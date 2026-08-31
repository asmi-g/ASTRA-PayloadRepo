import gym
from gym import spaces
import numpy as np
import pickle
import pywt

class NoiseReductionEnv(gym.Env):
    def __init__(self, signal_length=1000, window_size=10, flevel = 5, mode="inference", noise_model_path=None):
        super().__init__()
        assert mode in ("train", "inference"), f"mode must be 'train' or 'inference', got {mode}"
        self.mode = mode
        self.window_size = window_size
        self.signal_length = np.random.randint(signal_length/5, signal_length*2)
        self.denoiser = StatelessDenoisingEnv(window_size=window_size, level=flevel)
        self.reward_history = []
        self.observation_space = self.denoiser.observation_space
        self.action_space = self.denoiser.action_space

        # noise_model_path=None (default) -> unchanged OFT behavior, synthetic
        # white+pink+burst noise via _generate_signals(). Passing a path to a
        # noise_extractor.py pickle (e.g. noise_model_fs1.pkl) switches to
        # _generate_signals_from_noise_model() instead, for UN training.
        self.noise_model = None
        if noise_model_path is not None:
            with open(noise_model_path, "rb") as f:
                self.noise_model = pickle.load(f)

    def reset(self, clean_signal=None, noisy_signal=None):
        if self.mode == "train":
            if clean_signal is not None and noisy_signal is not None:
                self.clean = clean_signal.astype(np.float64)
                self.noisy = noisy_signal.astype(np.float64)
            elif self.noise_model is not None:
                self.clean, self.noisy = self._generate_signals_from_noise_model()
            else:
                self.clean, self.noisy = self._generate_signals()

            # scale by clean signal's own peak (training-time normalization)
            scale = np.max(np.abs(self.clean))
            if scale > 0:
                self.clean /= scale
                self.noisy /= scale
        else:
            # inference: use exactly what's passed in, no generation, no scaling
            # (scaling/alignment is handled upstream in inference.py)
            if clean_signal is None or noisy_signal is None:
                raise ValueError("inference mode requires clean_signal and noisy_signal to be provided")
            self.clean = clean_signal.astype(np.float64)
            self.noisy = noisy_signal.astype(np.float64)

        self.t = 0
        self.reward_history = []
        return self._get_state()

    def _get_state(self):
        state = self.noisy[self.t:self.t + self.window_size].astype(np.float32)
        if state.shape[0] != self.window_size:
            raise ValueError(f"State length {state.shape[0]} does not match expected window size {self.window_size}")
        return state

    def set_signal_window(self, clean_window, noisy_window):
        # inference-only helper: swap in a new window, always start at t=0
        self.clean = clean_window
        self.noisy = noisy_window
        self.t = 0

    def _generate_signals(self):
        x = np.arange(self.signal_length)
        t = x / 1e6  # F_SAMPLING
        f_signal = 100_000  # Hz

        clean = np.sin(2 * np.pi * f_signal * t)
        white_noise = np.random.normal(0, np.sqrt(0.1), size=self.signal_length)
        pink_noise = self._generate_pink_noise(self.signal_length, white_noise) * np.sqrt(0.1)
        noisy = clean + white_noise + pink_noise
        noisy = self._add_bursts(noisy)

        return clean, noisy

    def _generate_signals_from_noise_model(self):
        # UN ("updated noise") training signal: same clean tone as _generate_signals
        # (matches TX.py), but noise comes from a block-bootstrap of the real
        # flight-extracted residual (noise_extractor.py's noise_residual_sample)
        # instead of the synthetic white+pink+burst model. A Gaussian-innovation
        # simulation of the fitted AR(50) was considered and rejected: the real
        # residual has excess kurtosis in the tens-to-hundreds (ADC-quantization
        # spikes -- see task 1/3 findings), which an AR process driven by Gaussian
        # innovations cannot reproduce regardless of its coefficients. Bootstrapping
        # directly from the real samples preserves that spikiness exactly, with no
        # distributional assumption. No _add_bursts() call here -- the real
        # residual already contains the actual burst events; adding synthetic
        # ones on top would double-count them.
        x = np.arange(self.signal_length)
        t = x / 1e6  # F_SAMPLING, matches TX.py
        f_signal = 100_000  # Hz, matches TX.py

        clean = np.sin(2 * np.pi * f_signal * t)
        noise = self._block_bootstrap_noise(self.signal_length)

        # Rescale to the REAL noise-to-signal severity ratio, not just the
        # noise's raw captured magnitude. The bootstrapped sample's absolute
        # scale reflects the real flight's ~0.00474-amplitude clean signal;
        # pairing that raw magnitude with this unit-amplitude clean tone
        # would make training ~250x gentler (relatively) than the real
        # -33.8dB flight SNR, defeating the point of training on this noise
        # model in the first place.
        real_ratio = self.noise_model["noise_std"] / self.noise_model["clean_std"]
        target_noise_std = clean.std() * real_ratio
        current_noise_std = noise.std()
        if current_noise_std > 0:
            noise = noise * (target_noise_std / current_noise_std)

        noisy = clean + noise

        return clean, noisy

    def _block_bootstrap_noise(self, length, block_size=100):
        sample = self.noise_model["noise_residual_sample"]
        n_sample = len(sample)
        block_size = min(block_size, n_sample)
        blocks = []
        total = 0
        while total < length:
            start = np.random.randint(0, n_sample - block_size + 1)
            blocks.append(sample[start:start + block_size])
            total += block_size
        return np.concatenate(blocks)[:length]

    def _generate_pink_noise(self, size, white_noise):
        # NOTE: this filter (sqrt(|freq|), gain rising with frequency) actually
        # produces blue noise, not true 1/f pink noise (that would need to
        # divide by sqrt(|freq|) instead) -- named/documented as "pink" going
        # back to the original signal_simulation.py. A true 1/f fix was tried
        # and reverted: naive per-bin FFT division concentrates most of the
        # energy in the single lowest nonzero frequency bin, making pink noise
        # ~14-17x louder than white noise and unstably dependent on signal
        # length. Kept as-is (matches signal_simulation.py) and documented
        # as a known discrepancy for the paper rather than "fixed" further.
        freq = np.fft.fftfreq(size)
        pink_filter = np.sqrt(np.abs(freq))
        spectrum = np.fft.fft(white_noise / np.sqrt(0.1))
        spectrum *= pink_filter
        pink_noise = np.fft.ifft(spectrum)
        return np.real(pink_noise)

    def _add_bursts(self, signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
        noisy_signal = signal.copy()
        for i in range(len(signal)):
            if np.random.rand() < burst_probability:
                burst_start = i
                burst_end = min(i + burst_duration, len(signal))
                noisy_signal[burst_start:burst_end] += np.random.uniform(-burst_amplitude, burst_amplitude)
        return noisy_signal

    def step(self, action):
        clean_window = self.clean[self.t:self.t+self.window_size]
        noisy_window = self.noisy[self.t:self.t+self.window_size]

        _, reward, _, info = self.denoiser.step(noisy_window, action, clean_window)

        if self.mode == "train":
            self.t += 10
            self.t = min(self.t, len(self.noisy) - self.window_size)

        self.reward_history.append(reward)

        done = False
        if self.mode == "train":
            if len(self.reward_history) > 100:
                recent_rewards = self.reward_history[-10:]
                std_dev = np.std(recent_rewards)
                if std_dev < 1e-7:
                    done = True
            if self.t + self.window_size >= len(self.noisy):
                done = True
        # inference mode: never signals done from t/reward-plateau logic here;
        # inference.py's own loop controls when to stop

        return self._get_state(), reward, done, info


class StatelessDenoisingEnv(gym.Env):
    # unchanged
    def __init__(self, window_size=10, wavelet='db4', level=1):
        super().__init__()
        self.window_size = window_size
        self.wavelet = wavelet
        self.level = level
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def apply_filter(self, window, threshold_factor):
        x = np.asarray(window, dtype=np.float64)
        n = len(x)
        w = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(n, w.dec_len)
        lvl = max(1, min(self.level, max_level))
        coeffs = pywt.wavedec(x, self.wavelet, level=lvl, mode="periodization")
        # Floor sigma above 0: heavily-quantized input (long runs of exactly
        # repeated values, e.g. real flight data) can make >50% of the finest
        # detail band exactly zero, so its median is exactly 0 -- giving
        # lam=0 for EVERY threshold_factor (all collapse to the identical
        # answer). Worse, pywt.threshold's soft mode hits a 0/0 NaN internally
        # whenever a coefficient's magnitude AND the threshold value are both
        # exactly zero, which spreads through waverec and gets silently
        # zeroed by nan_to_num below -- so lam=0 doesn't even give the
        # identity filter it should, it gives a fully degenerate (all-zero)
        # output. A tiny nonzero floor avoids both: lam varies with
        # threshold_factor again, and pywt correctly resolves zero-magnitude
        # coefficients to 0 (verified) once the threshold value isn't also
        # exactly zero.
        sigma = max(np.median(np.abs(coeffs[-1])) / 0.6745, 1e-8) if coeffs[-1].size else 1e-8
        lam = threshold_factor * sigma * np.sqrt(2*np.log(max(n,2)))
        cA, details = coeffs[0], coeffs[1:]
        details = [pywt.threshold(c, lam, mode="soft") for c in details]
        y = pywt.waverec([cA] + details, self.wavelet, mode="periodization")[:n]
        y += (np.mean(x) - np.mean(y))
        return np.nan_to_num(y, copy=False)

    def step(self, window, action, clean_window=None):
        threshold = float(np.clip(action[0], -1.0, 1.0))
        # Range widened from [0.5, 2.5]: the true snr_improvement/signal_loss
        # optimum sits at threshold_factor~0.30 and correlation peaks near
        # ~0.05, both below the old floor of 0.5 -- this range now brackets
        # them. Floor kept strictly positive (not 0 or negative): a negative
        # threshold_factor makes pywt.threshold amplify coefficients instead
        # of shrinking them, and produces NaN outputs for any exactly-zero
        # coefficient, which would corrupt the filtered window entirely.
        threshold = np.interp(threshold, [-1, 1], [0.05, 2.5])
        filtered_window = self.apply_filter(window, threshold)

        reward = None
        if clean_window is not None:
            snr_raw = self._snr(clean_window, window)
            snr_filtered = self._snr(clean_window, filtered_window)
            snr_improvement = snr_filtered - snr_raw
            signal_loss = np.log1p(np.mean((filtered_window - clean_window)**2))
            correlation = np.corrcoef(filtered_window, clean_window)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            reward = snr_improvement - 1.25 * signal_loss + 0.25 * correlation
        else:
            snr_raw = None
            snr_filtered = None

        info = {
            "filtered_signal": filtered_window,
            "threshold_factor": threshold,
            "reward": reward,
            "SNR_raw": snr_raw,
            "SNR_filtered": snr_filtered
        }
        return window.astype(np.float32), reward, False, info

    def _snr(self, clean, test):
        noise = test - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

    def reset(self):
        pass

    def close(self):
        pass