import gym
from gym import spaces
import numpy as np
import pywt

class NoiseReductionEnv(gym.Env):
    def __init__(self, signal_length=1000, window_size=10):
        super().__init__()
        self.window_size = window_size
        self.signal_length = np.random.randint(200, 1000)
        self.denoiser = StatelessDenoisingEnv(window_size=window_size)
        self.reward_history = []
        self.observation_space = self.denoiser.observation_space
        self.action_space = self.denoiser.action_space

    def reset(self, clean_signal=None, noisy_signal=None):
        self.signal_length = np.random.randint(2000, 10000)
        if clean_signal is not None and noisy_signal is not None:
            self.clean = clean_signal.astype(np.float64)
            self.noisy = noisy_signal.astype(np.float64)
            scale = np.max(np.abs(self.clean))
            if scale > 0:
                self.clean /= scale
                self.noisy /= scale
        else:
            self.clean, self.noisy = self._generate_signals()
        self.t = 0
        self.reward_history = []
        return self._get_state()
    
    def _get_state(self):
        state = self.noisy[self.t:self.t + self.window_size].astype(np.float32)
        if state.shape[0] != self.window_size:
            raise ValueError(f"State length {state.shape[0]} does not match expected window size {self.window_size}")
        return state
    
    def set_signal_window(self, clean_window, noisy_window):
        self.clean = clean_window
        self.noisy = noisy_window
        self.t = 0

    def _generate_signals(self):
        x = np.arange(self.signal_length)
        t = x / 1e4  # F_SAMPLING = 10 kHz
        f_signal = 100  # Hz

        # Clean signal: sine wave
        clean = np.sin(2 * np.pi * f_signal * t)

        # White noise
        white_noise = np.random.normal(0, np.sqrt(0.1), size=self.signal_length)

        # Pink noise
        pink_noise = self._generate_pink_noise(self.signal_length, white_noise)

        # Add bursts
        noisy = clean + white_noise + pink_noise
        noisy = self._add_bursts(noisy)

        return clean, noisy

    def _generate_pink_noise(self, size, white_noise):
        freq = np.fft.fftfreq(size)
        pink_filter = np.sqrt(np.abs(freq) + 1e-6)  # avoid divide by zero
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
        #self.t += 1 #have to comment this out nduring inference.py
        self.reward_history.append(reward)
        
        done = False
        if len(self.reward_history) > 100:
            recent_rewards = self.reward_history[-10:]
            std_dev = np.std(recent_rewards)
            if std_dev < 1e-17:
                #print(f"std: {std_dev}. threshold = {std_dev < 1e-20}")
                done = True
        # if self.t + self.window_size >= len(self.noisy): # comment this out too during inference.py
        #     done = True

        return self._get_state(), reward, done, info


class StatelessDenoisingEnv(gym.Env):
    def __init__(self, window_size=10, wavelet='db4', level=1):
        super().__init__()

        self.window_size = window_size
        self.wavelet = wavelet
        self.level = level

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size,), dtype=np.float32
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def apply_filter(self, window, threshold_factor):
        x = np.asarray(window, dtype=np.float64)
        n = len(x)

        w = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(n, w.dec_len)
        lvl = max(1, min(self.level, max_level))
        coeffs = pywt.wavedec(x, self.wavelet, level=lvl, mode="periodization")

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
        lam = threshold_factor * sigma * np.sqrt(2*np.log(max(n,2)))  # universal-like
        cA, details = coeffs[0], coeffs[1:]
        details = [pywt.threshold(c, lam, mode="soft") for c in details]
        y = pywt.waverec([cA] + details, self.wavelet, mode="periodization")[:n]
        y += (np.mean(x) - np.mean(y))
        return np.nan_to_num(y, copy=False)

    def step(self, window, action, clean_window=None):
        threshold = float(np.clip(action[0], -1.0, 1.0))  # from SAC
        threshold = np.interp(threshold, [-1, 1], [0.5, 2.5]) 
        filtered_window = self.apply_filter(window, threshold)

        reward = None
        if clean_window is not None:
            snr_raw = self._snr(clean_window, window)
            snr_filtered = self._snr(clean_window, filtered_window)
            snr_improvement = snr_filtered - snr_raw
            signal_loss = np.log1p(np.mean((filtered_window - clean_window)**2))#/ (np.max(clean_window)**2 + 1e-6)

            correlation = np.corrcoef(filtered_window, clean_window)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            reward = (
                0.5 * snr_improvement
                - 1.25 * signal_loss
                + 0.25 * correlation
            )

            #reward = 1.5 * snr_improvement - 0.75 * signal_loss + 0.5 * correlation
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
        #print(f"threshold factor: {threshold}")

        return window.astype(np.float32), reward, False, info 

    def _snr(self, clean, test):
        noise = test - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

    def reset(self):
        # Not used
        pass

    def close(self):
        pass