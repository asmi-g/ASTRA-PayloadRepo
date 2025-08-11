import gym
from gym import spaces
import numpy as np
import pywt

class NoiseReductionEnv(gym.Env):
    """
    Episodic env that slides a window over a synthetic signal (training),
    or accepts single windows from inference.
    Obs: [noisy_window, rms, kurtosis, spectral_flatness, last_threshold]
         => shape = (window_size + 4,)
    Action: 1D in [-1, 1] -> threshold factor in [0.5, 2.5].
    """
    metadata = {"render.modes": []}

    def __init__(self, signal_length=1000, window_size=20, training=True, seed=None):
        super().__init__()
        self.window_size = int(window_size)
        self.signal_length = int(signal_length)
        self.training = bool(training)
        self.plateau_len = 20
        self.plateau_std = 1e-4

        self.rng = np.random.RandomState(seed)
        self.denoiser = StatelessDenoisingEnv(window_size=window_size, rng=self.rng)

        # Observation is window + 3 features + last_threshold
        obs_dim = self.window_size + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = self.denoiser.action_space

        # holders
        self.clean = None     # full sequence (training) or None (inference)
        self.noisy = None
        self.clean_window = None
        self.noisy_window = None
        self.t = 0
        self.s= 0.0
        self.last_threshold = 1.5
        self.reward_history = []

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.denoiser.rng = self.rng
        return [seed]

    def reset(self, clean_signal=None, noisy_signal=None, training=None):
        if training is not None:
            self.training = bool(training)

        if clean_signal is None or noisy_signal is None:
            # TRAINING: generate a full signal and pick a start window
            self.clean, self.noisy = self._generate_signals(self.signal_length)
            start_idx = self.rng.randint(0, len(self.noisy) - self.window_size)
            self.clean_window = self.clean[start_idx:start_idx + self.window_size]
            self.noisy_window = self.noisy[start_idx:start_idx + self.window_size]
            self.t = start_idx
        else:
            # INFERENCE: use provided windows directly (assumed window-sized)
            self.clean_window = np.asarray(clean_signal, dtype=np.float64)
            self.noisy_window = np.asarray(noisy_signal, dtype=np.float64)
            # normalize by clean amplitude to avoid scale drift
            self.s = np.max(np.abs(self.clean_window))
            if self.s > 0:
                self.clean_window = self.clean_window / self.s
                self.noisy_window = self.noisy_window / self.s
            # mark as single-window (no full sequences)
            self.clean = None
            self.noisy = None
            self.t = 0

        self.reward_history = []
        self.last_threshold = 1.5
        return self._get_state(self.noisy_window)

    # ---------- features & state ----------
    def _compute_features(self, window):
        rms = np.sqrt(np.mean(window**2))
        std = np.std(window) + 1e-12
        kurtosis = np.mean(((window - np.mean(window)) / std)**4)
        spec = np.abs(np.fft.rfft(window))
        spectral_flatness = np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)
        return np.array([rms, kurtosis, spectral_flatness], dtype=np.float32)

    def _get_state(self, noisy_window):
        feats = self._compute_features(noisy_window)
        obs = np.concatenate(
            [noisy_window.astype(np.float32), feats, [np.float32(self.last_threshold)]],
            axis=0
        ).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def set_signal_window(self, clean_window, noisy_window):
        # For external (inference) use: set one window only
        self.clean_window = np.asarray(clean_window, dtype=np.float64)
        self.noisy_window = np.asarray(noisy_window, dtype=np.float64)
        s = np.max(np.abs(self.clean_window))
        if s > 0:
            self.clean_window /= s
            self.noisy_window /= s
        self.clean = None
        self.noisy = None
        self.t = 0
        self.s = s

    # ---------- signal synthesis ----------
    def _generate_signals(self, n):
        x = np.arange(n)
        fs = 1e4
        f0 = 100.0
        t = x / fs

        clean = np.sin(2 * np.pi * f0 * t)

        white = self.rng.normal(0.0, np.sqrt(0.1), size=n)
        pink = self._generate_pink_noise(n, white)

        noisy = clean + white + pink
        noisy = self._add_bursts(noisy)
        return clean, noisy

    def _generate_pink_noise(self, n, white):
        freq = np.fft.rfftfreq(n)
        shape = np.sqrt(np.maximum(freq, 1e-6))
        spec = np.fft.rfft(white / np.sqrt(0.1))
        spec *= shape
        out = np.fft.irfft(spec, n=n)
        return np.real(out)

    def _add_bursts(self, sig, p=0.01, amp=3.0, dur=10):
        y = sig.copy()
        for i in range(len(sig)):
            if self.rng.rand() < p:
                j = min(i + dur, len(sig))
                y[i:j] += self.rng.uniform(-amp, amp)
        return y

    # ---------- RL step ----------
    def step(self, action):
        # choose correct window source
        if self.clean is not None and self.noisy is not None:
            c = self.clean[self.t:self.t + self.window_size]
            n = self.noisy[self.t:self.t + self.window_size]
        else:
            c = self.clean_window
            n = self.noisy_window

        _, reward, _, info = self.denoiser.step(n, action, c)
        self.last_threshold = float(info["threshold_factor"])

        if not np.isfinite(reward):
            reward = 0.0
        self.reward_history.append(reward)

        # --- always advance if we are traversing a full sequence ---
        if self.clean is not None and self.noisy is not None:
            self.t += 1

        done = False
        if self.clean is not None and self.noisy is not None:
            # plateau early-stop only when training (optional)
            if self.training and len(self.reward_history) >= self.plateau_len:
                if np.std(self.reward_history[-self.plateau_len:]) < self.plateau_std:
                    done = True

            # end-of-signal stop for both training and eval
            if self.t + self.window_size >= len(self.noisy):
                done = True

            # refresh current window if not done
            if not done:
                n = self.noisy[self.t:self.t + self.window_size]

        info['s'] = self.s
        obs = self._get_state(n)
        return obs, float(reward), done, info



class StatelessDenoisingEnv(gym.Env):
    """Stateless wavelet denoiser: thresholds details only."""
    def __init__(self, window_size=10, wavelet='db4', level=1, rng=None):
        super().__init__()
        self.window_size = int(window_size)
        self.wavelet = wavelet
        self.level = int(level)
        self.rng = rng or np.random.RandomState(None)

        # the inner env's obs is a plain window; wrapper adds features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32
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

    @staticmethod
    def _snr(clean, test):
        noise = test - clean
        sp = np.mean(clean ** 2)
        npow = np.mean(noise ** 2) + 1e-12
        return 10.0 * np.log10((sp + 1e-12) / npow)

    def step(self, window, action, clean_window=None):
        a = float(np.clip(action[0], -1.0, 1.0))
        threshold = np.interp(a, [-1.0, 1.0], [0.5, 2.5])

        filtered = self.apply_filter(window, threshold)

        reward = None
        snr_raw = snr_filt = None
        if clean_window is not None:
            snr_raw = self._snr(clean_window, window)
            snr_filt = self._snr(clean_window, filtered)
            snr_gain = snr_filt - snr_raw

            mse = np.mean((filtered - clean_window) ** 2)
            loss = mse/(np.var(clean_window) + 1e-12)

            if np.std(clean_window) < 1e-12 or np.std(filtered) < 1e-12:
                corr = 0.0
            else:
                c = np.corrcoef(filtered, clean_window)[0, 1]
                corr = c if np.isfinite(c) else 0.0

            reward = (0.8 * snr_gain) - (1.0 * loss) + (0.25 * corr)
            reward = np.clip(reward, -10, 10)

        info = {
            "filtered_signal": filtered,
            "threshold_factor": threshold,
            "reward": reward,
            "SNR_raw": snr_raw,
            "SNR_filtered": snr_filt,
        }
        # return original window as obs; wrapper builds the full state
        return window.astype(np.float32), float(0.0 if reward is None else reward), False, info

# # step 1: creating custom environment by subclassing gym.Env
# import gym
# from gym import spaces
# import numpy as np
# import pywt

# class NoiseReductionEnv(gym.Env):
#     def __init__(self):
#         super().__init__()

#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

#         self.step_size = 0.1
#         self.threshold_factor = 1.0
#         self.prev_reward = 0
#         self.reward_history = []
#         self.iteration = 0
#         self.no_signal = False

#         # Dummy init values
#         self.clean_signal = np.zeros(10)
#         self.raw_signal = np.zeros(10)
#         self.filtered_signal = np.zeros(10)

#     # helper functions
#     def set_signal_window(self, clean_signal, noisy_signal):
#         self.clean_signal = clean_signal.astype(np.float64)
#         self.raw_signal = noisy_signal.astype(np.float64)
#         if self.no_signal:
#             self.clean_signal = clean_signal[self.iteration:self.iteration+10].astype(np.float64)
#             self.raw_signal = noisy_signal[self.iteration:self.iteration+10].astype(np.float64)
#         scale = np.max(np.abs(self.clean_signal))
#         if scale > 0:
#             self.clean_signal /= scale
#             self.raw_signal /= scale

#     def apply_filter(self, signal, wavelet='db4', level=1, threshold_factor=1.0):
#         coeffs = pywt.wavedec(signal, wavelet, level=level)
#         sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#         threshold = threshold_factor * sigma
#         coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
#         return pywt.waverec(coeffs, wavelet)

#     def calculate_SNR(self, clean, noisy):
#         noise = noisy - clean
#         signal_power = np.mean(clean**2)
#         noise_power = np.mean(noise**2)
#         return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

#     def reset(self, seed=None, clean_signal=None, noisy_signal=None):
#         self.no_signal = False
#         if clean_signal is None or noisy_signal is None:
#             self.no_signal = True
#             print("signal not provided; generating random signal")
#             self.clean_signal = np.sin(np.linspace(0, 2*np.pi, 5000))
#             self.raw_signal = self.clean_signal + np.random.normal(0, 0.3, size=5000)
#         else:
#             self.clean_signal = clean_signal.astype(np.float64)
#             self.raw_signal = noisy_signal.astype(np.float64)

#         scale = np.max(np.abs(self.clean_signal))
#         if scale > 0:
#             self.clean_signal /= scale
#             self.raw_signal /= scale

#         self.threshold_factor = 1.0
#         self.filtered_signal = self.apply_filter(self.raw_signal)
#         self.prev_reward = 0
#         self.reward_history = []
#         self.iteration = 0

#         snr_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
#         snr_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

#         state = np.array([
#             self.iteration,
#             np.mean(self.clean_signal),
#             np.mean(self.raw_signal),
#             np.mean(self.filtered_signal),
#             self.threshold_factor,
#             snr_raw,
#             snr_filtered,
#             self.prev_reward,
#             0.0
#         ], dtype=np.float32)

#         return state


#     def step(self, action):
#         self.iteration += 1
#         delta = np.clip(float(action[0]), -1, 1)
#         self.threshold_factor += self.step_size * delta
#         self.threshold_factor = np.clip(self.threshold_factor, 0.0, 5.0)

#         self.filtered_signal = self.apply_filter(self.raw_signal, threshold_factor=self.threshold_factor)
#         snr_raw = self.calculate_SNR(self.clean_signal, self.raw_signal)
#         snr_filtered = self.calculate_SNR(self.clean_signal, self.filtered_signal)

#         # reward = snr_filtered - snr_raw
#         # if reward > 0:
#         #     reward += 0.2 * reward

#         # bias_penalty = np.abs(np.mean(self.filtered_signal - self.clean_signal))
#         # reward -= 0.05 * bias_penalty

#         # if abs(self.threshold_factor) < 0.01:
#         #     reward -= 0.02

#         snr_gain = snr_filtered - snr_raw
#         mae = np.mean(np.abs(self.filtered_signal - self.clean_signal))
#         clipping_threshold = 0.95 * np.max(np.abs(self.clean_signal))
#         clipping_penalty = np.mean(np.abs(self.filtered_signal) > clipping_threshold)
#         smoothness_penalty = np.mean(np.abs(np.diff(self.filtered_signal)))

#         reward = (
#             +1.0 * snr_gain
#             -0.5 * mae
#             -0.3 * clipping_penalty
#             -0.1 * smoothness_penalty
#         )

#         self.reward_history.append(reward)
#         self.prev_reward = reward

#         state = np.array([
#             self.iteration,
#             np.mean(self.clean_signal),
#             np.mean(self.raw_signal),
#             np.mean(self.filtered_signal),
#             self.threshold_factor,
#             snr_raw,
#             snr_filtered,
#             self.prev_reward,
#             np.mean(self.reward_history) if self.reward_history else 0.0
#         ], dtype=np.float32)

#         done = False
#         if self.iteration >= 100:
#             if len(self.reward_history) > 10:
#                 recent_rewards = self.reward_history[-10:]
#                 std_dev = np.std(recent_rewards)
#                 if std_dev < 1e-4:
#                     done = True

#         info = {
#             "iteration": self.iteration,
#             "clean_signal": self.clean_signal,
#             "noisy_signal": self.raw_signal,
#             "filtered_signal": self.filtered_signal,
#             "threshold_factor": self.threshold_factor,
#             "SNR_raw": snr_raw,
#             "SNR_filtered": snr_filtered,
#             "prev_reward": self.prev_reward,
#             "reward_history": self.reward_history
#         }

#         return state, reward, done, info

#     def close(self):
#         pass
