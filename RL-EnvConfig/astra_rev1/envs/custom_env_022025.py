import numpy as np
import pywt
import gym
from gym import spaces

def _spectral_flatness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    X = np.fft.rfft(x * np.hanning(len(x)))
    P = (np.abs(X) ** 2) / max(len(x), 1)
    eps = 1e-12
    gm = np.exp(np.mean(np.log(P + eps)))
    am = np.mean(P + eps)
    sf = float(gm / am) if am > 0 else 0.0
    return np.clip(sf, 0.0, 1.0)

def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0

def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2)
    if s2 <= 0:
        return 0.0
    k = np.mean(((x - m) ** 4)) / (s2 ** 2)
    return float(k)

def _snr(clean: np.ndarray, test: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if clean.size == 0 or test.size == 0:
        return 0.0
    noise = test - clean
    sp = np.mean(clean ** 2)
    npow = np.mean(noise ** 2)
    return float(10 * np.log10((sp + 1e-10) / (npow + 1e-10)))

def _apply_wavelet_filter(x: np.ndarray, wavelet='db4', level=1, threshold_factor=1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float64)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(x), w.dec_len)
    lvl = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    lam = threshold_factor * sigma * np.sqrt(2*np.log(max(len(x),2)))
    cA, details = coeffs[0], coeffs[1:]
    details = [pywt.threshold(c, lam, mode="soft") for c in details]
    y = pywt.waverec([cA] + details, wavelet, mode="periodization")[:len(x)]
    y += (np.mean(x) - np.mean(y))
    return np.nan_to_num(y, copy=False)

class NoiseReductionEnv(gym.Env):
    """
    Observation is a fixed-length feature vector (12 dims), not raw window samples.
    Works with ANY window length passed at runtime.
    """
    metadata = {"render.modes": []}

    def __init__(self, wavelet='db4', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = int(level)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.iteration = 0
        self.last_threshold = 1.0  # mapped to [0.5, 2.5]
        self.prev_reward = 0.0

        self.clean_window = np.zeros(10, dtype=np.float64)
        self.noisy_window = np.zeros(10, dtype=np.float64)
        self.filtered_window = np.zeros(10, dtype=np.float64)
        self.reward_history = []

        self.supervised = True

    def _map_action(self, a: float) -> float:
        a = float(np.clip(a, -1.0, 1.0))
        return float(np.interp(a, [-1, 1], [0.5, 2.5]))

    def _build_obs(self) -> np.ndarray:
        noisy = self.noisy_window
        filtered = self.filtered_window if self.filtered_window.size else noisy

        mean = float(np.mean(noisy)) if noisy.size else 0.0
        std = float(np.std(noisy)) if noisy.size else 0.0
        rms = _rms(noisy)
        kurt = _kurtosis(noisy)
        flat = _spectral_flatness(noisy)

        if self.supervised and self.clean_window.size == noisy.size and noisy.size >= 2:
            snr_raw = _snr(self.clean_window, noisy)
            snr_f = _snr(self.clean_window, filtered)
            gain = snr_f - snr_raw
            if np.std(self.clean_window) < 1e-12 or np.std(filtered) < 1e-12:
                corr = 0.0
            else:
                c = np.corrcoef(filtered, self.clean_window)[0, 1]
                corr = float(c) if np.isfinite(c) else 0.0
        else:
            snr_raw = 0.0
            snr_f = 0.0
            gain = 0.0
            corr = 0.0

        obs = np.array(
            [
                mean, std, rms, kurt, flat,
                float(self.last_threshold),
                float(snr_raw), float(snr_f), float(gain), float(corr),
                float(self.prev_reward),
                float(self.iteration) / 1e4,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self) -> float:
        if not self.supervised or self.clean_window.size != self.noisy_window.size:
            return 0.0
        snr_raw = _snr(self.clean_window, self.noisy_window)
        snr_f = _snr(self.clean_window, self.filtered_window)
        snr_gain = snr_f - snr_raw
        mse = np.mean((self.filtered_window - self.clean_window) ** 2)
        loss = np.log1p(mse)
        if np.std(self.clean_window) < 1e-12 or np.std(self.filtered_window) < 1e-12:
            corr = 0.0
        else:
            c = np.corrcoef(self.filtered_window, self.clean_window)[0, 1]
            corr = float(c) if np.isfinite(c) else 0.0
        reward = (0.8 * snr_gain) - (1.0 * loss) + (0.25 * corr)
        return float(np.clip(reward, -10, 10))

    def set_signal_window(self, clean_window: np.ndarray, noisy_window: np.ndarray):
        self.iteration = 0
        self.prev_reward = 0.0
        self.last_threshold = 1.0

        self.clean_window = np.asarray(clean_window, dtype=np.float64) if clean_window is not None else np.array([], dtype=np.float64)
        self.noisy_window = np.asarray(noisy_window, dtype=np.float64)

        if self.clean_window.size:
            scale = np.max(np.abs(self.clean_window))
            if scale > 0:
                self.clean_window = self.clean_window / scale
                self.noisy_window = self.noisy_window / scale

        self.filtered_window = self.noisy_window.copy()

    def reset(self, seed=None, clean_signal=None, noisy_signal=None):
        self.reward_history = []
        if clean_signal is None or noisy_signal is None:
            n = 5000
            t = np.linspace(0, 1, n, endpoint=False)
            clean = np.sin(2*np.pi*100*t)
            noisy = clean + np.random.normal(0, 0.3, size=n)
            L = 256
            i0 = np.random.randint(0, n - L)
            cwin = clean[i0:i0+L]
            nwin = noisy[i0:i0+L]
            self.set_signal_window(cwin, nwin)
        else:
            clean_signal = np.asarray(clean_signal, dtype=np.float64)
            noisy_signal = np.asarray(noisy_signal, dtype=np.float64)
            L = min(len(clean_signal), len(noisy_signal))
            self.set_signal_window(clean_signal[:L], noisy_signal[:L])

        return self._build_obs()

    def step(self, action):
        self.iteration += 1
        thr = self._map_action(float(action[0]))
        self.last_threshold = thr

        self.filtered_window = _apply_wavelet_filter(
            self.noisy_window, wavelet=self.wavelet, level=self.level, threshold_factor=thr
        )

        reward = self._compute_reward()
        self.prev_reward = reward
        self.reward_history.append(float(reward))

        done = False
        if len(self.reward_history) > 1000:
            recent_rewards = self.reward_history[-100:]
            std_dev = np.std(recent_rewards)
            if std_dev < 1e-17:
                done = True

        if self.iteration >= 5000:
            done = True
        
        info = {
            "filtered_signal": self.filtered_window,
            "threshold_factor": thr,
            "reward": reward,
            "SNR_raw": _snr(self.clean_window, self.noisy_window) if self.supervised and self.clean_window.size == self.noisy_window.size else None,
            "SNR_filtered": _snr(self.clean_window, self.filtered_window) if self.supervised and self.clean_window.size == self.noisy_window.size else None,
        }
        return self._build_obs(), reward, done, info
