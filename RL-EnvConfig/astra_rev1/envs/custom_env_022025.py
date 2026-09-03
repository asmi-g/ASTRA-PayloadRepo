#TODO: clean up comments

import gym
from gym import spaces
import numpy as np
import pickle
import pywt

class NoiseReductionEnv(gym.Env):
    def __init__(self, signal_length=1000, window_size=10, flevel = 5, mode="inference", noise_model_path=None, accurate=False, signal_gen="accurate"):
        super().__init__()
        assert mode in ("train", "inference"), f"mode must be 'train' or 'inference', got {mode}"
        assert signal_gen in ("accurate", "pink_bursts"), f"signal_gen must be 'accurate' or 'pink_bursts', got {signal_gen!r}"
        self.mode = mode
        # signal_gen selects the train-mode generator in reset():
        #   "accurate"     -> generate_accurate_signals()  (OAN), unless a
        #                     noise_model_path is given -> generate_accurate_flight_signals() (PFN)
        #   "pink_bursts"  -> generate_signals_pink_bursts()  (OPBN): white floor
        #                     + true-1/f pink only inside a Poisson-count of short
        #                     additive bursts, no continuous pink, no flare envelope.
        self.signal_gen = signal_gen
        # accurate=True routes reset() through the corrected generators
        # (generate_accurate_signals / generate_accurate_flight_signals).
        # For the OFT model: true 1/f pink noise (not the blue-noise filter),
        # white/pink normalised to the noise_power budget, and rare
        # FRED-shaped solar-radio-burst episodes (Poisson mean 0.3 per signal
        # by default -- pass n_flares=0 for a stationary before/after against
        # _generate_signals()). For the flight model it is a parametric
        # AR + heavy-tailed + measured rms_profile build instead of the block
        # bootstrap.
        self.window_size = window_size
        # signal_length is re-drawn on every train reset() (see _draw_signal_length)
        # so episodes vary in length ~ U[arg/5, arg*2). Keep an initial draw here
        # so code that reads self.signal_length before the first reset()
        # (e.g. SB3 check_env) still works.
        self._signal_length_arg = signal_length
        self.signal_length = self._draw_signal_length()
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

    def _draw_signal_length(self):
        # U[arg/5, arg*2) -- same range the constructor used, now sampled per
        # episode so training sees a range of episode lengths. Both accurate
        # generators handle a varying N cheaply: generate_accurate_signals()
        # has no per-N state, and generate_accurate_flight_signals() calibrates
        # its spike-ratio bisection at a fixed probe length with N absent from
        # the cache key (see that method), so a per-episode-varying length no
        # longer busts the cache.
        return int(np.random.randint(self._signal_length_arg / 5,
                                     self._signal_length_arg * 2))

    def reset(self, clean_signal=None, noisy_signal=None):
        if self.mode == "train":
            if clean_signal is not None and noisy_signal is not None:
                self.clean = clean_signal.astype(np.float64)
                self.noisy = noisy_signal.astype(np.float64)
            else:
                self.signal_length = self._draw_signal_length()
                if self.signal_gen == "pink_bursts":
                    self.clean, self.noisy = self.generate_signals_pink_bursts()
                elif self.noise_model is not None:
                    self.clean, self.noisy = self.generate_accurate_flight_signals()
                else:
                    self.clean, self.noisy = self.generate_accurate_signals()

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

    # ------------------------------------------------------------------
    # Physically-motivated cosmic-noise generation (task 1). Used by
    # generate_accurate_signals() below; kept separate from the legacy
    # _generate_signals() / _generate_pink_noise() so the old behaviour is
    # untouched and the two can be compared directly.
    # ------------------------------------------------------------------
#TODO: clean up/reduce no of functions

    def _pink_noise_1f(self, size, target_std, low_freq_corner_bins=1):
        # TRUE pink (1/f) noise. _generate_pink_noise() multiplies the
        # spectrum by sqrt(|f|), which is BLUE noise (power ~ f). The
        # galactic/cosmic radio background is conventionally modelled as
        # 1/f-type: power spectral density ~ 1/f, i.e. amplitude ~
        # 1/sqrt(f).  [M. S. Keshner, "1/f noise", Proc. IEEE 70(3):212-218,
        # 1982.]  The naive 1/sqrt(f) FFT filter is unstable (all the energy
        # piles into the lowest nonzero bin -- the reason the earlier
        # attempt was reverted); this uses random-phase synthesis with an
        # explicit low-frequency corner and then renormalises to target_std,
        # which is stable in signal length.
        freqs = np.fft.rfftfreq(size)
        mag = np.zeros_like(freqs)
        nz = freqs > 0
        corner = freqs[min(low_freq_corner_bins, len(freqs) - 1)] if len(freqs) > 1 else 1.0
        mag[nz] = 1.0 / np.sqrt(np.maximum(freqs[nz], corner))
        phases = np.random.uniform(0, 2 * np.pi, size=freqs.shape)
        x = np.fft.irfft(mag * np.exp(1j * phases), n=size)
        s = x.std()
        return x * (target_std / s) if s > 0 else x
 
    def _add_bursts(self, signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
        # Short impulsive bursts: sub-millisecond single-particle / cosmic-ray
        # strikes in the ADC and ISM-band RFI packets. Called by
        # generate_accurate_signals() when keep_impulsive_bursts=True.
        noisy_signal = signal.copy()
        for i in range(len(signal)):
            if np.random.rand() < burst_probability:
                burst_start = i
                burst_end = min(i + burst_duration, len(signal))
                noisy_signal[burst_start:burst_end] += np.random.uniform(-burst_amplitude, burst_amplitude)
        return noisy_signal

    def _pink_burst_mask(self, size, burst_len, n_bursts=0.5,
                         rise_frac=0.15, decay_tau_frac=0.35):
        # Additive 0..1 mask, zero outside a Poisson-count of short episodes.
        # Same fast-rise / slow-decay shape as _flare_envelope(), but here it
        # GATES an additive pink term instead of MULTIPLYING the noise std --
        # so between episodes the signal is exactly white at noise_power, and
        # during an episode a colored (1/f) component is layered on top.
        # Overlapping episodes are combined with max() (this is a mask in
        # [0,1], not a gain that should stack).
        mask = np.zeros(size)
        burst_len = int(min(burst_len, max(2, size // 4)))
        for _ in range(np.random.poisson(n_bursts)):
            start = np.random.randint(0, size)
            L = max(2, int(burst_len * np.random.uniform(0.5, 1.5)))
            rise = max(1, int(L * rise_frac))
            decay = max(1, L - rise)
            shape = np.empty(L)
            shape[:rise] = np.linspace(0.0, 1.0, rise)
            shape[rise:] = np.exp(-np.arange(decay) / (decay_tau_frac * decay))
            end = min(start + L, size)
            mask[start:end] = np.maximum(mask[start:end], shape[:end - start])
        return mask

    def generate_signals_pink_bursts(self, noise_power=0.1, burst_gain=4.0,
                                     burst_len=None, n_bursts=0.5,
                                     keep_impulsive_bursts=True, seed=None):
        # Variant of generate_accurate_signals() with NO continuous pink noise
        # and NO solar-flare envelope. The always-on floor is pure white at
        # noise_power; 1/f (pink) energy appears only inside a Poisson-count of
        # short ADDITIVE bursts:
        #     noisy = clean + white + pink * burst_mask   [+ impulses]
        #
        # Compare:
        #   generate_accurate_signals() : noisy = clean + (white + pink) * flare_env
        #       -- continuous true-1/f pink, plus a multiplicative FRED-shaped
        #          std-lift for solar radio bursts.
        #   signal_simulation.py (legacy prototype) :
        #       noisy = clean + white + BLUE(same white array) + impulses
        #       -- always-on, "pink" is really blue (sqrt|f|), correlated with
        #          white, continuous floor ends up ~2x noise_power.
        # Here the colored term is true 1/f, an INDEPENDENT draw, and only
        # present in a few episodes; the continuous floor is exactly
        # noise_power (white only).
        #
        # Knobs:
        #  - noise_power : variance of the continuous white floor.
        #  - burst_gain  : peak pink std during an episode, as a multiple of
        #                  the white std (4.0 -> +12 dB local excess; local
        #                  variance rises ~burst_gain**2 at the episode peak).
        #  - n_bursts    : Poisson mean of pink-burst onsets per signal
        #                  (0 -> _pink_burst_mask() is all-zeros, i.e. a plain
        #                  white-noise signal; use for a before/after).
        #  - burst_len   : episode duration in samples; defaults to
        #                  20 * window_size (>> one agent window, << signal;
        #                  capped to N//4 in _pink_burst_mask()).
        #  - keep_impulsive_bursts : add the short ±impulse hits via the
        #                  unchanged _add_bursts(), as in the other generators.
        if seed is not None:
            np.random.seed(seed)
        N = self.signal_length
        t = np.arange(N) / 1e6          # F_SAMPLING, matches _generate_signals
        f_signal = 100_000             # Hz, matches _generate_signals / TX.py
        clean = np.sin(2 * np.pi * f_signal * t)

        white_std = np.sqrt(noise_power)
        white = np.random.normal(0, white_std, N)

        pink = self._pink_noise_1f(N, white_std * burst_gain)
        if burst_len is None:
            burst_len = 20 * self.window_size
        mask = self._pink_burst_mask(N, burst_len=burst_len, n_bursts=n_bursts)

        noisy = clean + white + pink * mask
        if keep_impulsive_bursts:
            noisy = self._add_bursts(noisy)

        return clean, noisy

    def _bg_innovations(self, n, spike_prob, spike_ratio):
        # Bernoulli-Gaussian spike mixture: each sample is N(0,1), and with
        # probability spike_prob its std is multiplied by spike_ratio. Unit
        # variance overall. Sample excess kurtosis is stable (unlike a
        # near-df=4 Student-t) and tunable via spike_ratio:
        #   K_excess ~ 3[(1-p) + p r^4] / [(1-p) + p r^2]^2 - 3
        z = np.random.normal(0.0, 1.0, n)
        spike = np.random.random(n) < spike_prob
        z[spike] *= spike_ratio
        z /= np.sqrt((1.0 - spike_prob) + spike_prob * spike_ratio ** 2)
        return z

    def _rms_profile_envelope(self, n, rp, block=1000, mode="realrate"):
        # rp[i] is the noise RMS over `block` consecutive real samples, so at
        # real rate one rp point covers `block` samples. "realrate" expands
        # each point x block (a random contiguous slice if the signal is
        # shorter than the whole profile) -- this keeps the true drift
        # timescale. "stretch" maps the whole profile onto n (changes the
        # timescale); "tile" repeats it.
        rp = rp / (rp.mean() + 1e-12)
        if mode == "stretch":
            return np.interp(np.linspace(0, len(rp) - 1, n), np.arange(len(rp)), rp)
        if mode == "tile":
            reps = int(np.ceil(n / len(rp)))
            return np.tile(np.concatenate([rp, rp[::-1]]), reps)[:n]
        # realrate
        need = int(np.ceil(n / block))
        if need <= len(rp):
            start = np.random.randint(0, len(rp) - need + 1)
            sel = rp[start:start + need]
        else:
            reps = int(np.ceil(need / len(rp)))
            sel = np.tile(np.concatenate([rp, rp[::-1]]), reps)[:need]
        return np.repeat(sel, block)[:n]

    def generate_accurate_flight_signals(self, innov_excess_kurt=None,
                                         spike_prob=0.03, use_rms_profile=True,
                                         envelope_mode="realrate", seed=None):
        # Task 2: parametric ("does not resample the snippet") stand-in for
        # _generate_signals_from_noise_model(). Same clean tone as TX.py.
        # Flight noise is rebuilt from fitted quantities in noise_model_fs1.pkl
        # (extractor v4: segment 0 excluded -- see noise_extractor.py), NOT by
        # block-bootstrapping noise_residual_sample:
        #
        #   1. AR(50) colour  -- ar_params[1:] as all-pole coefficients. The
        #      fs1 background is essentially WHITE (per-segment PSD log-log
        #      slope ~ -0.04, lag-1 autocorr ~ 0.08), so this filter is close
        #      to identity; kept because it is data-driven and general.
        #   2. HEAVY-TAILED innovations -- a Bernoulli-Gaussian spike mixture
        #      whose spike_ratio is CALIBRATED so the finished noise's excess
        #      kurtosis matches the real AR-innovation kurtosis (~14 for the
        #      v4 representative segment; the old v3 value ~68 came from the
        #      anomalous segment 0). An AR + Gaussian-innovation sim cannot
        #      reproduce heavy tails at all (the reason bootstrap was used);
        #      a near-df=4 Student-t can in theory but its sample kurtosis is
        #      wildly unstable.
        #   3. NON-STATIONARITY -- the measured rms_profile envelope (~1.7x
        #      per-segment level drift across the flight, plus local wiggle).
        #      Default envelope_mode="realrate": one rms_profile point per 1000
        #      samples, exactly as measured, so an episode sees the drift at
        #      its TRUE rate (a ~500k-sample episode traverses ~500 of the
        #      16,500 profile points, i.e. about one real flight segment's
        #      worth of drift); a random contiguous slice is taken per episode
        #      so different episodes see different stretches of the flight.
        #      ("stretch" -- squash the whole profile onto the episode --
        #      compresses the drift timescale ~33x and is NOT physical; kept
        #      as an option only.) This whole term is the piece
        #      _generate_signals_from_noise_model() discards.
        #
        # Rescaled at the end to the real noise_std/clean_std severity ratio
        # (~47x), like the bootstrap path. No _add_bursts: the spikes are in
        # the innovations already.
        if self.noise_model is None:
            raise ValueError("generate_accurate_flight_signals needs noise_model_path set in __init__")
        if seed is not None:
            np.random.seed(seed)
        from scipy.signal import lfilter

        nm = self.noise_model
        ar = np.asarray(nm["ar_params"], dtype=np.float64)
        phi = ar[1:] if ar.size > 1 else np.array([])
        resid_std = float(nm.get("ar_resid_std", nm["noise_std"]))
        rp = np.asarray(nm["rms_profile"], dtype=np.float64) if "rms_profile" in nm else None

        N = self.signal_length
        t = np.arange(N) / 1e6
        f_signal = 100_000
        clean = np.sin(2 * np.pi * f_signal * t)

        if innov_excess_kurt is None:
            # prefer the stored AR-innovation kurtosis (extractor v4); fall
            # back to measuring it off the reference snippet for older pickles.
            innov_excess_kurt = nm.get("ar_resid_excess_kurt")
            if innov_excess_kurt is None:
                s = np.asarray(nm["noise_residual_sample"], dtype=np.float64)
                s = s - s.mean()
                innov_excess_kurt = float((s ** 4).mean() / (s.std() ** 4) - 3.0)
        target_k = max(float(innov_excess_kurt), 0.1)

        def build(spike_ratio, m):
            innov = self._bg_innovations(m, spike_prob, spike_ratio) * resid_std
            nz = lfilter([1.0], np.r_[1.0, -phi], innov) if phi.size else innov
            if use_rms_profile and rp is not None:
                nz = nz * self._rms_profile_envelope(m, rp, mode=envelope_mode)
            return nz

        # --- calibrate spike_ratio so the FINISHED noise (post AR + envelope)
        # hits the target excess kurtosis. Monotonic in spike_ratio, so
        # bisect; averaged over 2 draws for stability.
        # Cached: the pipeline is deterministic for a given noise_model, so
        # this is solved once and reused for every subsequent episode.
        #
        # The calibration is done at a FIXED probe length (CALIB_PROBE_N),
        # NOT at the episode's own N, and N is deliberately absent from the
        # cache key. This lets signal_length vary per episode (train_sb.py
        # draws U[len/5, len*2) every reset) without busting the cache and
        # re-running the 12-iteration bisection every episode. Under the
        # default envelope_mode "realrate" the rms-profile envelope's variance
        # does not depend on N (each point covers a fixed 1000 samples), so
        # the calibrated spike_ratio is genuinely N-independent. (Under the
        # legacy "stretch" mode the envelope variance drifts a few percent
        # with N, shifting the realised kurtosis a similar few percent --
        # still inside the 10% bisection tolerance.) ---
        CALIB_PROBE_N = 400_000
        cache_key = (spike_prob, envelope_mode, round(target_k, 2))
        if getattr(self, "_flight_spike_ratio_key", None) == cache_key:
            spike_ratio = self._flight_spike_ratio
        else:
            lo_r, hi_r, spike_ratio = 2.0, 60.0, 8.0
            probe_n = CALIB_PROBE_N
            for _ in range(12):
                spike_ratio = 0.5 * (lo_r + hi_r)
                ks = []
                for _ in range(2):
                    pk = build(spike_ratio, probe_n)
                    pk = pk - pk.mean()
                    ks.append((pk ** 4).mean() / (pk.std() ** 4) - 3.0)
                k_now = float(np.mean(ks))
                if abs(k_now - target_k) < 0.10 * target_k:
                    break
                lo_r, hi_r = (spike_ratio, hi_r) if k_now < target_k else (lo_r, spike_ratio)
            self._flight_spike_ratio_key = cache_key
            self._flight_spike_ratio = spike_ratio

        noise = build(spike_ratio, N) + float(nm.get("noise_mean", 0.0))

        real_ratio = nm["noise_std"] / nm["clean_std"]
        target_std = clean.std() * real_ratio
        cur = noise.std()
        if cur > 0:
            noise = noise * (target_std / cur)

        return clean, clean + noise

# not used
    def _flare_envelope(self, size, flare_len, flare_gain=6.0, n_flares=3,
                            rise_frac=0.15, decay_tau_frac=0.35):
            # Multiplicative envelope on the background noise std: 1.0 baseline,
            # rising to ~flare_gain during "flare" episodes. Solar radio bursts
            # at L/S band (1-3 GHz) raise a receiver's noise floor for minutes
            # to ~1 hour  [Cerruti et al., "Effect of intense December 2006
            # solar radio bursts on GPS receivers", Space Weather 6:S10D07,
            # 2008;  Nita et al., "Radio Frequency Interference from the Sun",
            # 2007].  A literal minute at 1 MHz sampling is 6e7 samples
            # (untrainable), so the ABSOLUTE timescale is compressed -- what is
            # preserved is flare_len >> window_size (noise non-stationary w.r.t.
            # the agent's window) and several flares per signal. Fast linear
            # rise, slow exponential decay, matching flare X-ray/microwave
            # light-curve shape. Overlapping flares ADD in gain (two simultaneous
            # bursts add in power, they do not multiply), so the envelope stays
            # bounded; flare_len is capped to size//4 so even a shortish signal
            # still gets several distinct flares.
            env = np.ones(size)
            flare_len = int(min(flare_len, max(2, size // 4)))
            n_onsets = np.random.poisson(n_flares)
            for _ in range(n_onsets):
                start = np.random.randint(0, size)
                L = max(2, int(flare_len * np.random.uniform(0.5, 1.5)))
                rise = max(1, int(L * rise_frac))
                decay = max(1, L - rise)
                shape = np.empty(L)
                shape[:rise] = np.linspace(0.0, 1.0, rise)
                shape[rise:] = np.exp(-np.arange(decay) / (decay_tau_frac * decay))
                end = min(start + L, size)
                env[start:end] += (flare_gain - 1.0) * shape[:end - start]
            return env
    
    def generate_accurate_signals(self, noise_power=0.1, pink_frac=0.1,
                                  flare_gain=6.0, flare_len=None, n_flares=0.3,
                                  keep_impulsive_bursts=True, seed=None):
        # Physically-motivated cosmic-noise model for the 2.4 GHz (S-band)
        # link in TX.py. Structure:
        #     noisy = clean + (white + pink) * flare_envelope   [+ impulses]
        # Compare _generate_signals(), which is
        #     noisy = clean + white + pink            [+ impulses]
        # with pink actually being BLUE (sqrt|f| filter) and a filtered copy
        # of the same white array (so continuous noise power lands ~2x over
        # the nominal noise_power budget). The differences here:
        #
        #   1. TRUE pink (1/f) instead of blue. At S-band the always-on floor
        #      is receiver thermal noise (galactic background is negligible
        #      above ~1 GHz), so it is WHITE-dominated with only a small 1/f
        #      contribution from SDR flicker / LO phase noise / gain drift ->
        #      pink_frac defaults to 0.1 (was an even split in the first cut;
        #      0.5 is unphysical for an in-band receiver).
        #   2. white and pink are INDEPENDENT draws and the sum is normalised
        #      to exactly noise_power (no correlation artifact).
        #   3. Solar radio bursts -- a multiplicative FRED-shaped envelope on
        #      the noise std, +10..25 dB floor lift for a stretch >> one agent
        #      window. n_flares is the Poisson MEAN of onsets per signal;
        #      0.3 -> ~74% of episodes have no flare, ~23% one, ~3% more,
        #      matching a mostly-quiet Sun with occasional strong events.
        #      [Cerruti et al. 2008; Nita et al. 2007.]
        #   4. short impulsive bursts (sub-ms particle / cosmic-ray strikes,
        #      ISM-band RFI) via the unchanged _add_bursts(), as in
        #      _generate_signals().
        #
        # Knobs:
        #  - pink_frac   : white/pink variance split (0 = all white, 1 = all pink).
        #  - n_flares    : Poisson mean of solar-burst onsets per signal.
        #                  0 -> _flare_envelope() returns all-ones (stationary;
        #                  use this for a clean before/after against
        #                  _generate_signals()).
        #  - flare_gain  : peak noise-std multiplier during a burst (~+15.6 dB
        #                  power at 6.0). Fixed per onset for now; a per-onset
        #                  heavy-tailed draw would be more SOC-realistic.
        #  - flare_len   : burst duration in samples; defaults to
        #                  30 * window_size (>> one window; absolute timescale
        #                  is compressed -- a real ~30 min burst is untrainable).
        if seed is not None:
            np.random.seed(seed)
        N = self.signal_length
        t = np.arange(N) / 1e6          # F_SAMPLING, matches _generate_signals
        f_signal = 100_000             # Hz, matches _generate_signals / TX.py
        clean = np.sin(2 * np.pi * f_signal * t)

        total_std = np.sqrt(noise_power)
        white = np.random.normal(0, total_std * np.sqrt(1.0 - pink_frac), N)
        pink = self._pink_noise_1f(N, total_std * np.sqrt(pink_frac))

        if flare_len is None:
            flare_len = 30 * self.window_size
        env = self._flare_envelope(N, flare_len=flare_len, flare_gain=flare_gain,
                                   n_flares=n_flares)

        noisy = clean + (white + pink) * env
        if keep_impulsive_bursts:
            noisy = self._add_bursts(noisy)

        return clean, noisy

    def step(self, action):
        clean_window = self.clean[self.t:self.t+self.window_size]
        noisy_window = self.noisy[self.t:self.t+self.window_size]

        _, reward, _, info = self.denoiser.step(noisy_window, action, clean_window)

        if self.mode == "train":
            self.t += 100
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
            signal_loss = None
            correlation = None

        info = {
            "filtered_signal": filtered_window,
            "threshold_factor": threshold,
            "reward": reward,
            "SNR_raw": snr_raw,
            "SNR_filtered": snr_filtered,
            "signal_loss": signal_loss,
            "correlation": correlation
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