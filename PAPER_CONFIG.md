# PAPER_CONFIG — reproducibility record

Single source of truth for the numbers and settings the paper needs to report.
Generated / verified 2026-09-01. Update this file whenever a config value
changes.

---

## 1. Software environment

| package | version |
|---|---|
| Python | 3.11.1 |
| stable-baselines3 | 1.6.2 |
| gym | 0.21.0 |
| numpy | 1.26.3 |
| scipy | 1.14.1 |
| pandas | 2.2.3 |
| PyWavelets (pywt) | 1.8.0 |

OS: Windows 11. RNG: `numpy.random` legacy global (`np.random.seed`) plus
Python `random.seed`; SAC seeded via SB3 `seed=` argument.

---

## 2. Signal chain / hardware (from `Scripts/SDR/TX.py`, `RX.py`)

| quantity | value |
|---|---|
| carrier (center_freq) | 2.400 GHz |
| sample rate | 1.000 MHz (1 Msps), complex IQ |
| transmitted waveform | single complex sinusoid, **100 kHz** baseband tone, unit amplitude, 0 phase (`analog.sig_source_c`, `GR_SIN_WAVE`) |
| TX radio | HackRF, VGA gain 25 dB (of 47), RF amp OFF |
| RX radio | HackRF, LNA gain 40 dB (max), VGA gain 0 dB (of 62), RF amp OFF |
| per-run capture cap | 50e6 samples (`blocks.head`) |
| GNU Radio | 3.10.10.0 |

No oscillator tolerance is specified in TX.py/RX.py. HackRF stock TCXO is
~20 ppm → up to ~48 kHz each at 2.4 GHz worst case. Observed per-segment CFO
in flight_signal_1 drifts up to ~14 kHz. **CFO search band = 30 kHz**
everywhere (`build_clean_noisy.py CFO_SEARCH_HZ`, `inference.py` /
`static_filter_realtime.py estimate_alignment(search_hz=)`).

---

## 3. Wavelet denoising filter (`StatelessDenoisingEnv.apply_filter`)

| parameter | value |
|---|---|
| wavelet | Daubechies-4 (`db4`) |
| decomposition level | 5 (capped at `pywt.dwt_max_level`) |
| DWT mode | `periodization` |
| threshold rule | soft, VisuShrink form: `lam = threshold_factor * sigma * sqrt(2 ln n)` |
| sigma estimate | `median(|finest detail coeffs|) / 0.6745`, floored at 1e-8 |
| coefficients thresholded | detail bands only (approximation kept) |
| post-processing | mean re-added (`y += mean(x) - mean(y)`), `nan_to_num` |
| filter window length `n` | 1000 samples |

The sigma floor (1e-8) is required: heavily quantized flight input drives the
finest detail band to >50% exact zeros → `median = 0` → every
`threshold_factor` collapses to an identical degenerate output without it.

---

## 4. RL problem definition (`NoiseReductionEnv`)

| item | value |
|---|---|
| observation | current 1000-sample noisy window (`Box(-inf, inf, (1000,))`) |
| action | scalar in `[-1, 1]` (`Box(-1, 1, (1,))`) |
| action → threshold_factor | linear map `[-1, 1] → [0.05, 2.5]` |
| reward | `snr_improvement - 1.25 * signal_loss + 0.25 * correlation` |
| `snr_improvement` | `SNR(clean, filtered) - SNR(clean, noisy)` per window, dB |
| `signal_loss` | `log1p(mean((filtered - clean)^2))` |
| `correlation` | Pearson `corr(filtered, clean)` (0 if NaN) |
| training stride | `t += 100` samples per env step |
| inference stride | 100 samples (`inference.py`, `static_filter_realtime.py`) |
| episode end (train) | window reaches end of signal, or last-10 reward std < 1e-7 |
| train-time normalization | clean & noisy divided by `max(|clean|)` |

Reward-weight rationale: **see `paper-notes/item4_reward_weights.md`** — one
open question flagged for the author there.

---

## 5. SAC training (`train_sb.py`)

Single seed per scheme (SEED = 42). `SAC("MlpPolicy", ...)` with **all SB3
1.6.2 defaults**:

| hyperparameter | value |
|---|---|
| learning_rate | 3e-4 |
| buffer_size | 1_000_000 |
| learning_starts | 100 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |
| **train_freq** | **(4, "step")** — overridden from default 1 |
| **gradient_steps** | **1** — set explicitly alongside train_freq |
| ent_coef | "auto" |
| target_update_interval | 1 |
| net_arch | [256, 256], ReLU (SB3 SAC default) |

`train_freq=(4,"step")` + `gradient_steps=1` → 1 gradient update per 4 env
steps (update-to-data ratio 0.25 vs the default 1.0). ~250k gradient updates
over a 1M-env-step run instead of ~1M → ~3–4× faster on this CPU-only box
(~7 h/model). Standard off-policy knob; mild sample-inefficiency cost,
expected small given the near-bandit reward landscape. Both parameters are set
explicitly because SB3's `gradient_steps=-1` default (other versions) would
otherwise keep the ratio at 1:1.

Run settings (identical for both schemes):

| setting | value |
|---|---|
| total timesteps | 1_000_000 |
| window size | 1000 |
| train signal length (nominal) | 500_000; env redraws `U[100_000, 1_000_000)` per episode |
| eval signal length (nominal) | 150_000; also redrawn per episode |
| eval frequency | every 50_000 steps (20 evals over 1M) |
| eval episodes | 10, deterministic |
| early stop | `StopTrainingOnNoModelImprovement(max_no_improvement_evals=6, min_evals=8)` |
| checkpoints | every 50_000 steps; `recent_` saves every 100_000 after 500_000 |

Model selection (Q5): **use the `final` model** provided the eval curve shows
a plateau and no late collapse. Verify with
`RL-EnvConfig/plot_eval_curve.py models/best_<RUN>` — it reads
`evaluations.npz`, prints the eval-reward-vs-timestep table, the best-vs-final
gap, a plateau/collapse flag, and saves `models/eval_curve_<RUN>.png`. If
`final` is within ~1 eval-std of `best` and the tail is flat → keep `final`;
if `final ≪ best` → use `best_` and note the degradation. Report the curve
and the choice in the paper.

### Naming
`SCHEME = "OAN"` (original accurate noise) or `"PFN"` (parametric flight
noise). Run tag: `<SCHEME>_W1000_<timestamp>_<timesteps>`.
Legacy artifacts on disk use the old `OFT_ACC` / `UN_ACC` / `OFT` / `UN`
tags — those predate this naming and are not the paper models.

---

## 6. Noise schemes

### OAN — `generate_accurate_signals()` defaults
| parameter | value |
|---|---|
| clean | 100 kHz sine, 1 Msps, unit amplitude |
| noise_power budget | 0.1 (variance) |
| pink fraction | 0.1 (10% of noise variance is true 1/f pink, 90% white) |
| pink synthesis | random-phase 1/f with low-freq corner at bin 1, renormalized to target std |
| flares | multiplicative FRED envelope on noise std; `flare_gain = 6.0` (~+15.6 dB), `flare_len = 30 * window = 30_000` samples, onset count `~Poisson(0.3)` per signal |
| impulsive bursts | `_add_bursts`: p=0.01/sample, amplitude U(-3, 3), duration 10 |

### PFN — `generate_accurate_flight_signals()` defaults
| parameter | value |
|---|---|
| source model | `Data/noise_model_fs1.pkl` (extractor v4, segment 0 excluded) |
| colour | AR(50), coefficients = `ar_params[1:]` as all-pole filter |
| innovations | Bernoulli-Gaussian spike mixture, `spike_prob = 0.03`, `spike_ratio` bisected so finished-noise excess kurtosis matches `ar_resid_excess_kurt` |
| non-stationarity | measured `rms_profile` envelope, **`envelope_mode = "realrate"`** (default): one profile point per 1000 samples, random contiguous slice per episode → drift at its true rate (~one flight segment's worth per episode) |
| severity rescale | final noise scaled to `noise_std/clean_std` from the pkl (~47.5) |
| spike-ratio calibration | done once at a fixed 400_000-sample probe, cached with N absent from the key; under `realrate` the envelope variance is N-independent so this is exact |

Q3 resolved: `"realrate"` is the physically faithful choice (`"stretch"`
compresses the whole-flight drift onto each ≤1 s episode, ~33× too fast).
Slice-dependence: realised noise excess kurtosis is 9–14 / drift 1.5–2.2 for
5 of 6 sampled seeds (on target with the AR-innovation kurtosis ~14 and
per-segment drift ~1.7), one spiky-slice outlier ~150 / ~5.6. See
`paper-notes/OPEN_QUESTIONS.md` Q3.

---

## 7. `noise_model_fs1.pkl` (extractor v4) — key values

Extractor: `Scripts/Intermediate-Scripts/noise_extractor.py`, reads
`Data/flight_signal_1_clean_noisy.csv`. Segment 0 excluded (5x-low RX gain
startup outlier, per-segment excess kurtosis ~6500). Characterization segment:
index 17 (mid-flight), first 50_000 samples. AR fit lags = 50.

| key | value |
|---|---|
| `ar_lags` | 50 |
| `ar_resid_std` | 0.22313 |
| `ar_resid_excess_kurt` | 13.92 |
| `noise_std` (usable segs) | 0.22676 |
| `noise_mean` | -0.05240 |
| `clean_std` | 0.0047758 |
| severity ratio `noise_std/clean_std` | 47.48 |
| `global_excess_kurt` (all usable segs) | 634.4 |
| `psd_flatness` | 0.1540 |
| `psd_slope` (log-log, 1–300 kHz) | -0.0377 (≈ white) |
| `autocorr_lag1` | 0.0791 |
| `rms_profile` | 16_500 points (1000-sample blocks); min 0.105 / mean 0.226 / max 3.61 |
| `per_segment_std_ratio` | 1.716 |
| `n_segments` | 34 (`excluded_segments = [0]`) |
| `char_segment_index` | 17 |

`noise_model_fs1_SEG0.pkl.bak` = the superseded v3 model (segment 0 included);
kept only for provenance.

---

## 8. Datasets

| file | what it is | provenance |
|---|---|---|
| `Data/flight_signal_1.csv` | raw flight capture, TX/RX IQ + magnitudes + ISO timestamps, 34 stitched ~500k segments, ~17e6 rows | field recording |
| `Data/flight_signal_1_clean_noisy.csv` | offline per-segment aligned Clean/Noisy real columns | `build_clean_noisy.py` — **analysis only, never fed to models/filters** |
| `Data/flight_signal_2.csv` / `_clean_noisy.csv` | a second, shorter flight capture | field recording — *not currently used; candidate held-out test for PFN* |
| `Data/simulated_signal_oan.csv` | OAN-scheme eval signal | `generate_simulated_signal_oan.py`, **seed 20260904** (first flare-containing draw: 4 runs, peak/median RMS 3.19), 500_000 samples |
| `Data/simulated_signal_pfn.csv` | PFN-scheme eval signal | `generate_simulated_signal_pfn.py`, **seed 20260907** (representative realrate slice: kurt 14.4, drift 2.2), 500_000 samples |
| `Data/simulated_signal_match_hz.csv` | legacy synthetic (white + *blue* "pink" + bursts) validation signal | `generate_simulated_signal.py`, seed 42 |
| `Data/simulated_signal_un_noise_model.csv` | legacy block-bootstrap flight-noise validation signal | `generate_un_validation_signal.py`, seed 123 |
| `Data/noise_model_fs1.pkl` | PFN parametric noise model (v4) | `noise_extractor.py` |
| `Data/signal_property_comparison.csv` | item-3 property table | `compare_signal_properties.py` |

### Measured signal properties (`compare_signal_properties.py`, 2026-09-01, regenerated sims)

| property | OAN_sim | PFN_sim | flight_1 |
|---|---|---|---|
| n samples | 500,000 | 500,000 | 17,000,000 |
| segments | 1 | 1 | 34 |
| duration (s, incl. dead gaps) | 0.5 | 0.5 | 3505.6 |
| clean tone amplitude est. | 1.00 | 1.00 | 0.0067 |
| clean dominant freq (Hz) | 100,000 | 100,000 | 126,900* |
| noise std | 0.664 | 33.57 | 0.226 |
| severity ratio (noise/clean std) | 0.94 | 47.5 | 47.8 |
| whole-signal SNR (dB) | +0.54 | −33.74 | −33.81 |
| per-window SNR mean ± std (dB) | +1.01 ± 1.82 | −33.39 ± 1.86 | −34.00 ± 2.75 |
| noise excess kurtosis | 9.4 | 14.4 | 769** |
| noise PSD log-log slope | −1.03 (true 1/f pink) | −0.05 (white) | −0.04 (white) |
| noise ACF lag-1 | 0.64 | 0.03 | 0.12 |
| rolling-RMS drift ratio (p99/p1) | 3.3 (incl. flare) | 2.2 | 2.8 |
| impulsive-burst runs per 1k samples | 1.20 | 4.46 | 0.021 |
| noisy exact-zero fraction | 0 | 0 | **0.949** |

\* CFO: the reconstructed flight clean reference is TX shifted by the fitted
per-segment CFO; the analyzed leading chunk sits ~27 kHz off nominal.
\*\* flight kurtosis 769 is a *global* figure dominated by inter-segment level
shifts + the segment-0 transient + ADC spikes. The *within-segment*
AR-innovation excess kurtosis is ~14 (`ar_resid_excess_kurt`), which is what
PFN targets.

---

## 9. Inference / baseline runtime constants

`inference.py` and `static_filter_realtime.py`:

| constant | value |
|---|---|
| window size | 1000 |
| stride | 100 |
| CFO search band | 30_000 Hz |
| flight calibration window | 20_000 samples |
| sim calibration window | 1_000 samples |
| recalibration interval | 100_000 samples |
| amplitude-scale EWMA alpha | 0.3 |
| flight clean rebuild | `A_hat * TX * exp(j 2π df_hat n / fs)`, DC removed from RX, common scale |

`static_filter_realtime.py`: `MODE="sweep"` searches `threshold_factor` in
`linspace(0.05, 6.0, 80)` over up to 3000 spread windows, then `MODE="apply"`
filters every window at the chosen fixed value. Best static `threshold_factor`
per source (fill from sweep runs):

| source | best tf (unconstrained) | best tf (≤ 2.5) | mean SNR improvement (dB) | n windows |
|---|---|---|---|---|
| oan_sim (seed 20260904) | 0.276 | 0.276 | +1.007 ± 0.443 | 4991 (all) |
| pfn_sim (seed 20260907) | 0.427 | 0.427 | +6.630 ± 0.808 | 4991 (all) |
| flight  | 6.0 (ceiling; SNR improvement monotonic in tf, no interior optimum) | 2.460 | +0.126 ± 0.669 (tf 2.46) — indistinguishable from zero | 169,991 (full pass, all 34 segments incl. seg 0) |

Flight: the db4 wavelet filter is **near-inert** on flight_signal_1 (95% exact
zeros → detail bands ~empty → σ floored → threshold ≈ 0 regardless of tf;
output/input energy ratio stays ≈ 0.97 even at tf = 6). See
`paper-notes/item10_static_filter_baseline.md`.

---

## 10. Known caveats to state in the paper

- **Single training seed per scheme.** RL is high-variance; results are one
  draw.
- **PFN is extracted from and evaluated on the same flight** (`flight_signal_1`);
  optimistic. `flight_signal_2` exists as a possible held-out test.
- **PFN does not reproduce**: the 95% exact-zero ADC quantization of the real
  RX, or the cross-segment level jumps (global kurtosis 769 vs PFN's 12).
- **OAN eval signal drew 0 solar-flare episodes** at seed 20260901 (Poisson
  mean 0.3 → ~74% chance of none). See `paper-notes/item2` note.
- **Flare timescale is compressed** — a real ~30-min solar radio burst is
  30_000 samples here, not 1.8e9.
- **Legacy "pink" noise was actually blue** (`sqrt|f|` filter); OAN's
  `_pink_noise_1f` corrects it to true 1/f (measured slope −1.08 vs legacy +1).
- Flight metrics depend on the live-alignment constants in §9; sensitivity not
  yet swept.
- `envelope_mode="stretch"` distorts the PFN drift timescale.
- **The wavelet filter has almost no leverage on flight_signal_1** (95% exact
  zeros → empty detail bands → threshold ≈ 0 for any `threshold_factor`). Both
  the static baseline and the RL agent's action are close to inconsequential
  on the real flight signal — a scheme-independent explanation for the flight
  performance gap that the paper should address directly.

---

## 11. Where the paper artifacts live

| artifact | path |
|---|---|
| this record | `PAPER_CONFIG.md` |
| flight signal description (item 1) | `paper-notes/item1_flight_signal_description.md` |
| signal comparison writeup + table (item 3) | `paper-notes/item3_signal_comparison.md` |
| reward-weights explanation (item 4) | `paper-notes/item4_reward_weights.md` |
| PFN noise model writeup (item 7) | `paper-notes/item7_pfn_noise_model.md` |
| property comparison data | `Data/signal_property_comparison.csv` |
| static-filter sweep results | `Data/<timestamp>_<tag>_static_sweep.csv` |
| open questions for the author | `paper-notes/OPEN_QUESTIONS.md` |
