"""
signal_analysis.py

Characterizes flight-signal (and optionally simulated-signal) data produced
by inference.py's signal-reconstruction export, i.e. files named like:

    Data/20260816_121659_fs1_signal_ws10s1.csv

with columns: Index, Clean Signal, Noisy Signal, filtered_signal

These files already contain aligned, jointly-scaled, real-valued clean/noisy
signals (see inference.py: estimate_alignment + apply_alignment), so no
IQ/alignment work is redone here -- this script only characterizes the
resulting signal/noise properties.

Because clean and noisy were divided by the SAME scale factor S when they
were produced, S cancels out of any *ratio* between them -- so a gain
(attenuation) estimate computed here is still physically meaningful even
though the absolute raw amplitudes aren't directly available in this file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis, skew, normaltest, kstest, ks_2samp, wasserstein_distance, norm
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
FLIGHT_CSV = "C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/flight_signal_1_clean_noisy.csv"
SIMULATED_CSV = "C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/simulated_signal_match_hz.csv"
COMPARE_TO_SIMULATED = True   # set False to only analyze FLIGHT_CSV

CHUNK_SIZE = 100_000            # for stationarity check
BURST_MAD_THRESHOLD = 6.0      # burst = |noise| more than this many MADs from median
OUTPUT_PREFIX = "signal_analysis"


# ── Loading ──────────────────────────────────────────────────────────────
def load_signal_csv(path):
    """Load an inference.py signal-reconstruction CSV, dedupe/sort by
    Index (multiple save_signal_data() flushes can append out of
    order or overlap slightly if the script was interrupted/restarted).
    If the file has no Index column, one is added (0..n-1) based
    on row order before any dedup/sort logic runs."""
    df = pd.read_csv(path)
    if 'Index' not in df.columns:
        df['Index'] = np.arange(len(df))
    df = df.drop_duplicates(subset='Index').sort_values('Index').reset_index(drop=True)
    return df


# ── Core metric functions ───────────────────────────────────────────────
def snr_db(clean, test):
    """SNR of `test` relative to `clean`. Matches the definition used in
    StatelessDenoisingEnv._snr, so numbers here are directly comparable to
    inference.py's logged SNR_raw / SNR_filtered."""
    noise = test - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def estimate_gain(clean, noisy):
    """Least-squares estimate of the scalar gain A such that noisy ≈ A*clean.
    Because clean and noisy share the same normalization scale S from
    inference.py (both divided by S), S cancels out of this ratio:
        A_est = <clean, noisy> / <clean, clean>
              = [<clean_raw, noisy_raw> / S^2] / [<clean_raw, clean_raw> / S^2]
    so this is a valid, scale-invariant attenuation estimate even without
    access to the original raw-amplitude data. This is the real-valued
    counterpart of the complex A_hat computed during alignment."""
    denom = np.dot(clean, clean)
    if denom == 0:
        return np.nan, np.nan
    A_est = np.dot(clean, noisy) / denom
    gain_db = 20 * np.log10(np.abs(A_est)) if np.abs(A_est) > 0 else -np.inf
    return A_est, gain_db


def detect_bursts(noise, mad_threshold=BURST_MAD_THRESHOLD):
    """Identify burst events as contiguous runs of samples where |noise|
    deviates from the median by more than `mad_threshold` MADs (robust to
    heavy-tailed/non-Gaussian noise, unlike a std-based threshold)."""
    med = np.median(noise)
    mad = np.median(np.abs(noise - med)) + 1e-12
    z = np.abs(noise - med) / (1.4826 * mad)  # 1.4826 makes MAD ~consistent with std for Gaussian
    is_burst = z > mad_threshold

    # group contiguous True runs into events
    events = []
    in_event = False
    start = None
    for i, flag in enumerate(is_burst):
        if flag and not in_event:
            in_event = True
            start = i
        elif not flag and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(is_burst) - 1))

    durations = [e[1] - e[0] + 1 for e in events]
    amplitudes = [np.max(np.abs(noise[e[0]:e[1] + 1])) for e in events]

    return {
        "n_events": len(events),
        "rate_per_1000": len(events) / len(noise) * 1000 if len(noise) > 0 else 0,
        "mean_duration": np.mean(durations) if durations else 0,
        "max_duration": np.max(durations) if durations else 0,
        "mean_amplitude": np.mean(amplitudes) if amplitudes else 0,
        "max_amplitude": np.max(amplitudes) if amplitudes else 0,
        "events": events,
    }


def noise_distribution_stats(noise):
    kurt = kurtosis(noise)
    sk = skew(noise)
    _, p_normal = normaltest(noise)
    mu, sigma = noise.mean(), noise.std()
    ks_stat, ks_p = kstest(noise, 'norm', args=(mu, sigma))
    return {
        "kurtosis": kurt, "skew": sk, "p_normal": p_normal,
        "ks_stat": ks_stat, "ks_p": ks_p, "mu": mu, "sigma": sigma,
    }


def stationarity_stats(noise, chunk_size=CHUNK_SIZE):
    chunks = [noise[i:i + chunk_size] for i in range(0, len(noise) - chunk_size, chunk_size)]
    if not chunks:
        return {"chunk_stds": [], "stationary": None}
    chunk_stds = [c.std() for c in chunks]
    chunk_means = [c.mean() for c in chunks]
    cv = np.std(chunk_stds) / (np.mean(chunk_stds) + 1e-12)
    return {
        "chunk_stds": chunk_stds, "chunk_means": chunk_means,
        "std_of_std": np.std(chunk_stds), "mean_std": np.mean(chunk_stds),
        "stationary": cv < 0.1,
    }


def spectral_stats(noise, nperseg=1024):
    freqs, psd = signal.welch(noise, nperseg=min(nperseg, len(noise)))
    log_f = np.log10(freqs[1:] + 1e-12)
    log_psd = np.log10(psd[1:] + 1e-12)
    slope, _ = np.polyfit(log_f, log_psd, 1)
    if abs(slope) < 0.3:
        color = "WHITE noise"
    elif abs(slope) < 1.5:
        color = "PINK/colored noise"
    else:
        color = "RED/strongly colored noise"
    return {"freqs": freqs, "psd": psd, "slope": slope, "color": color}


def autocorrelation(noise, max_lag=20):
    return [np.corrcoef(noise[:-lag], noise[lag:])[0, 1] for lag in range(1, max_lag + 1)]

def compare_noise_distributions(noise_a, noise_b, label_a="Flight", label_b="Simulated"):
    """Direct statistical distance between two noise samples (not vs. Gaussian)."""
    ks_stat, ks_p = ks_2samp(noise_a, noise_b)
    w_dist = wasserstein_distance(noise_a, noise_b)  # "earth mover's distance", same units as the signal

    return {
        "ks_stat": ks_stat,       # 0 = identical distributions, up to 1 = fully separated
        "ks_p": ks_p,             # p < 0.05 -> distributions are statistically distinguishable
        "wasserstein": w_dist,    # avg amount of "mass" you'd have to move to turn one dist into the other
        "std_ratio": noise_a.std() / noise_b.std(),
        "kurtosis_diff": kurtosis(noise_a) - kurtosis(noise_b),
    }


def compare_spectra(psd_a, freqs_a, psd_b, freqs_b):
    """Spectral distance -- interpolate onto a shared frequency grid, then
    compare log-power at each frequency."""
    common_freqs = np.linspace(max(freqs_a[1], freqs_b[1]), min(freqs_a[-1], freqs_b[-1]), 200)
    log_a = np.interp(common_freqs, freqs_a, np.log10(psd_a + 1e-12))
    log_b = np.interp(common_freqs, freqs_b, np.log10(psd_b + 1e-12))
    return {
        "mean_log_power_gap_db": 10 * np.mean(log_a - log_b),  # avg dB difference across spectrum
        "rmse_log_power": np.sqrt(np.mean((log_a - log_b) ** 2)),
    }


# ── Full analysis pipeline for one signal file ──────────────────────────
def analyze_signal(csv_path, label):
    df = load_signal_csv(csv_path)
    clean = df['Clean Signal'].to_numpy(dtype=np.float64)
    noisy = df['Noisy Signal'].to_numpy(dtype=np.float64)
    filtered = df['filtered_signal'].to_numpy(dtype=np.float64) if 'filtered_signal' in df.columns else None

    noise = noisy - clean
    A_est, gain_db = estimate_gain(clean, noisy)

    stats = {
        "label": label,
        "n_samples": len(df),
        "clean": clean, "noisy": noisy, "filtered": filtered, "noise": noise,
        "clean_mean": clean.mean(), "clean_std": clean.std(),
        "noisy_mean": noisy.mean(), "noisy_std": noisy.std(),
        "noise_mean": noise.mean(), "noise_std": noise.std(),
        "A_est": A_est, "gain_db": gain_db,
        "snr_raw_db": snr_db(clean, noisy),
        "distribution": noise_distribution_stats(noise),
        "stationarity": stationarity_stats(noise),
        "spectral": spectral_stats(noise),
        "autocorr": autocorrelation(noise),
        "bursts": detect_bursts(noise),
    }

    if filtered is not None:
        stats["snr_filtered_db"] = snr_db(clean, filtered)
        stats["snr_improvement_db"] = stats["snr_filtered_db"] - stats["snr_raw_db"]

    return stats


# ── Printing ──────────────────────────────────────────────────────────────
def print_report(s):
    print("=" * 70)
    print(f"SIGNAL ANALYSIS — {s['label']}  (N = {s['n_samples']:,} samples)")
    print("=" * 70)

    print("\n-- Amplitude / Attenuation --")
    print(f"  Clean  — mean: {s['clean_mean']:.6g}, std: {s['clean_std']:.6g}")
    print(f"  Noisy  — mean: {s['noisy_mean']:.6g}, std: {s['noisy_std']:.6g}")
    print(f"  Estimated gain (noisy ≈ A·clean): |A| = {abs(s['A_est']):.6g}  ({s['gain_db']:.2f} dB)")

    print("\n-- SNR --")
    print(f"  SNR (raw):      {s['snr_raw_db']:.2f} dB")
    if 'snr_filtered_db' in s:
        print(f"  SNR (filtered): {s['snr_filtered_db']:.2f} dB")
        print(f"  SNR improvement: {s['snr_improvement_db']:.2f} dB")

    d = s['distribution']
    print("\n-- Noise Distribution --")
    print(f"  Kurtosis (excess): {d['kurtosis']:.4f}  (Gaussian=0, impulsive >> 0)")
    print(f"  Skewness:          {d['skew']:.4f}  (symmetric=0)")
    print(f"  Normality p:       {d['p_normal']:.6f}  "
          f"({'likely Gaussian' if d['p_normal'] > 0.05 else 'NOT Gaussian'})")
    print(f"  KS vs Gaussian:    stat={d['ks_stat']:.4f}, p={d['ks_p']:.6f}  "
          f"({'consistent' if d['ks_p'] > 0.05 else 'deviates'})")

    st = s['stationarity']
    print("\n-- Stationarity --")
    if st['stationary'] is None:
        print("  Not enough samples for chunked stationarity check.")
    else:
        print(f"  Noise std across {len(st['chunk_stds'])} chunks — "
              f"mean: {st['mean_std']:.6g}, std-of-std: {st['std_of_std']:.6g}")
        print(f"  Verdict: {'STATIONARY' if st['stationary'] else 'NON-STATIONARY (std drifts)'}")

    sp = s['spectral']
    print("\n-- Spectral --")
    print(f"  PSD log-log slope: {sp['slope']:.4f}  → {sp['color']}")

    ac = s['autocorr']
    correlated = any(abs(v) > 0.05 for v in ac)
    print("\n-- Autocorrelation --")
    print(f"  Lags 1-20 max |corr|: {max(abs(v) for v in ac):.4f}  "
          f"({'correlated (colored)' if correlated else 'uncorrelated (white-ish)'})")

    b = s['bursts']
    print("\n-- Burst / Anomaly Events --")
    print(f"  Detected events:      {b['n_events']}  ({b['rate_per_1000']:.3f} per 1000 samples)")
    if b['n_events'] > 0:
        print(f"  Mean duration:         {b['mean_duration']:.1f} samples")
        print(f"  Max duration:          {b['max_duration']} samples")
        print(f"  Mean peak amplitude:   {b['mean_amplitude']:.4g}")
        print(f"  Max peak amplitude:    {b['max_amplitude']:.4g}")
    print()


def print_comparison(flight, sim):
    print("=" * 70)
    print("FLIGHT vs SIMULATED — SUMMARY COMPARISON")
    print("=" * 70)
    rows = [
        ("SNR (raw, dB)", flight['snr_raw_db'], sim['snr_raw_db']),
        ("Attenuation (dB)", flight['gain_db'], sim['gain_db']),
        ("Noise std", flight['noise_std'], sim['noise_std']),
        ("Noise kurtosis", flight['distribution']['kurtosis'], sim['distribution']['kurtosis']),
        ("Noise skew", flight['distribution']['skew'], sim['distribution']['skew']),
        ("PSD slope (noise color)", flight['spectral']['slope'], sim['spectral']['slope']),
        ("Stationary?", flight['stationarity']['stationary'], sim['stationarity']['stationary']),
        ("Burst rate (/1000 samples)", flight['bursts']['rate_per_1000'], sim['bursts']['rate_per_1000']),
        ("Burst mean amplitude", flight['bursts']['mean_amplitude'], sim['bursts']['mean_amplitude']),
    ]
    print(f"  {'Metric':<28}{'Flight':>18}{'Simulated':>18}")
    print("  " + "-" * 64)
    for name, fv, sv in rows:
        fv_str = f"{fv:.4f}" if isinstance(fv, (int, float, np.floating)) else str(fv)
        sv_str = f"{sv:.4f}" if isinstance(sv, (int, float, np.floating)) else str(sv)
        print(f"  {name:<28}{fv_str:>18}{sv_str:>18}")

    gap = flight['snr_raw_db'] - sim['snr_raw_db']
    print(f"\n  SNR gap (flight − simulated): {gap:.2f} dB")
    print(f"  → the flight environment is ~{abs(gap):.1f} dB harder than the simulator's validation case.")
    dist_cmp = compare_noise_distributions(flight['noise'], sim['noise'])
    spec_cmp = compare_spectra(flight['spectral']['psd'], flight['spectral']['freqs'],
                                sim['spectral']['psd'], sim['spectral']['freqs'])

    print("\n-- Direct Distribution Distance (Flight vs Simulated) --")
    print(f"  KS statistic:        {dist_cmp['ks_stat']:.4f}  (0=identical, 1=fully separated)")
    print(f"  KS p-value:          {dist_cmp['ks_p']:.2e}  "
          f"({'DISTINGUISHABLE' if dist_cmp['ks_p'] < 0.05 else 'not statistically distinguishable'})")
    print(f"  Wasserstein distance: {dist_cmp['wasserstein']:.4f}")
    print(f"  Std ratio (flight/sim): {dist_cmp['std_ratio']:.2f}x")
    print(f"  Kurtosis gap:          {dist_cmp['kurtosis_diff']:.2f}")

    print("\n-- Direct Spectral Distance --")
    print(f"  Mean log-power gap:  {spec_cmp['mean_log_power_gap_db']:.2f} dB")
    print(f"  RMSE (log power):    {spec_cmp['rmse_log_power']:.4f}")


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_single(s, prefix=OUTPUT_PREFIX):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Signal Analysis — {s['label']}", fontsize=14)

    axes[0, 0].plot(s['noise'][:5000], lw=0.5, color='steelblue')
    axes[0, 0].set_title("Noise (first 5000 samples)")
    axes[0, 0].set_xlabel("Sample"); axes[0, 0].set_ylabel("Amplitude")

    d = s['distribution']
    axes[0, 1].hist(s['noise'], bins=100, density=True, alpha=0.7, color='steelblue', label=s['label'])
    x_fit = np.linspace(s['noise'].min(), s['noise'].max(), 300)
    axes[0, 1].plot(x_fit, norm.pdf(x_fit, d['mu'], d['sigma']), 'r--', lw=2, label='Gaussian fit')
    axes[0, 1].set_title("Noise Distribution"); axes[0, 1].legend()

    sp = s['spectral']
    axes[0, 2].loglog(sp['freqs'][1:], sp['psd'][1:], color='steelblue')
    axes[0, 2].set_title(f"Noise PSD (slope={sp['slope']:.2f})")
    axes[0, 2].set_xlabel("Frequency"); axes[0, 2].set_ylabel("Power")

    axes[1, 0].bar(range(1, 21), s['autocorr'], color='steelblue')
    axes[1, 0].axhline(0.05, color='r', ls='--', lw=1)
    axes[1, 0].axhline(-0.05, color='r', ls='--', lw=1)
    axes[1, 0].set_title("Noise Autocorrelation"); axes[1, 0].set_xlabel("Lag")

    st = s['stationarity']
    axes[1, 1].plot(st['chunk_stds'], color='steelblue')
    axes[1, 1].set_title("Noise Std per Chunk (Stationarity)")
    axes[1, 1].set_xlabel("Chunk"); axes[1, 1].set_ylabel("Std")

    if s['filtered'] is not None:
        err_raw = s['noisy'] - s['clean']
        err_filt = s['filtered'] - s['clean']
        axes[1, 2].hist(err_raw, bins=100, density=True, alpha=0.5, color='red', label='raw error')
        axes[1, 2].hist(err_filt, bins=100, density=True, alpha=0.5, color='green', label='filtered error')
        axes[1, 2].set_title("Raw vs Filtered Error"); axes[1, 2].legend()
    else:
        axes[1, 2].hist(s['clean'], bins=100, density=True, color='orange', alpha=0.7)
        axes[1, 2].set_title("Clean Signal Distribution")

    plt.tight_layout()
    fname = f"{prefix}_{s['label'].lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"[INFO] Saved plot: {fname}")
    plt.show()


def plot_comparison(flight, sim, prefix=OUTPUT_PREFIX):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Flight vs Simulated — Noise Comparison", fontsize=14)

    axes[0].hist(flight['noise'], bins=100, density=True, alpha=0.5, color='steelblue', label='Flight')
    axes[0].hist(sim['noise'], bins=100, density=True, alpha=0.5, color='orange', label='Simulated')
    axes[0].set_title("Noise Distribution Overlay"); axes[0].legend()

    axes[1].loglog(flight['spectral']['freqs'][1:], flight['spectral']['psd'][1:],
                    color='steelblue', label='Flight')
    axes[1].loglog(sim['spectral']['freqs'][1:], sim['spectral']['psd'][1:],
                    color='orange', label='Simulated')
    axes[1].set_title("Noise PSD Overlay"); axes[1].legend()
    axes[1].set_xlabel("Frequency"); axes[1].set_ylabel("Power")

    plt.tight_layout()
    fname = f"{prefix}_comparison.png"
    plt.savefig(fname, dpi=150)
    print(f"[INFO] Saved plot: {fname}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flight_stats = analyze_signal(FLIGHT_CSV, "Flight Signal")
    print_report(flight_stats)
    plot_single(flight_stats)

    if COMPARE_TO_SIMULATED:
        sim_stats = analyze_signal(SIMULATED_CSV, "Simulated Signal")
        print_report(sim_stats)
        plot_single(sim_stats)

        print_comparison(flight_stats, sim_stats)
        plot_comparison(flight_stats, sim_stats)