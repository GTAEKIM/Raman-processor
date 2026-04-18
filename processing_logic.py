import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import f as f_dist
from typing import Tuple, Dict, Any, Callable, Optional, List
import os
import logging
import threading
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from pybaselines.morphological import mor
from pybaselines.whittaker import airpls, arpls, asls
from pybaselines.smooth import snip
from joblib import Parallel, delayed

# lmfit is optional at import time — peak fitting degrades gracefully if absent
try:
    from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel
    _LMFIT_AVAILABLE = True
except ImportError:
    _LMFIT_AVAILABLE = False
    logging.warning("lmfit not installed — peak fitting disabled.")


class ProcessingStage(Enum):
    RAW = "raw"
    SMOOTHED = "smoothed"
    BASELINE_CORRECTED = "baseline_corrected"
    NORMALIZED = "normalized"


# MAD scaling constant for normal distribution (1 / Q(0.75) of standard normal)
MAD_SCALE_FACTOR = 0.6745

# Polynomial baseline iteration defaults
BASELINE_MAX_ITERATIONS = 100
BASELINE_CONVERGENCE_TOL = 1e-9
BASELINE_ALPHA = 0.99 * 0.5

# Algorithm family mapping for baseline dispatch
POLYNOMIAL_ALGOS = {'atq', 'stq', 'ah', 'sh'}
WHITTAKER_ALGOS = {'airpls', 'arpls', 'asls'}

# Supported normalization methods
NORMALIZATION_METHODS = {'none', 'snv', 'vector', 'area', 'minmax', 'maxpeak'}

# PCA scaling methods
PCA_SCALING_METHODS = {'auto', 'mean', 'pareto', 'none'}

# Peak profile models (requires lmfit)
PEAK_PROFILES = {'gaussian', 'lorentzian', 'voigt', 'pseudovoigt'}


# ───────────────────────── Derivative Spectra ─────────────────────────


def compute_derivative(
    y: np.ndarray,
    order: int = 1,
    window: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute nth-order derivative spectrum using Savitzky-Golay.

    Derivative spectra remove slow-varying baselines naturally and
    enhance sharp Raman features. Common: 1st-order for baseline removal,
    2nd-order for peak sharpening / overlapped-peak resolution.
    """
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, len(y) - 1))
    if window % 2 == 0:
        window -= 1
    polyorder = max(order + 1, min(polyorder, window - 1))
    return savgol_filter(y, window_length=window, polyorder=polyorder, deriv=order)


# ───────────────────────── Quality Control ─────────────────────────


def compute_snr(
    y: np.ndarray,
    x: np.ndarray,
    noise_region: Tuple[float, float] = (1800.0, 2000.0),
) -> float:
    """Estimate SNR (dB) from signal peak-to-peak vs. std in a silent Raman region.

    Default noise region is the Raman silent region (~1800-2000 cm^-1) which
    contains no vibrational features for most organic/biological samples.
    Returns np.inf if noise is zero, NaN if region is empty.
    """
    mask = (x >= noise_region[0]) & (x <= noise_region[1])
    if not np.any(mask):
        return float('nan')
    noise_std = float(np.std(y[mask]))
    signal_pp = float(np.max(y) - np.min(y))
    if noise_std == 0:
        return float('inf')
    return 20.0 * np.log10(signal_pp / noise_std)


def detect_saturation(y: np.ndarray, relative_threshold: float = 0.98) -> bool:
    """Heuristic: flag if any point is within `relative_threshold` of max,
    and flat plateaus of ≥3 consecutive points at max exist.
    """
    if len(y) < 3:
        return False
    y_max = float(np.max(y))
    if y_max <= 0:
        return False
    flat = np.abs(y - y_max) <= (1.0 - relative_threshold) * y_max
    # Flag if ≥3 consecutive saturated samples
    run = 0
    for f in flat:
        if f:
            run += 1
            if run >= 3:
                return True
        else:
            run = 0
    return False


def compute_spectrum_qc(
    y_raw: np.ndarray,
    y_final: np.ndarray,
    x: np.ndarray,
    noise_region: Tuple[float, float] = (1800.0, 2000.0),
    sat_threshold: float = 0.98,
) -> Dict[str, Any]:
    """Return a per-spectrum QC dict."""
    snr = compute_snr(y_final, x, noise_region=noise_region)
    saturated = detect_saturation(y_raw, relative_threshold=sat_threshold)
    has_nan = bool(np.any(~np.isfinite(y_final)))
    flag = "OK"
    if saturated:
        flag = "SATURATED"
    elif has_nan:
        flag = "NAN_OR_INF"
    elif np.isfinite(snr) and snr < 10.0:
        flag = "LOW_SNR"
    return {
        "snr_db": snr,
        "saturated": saturated,
        "has_nan": has_nan,
        "flag": flag,
    }


# ───────────────────────── Peak Detection & Fitting ─────────────────────────


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    prominence: Optional[float] = None,
    prominence_percent: float = 2.0,
    min_distance_pts: int = 5,
    height: Optional[float] = None,
    width_pts: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect peaks via scipy.signal.find_peaks.

    prominence_percent: if `prominence` is None, uses percent-of-(max-min) heuristic.
    Returns dict with indices, x positions, heights, prominences, and FWHM (cm^-1).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if prominence is None:
        y_range = float(np.max(y) - np.min(y))
        prominence = max(1e-12, prominence_percent / 100.0 * y_range)

    peaks, props = find_peaks(
        y,
        prominence=prominence,
        distance=max(1, int(min_distance_pts)),
        height=height,
        width=width_pts,
    )

    # FWHM in x-units
    if len(peaks) > 0:
        widths_pts, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
        # Convert fractional index positions to x via linear interpolation
        def _idx_to_x(idx_frac):
            lo = int(np.floor(idx_frac))
            hi = min(lo + 1, len(x) - 1)
            frac = idx_frac - lo
            return x[lo] * (1 - frac) + x[hi] * frac
        left_x = np.array([_idx_to_x(i) for i in left_ips])
        right_x = np.array([_idx_to_x(i) for i in right_ips])
        fwhm_x = np.abs(right_x - left_x)
    else:
        fwhm_x = np.array([])

    return {
        "indices": peaks,
        "positions": x[peaks] if len(peaks) else np.array([]),
        "heights": y[peaks] if len(peaks) else np.array([]),
        "prominences": props.get("prominences", np.array([])),
        "fwhm": fwhm_x,
        "prominence_used": prominence,
    }


def fit_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peak_positions: np.ndarray,
    profile: str = 'gaussian',
    fit_window: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit a multi-peak model using lmfit.

    profile: 'gaussian' | 'lorentzian' | 'voigt' | 'pseudovoigt'
    fit_window: if given, restrict fit to x within [min(pos)-w, max(pos)+w] cm^-1.
    """
    if not _LMFIT_AVAILABLE:
        raise RuntimeError("lmfit is not installed. Run: pip install lmfit")
    if profile not in PEAK_PROFILES:
        raise ValueError(f"Unknown peak profile: {profile}")
    if len(peak_positions) == 0:
        raise ValueError("No peaks supplied for fitting.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if fit_window is not None:
        lo = float(np.min(peak_positions)) - fit_window
        hi = float(np.max(peak_positions)) + fit_window
        mask = (x >= lo) & (x <= hi)
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y

    model_cls = {
        'gaussian': GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt': VoigtModel,
        'pseudovoigt': PseudoVoigtModel,
    }[profile]

    composite = None
    params = None
    x_range = float(np.max(x_fit) - np.min(x_fit))
    approx_sigma = max(x_range / 100.0, 1.0)

    for i, pos in enumerate(peak_positions):
        prefix = f"p{i}_"
        m = model_cls(prefix=prefix)
        # Amplitude guess from y near pos
        near_idx = int(np.argmin(np.abs(x_fit - pos)))
        approx_amp = float(y_fit[near_idx]) * approx_sigma * 2.5
        p = m.make_params()
        p[f"{prefix}center"].set(value=float(pos), min=float(pos) - approx_sigma * 5,
                                 max=float(pos) + approx_sigma * 5)
        p[f"{prefix}sigma"].set(value=approx_sigma, min=1e-6, max=x_range)
        p[f"{prefix}amplitude"].set(value=max(approx_amp, 1e-6), min=0)
        if composite is None:
            composite = m
            params = p
        else:
            composite = composite + m
            params.update(p)

    result = composite.fit(y_fit, params, x=x_fit)

    # Per-peak summary
    peaks_out: List[Dict[str, Any]] = []
    for i in range(len(peak_positions)):
        prefix = f"p{i}_"
        c = result.params.get(f"{prefix}center")
        s = result.params.get(f"{prefix}sigma")
        a = result.params.get(f"{prefix}amplitude")
        # FWHM / height: use computed component result for robustness
        comp_vals = result.eval_components(x=x_fit).get(prefix, None)
        if comp_vals is not None and len(comp_vals) > 0:
            height = float(np.max(comp_vals))
            area = float(np.trapezoid(comp_vals, x_fit))
        else:
            height = float('nan')
            area = float('nan')
        # FWHM conversions
        if profile == 'gaussian':
            fwhm = float(2.3548200 * s.value) if s is not None else float('nan')
        elif profile == 'lorentzian':
            fwhm = float(2.0 * s.value) if s is not None else float('nan')
        else:
            fwhm = float(2.3548200 * s.value) if s is not None else float('nan')
        peaks_out.append({
            "index": i,
            "center": float(c.value) if c is not None else float('nan'),
            "sigma": float(s.value) if s is not None else float('nan'),
            "amplitude": float(a.value) if a is not None else float('nan'),
            "fwhm": fwhm,
            "height": height,
            "area": area,
        })

    ss_res = float(np.sum((y_fit - result.best_fit) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        "profile": profile,
        "peaks": peaks_out,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "best_fit": result.best_fit,
        "components": result.eval_components(x=x_fit),
        "r_squared": r_squared,
        "chisqr": float(result.chisqr),
        "redchi": float(result.redchi),
        "n_peaks": len(peak_positions),
        "report": result.fit_report(),
    }


# ───────────────────────── PCA Scaling Helpers ─────────────────────────


def apply_pca_scaling(data: np.ndarray, method: str = 'auto') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply scaling before PCA.

    - 'auto'   : mean-center + unit variance (StandardScaler default)
    - 'mean'   : mean-center only (preserves relative variance)
    - 'pareto' : mean-center, divide by sqrt(std) — between 'mean' and 'auto'
    - 'none'   : no scaling
    """
    if method not in PCA_SCALING_METHODS:
        raise ValueError(f"Unknown PCA scaling method: {method}")

    data = np.asarray(data, dtype=float)
    if method == 'auto':
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        return scaled, {"mean_": scaler.mean_, "scale_": scaler.scale_}
    if method == 'mean':
        mean_ = data.mean(axis=0)
        return data - mean_, {"mean_": mean_, "scale_": np.ones_like(mean_)}
    if method == 'pareto':
        mean_ = data.mean(axis=0)
        std_ = data.std(axis=0)
        sqrt_std = np.sqrt(np.where(std_ == 0, 1.0, std_))
        return (data - mean_) / sqrt_std, {"mean_": mean_, "scale_": sqrt_std}
    return data.copy(), {"mean_": np.zeros(data.shape[1]), "scale_": np.ones(data.shape[1])}


# ───────────────────────── Normalization ─────────────────────────


def normalize_spectrum(y: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize a single spectrum using the specified method.

    - snv: (y - mean) / std
    - vector: y / ||y||_2
    - area: y / integral(|y|)
    - minmax: (y - min) / (max - min)
    - maxpeak: y / max(y)
    - none: returns y unchanged
    """
    if method == 'none' or method is None:
        return y

    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"Unknown normalization method: {method}")

    y = np.asarray(y, dtype=float)

    if method == 'snv':
        std = np.std(y)
        if std == 0:
            return np.zeros_like(y)
        return (y - np.mean(y)) / std

    if method == 'vector':
        norm = np.linalg.norm(y)
        if norm == 0:
            return y.copy()
        return y / norm

    if method == 'area':
        area = np.trapezoid(np.abs(y))
        if area == 0:
            return y.copy()
        return y / area

    if method == 'minmax':
        y_min, y_max = np.min(y), np.max(y)
        rng = y_max - y_min
        if rng == 0:
            return np.zeros_like(y)
        return (y - y_min) / rng

    if method == 'maxpeak':
        y_max = np.max(y)
        if y_max == 0:
            return y.copy()
        return y / y_max

    return y


# ───────────────────────── DataProcessor ─────────────────────────


class DataProcessor:
    def __init__(self):
        self.x_full: Optional[np.ndarray] = None
        self.y_raw_full: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None
        self.sample_names: list[str] = []
        self._lock = threading.Lock()
        logging.info("DataProcessor initialized.")

    def load_data(self, filepath: str) -> int:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in ('.csv',):
                df_raw = pd.read_csv(filepath, header=None)
            elif ext in ('.xlsx', '.xls'):
                df_raw = pd.read_excel(filepath, header=None)
            elif ext in ('.txt', '.asc', '.dat'):
                df_raw = _load_text_file(filepath)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            logging.error(f"Failed to load data from {filepath}: {e}")
            raise ValueError(f"Could not read file. Error: {e}")

        if df_raw.shape[0] < 2 or df_raw.shape[1] < 2:
            raise ValueError("Data must have at least two rows and two columns.")

        self.x_full = df_raw.iloc[0, 1:].values.astype(float)
        self.y_raw_full = df_raw.iloc[1:, 1:].values.astype(float).T
        self.sample_names = [str(name) for name in df_raw.iloc[1:, 0].tolist()]

        self.x = self.x_full.copy()
        self.y_raw = self.y_raw_full.copy()

        if self.x.size == 0 or self.y_raw.size == 0:
            raise ValueError("Loaded data is empty after processing.")

        num_spectra = self.y_raw.shape[1]
        logging.info(f"Loaded {num_spectra} spectra from {filepath}")
        return num_spectra

    def filter_data_by_range(self, lower_bound: float, upper_bound: float) -> bool:
        if self.x_full is None or self.y_raw_full is None:
            return False
        if lower_bound >= upper_bound:
            return False

        mask = (self.x_full >= lower_bound) & (self.x_full <= upper_bound)
        if not np.any(mask):
            self.x = self.x_full.copy()
            self.y_raw = self.y_raw_full.copy()
            logging.warning("No data points in the specified range. Resetting to full range.")
            return False

        self.x = self.x_full[mask]
        self.y_raw = self.y_raw_full[mask, :]
        logging.info(f"Data filtered to range [{lower_bound}, {upper_bound}].")
        return True

    # ── Cosmic ray removal ─────────────────────────────────────────

    def remove_cosmic_rays(self, y_data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        y = y_data.copy()
        n = len(y)
        if n < 3:
            return y

        diff = np.diff(y)
        median_diff = np.median(diff)
        mad_diff = np.median(np.abs(diff - median_diff))
        if mad_diff == 0:
            return y

        mod_z_score = MAD_SCALE_FACTOR * (diff - median_diff) / mad_diff
        spikes = np.abs(mod_z_score) > threshold

        for i in np.where(spikes)[0]:
            spike_idx = i + 1
            left = max(0, spike_idx - 1)
            right = min(n - 1, spike_idx + 1)
            if left == spike_idx:
                y[spike_idx] = y[right]
            elif right == spike_idx:
                y[spike_idx] = y[left]
            else:
                y[spike_idx] = (y[left] + y[right]) / 2.0

        return y

    # ── Smoothing ──────────────────────────────────────────────────

    def apply_sg_filter(
        self, y_data: np.ndarray, poly_order: int, frame_window: int
    ) -> np.ndarray:
        if frame_window % 2 == 0:
            frame_window += 1
        frame_window = max(3, min(frame_window, len(y_data) - 1))
        if frame_window % 2 == 0:
            frame_window -= 1
        poly_order = min(poly_order, frame_window - 1)
        return savgol_filter(y_data, window_length=frame_window, polyorder=poly_order)

    # ── Baseline: unified dispatch ─────────────────────────────────

    def compute_baseline(
        self,
        y_mid: np.ndarray,
        algorithm: str,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Dispatch to the appropriate baseline algorithm and return the baseline."""
        if algorithm in POLYNOMIAL_ALGOS:
            baseline, _ = self.baseline_polynomial(
                self.x,
                y_mid,
                ord=int(params.get('order', 2)),
                s=float(params.get('threshold', 0.01)),
                fct=algorithm,
            )
            return baseline

        if algorithm == 'mor':
            baseline, _ = self.baseline_morphological(
                y_mid, half_window=int(params.get('half_window', 25))
            )
            return baseline

        if algorithm == 'airpls':
            baseline, _ = airpls(
                y_mid,
                lam=float(params.get('lam', 1e6)),
                diff_order=int(params.get('diff_order', 2)),
            )
            return baseline

        if algorithm == 'arpls':
            baseline, _ = arpls(
                y_mid,
                lam=float(params.get('lam', 1e5)),
                diff_order=int(params.get('diff_order', 2)),
            )
            return baseline

        if algorithm == 'asls':
            baseline, _ = asls(
                y_mid,
                lam=float(params.get('lam', 1e6)),
                p=float(params.get('p', 0.01)),
                diff_order=int(params.get('diff_order', 2)),
            )
            return baseline

        if algorithm == 'snip':
            baseline, _ = snip(
                y_mid,
                max_half_window=int(params.get('max_half_window', 40)),
                filter_order=int(params.get('filter_order', 2)),
                decreasing=bool(params.get('decreasing', False)),
            )
            return baseline

        raise ValueError(f"Unknown baseline algorithm: {algorithm}")

    def baseline_polynomial(
        self,
        n: np.ndarray,
        y: np.ndarray,
        ord: int,
        s: float,
        fct: str,
    ) -> Tuple[np.ndarray, int]:
        n = np.asarray(n, dtype=float)
        y = np.asarray(y, dtype=float)

        sort_indices = np.argsort(n)
        n_sorted = n[sort_indices]
        y_sorted = y[sort_indices]

        N = len(n)
        maxy = np.max(y_sorted)
        miny = np.min(y_sorted)
        dely = (maxy - miny) / 2.0

        if dely == 0:
            return np.full_like(y, maxy), 0

        y_scaled = (y_sorted - maxy) / dely + 1

        if N > 1:
            n_range = n_sorted[-1] - n_sorted[0]
            if n_range > 0:
                n_scaled = 2 * (n_sorted - n_sorted[-1]) / n_range + 1
            else:
                n_scaled = np.zeros_like(n_sorted)
        else:
            n_scaled = np.zeros_like(n_sorted)

        T = np.vander(n_scaled, ord + 1, increasing=True)
        a, _, _, _ = np.linalg.lstsq(T, y_scaled, rcond=None)
        z = T @ a

        alpha = BASELINE_ALPHA
        zp = np.ones(N)
        it = 0

        while it < BASELINE_MAX_ITERATIONS:
            norm_zp = np.sum(zp ** 2)
            if norm_zp > 0 and np.sum((z - zp) ** 2) / norm_zp <= BASELINE_CONVERGENCE_TOL:
                break
            it += 1
            zp = z.copy()
            res = y_scaled - z

            if fct == 'sh':
                d = np.where(
                    np.abs(res) < s,
                    res * (2 * alpha - 1),
                    np.where(res <= -s, -alpha * 2 * s - res, alpha * 2 * s - res),
                )
            elif fct == 'ah':
                d = np.where(res < s, res * (2 * alpha - 1), alpha * 2 * s - res)
            elif fct == 'stq':
                d = np.where(np.abs(res) < s, res * (2 * alpha - 1), -res)
            elif fct == 'atq':
                d = np.where(res < s, res * (2 * alpha - 1), -res)
            else:
                raise ValueError(f"Unknown baseline function type: {fct}")

            a, _, _, _ = np.linalg.lstsq(T, y_scaled + d, rcond=None)
            z = T @ a

        z_rescaled = (z - 1) * dely + maxy
        inv_sort_indices = np.argsort(sort_indices)
        return z_rescaled[inv_sort_indices], it

    def baseline_morphological(
        self, y_data: np.ndarray, half_window: int
    ) -> Tuple[np.ndarray, dict]:
        baseline, params = mor(y_data, half_window=half_window)
        return baseline, params

    # ── Full single-spectrum pipeline ──────────────────────────────

    def process_single_spectrum(
        self, y_raw: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """Process a single spectrum through the full pipeline and return y_final."""
        y_current = y_raw.copy()

        if params.get('preprocessing', {}).get('apply_cosmic_ray', False):
            threshold = params.get('preprocessing', {}).get('cosmic_ray_threshold', 5.0)
            y_current = self.remove_cosmic_rays(y_current, threshold=threshold)

        y_mid = self.apply_sg_filter(
            y_current,
            int(params['smoothing']['sg_poly_order']),
            int(params['smoothing']['sg_frame_window']),
        )

        baseline_algo = params['baseline']['algorithm']
        baseline_params = params['baseline']['params']
        baseline = self.compute_baseline(y_mid, baseline_algo, baseline_params)

        y_final = y_mid - baseline

        # Optional derivative transformation (baseline alternative / peak sharpening)
        deriv_cfg = params.get('derivative', {})
        if isinstance(deriv_cfg, dict) and deriv_cfg.get('enabled', False):
            y_final = compute_derivative(
                y_final,
                order=int(deriv_cfg.get('order', 1)),
                window=int(deriv_cfg.get('window', 15)),
                polyorder=int(deriv_cfg.get('polyorder', 3)),
            )

        norm_method = params.get('normalization', 'none')
        if norm_method and norm_method != 'none':
            y_final = normalize_spectrum(y_final, method=norm_method)

        return y_final

    # ── Batch processing (serial or parallel) ──────────────────────

    def run_batch_processing(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.y_raw is None or self.x is None:
            raise RuntimeError("Data not loaded.")

        total_spectra = self.y_raw.shape[1]
        n_points = len(self.x)

        if parallel and total_spectra > 1:
            results, statuses = self._run_batch_parallel(
                total_spectra, n_points, params, n_jobs, progress_callback
            )
        else:
            results, statuses = self._run_batch_serial(
                total_spectra, n_points, params, progress_callback
            )

        valid_indices = [i for i, ok in enumerate(statuses) if ok]
        valid_names = [self.sample_names[i] for i in valid_indices]
        valid_data = results[:, valid_indices]

        processed_df = pd.DataFrame(valid_data, columns=valid_names)
        processed_df.insert(0, 'Raman shift (cm-1)', self.x)

        # Per-spectrum QC
        qc_cfg = params.get('qc', {}) if isinstance(params, dict) else {}
        noise_region = tuple(qc_cfg.get('snr_noise_region', (1800.0, 2000.0)))
        sat_thr = float(qc_cfg.get('saturation_relative', 0.98))

        qc_rows = []
        for idx in valid_indices:
            qc = compute_spectrum_qc(
                self.y_raw[:, idx],
                results[:, idx],
                self.x,
                noise_region=noise_region,
                sat_threshold=sat_thr,
            )
            qc["sample"] = self.sample_names[idx]
            qc_rows.append(qc)
        qc_df = pd.DataFrame(qc_rows, columns=['sample', 'snr_db', 'saturated', 'has_nan', 'flag'])

        n_flagged = int(np.sum(qc_df['flag'] != 'OK')) if not qc_df.empty else 0
        summary = {
            "processed": len(valid_indices),
            "failed": total_spectra - len(valid_indices),
            "qc_flagged": n_flagged,
            "qc_df": qc_df,
        }
        return processed_df, summary

    def _run_batch_serial(
        self,
        total: int,
        n_points: int,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> Tuple[np.ndarray, list[bool]]:
        results = np.full((n_points, total), np.nan, dtype=float)
        statuses = [False] * total

        for i in range(total):
            try:
                y_final = self.process_single_spectrum(self.y_raw[:, i], params)
                results[:, i] = y_final
                statuses[i] = True
            except Exception as e:
                logging.warning(
                    f"Failed to process spectrum {i + 1} ({self.sample_names[i]}): {e}"
                )
            if progress_callback:
                progress_callback(i + 1, total)

        return results, statuses

    def _run_batch_parallel(
        self,
        total: int,
        n_points: int,
        params: Dict[str, Any],
        n_jobs: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> Tuple[np.ndarray, list[bool]]:
        # Snapshot inputs so workers share read-only arrays
        y_all = self.y_raw
        x = self.x
        sample_names = self.sample_names

        # Using threading backend — pybaselines/numpy release the GIL
        # and avoid pickling the DataProcessor instance.
        outputs = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_process_worker)(i, y_all[:, i], x, params, sample_names[i])
            for i in range(total)
        )

        results = np.full((n_points, total), np.nan, dtype=float)
        statuses = [False] * total
        for idx, (ok, y_final) in enumerate(outputs):
            if ok:
                results[:, idx] = y_final
                statuses[idx] = True
            if progress_callback:
                progress_callback(idx + 1, total)

        return results, statuses

    # ── PCA ────────────────────────────────────────────────────────

    def perform_pca(
        self,
        processed_df: pd.DataFrame,
        n_components: int = 10,
        scaling: str = 'auto',
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Perform PCA with diagnostics (Hotelling T², Q-residuals, scaling options).

        scaling: 'auto' | 'mean' | 'pareto' | 'none'
        confidence: for T²/Q control limits (0.95 / 0.99 typical)
        """
        if processed_df.shape[1] < 2:
            raise ValueError("PCA requires at least 2 samples.")

        raman_shifts = processed_df['Raman shift (cm-1)'].values
        intensity_data = processed_df.iloc[:, 1:].values
        data_transposed = intensity_data.T
        n_samples = data_transposed.shape[0]

        data_scaled, scale_info = apply_pca_scaling(data_transposed, method=scaling)

        n_components = min(n_components, n_samples, data_scaled.shape[1])

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data_scaled)

        eigenvalues = pca.explained_variance_ * (n_samples - 1)
        kaiser_components = int(np.sum(eigenvalues > 1))

        # Hotelling T² = Σ (t_i² / λ_i)
        explained_variance = pca.explained_variance_
        safe_var = np.where(explained_variance > 0, explained_variance, 1.0)
        t2 = np.sum((scores ** 2) / safe_var, axis=1)

        # T² confidence limit via F-distribution
        # T²_lim = k(n-1)/(n-k) * F(alpha; k, n-k)
        k = n_components
        n = n_samples
        if n - k > 0:
            f_crit = f_dist.ppf(confidence, k, n - k)
            t2_limit = k * (n - 1) / (n - k) * f_crit
        else:
            t2_limit = float('nan')

        # Q-residuals (SPE) = Σ (x - x_hat)²  per sample
        reconstruction = scores @ pca.components_
        residuals = data_scaled - reconstruction
        q_residuals = np.sum(residuals ** 2, axis=1)

        # Jackson-Mudholkar Q-limit
        # Requires eigenvalues of residual subspace — use full PCA to estimate
        full_pca = PCA(n_components=min(n_samples, data_scaled.shape[1]))
        full_pca.fit(data_scaled)
        all_eig = full_pca.explained_variance_
        if k < len(all_eig):
            residual_eig = all_eig[k:]
            theta1 = float(np.sum(residual_eig))
            theta2 = float(np.sum(residual_eig ** 2))
            theta3 = float(np.sum(residual_eig ** 3))
            if theta1 > 0 and theta2 > 0:
                h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 ** 2)
                from scipy.stats import norm
                c_alpha = norm.ppf(confidence)
                try:
                    q_limit = theta1 * (
                        c_alpha * np.sqrt(2.0 * theta2 * h0 ** 2) / theta1
                        + 1.0
                        + theta2 * h0 * (h0 - 1.0) / (theta1 ** 2)
                    ) ** (1.0 / h0)
                    q_limit = float(q_limit)
                except (ValueError, ZeroDivisionError):
                    q_limit = float('nan')
            else:
                q_limit = float('nan')
        else:
            q_limit = float('nan')

        results = {
            "scores": scores,
            "loadings": pca.components_.T,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "eigenvalues": eigenvalues,
            "kaiser_components": kaiser_components,
            "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "n_samples": n_samples,
            "n_variables": data_scaled.shape[1],
            "n_components_selected": n_components,
            "sample_names": processed_df.columns[1:].tolist(),
            "raman_shifts": raman_shifts,
            "scaling_method": scaling,
            "hotelling_t2": t2,
            "t2_limit": t2_limit,
            "q_residuals": q_residuals,
            "q_limit": q_limit,
            "confidence": confidence,
        }
        logging.info(
            f"PCA ({scaling} scaling) performed: {n_components} components, "
            f"T2_lim={t2_limit:.3g}, Q_lim={q_limit:.3g}"
        )
        return results

    # ── NMF ────────────────────────────────────────────────────────

    def perform_nmf(
        self,
        processed_df: pd.DataFrame,
        n_components: int,
        init: str = 'nndsvda',
        random_state: int = 0,
    ) -> Dict[str, Any]:
        n_samples = processed_df.shape[1] - 1  # exclude Raman shift column
        if n_components < 2:
            raise ValueError("Number of components must be 2 or greater.")
        if n_components > n_samples:
            raise ValueError(
                f"Number of components ({n_components}) cannot exceed "
                f"number of samples ({n_samples})."
            )

        raman_shifts = processed_df['Raman shift (cm-1)'].values
        intensity_data = processed_df.iloc[:, 1:].values
        data_non_negative = np.maximum(0, intensity_data)
        data_transposed = data_non_negative.T

        model = NMF(
            n_components=n_components,
            init=init,
            random_state=random_state,
            max_iter=2000,
            tol=1e-5,
        )
        weights = model.fit_transform(data_transposed)
        components = model.components_

        results = {
            "weights": weights,
            "components": components.T,
            "reconstruction_error": float(model.reconstruction_err_),
            "n_iter": int(model.n_iter_),
            "n_components": n_components,
            "init": init,
            "random_state": random_state,
            "sample_names": processed_df.columns[1:].tolist(),
            "raman_shifts": raman_shifts,
        }
        logging.info(
            f"NMF performed with {n_components} components (init={init}, iters={model.n_iter_})."
        )
        return results


# ── Text / ASCII file loader ──────────────────────────────────────


def _load_text_file(filepath: str) -> pd.DataFrame:
    """Load a plain-text Raman data file and return a DataFrame in the standard format.

    Handles two layout types automatically:

    Type A – Multi-spectrum table (same format as CSV/Excel):
        Row 0   : [empty/label] | shift_1 | shift_2 | ...
        Row 1.. : sample_name  | int_1   | int_2   | ...

    Type B – Single-spectrum two-column (wavenumber | intensity), one pair per row:
        400.0   123.4
        401.0   125.1
        ...
        → Converted to Type-A shape with sample name = filename stem.

    Features:
    - Auto-detects separator: tab → comma → semicolon → whitespace
    - Skips comment lines starting with '#' or '%'
    - Skips blank lines
    - Tolerates UTF-8 and Latin-1 encodings
    """
    # Read raw lines, stripping comments and blanks
    raw_lines: list[str] = []
    for encoding in ('utf-8', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(('#', '%')):
                        raw_lines.append(stripped)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode file with UTF-8 or Latin-1 encoding.")

    if not raw_lines:
        raise ValueError("File contains no data after stripping comment/blank lines.")

    # Detect separator from first non-empty line
    first = raw_lines[0]
    if '\t' in first:
        sep = '\t'
    elif ',' in first:
        sep = ','
    elif ';' in first:
        sep = ';'
    else:
        sep = r'\s+'   # generic whitespace

    # Parse into a list-of-lists (all numeric where possible)
    import re
    rows: list[list[str]] = []
    for line in raw_lines:
        if sep == r'\s+':
            parts = re.split(r'\s+', line)
        else:
            parts = line.split(sep)
        rows.append([p.strip() for p in parts])

    # Remove completely empty trailing tokens from each row
    rows = [[c for c in row if c != ''] for row in rows if any(c != '' for c in row)]

    # Check if all tokens in the first row are numeric → Type B (two-column)
    def _is_numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    first_row_numeric = all(_is_numeric(c) for c in rows[0])
    all_rows_two_cols = all(len(r) == 2 for r in rows)

    if first_row_numeric and all_rows_two_cols:
        # Type B: wavenumber | intensity single-spectrum file
        sample_name = os.path.splitext(os.path.basename(filepath))[0]
        shifts = [float(r[0]) for r in rows]
        intensities = [float(r[1]) for r in rows]
        # Build Type-A DataFrame:
        #   row0: '' | shift_0 | shift_1 | ...
        #   row1: sample_name | int_0 | int_1 | ...
        header_row = [''] + [str(s) for s in shifts]
        data_row = [sample_name] + [str(v) for v in intensities]
        df = pd.DataFrame([header_row, data_row])
        logging.info(f"Detected Type-B (2-column single spectrum): {len(shifts)} points.")
        return df

    # Type A: multi-spectrum table — parse as-is
    df = pd.DataFrame(rows)
    logging.info(f"Detected Type-A (multi-spectrum table): {df.shape[0]} rows x {df.shape[1]} cols.")
    return df


# ── Worker helper for parallel batch (module-level for pickling) ──


def _process_worker(
    idx: int,
    y_raw: np.ndarray,
    x: np.ndarray,
    params: Dict[str, Any],
    sample_name: str,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Standalone worker — avoids pickling the DataProcessor itself."""
    try:
        # Lightweight local processor binding x for polynomial baselines
        dp = DataProcessor()
        dp.x = x
        y_final = dp.process_single_spectrum(y_raw, params)
        return True, y_final
    except Exception as e:
        logging.warning(f"Parallel batch failed on spectrum {idx + 1} ({sample_name}): {e}")
        return False, None
