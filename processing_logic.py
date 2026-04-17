import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Any, Callable, Optional
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
        if ext == '.csv':
            df_raw = pd.read_csv(filepath, header=None)
        elif ext in ('.xlsx', '.xls'):
            df_raw = pd.read_excel(filepath, header=None)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

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

        summary = {
            "processed": len(valid_indices),
            "failed": total_spectra - len(valid_indices),
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
        self, processed_df: pd.DataFrame, n_components: int = 10
    ) -> Dict[str, Any]:
        if processed_df.shape[1] < 2:
            raise ValueError("PCA requires at least 2 samples.")

        raman_shifts = processed_df['Raman shift (cm-1)'].values
        intensity_data = processed_df.iloc[:, 1:].values
        data_transposed = intensity_data.T
        n_samples = data_transposed.shape[0]

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_transposed)

        n_components = min(n_components, n_samples, data_scaled.shape[1])

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(data_scaled)

        eigenvalues = pca.explained_variance_ * (n_samples - 1)
        kaiser_components = int(np.sum(eigenvalues > 1))

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
        }
        logging.info(f"PCA performed with {n_components} components.")
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
