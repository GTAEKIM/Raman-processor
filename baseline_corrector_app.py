import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any, Callable
import numpy as np
from processing_logic import DataProcessor, POLYNOMIAL_ALGOS, WHITTAKER_ALGOS


# Default parameters per algorithm family
_FAMILY_DEFAULTS = {
    'polynomial': {'order': 2, 'threshold': 0.01},
    'morphological': {'half_window': 25},
    'whittaker': {'lam': 1e6, 'diff_order': 2, 'p': 0.01},
    'snip': {'max_half_window': 40, 'filter_order': 2},
}


def _algo_family(short_code: str) -> str:
    if short_code in POLYNOMIAL_ALGOS:
        return 'polynomial'
    if short_code == 'mor':
        return 'morphological'
    if short_code in WHITTAKER_ALGOS:
        return 'whittaker'
    if short_code == 'snip':
        return 'snip'
    return 'polynomial'


class BaselineCorrectorWindow(tk.Toplevel):
    """Interactive baseline adjustment sub-window with real-time preview.

    Supports polynomial, morphological, Whittaker (airPLS/arPLS/ASLS), and SNIP.
    """

    def __init__(
        self,
        parent,
        processor: DataProcessor,
        y_mid: np.ndarray,
        current_params: Dict[str, Any],
        algorithms: Dict[str, str],
        callback: Callable[[np.ndarray, Dict[str, Any]], None],
    ):
        super().__init__(parent)
        self.title("Baseline Corrector")
        self.geometry("900x550")

        self.processor = processor
        self.x_data = processor.x
        self.y_data = y_mid
        self.callback = callback
        self.algorithms_map = algorithms
        self.reverse_algo_map = {v: k for k, v in algorithms.items()}
        self.last_baseline: np.ndarray | None = None

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        init_algo = current_params.get("algorithm", "airpls")
        self.algorithm_code = tk.StringVar(value=init_algo)
        self.algorithm_fullname = tk.StringVar(
            value=self.algorithms_map.get(init_algo, init_algo)
        )

        # Parameter variables (merged from defaults + provided)
        merged = self._merge_params(current_params.get("params", {}))
        self.poly_order = tk.IntVar(value=int(merged.get('order', 2)))
        self.threshold = tk.DoubleVar(value=float(merged.get('threshold', 0.01)))
        self.half_window = tk.IntVar(value=int(merged.get('half_window', 25)))
        self.lam = tk.DoubleVar(value=float(merged.get('lam', 1e6)))
        self.diff_order = tk.IntVar(value=int(merged.get('diff_order', 2)))
        self.p_asls = tk.DoubleVar(value=float(merged.get('p', 0.01)))
        self.max_half_window = tk.IntVar(value=int(merged.get('max_half_window', 40)))
        self.filter_order = tk.IntVar(value=int(merged.get('filter_order', 2)))

        self._create_widgets()
        self._on_algo_select()
        self.grab_set()

    def _merge_params(self, provided: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for defaults in _FAMILY_DEFAULTS.values():
            merged.update(defaults)
        merged.update(provided)
        return merged

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, sticky="nsew")

        ttk.Label(control_frame, text="Algorithm:").pack(fill="x", pady=(0, 2))
        self.algo_combo = ttk.Combobox(
            control_frame,
            textvariable=self.algorithm_fullname,
            values=list(self.algorithms_map.values()),
            state="readonly",
        )
        self.algo_combo.pack(fill="x", pady=(0, 10))
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_select)

        self.param_container = ttk.Frame(control_frame)
        self.param_container.pack(fill="both", expand=True)
        self._build_param_frames()

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side="bottom", fill="x")
        ttk.Button(button_frame, text="Apply", command=self._on_apply).pack(
            side="left", expand=True, padx=2
        )
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(
            side="left", expand=True, padx=2
        )

    # ── Parameter frames per algorithm family ───────────────────────

    def _build_param_frames(self):
        self.frames: Dict[str, ttk.Frame] = {}

        # Polynomial
        f = ttk.Frame(self.param_container)
        self._labeled_slider_entry(
            f, "Polynomial Order:", self.poly_order, 0, 20, is_int=True
        )
        self._labeled_entry(f, "Threshold:", self.threshold)
        self.frames['polynomial'] = f

        # Morphological
        f = ttk.Frame(self.param_container)
        self._labeled_slider_entry(
            f, "Half Window Size:", self.half_window, 1, 100, is_int=True
        )
        self.frames['morphological'] = f

        # Whittaker (airPLS / arPLS / ASLS)
        f = ttk.Frame(self.param_container)
        self._labeled_entry(f, "Lambda (\u03bb):", self.lam)
        ttk.Label(
            f,
            text="Typical range: 1e3 - 1e9",
            foreground="gray",
        ).pack(anchor="w", pady=(0, 8))
        self._labeled_slider_entry(
            f, "Differential Order:", self.diff_order, 1, 3, is_int=True
        )
        # ASLS-specific: asymmetry weight p
        self.asls_only_frame = ttk.Frame(f)
        self._labeled_entry(self.asls_only_frame, "Asymmetry p (ASLS only):", self.p_asls)
        ttk.Label(
            self.asls_only_frame,
            text="Typical: 0.001 - 0.1",
            foreground="gray",
        ).pack(anchor="w", pady=(0, 8))
        self.asls_only_frame.pack(fill="x")
        self.frames['whittaker'] = f

        # SNIP
        f = ttk.Frame(self.param_container)
        self._labeled_slider_entry(
            f, "Max Half Window:", self.max_half_window, 5, 200, is_int=True
        )
        self._labeled_slider_entry(
            f, "Filter Order (2/4/6/8):", self.filter_order, 2, 8, is_int=True
        )
        self.frames['snip'] = f

    def _labeled_entry(self, parent, label_text: str, variable):
        ttk.Label(parent, text=label_text).pack(fill="x", pady=(0, 2))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.pack(fill="x", pady=(0, 8))
        entry.bind("<Return>", self._update_plot)
        entry.bind("<FocusOut>", self._update_plot)

    def _labeled_slider_entry(
        self, parent, label_text: str, variable, from_, to, is_int: bool
    ):
        ttk.Label(parent, text=label_text).pack(fill="x", pady=(0, 2))
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 8))
        entry = ttk.Entry(row, textvariable=variable, width=6)
        entry.pack(side="right", padx=(5, 0))
        entry.bind("<Return>", self._update_plot)
        entry.bind("<FocusOut>", self._update_plot)
        ttk.Scale(
            row,
            from_=from_,
            to=to,
            orient="horizontal",
            variable=variable,
            command=self._update_plot,
        ).pack(side="left", fill="x", expand=True)

    # ── Event handlers ─────────────────────────────────────────────

    def _on_algo_select(self, event=None):
        selected_fullname = self.algorithm_fullname.get()
        if selected_fullname not in self.reverse_algo_map:
            return
        short_code = self.reverse_algo_map[selected_fullname]
        self.algorithm_code.set(short_code)
        family = _algo_family(short_code)

        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[family].pack(fill="both", expand=True)

        # Toggle ASLS-only row in Whittaker family
        if family == 'whittaker':
            if short_code == 'asls':
                self.asls_only_frame.pack(fill="x")
            else:
                self.asls_only_frame.pack_forget()

        self._update_plot()

    def _collect_current_params(self) -> Dict[str, Any]:
        short = self.algorithm_code.get()
        family = _algo_family(short)
        if family == 'polynomial':
            return {
                'order': self.poly_order.get(),
                'threshold': self.threshold.get(),
            }
        if family == 'morphological':
            return {'half_window': self.half_window.get()}
        if family == 'whittaker':
            params = {
                'lam': self.lam.get(),
                'diff_order': self.diff_order.get(),
            }
            if short == 'asls':
                params['p'] = self.p_asls.get()
            return params
        if family == 'snip':
            return {
                'max_half_window': self.max_half_window.get(),
                'filter_order': self.filter_order.get(),
            }
        return {}

    def _update_plot(self, event=None):
        try:
            algo_code = self.algorithm_code.get()
            params = self._collect_current_params()
            baseline = self.processor.compute_baseline(self.y_data, algo_code, params)
        except (tk.TclError, ValueError, RuntimeError):
            return
        except Exception:
            return

        self.ax.clear()
        self.ax.plot(self.x_data, self.y_data, 'b-', label="SG Filtered")
        self.ax.plot(self.x_data, baseline, 'm--', label="Estimated Baseline")
        self.ax.plot(
            self.x_data,
            self.y_data - baseline,
            'r-',
            alpha=0.5,
            label="Corrected",
        )
        self.ax.set_title("Real-time Baseline Adjustment")
        self.ax.set_xlabel("Raman shift (cm\u207b\u00b9)")
        self.ax.set_ylabel("Intensity (AU)")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()
        self.last_baseline = baseline

    def _on_apply(self):
        if self.last_baseline is not None:
            final_params = {
                "algorithm": self.algorithm_code.get(),
                "params": self._collect_current_params(),
            }
            self.callback(self.last_baseline, final_params)
        self._close()

    def _on_cancel(self):
        self._close()

    def _close(self):
        plt.close(self.fig)
        self.destroy()
