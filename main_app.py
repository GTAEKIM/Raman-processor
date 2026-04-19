import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import json
import logging
import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import os
import sys

from processing_logic import DataProcessor, ProcessingStage, normalize_spectrum, compute_derivative
from baseline_corrector_app import BaselineCorrectorWindow
from pca_result_window import PCAResultWindow
from nmf_result_window import NMFResultWindow
from peak_analysis_window import PeakAnalysisWindow
from clustering_window import ClusteringWindow
from mcr_als_window import MCRALSWindow
from calibration_window import CalibrationWindow

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class CustomToolbar(NavigationToolbar2Tk):
    def set_message(self, s):
        pass


class RamanProcessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Raman Spectroscopy Processor v2.3")
        self.root.geometry("1280x820")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.processor = DataProcessor()
        self.config = self._load_config()

        # Spectrum state
        self.y_raw: Optional[np.ndarray] = None
        self.y_processed: Optional[np.ndarray] = None
        self.y_mid: Optional[np.ndarray] = None
        self.y_final: Optional[np.ndarray] = None
        self.baseline: Optional[np.ndarray] = None
        self.current_selection_idx: Optional[int] = None
        self.batch_result_df: Optional[pd.DataFrame] = None
        self._batch_lock = threading.Lock()
        self.last_processing_stage = ProcessingStage.RAW

        # UI variables from config defaults
        defaults = self.config.get("defaults", {})
        self.lower_bound = tk.DoubleVar(value=defaults.get("lower_bound", 400.0))
        self.upper_bound = tk.DoubleVar(value=defaults.get("upper_bound", 3300.0))
        self.sg_poly_order = tk.IntVar(value=defaults.get("sg_poly_order", 1))
        self.sg_frame_window = tk.IntVar(value=defaults.get("sg_frame_window", 15))
        self.baseline_params = defaults.get(
            "baseline",
            {"algorithm": "airpls", "params": {"lam": 1e6, "diff_order": 2}},
        )

        # Normalization
        self.normalization_methods = self.config.get(
            "normalization_methods",
            {"none": "None", "snv": "SNV", "vector": "Vector", "area": "Area",
             "minmax": "Min-Max", "maxpeak": "Max Peak"},
        )
        self._norm_code_to_name = self.normalization_methods
        self._norm_name_to_code = {v: k for k, v in self.normalization_methods.items()}
        init_norm = defaults.get("normalization", "none")
        self.normalization_code = tk.StringVar(value=init_norm)
        self.normalization_name = tk.StringVar(
            value=self._norm_code_to_name.get(init_norm, "None")
        )

        # Batch
        batch_defaults = defaults.get("batch", {"parallel": True, "n_jobs": -1})
        self.batch_parallel = tk.BooleanVar(value=batch_defaults.get("parallel", True))
        self.batch_n_jobs = tk.IntVar(value=batch_defaults.get("n_jobs", -1))

        # NMF
        nmf_defaults = defaults.get(
            "nmf", {"init": "nndsvda", "n_components": 3, "random_state": 0}
        )
        self.nmf_init_methods = self.config.get(
            "nmf_init_methods",
            {"nndsvda": "NNDSVDA (recommended)", "nndsvd": "NNDSVD",
             "nndsvdar": "NNDSVDAR", "random": "Random"},
        )
        self._nmf_code_to_name = self.nmf_init_methods
        self._nmf_name_to_code = {v: k for k, v in self.nmf_init_methods.items()}
        self.nmf_init_code = tk.StringVar(value=nmf_defaults.get("init", "nndsvda"))
        self.nmf_init_name = tk.StringVar(
            value=self._nmf_code_to_name.get(nmf_defaults.get("init", "nndsvda"), "")
        )
        self.nmf_random_state = tk.IntVar(value=nmf_defaults.get("random_state", 0))
        self.nmf_n_components = tk.IntVar(value=nmf_defaults.get("n_components", 3))

        # Derivative transform
        deriv_defaults = defaults.get(
            "derivative", {"enabled": False, "order": 1, "window": 15, "polyorder": 3}
        ) if False else self.config.get(
            "derivative", {"enabled": False, "order": 1, "window": 15, "polyorder": 3}
        )
        self.deriv_enabled = tk.BooleanVar(value=deriv_defaults.get("enabled", False))
        self.deriv_order = tk.IntVar(value=deriv_defaults.get("order", 1))
        self.deriv_window = tk.IntVar(value=deriv_defaults.get("window", 15))
        self.deriv_polyorder = tk.IntVar(value=deriv_defaults.get("polyorder", 3))

        # PCA scaling
        self.pca_scaling_methods = self.config.get(
            "pca_scaling_methods",
            {"auto": "Auto-scale (unit variance)", "mean": "Mean-center only",
             "pareto": "Pareto (sqrt std)", "none": "No scaling"},
        )
        self._pca_code_to_name = self.pca_scaling_methods
        self._pca_name_to_code = {v: k for k, v in self.pca_scaling_methods.items()}
        self.pca_scaling_code = tk.StringVar(value="auto")
        self.pca_scaling_name = tk.StringVar(
            value=self._pca_code_to_name.get("auto", "Auto-scale (unit variance)")
        )
        self.pca_confidence = tk.DoubleVar(value=0.95)

        # QC config passthrough
        self.qc_cfg = self.config.get("qc", {
            "snr_noise_region": [1800, 2000], "saturation_relative": 0.98,
        })

        # Peak detection config
        self.peak_cfg = self.config.get("peak_detection", {
            "prominence_percent": 2.0, "min_distance": 5, "default_profile": "gaussian",
        })

        # Display / preprocess
        self.show_raw = tk.BooleanVar(value=True)
        self.show_smooth = tk.BooleanVar(value=True)
        self.show_baseline = tk.BooleanVar(value=True)
        self.show_final = tk.BooleanVar(value=True)
        self.apply_cosmic_ray = tk.BooleanVar(value=False)

        self.icon_images: Dict[str, tk.PhotoImage] = {}
        self._build_ui()
        self._update_status("Ready. Please import a data file.")

    # ── Config ──────────────────────────────────────────────────────────

    def _load_config(self) -> Dict[str, Any]:
        config_path = resource_path("config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("config.json not found, using defaults.")
            return {"defaults": {}, "validation": {}, "baseline_algorithms": {}}

    # ── UI Construction ─────────────────────────────────────────────────

    def _build_ui(self):
        self._build_toolbar()

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = self._build_control_sidebar(main_pane)
        main_pane.add(left_panel, weight=1)

        right_panel = self._build_plot_panel(main_pane)
        main_pane.add(right_panel, weight=3)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(
            status_frame, mode='determinate', length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        self.progress_bar.pack_forget()

        self.status_bar = ttk.Label(
            status_frame, text="Ready", anchor=tk.W, relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _load_icon(self, filename: str) -> Optional[tk.PhotoImage]:
        try:
            path = resource_path(os.path.join('icons', filename))
            img = tk.PhotoImage(file=path)
            self.icon_images[filename] = img
            return img
        except tk.TclError:
            logging.warning(f"Icon not found: {filename}")
            return None

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_configs = [
            ("folder-open-outline.png", "Import Data", self._import_data),
            ("parameters-import.png", "Import Parameters", self._import_params),
        ]
        for icon_file, text, cmd in btn_configs:
            icon = self._load_icon(icon_file)
            btn = ttk.Button(toolbar, image=icon, text=text, compound=tk.LEFT, command=cmd)
            btn.pack(side=tk.LEFT, padx=2, pady=2)

    def _build_control_sidebar(self, parent) -> ttk.Frame:
        # Scrollable container so the sidebar doesn't overflow on small screens
        outer = ttk.Frame(parent, width=310)
        outer.pack_propagate(False)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, width=290)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(canvas)
        panel_id = canvas.create_window((0, 0), window=panel, anchor="nw")

        def _on_panel_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_config(event):
            canvas.itemconfig(panel_id, width=event.width)

        panel.bind("<Configure>", _on_panel_config)
        canvas.bind("<Configure>", _on_canvas_config)

        # Mouse wheel scrolling
        def _on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)

        self._build_sidebar_sections(panel)
        return outer

    def _build_sidebar_sections(self, panel: ttk.Frame):
        # 1. Data Load & Select
        file_frame = ttk.LabelFrame(panel, text="1. Data Load & Select")
        file_frame.pack(fill="x", pady=5, padx=2)
        self.info_label = ttk.Label(file_frame, text="No file loaded.", wraplength=250)
        self.info_label.pack(padx=5, pady=5)
        self.listbox = tk.Listbox(file_frame, height=8)
        self.listbox.pack(fill="x", expand=True, padx=5, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self._on_listbox_select)

        # 2. Pre-process
        preproc_frame = ttk.LabelFrame(panel, text="2. Pre-process")
        preproc_frame.pack(fill="x", pady=5, padx=2)
        range_frame = ttk.Frame(preproc_frame)
        range_frame.pack(fill="x", pady=5)
        ttk.Label(range_frame, text="Range:").pack(side="left", padx=5)
        ttk.Entry(range_frame, textvariable=self.lower_bound, width=7).pack(side="left")
        ttk.Label(range_frame, text="-").pack(side="left", padx=2)
        ttk.Entry(range_frame, textvariable=self.upper_bound, width=7).pack(side="left")
        ttk.Button(
            range_frame, text="Apply", command=self._apply_range_filter, width=6
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            preproc_frame,
            text="Remove Cosmic Rays",
            variable=self.apply_cosmic_ray,
            command=self._clear_and_redraw_raw,
        ).pack(anchor="w", padx=5)
        ttk.Button(
            preproc_frame, text="Wavelength Calibration...",
            command=self._open_calibration,
        ).pack(fill="x", pady=4, padx=5)

        # 3. Smooth
        smoothing_frame = ttk.LabelFrame(panel, text="3. Smooth (Savitzky-Golay)")
        smoothing_frame.pack(fill="x", pady=5, padx=2)
        sg_param_frame = ttk.Frame(smoothing_frame)
        sg_param_frame.pack(fill="x", pady=5)
        ttk.Label(sg_param_frame, text="Order:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(sg_param_frame, textvariable=self.sg_poly_order, width=5).grid(row=0, column=1)
        ttk.Label(sg_param_frame, text="Window:").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(sg_param_frame, textvariable=self.sg_frame_window, width=5).grid(row=0, column=3)
        ttk.Button(
            smoothing_frame, text="Apply Smoothing", command=self._apply_smoothing
        ).pack(fill="x", pady=5, padx=5)

        # 4. Baseline
        baseline_frame = ttk.LabelFrame(panel, text="4. Baseline Correction")
        baseline_frame.pack(fill="x", pady=5, padx=2)
        ttk.Button(
            baseline_frame, text="Adjust Baseline...", command=self._open_baseline_corrector
        ).pack(fill="x", pady=5, padx=5)
        self.baseline_algo_label = ttk.Label(
            baseline_frame,
            text=self._baseline_summary(),
            foreground="gray",
            wraplength=250,
        )
        self.baseline_algo_label.pack(fill="x", padx=5, pady=(0, 5))

        # 5. Normalization
        norm_frame = ttk.LabelFrame(panel, text="5. Normalization")
        norm_frame.pack(fill="x", pady=5, padx=2)
        combo = ttk.Combobox(
            norm_frame,
            textvariable=self.normalization_name,
            values=list(self._norm_code_to_name.values()),
            state="readonly",
        )
        combo.pack(fill="x", padx=5, pady=5)
        combo.bind("<<ComboboxSelected>>", self._on_norm_change)
        ttk.Button(
            norm_frame, text="Apply Normalization", command=self._apply_normalization
        ).pack(fill="x", padx=5, pady=(0, 5))

        # 5b. Derivative (applied after baseline, before normalization)
        deriv_frame = ttk.LabelFrame(panel, text="5b. Derivative (optional)")
        deriv_frame.pack(fill="x", pady=5, padx=2)
        ttk.Checkbutton(
            deriv_frame,
            text="Apply derivative to Final",
            variable=self.deriv_enabled,
            command=self._on_derivative_toggle,
        ).pack(anchor="w", padx=5)
        drow = ttk.Frame(deriv_frame); drow.pack(fill="x", padx=5, pady=2)
        ttk.Label(drow, text="Order:").pack(side="left")
        ttk.Entry(drow, textvariable=self.deriv_order, width=4).pack(side="left", padx=2)
        ttk.Label(drow, text="Win:").pack(side="left", padx=(6, 0))
        ttk.Entry(drow, textvariable=self.deriv_window, width=4).pack(side="left", padx=2)
        ttk.Label(drow, text="Poly:").pack(side="left", padx=(6, 0))
        ttk.Entry(drow, textvariable=self.deriv_polyorder, width=4).pack(side="left", padx=2)

        # 5c. Peak analysis
        peak_frame = ttk.LabelFrame(panel, text="5c. Peak Analysis")
        peak_frame.pack(fill="x", pady=5, padx=2)
        ttk.Button(
            peak_frame, text="Detect & Fit Peaks...", command=self._open_peak_analysis
        ).pack(fill="x", padx=5, pady=5)

        # 6. Export
        export_frame = ttk.LabelFrame(panel, text="6. Export")
        export_frame.pack(fill="x", pady=5, padx=2)
        ttk.Button(
            export_frame, text="Export Smoothed Data...", command=self._export_smoothed_data
        ).pack(fill="x", pady=2, padx=5)
        ttk.Button(
            export_frame, text="Export Final Data...", command=self._export_final_data
        ).pack(fill="x", pady=2, padx=5)

        # 7. Batch
        batch_frame = ttk.LabelFrame(panel, text="7. Batch & Multi-spectrum Analysis")
        batch_frame.pack(fill="x", pady=5, padx=2)
        ttk.Checkbutton(
            batch_frame,
            text="Parallel processing (joblib)",
            variable=self.batch_parallel,
        ).pack(anchor="w", padx=5)
        njobs_row = ttk.Frame(batch_frame)
        njobs_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(njobs_row, text="n_jobs (-1 = all cores):").pack(side="left")
        ttk.Entry(njobs_row, textvariable=self.batch_n_jobs, width=5).pack(side="right")
        ttk.Button(
            batch_frame, text="Run Batch Process...", command=self._start_batch_process
        ).pack(fill="x", pady=5, padx=5)

        ttk.Separator(batch_frame, orient="horizontal").pack(fill="x", pady=5)

        pca_scaling_row = ttk.Frame(batch_frame)
        pca_scaling_row.pack(fill="x", pady=2, padx=5)
        ttk.Label(pca_scaling_row, text="PCA Scaling:").pack(side="left")
        pca_combo = ttk.Combobox(
            pca_scaling_row,
            textvariable=self.pca_scaling_name,
            values=list(self._pca_code_to_name.values()),
            state="readonly",
            width=22,
        )
        pca_combo.pack(side="right")
        pca_combo.bind("<<ComboboxSelected>>", self._on_pca_scaling_change)

        pca_frame = ttk.Frame(batch_frame)
        pca_frame.pack(fill="x", pady=2, padx=5)
        ttk.Button(pca_frame, text="Run PCA", command=self._run_pca_analysis).pack(side="left")
        ttk.Label(pca_frame, text="Conf:").pack(side="right")
        ttk.Entry(pca_frame, textvariable=self.pca_confidence, width=6).pack(side="right", padx=2)

        nmf_frame = ttk.Frame(batch_frame)
        nmf_frame.pack(fill="x", pady=2, padx=5)
        ttk.Button(nmf_frame, text="Run NMF", command=self._run_nmf_analysis).pack(side="left")
        ttk.Entry(nmf_frame, textvariable=self.nmf_n_components, width=5).pack(side="right")
        ttk.Label(nmf_frame, text="Components:").pack(side="right")

        nmf_init_row = ttk.Frame(batch_frame)
        nmf_init_row.pack(fill="x", pady=2, padx=5)
        ttk.Label(nmf_init_row, text="NMF Init:").pack(side="left")
        nmf_combo = ttk.Combobox(
            nmf_init_row,
            textvariable=self.nmf_init_name,
            values=list(self._nmf_code_to_name.values()),
            state="readonly",
            width=22,
        )
        nmf_combo.pack(side="right")
        nmf_combo.bind("<<ComboboxSelected>>", self._on_nmf_init_change)

        rs_row = ttk.Frame(batch_frame)
        rs_row.pack(fill="x", padx=5, pady=(2, 5))
        ttk.Label(rs_row, text="NMF Random State:").pack(side="left")
        ttk.Entry(rs_row, textvariable=self.nmf_random_state, width=6).pack(side="right")

        ttk.Separator(batch_frame, orient="horizontal").pack(fill="x", pady=5)
        ttk.Button(batch_frame, text="Clustering (HCA / K-means / UMAP)...",
                   command=self._run_clustering).pack(fill="x", pady=2, padx=5)
        ttk.Button(batch_frame, text="Run MCR-ALS...",
                   command=self._run_mcr_als).pack(fill="x", pady=2, padx=5)

    def _build_plot_panel(self, parent) -> ttk.Frame:
        panel = ttk.Frame(parent)
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(panel)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.ax.set_visible(False)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        CustomToolbar(self.canvas, toolbar_frame)

        toggle_frame = ttk.LabelFrame(panel, text="Display Options")
        toggle_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        for text, var in [
            ("Raw", self.show_raw),
            ("Smooth", self.show_smooth),
            ("Baseline", self.show_baseline),
            ("Final", self.show_final),
        ]:
            ttk.Checkbutton(
                toggle_frame, text=text, variable=var, command=self._update_display
            ).pack(side="left", padx=5)

        return panel

    # ── Status & Progress ───────────────────────────────────────────────

    def _update_status(self, text: str):
        self.status_bar.config(text=text)

    def _show_progress(self, current: int, total: int):
        if not self.progress_bar.winfo_ismapped():
            self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = current

    def _hide_progress(self):
        self.progress_bar['value'] = 0
        self.progress_bar.pack_forget()

    def _baseline_summary(self) -> str:
        algo = self.baseline_params.get('algorithm', '?')
        fullname = self.config.get("baseline_algorithms", {}).get(algo, algo)
        params = self.baseline_params.get('params', {})
        parts = ", ".join(f"{k}={_fmt_num(v)}" for k, v in params.items())
        return f"{fullname}\n({parts})"

    # ── Plot ────────────────────────────────────────────────────────────

    def _setup_plot(self, title: str, normalized: bool = False):
        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel("Raman shift (cm\u207b\u00b9)")
        if normalized:
            self.ax.set_ylabel("Normalized Intensity")
        else:
            self.ax.set_ylabel("Intensity (AU)")
        self.ax.grid(True)

    def _update_display(self):
        is_normalized = self.last_processing_stage == ProcessingStage.NORMALIZED
        self._setup_plot("Raman Spectrum", normalized=is_normalized)

        if self.current_selection_idx is None or self.processor.y_raw is None:
            self.ax.set_visible(False)
            self.canvas.draw()
            return

        self.ax.set_visible(True)
        idx = self.current_selection_idx
        self.ax.set_title(f"Spectrum: {self.processor.sample_names[idx]}")

        y_raw_selected = self.processor.y_raw[:, idx]
        x = self.processor.x

        plotted = False
        if self.show_raw.get() and y_raw_selected is not None:
            self.ax.plot(x, y_raw_selected, label="Raw", color='gray', alpha=0.7)
            plotted = True
        if self.show_smooth.get() and self.y_mid is not None:
            self.ax.plot(x, self.y_mid, label="Smooth", color='blue')
            plotted = True
        if self.show_baseline.get() and self.baseline is not None:
            self.ax.plot(x, self.baseline, label="Baseline", color='magenta', linestyle='--')
            plotted = True
        if self.show_final.get() and self.y_final is not None:
            norm_code = self.normalization_code.get()
            label = f"Final ({norm_code.upper()})" if norm_code != 'none' else "Final"
            self.ax.plot(x, self.y_final, label=label, color='red')
            plotted = True

        if plotted:
            self.ax.legend()

        # If normalized Final is shown together with non-normalized traces,
        # show a text warning about the scale mismatch
        if is_normalized and (self.show_raw.get() or self.show_smooth.get()):
            self.ax.text(
                0.01, 0.99,
                "⚠ Scale mismatch: Final is normalized, Raw/Smooth are not",
                transform=self.ax.transAxes,
                fontsize=8, color='darkorange', va='top',
            )

        self.fig.tight_layout()
        self.canvas.draw()

    # ── Data Import ─────────────────────────────────────────────────────

    def _import_data(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("All supported", "*.xlsx *.xls *.csv *.txt *.asc *.dat"),
                ("Excel", "*.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Text / ASCII", "*.txt *.asc *.dat"),
                ("All files", "*.*"),
            ]
        )
        if not filepath:
            return

        try:
            num_spectra = self.processor.load_data(filepath)
            self.info_label.config(
                text=f"File: {os.path.basename(filepath)}\nSpectra: {num_spectra}"
            )
            self.listbox.delete(0, tk.END)
            with self._batch_lock:
                self.batch_result_df = None
            for name in self.processor.sample_names:
                self.listbox.insert(tk.END, name)
            if num_spectra > 0:
                self.listbox.selection_set(0)
                self.current_selection_idx = 0
                self._apply_range_filter()
            self._update_status(f"Loaded {num_spectra} spectra from {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            self._update_status("Import failed.")

    def _import_params(self):
        filepath = filedialog.askopenfilename(
            title="Select a parameter file",
            filetypes=[("JSON files", "*.json")],
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                params = json.load(f)

            self.lower_bound.set(params['range']['lower_bound'])
            self.upper_bound.set(params['range']['upper_bound'])
            self.sg_poly_order.set(params['smoothing']['sg_poly_order'])
            self.sg_frame_window.set(params['smoothing']['sg_frame_window'])
            self.baseline_params = params['baseline']
            self.baseline_algo_label.config(text=self._baseline_summary())
            if 'preprocessing' in params:
                self.apply_cosmic_ray.set(
                    params['preprocessing'].get('apply_cosmic_ray', False)
                )
            if 'normalization' in params:
                norm_code = params['normalization']
                if norm_code in self._norm_code_to_name:
                    self.normalization_code.set(norm_code)
                    self.normalization_name.set(self._norm_code_to_name[norm_code])
            self._apply_range_filter()
            messagebox.showinfo("Success", f"Parameters loaded from {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror(
                "Parameter Import Error", f"Failed to load parameters.\nError: {e}"
            )

    # ── Selection & Processing ──────────────────────────────────────────

    def _on_listbox_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        new_idx = sel[0]
        if new_idx == self.current_selection_idx:
            return

        self.current_selection_idx = new_idx
        stage = self.last_processing_stage
        if stage in (ProcessingStage.BASELINE_CORRECTED, ProcessingStage.NORMALIZED):
            self._run_full_processing()
        elif stage == ProcessingStage.SMOOTHED:
            self._apply_smoothing()
        else:
            self._clear_and_redraw_raw()

    def _apply_range_filter(self):
        if self.processor.x_full is None:
            return
        try:
            lower = self.lower_bound.get()
            upper = self.upper_bound.get()
            if lower >= upper:
                messagebox.showerror("Range Error", "Lower bound must be less than upper bound.")
                return
        except tk.TclError:
            messagebox.showerror("Input Error", "Enter valid numbers for range.")
            return

        if not self.processor.filter_data_by_range(lower, upper):
            messagebox.showwarning(
                "Range Warning", "No data in the specified range. Displaying full range."
            )
        self._clear_and_redraw_raw()

    def _clear_and_redraw_raw(self):
        self.y_processed = None
        self.y_mid = None
        self.y_final = None
        self.baseline = None
        self.last_processing_stage = ProcessingStage.RAW
        self._update_display()

    def _apply_smoothing(self):
        if self.current_selection_idx is None:
            messagebox.showwarning("Warning", "Please select a spectrum first.")
            return
        if not self._validate_sg_params():
            return

        self.baseline = None
        self.y_final = None

        self.y_raw = self.processor.y_raw[:, self.current_selection_idx]
        current_y = self.y_raw.copy()

        if self.apply_cosmic_ray.get():
            current_y = self.processor.remove_cosmic_rays(current_y)

        self.y_processed = current_y
        self.y_mid = self.processor.apply_sg_filter(
            self.y_processed, self.sg_poly_order.get(), self.sg_frame_window.get()
        )
        self.last_processing_stage = ProcessingStage.SMOOTHED
        self._update_display()
        self._update_status(
            f"Smoothing applied to {self.processor.sample_names[self.current_selection_idx]}"
        )

    def _apply_baseline(self):
        if self.y_mid is None:
            messagebox.showwarning("Warning", "Please apply smoothing first.")
            return

        algo = self.baseline_params['algorithm']
        params = self.baseline_params['params']
        try:
            self.baseline = self.processor.compute_baseline(self.y_mid, algo, params)
        except Exception as e:
            messagebox.showerror("Baseline Error", f"Failed: {e}")
            return

        self.y_final = self.y_mid - self.baseline
        # Apply derivative transform if enabled (before normalization)
        if self.deriv_enabled.get():
            try:
                self.y_final = compute_derivative(
                    self.y_final,
                    order=self.deriv_order.get(),
                    window=self.deriv_window.get(),
                    polyorder=self.deriv_polyorder.get(),
                )
            except Exception as e:
                messagebox.showerror("Derivative Error", f"Failed: {e}")
                return
        self.last_processing_stage = ProcessingStage.BASELINE_CORRECTED
        self._update_display()
        self._update_status("Baseline correction applied.")

    def _apply_normalization(self):
        if self.y_final is None:
            messagebox.showwarning(
                "Warning", "Please apply smoothing and baseline correction first."
            )
            return
        method = self.normalization_code.get()
        if method == 'none':
            # Restore display toggles and re-derive y_final
            self.show_raw.set(True)
            self.show_smooth.set(True)
            self.show_baseline.set(True)
            if self.baseline is not None and self.y_mid is not None:
                self.y_final = self.y_mid - self.baseline
                self.last_processing_stage = ProcessingStage.BASELINE_CORRECTED
            self._update_display()
            self._update_status("Normalization cleared — display restored.")
            return

        # Always re-compute from (y_mid - baseline) [+ derivative if enabled]
        # to avoid compounding effects
        if self.y_mid is not None and self.baseline is not None:
            base = self.y_mid - self.baseline
            if self.deriv_enabled.get():
                try:
                    base = compute_derivative(
                        base,
                        order=self.deriv_order.get(),
                        window=self.deriv_window.get(),
                        polyorder=self.deriv_polyorder.get(),
                    )
                except Exception as e:
                    messagebox.showerror("Derivative Error", f"Failed: {e}")
                    return
        else:
            base = self.y_final
        self.y_final = normalize_spectrum(base, method=method)
        self.last_processing_stage = ProcessingStage.NORMALIZED

        # Hide Raw / Smooth / Baseline — their scale is incompatible with [0,1] or unit-norm
        self.show_raw.set(False)
        self.show_smooth.set(False)
        self.show_baseline.set(False)
        self.show_final.set(True)

        self._update_display()
        self._update_status(
            f"Normalization applied: {method.upper()}  "
            f"(Raw/Smooth/Baseline hidden — scale mismatch)"
        )

    def _on_derivative_toggle(self):
        """Re-run pipeline if we already have processed data."""
        if self.last_processing_stage in (
            ProcessingStage.BASELINE_CORRECTED, ProcessingStage.NORMALIZED
        ):
            self._run_full_processing()

    def _on_pca_scaling_change(self, event=None):
        name = self.pca_scaling_name.get()
        code = self._pca_name_to_code.get(name, "auto")
        self.pca_scaling_code.set(code)

    def _open_peak_analysis(self):
        if self.y_final is None or self.processor.x is None:
            messagebox.showwarning(
                "Peak Analysis",
                "Run smoothing + baseline correction first so a Final spectrum is available.",
            )
            return
        idx = self.current_selection_idx
        name = self.processor.sample_names[idx] if idx is not None else ""
        PeakAnalysisWindow(
            parent=self.root,
            x=self.processor.x,
            y=self.y_final,
            sample_name=name,
            default_prominence_percent=float(self.peak_cfg.get("prominence_percent", 2.0)),
            default_min_distance=int(self.peak_cfg.get("min_distance", 5)),
            default_profile=str(self.peak_cfg.get("default_profile", "gaussian")),
        )

    def _on_norm_change(self, event=None):
        name = self.normalization_name.get()
        code = self._norm_name_to_code.get(name, 'none')
        self.normalization_code.set(code)

    def _on_nmf_init_change(self, event=None):
        name = self.nmf_init_name.get()
        code = self._nmf_name_to_code.get(name, 'nndsvda')
        self.nmf_init_code.set(code)

    def _run_full_processing(self):
        if self.current_selection_idx is None:
            if self.listbox.size() > 0:
                self.current_selection_idx = 0
            else:
                messagebox.showwarning(
                    "Warning", "Please import data and select a spectrum first."
                )
                return
        self._apply_smoothing()
        if self.y_mid is not None:
            self._apply_baseline()
            if self.normalization_code.get() != 'none':
                self._apply_normalization()
            self._update_status(
                f"Processing complete for {self.processor.sample_names[self.current_selection_idx]}"
            )

    def _open_baseline_corrector(self):
        if self.y_mid is None:
            messagebox.showwarning("Warning", "Please apply smoothing first.")
            return
        BaselineCorrectorWindow(
            parent=self.root,
            processor=self.processor,
            y_mid=self.y_mid,
            current_params=self.baseline_params,
            algorithms=self.config.get("baseline_algorithms", {}),
            callback=self._on_baseline_applied,
        )

    def _on_baseline_applied(self, final_baseline: np.ndarray, final_params: Dict[str, Any]):
        self.baseline_params = final_params
        self.baseline = final_baseline
        self.y_final = self.y_mid - self.baseline
        if self.deriv_enabled.get():
            try:
                self.y_final = compute_derivative(
                    self.y_final,
                    order=self.deriv_order.get(),
                    window=self.deriv_window.get(),
                    polyorder=self.deriv_polyorder.get(),
                )
            except Exception as e:
                messagebox.showerror("Derivative Error", f"Failed: {e}")
                return
        self.last_processing_stage = ProcessingStage.BASELINE_CORRECTED
        self.baseline_algo_label.config(text=self._baseline_summary())
        if self.normalization_code.get() != 'none':
            self._apply_normalization()
        else:
            self._update_display()

    # ── Validation ──────────────────────────────────────────────────────

    def _validate_sg_params(self) -> bool:
        try:
            order = self.sg_poly_order.get()
            window = self.sg_frame_window.get()
            if order >= window:
                raise ValueError("'Polynomial Order' must be strictly less than 'Frame Window'.")
            if window % 2 == 0:
                raise ValueError("'Frame Window' must be an odd number.")
            if self.processor.x is not None and (window < 3 or window >= len(self.processor.x)):
                raise ValueError(
                    f"'Frame Window' must be >= 3 and less than the data points ({len(self.processor.x)})."
                )
            return True
        except tk.TclError:
            messagebox.showerror(
                "Parameter Error", "Please enter valid integer numbers for SG parameters."
            )
            return False
        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
            logging.error(f"Parameter validation failed: {e}")
            return False

    # ── Export ──────────────────────────────────────────────────────────

    def _collect_current_params(self) -> Dict[str, Any]:
        return {
            "range": {
                "lower_bound": self.lower_bound.get(),
                "upper_bound": self.upper_bound.get(),
            },
            "smoothing": {
                "sg_poly_order": self.sg_poly_order.get(),
                "sg_frame_window": self.sg_frame_window.get(),
            },
            "baseline": self.baseline_params,
            "normalization": self.normalization_code.get(),
            "preprocessing": {
                "apply_cosmic_ray": self.apply_cosmic_ray.get(),
            },
            "batch": {
                "parallel": self.batch_parallel.get(),
                "n_jobs": self.batch_n_jobs.get(),
            },
            "nmf": {
                "init": self.nmf_init_code.get(),
                "n_components": self.nmf_n_components.get(),
                "random_state": self.nmf_random_state.get(),
            },
            "derivative": {
                "enabled": self.deriv_enabled.get(),
                "order": self.deriv_order.get(),
                "window": self.deriv_window.get(),
                "polyorder": self.deriv_polyorder.get(),
            },
            "pca": {
                "scaling": self.pca_scaling_code.get(),
                "confidence": self.pca_confidence.get(),
            },
            "qc": self.qc_cfg,
        }

    def _export_data(self, data_dict: Dict[str, np.ndarray], default_filename: str):
        if self.current_selection_idx is None:
            messagebox.showwarning("Warning", "Please select a spectrum to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=default_filename,
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
        )
        if not filepath:
            return

        try:
            export_df = pd.DataFrame(data_dict).set_index('Raman shift (cm-1)').T
            if filepath.endswith('.csv'):
                export_df.to_csv(filepath, header=True, index=True)
            else:
                export_df.to_excel(filepath, header=True, index=True)

            params_to_save = self._collect_current_params()
            json_filepath = os.path.splitext(filepath)[0] + '.json'
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=4)

            messagebox.showinfo(
                "Success", f"Data and parameters exported to\n{os.path.dirname(filepath)}"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save file.\nError: {e}")

    def _export_smoothed_data(self):
        if self.y_mid is None:
            messagebox.showwarning("Warning", "No smoothed data to export.")
            return
        idx = self.current_selection_idx
        name = self.processor.sample_names[idx]
        self._export_data(
            {'Raman shift (cm-1)': self.processor.x, name: self.y_mid},
            f"Smoothed_{name}",
        )

    def _export_final_data(self):
        if self.y_final is None:
            messagebox.showwarning("Warning", "No final corrected data to export.")
            return
        idx = self.current_selection_idx
        name = self.processor.sample_names[idx]
        self._export_data(
            {'Raman shift (cm-1)': self.processor.x, name: self.y_final},
            f"Final_{name}",
        )

    # ── Batch Processing ────────────────────────────────────────────────

    def _start_batch_process(self):
        if self.processor.y_raw is None:
            messagebox.showerror("Error", "Please import data first.")
            return
        if not self._validate_sg_params():
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Batch-Processed File As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not output_path:
            return

        if os.path.exists(output_path):
            if not messagebox.askyesno(
                "Confirm Overwrite", f"'{os.path.basename(output_path)}' already exists. Overwrite?"
            ):
                return

        params = self._collect_current_params()
        thread = threading.Thread(
            target=self._batch_worker, args=(output_path, params), daemon=True
        )
        thread.start()

    def _batch_worker(self, output_path: str, params: Dict[str, Any]):
        self.root.after(0, self._update_status, "Batch processing started...")
        start_time = time.time()
        parallel = self.batch_parallel.get()
        n_jobs = self.batch_n_jobs.get()

        try:
            def progress_cb(current: int, total: int):
                self.root.after(0, self._update_status, f"Processing {current}/{total}...")
                self.root.after(0, self._show_progress, current, total)

            processed_df, summary = self.processor.run_batch_processing(
                params, progress_cb, parallel=parallel, n_jobs=n_jobs
            )

            with self._batch_lock:
                self.batch_result_df = processed_df

            qc_df = summary.get('qc_df')
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                processed_df.set_index('Raman shift (cm-1)').T.to_excel(
                    writer, sheet_name='Processed', header=True, index=True
                )
                if qc_df is not None and not qc_df.empty:
                    qc_df.to_excel(writer, sheet_name='QC', index=False)

            json_filepath = os.path.splitext(output_path)[0] + '.json'
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)

            elapsed = time.time() - start_time
            mode = f"parallel (n_jobs={n_jobs})" if parallel else "serial"
            qc_flagged = summary.get('qc_flagged', 0)
            result_message = (
                f"Batch process complete!\n\n"
                f"Mode: {mode}\n"
                f"Processed: {summary['processed']}\n"
                f"Failed: {summary['failed']}\n"
                f"QC Flagged: {qc_flagged}\n"
                f"Time: {elapsed:.2f}s"
            )
            self.root.after(0, lambda: messagebox.showinfo("Success", result_message))
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Batch Processing Error", str(e))
            )
            logging.error(f"Batch worker failed: {e}")
        finally:
            self.root.after(0, self._hide_progress)
            self.root.after(0, self._update_status, "Ready.")

    # ── PCA / NMF ───────────────────────────────────────────────────────

    def _run_pca_analysis(self):
        with self._batch_lock:
            batch_df = self.batch_result_df

        if batch_df is None:
            messagebox.showwarning("No Data", "Please run Batch Process first.")
            return

        try:
            analysis_params = self._collect_current_params()
            pca_results = self.processor.perform_pca(
                batch_df,
                scaling=self.pca_scaling_code.get(),
                confidence=float(self.pca_confidence.get()),
            )
            PCAResultWindow(self.root, pca_results, analysis_params)
        except Exception as e:
            messagebox.showerror("PCA Error", f"Failed to perform PCA.\nError: {e}")
            logging.error(f"PCA analysis failed: {e}")

    def _run_clustering(self):
        with self._batch_lock:
            batch_df = self.batch_result_df
        if batch_df is None:
            messagebox.showwarning("No Data", "Please run Batch Process first.")
            return
        try:
            ClusteringWindow(self.root, batch_df)
        except Exception as e:
            messagebox.showerror("Clustering Error", f"Failed: {e}")
            logging.error(f"Clustering failed: {e}")

    def _run_mcr_als(self):
        with self._batch_lock:
            batch_df = self.batch_result_df
        if batch_df is None:
            messagebox.showwarning("No Data", "Please run Batch Process first.")
            return
        try:
            MCRALSWindow(self.root, batch_df)
        except Exception as e:
            messagebox.showerror("MCR-ALS Error", f"Failed: {e}")
            logging.error(f"MCR-ALS failed: {e}")

    def _open_calibration(self):
        if self.processor.x is None:
            messagebox.showwarning("Calibration", "Please import data first.")
            return
        CalibrationWindow(
            self.root, self.processor,
            on_applied=self._on_calibration_applied,
        )

    def _on_calibration_applied(self):
        """Called after calibration — refresh the current view."""
        self._clear_and_redraw_raw()
        self._update_status("Calibration applied to Raman shift axis.")

    def _run_nmf_analysis(self):
        with self._batch_lock:
            batch_df = self.batch_result_df

        if batch_df is None:
            messagebox.showwarning("No Data", "Please run Batch Process first.")
            return

        try:
            n_components = self.nmf_n_components.get()
            n_samples = batch_df.shape[1] - 1
            if n_components < 2:
                messagebox.showerror("NMF Error", "Number of components must be 2 or greater.")
                return
            if n_components > n_samples:
                messagebox.showerror(
                    "NMF Error",
                    f"Number of components ({n_components}) cannot exceed "
                    f"number of samples ({n_samples}).",
                )
                return

            self._update_status("Running NMF analysis... please wait.")
            self.root.update_idletasks()

            nmf_results = self.processor.perform_nmf(
                batch_df,
                n_components,
                init=self.nmf_init_code.get(),
                random_state=self.nmf_random_state.get(),
            )
            NMFResultWindow(self.root, nmf_results)
            self._update_status("NMF analysis complete.")
        except Exception as e:
            messagebox.showerror("NMF Error", f"Failed to perform NMF.\nError: {e}")
            self._update_status("NMF analysis failed.")

    # ── Cleanup ─────────────────────────────────────────────────────────

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            plt.close(self.fig)
            self.root.destroy()


def _fmt_num(v) -> str:
    try:
        fv = float(v)
        if abs(fv) >= 1000 or (0 < abs(fv) < 0.01):
            return f"{fv:.2e}"
        if fv == int(fv):
            return str(int(fv))
        return f"{fv:.3g}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    root = tk.Tk()
    app = RamanProcessorApp(root)
    root.mainloop()
