"""Wavelength (Raman shift) calibration window.

Workflow:
1. Pick a reference standard (Si, polystyrene, cyclohexane, etc.) —
   or enter custom reference peaks.
2. Auto-detect peaks on the currently-selected spectrum (or enter
   measured peak centers manually).
3. Match measured <-> reference peaks (nearest-by-default; editable).
4. Fit polynomial shift (order 1 or 2) and view residuals + RMS.
5. Apply correction to the DataProcessor's Raman shift axis.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional
import numpy as np

from processing_logic import (
    CALIBRATION_STANDARDS,
    detect_peaks,
    fit_calibration_polynomial,
    apply_calibration,
)


class CalibrationWindow(tk.Toplevel):
    def __init__(self, parent, processor, on_applied=None):
        super().__init__(parent)
        self.title("Wavelength Calibration")
        self.geometry("1050x680")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.processor = processor
        self.on_applied = on_applied
        self.figures: list[plt.Figure] = []
        self.fit_result: Optional[dict] = None

        # UI vars
        self.standard_var = tk.StringVar(value="silicon")
        self.custom_ref_var = tk.StringVar(value="")
        self.measured_var = tk.StringVar(value="")
        self.order_var = tk.IntVar(value=1)
        self.prominence_pct = tk.DoubleVar(value=2.0)
        self.tolerance_var = tk.DoubleVar(value=15.0)  # cm^-1

        self._build_ui()
        self._seed_reference_from_standard()
        self.grab_set()

    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.Frame(pane, width=320); ctrl.pack_propagate(False)
        pane.add(ctrl, weight=0)
        self._build_controls(ctrl)

        right = ttk.Frame(pane); pane.add(right, weight=1)
        self.fig, (self.ax_spec, self.ax_resid) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
        self.figures.append(self.fig)
        FigureCanvasTkAgg(self.fig, master=right).get_tk_widget().pack(fill="both", expand=True)
        self._draw_initial()

    def _build_controls(self, parent):
        std = ttk.LabelFrame(parent, text="1. Reference Standard")
        std.pack(fill="x", pady=5)
        row = ttk.Frame(std); row.pack(fill="x", padx=5, pady=3)
        ttk.Label(row, text="Standard:").pack(side="left")
        cb = ttk.Combobox(
            row, textvariable=self.standard_var,
            values=list(CALIBRATION_STANDARDS.keys()) + ["custom"],
            state="readonly", width=16,
        )
        cb.pack(side="right"); cb.bind("<<ComboboxSelected>>", lambda e: self._seed_reference_from_standard())
        ttk.Label(std, text="Reference peaks (cm⁻¹, comma-separated):",
                  foreground="gray").pack(anchor="w", padx=5, pady=(5, 0))
        ttk.Entry(std, textvariable=self.custom_ref_var).pack(fill="x", padx=5, pady=3)

        det = ttk.LabelFrame(parent, text="2. Measured Peaks")
        det.pack(fill="x", pady=5)
        row = ttk.Frame(det); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Detect prominence (%):").pack(side="left")
        ttk.Entry(row, textvariable=self.prominence_pct, width=6).pack(side="right")
        ttk.Button(det, text="Auto-detect on current spectrum",
                   command=self._auto_detect).pack(fill="x", padx=5, pady=3)
        ttk.Label(det, text="Measured peaks (cm⁻¹, comma-separated):",
                  foreground="gray").pack(anchor="w", padx=5, pady=(5, 0))
        ttk.Entry(det, textvariable=self.measured_var).pack(fill="x", padx=5, pady=3)

        match = ttk.LabelFrame(parent, text="3. Match & Fit")
        match.pack(fill="x", pady=5)
        row = ttk.Frame(match); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Match tolerance (cm⁻¹):").pack(side="left")
        ttk.Entry(row, textvariable=self.tolerance_var, width=6).pack(side="right")
        row = ttk.Frame(match); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Polynomial order:").pack(side="left")
        ttk.Combobox(row, textvariable=self.order_var, values=[1, 2, 3],
                     state="readonly", width=5).pack(side="right")
        ttk.Button(match, text="Run Fit", command=self._run_fit).pack(fill="x", padx=5, pady=5)

        apply_frame = ttk.LabelFrame(parent, text="4. Apply")
        apply_frame.pack(fill="x", pady=5)
        self.fit_summary = ttk.Label(apply_frame, text="(no fit yet)", foreground="gray",
                                     wraplength=280, justify="left")
        self.fit_summary.pack(anchor="w", padx=5, pady=3)
        ttk.Button(apply_frame, text="Apply Calibration to Axis",
                   command=self._apply).pack(fill="x", padx=5, pady=5)

    def _parse_list(self, text: str) -> np.ndarray:
        txt = text.strip()
        if not txt:
            return np.array([])
        try:
            return np.array([float(p.strip()) for p in txt.split(",") if p.strip()], dtype=float)
        except ValueError:
            raise ValueError("Could not parse peak list — must be comma-separated numbers.")

    def _seed_reference_from_standard(self):
        name = self.standard_var.get()
        if name in CALIBRATION_STANDARDS:
            vals = CALIBRATION_STANDARDS[name]
            self.custom_ref_var.set(", ".join(str(v) for v in vals))
        elif name == "custom":
            pass  # leave editable

    def _auto_detect(self):
        if self.processor.x is None or self.processor.y_raw is None:
            messagebox.showwarning("Calibration", "No data loaded.")
            return
        # Use first (or any) spectrum
        y = self.processor.y_raw[:, 0]
        x = self.processor.x
        info = detect_peaks(x, y, prominence_percent=float(self.prominence_pct.get()))
        self.measured_var.set(", ".join(f"{p:.1f}" for p in info["positions"]))
        self._redraw()

    def _run_fit(self):
        try:
            ref = self._parse_list(self.custom_ref_var.get())
            meas = self._parse_list(self.measured_var.get())
            if len(ref) == 0 or len(meas) == 0:
                raise ValueError("Both reference and measured peak lists are required.")

            tol = float(self.tolerance_var.get())
            # Greedy nearest-pair matching within tolerance
            matched_meas: list[float] = []
            matched_ref: list[float] = []
            used_ref = set()
            for m in meas:
                diffs = np.abs(ref - m)
                order = np.argsort(diffs)
                for idx in order:
                    if idx in used_ref:
                        continue
                    if diffs[idx] <= tol:
                        matched_meas.append(float(m))
                        matched_ref.append(float(ref[idx]))
                        used_ref.add(int(idx))
                    break
            if len(matched_meas) < self.order_var.get() + 1:
                raise ValueError(
                    f"Only {len(matched_meas)} peaks matched within tolerance — "
                    f"need at least {self.order_var.get() + 1} for order-{self.order_var.get()} fit."
                )

            self.fit_result = fit_calibration_polynomial(
                np.array(matched_meas),
                np.array(matched_ref),
                order=self.order_var.get(),
            )
        except Exception as e:
            messagebox.showerror("Calibration Fit", str(e))
            return

        coef_str = ", ".join(f"{c:.6g}" for c in self.fit_result["coefficients"])
        self.fit_summary.config(
            text=(
                f"Matched: {len(self.fit_result['measured'])} peaks\n"
                f"Order {self.fit_result['order']}: [{coef_str}]\n"
                f"RMS residual: {self.fit_result['rms']:.3f} cm⁻¹"
            ),
            foreground="black",
        )
        self._redraw()

    def _draw_initial(self):
        self.ax_spec.clear(); self.ax_resid.clear()
        if self.processor.x is not None and self.processor.y_raw is not None:
            self.ax_spec.plot(self.processor.x, self.processor.y_raw[:, 0],
                              color="gray", lw=1, label="Spectrum[0]")
        self.ax_spec.set_title("Calibration")
        self.ax_spec.set_xlabel("Raman shift (cm⁻¹)")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.legend(loc="upper right", fontsize=8)
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_resid.set_xlabel("Measured (cm⁻¹)"); self.ax_resid.set_ylabel("Residual")
        self.ax_resid.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _redraw(self):
        self._draw_initial()
        try:
            ref = self._parse_list(self.custom_ref_var.get())
            meas = self._parse_list(self.measured_var.get())
        except Exception:
            ref, meas = np.array([]), np.array([])

        for p in ref:
            self.ax_spec.axvline(p, color="green", lw=0.8, alpha=0.5)
        for p in meas:
            self.ax_spec.axvline(p, color="red", lw=0.8, alpha=0.7, linestyle="--")

        if self.fit_result is not None:
            self.ax_resid.clear()
            self.ax_resid.scatter(self.fit_result["measured"], self.fit_result["residuals"])
            self.ax_resid.axhline(0, color="black", lw=0.5)
            self.ax_resid.set_xlabel("Measured (cm⁻¹)")
            self.ax_resid.set_ylabel("Residual (cm⁻¹)")
            self.ax_resid.set_title(f"RMS = {self.fit_result['rms']:.3f} cm⁻¹")
            self.ax_resid.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _apply(self):
        if self.fit_result is None:
            messagebox.showwarning("Calibration", "Run a fit first.")
            return
        if not messagebox.askyesno(
            "Apply Calibration",
            f"This will modify the Raman shift axis (order {self.fit_result['order']} polynomial).\n"
            f"RMS = {self.fit_result['rms']:.3f} cm⁻¹\n\nContinue?",
        ):
            return
        try:
            self.processor.recalibrate_axis(self.fit_result["coefficients"])
        except Exception as e:
            messagebox.showerror("Apply Error", str(e))
            return
        if self.on_applied is not None:
            try:
                self.on_applied()
            except Exception:
                pass
        messagebox.showinfo("Success", "Calibration applied to Raman shift axis.")
        self.destroy()

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
