"""Interactive peak detection and fitting window.

Workflow:
1. Auto-detect peaks via scipy.signal.find_peaks (adjustable prominence/distance).
2. Review detected peaks on the plot; edit the center-list if needed.
3. Fit a multi-peak model (Gaussian / Lorentzian / Voigt / PseudoVoigt) with lmfit.
4. Inspect residuals, R², and per-peak center/FWHM/area/amplitude.
5. Export full fit report + peak table to Excel.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional
import numpy as np
import pandas as pd

from processing_logic import detect_peaks, fit_peaks, PEAK_PROFILES


class PeakAnalysisWindow(tk.Toplevel):
    def __init__(
        self,
        parent,
        x: np.ndarray,
        y: np.ndarray,
        sample_name: str = "",
        default_prominence_percent: float = 2.0,
        default_min_distance: int = 5,
        default_profile: str = "gaussian",
    ):
        super().__init__(parent)
        self.title(f"Peak Analysis — {sample_name}" if sample_name else "Peak Analysis")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sample_name = sample_name

        self.peaks_info: Optional[dict] = None
        self.fit_result: Optional[dict] = None
        self.figures: list[plt.Figure] = []

        # UI vars
        self.prominence_pct = tk.DoubleVar(value=default_prominence_percent)
        self.min_distance = tk.IntVar(value=default_min_distance)
        self.profile_var = tk.StringVar(value=default_profile)
        self.fit_window_var = tk.StringVar(value="")  # empty = full range
        self.edit_centers_var = tk.StringVar(value="")

        self._build_ui()
        self._run_detection()
        self.grab_set()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root_pane = ttk.PanedWindow(self, orient="horizontal")
        root_pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: controls
        ctrl = ttk.Frame(root_pane, width=280)
        ctrl.pack_propagate(False)
        root_pane.add(ctrl, weight=0)
        self._build_controls(ctrl)

        # Right: plot + results
        right = ttk.Frame(root_pane)
        root_pane.add(right, weight=1)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        self.plot_tab = ttk.Frame(notebook)
        self.table_tab = ttk.Frame(notebook)
        self.report_tab = ttk.Frame(notebook)
        notebook.add(self.plot_tab, text="Spectrum & Fit")
        notebook.add(self.table_tab, text="Peak Table")
        notebook.add(self.report_tab, text="Fit Report")

        self._build_plot(self.plot_tab)
        self._build_table(self.table_tab)
        self._build_report(self.report_tab)

    def _build_controls(self, parent):
        det = ttk.LabelFrame(parent, text="1. Detection")
        det.pack(fill="x", pady=5)
        row = ttk.Frame(det); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Prominence (% of range):").pack(side="left")
        ttk.Entry(row, textvariable=self.prominence_pct, width=6).pack(side="right")
        row = ttk.Frame(det); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Min distance (pts):").pack(side="left")
        ttk.Entry(row, textvariable=self.min_distance, width=6).pack(side="right")
        ttk.Button(det, text="Re-detect Peaks", command=self._run_detection).pack(
            fill="x", padx=5, pady=5
        )
        self.detected_label = ttk.Label(det, text="Detected: 0", foreground="gray")
        self.detected_label.pack(anchor="w", padx=5)

        edit = ttk.LabelFrame(parent, text="2. Edit Peak Centers (cm⁻¹)")
        edit.pack(fill="x", pady=5)
        ttk.Label(
            edit, text="Comma-separated. Empty = use auto-detected.",
            foreground="gray", wraplength=260, justify="left"
        ).pack(anchor="w", padx=5, pady=(2, 0))
        ttk.Entry(edit, textvariable=self.edit_centers_var).pack(
            fill="x", padx=5, pady=5
        )

        fit = ttk.LabelFrame(parent, text="3. Fit")
        fit.pack(fill="x", pady=5)
        row = ttk.Frame(fit); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Profile:").pack(side="left")
        ttk.Combobox(
            row, textvariable=self.profile_var,
            values=sorted(PEAK_PROFILES), state="readonly", width=14,
        ).pack(side="right")
        row = ttk.Frame(fit); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Fit window (cm⁻¹, blank=all):").pack(side="left")
        ttk.Entry(row, textvariable=self.fit_window_var, width=7).pack(side="right")
        ttk.Button(fit, text="Run Fit", command=self._run_fit).pack(
            fill="x", padx=5, pady=5
        )

        exp = ttk.LabelFrame(parent, text="4. Export")
        exp.pack(fill="x", pady=5)
        ttk.Button(exp, text="Export Fit to Excel...", command=self._export).pack(
            fill="x", padx=5, pady=5
        )

    def _build_plot(self, parent):
        self.fig, (self.ax, self.ax_resid) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        self.figures.append(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_table(self, parent):
        cols = ("index", "center", "fwhm", "amplitude", "height", "area")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_report(self, parent):
        self.report_text = tk.Text(parent, wrap="word", font=("Courier New", 9))
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.report_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    # ── Actions ─────────────────────────────────────────────────────────
    def _run_detection(self):
        try:
            self.peaks_info = detect_peaks(
                self.x, self.y,
                prominence_percent=float(self.prominence_pct.get()),
                min_distance_pts=int(self.min_distance.get()),
            )
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
            return
        n = len(self.peaks_info["indices"])
        self.detected_label.config(text=f"Detected: {n}")
        # Populate edit box with detected centers as starting point
        if n > 0:
            self.edit_centers_var.set(
                ", ".join(f"{p:.1f}" for p in self.peaks_info["positions"])
            )
        self._redraw(show_fit=False)

    def _parse_centers(self) -> np.ndarray:
        txt = self.edit_centers_var.get().strip()
        if not txt:
            return np.asarray(self.peaks_info["positions"], dtype=float) if self.peaks_info else np.array([])
        try:
            parts = [float(p.strip()) for p in txt.split(",") if p.strip()]
            return np.array(parts, dtype=float)
        except ValueError:
            raise ValueError("Could not parse peak centers — must be comma-separated numbers.")

    def _run_fit(self):
        try:
            centers = self._parse_centers()
            if len(centers) == 0:
                messagebox.showwarning("Fit", "No peaks to fit.")
                return
            fw_str = self.fit_window_var.get().strip()
            fit_window = float(fw_str) if fw_str else None
            self.fit_result = fit_peaks(
                self.x, self.y, centers,
                profile=self.profile_var.get(),
                fit_window=fit_window,
            )
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))
            return

        self._populate_table()
        self._populate_report()
        self._redraw(show_fit=True)

    def _populate_table(self):
        self.tree.delete(*self.tree.get_children())
        if not self.fit_result:
            return
        for p in self.fit_result["peaks"]:
            self.tree.insert(
                "", "end",
                values=(
                    p["index"],
                    f"{p['center']:.2f}",
                    f"{p['fwhm']:.2f}",
                    f"{p['amplitude']:.3g}",
                    f"{p['height']:.3g}",
                    f"{p['area']:.3g}",
                ),
            )

    def _populate_report(self):
        self.report_text.delete("1.0", "end")
        if not self.fit_result:
            return
        header = (
            f"Profile: {self.fit_result['profile']}\n"
            f"R² = {self.fit_result['r_squared']:.6f}   "
            f"χ² = {self.fit_result['chisqr']:.4g}   "
            f"reduced χ² = {self.fit_result['redchi']:.4g}\n"
            f"n_peaks = {self.fit_result['n_peaks']}\n"
            "─" * 60 + "\n\n"
        )
        self.report_text.insert("end", header)
        self.report_text.insert("end", self.fit_result["report"])

    def _redraw(self, show_fit: bool):
        self.ax.clear()
        self.ax_resid.clear()

        self.ax.plot(self.x, self.y, color="gray", lw=1, label="Spectrum")

        # Detected peaks
        if self.peaks_info and len(self.peaks_info["positions"]) > 0:
            self.ax.scatter(
                self.peaks_info["positions"],
                self.peaks_info["heights"],
                color="blue", marker="v", zorder=5, label="Detected",
            )

        if show_fit and self.fit_result is not None:
            xf = self.fit_result["x_fit"]
            self.ax.plot(xf, self.fit_result["best_fit"], color="red", lw=1.5, label="Fit")
            # Individual components
            for name, comp in self.fit_result["components"].items():
                self.ax.fill_between(xf, comp, alpha=0.2)
            # Residuals
            residuals = self.fit_result["y_fit"] - self.fit_result["best_fit"]
            self.ax_resid.plot(xf, residuals, color="purple", lw=0.8)
            self.ax_resid.axhline(0, color="black", lw=0.5)
            self.ax_resid.set_ylabel("Residual")

        self.ax.set_title(f"Peak Analysis: {self.sample_name}" if self.sample_name else "Peak Analysis")
        self.ax.set_ylabel("Intensity")
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax_resid.set_xlabel("Raman shift (cm⁻¹)")
        self.ax_resid.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _export(self):
        if self.fit_result is None:
            messagebox.showwarning("Export", "Run a fit first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Peak Fit As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            peaks_df = pd.DataFrame(self.fit_result["peaks"])
            info_rows = [
                ("Sample", self.sample_name),
                ("Profile", self.fit_result["profile"]),
                ("R_squared", self.fit_result["r_squared"]),
                ("chisqr", self.fit_result["chisqr"]),
                ("reduced_chisqr", self.fit_result["redchi"]),
                ("n_peaks", self.fit_result["n_peaks"]),
            ]
            info_df = pd.DataFrame(info_rows, columns=["Category", "Value"])
            xf = self.fit_result["x_fit"]
            trace_df = pd.DataFrame({
                "Raman shift (cm-1)": xf,
                "Observed": self.fit_result["y_fit"],
                "Fit": self.fit_result["best_fit"],
                "Residual": self.fit_result["y_fit"] - self.fit_result["best_fit"],
            })
            for name, comp in self.fit_result["components"].items():
                trace_df[name.rstrip("_")] = comp

            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                info_df.to_excel(writer, sheet_name="Fit_Info", index=False)
                peaks_df.to_excel(writer, sheet_name="Peaks", index=False)
                trace_df.to_excel(writer, sheet_name="Traces", index=False)

            messagebox.showinfo("Success", f"Fit results exported to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
