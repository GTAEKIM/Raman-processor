"""Hyperspectral Raman mapping window.

Features:
- Load an X/Y/shift wide-table mapping file.
- Band-integrated 2D heatmap (trapezoid / sum / max / mean).
- Click a pixel to view its spectrum on the right panel.
- Export the heatmap to a CSV/Excel; flatten cube -> batch DataFrame
  and send to PCA / NMF / MCR-ALS via the main app's batch pipeline.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from processing_logic import (
    load_mapping_file,
    integrate_band,
    cube_to_batch_df,
)


class MappingWindow(tk.Toplevel):
    def __init__(self, parent, on_cube_to_batch=None):
        super().__init__(parent)
        self.title("Hyperspectral Raman Mapping")
        self.geometry("1200x720")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Callback to push cube-as-DataFrame into main app's batch slot
        self.on_cube_to_batch = on_cube_to_batch

        self.mapping: Optional[Dict[str, Any]] = None
        self.current_map: Optional[np.ndarray] = None
        self.figures: list[plt.Figure] = []
        self.selected_pixel: Optional[tuple[int, int]] = None

        # UI vars
        self.band_lo = tk.DoubleVar(value=1550.0)
        self.band_hi = tk.DoubleVar(value=1650.0)
        self.method_var = tk.StringVar(value="trapezoid")
        self.cmap_var = tk.StringVar(value="viridis")

        self._build_ui()
        self.grab_set()

    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: controls
        ctrl = ttk.Frame(pane, width=260); ctrl.pack_propagate(False)
        pane.add(ctrl, weight=0)
        self._build_controls(ctrl)

        # Center: heatmap
        mid = ttk.Frame(pane); pane.add(mid, weight=2)
        self.fig_map, self.ax_map = plt.subplots(figsize=(6, 5.5))
        self.figures.append(self.fig_map)
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=mid)
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_map.mpl_connect("button_press_event", self._on_map_click)
        self._draw_map_placeholder()

        # Right: spectrum panel
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(6, 5.5))
        self.figures.append(self.fig_spec)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=right)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)
        self._draw_spec_placeholder()

        self.status = ttk.Label(self, text="Load a mapping file to begin.",
                                foreground="gray")
        self.status.pack(side="bottom", fill="x", padx=8, pady=4)

    def _build_controls(self, parent):
        io = ttk.LabelFrame(parent, text="File")
        io.pack(fill="x", pady=5)
        ttk.Button(io, text="Load Mapping File...", command=self._load_file).pack(
            fill="x", padx=5, pady=5
        )
        self.file_label = ttk.Label(io, text="(none)", foreground="gray", wraplength=220)
        self.file_label.pack(anchor="w", padx=5)

        band = ttk.LabelFrame(parent, text="Spectral Band (cm⁻¹)")
        band.pack(fill="x", pady=5)
        row = ttk.Frame(band); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Low:").pack(side="left")
        ttk.Entry(row, textvariable=self.band_lo, width=8).pack(side="right")
        row = ttk.Frame(band); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="High:").pack(side="left")
        ttk.Entry(row, textvariable=self.band_hi, width=8).pack(side="right")
        row = ttk.Frame(band); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Method:").pack(side="left")
        ttk.Combobox(
            row, textvariable=self.method_var,
            values=["trapezoid", "sum", "max", "mean"],
            state="readonly", width=10,
        ).pack(side="right")
        row = ttk.Frame(band); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Colormap:").pack(side="left")
        ttk.Combobox(
            row, textvariable=self.cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "hot", "jet", "gray"],
            state="readonly", width=10,
        ).pack(side="right")
        ttk.Button(band, text="Compute Map", command=self._compute_map).pack(
            fill="x", padx=5, pady=5
        )

        send = ttk.LabelFrame(parent, text="Send to Batch")
        send.pack(fill="x", pady=5)
        ttk.Label(
            send, text="Flatten cube into a batch DataFrame\n"
                      "(each pixel = one sample) and push\nto the main app for PCA/NMF/MCR-ALS.",
            foreground="gray", justify="left", wraplength=220,
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Button(send, text="Send Cube to Batch Pipeline",
                   command=self._send_to_batch).pack(fill="x", padx=5, pady=5)

        exp = ttk.LabelFrame(parent, text="Export")
        exp.pack(fill="x", pady=5)
        ttk.Button(exp, text="Export Map to Excel...",
                   command=self._export_map).pack(fill="x", padx=5, pady=5)

    # ── Plot helpers ────────────────────────────────────────────────────
    def _draw_map_placeholder(self):
        self.ax_map.clear()
        self.ax_map.text(0.5, 0.5, "Load a mapping file", ha="center", va="center",
                         transform=self.ax_map.transAxes, color="gray")
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        self.fig_map.tight_layout()
        self.canvas_map.draw()

    def _draw_spec_placeholder(self):
        self.ax_spec.clear()
        self.ax_spec.text(0.5, 0.5, "Click a pixel to view its spectrum",
                          ha="center", va="center",
                          transform=self.ax_spec.transAxes, color="gray")
        self.ax_spec.set_xticks([]); self.ax_spec.set_yticks([])
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    # ── Actions ─────────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Mapping files", "*.csv *.xlsx *.xls *.txt *.asc *.dat"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.mapping = load_mapping_file(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return
        import os
        self.file_label.config(text=os.path.basename(path))
        cube = self.mapping["cube"]
        self.status.config(
            text=f"Loaded {cube.shape[1]}×{cube.shape[0]} px × {cube.shape[2]} shifts"
        )
        # Seed band from data range middle
        rs = self.mapping["raman_shifts"]
        self.band_lo.set(float(rs.min() + (rs.max() - rs.min()) * 0.4))
        self.band_hi.set(float(rs.min() + (rs.max() - rs.min()) * 0.6))
        self._compute_map()

    def _compute_map(self):
        if self.mapping is None:
            return
        try:
            band = (float(self.band_lo.get()), float(self.band_hi.get()))
            self.current_map = integrate_band(
                self.mapping["cube"], self.mapping["raman_shifts"],
                band=band, method=self.method_var.get(),
            )
        except Exception as e:
            messagebox.showerror("Compute Error", str(e))
            return
        self._draw_map()

    def _draw_map(self):
        self.ax_map.clear()
        x = self.mapping["x_coords"]; y = self.mapping["y_coords"]
        extent = [x.min(), x.max(), y.min(), y.max()]
        im = self.ax_map.imshow(
            self.current_map, origin="lower", aspect="auto",
            extent=extent, cmap=self.cmap_var.get(),
        )
        self.fig_map.colorbar(im, ax=self.ax_map, label="Integrated intensity")
        self.ax_map.set_title(
            f"Band {self.band_lo.get():.0f}–{self.band_hi.get():.0f} cm⁻¹  "
            f"({self.method_var.get()})"
        )
        self.ax_map.set_xlabel("X"); self.ax_map.set_ylabel("Y")
        if self.selected_pixel is not None:
            xi, yi = self.selected_pixel
            self.ax_map.scatter([x[xi]], [y[yi]], s=80, facecolor="none",
                                edgecolor="red", linewidth=2)
        self.fig_map.tight_layout()
        self.canvas_map.draw()

    def _on_map_click(self, event):
        if self.mapping is None or event.inaxes != self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = self.mapping["x_coords"]; y = self.mapping["y_coords"]
        xi = int(np.argmin(np.abs(x - event.xdata)))
        yi = int(np.argmin(np.abs(y - event.ydata)))
        self.selected_pixel = (xi, yi)
        self._draw_map()
        self._draw_spectrum_at(xi, yi)

    def _draw_spectrum_at(self, xi: int, yi: int):
        spec = self.mapping["cube"][yi, xi, :]
        rs = self.mapping["raman_shifts"]
        self.ax_spec.clear()
        self.ax_spec.plot(rs, spec, color="black", lw=1)
        lo, hi = sorted([float(self.band_lo.get()), float(self.band_hi.get())])
        self.ax_spec.axvspan(lo, hi, color="orange", alpha=0.2, label="Band")
        self.ax_spec.set_title(
            f"Pixel (x={self.mapping['x_coords'][xi]:.2f}, "
            f"y={self.mapping['y_coords'][yi]:.2f})"
        )
        self.ax_spec.set_xlabel("Raman shift (cm⁻¹)")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.legend(fontsize=8); self.ax_spec.grid(True, alpha=0.3)
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    def _send_to_batch(self):
        if self.mapping is None:
            messagebox.showwarning("Send", "Load a mapping file first.")
            return
        if self.on_cube_to_batch is None:
            messagebox.showerror("Send", "No callback to main app registered.")
            return
        try:
            df = cube_to_batch_df(self.mapping["cube"], self.mapping["raman_shifts"])
            self.on_cube_to_batch(df)
            messagebox.showinfo(
                "Success",
                f"Pushed {df.shape[1] - 1} pixel spectra to batch pipeline.\n"
                "You can now run PCA / NMF / MCR-ALS / Clustering from the main window.",
            )
        except Exception as e:
            messagebox.showerror("Send Error", str(e))

    def _export_map(self):
        if self.current_map is None:
            messagebox.showwarning("Export", "Compute a map first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Map",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            df = pd.DataFrame(
                self.current_map,
                index=[f"y={v:.3f}" for v in self.mapping["y_coords"]],
                columns=[f"x={v:.3f}" for v in self.mapping["x_coords"]],
            )
            if path.endswith(".csv"):
                df.to_csv(path)
            else:
                df.to_excel(path)
            messagebox.showinfo("Success", f"Map exported to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
