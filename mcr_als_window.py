"""MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares) window.

Decomposes a Raman dataset D (n_samples x n_wavenumbers) into physically
interpretable pure-component spectra (ST) and concentration profiles (C)
with non-negativity constraints.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any
import numpy as np
import pandas as pd

from processing_logic import perform_mcr_als, _PYMCR_AVAILABLE


class MCRALSWindow(tk.Toplevel):
    def __init__(self, parent, processed_df: pd.DataFrame):
        super().__init__(parent)
        self.title("MCR-ALS Results")
        self.geometry("950x650")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.processed_df = processed_df
        self.results: Dict[str, Any] = {}
        self.figures: list[plt.Figure] = []

        self.n_components = tk.IntVar(value=3)
        self.nonneg_c = tk.BooleanVar(value=True)
        self.nonneg_s = tk.BooleanVar(value=True)
        self.norm_s = tk.BooleanVar(value=False)
        self.max_iter = tk.IntVar(value=200)
        self.init_method = tk.StringVar(value="nmf")
        self.random_state = tk.IntVar(value=0)

        self._build_ui()

        if not _PYMCR_AVAILABLE:
            messagebox.showerror(
                "pymcr missing",
                "pymcr is not installed.\nRun: pip install pymcr",
                parent=self,
            )
        self.grab_set()

    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.Frame(pane, width=280); ctrl.pack_propagate(False)
        pane.add(ctrl, weight=0)
        self._build_controls(ctrl)

        right = ttk.Frame(pane); pane.add(right, weight=1)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self.spec_tab = ttk.Frame(self.notebook)
        self.conc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spec_tab, text="Pure Spectra (ST)")
        self.notebook.add(self.conc_tab, text="Concentrations (C)")

        ttk.Label(self.spec_tab, text="Run MCR-ALS to populate.", foreground="gray").pack(expand=True)
        ttk.Label(self.conc_tab, text="Run MCR-ALS to populate.", foreground="gray").pack(expand=True)

        self.info_label = ttk.Label(self, text="", foreground="gray")
        self.info_label.pack(side="bottom", fill="x", padx=8, pady=4)

    def _build_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Parameters")
        f.pack(fill="x", pady=5)
        row = ttk.Frame(f); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Components:").pack(side="left")
        ttk.Entry(row, textvariable=self.n_components, width=5).pack(side="right")
        row = ttk.Frame(f); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Init:").pack(side="left")
        ttk.Combobox(row, textvariable=self.init_method,
                     values=["nmf", "random"], state="readonly", width=10).pack(side="right")
        row = ttk.Frame(f); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Random state:").pack(side="left")
        ttk.Entry(row, textvariable=self.random_state, width=5).pack(side="right")
        row = ttk.Frame(f); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Max iter:").pack(side="left")
        ttk.Entry(row, textvariable=self.max_iter, width=5).pack(side="right")

        c = ttk.LabelFrame(parent, text="Constraints")
        c.pack(fill="x", pady=5)
        ttk.Checkbutton(c, text="Non-neg concentrations (C >= 0)",
                        variable=self.nonneg_c).pack(anchor="w", padx=5)
        ttk.Checkbutton(c, text="Non-neg spectra (S >= 0)",
                        variable=self.nonneg_s).pack(anchor="w", padx=5)
        ttk.Checkbutton(c, text="Unit-norm spectra",
                        variable=self.norm_s).pack(anchor="w", padx=5)

        ttk.Button(parent, text="Run MCR-ALS", command=self._run).pack(
            fill="x", padx=5, pady=8
        )
        ttk.Button(parent, text="Export to Excel...", command=self._export).pack(
            fill="x", padx=5, pady=5
        )

    def _run(self):
        if not _PYMCR_AVAILABLE:
            messagebox.showerror("MCR-ALS", "pymcr not installed.")
            return
        try:
            self.results = perform_mcr_als(
                self.processed_df,
                n_components=self.n_components.get(),
                nonneg_concentrations=self.nonneg_c.get(),
                nonneg_spectra=self.nonneg_s.get(),
                norm_spectra=self.norm_s.get(),
                max_iter=self.max_iter.get(),
                init=self.init_method.get(),
                random_state=self.random_state.get(),
            )
        except Exception as e:
            messagebox.showerror("MCR-ALS Error", str(e))
            return

        self._draw_spectra()
        self._draw_concentrations()
        self.info_label.config(
            text=(
                f"n_components={self.results['n_components']}  |  "
                f"n_iter={self.results['n_iter']}  |  "
                f"LOF={self.results['lof_percent']:.3f}%  |  "
                f"init={self.results['init']}"
            )
        )

    def _draw_spectra(self):
        for w in self.spec_tab.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        self.figures.append(fig)
        spectra = self.results["spectra"]  # (n_wn, k)
        x = self.results["raman_shifts"]
        for i in range(spectra.shape[1]):
            ax.plot(x, spectra[:, i], label=f"Component {i + 1}")
        ax.set_title("MCR-ALS Pure Component Spectra")
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensity (AU)")
        ax.legend(); ax.grid(True, alpha=0.4)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.spec_tab).get_tk_widget().pack(fill="both", expand=True)

    def _draw_concentrations(self):
        for w in self.conc_tab.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        self.figures.append(fig)
        C = self.results["concentrations"]  # (n_samples, k)
        names = self.results["sample_names"]
        n_samples, k = C.shape
        indices = np.arange(n_samples)
        w = 0.8 / k
        for i in range(k):
            ax.bar(indices + i * w, C[:, i], w, label=f"Component {i + 1}")
        ax.set_title("MCR-ALS Concentration Profiles")
        ax.set_xlabel("Sample"); ax.set_ylabel("Concentration")
        ax.set_xticks(indices + w * (k - 1) / 2)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend(); ax.grid(axis="y")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.conc_tab).get_tk_widget().pack(fill="both", expand=True)

    def _export(self):
        if not self.results:
            messagebox.showwarning("Export", "Run MCR-ALS first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save MCR-ALS Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            C = self.results["concentrations"]
            ST = self.results["spectra"]
            names = self.results["sample_names"]
            shifts = self.results["raman_shifts"]
            k = C.shape[1]

            conc_df = pd.DataFrame(C, columns=[f"Component_{i + 1}" for i in range(k)])
            conc_df.insert(0, "Sample_Name", names)

            spec_df = pd.DataFrame(ST, columns=[f"Component_{i + 1}" for i in range(k)])
            spec_df.insert(0, "Raman Shift (cm-1)", shifts)

            info_df = pd.DataFrame(
                [
                    ("n_components", self.results["n_components"]),
                    ("n_iter", self.results["n_iter"]),
                    ("LOF_percent", self.results["lof_percent"]),
                    ("init", self.results["init"]),
                    ("nonneg_concentrations", self.results["nonneg_concentrations"]),
                    ("nonneg_spectra", self.results["nonneg_spectra"]),
                ],
                columns=["Category", "Value"],
            )

            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                info_df.to_excel(writer, sheet_name="MCR_Info", index=False)
                conc_df.to_excel(writer, sheet_name="Concentrations", index=False)
                spec_df.to_excel(writer, sheet_name="Pure_Spectra", index=False)

            messagebox.showinfo("Success", f"MCR-ALS results exported to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
