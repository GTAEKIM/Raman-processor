import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any
import numpy as np
import pandas as pd


class NMFResultWindow(tk.Toplevel):
    """NMF analysis results window with tabbed plots and Excel export."""

    def __init__(self, parent, nmf_results: Dict[str, Any]):
        super().__init__(parent)
        self.title("NMF Results")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.results = nmf_results
        self.figures: list[plt.Figure] = []

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True)

        components_tab = ttk.Frame(notebook)
        weights_tab = ttk.Frame(notebook)
        notebook.add(components_tab, text="Components Plot")
        notebook.add(weights_tab, text="Weights Plot")

        self._create_components_plot(components_tab)
        self._create_weights_plot(weights_tab)

        # Info banner showing init / iterations / reconstruction error
        info_parts = [f"Components: {self.results.get('n_components', '?')}"]
        if 'init' in self.results:
            info_parts.append(f"Init: {self.results['init']}")
        if 'n_iter' in self.results:
            info_parts.append(f"Iters: {self.results['n_iter']}")
        if 'reconstruction_error' in self.results:
            info_parts.append(f"Recon Err: {self.results['reconstruction_error']:.4f}")
        ttk.Label(
            main_frame,
            text="  |  ".join(info_parts),
            foreground="gray",
        ).pack(side="bottom", fill="x", pady=(5, 0))

        self._create_export_button(main_frame)
        self.grab_set()

    def _embed_figure(self, parent_tab: ttk.Frame, fig: plt.Figure):
        self.figures.append(fig)
        canvas = FigureCanvasTkAgg(fig, master=parent_tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_export_button(self, parent_frame):
        export_frame = ttk.Frame(parent_frame, padding=(0, 10, 0, 0))
        export_frame.pack(side="bottom", fill="x")
        ttk.Button(
            export_frame, text="Export Results to Excel...", command=self._export_results
        ).pack(anchor="e")

    def _create_components_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        components = self.results["components"]
        raman_shifts = self.results["raman_shifts"]

        for i in range(components.shape[1]):
            ax.plot(raman_shifts, components[:, i], label=f"Component {i + 1}")

        ax.set_title("NMF Components (Spectral Signatures)")
        ax.set_xlabel("Raman shift (cm\u207b\u00b9)")
        ax.set_ylabel("Intensity (AU)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_weights_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        weights = self.results["weights"]
        sample_names = self.results["sample_names"]
        n_samples = len(sample_names)
        n_components = weights.shape[1]

        bar_width = 0.8 / n_components
        indices = np.arange(n_samples)

        for i in range(n_components):
            ax.bar(
                indices + i * bar_width,
                weights[:, i],
                bar_width,
                label=f"Component {i + 1}",
            )

        ax.set_title("NMF Component Weights per Sample")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Relative Contribution (Weight)")
        ax.set_xticks(indices + bar_width * (n_components - 1) / 2)
        ax.set_xticklabels(sample_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y')
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _export_results(self):
        filepath = filedialog.asksaveasfilename(
            title="Save NMF Results As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not filepath:
            return

        try:
            weights = self.results["weights"]
            sample_names = self.results["sample_names"]
            comp_cols_w = [f"Component_{i + 1}_Weight" for i in range(weights.shape[1])]
            weights_df = pd.DataFrame(weights, columns=comp_cols_w)
            weights_df.insert(0, "Sample_Name", sample_names)

            components = self.results["components"]
            raman_shifts = self.results["raman_shifts"]
            comp_cols_c = [f"Component_{i + 1}" for i in range(components.shape[1])]
            components_df = pd.DataFrame(components, columns=comp_cols_c)
            components_df.insert(0, "Raman Shift (cm-1)", raman_shifts)

            # Metadata sheet
            info_rows = [
                ("Category", "Value"),
                ("n_components", self.results.get('n_components', '')),
                ("init", self.results.get('init', '')),
                ("random_state", self.results.get('random_state', '')),
                ("n_iter", self.results.get('n_iter', '')),
                ("reconstruction_error", self.results.get('reconstruction_error', '')),
            ]
            info_df = pd.DataFrame(info_rows[1:], columns=info_rows[0])

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                info_df.to_excel(writer, sheet_name="Analysis_Info", index=False)
                weights_df.to_excel(writer, sheet_name="Weights (Scores)", index=False)
                components_df.to_excel(writer, sheet_name="Components (Loadings)", index=False)

            messagebox.showinfo("Success", f"NMF results successfully exported to\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export NMF results. Error: {e}")

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
