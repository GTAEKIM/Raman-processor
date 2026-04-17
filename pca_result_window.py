import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any
import numpy as np
import pandas as pd
import datetime


class PCAResultWindow(tk.Toplevel):
    """PCA analysis results window with tabbed plots and Excel export."""

    def __init__(self, parent, pca_results: Dict[str, Any], analysis_params: Dict[str, Any]):
        super().__init__(parent)
        self.title("PCA Results")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.results = pca_results
        self.analysis_params = analysis_params
        self.figures: list[plt.Figure] = []

        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True)

        scores_tab = ttk.Frame(notebook)
        loadings_tab = ttk.Frame(notebook)
        variance_tab = ttk.Frame(notebook)
        notebook.add(scores_tab, text="Scores Plot")
        notebook.add(loadings_tab, text="Loadings Plot")
        notebook.add(variance_tab, text="Explained Variance")

        self._create_scores_plot(scores_tab)
        self._create_loadings_plot(loadings_tab)
        self._create_variance_plot(variance_tab)

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

    def _create_scores_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        scores = self.results["scores"]
        sample_names = self.results["sample_names"]
        variance = self.results["explained_variance_ratio"]

        pc1, pc2 = scores[:, 0], scores[:, 1]
        ax.scatter(pc1, pc2)
        for i, name in enumerate(sample_names):
            ax.text(pc1[i], pc2[i], f' {name}', fontsize=9)

        ax.set_title("PCA Scores Plot (PC1 vs PC2)")
        ax.set_xlabel(f"PC1 ({variance[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({variance[1] * 100:.2f}%)")
        ax.grid(True)
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_loadings_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        loadings = self.results["loadings"]
        raman_shifts = self.results["raman_shifts"]

        ax.plot(raman_shifts, loadings[:, 0], label="PC1 Loadings")
        ax.plot(raman_shifts, loadings[:, 1], label="PC2 Loadings")
        ax.set_title("PCA Loadings Plot")
        ax.set_xlabel("Raman shift (cm\u207b\u00b9)")
        ax.set_ylabel("Loading Value")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_variance_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        variance = self.results["explained_variance_ratio"]
        pc_indices = np.arange(1, len(variance) + 1)

        ax.bar(pc_indices, variance * 100, alpha=0.8)
        ax.plot(pc_indices, np.cumsum(variance * 100), 'r-o', label="Cumulative Variance")
        ax.set_title("Explained Variance by Principal Component")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_xticks(pc_indices)
        ax.legend()
        ax.grid(axis='y')
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _export_results(self):
        filepath = filedialog.asksaveasfilename(
            title="Save PCA Results As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not filepath:
            return

        try:
            # Baseline parameter string
            baseline_p = self.analysis_params['baseline']['params']
            algo = self.analysis_params['baseline']['algorithm']
            if algo == 'mor':
                baseline_str = f"Algorithm: mor, Half Window: {baseline_p.get('half_window')}"
            else:
                baseline_str = (
                    f"Algorithm: {algo}, "
                    f"Order: {baseline_p.get('order')}, "
                    f"Threshold: {baseline_p.get('threshold')}"
                )

            info_data = [
                ("Category", "Detail"),
                ("Analysis Information", ""),
                ("Analysis Date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ("Software", "Raman Spectroscopy Processor v2.0"),
                ("", ""),
                ("Analysis Parameters", ""),
                ("X-Axis Range", f"{self.analysis_params['range']['lower_bound']} - {self.analysis_params['range']['upper_bound']}"),
                ("Cosmic Ray Removal", self.analysis_params['preprocessing']['apply_cosmic_ray']),
                ("Smoothing (SG)", f"Poly Order: {self.analysis_params['smoothing']['sg_poly_order']}, Window: {self.analysis_params['smoothing']['sg_frame_window']}"),
                ("Baseline Correction", baseline_str),
                ("", ""),
                ("Data Information", ""),
                ("Original Dimensions", f"{self.results['n_samples']} samples, {self.results['n_variables']} variables"),
                ("Principal Components", self.results['n_components_selected']),
                ("Preprocessing", "Standard Scaling (Centering & Unit Variance)"),
                ("", ""),
                ("Performance Metrics", ""),
                ("Total Explained Variance", f"{self.results['total_explained_variance'] * 100:.2f}%"),
                ("Kaiser Criterion (Eigenvalue > 1)", f"{self.results['kaiser_components']} components suggested"),
            ]
            info_df = pd.DataFrame(info_data[1:], columns=info_data[0])

            scores = self.results["scores"]
            sample_names = self.results["sample_names"]
            pc_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
            scores_df = pd.DataFrame(scores, columns=pc_cols)
            scores_df.insert(0, "Sample_Name", sample_names)

            loadings = self.results["loadings"]
            raman_shifts = self.results["raman_shifts"]
            loading_cols = [f"PC{i + 1}_Loading" for i in range(loadings.shape[1])]
            loadings_df = pd.DataFrame(loadings, columns=loading_cols)
            loadings_df.insert(0, "Raman Shift (cm-1)", raman_shifts)

            variance = self.results["explained_variance_ratio"]
            eigenvalues = self.results["eigenvalues"]
            pc_indices = [f"PC{i + 1}" for i in range(len(variance))]
            variance_df = pd.DataFrame({
                "Principal_Component": pc_indices,
                "Eigenvalue": eigenvalues,
                "Explained_Variance_Ratio": variance,
                "Cumulative_Variance": np.cumsum(variance),
            })

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                info_df.to_excel(writer, sheet_name="Analysis_Info", index=False)
                scores_df.to_excel(writer, sheet_name="Scores", index=False)
                loadings_df.to_excel(writer, sheet_name="Loadings", index=False)
                variance_df.to_excel(writer, sheet_name="Explained Variance", index=False)

            messagebox.showinfo("Success", f"PCA results successfully exported to\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PCA results. Error: {e}")

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
