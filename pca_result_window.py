import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any
import numpy as np
import pandas as pd
import datetime


class PCAResultWindow(tk.Toplevel):
    """PCA analysis results window with scores, loadings, variance,
    Scree, and Hotelling T² / Q-residuals diagnostics."""

    def __init__(self, parent, pca_results: Dict[str, Any], analysis_params: Dict[str, Any]):
        super().__init__(parent)
        self.title("PCA Results")
        self.geometry("1000x680")
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
        scree_tab = ttk.Frame(notebook)
        diag_tab = ttk.Frame(notebook)
        notebook.add(scores_tab, text="Scores")
        notebook.add(loadings_tab, text="Loadings")
        notebook.add(variance_tab, text="Explained Variance")
        notebook.add(scree_tab, text="Scree / Eigenvalues")
        notebook.add(diag_tab, text="T² / Q Diagnostics")

        self._create_scores_plot(scores_tab)
        self._create_loadings_plot(loadings_tab)
        self._create_variance_plot(variance_tab)
        self._create_scree_plot(scree_tab)
        self._create_diagnostics_plot(diag_tab)

        # Info banner
        info_parts = [
            f"Scaling: {self.results.get('scaling_method', 'auto')}",
            f"PCs: {self.results.get('n_components_selected', '?')}",
            f"Total Var: {self.results.get('total_explained_variance', 0) * 100:.2f}%",
            f"Kaiser: {self.results.get('kaiser_components', '?')}",
        ]
        t2_lim = self.results.get("t2_limit", float('nan'))
        q_lim = self.results.get("q_limit", float('nan'))
        if np.isfinite(t2_lim):
            info_parts.append(f"T²_lim: {t2_lim:.3g}")
        if np.isfinite(q_lim):
            info_parts.append(f"Q_lim: {q_lim:.3g}")
        ttk.Label(
            main_frame, text="  |  ".join(info_parts), foreground="gray"
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

    def _create_scores_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        scores = self.results["scores"]
        sample_names = self.results["sample_names"]
        variance = self.results["explained_variance_ratio"]

        if scores.shape[1] < 2:
            ax.text(0.5, 0.5, "Need ≥ 2 PCs for scores plot",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            pc1, pc2 = scores[:, 0], scores[:, 1]
            ax.scatter(pc1, pc2)
            for i, name in enumerate(sample_names):
                ax.text(pc1[i], pc2[i], f" {name}", fontsize=9)
            ax.set_title("PCA Scores Plot (PC1 vs PC2)")
            ax.set_xlabel(f"PC1 ({variance[0] * 100:.2f}%)")
            ax.set_ylabel(f"PC2 ({variance[1] * 100:.2f}%)")
            ax.grid(True)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_loadings_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        loadings = self.results["loadings"]
        raman_shifts = self.results["raman_shifts"]

        for i in range(min(3, loadings.shape[1])):
            ax.plot(raman_shifts, loadings[:, i], label=f"PC{i + 1}")
        ax.set_title("PCA Loadings")
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

        ax.bar(pc_indices, variance * 100, alpha=0.8, label="Explained")
        ax.plot(pc_indices, np.cumsum(variance * 100), "r-o", label="Cumulative")
        ax.set_title("Explained Variance by Principal Component")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_xticks(pc_indices)
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_scree_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        eig = self.results["eigenvalues"]
        idx = np.arange(1, len(eig) + 1)
        ax.plot(idx, eig, "b-o", label="Eigenvalue")
        ax.axhline(1.0, color="red", linestyle="--", label="Kaiser (λ=1)")
        ax.set_title("Scree Plot (eigenvalues)")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Eigenvalue (λ)")
        ax.set_xticks(idx)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.4)
        fig.tight_layout()
        self._embed_figure(parent_tab, fig)

    def _create_diagnostics_plot(self, parent_tab):
        fig, ax = plt.subplots(figsize=(7, 5))
        t2 = self.results.get("hotelling_t2")
        q = self.results.get("q_residuals")
        names = self.results["sample_names"]
        t2_lim = self.results.get("t2_limit", float("nan"))
        q_lim = self.results.get("q_limit", float("nan"))

        if t2 is None or q is None:
            ax.text(0.5, 0.5, "Diagnostics unavailable", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(t2, q)
            for i, name in enumerate(names):
                ax.text(t2[i], q[i], f" {name}", fontsize=8)
            if np.isfinite(t2_lim):
                ax.axvline(t2_lim, color="red", linestyle="--",
                           label=f"T² limit ({self.results.get('confidence', 0.95)*100:.0f}%)")
            if np.isfinite(q_lim):
                ax.axhline(q_lim, color="orange", linestyle="--",
                           label=f"Q limit ({self.results.get('confidence', 0.95)*100:.0f}%)")
            ax.set_title("Hotelling T² vs Q-residuals (Outlier Diagnostic)")
            ax.set_xlabel("Hotelling T²")
            ax.set_ylabel("Q-residual (SPE)")
            ax.legend()
            ax.grid(True, alpha=0.4)
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
            baseline_p = self.analysis_params["baseline"]["params"]
            algo = self.analysis_params["baseline"]["algorithm"]
            if algo == "mor":
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
                ("Software", "Raman Spectroscopy Processor v2.2"),
                ("", ""),
                ("Analysis Parameters", ""),
                ("X-Axis Range", f"{self.analysis_params['range']['lower_bound']} - {self.analysis_params['range']['upper_bound']}"),
                ("Cosmic Ray Removal", self.analysis_params["preprocessing"]["apply_cosmic_ray"]),
                ("Smoothing (SG)", f"Poly Order: {self.analysis_params['smoothing']['sg_poly_order']}, Window: {self.analysis_params['smoothing']['sg_frame_window']}"),
                ("Baseline Correction", baseline_str),
                ("Normalization", self.analysis_params.get("normalization", "none")),
                ("", ""),
                ("PCA Settings", ""),
                ("Scaling Method", self.results.get("scaling_method", "auto")),
                ("Principal Components", self.results["n_components_selected"]),
                ("Confidence Level", f"{self.results.get('confidence', 0.95) * 100:.1f}%"),
                ("", ""),
                ("Performance Metrics", ""),
                ("Total Explained Variance", f"{self.results['total_explained_variance'] * 100:.2f}%"),
                ("Kaiser Criterion (λ>1)", f"{self.results['kaiser_components']} components"),
                ("T² Limit", f"{self.results.get('t2_limit', float('nan')):.4g}"),
                ("Q Limit", f"{self.results.get('q_limit', float('nan')):.4g}"),
            ]
            info_df = pd.DataFrame(info_data[1:], columns=info_data[0])

            scores = self.results["scores"]
            sample_names = self.results["sample_names"]
            pc_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
            scores_df = pd.DataFrame(scores, columns=pc_cols)
            scores_df.insert(0, "Sample_Name", sample_names)
            # Append diagnostics
            t2 = self.results.get("hotelling_t2")
            q = self.results.get("q_residuals")
            if t2 is not None:
                scores_df["Hotelling_T2"] = t2
            if q is not None:
                scores_df["Q_residual"] = q
            if t2 is not None and np.isfinite(self.results.get("t2_limit", float("nan"))):
                scores_df["T2_exceeds_limit"] = t2 > self.results["t2_limit"]
            if q is not None and np.isfinite(self.results.get("q_limit", float("nan"))):
                scores_df["Q_exceeds_limit"] = q > self.results["q_limit"]

            loadings = self.results["loadings"]
            raman_shifts = self.results["raman_shifts"]
            loading_cols = [f"PC{i + 1}_Loading" for i in range(loadings.shape[1])]
            loadings_df = pd.DataFrame(loadings, columns=loading_cols)
            loadings_df.insert(0, "Raman Shift (cm-1)", raman_shifts)

            variance = self.results["explained_variance_ratio"]
            eigenvalues = self.results["eigenvalues"]
            variance_df = pd.DataFrame({
                "Principal_Component": [f"PC{i + 1}" for i in range(len(variance))],
                "Eigenvalue": eigenvalues,
                "Explained_Variance_Ratio": variance,
                "Cumulative_Variance": np.cumsum(variance),
            })

            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                info_df.to_excel(writer, sheet_name="Analysis_Info", index=False)
                scores_df.to_excel(writer, sheet_name="Scores_Diagnostics", index=False)
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
