"""Clustering window: HCA dendrogram, K-means (elbow + silhouette),
and optional UMAP 2D embedding.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd

from processing_logic import (
    perform_hca, perform_kmeans, perform_umap,
    prepare_cluster_matrix,
    CLUSTERING_LINKAGE, CLUSTERING_METRICS,
    _UMAP_AVAILABLE,
)


class ClusteringWindow(tk.Toplevel):
    def __init__(self, parent, processed_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Clustering Analysis")
        self.geometry("1050x700")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.data, self.sample_names = prepare_cluster_matrix(processed_df)

        self.hca_result = None
        self.kmeans_result = None
        self.umap_coords = None
        self.figures: list[plt.Figure] = []

        # UI vars
        self.hca_method = tk.StringVar(value="ward")
        self.hca_metric = tk.StringVar(value="euclidean")
        self.n_clusters = tk.IntVar(value=3)
        self.km_random_state = tk.IntVar(value=0)
        self.umap_neighbors = tk.IntVar(value=min(15, max(2, self.data.shape[0] - 1)))
        self.umap_min_dist = tk.DoubleVar(value=0.1)

        self._build_ui()
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

        self.hca_tab = ttk.Frame(self.notebook)
        self.km_tab = ttk.Frame(self.notebook)
        self.umap_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hca_tab, text="HCA Dendrogram")
        self.notebook.add(self.km_tab, text="K-means Sweep")
        self.notebook.add(self.umap_tab, text="UMAP 2D")

        self._init_empty(self.hca_tab, "Click 'Run HCA' to compute dendrogram")
        self._init_empty(self.km_tab, "Click 'Run K-means' to compute clusters + sweep")
        if _UMAP_AVAILABLE:
            self._init_empty(self.umap_tab, "Click 'Run UMAP' to compute 2D embedding")
        else:
            ttk.Label(
                self.umap_tab,
                text="umap-learn is not installed.\nRun: pip install umap-learn",
                foreground="red",
            ).pack(expand=True)

    def _build_controls(self, parent):
        info = ttk.LabelFrame(parent, text="Data")
        info.pack(fill="x", pady=5)
        ttk.Label(
            info,
            text=f"Samples: {self.data.shape[0]}\nFeatures: {self.data.shape[1]}",
            foreground="gray",
        ).pack(anchor="w", padx=5, pady=3)

        common = ttk.LabelFrame(parent, text="Common")
        common.pack(fill="x", pady=5)
        row = ttk.Frame(common); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="n_clusters:").pack(side="left")
        ttk.Entry(row, textvariable=self.n_clusters, width=5).pack(side="right")

        hca = ttk.LabelFrame(parent, text="HCA")
        hca.pack(fill="x", pady=5)
        row = ttk.Frame(hca); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Linkage:").pack(side="left")
        ttk.Combobox(row, textvariable=self.hca_method,
                     values=sorted(CLUSTERING_LINKAGE), state="readonly", width=12).pack(side="right")
        row = ttk.Frame(hca); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Metric:").pack(side="left")
        ttk.Combobox(row, textvariable=self.hca_metric,
                     values=sorted(CLUSTERING_METRICS), state="readonly", width=12).pack(side="right")
        ttk.Button(hca, text="Run HCA", command=self._run_hca).pack(fill="x", padx=5, pady=5)

        km = ttk.LabelFrame(parent, text="K-means")
        km.pack(fill="x", pady=5)
        row = ttk.Frame(km); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Random state:").pack(side="left")
        ttk.Entry(row, textvariable=self.km_random_state, width=5).pack(side="right")
        ttk.Button(km, text="Run K-means", command=self._run_kmeans).pack(fill="x", padx=5, pady=5)

        um = ttk.LabelFrame(parent, text="UMAP")
        um.pack(fill="x", pady=5)
        row = ttk.Frame(um); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="n_neighbors:").pack(side="left")
        ttk.Entry(row, textvariable=self.umap_neighbors, width=5).pack(side="right")
        row = ttk.Frame(um); row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="min_dist:").pack(side="left")
        ttk.Entry(row, textvariable=self.umap_min_dist, width=5).pack(side="right")
        ttk.Button(um, text="Run UMAP", command=self._run_umap).pack(fill="x", padx=5, pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)
        ttk.Button(parent, text="Export Labels to Excel...", command=self._export).pack(
            fill="x", padx=5, pady=5
        )

    def _init_empty(self, parent, msg):
        for w in parent.winfo_children():
            w.destroy()
        ttk.Label(parent, text=msg, foreground="gray").pack(expand=True)

    def _embed(self, parent, fig):
        for w in parent.winfo_children():
            w.destroy()
        self.figures.append(fig)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _run_hca(self):
        try:
            self.hca_result = perform_hca(
                self.data,
                method=self.hca_method.get(),
                metric=self.hca_metric.get(),
                n_clusters=self.n_clusters.get(),
            )
        except Exception as e:
            messagebox.showerror("HCA Error", str(e)); return
        fig, ax = plt.subplots(figsize=(9, 5))
        dendrogram(self.hca_result["linkage"], labels=self.sample_names, ax=ax,
                   leaf_rotation=60)
        ax.set_title(
            f"HCA dendrogram  (linkage={self.hca_result['method']}, "
            f"metric={self.hca_result['metric']}, k={self.hca_result['n_clusters']})"
        )
        ax.set_ylabel("Distance")
        fig.tight_layout()
        self._embed(self.hca_tab, fig)
        self.notebook.select(self.hca_tab)

    def _run_kmeans(self):
        try:
            self.kmeans_result = perform_kmeans(
                self.data,
                n_clusters=self.n_clusters.get(),
                random_state=self.km_random_state.get(),
            )
        except Exception as e:
            messagebox.showerror("K-means Error", str(e)); return
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ks = self.kmeans_result["sweep_ks"]
        if ks:
            axes[0].plot(ks, self.kmeans_result["sweep_inertias"], "o-")
            axes[0].set_title("Elbow (Inertia vs k)")
            axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
            axes[0].grid(True, alpha=0.4)

            sil = self.kmeans_result["sweep_silhouettes"]
            axes[1].plot(ks, sil, "o-", color="green")
            axes[1].set_title("Silhouette vs k")
            axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
            axes[1].axvline(self.kmeans_result["n_clusters"], color="red",
                            linestyle="--", label=f"selected k={self.kmeans_result['n_clusters']}")
            axes[1].legend(); axes[1].grid(True, alpha=0.4)
        fig.tight_layout()
        self._embed(self.km_tab, fig)
        self.notebook.select(self.km_tab)

    def _run_umap(self):
        if not _UMAP_AVAILABLE:
            messagebox.showerror("UMAP", "umap-learn not installed.")
            return
        try:
            self.umap_coords = perform_umap(
                self.data,
                n_neighbors=self.umap_neighbors.get(),
                min_dist=self.umap_min_dist.get(),
            )
        except Exception as e:
            messagebox.showerror("UMAP Error", str(e)); return

        fig, ax = plt.subplots(figsize=(7, 6))
        # Color by k-means labels if available
        if self.kmeans_result is not None:
            labels = self.kmeans_result["labels"]
            sc = ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1],
                            c=labels, cmap="tab10", s=60)
            plt.colorbar(sc, ax=ax, label="K-means cluster")
        else:
            ax.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], s=60)
        for i, name in enumerate(self.sample_names):
            ax.text(self.umap_coords[i, 0], self.umap_coords[i, 1], f" {name}", fontsize=8)
        ax.set_title("UMAP 2D Embedding")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._embed(self.umap_tab, fig)
        self.notebook.select(self.umap_tab)

    def _export(self):
        path = filedialog.asksaveasfilename(
            title="Save Clustering Labels",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            df = pd.DataFrame({"Sample": self.sample_names})
            if self.hca_result is not None:
                df[f"HCA_label_k{self.hca_result['n_clusters']}"] = self.hca_result["labels"]
            if self.kmeans_result is not None:
                df[f"KMeans_label_k{self.kmeans_result['n_clusters']}"] = self.kmeans_result["labels"]
            if self.umap_coords is not None:
                df["UMAP_1"] = self.umap_coords[:, 0]
                df["UMAP_2"] = self.umap_coords[:, 1]
            df.to_excel(path, index=False)
            messagebox.showinfo("Success", f"Labels exported to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
