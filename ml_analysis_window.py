"""Machine-learning / chemometrics analysis window.

Supervised:  PLS-DA (classification), PLS regression (quantification),
             Random Forest / SVM / kNN classifiers (with cross-validation).
Unsupervised: t-SNE 2-D embedding, spectral-library matching (cosine / SAM).

Labels/targets come from an editable per-sample table (or an imported
"sample,label" CSV). Classification uses categorical labels; regression parses
them as numbers. t-SNE and spectral matching do not require labels.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing_logic import (
    ml_matrix, perform_plsda, perform_plsr, perform_classifier,
    perform_tsne, spectral_match,
)
from ui_helpers import scrolled

_MODELS = {
    "PLS-DA (classification)": "plsda",
    "PLS regression (quantify)": "plsr",
    "Random Forest": "rf",
    "SVM": "svm",
    "k-Nearest Neighbours": "knn",
    "t-SNE (unsupervised)": "tsne",
    "Spectral matching": "match",
}
_SUPERVISED = {"plsda", "plsr", "rf", "svm", "knn"}


class MLAnalysisWindow(tk.Toplevel):
    def __init__(self, parent, processed_df: pd.DataFrame):
        super().__init__(parent)
        self.title("ML / Chemometrics Analysis")
        self.geometry("1240x760")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.processed_df = processed_df
        self.X, self.sample_names, self.shifts = ml_matrix(processed_df)
        self.labels: Dict[str, str] = {n: "" for n in self.sample_names}
        self.result: Optional[Dict[str, Any]] = None
        self.figures: List[plt.Figure] = []

        self.model_name = tk.StringVar(value=list(_MODELS.keys())[0])
        self.n_components = tk.IntVar(value=2)
        self.cv_folds = tk.IntVar(value=5)
        self.perplexity = tk.DoubleVar(value=15.0)
        self.match_metric = tk.StringVar(value="cosine")
        self.query_sample = tk.StringVar(value=self.sample_names[0] if self.sample_names else "")

        self._build_ui()
        self._refresh_label_tree()
        self.grab_set()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(pane, width=360); left.pack_propagate(False)
        pane.add(left, weight=0)
        self._build_left(left)

        right = ttk.Frame(pane); pane.add(right, weight=3)
        self._build_right(right)

    def _build_left(self, parent):
        mdl = ttk.LabelFrame(parent, text="Model")
        mdl.pack(fill="x", padx=4, pady=4)
        cb = ttk.Combobox(mdl, textvariable=self.model_name,
                          values=list(_MODELS.keys()), state="readonly")
        cb.pack(fill="x", padx=5, pady=5)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())

        self.param_frame = ttk.LabelFrame(parent, text="Parameters")
        self.param_frame.pack(fill="x", padx=4, pady=4)
        self._build_params()

        lab = ttk.LabelFrame(parent, text="Labels / targets")
        lab.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Label(lab, text="Double-click a row to set its label/target.",
                  foreground="gray").pack(anchor="w", padx=4)
        self.label_tree = scrolled(
            lab, lambda c: ttk.Treeview(c, columns=("sample", "label"),
                                        show="headings", height=10))
        self.label_tree.heading("sample", text="sample")
        self.label_tree.heading("label", text="label / target")
        self.label_tree.column("sample", width=170, anchor="w", stretch=False)
        self.label_tree.column("label", width=120, anchor="w", stretch=False)
        self.label_tree.bind("<Double-1>", lambda e: self._edit_label())
        lrow = ttk.Frame(lab); lrow.pack(fill="x", padx=4, pady=2)
        ttk.Button(lrow, text="Import labels CSV...",
                   command=self._import_labels).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(lrow, text="Clear",
                   command=self._clear_labels).pack(side="left", expand=True, fill="x", padx=1)

        ttk.Button(parent, text="Run", style="Accent.TButton",
                   command=self._run).pack(fill="x", padx=6, pady=6)

    def _build_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        code = _MODELS[self.model_name.get()]
        if code in ("plsda", "plsr"):
            self._param_row("PLS components:", self.n_components)
            self._param_row("CV folds:", self.cv_folds)
        elif code in ("rf", "svm", "knn"):
            self._param_row("CV folds:", self.cv_folds)
        elif code == "tsne":
            self._param_row("Perplexity:", self.perplexity)
        elif code == "match":
            row = ttk.Frame(self.param_frame); row.pack(fill="x", padx=5, pady=3)
            ttk.Label(row, text="Query sample:").pack(side="left")
            ttk.Combobox(row, textvariable=self.query_sample,
                         values=self.sample_names, state="readonly",
                         width=16).pack(side="right")
            row2 = ttk.Frame(self.param_frame); row2.pack(fill="x", padx=5, pady=3)
            ttk.Label(row2, text="Metric:").pack(side="left")
            ttk.Combobox(row2, textvariable=self.match_metric,
                         values=["cosine", "sam"], state="readonly",
                         width=10).pack(side="right")

    def _param_row(self, label, var):
        row = ttk.Frame(self.param_frame); row.pack(fill="x", padx=5, pady=3)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=var, width=8).pack(side="right")

    def _on_model_change(self):
        self._build_params()

    def _build_right(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        mt = ttk.Frame(self.nb); self.nb.add(mt, text="Metrics")
        self.metrics_text = scrolled(
            mt, lambda c: tk.Text(c, wrap="none", height=10,
                                  font=("Consolas", 10)),
            vertical=True, horizontal=True)

        pl = ttk.Frame(self.nb); self.nb.add(pl, text="Plot")
        self.fig_main, self.ax_main = plt.subplots(figsize=(6, 5))
        self.figures.append(self.fig_main)
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=pl)
        self.canvas_main.get_tk_widget().pack(fill="both", expand=True)

        im = ttk.Frame(self.nb); self.nb.add(im, text="Importance / VIP")
        self.fig_imp, self.ax_imp = plt.subplots(figsize=(6, 4))
        self.figures.append(self.fig_imp)
        self.canvas_imp = FigureCanvasTkAgg(self.fig_imp, master=im)
        self.canvas_imp.get_tk_widget().pack(fill="both", expand=True)

        ttk.Button(parent, text="Export results to Excel...",
                   command=self._export).pack(fill="x", padx=4, pady=4)

    # ── Labels ──────────────────────────────────────────────────────────
    def _refresh_label_tree(self):
        self.label_tree.delete(*self.label_tree.get_children())
        for n in self.sample_names:
            self.label_tree.insert("", "end", values=(n, self.labels.get(n, "")))

    def _edit_label(self):
        sel = self.label_tree.selection()
        if not sel:
            return
        name = self.label_tree.item(sel[0], "values")[0]
        dlg = tk.Toplevel(self); dlg.title("Label"); dlg.grab_set(); dlg.transient(self)
        ttk.Label(dlg, text=f"Label / target for\n{name}").pack(padx=8, pady=6)
        v = tk.StringVar(value=self.labels.get(name, ""))
        e = ttk.Entry(dlg, textvariable=v, width=24); e.pack(padx=8, pady=4); e.focus_set()

        def save(*_):
            self.labels[name] = v.get().strip()
            self._refresh_label_tree()
            dlg.destroy()
        e.bind("<Return>", save)
        ttk.Button(dlg, text="Save", command=save).pack(pady=6)

    def _import_labels(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path, header=None)
            mapping = {str(r[0]): str(r[1]) for _, r in df.iterrows()}
            n = 0
            for name in self.sample_names:
                if name in mapping:
                    self.labels[name] = mapping[name]; n += 1
            self._refresh_label_tree()
            messagebox.showinfo("Import", f"Matched labels for {n}/{len(self.sample_names)} samples.")
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def _clear_labels(self):
        self.labels = {n: "" for n in self.sample_names}
        self._refresh_label_tree()

    def _labelled(self):
        """Return (X_subset, names_subset, labels_subset) for samples with labels."""
        idx = [i for i, n in enumerate(self.sample_names) if self.labels.get(n, "").strip()]
        if not idx:
            return None, None, None
        X = self.X[idx, :]
        names = [self.sample_names[i] for i in idx]
        lab = [self.labels[self.sample_names[i]].strip() for i in idx]
        return X, names, lab

    # ── Run ─────────────────────────────────────────────────────────────
    def _run(self):
        code = _MODELS[self.model_name.get()]
        try:
            if code == "tsne":
                self._run_tsne()
            elif code == "match":
                self._run_match()
            elif code == "plsr":
                self._run_plsr()
            elif code == "plsda":
                self._run_plsda()
            else:
                self._run_classifier(code)
        except Exception as e:
            messagebox.showerror("Run Error", str(e))

    def _need_labels(self):
        X, names, lab = self._labelled()
        if X is None:
            messagebox.showwarning(
                "Labels", "Assign labels/targets to at least some samples first "
                "(double-click rows or import a CSV).")
        return X, names, lab

    def _run_plsda(self):
        X, names, lab = self._need_labels()
        if X is None:
            return
        r = perform_plsda(X, lab, n_components=self.n_components.get(),
                          cv_folds=self.cv_folds.get())
        self.result = {**r, "sample_names": names, "labels": lab}
        self._show_metrics(
            f"PLS-DA  ({r['n_components']} comps, {r['cv_folds']}-fold CV)\n"
            f"Classes: {list(r['classes'])}\n"
            f"Train accuracy: {r['train_accuracy']:.3f}\n"
            f"CV accuracy:    {r['cv_accuracy']:.3f}\n\n"
            + self._confusion_text(r['confusion'], r['classes']))
        self._plot_scores(r['scores'], lab, "PLS-DA scores (LV1 vs LV2)")
        self._plot_vector(r['vip'], "VIP", "VIP score")

    def _run_plsr(self):
        X, names, lab = self._need_labels()
        if X is None:
            return
        try:
            y = [float(v) for v in lab]
        except ValueError:
            messagebox.showerror("PLSR", "Regression needs numeric targets.")
            return
        r = perform_plsr(X, y, n_components=self.n_components.get(),
                         cv_folds=self.cv_folds.get())
        self.result = {**r, "sample_names": names}
        self._show_metrics(
            f"PLS regression  ({r['n_components']} comps, {r['cv_folds']}-fold CV)\n"
            f"R² train: {r['r2_train']:.3f}\n"
            f"R² CV:    {r['r2_cv']:.3f}\n"
            f"RMSECV:   {r['rmsecv']:.4g}")
        self._plot_pred_vs_actual(r['y'], r['pred_train'], r['pred_cv'])
        self._plot_vector(r['vip'], "VIP", "VIP score")

    def _run_classifier(self, code):
        X, names, lab = self._need_labels()
        if X is None:
            return
        r = perform_classifier(X, lab, model=code, cv_folds=self.cv_folds.get())
        self.result = {**r, "sample_names": names, "labels": lab}
        self._show_metrics(
            f"{r['method']} classifier  ({r['cv_folds']}-fold CV)\n"
            f"Classes: {list(r['classes'])}\n"
            f"CV accuracy: {r['cv_accuracy']:.3f}\n\n"
            + self._confusion_text(r['confusion'], r['classes']))
        self._plot_confusion(r['confusion'], r['classes'])
        if r['importance'] is not None:
            self._plot_vector(r['importance'], "Feature importance", "importance")
        else:
            self._clear_importance("Feature importance not available for this model.")

    def _run_tsne(self):
        r = perform_tsne(self.X, perplexity=self.perplexity.get())
        self.result = {"method": "t-SNE", "embedding": r["embedding"],
                       "sample_names": self.sample_names}
        _, _, lab = self._labelled()
        lab_full = [self.labels.get(n, "") for n in self.sample_names]
        self._show_metrics(f"t-SNE 2-D embedding\nperplexity = {r['perplexity']:.1f}\n"
                           f"{self.X.shape[0]} samples")
        self._plot_scores(r["embedding"], lab_full, "t-SNE embedding")
        self._clear_importance("t-SNE has no per-feature importance.")

    def _run_match(self):
        q = self.query_sample.get()
        if q not in self.processed_df.columns:
            messagebox.showwarning("Match", "Pick a query sample.")
            return
        library = {n: self.processed_df[n].values for n in self.sample_names if n != q}
        ranked = spectral_match(self.processed_df[q].values, library,
                                metric=self.match_metric.get())
        self.result = {"method": "Spectral match", "query": q,
                       "ranked": ranked}
        lines = [f"Spectral matching vs '{q}'  (metric={self.match_metric.get()})", ""]
        for name, score in ranked[:15]:
            lines.append(f"  {score:.4f}   {name}")
        self._show_metrics("\n".join(lines))
        names = [n for n, _ in ranked[:15]]
        scores = [s for _, s in ranked[:15]]
        self.ax_main.clear()
        self.ax_main.barh(range(len(names)), scores, color="#3b7dd8")
        self.ax_main.set_yticks(range(len(names)))
        self.ax_main.set_yticklabels(names, fontsize=7)
        self.ax_main.invert_yaxis()
        self.ax_main.set_xlabel("Similarity")
        self.ax_main.set_title(f"Top matches to {q}")
        self.fig_main.tight_layout(); self.canvas_main.draw()
        self._clear_importance("Spectral match has no per-feature importance.")

    # ── Rendering helpers ───────────────────────────────────────────────
    def _show_metrics(self, text):
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", text)
        self.nb.select(0)

    def _confusion_text(self, cm, classes):
        if cm is None:
            return "Confusion matrix: (CV not available — too few samples per class)"
        hdr = "Confusion (rows=true, cols=pred):\n      " + "  ".join(f"{c:>6}" for c in classes)
        rows = []
        for i, c in enumerate(classes):
            rows.append(f"{c:>5} " + "  ".join(f"{v:>6}" for v in cm[i]))
        return hdr + "\n" + "\n".join(rows)

    def _plot_scores(self, scores, labels, title):
        self.ax_main.clear()
        labels = list(labels)
        uniq = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        for i, u in enumerate(uniq):
            idx = [j for j, l in enumerate(labels) if l == u]
            self.ax_main.scatter(scores[idx, 0], scores[idx, 1],
                                 color=cmap(i % 10), label=(u or "(unlabelled)"), s=40)
        self.ax_main.set_title(title)
        self.ax_main.set_xlabel("Dim 1"); self.ax_main.set_ylabel("Dim 2")
        if uniq:
            self.ax_main.legend(fontsize=8)
        self.fig_main.tight_layout(); self.canvas_main.draw()

    def _plot_pred_vs_actual(self, y, pred_train, pred_cv):
        self.ax_main.clear()
        self.ax_main.scatter(y, pred_train, label="train", s=40, color="#3b7dd8")
        if pred_cv is not None:
            self.ax_main.scatter(y, pred_cv, label="CV", s=40, facecolor="none",
                                 edgecolor="#d8663b")
        lo = float(min(np.min(y), np.min(pred_train)))
        hi = float(max(np.max(y), np.max(pred_train)))
        self.ax_main.plot([lo, hi], [lo, hi], "k--", lw=1)
        self.ax_main.set_xlabel("Actual"); self.ax_main.set_ylabel("Predicted")
        self.ax_main.set_title("Predicted vs actual"); self.ax_main.legend(fontsize=8)
        self.fig_main.tight_layout(); self.canvas_main.draw()

    def _plot_confusion(self, cm, classes):
        self.ax_main.clear()
        if cm is None:
            self.ax_main.text(0.5, 0.5, "CV not available", ha="center", va="center",
                              transform=self.ax_main.transAxes, color="gray")
            self.canvas_main.draw(); return
        im = self.ax_main.imshow(cm, cmap="Blues")
        self.ax_main.set_xticks(range(len(classes))); self.ax_main.set_xticklabels(classes)
        self.ax_main.set_yticks(range(len(classes))); self.ax_main.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                self.ax_main.text(j, i, str(cm[i, j]), ha="center", va="center")
        self.ax_main.set_xlabel("Predicted"); self.ax_main.set_ylabel("True")
        self.ax_main.set_title("Confusion matrix (CV)")
        self.fig_main.colorbar(im, ax=self.ax_main)
        self.fig_main.tight_layout(); self.canvas_main.draw()

    def _plot_vector(self, vec, title, ylabel):
        self.ax_imp.clear()
        if vec is None or len(vec) != len(self.shifts):
            self.ax_imp.plot(vec)
        else:
            self.ax_imp.plot(self.shifts, vec, color="#444", lw=0.9)
            self.ax_imp.set_xlabel("Raman shift (cm⁻¹)")
        self.ax_imp.set_ylabel(ylabel); self.ax_imp.set_title(title)
        self.fig_imp.tight_layout(); self.canvas_imp.draw()

    def _clear_importance(self, msg):
        self.ax_imp.clear()
        self.ax_imp.text(0.5, 0.5, msg, ha="center", va="center",
                         transform=self.ax_imp.transAxes, color="gray")
        self.ax_imp.set_xticks([]); self.ax_imp.set_yticks([])
        self.fig_imp.tight_layout(); self.canvas_imp.draw()

    # ── Export ──────────────────────────────────────────────────────────
    def _export(self):
        if self.result is None:
            messagebox.showwarning("Export", "Run a model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", initialfile="ml_results.xlsx",
            filetypes=[("Excel files", "*.xlsx")])
        if not path:
            return
        try:
            r = self.result
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                method = r.get("method", "result")
                if method == "Spectral match":
                    pd.DataFrame(r["ranked"], columns=["sample", "similarity"]).to_excel(
                        w, sheet_name="Matches", index=False)
                elif method == "t-SNE":
                    pd.DataFrame(r["embedding"], index=r["sample_names"],
                                 columns=["tSNE1", "tSNE2"]).to_excel(w, sheet_name="Embedding")
                elif method == "PLSR":
                    d = {"actual": r["y"], "pred_train": r["pred_train"]}
                    if r["pred_cv"] is not None:
                        d["pred_cv"] = r["pred_cv"]
                    pd.DataFrame(d, index=r["sample_names"]).to_excel(w, sheet_name="Predictions")
                    pd.DataFrame({"raman_shift": self.shifts, "vip": r["vip"],
                                  "coef": r["coef"]}).to_excel(w, sheet_name="VIP_Coef", index=False)
                    pd.DataFrame({"metric": ["r2_train", "r2_cv", "rmsecv"],
                                  "value": [r["r2_train"], r["r2_cv"], r["rmsecv"]]}).to_excel(
                        w, sheet_name="Metrics", index=False)
                else:  # PLS-DA / classifiers
                    if "scores" in r:
                        pd.DataFrame(r["scores"], index=r["sample_names"]).to_excel(
                            w, sheet_name="Scores")
                    if r.get("vip") is not None:
                        pd.DataFrame({"raman_shift": self.shifts, "vip": r["vip"]}).to_excel(
                            w, sheet_name="VIP", index=False)
                    if r.get("importance") is not None:
                        pd.DataFrame({"raman_shift": self.shifts,
                                      "importance": r["importance"]}).to_excel(
                            w, sheet_name="Importance", index=False)
                    if r.get("confusion") is not None:
                        pd.DataFrame(r["confusion"], index=r["classes"],
                                     columns=r["classes"]).to_excel(w, sheet_name="Confusion")
                    acc = r.get("cv_accuracy", float("nan"))
                    pd.DataFrame({"metric": ["cv_accuracy"], "value": [acc]}).to_excel(
                        w, sheet_name="Metrics", index=False)
            messagebox.showinfo("Export", f"Results saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
