"""Microalgae band-analysis window.

Import (or reuse batch-processed) Raman spectra, measure named spectral band
intensities (pigment / lipid / saturated-FA / PUFA / protein / carbohydrate /
total-biomass), and compute compositional ratios (unsaturation, saturation
index, PUFA index, lipid/protein, carotenoid/lipid, starch/lipid,
total-lipid/biomass).

The band library and ratio definitions are literature-based defaults loaded from
config.json; they are editable in-window and can be saved back to config.

Scientific caveats (shown in the UI):
- Amide I (~1650) overlaps lipid C=C (~1650): the I1658/I1441 unsaturation ratio
  is cleanest on lipid-dominated spectra; use amide III (~1250) for protein.
- Phe (~1004) nearly overlaps carotenoid v3 (~1008); confirm pigment with the
  carotenoid v1 (~1520) band.
"""

import os
import json
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing_logic import band_metrics, compute_band_ratios, microalgae_report
from ui_helpers import scrolled


_CAVEAT = (
    "Note: amide I (~1650) overlaps lipid C=C (~1650) — unsaturation ratio is "
    "cleanest on lipid-rich spectra; use amide III (~1250) for protein. "
    "Phe (~1004) overlaps carotenoid v3 (~1008)."
)


class MicroalgaeWindow(tk.Toplevel):
    def __init__(self, parent, processed_df: pd.DataFrame,
                 config: Dict[str, Any], config_path: Optional[str] = None):
        super().__init__(parent)
        self.title("Microalgae Band Analysis")
        self.geometry("1280x760")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.processed_df = processed_df
        self.config = config
        self.config_path = config_path
        self.x = processed_df.iloc[:, 0].values.astype(float)
        self.sample_names = list(processed_df.columns[1:])

        # Working copies of the editable library
        self.bands: List[Dict[str, Any]] = [dict(b) for b in config.get("microalgae_bands", [])]
        self.ratios: List[Dict[str, Any]] = [dict(r) for r in config.get("microalgae_ratios", [])]
        self.references: List[Dict[str, Any]] = [dict(r) for r in config.get("microalgae_references", [])]
        self._ref_by_key = {r.get("key", ""): r for r in self.references}

        mcfg = config.get("microalgae", {})
        self.method_var = tk.StringVar(value=mcfg.get("default_method", "height"))
        self.local_bl_var = tk.BooleanVar(value=bool(mcfg.get("local_baseline", True)))
        self.sample_var = tk.StringVar(value=self.sample_names[0] if self.sample_names else "")

        self.intensity_df: Optional[pd.DataFrame] = None
        self.ratio_df: Optional[pd.DataFrame] = None
        self.figures: List[plt.Figure] = []

        self._build_ui()
        self._refresh_band_tree()
        self._refresh_ratio_tree()
        self._compute()
        self.grab_set()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(pane, width=380)
        left.pack_propagate(False)
        pane.add(left, weight=0)
        self._build_left(left)

        mid = ttk.Frame(pane)
        pane.add(mid, weight=2)
        self._build_center(mid)

        right = ttk.Frame(pane, width=430)
        right.pack_propagate(False)
        pane.add(right, weight=2)
        self._build_right(right)

        cap = ttk.Label(self, text=_CAVEAT, foreground="gray",
                        wraplength=1240, justify="left")
        cap.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    def _build_left(self, parent):
        # Method controls
        opt = ttk.LabelFrame(parent, text="Measurement")
        opt.pack(fill="x", padx=4, pady=4)
        row = ttk.Frame(opt); row.pack(fill="x", padx=5, pady=3)
        ttk.Label(row, text="Metric:").pack(side="left")
        ttk.Combobox(row, textvariable=self.method_var,
                     values=["height", "area", "mean"], state="readonly",
                     width=10).pack(side="right")
        ttk.Checkbutton(opt, text="Subtract local linear baseline (recommended)",
                        variable=self.local_bl_var).pack(anchor="w", padx=5, pady=2)
        ttk.Button(opt, text="Compute", command=self._compute).pack(
            fill="x", padx=5, pady=4)

        # Band library
        bl = ttk.LabelFrame(parent, text="Band library (cm⁻¹)")
        bl.pack(fill="both", expand=True, padx=4, pady=4)
        cols = ("name", "class", "lo", "hi", "ref")
        self.band_tree = scrolled(
            bl, lambda c: ttk.Treeview(c, columns=cols, show="headings", height=10))
        for c, w in zip(cols, (135, 80, 48, 48, 75)):
            self.band_tree.heading(c, text=c)
            self.band_tree.column(c, width=w, anchor="w", stretch=False)
        self.band_tree.bind("<Double-1>", lambda e: self._edit_band())
        brow = ttk.Frame(bl); brow.pack(fill="x", padx=4, pady=2)
        ttk.Button(brow, text="Add", command=self._add_band).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(brow, text="Edit", command=self._edit_band).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(brow, text="Remove", command=self._remove_band).pack(side="left", expand=True, fill="x", padx=1)

        # Ratio library
        rl = ttk.LabelFrame(parent, text="Ratios")
        rl.pack(fill="both", expand=True, padx=4, pady=4)
        rcols = ("name", "numerator", "denominator", "ref")
        self.ratio_tree = scrolled(
            rl, lambda c: ttk.Treeview(c, columns=rcols, show="headings", height=6))
        for c, w in zip(rcols, (120, 100, 100, 70)):
            self.ratio_tree.heading(c, text=c)
            self.ratio_tree.column(c, width=w, anchor="w", stretch=False)
        self.ratio_tree.bind("<Double-1>", lambda e: self._edit_ratio())
        rrow = ttk.Frame(rl); rrow.pack(fill="x", padx=4, pady=2)
        ttk.Button(rrow, text="Add", command=self._add_ratio).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(rrow, text="Edit", command=self._edit_ratio).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(rrow, text="Remove", command=self._remove_ratio).pack(side="left", expand=True, fill="x", padx=1)

        # Library persistence
        lib = ttk.Frame(parent); lib.pack(fill="x", padx=4, pady=4)
        ttk.Button(lib, text="Reset to defaults", command=self._reset_defaults).pack(
            side="left", expand=True, fill="x", padx=1)
        ttk.Button(lib, text="Save to config", command=self._save_to_config).pack(
            side="left", expand=True, fill="x", padx=1)

    def _build_center(self, parent):
        top = ttk.Frame(parent); top.pack(fill="x", padx=4, pady=2)
        ttk.Label(top, text="Sample:").pack(side="left")
        combo = ttk.Combobox(top, textvariable=self.sample_var,
                             values=self.sample_names, state="readonly", width=28)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self._draw_spectrum())

        self.fig_spec, self.ax_spec = plt.subplots(figsize=(6, 5.5))
        self.figures.append(self.fig_spec)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=parent)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Intensities tab
        it = ttk.Frame(nb); nb.add(it, text="Intensities")
        self.int_tree = scrolled(it, lambda c: ttk.Treeview(c, show="headings", height=12))

        # Ratios tab
        rt = ttk.Frame(nb); nb.add(rt, text="Ratios")
        self.rat_tree = scrolled(rt, lambda c: ttk.Treeview(c, show="headings", height=12))

        # Bar chart tab
        bc = ttk.Frame(nb); nb.add(bc, text="Bar chart")
        selrow = ttk.Frame(bc); selrow.pack(fill="x", padx=4, pady=2)
        ttk.Label(selrow, text="Show:").pack(side="left")
        self.bar_target = tk.StringVar(value="")
        self.bar_combo = ttk.Combobox(selrow, textvariable=self.bar_target,
                                      state="readonly", width=28)
        self.bar_combo.pack(side="left", padx=4)
        self.bar_combo.bind("<<ComboboxSelected>>", lambda e: self._draw_bar())
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(5, 4.5))
        self.figures.append(self.fig_bar)
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=bc)
        self.canvas_bar.get_tk_widget().pack(fill="both", expand=True)

        # Sources tab — literature references for the band library
        src = ttk.Frame(nb); nb.add(src, text="Sources")
        ttk.Label(src, text="Literature sources for the band library. "
                            "Double-click a row to open it in your browser.",
                  foreground="gray", wraplength=400, justify="left").pack(
            anchor="w", padx=4, pady=(4, 2))
        scols = ("key", "citation")
        self.src_tree = scrolled(
            src, lambda c: ttk.Treeview(c, columns=scols, show="headings", height=8))
        self.src_tree.heading("key", text="key")
        self.src_tree.column("key", width=80, anchor="w", stretch=False)
        self.src_tree.heading("citation", text="citation")
        self.src_tree.column("citation", width=320, anchor="w")
        for r in self.references:
            self.src_tree.insert("", "end", values=(r.get("key", ""), r.get("citation", "")))
        self.src_tree.bind("<Double-1>", self._open_reference)
        ttk.Button(src, text="Open selected source in browser",
                   command=self._open_reference).pack(fill="x", padx=4, pady=(0, 4))

        ttk.Button(parent, text="Export to Excel...",
                   command=self._export).pack(fill="x", padx=4, pady=4)

    def _open_reference(self, event=None):
        sel = self.src_tree.selection()
        if not sel:
            return
        key = self.src_tree.item(sel[0], "values")[0]
        ref = self._ref_by_key.get(key)
        if ref and ref.get("url"):
            try:
                webbrowser.open(ref["url"])
                self.title(f"Microalgae Band Analysis — opened {key}")
            except Exception as e:
                messagebox.showerror("Open source", str(e))

    # ── Library tree helpers ───────────────────────────────────────────
    def _refresh_band_tree(self):
        self.band_tree.delete(*self.band_tree.get_children())
        for b in self.bands:
            self.band_tree.insert("", "end", values=(
                b.get("name", ""), b.get("class", ""), b.get("lo", ""),
                b.get("hi", ""), b.get("ref", "")))

    def _refresh_ratio_tree(self):
        self.ratio_tree.delete(*self.ratio_tree.get_children())
        for r in self.ratios:
            self.ratio_tree.insert("", "end", values=(
                r.get("name", ""), r.get("numerator", ""),
                r.get("denominator", ""), r.get("ref", "")))

    def _selected_index(self, tree) -> Optional[int]:
        sel = tree.selection()
        if not sel:
            return None
        return tree.index(sel[0])

    def _add_band(self):
        self._band_dialog(None)

    def _edit_band(self):
        i = self._selected_index(self.band_tree)
        if i is None:
            messagebox.showinfo("Edit", "Select a band row first.")
            return
        self._band_dialog(i)

    def _remove_band(self):
        i = self._selected_index(self.band_tree)
        if i is None:
            return
        del self.bands[i]
        self._refresh_band_tree()

    def _band_dialog(self, index: Optional[int]):
        b = self.bands[index] if index is not None else {"name": "", "class": "", "lo": 0.0, "hi": 0.0, "ref": ""}
        dlg = tk.Toplevel(self); dlg.title("Band"); dlg.grab_set()
        dlg.transient(self)
        vars_ = {
            "name": tk.StringVar(value=b.get("name", "")),
            "class": tk.StringVar(value=b.get("class", "")),
            "lo": tk.StringVar(value=str(b.get("lo", ""))),
            "hi": tk.StringVar(value=str(b.get("hi", ""))),
        }
        ref_v = tk.StringVar(value=b.get("ref", ""))
        ref_keys = [r.get("key", "") for r in self.references]
        for i, (lbl, key) in enumerate([("Name", "name"), ("Class", "class"),
                                        ("Low (cm⁻¹)", "lo"), ("High (cm⁻¹)", "hi")]):
            ttk.Label(dlg, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(dlg, textvariable=vars_[key], width=24).grid(row=i, column=1, padx=6, pady=4)
        ttk.Label(dlg, text="Reference").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(dlg, textvariable=ref_v, values=ref_keys, width=22).grid(
            row=4, column=1, padx=6, pady=4)

        def save():
            try:
                new = {
                    "name": vars_["name"].get().strip(),
                    "class": vars_["class"].get().strip(),
                    "lo": float(vars_["lo"].get()),
                    "hi": float(vars_["hi"].get()),
                    "ref": ref_v.get().strip(),
                }
            except ValueError:
                messagebox.showerror("Band", "Low/High must be numbers.", parent=dlg)
                return
            if not new["name"]:
                messagebox.showerror("Band", "Name is required.", parent=dlg)
                return
            if index is not None:
                self.bands[index] = new
            else:
                self.bands.append(new)
            self._refresh_band_tree()
            dlg.destroy()

        ttk.Button(dlg, text="Save", command=save).grid(row=5, column=0, columnspan=2, pady=8)

    def _add_ratio(self):
        self._ratio_dialog(None)

    def _edit_ratio(self):
        i = self._selected_index(self.ratio_tree)
        if i is None:
            messagebox.showinfo("Edit", "Select a ratio row first.")
            return
        self._ratio_dialog(i)

    def _remove_ratio(self):
        i = self._selected_index(self.ratio_tree)
        if i is None:
            return
        del self.ratios[i]
        self._refresh_ratio_tree()

    def _ratio_dialog(self, index: Optional[int]):
        r = self.ratios[index] if index is not None else {"name": "", "numerator": "", "denominator": "", "ref": ""}
        names = [b["name"] for b in self.bands]
        ref_keys = [rr.get("key", "") for rr in self.references]
        dlg = tk.Toplevel(self); dlg.title("Ratio"); dlg.grab_set(); dlg.transient(self)
        name_v = tk.StringVar(value=r.get("name", ""))
        num_v = tk.StringVar(value=r.get("numerator", ""))
        den_v = tk.StringVar(value=r.get("denominator", ""))
        ref_v = tk.StringVar(value=r.get("ref", ""))
        ttk.Label(dlg, text="Name").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(dlg, textvariable=name_v, width=26).grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(dlg, text="Numerator").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(dlg, textvariable=num_v, values=names, state="readonly",
                     width=24).grid(row=1, column=1, padx=6, pady=4)
        ttk.Label(dlg, text="Denominator").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(dlg, textvariable=den_v, values=names, state="readonly",
                     width=24).grid(row=2, column=1, padx=6, pady=4)
        ttk.Label(dlg, text="Reference").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(dlg, textvariable=ref_v, values=ref_keys, width=22).grid(
            row=3, column=1, padx=6, pady=4)

        def save():
            new = {"name": name_v.get().strip(),
                   "numerator": num_v.get(), "denominator": den_v.get(),
                   "ref": ref_v.get().strip()}
            if not (new["name"] and new["numerator"] and new["denominator"]):
                messagebox.showerror("Ratio", "Name/numerator/denominator are required.", parent=dlg)
                return
            if index is not None:
                self.ratios[index] = new
            else:
                self.ratios.append(new)
            self._refresh_ratio_tree()
            dlg.destroy()

        ttk.Button(dlg, text="Save", command=save).grid(row=4, column=0, columnspan=2, pady=8)

    def _reset_defaults(self):
        if not messagebox.askyesno("Reset", "Reset band library and ratios to config defaults?"):
            return
        self.bands = [dict(b) for b in self.config.get("microalgae_bands", [])]
        self.ratios = [dict(r) for r in self.config.get("microalgae_ratios", [])]
        self._refresh_band_tree(); self._refresh_ratio_tree()

    def _save_to_config(self):
        self.config["microalgae_bands"] = [dict(b) for b in self.bands]
        self.config["microalgae_ratios"] = [dict(r) for r in self.ratios]
        self.config["microalgae_references"] = [dict(r) for r in self.references]
        if not self.config_path:
            messagebox.showwarning("Save", "No config path known; changes kept in memory only.")
            return
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Save", f"Band library saved to\n{self.config_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ── Compute & display ───────────────────────────────────────────────
    def _compute(self):
        if not self.bands:
            messagebox.showwarning("Compute", "No bands defined.")
            return
        try:
            self.intensity_df, self.ratio_df = microalgae_report(
                self.processed_df, self.bands, self.ratios,
                method=self.method_var.get(), local_baseline=self.local_bl_var.get(),
            )
        except Exception as e:
            messagebox.showerror("Compute Error", str(e))
            return
        self._fill_table(self.int_tree, self.intensity_df)
        self._fill_table(self.rat_tree, self.ratio_df)
        # Populate bar-chart target list (bands + ratios)
        targets = list(self.intensity_df.columns) + ["[ratio] " + c for c in self.ratio_df.columns]
        self.bar_combo.configure(values=targets)
        if targets and not self.bar_target.get():
            self.bar_target.set(targets[0])
        self._draw_spectrum()
        self._draw_bar()

    def _fill_table(self, tree, df: pd.DataFrame):
        tree.delete(*tree.get_children())
        cols = ["sample"] + list(df.columns)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            # stretch=False + fixed width so the horizontal scrollbar engages
            # when there are more/wider columns than the panel width.
            tree.column(c, width=max(80, min(170, 9 * len(str(c)))),
                        anchor="center", stretch=False)
        for name, row in df.iterrows():
            vals = [name] + [("" if pd.isna(v) else f"{v:.3g}") for v in row.values]
            tree.insert("", "end", values=vals)

    def _draw_spectrum(self):
        name = self.sample_var.get()
        if not name or name not in self.processed_df.columns:
            return
        y = self.processed_df[name].values.astype(float)
        self.ax_spec.clear()
        self.ax_spec.plot(self.x, y, color="black", lw=0.9)
        # Shade band windows, colour-coded by class
        classes = sorted({b.get("class", "") for b in self.bands})
        cmap = plt.get_cmap("tab10")
        cmap_by = {c: cmap(i % 10) for i, c in enumerate(classes)}
        seen = set()
        for b in self.bands:
            lo, hi = sorted((float(b["lo"]), float(b["hi"])))
            cls = b.get("class", "")
            lbl = cls if cls not in seen else None
            seen.add(cls)
            self.ax_spec.axvspan(lo, hi, color=cmap_by.get(cls, "gray"),
                                 alpha=0.18, label=lbl)
        self.ax_spec.set_title(f"{name} — bands shaded by class")
        self.ax_spec.set_xlabel("Raman shift (cm⁻¹)")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.legend(fontsize=7, ncol=2, loc="upper right")
        self.ax_spec.grid(True, alpha=0.3)
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    def _draw_bar(self):
        target = self.bar_target.get()
        if not target or self.intensity_df is None:
            return
        if target.startswith("[ratio] "):
            col = target[len("[ratio] "):]
            series = self.ratio_df[col]
            ylabel = "Ratio"
        else:
            series = self.intensity_df[target]
            ylabel = "Intensity"
        self.ax_bar.clear()
        self.ax_bar.bar(range(len(series)), series.values, color="#3b7dd8")
        self.ax_bar.set_xticks(range(len(series)))
        self.ax_bar.set_xticklabels(series.index, rotation=45, ha="right", fontsize=7)
        self.ax_bar.set_ylabel(ylabel)
        self.ax_bar.set_title(target)
        self.ax_bar.grid(True, axis="y", alpha=0.3)
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()

    def _export(self):
        if self.intensity_df is None:
            messagebox.showwarning("Export", "Compute first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Microalgae Report",
            defaultextension=".xlsx",
            initialfile="microalgae_report.xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                self.intensity_df.to_excel(w, sheet_name="Intensities")
                self.ratio_df.to_excel(w, sheet_name="Ratios")
                pd.DataFrame(self.bands).to_excel(w, sheet_name="BandDefinitions", index=False)
                pd.DataFrame(self.ratios).to_excel(w, sheet_name="RatioDefinitions", index=False)
                if self.references:
                    pd.DataFrame(self.references).to_excel(w, sheet_name="References", index=False)
                pd.DataFrame({
                    "setting": ["method", "local_baseline"],
                    "value": [self.method_var.get(), self.local_bl_var.get()],
                }).to_excel(w, sheet_name="Settings", index=False)
            messagebox.showinfo("Export", f"Report saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _on_closing(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
        self.destroy()
