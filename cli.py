"""Headless CLI for batch-processing Raman spectra.

Example:
    python cli.py input.xlsx output.xlsx --params params.json --parallel
    python cli.py input.csv output.xlsx --pca --nmf 3
    python cli.py input.xlsx output.xlsx --range 400 3300 --baseline airpls \\
        --normalize snv --cosmic --parallel --n-jobs -1

No GUI — useful for scripting, automation, and reproducibility.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from processing_logic import (
    DataProcessor,
    load_baseline_plugins,
)


def _build_params(args) -> Dict[str, Any]:
    if args.params:
        with open(args.params, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Build from flags
    return {
        "range": {"lower_bound": args.range[0], "upper_bound": args.range[1]},
        "smoothing": {
            "sg_poly_order": args.sg_order,
            "sg_frame_window": args.sg_window,
        },
        "baseline": {
            "algorithm": args.baseline,
            "params": _default_baseline_params(args.baseline),
        },
        "normalization": args.normalize,
        "preprocessing": {"apply_cosmic_ray": args.cosmic},
        "derivative": {"enabled": False, "order": 1, "window": 15, "polyorder": 3},
    }


def _default_baseline_params(algo: str) -> Dict[str, Any]:
    defaults = {
        "airpls": {"lam": 1e6, "diff_order": 2},
        "arpls":  {"lam": 1e5, "diff_order": 2},
        "asls":   {"lam": 1e6, "p": 0.01, "diff_order": 2},
        "snip":   {"max_half_window": 40, "filter_order": 2},
        "atq":    {"order": 2, "threshold": 0.01},
        "stq":    {"order": 2, "threshold": 0.01},
        "ah":     {"order": 2, "threshold": 0.01},
        "sh":     {"order": 2, "threshold": 0.01},
        "mor":    {"half_window": 25},
    }
    return defaults.get(algo, {})


def _progress(current: int, total: int) -> None:
    bar_len = 30
    frac = current / total if total else 0
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Headless Raman batch processor")
    p.add_argument("input", help="Input data file (xlsx/csv/txt/asc/dat)")
    p.add_argument("output", help="Output Excel file (.xlsx)")
    p.add_argument("--params", help="JSON parameter file (overrides flags)")

    # Pipeline flags
    p.add_argument("--range", nargs=2, type=float, default=[400.0, 3300.0],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--sg-order", type=int, default=1)
    p.add_argument("--sg-window", type=int, default=15)
    p.add_argument("--baseline", default="airpls",
                   choices=["airpls", "arpls", "asls", "snip",
                            "atq", "stq", "ah", "sh", "mor"])
    p.add_argument("--normalize", default="none",
                   choices=["none", "snv", "vector", "area", "minmax", "maxpeak"])
    p.add_argument("--cosmic", action="store_true", help="Remove cosmic ray spikes")

    # Batch
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)

    # Analyses
    p.add_argument("--pca", action="store_true")
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--pca-scaling", default="auto",
                   choices=["auto", "mean", "pareto", "none"])
    p.add_argument("--nmf", type=int, metavar="K",
                   help="Run NMF with K components")

    # Plugins
    p.add_argument("--plugin-dir", default=os.path.join(os.path.dirname(__file__), "plugins"),
                   help="Directory containing baseline plugins (default: ./plugins)")

    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Auto-load plugins
    try:
        n_plug = load_baseline_plugins(args.plugin_dir)
        if n_plug:
            print(f"Loaded {n_plug} baseline plugin(s) from {args.plugin_dir}")
    except Exception as e:
        print(f"Warning: plugin loading failed: {e}")

    params = _build_params(args)

    # Load data
    processor = DataProcessor()
    n = processor.load_data(args.input)
    print(f"Loaded {n} spectra from {args.input}")

    # Apply range
    lo = params["range"]["lower_bound"]; hi = params["range"]["upper_bound"]
    processor.filter_data_by_range(lo, hi)

    # Batch process
    t0 = time.time()
    processed_df, summary = processor.run_batch_processing(
        params, progress_callback=_progress,
        parallel=args.parallel, n_jobs=args.n_jobs,
    )
    elapsed = time.time() - t0
    print(f"Batch done: processed={summary['processed']}  failed={summary['failed']}  "
          f"QC flagged={summary.get('qc_flagged', 0)}  time={elapsed:.2f}s")

    # Write output workbook (+ optional PCA / NMF sheets)
    qc_df = summary.get('qc_df')
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        processed_df.set_index("Raman shift (cm-1)").T.to_excel(
            writer, sheet_name="Processed", header=True, index=True
        )
        if qc_df is not None and not qc_df.empty:
            qc_df.to_excel(writer, sheet_name="QC", index=False)

        if args.pca:
            try:
                pca_res = processor.perform_pca(
                    processed_df,
                    n_components=args.pca_components,
                    scaling=args.pca_scaling,
                )
                pd.DataFrame(
                    pca_res["scores"],
                    index=pca_res["sample_names"],
                    columns=[f"PC{i + 1}" for i in range(pca_res["scores"].shape[1])],
                ).to_excel(writer, sheet_name="PCA_Scores")
                pd.DataFrame({
                    "Principal_Component": [f"PC{i + 1}" for i in range(len(pca_res["explained_variance_ratio"]))],
                    "Eigenvalue": pca_res["eigenvalues"],
                    "Explained_Variance_Ratio": pca_res["explained_variance_ratio"],
                    "Cumulative": np.cumsum(pca_res["explained_variance_ratio"]),
                }).to_excel(writer, sheet_name="PCA_Variance", index=False)
                print(f"PCA: {pca_res['n_components_selected']} PCs, "
                      f"{pca_res['total_explained_variance']*100:.2f}% total variance")
            except Exception as e:
                print(f"PCA failed: {e}")

        if args.nmf is not None and args.nmf > 0:
            try:
                nmf_res = processor.perform_nmf(processed_df, n_components=args.nmf)
                pd.DataFrame(
                    nmf_res["weights"],
                    index=nmf_res["sample_names"],
                    columns=[f"Component_{i + 1}" for i in range(args.nmf)],
                ).to_excel(writer, sheet_name="NMF_Weights")
                comp_df = pd.DataFrame(
                    nmf_res["components"],
                    columns=[f"Component_{i + 1}" for i in range(args.nmf)],
                )
                comp_df.insert(0, "Raman Shift (cm-1)", nmf_res["raman_shifts"])
                comp_df.to_excel(writer, sheet_name="NMF_Components", index=False)
                print(f"NMF: {args.nmf} components, reconstruction error = "
                      f"{nmf_res['reconstruction_error']:.4f}")
            except Exception as e:
                print(f"NMF failed: {e}")

    # Sidecar JSON parameters
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"Output written to {args.output}")
    print(f"Parameters saved to {json_path}")


if __name__ == "__main__":
    main()
