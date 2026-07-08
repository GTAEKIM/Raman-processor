"""Smoke tests for microalgae band analysis and a few core helpers.

Run: python -m pytest tests/ -q   (or: python tests/test_microalgae.py)
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing_logic import (  # noqa: E402
    band_metrics, compute_band_ratios, microalgae_report, normalize_spectrum,
)


def _synthetic(x, peaks):
    y = 5.0 + x * 2e-3  # sloped fluorescence background
    for c, a in peaks:
        y = y + a * np.exp(-((x - c) ** 2) / (2 * 8.0 ** 2))
    return y


def test_band_metrics_and_ratios():
    x = np.linspace(400, 3100, 1400)
    y = _synthetic(x, [(1440, 100), (1655, 60), (1520, 80), (478, 50),
                       (2850, 120), (3011, 30), (1003, 40), (1265, 25)])
    bands = [
        {"name": "CH2bend", "lo": 1430, "hi": 1455},
        {"name": "C=C", "lo": 1645, "hi": 1670},
        {"name": "starch", "lo": 468, "hi": 488},
    ]
    m = band_metrics(x, y, bands, method="height", local_baseline=True)
    # All three windows have a peak -> positive height
    assert m["CH2bend"] > 0 and m["C=C"] > 0 and m["starch"] > 0
    # Unsaturation ratio in a sane range (peaks 60/100 before local-baseline)
    r = compute_band_ratios(m, [{"name": "unsat", "numerator": "C=C",
                                 "denominator": "CH2bend"}])
    assert 0.3 < r["unsat"] < 0.9


def test_ratio_divide_by_zero_is_nan():
    m = {"a": 10.0, "b": 0.0}
    r = compute_band_ratios(m, [{"name": "x", "numerator": "a", "denominator": "b"}])
    assert np.isnan(r["x"])


def test_missing_band_window_is_nan():
    x = np.linspace(400, 1800, 200)
    y = _synthetic(x, [(1440, 100)])
    # Band entirely outside the axis -> NaN
    m = band_metrics(x, y, [{"name": "far", "lo": 2800, "hi": 2900}])
    assert np.isnan(m["far"])


def test_report_scale_invariance_of_height_ratios():
    x = np.linspace(400, 3100, 1200)
    y = _synthetic(x, [(1440, 100), (1655, 60)])
    df = pd.DataFrame({"s1": y, "s2": y * 2.0})
    df.insert(0, "Raman shift (cm-1)", x)
    bands = [{"name": "CH2bend", "lo": 1430, "hi": 1455},
             {"name": "C=C", "lo": 1645, "hi": 1670}]
    ratios = [{"name": "unsat", "numerator": "C=C", "denominator": "CH2bend"}]
    idf, rdf = microalgae_report(df, bands, ratios)
    assert idf.shape == (2, 2) and rdf.shape == (2, 1)
    # Height ratios are invariant to a global intensity scale
    assert np.isclose(rdf.loc["s1", "unsat"], rdf.loc["s2", "unsat"])


def test_config_default_library_loads():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    assert len(cfg["microalgae_bands"]) >= 15
    assert len(cfg["microalgae_ratios"]) >= 5
    # Every ratio references defined band names
    names = {b["name"] for b in cfg["microalgae_bands"]}
    for r in cfg["microalgae_ratios"]:
        assert r["numerator"] in names, r
        assert r["denominator"] in names, r
    # Every band/ratio 'ref' resolves to a known reference key with a URL
    refs = {r["key"]: r for r in cfg.get("microalgae_references", [])}
    assert refs, "no microalgae_references defined"
    for r in refs.values():
        assert r.get("url"), r
        assert r.get("citation"), r
    for item in cfg["microalgae_bands"] + cfg["microalgae_ratios"]:
        if item.get("ref"):
            assert item["ref"] in refs, f"unknown ref: {item}"


def test_normalize_snv_zero_mean_unit_std():
    y = np.random.RandomState(0).rand(500) * 10 + 3
    z = normalize_spectrum(y, method="snv")
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
