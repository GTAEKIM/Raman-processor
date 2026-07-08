"""Smoke tests for the ML / chemometrics functions."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing_logic import (  # noqa: E402
    ml_matrix, perform_plsda, perform_plsr, perform_classifier,
    perform_tsne, spectral_match,
)


def _two_class_df(seed=0, n_each=6, noise=0.4):
    rng = np.random.RandomState(seed)
    x = np.linspace(400, 1800, 250)
    cols, labels = {}, []
    for i in range(2 * n_each):
        cls = "A" if i < n_each else "B"
        c = 1000 if cls == "A" else 1400
        y = 1 + 100 * np.exp(-((x - c) ** 2) / (2 * 10.0 ** 2)) + noise * rng.randn(250)
        cols[f"{cls}{i}"] = y
        labels.append(cls)
    df = pd.DataFrame(cols)
    df.insert(0, "Raman shift (cm-1)", x)
    return df, labels


def test_ml_matrix_shape():
    df, labels = _two_class_df()
    X, names, shifts = ml_matrix(df)
    assert X.shape == (12, 250)
    assert len(names) == 12 and len(shifts) == 250


def test_plsda_separates_classes():
    df, labels = _two_class_df()
    X, _, _ = ml_matrix(df)
    r = perform_plsda(X, labels, n_components=2, cv_folds=4)
    assert r["train_accuracy"] > 0.9
    assert r["cv_accuracy"] > 0.8
    assert len(r["vip"]) == X.shape[1]
    assert r["confusion"].shape == (2, 2)


def test_plsr_recovers_gradient():
    df, _ = _two_class_df(noise=0.05)
    X, _, _ = ml_matrix(df)
    y = np.linspace(0, 11, X.shape[0])  # smooth target
    r = perform_plsr(X, y, n_components=3, cv_folds=4)
    assert r["r2_train"] > 0.5
    assert len(r["coef"]) == X.shape[1]


def test_classifiers_run():
    df, labels = _two_class_df()
    X, _, _ = ml_matrix(df)
    for model in ("rf", "svm", "knn"):
        r = perform_classifier(X, labels, model=model, cv_folds=4)
        assert 0.0 <= r["cv_accuracy"] <= 1.0
        assert r["confusion"].shape == (2, 2)


def test_tsne_embeds_2d():
    df, _ = _two_class_df()
    X, _, _ = ml_matrix(df)
    r = perform_tsne(X, perplexity=4)
    assert r["embedding"].shape == (X.shape[0], 2)


def test_spectral_match_self_first():
    df, _ = _two_class_df()
    X, names, _ = ml_matrix(df)
    lib = {n: df[n].values for n in names}
    ranked = spectral_match(df[names[0]].values, lib, metric="cosine")
    assert ranked[0][0] == names[0]
    # SAM similarity in [0, 1]
    ranked_sam = spectral_match(df[names[0]].values, lib, metric="sam")
    assert all(0.0 <= s <= 1.0 for _, s in ranked_sam)


def test_plsda_requires_two_classes():
    df, _ = _two_class_df()
    X, _, _ = ml_matrix(df)
    try:
        perform_plsda(X, ["A"] * X.shape[0])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
