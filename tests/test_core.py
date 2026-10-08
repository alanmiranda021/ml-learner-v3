"""Testes mínimos dos módulos de core (independentes da UI)."""
import numpy as np
import pandas as pd
import pytest

from core.preprocess import (MADScaler, TanhScaler, Winsorizer,
                             build_full_pipeline, multivariate_outlier_report)
from core.evaluate import holdout, make_pipeline, cross_val
from core.io import load_table, validate_test_columns
from core.models import REGRESSION


def test_winsorizer_clip():
    X = np.array([[1.0], [2.0], [1000.0], [3.0]])
    w = Winsorizer(k=1.5).fit(X)
    out = w.transform(X)
    assert out.max() < 1000.0


def test_mad_scaler_mediana_zero():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    m = MADScaler().fit(X)
    out = m.transform(X)
    assert abs(np.median(out)) < 1e-9


def test_tanh_scaler_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    t = TanhScaler().fit(X)
    out = t.transform(X)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_multivariate_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[0] = [50, 50, 50]
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    out = multivariate_outlier_report(df, method="LOF", contamination=0.02)
    assert out["_outlier"].sum() >= 1


def test_pipeline_holdout_regression():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = X["a"] * 2 + X["b"] - X["c"] + rng.normal(scale=0.1, size=200)
    pre = build_full_pipeline(list("abc"), [], scaler="Z-score (Standard)",
                              winsor=True, k=1.5, imputer="Mediana")
    model, _ = REGRESSION["Regressão Linear"]
    pipe = make_pipeline(pre, model, None, False, task="regression")
    m, _ = holdout(pipe, X, y, "regression", test_size=0.2, seed=0)
    assert m["R2"] > 0.8


def test_savgol_is_causal():
    from core.preprocess import CausalSmoother
    x1 = pd.DataFrame({"x": np.arange(30, dtype=float)})
    x2 = x1.copy()
    x2.loc[12, "x"] = 10000.0
    s = CausalSmoother("Savitzky-Golay", window=5)
    y1 = s.transform(x1)[:, 0]
    y2 = s.transform(x2)[:, 0]
    # O ponto futuro t=12 não pode alterar t=10.
    assert np.isclose(y1[10], y2[10])


def test_savgol_with_missing_values_does_not_use_future_data():
    from core.preprocess import CausalSmoother
    x1 = pd.DataFrame({"x": np.arange(20, dtype=float)})
    x1.loc[6, "x"] = np.nan
    x2 = x1.copy()
    x2.loc[7, "x"] = 10000.0
    smoother = CausalSmoother("Savitzky-Golay", window=5)
    y1 = smoother.transform(x1)[:, 0]
    y2 = smoother.transform(x2)[:, 0]
    assert np.isclose(y1[6], y2[6])


def test_holdout_keeps_true_oos_test_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = X["a"] - 0.5 * X["b"] + rng.normal(scale=.1, size=120)
    pre = build_full_pipeline(list("abc"), [], scaler="Z-score (Standard)")
    model, _ = REGRESSION["Regressão Linear"]
    pipe = make_pipeline(pre, model, None, False, task="regression")
    _, extra = holdout(pipe, X, y, "regression", test_size=.2, seed=42)
    assert "X_test" in extra and "y_test" in extra
    assert len(extra["X_test"]) == len(extra["y_test"]) == 24


def test_validate_test_columns_reports_missing_fields():
    test_df = pd.DataFrame({"a": [1], "target": [2]})
    with pytest.raises(ValueError, match="b"):
        validate_test_columns(test_df, ["a", "b"], "target")


def test_load_hdf5_table():
    h5py = pytest.importorskip("h5py")
    import io
    raw = io.BytesIO()
    with h5py.File(raw, "w") as handle:
        handle.create_dataset("dados", data=np.array([[1, 2], [3, 4]]))
    frame = load_table("dados.h5", raw.getvalue())
    assert frame.shape == (2, 2)
