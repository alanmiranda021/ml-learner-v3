import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from core.evaluate import make_pipeline
from core.preprocess import build_preprocessor
from core.reduction import ValidatedReduction
from core.explain import estimator_input, explain_pipeline


def test_explained_prediction_and_component_names():
    df = pd.DataFrame({"x": np.arange(20, dtype=float), "cat": ["a", "b"]*10})
    df.loc[0, "x"] = np.nan
    pipe = make_pipeline(build_preprocessor(["x"], ["cat"]), Ridge(),
                         reduction=ValidatedReduction("kpca", 2)).fit(df, np.arange(20))
    model, transformed, names = estimator_input(pipe, df)
    np.testing.assert_allclose(model.predict(transformed), pipe.predict(df))
    assert list(names) == ["kpca_component_1", "kpca_component_2"]


def test_shap_explanation_additivity():
    pytest.importorskip("shap")
    df = pd.DataFrame({"x": np.arange(12, dtype=float), "y": np.arange(12)%2})
    pipe = make_pipeline(build_preprocessor(["x", "y"], []), Ridge(),
                         reduction=ValidatedReduction("kpca", 2)).fit(df, df.x)
    result = explain_pipeline(pipe, df.iloc[:8], df.iloc[8:], max_samples=2, max_background=5)
    values = result["values"]
    np.testing.assert_allclose(values.base_values + values.values.sum(axis=1), pipe.predict(df.iloc[8:10]), atol=1e-6)
