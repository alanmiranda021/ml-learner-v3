import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from core.preprocess import build_preprocessor
from core.reduction import ValidatedReduction
from core.evaluate import make_pipeline, cross_val


@pytest.mark.parametrize("method", ["none", "pca", "kpca"])
def test_reduction_inside_folds_with_categories(method):
    df = pd.DataFrame({"x": np.arange(30, dtype=float), "cat": ["a", "b"]*15})
    df.loc[0, "x"] = np.nan
    pre = build_preprocessor(["x"], ["cat"])
    pipe = make_pipeline(pre, Ridge(), reduction=ValidatedReduction(method, 2))
    scores = cross_val(pipe, df, np.arange(30), "regression", k=3)
    assert np.isfinite(scores["MAE"]).all()
    pipe.fit(df, np.arange(30))
    assert pipe.predict(df[:2]).shape == (2,)
    assert pipe["reduce"].n_features_in_ == 3


def test_invalid_components_gamma_and_memory():
    for reducer in [ValidatedReduction("kpca", 10), ValidatedReduction("kpca", 2, -1),
                    ValidatedReduction("pca", 3), ValidatedReduction("kpca", 2, max_samples=2)]:
        with pytest.raises(ValueError):
            reducer.fit(np.ones((5, 2)))


def test_constant_repeated_samples():
    reducer = ValidatedReduction("kpca", 2).fit(np.ones((8, 3)))
    assert np.isfinite(reducer.transform(np.ones((2, 3)))).all()
