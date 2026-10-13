import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from core.evaluate import make_pipeline
from core.preprocess import build_preprocessor
from core.reduction import ValidatedReduction
from core.predict import make_bundle, to_bytes, load_bundle, predict_table


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_bundle_roundtrip_categories_missing_and_kpca(task):
    df = pd.DataFrame({"x": np.arange(30, dtype=float), "cat": ["a", "b"]*15})
    df.loc[0, "x"] = np.nan
    y = np.arange(30)%2 if task == "classification" else np.arange(30)
    df["target"] = y
    model = LogisticRegression() if task == "classification" else Ridge()
    pipe = make_pipeline(build_preprocessor(["x"], ["cat"]), model, task=task,
                         reduction=ValidatedReduction("kpca", 2)).fit(df, y)
    r = dict(data=df, model=pipe, features=["x", "cat"], task=task, target="target", best="test", test_metrics={})
    bundle = load_bundle(to_bytes(make_bundle(r, pipe)))
    out = predict_table(bundle, df[["cat", "x"]])
    np.testing.assert_allclose(out.previsto_target, pipe.predict(df))
    with pytest.raises(ValueError):
        predict_table(bundle, df[["cat"]])
    with pytest.raises(ValueError):
        predict_table(bundle, df.assign(x="oops"))
