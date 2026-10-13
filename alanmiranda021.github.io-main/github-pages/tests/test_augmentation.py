import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from core.augmentation import make_sampler
from core.evaluate import make_pipeline, holdout
from core.preprocess import build_preprocessor


class SpySampler(BaseEstimator):
    calls = []
    def fit_resample(self, X, y):
        type(self).calls.append(set(X[:, 0]))
        return X, y


def test_sampler_external_ids_and_inference():
    pytest.importorskip("imblearn")
    SpySampler.calls = []
    df = pd.DataFrame({"id_value": np.arange(40, dtype=float)})
    pre = build_preprocessor(["id_value"], [], scaler="Nenhum", winsor=False)
    pipe = make_pipeline(pre, LogisticRegression(), task="classification", sampler=SpySampler())
    _, extra = holdout(pipe, df, np.array([0, 1]*20), "classification")
    assert len(SpySampler.calls) == 1
    assert not SpySampler.calls[0] & set(df.iloc[extra["test_indices"]].id_value)
    pipe.predict(df)
    assert len(SpySampler.calls) == 1


def test_incompatible_augmentation():
    with pytest.raises(ValueError):
        make_sampler("smote", task="regression")
    with pytest.raises(ValueError):
        make_sampler("sae", categorical=True)
