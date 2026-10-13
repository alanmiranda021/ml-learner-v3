import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from core.splitters import holdout_indices, cv_indices


class Spy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.seen_ = set(X[:, 0])
        return self

    def transform(self, X):
        return X


def test_external_ids_never_fitted():
    X = np.column_stack([np.arange(40), np.ones(40)])
    y = np.arange(40)
    tr, te = holdout_indices(y, "regression")
    pipe = Pipeline([("spy", Spy()), ("model", Ridge())]).fit(X[tr], y[tr])
    pipe.predict(X[te])
    assert not pipe["spy"].seen_ & set(X[te, 0])


def test_group_and_time_invariants():
    y = np.arange(40)
    groups = np.repeat(np.arange(8), 5)
    for tr, va in cv_indices(y, "regression", k=4, groups=groups):
        assert not set(groups[tr]) & set(groups[va])
    tr, te = holdout_indices(y, "regression", groups=groups)
    assert not set(groups[tr]) & set(groups[te])
    times = np.repeat(np.arange(20), 2)[::-1]
    for tr, va in cv_indices(y, "regression", k=3, times=times, gap=1):
        assert times[tr].max() < times[va].min()
    tr, te = holdout_indices(y, "regression", times=times)
    assert times[tr].max() < times[te].min()
