import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.evaluate import cross_val


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("repeats", [1, 2])
def test_progress_preserves_validation_metrics(task, repeats):
    if task == "classification":
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        model = LogisticRegression()
    else:
        X, y = make_regression(n_samples=60, n_features=5, noise=5, random_state=42)
        model = Ridge()
    expected = cross_val(model, X, y, task, k=3, repeats=repeats)
    updates = []
    actual = cross_val(model, X, y, task, k=3, repeats=repeats,
                       progress=lambda p, text: updates.append((p, text)))
    for metric in expected:
        np.testing.assert_allclose(actual[metric], expected[metric])
    values = [p for p, _ in updates]
    assert values[0] == 0
    assert values[-1] == 1
    assert values == sorted(values)
    assert len(updates) == 2 * 3 * repeats
