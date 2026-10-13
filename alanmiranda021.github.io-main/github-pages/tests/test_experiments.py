import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from core.experiments import run_experiment, scenario_configs
from core.task_config import TaskConfig


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generic_experiment_and_reserved_test(task):
    maker = make_classification if task == "classification" else make_regression
    X, y = maker(n_samples=80, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["target"] = y
    models = ["Regressão Logística"] if task == "classification" else ["Ridge"]
    result = run_experiment(df, TaskConfig(task, "target", list("abcd")),
                            scenario_configs(task), models, k=2)
    assert len(result["leaderboard"]) == 6
    folds = result["manifest"]["folds"]
    for fold in folds["validation"]:
        assert not set(fold["train"]) & set(folds["test"])
        assert not set(fold["validation"]) & set(folds["test"])
    assert all(np.isfinite(list(result["test_metrics"].values())))


def test_four_scenarios_real_sae():
    pytest.importorskip("imblearn")
    X, y = make_classification(n_samples=100, n_features=4, weights=[.65, .35], random_state=12)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["target"] = y
    r = run_experiment(df, TaskConfig("classification", "target", list("abcd")),
                       scenario_configs("classification", "article3", sae_epochs=5),
                       ["SVM (RBF)", "Random Forest"], k=2)
    assert len(r["leaderboard"]) == 12
    assert any(d["augmentation"] for d in r["manifest"]["metrics"]["validation"])
