import numpy as np
import pandas as pd
import pytest
import core.auto as auto
from core.task_config import TaskConfig


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_auto_preserves_explicit_task_and_predictors(task, monkeypatch):
    names = ["Baseline (média)", "Ridge"] if task == "regression" else ["Baseline (maioria)", "Regressão Logística"]
    monkeypatch.setattr(auto, "ORDER", names)
    monkeypatch.setattr(auto, "FAST", set(names))
    x = np.arange(80, dtype=float)
    df = pd.DataFrame({"x": x, "z": np.sin(x), "target": x%2 if task == "classification" else x*.5})
    cfg = TaskConfig(task, "target", ["x", "z"])
    r = auto.run_auto(df, task_config=cfg, budget_s=30)
    assert r["task"] == task
    assert "x" in r["features"]  # high correlation is not automatic leakage exclusion
    assert len(r["X_test"]) == 16
    assert not set(r["train_indices"]) & set(r["test_indices"])


def test_auto_feature_filter_uses_only_training(monkeypatch):
    monkeypatch.setattr(auto, "ORDER", ["Ridge"])
    monkeypatch.setattr(auto, "FAST", {"Ridge"})
    df = pd.DataFrame({"x": np.arange(80, dtype=float), "train_constant": [0.]*64+list(np.arange(16)),
                       "time": np.arange(80), "target": np.arange(80)*.5})
    cfg = TaskConfig("regression", "target", ["x", "train_constant"], time_column="time", split_strategy="time")
    r = auto.run_auto(df, task_config=cfg, budget_s=30)
    assert r["features"] == ["x"]
