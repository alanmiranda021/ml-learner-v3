import numpy as np
import pandas as pd
from core.task_config import TaskConfig
from core.experiments import run_experiment
from core.splitters import cv_indices


def test_nested_group_cv_and_manifest():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=80), "group": np.repeat(np.arange(20), 4)})
    df["target"] = 3*df.x + rng.normal(size=80)
    config = TaskConfig("regression", "target", ["x"], group_column="group", split_strategy="group")
    result = run_experiment(df, config, model_names=["Ridge"], k=2, tune=True)
    f = result["manifest"]["folds"]
    assert not set(df.group.iloc[f["train"]]) & set(df.group.iloc[f["test"]])
    for fold in f["validation"]:
        assert not set(df.group.iloc[fold["train"]]) & set(df.group.iloc[fold["validation"]])
    assert "model__alpha" in result["best_params"]


def test_temporal_iso_strings_not_lexicographic():
    times = ["2025-01-01T00:00:00-03:00", "2025-01-01T01:00:00Z", "2025-01-02T00:00:00Z",
             "2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z", "2025-01-05T00:00:00Z"]
    splits = cv_indices(np.arange(6), "regression", k=2, times=times)
    first_train = splits[0][0]
    assert list(first_train) == [1, 0]
