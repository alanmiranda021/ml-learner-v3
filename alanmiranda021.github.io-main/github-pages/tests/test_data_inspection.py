import io
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from core.data_inspection import inspect_mat, mat_to_table, profile_table
from core.task_config import TaskConfig, suggest_task
from core.io import load_table


def test_numeric_few_values_remains_regression():
    df = pd.DataFrame({"x": range(6), "y": [1., 2.] * 3})
    assert suggest_task(df.y)[0] == "regression"
    assert TaskConfig("regression", "y", ["x"]).validate(df)
    with pytest.raises(ValueError):
        TaskConfig("regression", "y", ["y"]).validate(df)


def test_mat_inspection_and_explicit_orientation():
    buf = io.BytesIO()
    savemat(buf, {"a": np.arange(12).reshape(3, 4), "b": np.ones((2, 2)),
                  "struct": {"x": [1, 2]}, "cube": np.ones((2, 2, 2))})
    raw = buf.getvalue()
    assert len(inspect_mat(raw)["variables"]) == 4
    assert mat_to_table(raw, "a", "columns").shape == (4, 3)
    with pytest.raises(ValueError):
        mat_to_table(raw, "struct")
    with pytest.raises(ValueError):
        inspect_mat(b"corrupt")


def test_mat_v73(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "data.mat"
    with h5py.File(p, "w") as h:
        h["data"] = np.arange(12).reshape(3, 4)
    raw = p.read_bytes()
    assert inspect_mat(raw)["format"] == "v7.3"
    assert mat_to_table(raw, "data").shape == (3, 4)


def test_loaders_and_profile():
    df = load_table("d.csv", b"x;y\n1;2\n3;4\n")
    assert df.shape == (2, 2)
    out = io.BytesIO()
    df.to_excel(out, index=False)
    pd.testing.assert_frame_equal(df, load_table("d.xlsx", out.getvalue()))
    assert profile_table(pd.DataFrame({"x": [1., np.inf, np.nan]}))["columns"][0]["infinitos"] == 1
    with pytest.raises(ValueError):
        load_table("d.exe", b"")
