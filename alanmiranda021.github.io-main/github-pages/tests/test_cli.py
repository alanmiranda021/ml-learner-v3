import json
import sys
import numpy as np
import pandas as pd
import pytest
from core.predict import load_bundle, predict_table


def test_yaml_cli_exports_reviewable_artifacts(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    from experiment_cli import main
    df = pd.DataFrame({"x": np.arange(40, dtype=float), "target": np.arange(40)*.25})
    data = tmp_path / "data.csv"
    config = tmp_path / "config.yaml"
    df.to_csv(data, index=False)
    config.write_text(yaml.safe_dump({"task": {"task_type": "regression", "target": "target", "features": ["x"]},
                                      "models": ["Ridge"], "folds": 2,
                                      "scenarios": [{"id": "R1", "reduction": "none", "augmentation": "none"}]}), encoding="utf-8")
    output = tmp_path / "artifacts"
    monkeypatch.setattr(sys, "argv", ["experiment_cli.py", str(data), "--config", str(config), "--output", str(output)])
    main()
    folder, = output.iterdir()
    assert {"manifest.json", "config.json", "metrics.csv", "report.md", "model.joblib"} <= {p.name for p in folder.iterdir()}
    assert not (folder / "predictions.csv").exists()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["folds"]["test"]) == 8
    bundle = load_bundle((folder / "model.joblib").read_bytes())
    assert np.isfinite(predict_table(bundle, df).previsto_target).all()
