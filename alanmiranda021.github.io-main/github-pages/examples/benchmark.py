"""Benchmark sintético reproduzível; nunca representar como resultado em dados reais."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification, make_regression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.task_config import TaskConfig
from core.experiments import run_experiment, scenario_configs
from core.experiment_tracking import json_bytes
from core.auto import report_md
from core.predict import make_bundle, to_bytes


def main():
    destination = ROOT / "examples" / "results"
    destination.mkdir(exist_ok=True)
    summary = []
    for task in ("classification", "regression"):
        if task == "classification":
            X, y = make_classification(n_samples=120, n_features=4, n_informative=3,
                                       n_redundant=0, weights=[.65, .35], random_state=17)
            scenarios = scenario_configs(task, "article3", components=2, sae_epochs=10)
            models = ["SVM (RBF)", "Random Forest"]
        else:
            X, y = make_regression(n_samples=120, n_features=4, noise=10, random_state=17)
            scenarios = scenario_configs(task, components=2)
            models = ["Ridge", "SVR (RBF)", "Random Forest"]
        frame = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
        frame["target"] = y
        cfg = TaskConfig(task, "target", ["f1", "f2", "f3", "f4"], random_state=42)
        r = run_experiment(frame, cfg, scenarios, models, k=2)
        (destination / f"{task}_manifest.json").write_bytes(json_bytes(r["manifest"]))
        r["leaderboard"].to_csv(destination / f"{task}_metrics.csv", index=False)
        (destination / f"{task}_report.md").write_text(report_md(r), encoding="utf-8")
        bundle = make_bundle(r, r["model"], "treino externo sintético; teste preservado", schema_data=r["X_train"])
        bundle["n_rows"] = r["n_train"]
        (destination / f"{task}_model.joblib").write_bytes(to_bytes(bundle))
        summary.append(dict(task=task, best=r["best"], metrics=r["test_metrics"],
                            train=r["n_train"], test=r["n_test"], seconds=r["seconds"]))
    (destination / "summary.json").write_bytes(json_bytes(summary))
    print(json_bytes(summary).decode("utf-8"))


if __name__ == "__main__":
    main()
