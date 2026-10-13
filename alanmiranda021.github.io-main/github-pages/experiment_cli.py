"""Executa configurações YAML locais e exporta artefatos sem dados brutos."""
import argparse
from pathlib import Path
from core.io import load_table
from core.task_config import TaskConfig
from core.experiments import run_experiment, scenario_configs, ablation_configs
from core.experiment_tracking import json_bytes
from core.auto import report_md
from core.predict import make_bundle, to_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--variable", help="Variável MATLAB explicitamente selecionada")
    parser.add_argument("--orientation", choices=["rows", "columns"], default="rows")
    parser.add_argument("--output", type=Path, default=Path("experiments_output"))
    parser.add_argument("--export-predictions", action="store_true", help="Exportar alvo real e previsão do teste")
    args = parser.parse_args()
    import yaml
    options = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    task = dict(options["task"])
    if args.target:
        task["target"] = args.target
    if args.features:
        task["features"] = args.features
    if not task.get("target") or not task.get("features"):
        parser.error("Defina alvo e preditores no YAML ou com --target e --features; não serão deduzidos.")
    cfg = TaskConfig(**task)
    df = load_table(args.dataset.name, args.dataset.read_bytes(), args.variable, args.orientation)
    protocol = options.get("protocol", "generic")
    scenarios = options.get("scenarios")
    if scenarios is None:
        scenarios = ablation_configs(options.get("components", 2), options.get("gamma", .1)) if protocol == "ablation" else \
                    scenario_configs(cfg.task_type, protocol, options.get("components", 2), options.get("gamma", .1), options.get("sae_epochs", 100))
    result = run_experiment(df, cfg, scenarios, options.get("models"), options.get("folds", 3),
                            options.get("test_size", .2), options.get("tune", False), options.get("gap", 0))
    destination = args.output / result["manifest"]["experiment_id"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_bytes(json_bytes(result["manifest"]))
    (destination / "config.json").write_bytes(json_bytes(result["manifest"]["config"]))
    (destination / "report.md").write_text(report_md(result), encoding="utf-8")
    result["leaderboard"].to_csv(destination / "metrics.csv", index=False)
    bundle = make_bundle(result, result["model"], "treino externo; teste preservado", schema_data=result["X_train"])
    bundle["n_rows"] = result["n_train"]
    (destination / "model.joblib").write_bytes(to_bytes(bundle))
    if args.export_predictions:
        import pandas as pd
        pd.DataFrame({"real": result["y_true"], "previsto": result["y_pred"]}).to_csv(destination / "predictions.csv", index=False)
    print(f"Experimento concluído: {destination}")
    print(result["test_metrics"])


if __name__ == "__main__":
    main()
