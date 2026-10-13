"""Protocolos comparáveis: mesmos folds, tuning interno e teste externo reservado."""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from core.augmentation import make_sampler
from core.evaluate import make_pipeline, clf_metrics, reg_metrics
from core.experiment_tracking import manifest
from core.models import CLASSIFICATION, REGRESSION
from core.preprocess import build_preprocessor
from core.reduction import ValidatedReduction
from core.splitters import holdout_indices, cv_indices


def scenario_configs(task, protocol="generic", components=2, gamma=.1, sae_epochs=100):
    if protocol == "article3" and task != "classification":
        raise ValueError("Quatro cenários destinam-se à classificação.")
    if protocol == "article3":
        return [dict(id="E1", augmentation="none", reduction="none"),
                dict(id="E2", augmentation="sae", balance_fraction=.5, reduction="none", epochs=sae_epochs),
                dict(id="E3", augmentation="sae", balance_fraction=1., reduction="none", epochs=sae_epochs),
                dict(id="E4", augmentation="sae", balance_fraction=1., reduction="kpca", epochs=sae_epochs,
                     components=components, gamma=gamma)]
    return [dict(id="R1" if task == "regression" else "C1", augmentation="none", reduction="none"),
            dict(id="R2" if task == "regression" else "C2", augmentation="none", reduction="pca", components=components),
            dict(id="R3" if task == "regression" else "C3", augmentation="none", reduction="kpca", components=components, gamma=gamma)]


def ablation_configs(components=2, gamma=.1):
    rows = []
    for augmentation in ("none", "class_weight", "smote", "adasyn", "sae_partial", "sae_full"):
        for reduction in ("none", "pca", "kpca"):
            method = "sae" if augmentation.startswith("sae_") else augmentation
            rows.append(dict(id=f"{augmentation}_{reduction}", augmentation=method, reduction=reduction,
                             balance_fraction=.5 if augmentation == "sae_partial" else 1.,
                             components=components, gamma=gamma))
    return rows


def run_experiment(df, config, scenarios=None, model_names=None, k=3, test_size=.2,
                   tune=False, gap=0, progress=None):
    started = time.perf_counter()
    config.validate(df)
    if config.split_strategy == "random" and df.duplicated(subset=[*config.features, config.target]).any():
        raise ValueError("Linhas duplicadas podem atravessar folds aleatórios. Escolha grupos de origem ou deduplique explicitamente.")
    if not 2 <= k <= 10:
        raise ValueError("Número de folds deve estar entre 2 e 10.")
    if not 0 < test_size < 1:
        raise ValueError("Fração de teste inválida.")
    X, y = df[config.features], df[config.target]
    groups = df[config.group_column].to_numpy() if config.split_strategy == "group" else None
    times = df[config.time_column].to_numpy() if config.split_strategy == "time" else None
    tr, te = holdout_indices(y, config.task_type, test_size, config.random_state, groups, times, gap)
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    gt, tt = groups[tr] if groups is not None else None, times[tr] if times is not None else None
    folds = cv_indices(ytr, config.task_type, k, config.random_state, gt, tt, gap)
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in config.features if c not in numeric]
    zoo = CLASSIFICATION if config.task_type == "classification" else REGRESSION
    names = model_names or (["SVM (RBF)", "Random Forest"] if config.task_type == "classification" else ["SVR (RBF)", "Random Forest"])
    baseline = "Baseline (maioria)" if config.task_type == "classification" else "Baseline (média)"
    names = list(dict.fromkeys([baseline, *names]))
    if any(n not in zoo for n in names):
        raise ValueError("Modelo não pertence ao catálogo desta tarefa.")
    scenarios = scenarios or [dict(id="baseline", augmentation="none", reduction="none")]
    if len(scenarios)*len(names)*k > 600:
        raise ValueError("Experimento excede orçamento de 600 ajustes externos.")
    if len({s["id"] for s in scenarios}) != len(scenarios):
        raise ValueError("IDs de cenários devem ser únicos.")
    metric = clf_metrics if config.task_type == "classification" else reg_metrics
    key = "F1 (macro)" if config.task_type == "classification" else "R2"
    rows, detailed, specs = [], [], {}
    total = len(names)*len(scenarios)*len(folds)
    done = 0
    for scenario in scenarios:
        for name in names:
            label = f"{scenario['id']} / {name}"
            model, grid = zoo[name]
            model = clone(model)
            if "random_state" in model.get_params():
                model.set_params(random_state=config.random_state)
            method = scenario.get("augmentation", "none")
            if name == baseline:
                method = "none"
            if method == "class_weight":
                if config.task_type != "classification" or "class_weight" not in model.get_params():
                    raise ValueError("Modelo não suporta class_weight.")
                model.set_params(class_weight="balanced")
                method = "none"
            sampler_args = {"balance_fraction": scenario.get("balance_fraction", 1.),
                            "epochs": scenario.get("epochs", 100)} if method == "sae" else {}
            sampler = make_sampler(method, config.task_type, bool(categorical), config.random_state, **sampler_args)
            pre = build_preprocessor(numeric, categorical)
            reduction = ValidatedReduction(scenario.get("reduction", "none"), scenario.get("components", 2),
                                           scenario.get("gamma", .1), random_state=config.random_state)
            pipe = make_pipeline(pre, model, grid, False, task=config.task_type, sampler=sampler, reduction=reduction)
            scores, elapsed = [], 0.
            for fold_id, (ft, fv) in enumerate(folds):
                start = time.perf_counter()
                inner = cv_indices(ytr.iloc[ft], config.task_type, k, config.random_state,
                                   gt[ft] if gt is not None else None, tt[ft] if tt is not None else None, gap) if tune else k
                fold_pipe = make_pipeline(pre, model, grid, tune, inner, config.task_type, sampler, reduction)
                fold_pipe.fit(Xtr.iloc[ft], ytr.iloc[ft])
                score = metric(ytr.iloc[fv], fold_pipe.predict(Xtr.iloc[fv]))
                if not all(np.isfinite(list(score.values()))):
                    raise ValueError("Métricas não finitas; revise o tamanho dos folds e o alvo.")
                scores.append(score)
                elapsed += time.perf_counter()-start
                estimator = getattr(fold_pipe, "best_estimator_", fold_pipe)
                diag = getattr(estimator.named_steps.get("samp"), "diagnostics_", [])
                detailed.append({"model": label, "fold": fold_id, "metrics": score, "augmentation": diag})
                done += 1
                if progress:
                    progress(done/total, f"{label} · fold {fold_id+1}/{k}")
            means = {m: float(np.mean([s[m] for s in scores])) for m in scores[0]}
            deviations = {f"{m} (±)": float(np.std([s[m] for s in scores])) for m in scores[0]}
            rows.append({"Modelo": label, **means, **deviations, "tempo (s)": elapsed})
            specs[label] = (pre, model, grid, sampler, reduction)
    # Choose using validation only; evaluate exactly one winner on the external test.
    board = pd.DataFrame(rows).sort_values(key, ascending=False, kind="stable").reset_index(drop=True)
    best = board.loc[0, "Modelo"]
    pre, model, grid, sampler, reduction = specs[best]
    inner = cv_indices(ytr, config.task_type, k, config.random_state, gt, tt, gap) if tune else k
    winner = make_pipeline(pre, model, grid, tune, inner, config.task_type, sampler, reduction).fit(Xtr, ytr)
    pred = winner.predict(Xte)
    test = metric(yte, pred)
    protocol = {"task": config.to_dict(), "scenarios": scenarios, "models": names, "folds": k,
                "test_size": test_size, "tune": tune, "gap": gap,
                "methodology": "inspirado/sae_extended; percentuais configuráveis, não reprodução exata"}
    fold_ids = {"train": tr.tolist(), "test": te.tolist(),
                "validation": [{"train": tr[ft].tolist(), "validation": tr[fv].tolist()} for ft, fv in folds]}
    track = manifest(df, protocol, fold_ids, {"validation": detailed, "external_test": test})
    result = {"task": config.task_type, "target": config.target, "features": config.features,
              "model": winner, "best": best, "best_params": getattr(winner, "best_params_", {}),
              "leaderboard": board, "test_metrics": test, "manifest": track, "task_config": config.to_dict(),
              "n_train": len(tr), "n_test": len(te), "X_train": Xtr, "X_test": Xte,
              "y_true": np.asarray(yte), "y_pred": np.asarray(pred), "data": df,
              "log": ["Teste externo reservado; somente vencedor selecionado na CV é avaliado."],
              "adjustments": ["Ordem: imputação/IQR/escala/categorias → aumento → redução completa → modelo."],
              "profile": {"n": len(Xtr)}, "seed": config.random_state, "seconds": time.perf_counter()-started}
    if config.task_type == "classification":
        result["per_class"] = classification_report(yte, pred, output_dict=True, zero_division=0)
    return result
