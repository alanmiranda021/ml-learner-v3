"""Protocolo do Artigo 3 adaptado: regressão naval exploratória, sem torch.

Preparação em cache por partição, só durante a execução, para reutilizar o SAE
entre modelos e sorteios. Cada dobra interna tem seu próprio ajuste completo.
"""
from dataclasses import dataclass, asdict
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterSampler, LeaveOneGroupOut
from core.augment import SAESampler, cell_labels
from core.evaluate import reg_metrics, cell_cv_indices
from core.experiment_tracking import manifest
from core.models import naval_model
from core.preprocess import build_preprocessor
from core.reduction import ValidatedReduction

FEATURES = ("ship_type", "route_id", "distance", "fuel_type", "weather_conditions", "engine_efficiency")
SCENARIOS = {"E1": (0., False), "E2": (.9, False), "E3": (1., False),
             "E4": (1., True), "E4b": (0., True), "E3-orig": (1., False),
             "E2-gaussian": (.9, False), "E3-gaussian": (1., False)}
LIMITATIONS = [
    "Resultados exploratórios: protocolo do Artigo 3 adaptado, tarefa e métrica diferentes.",
    "Unidades precisam de confirmação antes da interpretação física.",
    "Disponibilidade de engine_efficiency no instante da previsão precisa de confirmação.",
    "Sem ID de embarcação nem data: não há validação por embarcação ou temporal.",
    "Possível origem sintética dos dados; não transferir conclusões para uma frota real.",
    "SAE não garante novidade, coerência física ou melhora; plausibilidade é uma heurística.",
    "Dobras compartilham observações: não são usados valores-p.",
    "Beta=1, rho=.05, neurônios, otimizador e geração são suposições/adaptações do projeto.",
    "E3-orig aumenta antes da divisão: validação possivelmente otimista, excluída do ranking.",
]


@dataclass
class NavalConfig:
    target: str = "fuel_consumption"
    excluded: tuple = ("CO2_emissions",)
    cell_columns: tuple = ("ship_type", "route_id")
    mode: str = "rapido"
    partial_fraction: float = .9
    hidden_grid: tuple = (3, 4, 5, 6)
    gamma_grid: tuple = (.01, .1, 1., 10., 100.)
    components_grid: tuple = (5, 8, 11)
    repeats: int = 3
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 5
    draws: int = 0
    maxiter: int = 0
    models: tuple = ("SVR (RBF)", "Random Forest")
    scenarios: tuple = tuple(SCENARIOS)
    sensitivity: bool = True
    units: str = "Não confirmadas"
    filename: str = "tabela carregada"
    source_sha256: str = ""

    def effective(self):
        cfg = NavalConfig(**asdict(self))
        if cfg.mode not in ("rapido", "completo"):
            raise ValueError("Modo deve ser rapido ou completo.")
        if cfg.mode == "rapido":
            cfg.cell_columns = ("ship_type",)
            cfg.repeats = 1
        cfg.draws = cfg.draws or (8 if cfg.mode == "rapido" else 20)
        cfg.maxiter = cfg.maxiter or (50 if cfg.mode == "rapido" else 400)
        if cfg.mode == "rapido" and cfg.maxiter > 50:
            raise ValueError("Modo rápido permite no máximo 50 iterações SAE.")
        return cfg


def validate_naval(df, cfg):
    if df.empty:
        raise ValueError("Tabela vazia.")
    if cfg.target not in ("fuel_consumption", "CO2_emissions") or cfg.target not in df:
        raise ValueError("Alvo naval ausente ou não suportado.")
    forbidden = "CO2_emissions" if cfg.target == "fuel_consumption" else "fuel_consumption"
    if forbidden not in cfg.excluded:
        raise ValueError(f"Vazamento: {forbidden} deve ser excluída para prever {cfg.target}.")
    features = [c for c in FEATURES if c not in cfg.excluded]
    if set(features) != set(FEATURES) or any(c not in df for c in features):
        raise ValueError("O protocolo exige os seis preditores navais definidos no documento.")
    if not cfg.cell_columns or not set(cfg.cell_columns).issubset({"ship_type", "route_id"}):
        raise ValueError("Células devem usar ship_type e/ou route_id.")
    y = df[cfg.target]
    if not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y.to_numpy(float)).all() or (y <= 0).any():
        raise ValueError("Alvo deve ser numérico, finito e estritamente positivo.")
    if df[list(cfg.cell_columns)].isna().any().any():
        raise ValueError("Células não podem ter valores ausentes.")
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in ("distance", "engine_efficiency")):
        raise ValueError("distance e engine_efficiency devem ser numéricas.")
    if (not 0 < cfg.partial_fraction <= 1 or cfg.repeats < 1 or cfg.draws < 1 or cfg.maxiter < 1
            or min(cfg.outer_folds, cfg.inner_folds) < 2 or not cfg.models or not cfg.scenarios
            or not cfg.hidden_grid or not cfg.gamma_grid or not cfg.components_grid):
        raise ValueError("Configuração naval inválida ou grades vazias.")
    if any(s not in SCENARIOS for s in cfg.scenarios) or "E1" not in cfg.scenarios:
        raise ValueError("Cenários desconhecidos ou E1 ausente.")
    counts = pd.Series(cell_labels(df, cfg.cell_columns)).value_counts()
    if counts.min() < 15:
        warnings.warn("Célula com poucas linhas; dobras internas ou aumento podem falhar.", UserWarning)
    if df.duplicated(subset=[*features, cfg.target]).any():
        raise ValueError("Linhas duplicadas: deduplique explicitamente antes da divisão aleatória.")
    return features


class NavalReduction(ValidatedReduction):
    def fit(self, X, y=None):
        if self.method == "kpca" and self.n_components >= X.shape[1]:
            raise ValueError("KPCA exige n_components menor que a dimensão de entrada.")
        return super().fit(X, y)


def paired_comparisons(details, expected_folds=None):
    rows = []
    for name in sorted({d["model"] for d in details}):
        base = {d["fold"]: d["metrics"]["RMSE"] for d in details if d["model"] == name and d["scenario"] == "E1"}
        for scenario in sorted({d["scenario"] for d in details} - {"E1", "E3-orig"}):
            folds = [d for d in details if d["model"] == name and d["scenario"] == scenario and d["fold"] in base]
            if not folds:
                continue
            delta = np.array([d["metrics"]["RMSE"]-base[d["fold"]] for d in folds])
            fraction = float(np.mean(delta < 0))
            complete = len(folds) == len(base) and (expected_folds is None or len(folds) == expected_folds)
            rows.append(dict(scenario=scenario, model=name, delta_rmse=float(delta.mean()),
                std=float(delta.std()), fraction_improved=fraction, complete=complete,
                conclusion="melhora em pelo menos 80% das dobras" if complete and fraction >= .8 and delta.mean() < 0 else "sem ganho demonstrado"))
    return rows


def rank_results(rows):
    eligible = [r for r in rows if r["scenario"] != "E3-orig" and r.get("complete", True)]
    if not eligible:
        return []
    ordered, remaining = [], sorted(eligible, key=lambda r: r["RMSE"])
    while remaining:
        reference = remaining[0]
        tied = [r for r in remaining if r["RMSE"]-reference["RMSE"] <= max(reference["RMSE_std"], r["RMSE_std"])]
        tied.sort(key=lambda r: (r["RMSE_std"], r["seconds"], r["RMSE"]))
        ordered.extend([{**r, "technical_tie_group": len(ordered)+1} for r in tied])
        remaining = [r for r in remaining if r not in tied]
    return ordered


def run_naval(df, config=None, progress=None):
    from core.env import IN_BROWSER
    started = time.perf_counter()
    original = config or NavalConfig()
    cfg = original.effective()
    if IN_BROWSER and cfg.mode != "rapido":
        raise ValueError("No navegador está disponível apenas o modo rápido; execute o completo no Python local.")
    features = validate_naval(df, cfg)
    source = df.reset_index(drop=True).copy()
    labels = cell_labels(source, cfg.cell_columns)
    tr, te = train_test_split(np.arange(len(source)), test_size=.2, stratify=labels, random_state=cfg.seed)
    dev, final = source.iloc[tr], source.iloc[te]
    def splits(frame, k, seed):
        return cell_cv_indices(cell_labels(frame, cfg.cell_columns), k=k, seed=seed)
    folds = [(r, i, a, b) for r in range(cfg.repeats)
             for i, (a, b) in enumerate(splits(dev, cfg.outer_folds, cfg.seed+r))]
    cache, reduction_cache = {}, {}
    def spec(scenario):
        fraction, kpca = SCENARIOS[scenario]
        if scenario.startswith("E2"):
            fraction = cfg.partial_fraction
        return fraction, kpca, "gaussian" if scenario.endswith("gaussian") else "sae"
    def sampler(fraction, generator):
        return SAESampler(cell_columns=cfg.cell_columns, target=cfg.target, fraction=fraction,
            hidden_grid=cfg.hidden_grid, maxiter=cfg.maxiter, random_state=cfg.seed, generator=generator)
    def prepare(train, validation, scenario, contaminated=False):
        fraction, _, generator = spec(scenario)
        if contaminated:
            fraction = 0.
        key = (tuple(train.index), tuple(validation.index), fraction, generator, contaminated)
        if key not in cache:
            X, y = train[features].copy(), train[cfg.target].to_numpy()
            numeric = ["distance", "engine_efficiency"]
            cats = [c for c in features if c not in numeric]
            pre = build_preprocessor(numeric, cats, winsor=False).fit(X)
            # Raw imputations used by the sampler also fit only this training partition.
            fill = {c: (X[c].median() if c in numeric else X[c].mode().iloc[0]) for c in features if X[c].isna().any()}
            X = X.fillna(fill)
            aug = sampler(fraction, generator) if fraction else None
            xa, ya = aug.fit_resample(X, y) if aug else (X, y)
            cache[key] = (pre.transform(xa), ya, pre.transform(validation[features]), pre, aug)
            if len(cache) > 32:
                del cache[next(iter(cache))]
        return key, cache[key]
    def transformed(key, prepared, params, kpca):
        a, y, b, pre, aug = prepared
        n, gamma = params.get("kpca_components", 0), params.get("kpca_gamma", .1)
        rk = (key, n, gamma)
        if kpca:
            if rk not in reduction_cache:
                reduction = NavalReduction("kpca", n, gamma, random_state=cfg.seed).fit(a)
                reduction_cache[rk] = (reduction.transform(a), reduction.transform(b), reduction)
                if len(reduction_cache) > 12:
                    del reduction_cache[next(iter(reduction_cache))]
            aa, bb, reduction = reduction_cache[rk]
        else:
            aa, bb, reduction = a, b, None
        return aa, y, bb, pre, aug, reduction
    failures = []
    def search(train, validation, scenario, name, contaminated=False):
        model, grid = naval_model(name, cfg.seed)
        kpca = spec(scenario)[1]
        if kpca:
            grid = {**grid, "kpca_components": list(cfg.components_grid), "kpca_gamma": list(cfg.gamma_grid)}
        candidates = list(ParameterSampler(grid, n_iter=cfg.draws, random_state=cfg.seed))
        internal = splits(train, cfg.inner_folds, cfg.seed)
        best_score, best_params = np.inf, None
        for params in candidates:
            scores = []
            try:
                for a, b in internal:
                    key, ready = prepare(train.iloc[a], train.iloc[b], scenario, contaminated)
                    xa, ya, xb, _, _, _ = transformed(key, ready, params, kpca)
                    fitted = clone(model).set_params(**{k: v for k, v in params.items() if not k.startswith("kpca_")}).fit(xa, ya)
                    scores.append(reg_metrics(train.iloc[b][cfg.target], fitted.predict(xb))["RMSE"])
                score = np.mean(scores)
                if np.isfinite(score) and score < best_score:
                    best_score, best_params = score, params
            except (ValueError, FloatingPointError) as exc:
                failures.append(dict(scenario=scenario, model=name, stage="candidato interno", error=str(exc), params=str(params)))
        if best_params is None:
            raise ValueError("Todos os candidatos falharam; veja falhas e dimensões KPCA.")
        key, ready = prepare(train, validation, scenario, contaminated)
        xa, ya, xb, pre, aug, reduction = transformed(key, ready, best_params, kpca)
        fitted = clone(model).set_params(**{k: v for k, v in best_params.items() if not k.startswith("kpca_")}).fit(xa, ya)
        pred = fitted.predict(xb)
        evidence = dict(params=best_params, inner_rmse=float(best_score),
                        augmentation=aug.diagnostics_ if aug else [],
                        train_indices=train.index.tolist(), validation_indices=validation.index.tolist(),
                        eigenvalues=reduction.reducer_.eigenvalues_.tolist() if reduction else [])
        return pred, evidence
    rows, details, diagnostic_source = [], [], None
    total = len(cfg.scenarios)*len(cfg.models)*len(folds)
    done = 0
    for scenario in cfg.scenarios:
        contaminated = scenario == "E3-orig"
        if contaminated:
            try:
                aug = sampler(1., "sae")
                x, y = aug.fit_resample(dev[features], dev[cfg.target])
                diagnostic_source = x.assign(**{cfg.target: y})
                # Separate index namespace prevents cache aliasing with real partitions.
                diagnostic_source.index = np.arange(len(diagnostic_source)) + len(source)
            except Exception as exc:
                failures.append(dict(scenario=scenario, stage="aumento antes da divisão", error=str(exc)))
                done += len(cfg.models)*len(folds)
                continue
        frame = diagnostic_source if contaminated else dev
        used_folds = [(r, i, a, b) for r in range(cfg.repeats)
                      for i, (a, b) in enumerate(splits(frame, cfg.outer_folds, cfg.seed+r))] if contaminated else folds
        for name in cfg.models:
            scores, elapsed = [], time.perf_counter()
            for repeat, fold, a, b in used_folds:
                try:
                    pred, evidence = search(frame.iloc[a], frame.iloc[b], scenario, name, contaminated)
                    metrics = reg_metrics(frame.iloc[b][cfg.target], pred)
                    if not all(np.isfinite(list(metrics.values()))):
                        raise ValueError("Métricas não finitas nesta dobra.")
                    scores.append(metrics)
                    details.append(dict(scenario=scenario, model=name, fold=repeat*cfg.outer_folds+fold,
                                        metrics=metrics, **evidence))
                except Exception as exc:
                    failures.append(dict(scenario=scenario, model=name, fold=repeat*cfg.outer_folds+fold, error=str(exc)))
                done += 1
                if progress:
                    progress(.85*done/total, f"{scenario} / {name} · dobra {fold+1}, repetição {repeat+1}")
            if scores:
                rows.append(dict(scenario=scenario, model=name, complete=len(scores) == len(used_folds),
                    folds=len(scores), seconds=time.perf_counter()-elapsed,
                    **{m: float(np.mean([s[m] for s in scores])) for m in ("RMSE", "MAE", "R2")},
                    **{m+"_std": float(np.std([s[m] for s in scores])) for m in ("RMSE", "MAE", "R2")}))
    ranking = rank_results(rows)
    # Choose the validation winner, then unlock final test exactly once.
    test, predictions, sensitivity = None, None, []
    if ranking:
        winner = ranking[0]
        try:
            pred, evidence = search(dev, final, winner["scenario"], winner["model"])
            test = dict(scenario=winner["scenario"], model=winner["model"], metrics=reg_metrics(final[cfg.target], pred), **evidence)
            predictions = dict(actual=final[cfg.target].tolist(), predicted=pred.tolist(),
                               cells=cell_labels(final, cfg.cell_columns).tolist(), indices=te.tolist())
        except Exception as exc:
            failures.append(dict(stage="teste final", error=str(exc)))
        if cfg.sensitivity:
            # Development data only: sensitivity never reuses the final holdout.
            for column in ("route_id", "ship_type"):
                for a, b in LeaveOneGroupOut().split(dev, groups=dev[column]):
                    for scenario in dict.fromkeys(["E1", winner["scenario"]]):
                        try:
                            pred, _ = search(dev.iloc[a], dev.iloc[b], scenario, winner["model"])
                            sensitivity.append(dict(group_column=column, held_out=str(dev.iloc[b][column].iloc[0]),
                                scenario=scenario, model=winner["model"], **reg_metrics(dev.iloc[b][cfg.target], pred)))
                        except Exception as exc:
                            failures.append(dict(stage="sensibilidade", group_column=column, scenario=scenario, error=str(exc)))
    audit = {"rows": len(source), "missing": int(source.isna().sum().sum()),
             "duplicates": int(source.duplicated().sum()),
             "cells": pd.Series(labels).value_counts().to_dict(),
             "dictionary": [{"column": c, "dtype": str(source[c].dtype), "unique": int(source[c].nunique())} for c in source]}
    if {"CO2_emissions", "fuel_consumption"}.issubset(source):
        ratio = source.CO2_emissions/source.fuel_consumption
        audit["co2_fuel_ratio"] = dict(min=float(ratio.min()), max=float(ratio.max()), mean=float(ratio.mean()), std=float(ratio.std()))
        audit["co2_fuel_correlation"] = float(source.CO2_emissions.corr(source.fuel_consumption))
    audit["correlations"] = source.select_dtypes("number").corr()[cfg.target].dropna().to_dict()
    duration = time.perf_counter()-started
    protocol = dict(requested=asdict(original), effective=asdict(cfg), features=features,
                    final_train_indices=tr.tolist(), final_test_indices=te.tolist(),
                    final_test_synthetic=0, final_test_evaluations=int(test is not None),
                    sampler_scope="treino de cada dobra interna/externa; E3-orig é diagnóstico contaminado")
    provenance = manifest(source, protocol, [{"repeat": r, "fold": i, "train": tr[a].tolist(), "validation": tr[b].tolist()} for r, i, a, b in folds], test["metrics"] if test else {})
    result = dict(protocol="protocolo do Artigo 3 adaptado", exploratory=True, config=protocol,
        validation=rows, ranking=ranking, folds=details, comparisons=paired_comparisons(details, len(folds)),
        test=test, predictions=predictions, sensitivity=sensitivity, failures=failures,
        audit=audit, limitations=LIMITATIONS, manifest=provenance, seconds=duration,
        source_sha256=cfg.source_sha256, dataset_sha256=provenance["dataset_hash"])
    if progress:
        progress(1., "Concluído; preparando relatório")
    return result
