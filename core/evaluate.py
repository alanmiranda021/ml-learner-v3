"""Estratégias de divisão e métricas — sem vazamento.

Suporta: Hold-out, Hold-out 3-way, K-fold, K-fold repetido, GroupKFold,
StratifiedGroupKFold, TimeSeriesSplit e CV aninhada.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline

REG_SCORING = {"R2": "r2",
               "RMSE": "neg_root_mean_squared_error",
               "MAE": "neg_mean_absolute_error"}
CLF_SCORING = {"Acurácia": "accuracy",
               "F1 (macro)": "f1_macro",
               "Precisão (macro)": "precision_macro",
               "Recall (macro)": "recall_macro"}


# ---------------------------------------------------------------------------
# Construção de pipelines (com ou sem sampler)
# ---------------------------------------------------------------------------
def make_pipeline(pre, model, grid=None, tune: bool = False, cv_inner: int = 5,
                  task: str = "regression", sampler=None):
    """Monta Pipeline (ou ImbPipeline se houver sampler) e envolve em GridSearchCV."""
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([("pre", clone(pre)), ("samp", sampler),
                            ("model", clone(model))])
    else:
        pipe = Pipeline([("pre", clone(pre)), ("model", clone(model))])
    if tune and grid:
        scoring = "r2" if task == "regression" else "f1_macro"
        pipe = GridSearchCV(pipe,
                            {f"model__{k}": v for k, v in grid.items()},
                            cv=cv_inner, scoring=scoring, n_jobs=-1,
                            refit=True)
    return pipe


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def reg_metrics(y, p):
    return {"R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}


def clf_metrics(y, p):
    return {"Acurácia": float(accuracy_score(y, p)),
            "F1 (macro)": float(f1_score(y, p, average="macro")),
            "Precisão (macro)": float(precision_score(y, p, average="macro",
                                                       zero_division=0)),
            "Recall (macro)": float(recall_score(y, p, average="macro",
                                                  zero_division=0))}


# ---------------------------------------------------------------------------
# Hold-out simples
# ---------------------------------------------------------------------------
def holdout(pipe, X, y, task, test_size: float = 0.2, seed: int = 42):
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=strat)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    m = reg_metrics(yte, pred) if task == "regression" else clf_metrics(yte, pred)
    extra = {"y_true": np.asarray(yte), "y_pred": np.asarray(pred), "model": pipe,
             "X_test": Xte.copy(), "y_test": yte.copy()}
    if task == "classification":
        labels = sorted(pd.unique(y))
        extra["cm"] = confusion_matrix(yte, pred, labels=labels)
        extra["labels"] = labels
        if hasattr(pipe, "predict_proba"):
            try:
                extra["y_proba"] = pipe.predict_proba(Xte)
            except (AttributeError, TypeError, ValueError):
                # Alguns classificadores expõem a API, mas não conseguem
                # produzir probabilidades nesta configuração.
                pass
    if hasattr(pipe, "best_params_"):
        extra["best_params"] = pipe.best_params_
    return m, extra


# ---------------------------------------------------------------------------
# Hold-out 3-way (treino/validação/teste) — slide 4
# ---------------------------------------------------------------------------
def holdout_3way(pipe, X, y, task, val_size: float = 0.1, test_size: float = 0.2,
                 seed: int = 42):
    strat = y if task == "classification" else None
    X_tmp, Xte, y_tmp, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat)
    # Proporção da validação DENTRO do restante
    val_rel = val_size / (1.0 - test_size)
    strat2 = y_tmp if task == "classification" else None
    Xtr, Xval, ytr, yval = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=seed, stratify=strat2)

    # Se for GridSearchCV, ele usa a validação interna; aqui usamos Xval como
    # conjunto final para reportar. Treinamos no treino e reportamos no val.
    pipe.fit(Xtr, ytr)
    pred_val = pipe.predict(Xval)
    m_val = reg_metrics(yval, pred_val) if task == "regression" else clf_metrics(yval, pred_val)

    # Em seguida, retreinamos com treino+validação e reportamos no teste
    pipe_final = clone(pipe)
    X_train_full = pd.concat([Xtr, Xval])
    y_train_full = pd.concat([ytr, yval])
    pipe_final.fit(X_train_full, y_train_full)
    pred_te = pipe_final.predict(Xte)
    m_te = reg_metrics(yte, pred_te) if task == "regression" else clf_metrics(yte, pred_te)

    extra = {"y_true": np.asarray(yte), "y_pred": np.asarray(pred_te),
             "X_test": Xte.copy(), "y_test": yte.copy(),
             "model": pipe_final,
             "y_val_true": np.asarray(yval), "y_val_pred": np.asarray(pred_val),
             "m_val": m_val}
    if task == "classification":
        labels = sorted(pd.unique(y))
        extra["cm"] = confusion_matrix(yte, pred_te, labels=labels)
        extra["labels"] = labels
    if hasattr(pipe_final, "best_params_"):
        extra["best_params"] = pipe_final.best_params_
    elif hasattr(pipe, "best_params_"):
        extra["best_params"] = pipe.best_params_
    return m_te, extra


# ---------------------------------------------------------------------------
# Teste separado (arquivo dedicado)
# ---------------------------------------------------------------------------
def evaluate_test_set(pipe, X_train, y_train, X_test, y_test, task):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    m = reg_metrics(y_test, pred) if task == "regression" else clf_metrics(y_test, pred)
    extra = {"y_true": np.asarray(y_test), "y_pred": np.asarray(pred),
             "X_test": X_test.copy(), "y_test": y_test.copy(), "model": pipe}
    if task == "classification":
        labels = sorted(pd.unique(y_train))
        extra["cm"] = confusion_matrix(y_test, pred, labels=labels)
        extra["labels"] = labels
    if hasattr(pipe, "best_params_"):
        extra["best_params"] = pipe.best_params_
    return m, extra


# ---------------------------------------------------------------------------
# CV clássica: K-fold e K-fold repetido
# ---------------------------------------------------------------------------
def cross_val(pipe, X, y, task, k: int = 5, repeats: int = 1, seed: int = 42):
    if task == "classification":
        cv = (RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats,
                                      random_state=seed)
              if repeats > 1 else
              StratifiedKFold(k, shuffle=True, random_state=seed))
        sc = CLF_SCORING
    else:
        cv = (RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
              if repeats > 1 else
              KFold(k, shuffle=True, random_state=seed))
        sc = REG_SCORING
    res = cross_validate(pipe, X, y, cv=cv, scoring=sc, n_jobs=-1)
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    return out


# ---------------------------------------------------------------------------
# GroupKFold / StratifiedGroupKFold — slide 7-8 (por regime/período)
# ---------------------------------------------------------------------------
def group_cross_val(pipe, X, y, groups, task, k: int = 5, seed: int = 42):
    """CV respeitando grupos (ex.: velocidade, período, navio)."""
    if task == "classification":
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        sc = CLF_SCORING
    else:
        cv = GroupKFold(n_splits=k)
        sc = REG_SCORING
    res = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=sc, n_jobs=-1)
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    return out


# ---------------------------------------------------------------------------
# TimeSeriesSplit — slide 8 (períodos)
# ---------------------------------------------------------------------------
def timeseries_cv(pipe, X, y, task, k: int = 5):
    cv = TimeSeriesSplit(n_splits=k)
    sc = REG_SCORING if task == "regression" else CLF_SCORING
    res = cross_validate(pipe, X, y, cv=cv, scoring=sc, n_jobs=-1)
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    return out


# ---------------------------------------------------------------------------
# CV aninhada — slide 12 (treino+validação+teste)
# ---------------------------------------------------------------------------
def nested_cv(pipe, X, y, task, k_outer: int = 5, k_inner: int = 3,
              seed: int = 42):
    """CV aninhada.

    Assume que `pipe` já é um GridSearchCV (com cv=k_inner) OU um pipeline puro.
    O outer loop mede desempenho honesto; o inner loop (já em GridSearchCV)
    seleciona hiperparâmetros.
    """
    if task == "classification":
        cv_outer = StratifiedKFold(k_outer, shuffle=True, random_state=seed)
        sc = CLF_SCORING
    else:
        cv_outer = KFold(k_outer, shuffle=True, random_state=seed)
        sc = REG_SCORING
    res = cross_validate(pipe, X, y, cv=cv_outer, scoring=sc, n_jobs=-1,
                         return_estimator=True)
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    # Hiperparâmetros escolhidos por fold (se houver GridSearchCV)
    best = [e.best_params_ for e in res["estimator"]
            if hasattr(e, "best_params_")]
    return out, best
