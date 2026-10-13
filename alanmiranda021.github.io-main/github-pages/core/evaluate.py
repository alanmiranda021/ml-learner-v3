"""Estratégias de divisão (Hold-out, K-fold, K-fold repetido) e métricas."""
from core.env import NJOBS
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

REG_SCORING = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}
CLF_SCORING = {"Acurácia": "accuracy", "F1 (macro)": "f1_macro",
               "Precisão (macro)": "precision_macro", "Recall (macro)": "recall_macro"}


def make_pipeline(pre, model, grid=None, tune=False, cv_inner=5, task="regression", sampler=None):
    if sampler is not None:  # ex.: ADASYN/SMOTE — só aplicado ao treino
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([("pre", clone(pre)), ("samp", sampler), ("model", clone(model))])
    else:
        pipe = Pipeline([("pre", clone(pre)), ("model", clone(model))])
    if tune and grid:
        scoring = "r2" if task == "regression" else "f1_macro"
        pipe = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                            cv=cv_inner, scoring=scoring, n_jobs=NJOBS)
    return pipe


def reg_metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": mean_absolute_error(y, p)}


def clf_metrics(y, p):
    return {"Acurácia": accuracy_score(y, p), "F1 (macro)": f1_score(y, p, average="macro"),
            "Precisão (macro)": precision_score(y, p, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y, p, average="macro", zero_division=0)}


def holdout(pipe, X, y, task, test_size=0.2, seed=42):
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    m = reg_metrics(yte, pred) if task == "regression" else clf_metrics(yte, pred)
    extra = {"y_true": np.asarray(yte), "y_pred": np.asarray(pred), "model": pipe}
    if task == "classification":
        labels = sorted(pd.unique(y))
        extra["cm"], extra["labels"] = confusion_matrix(yte, pred, labels=labels), labels
    if hasattr(pipe, "best_params_"):
        extra["best_params"] = pipe.best_params_
    return m, extra


def cross_val(pipe, X, y, task, k=5, repeats=1, seed=42, progress=None):
    if task == "classification":
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed) if repeats > 1 \
            else StratifiedKFold(k, shuffle=True, random_state=seed)
        sc = CLF_SCORING
    else:
        cv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=seed) if repeats > 1 \
            else KFold(k, shuffle=True, random_state=seed)
        sc = REG_SCORING
    if progress is None:
        res = cross_validate(pipe, X, y, cv=cv, scoring=sc, n_jobs=1)
    else:
        total = cv.get_n_splits(X, y)
        parts = []
        for i, split in enumerate(cv.split(X, y)):
            progress(i / total, f"Validação {i + 1}/{total}: treinando e testando")
            parts.append(cross_validate(pipe, X, y, cv=[split], scoring=sc, n_jobs=1,
                                        error_score="raise"))
            progress((i + 1) / total, f"Validação {i + 1}/{total} concluída")
        res = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    return out
