"""Estratégias de divisão (Hold-out, K-fold, K-fold repetido) e métricas."""
from core.env import NJOBS
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             balanced_accuracy_score, matthews_corrcoef, median_absolute_error)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

REG_SCORING = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "MedAE": "neg_median_absolute_error"}
CLF_SCORING = {"Acurácia": "accuracy", "Acurácia balanceada": "balanced_accuracy", "F1 (macro)": "f1_macro",
               "Precisão (macro)": "precision_macro", "Recall (macro)": "recall_macro"}


def make_pipeline(pre, model, grid=None, tune=False, cv_inner=5, task="regression", sampler=None, reduction=None):
    if sampler is not None and task != "classification":
        raise ValueError("Samplers de classificação não podem aumentar regressão.")
    steps = [("pre", clone(pre))]
    if sampler is not None:
        steps.append(("samp", clone(sampler)))
    if reduction is not None:
        steps.append(("reduce", clone(reduction)))
    steps.append(("model", clone(model)))
    if sampler is not None:  # ex.: ADASYN/SMOTE — só aplicado ao treino
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline(steps)
    else:
        pipe = Pipeline(steps)
    if tune and grid:
        scoring = "r2" if task == "regression" else "f1_macro"
        pipe = GridSearchCV(pipe, {k if "__" in k else f"model__{k}": v for k, v in grid.items()},
                            cv=cv_inner, scoring=scoring, n_jobs=NJOBS)
    return pipe


def reg_metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": mean_absolute_error(y, p), "MedAE": median_absolute_error(y, p)}


def clf_metrics(y, p):
    return {"Acurácia": accuracy_score(y, p), "Acurácia balanceada": balanced_accuracy_score(y, p),
            "MCC": matthews_corrcoef(y, p), "F1 (macro)": f1_score(y, p, average="macro", zero_division=0),
            "F1 (weighted)": f1_score(y, p, average="weighted", zero_division=0),
            "Precisão (macro)": precision_score(y, p, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y, p, average="macro", zero_division=0)}


def holdout(pipe, X, y, task, test_size=0.2, seed=42, groups=None, times=None, gap=0):
    from core.splitters import holdout_indices, cv_indices
    tr, te = holdout_indices(y, task, test_size, seed, groups, times, gap)
    take = lambda a, idx: a.iloc[idx] if hasattr(a, "iloc") else a[idx]
    Xtr, Xte, ytr, yte = take(X, tr), take(X, te), take(y, tr), take(y, te)
    if isinstance(pipe, GridSearchCV) and (groups is not None or times is not None):
        pipe.cv = cv_indices(ytr, task, k=3, seed=seed,
            groups=take(groups, tr) if groups is not None else None,
            times=take(times, tr) if times is not None else None, gap=gap)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    m = reg_metrics(yte, pred) if task == "regression" else clf_metrics(yte, pred)
    extra = {"y_true": np.asarray(yte), "y_pred": np.asarray(pred), "model": pipe,
             "train_indices": tr, "test_indices": te, "X_train": Xtr, "X_test": Xte}
    if task == "classification":
        labels = sorted(pd.unique(y))
        extra["cm"], extra["labels"] = confusion_matrix(yte, pred, labels=labels), labels
    if hasattr(pipe, "best_params_"):
        extra["best_params"] = pipe.best_params_
    return m, extra


def cross_val(pipe, X, y, task, k=5, repeats=1, seed=42, progress=None, groups=None, times=None, gap=0):
    if isinstance(pipe, GridSearchCV) and (groups is not None or times is not None):
        raise ValueError("Use experiments.run_experiment para CV aninhada com grupos/tempo.")
    if task == "classification":
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed) if repeats > 1 \
            else StratifiedKFold(k, shuffle=True, random_state=seed)
        sc = CLF_SCORING
    else:
        cv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=seed) if repeats > 1 \
            else KFold(k, shuffle=True, random_state=seed)
        sc = REG_SCORING
    if groups is not None or times is not None:
        from core.splitters import cv_indices
        if repeats != 1:
            raise ValueError("Repetições aleatórias não se aplicam a grupos/tempo.")
        splits = cv_indices(y, task, k, seed, groups, times, gap)
    else:
        splits = list(cv.split(X, y))
    if progress is None:
        res = cross_validate(pipe, X, y, cv=splits, scoring=sc, n_jobs=1, error_score="raise")
    else:
        total = len(splits)
        parts = []
        for i, split in enumerate(splits):
            progress(i / total, f"Validação {i + 1}/{total}: treinando e testando")
            parts.append(cross_validate(pipe, X, y, cv=[split], scoring=sc, n_jobs=1,
                                        error_score="raise"))
            progress((i + 1) / total, f"Validação {i + 1}/{total} concluída")
        res = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    out = {}
    for name in sc:
        v = res[f"test_{name}"]
        v = -v if name in ("RMSE", "MAE", "MedAE") else v
        out[name] = (float(v.mean()), float(v.std()))
    return out
