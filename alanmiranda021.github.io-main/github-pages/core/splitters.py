"""Partições externas e internas que preservam grupos e cronologia."""
import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedKFold, TimeSeriesSplit, train_test_split)


def holdout_indices(y, task, test_size=.2, seed=42, groups=None, times=None, gap=0):
    idx = np.arange(len(y))
    if groups is not None and times is not None:
        raise ValueError("Escolha split agrupado ou temporal; protocolo conjunto ainda não suportado.")
    if groups is not None:
        if pd.isna(groups).any() or len(pd.unique(groups)) < 2:
            raise ValueError("Holdout agrupado requer pelo menos dois grupos sem ausentes.")
        return next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(idx, y, groups))
    if times is not None:
        if gap < 0 or pd.isna(times).any():
            raise ValueError("Tempo sem ausentes e gap não negativo são obrigatórios.")
        order = np.argsort(np.asarray(times), kind="stable")
        cut = int(len(order) * (1 - test_size))
        if cut <= gap or cut >= len(order):
            raise ValueError("Divisão temporal não deixa treino/teste suficientes.")
        boundary = np.asarray(times)[order[cut]]
        train = order[:cut - gap]
        # Equal timestamps cannot cross the boundary.
        train = train[np.asarray(times)[train] < boundary]
        if not len(train):
            raise ValueError("Tempo não permite treino anterior ao teste.")
        return train, order[cut:]
    return train_test_split(idx, test_size=test_size, random_state=seed,
                            stratify=y if task == "classification" else None)


def cv_indices(y, task, k=5, seed=42, groups=None, times=None, gap=0):
    idx = np.arange(len(y))
    if groups is not None and times is not None:
        raise ValueError("Split conjunto grupo/tempo não implementado.")
    if groups is not None:
        if pd.isna(groups).any() or len(pd.unique(groups)) < k:
            raise ValueError(f"São necessários pelo menos {k} grupos completos.")
        return list(GroupKFold(k).split(idx, y, groups))
    if times is not None:
        if pd.isna(times).any():
            raise ValueError("Tempo contém ausentes.")
        order = np.argsort(np.asarray(times), kind="stable")
        splits = []
        for tr, va in TimeSeriesSplit(k, gap=gap).split(order):
            tr, va = order[tr], order[va]
            tr = tr[np.asarray(times)[tr] < np.asarray(times)[va].min()]
            if not len(tr):
                raise ValueError("Timestamps repetidos deixam fold temporal sem treino.")
            splits.append((tr, va))
        return splits
    cv = StratifiedKFold(k, shuffle=True, random_state=seed) if task == "classification" else KFold(k, shuffle=True, random_state=seed)
    return list(cv.split(idx, y))
