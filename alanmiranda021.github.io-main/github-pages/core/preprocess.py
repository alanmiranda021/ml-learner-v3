"""Limpeza de ruído e pré-processamento (sem vazamento de dados)."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, PowerTransformer, RobustScaler,
                                   StandardScaler)

SCALERS = {"Nenhum": None, "Z-score (Standard)": StandardScaler, "Min-Max [0,1]": MinMaxScaler,
           "Robust (mediana/IQR)": RobustScaler,
           "Power (Yeo-Johnson)": PowerTransformer}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita outliers pelo critério IQR. Os limites são aprendidos SÓ no treino."""
    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_, self.hi_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def outlier_report(df: pd.DataFrame, k=1.5) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes("number"):
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        n = int(((s < q1 - k * iqr) | (s > q3 + k * iqr)).sum())
        rows.append({"coluna": c, "outliers": n, "%": round(100 * n / max(len(s), 1), 2)})
    return pd.DataFrame(rows)


def remove_outlier_rows(df, cols, k=1.5):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        mask &= df[c].between(q1 - k * iqr, q3 + k * iqr) | df[c].isna()
    return df[mask]


def smooth_columns(df, cols, method="Média móvel", window=5):
    """Suavização para sinais ordenados no tempo (reduz ruído de medição)."""
    out = df.copy()
    window = max(3, int(window) | 1) if method.startswith("Savitzky") else max(1, int(window))
    for c in cols:
        if method == "Média móvel":
            out[c] = out[c].rolling(window, center=True, min_periods=1).mean()
        elif method == "Mediana móvel":
            out[c] = out[c].rolling(window, center=True, min_periods=1).median()
        else:
            v = out[c].interpolate().bfill().ffill().to_numpy()
            if len(v) > window:
                out[c] = savgol_filter(v, window, 2)
    return out


def build_preprocessor(num_cols, cat_cols, scaler="Z-score (Standard)", winsor=True, k=1.5, pca=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if winsor:
        steps.append(("win", Winsorizer(k)))
    if SCALERS[scaler]:
        steps.append(("sc", SCALERS[scaler]()))
    if pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    tfs = [("num", Pipeline(steps), list(num_cols))]
    if cat_cols:
        tfs.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                     ("oh", OneHotEncoder(handle_unknown="ignore"))]), list(cat_cols)))
    return ColumnTransformer(tfs)
