"""Pré-processamento robusto e sem vazamento de dados.

Todos os transformers aprendem parâmetros SÓ no treino (fit), e são aplicados
no teste via `transform`. Isso evita data leakage em pipelines do sklearn.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, RobustScaler, StandardScaler)


# ---------------------------------------------------------------------------
# Escalonadores disponíveis (nome amigável -> classe sklearn ou None)
# ---------------------------------------------------------------------------
SCALERS = {
    "Nenhum": None,
    "Z-score (Standard)": StandardScaler,
    "Min-Max [0,1]": MinMaxScaler,
    "Robust (mediana/IQR)": RobustScaler,
    "MAD (mediana/MAD)": None,     # implementado abaixo
    "Tanh (Hampel)": None,          # implementado abaixo
}


# ---------------------------------------------------------------------------
# Transformers customizados
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita outliers por IQR. Aprende limites SÓ no treino."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_ = q1 - self.k * iqr
        self.hi_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class MADScaler(BaseEstimator, TransformerMixin):
    """Escalonamento por mediana e MAD (robusto a outliers).

    Referência: Zhu et al. (2018), seção 3.2.2, Eq. (6).
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.median_), axis=0)
        self.mad_ = np.where(mad < self.eps, 1.0, mad)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.median_) / self.mad_


class TanhScaler(BaseEstimator, TransformerMixin):
    """Escalonamento por tanh-estimador de Hampel (robusto + eficiente).

    Referência: Zhu et al. (2018), seção 3.2.2, Eqs. (7)-(8).
    Reduz a influência de caudas longas sem descartar dados.
    """

    def __init__(self, a: float = 1.5, b: float = 3.0, c: float = 4.5):
        self.a = a
        self.b = b
        self.c = c

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        # MAD como escala robusta
        mad = np.nanmedian(np.abs(X - self.median_), axis=0)
        self.scale_ = np.where(mad < 1e-9, 1.0, mad) / 0.6745  # consistente c/ sigma
        return self

    def transform(self, X):
        z = (np.asarray(X, dtype=float) - self.median_) / self.scale_
        return 0.5 * (np.tanh(0.01 * z) + 1.0)


class CausalSmoother(BaseEstimator, TransformerMixin):
    """Suavização causal (só passado) para séries temporais.

    NÃO usa `center=True` — evita vazamento do futuro. Deve ser aplicado dentro
    do pipeline para que o teste nunca influencie o treino.
    """

    def __init__(self, method: str = "Média móvel", window: int = 5):
        self.method = method
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        w = max(1, int(self.window))
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = pd.Series(X[:, j])
            if self.method == "Média móvel":
                out[:, j] = col.rolling(w, min_periods=1).mean().to_numpy()
            elif self.method == "Mediana móvel":
                out[:, j] = col.rolling(w, min_periods=1).median().to_numpy()
            elif self.method.startswith("Savitzky"):
                # Savitzky-Golay causal: usa somente amostras atuais/passadas.
                # A janela é "trailing"; não há acesso a t+1, t+2, ...
                # ``interpolate`` e ``bfill`` usam uma observação futura em
                # lacunas. Mantemos apenas o último valor observado; lacunas
                # iniciais seguem para o imputer, ajustado no treino.
                v = col.ffill().to_numpy()
                if len(v) <= 2:
                    out[:, j] = v
                    continue
                poly = min(2, max(0, len(v) - 1))
                for i in range(len(v)):
                    max_w = min(w, i + 1)
                    # janela ímpar; no início, reduz para 1/3/5...
                    ww = max_w if max_w % 2 else max_w - 1
                    if ww <= poly:
                        out[i, j] = v[i]
                        continue
                    coeff = savgol_coeffs(ww, poly, pos=ww - 1, use="dot")
                    out[i, j] = np.dot(coeff, v[i - ww + 1:i + 1])
            else:
                out[:, j] = X[:, j]
        return out


# ---------------------------------------------------------------------------
# Diagnóstico de outliers (fora do pipeline, para EDA)
# ---------------------------------------------------------------------------
def outlier_report(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Relatório univariado de outliers por IQR (não remove nada)."""
    rows = []
    for c in df.select_dtypes("number"):
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((s < q1 - k * iqr) | (s > q3 + k * iqr)).sum())
        rows.append({"coluna": c, "outliers": n,
                     "%": round(100 * n / max(len(s), 1), 2)})
    return pd.DataFrame(rows)


def multivariate_outlier_report(df: pd.DataFrame,
                                method: str = "LOF",
                                contamination: float = 0.05,
                                seed: int = 42) -> pd.DataFrame:
    """Detecta outliers multivariados (LOF ou Isolation Forest).

    Retorna o mesmo DataFrame com colunas `_outlier` (bool) e `_score` (float).
    Referência: Zhu et al. (2018), seção 3.1.1.2.
    """
    num = df.select_dtypes("number").dropna()
    if num.empty:
        return pd.DataFrame(columns=list(df.columns) + ["_outlier", "_score"])
    if method == "LOF":
        model = LocalOutlierFactor(n_neighbors=min(20, max(2, len(num) - 1)),
                                   contamination=contamination)
        labels = model.fit_predict(num)
        score = -model.negative_outlier_factor_
    elif method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=seed)
        labels = model.fit_predict(num)
        score = -model.score_samples(num)
    else:
        raise ValueError(f"Método desconhecido: {method}")

    out = df.copy()
    out["_outlier"] = False
    out["_score"] = np.nan
    out.loc[num.index, "_outlier"] = (labels == -1)
    out.loc[num.index, "_score"] = score
    return out


# ---------------------------------------------------------------------------
# Construção do pré-processador (dentro do pipeline)
# ---------------------------------------------------------------------------
def _build_scaler(name: str):
    """Instancia o escalonador escolhido (ou None)."""
    if name == "MAD (mediana/MAD)":
        return MADScaler()
    if name == "Tanh (Hampel)":
        return TanhScaler()
    cls = SCALERS.get(name)
    return cls() if cls is not None else None


def _build_numeric_steps(scaler: str, winsor: bool, k: float,
                         imputer: str, smooth: str, window: int,
                         seed: int = 42):
    """Monta os passos do sub-pipeline numérico."""
    steps = []
    if smooth and smooth != "Nenhuma":
        steps.append(("smooth", CausalSmoother(smooth, window)))
    if imputer == "MICE (IterativeImputer)":
        steps.append(("imp", IterativeImputer(random_state=seed, max_iter=10)))
    else:
        steps.append(("imp", SimpleImputer(strategy="median")))
    if winsor:
        steps.append(("win", Winsorizer(k)))
    sc = _build_scaler(scaler)
    if sc is not None:
        steps.append(("sc", sc))
    return steps


def build_preprocessor(num_cols, cat_cols,
                       scaler: str = "Z-score (Standard)",
                       winsor: bool = True, k: float = 1.5,
                       imputer: str = "Mediana",
                       smooth: str = "Nenhuma", window: int = 5,
                       use_pca: bool = False, pca_var: float = 0.95,
                       seed: int = 42) -> ColumnTransformer:
    """ColumnTransformer final.

    - O PCA é aplicado DEPOIS do ColumnTransformer (via Pipeline externo), no app.
      Aqui só montamos o CT. Use `build_full_pipeline` para incluir PCA.
    """
    num_steps = _build_numeric_steps(scaler, winsor, k, imputer, smooth, window, seed)
    tfs = [("num", Pipeline(num_steps), list(num_cols))]
    if cat_cols:
        tfs.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            list(cat_cols),
        ))
    return ColumnTransformer(tfs, remainder="drop")


def build_full_pipeline(num_cols, cat_cols, *,
                        scaler: str = "Z-score (Standard)",
                        winsor: bool = True, k: float = 1.5,
                        imputer: str = "Mediana",
                        smooth: str = "Nenhuma", window: int = 5,
                        use_pca: bool = False, pca_var: float = 0.95,
                        seed: int = 42) -> Pipeline:
    """Pipeline de pré-processamento completo: CT + (PCA opcional).

    `pca_var` pode ser fração de variância (ex.: 0.95) ou número inteiro de
    componentes (ex.: 10).
    """
    ct = build_preprocessor(num_cols, cat_cols, scaler, winsor, k,
                            imputer, smooth, window, seed=seed)
    steps = [("ct", ct)]
    if use_pca:
        n_components = pca_var if (isinstance(pca_var, int) or pca_var >= 1) else pca_var
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    return Pipeline(steps)
