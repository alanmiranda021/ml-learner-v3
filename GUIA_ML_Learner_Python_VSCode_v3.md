# 📊 Guia v3 — "MATLAB Learner" em Python (web) no VS Code

> Sistema web que substitui o fluxo **Regression/Classification Learner** do MATLAB:
> carrega planilha → limpa ruído → escalona → divide (Hold-out / K-fold / GroupKFold / TimeSeriesSplit / CV aninhada) → treina vários algoritmos → compara métricas → interpreta (SHAP, permutação, calibração, VIF) → exporta modelo e tabelas LaTeX.
> **Autocontido**: todos os arquivos referenciados aqui existem e são mutuamente consistentes. Não há divergência entre o guia e o código.

---

## 1. É viável e o que mudou em relação à v1

**Sim, é viável e é uma das tarefas mais bem cobertas pelo ecossistema Python.** Além do fluxo clássico (hold-out, K-fold, K-fold repetido, escalonamento), esta v3 corrige problemas de vazamento da v1 e adiciona técnicas modernas.

### Correções da v1

1. **Suavização causal** — a v1 usava `rolling(center=True)` antes do split, vazando informação do futuro. Agora a suavização é **causal** (só passado), aplicada por `causal_smooth`, e **não** usa o conjunto de teste para aprender nada.
2. **Remoção de outliers fora do split** — a v1 calculava quartis no dataset inteiro. Agora a remoção é opcional, marcada como **EDA** (não pré-processamento) e há um aviso explícito.
3. **PCA global** — a v1 colocava o PCA dentro do sub-pipeline numérico, deixando as categóricas fora. Agora o PCA ocorre **após** o `ColumnTransformer`, sobre todo o espaço transformado.
4. **`permutation_importance`** — a v1 media no conjunto inteiro (in-sample). Agora mede **no conjunto de teste** (out-of-sample), que é o correto academicamente.
5. **Arquivos unificados** — havia dois `app.py` e dois `analysis.py`. Aqui há **um de cada**.

### Técnicas modernas agregadas

| Técnica | Onde | Motivação (paper Zhu et al., 2018) |
|---|---|---|
| `StratifiedGroupKFold` / `GroupKFold` | `core/evaluate.py` | Slid. 7–8 (split por regime/período) |
| `TimeSeriesSplit` | `core/evaluate.py` | Slid. 8 (períodos distintos de medição) |
| CV aninhada (`nested_cv`) | `core/evaluate.py` | Slid. 12 (treino+validação+teste) |
| Hold-out 3-way | `core/evaluate.py` | Slid. 4 (treino/validação/teste) |
| MAD scaling | `core/preprocess.py` | Seç. 3.2.2 do paper |
| Tanh-estimator (Hampel) | `core/preprocess.py` | Seç. 3.2.2 do paper |
| LOF + Isolation Forest | `core/preprocess.py` | Seç. 3.1.1.2 do paper |
| `IterativeImputer` (MICE) | `core/preprocess.py` | Seç. 3.1.2.2 do paper |
| SHAP values | `core/interpret.py` | Explicabilidade moderna |
| Curva de calibração | `core/analysis.py` | Classificação honesta |
| Curva de aprendizado | `core/analysis.py` | Diagnóstico over/underfit |
| VIF (multicolinearidade) | `core/analysis.py` | Diagnóstico de regressão |
| Semente global (reprodutibilidade) | `core/__init__.py` | Boas práticas |

### Limitações honestas (mantidas da v1)

1. **`.mat` com `table` do MATLAB** não é lido pelo Python. Solução de 1 linha no MATLAB:
   ```matlab
   load('seu.mat'); writetable(suaTabela, 'saida.csv')
   ```
   O app detecta e mostra essa mensagem.
2. **Resultados não idênticos bit a bit** aos do MATLAB (solvers e inicializações diferem). Compare tendências.
3. **MATLAB não é obsoleto** — é padrão em engenharia. Python é complemento.
4. **Vazamento sutil**: `MDO (m³)` na base de regressão é `consumo/h × duração`. O app alerta quando correlação > 0,98.

### Resultados de referência (v1, para comparação)

| Base | Modelo | Hold-out 80/20 |
|---|---|---|
| Regressão navio (Speed, Draft Bow, Beaufort → MDO m³/h) | Linear | R² 0,54 |
| | SVR otimizado | R² 0,76 |
| | Random Forest | R² 0,87 |
| Classificação motor (13 classes) | Random Forest | Acurácia 0,83 |
| | SVM otimizado | Acurácia 0,84 |
| | KNN, 5-fold × 2 | 0,59 ± 0,02 |

---

## 2. Equivalência MATLAB → Python v3

| MATLAB | Este sistema |
|---|---|
| `readtable` / `uigetfile` | Upload CSV/XLSX/MAT/Parquet |
| Seleção manual `[6,7,9]` | Seleção por nome |
| `rmoutliers`, `filloutliers` | Winsorização IQR + LOF/IsoForest (EDA) + Imputer (pipeline) |
| `smoothdata` | Suavização causal (média, mediana, Savitzky-Golay) |
| `normalize` | Z-score, Min-Max, Robust, **MAD**, **Tanh** |
| Regression/Classification Learner | Catálogo de 11 regressores e 10 classificadores |
| Hold-out, Cross-validation | Hold-out, Hold-out 3-way, K-fold, K-fold repetido, **GroupKFold**, **StratifiedGroupKFold**, **TimeSeriesSplit**, **CV aninhada** |
| Hyperparameter optimization | `GridSearchCV` |
| Confusion matrix, Predicted vs Actual | Plotly |
| `cvpartition(..., 'Stratify', true)` | `StratifiedKFold` / `StratifiedGroupKFold` |
| `cvpartition(..., 'Holdout', 0.2)` | `train_test_split(test_size=0.2, stratify=y)` |
| Export model | `.joblib` |
| ADASYN usado na base de óleo | SMOTE/ADASYN via `imbalanced-learn` (só no treino) |

---

## 3. Arquitetura do projeto

```
ml-learner/
├── app.py                    # interface (Streamlit) — única
├── requirements.txt
├── CLAUDE.md                 # skill de IA (opcional)
├── README.md
├── .gitignore
├── core/
│   ├── __init__.py           # seed global + versão
│   ├── io.py                 # leitura CSV/XLSX/MAT/Parquet
│   ├── preprocess.py         # transformers robustos, pipeline, LOF
│   ├── models.py             # catálogo de algoritmos
│   ├── evaluate.py           # estratégias de validação + métricas
│   ├── analysis.py           # EDA, diagnósticos, VIF, LaTeX
│   └── interpret.py          # permutação + SHAP
└── tests/
    └── test_core.py          # pytest mínimo
```

---

## 4. Pré-requisitos

- **Python 3.10+** — <https://www.python.org/downloads/> (marque *Add to PATH*)
- **VS Code** — <https://code.visualstudio.com/>
- Extensões: **Python** e **Pylance**
- **Git** (opcional)

---

## 5. Setup

### 5.1 Criar o projeto

```bash
mkdir ml-learner && cd ml-learner && code .
```

### 5.2 Ambiente virtual

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate
```

> Se o PowerShell bloquear: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`.

### 5.3 Estrutura de pastas

```bash
mkdir core tests
```

### 5.4 `requirements.txt`

```text
streamlit>=1.40
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
scipy>=1.11
openpyxl>=3.1
xlrd>=2.0
pyarrow>=15.0
h5py>=3.10
plotly>=5.20
joblib>=1.3
imbalanced-learn>=0.12
shap>=0.45
pytest>=8.0
tabulate>=0.9
```

```bash
pip install -r requirements.txt
```

> **Nota sobre `shap`**: opcional. Se falhar na instalação (Windows antigo), o app usa apenas permutação. Nada mais quebra.

### 5.5 `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
dados/
*.joblib
*.xlsx
```

---


## 5b. Correções desta v3

1. **SHAP:** o `X` bruto é transformado pelo passo `pre` antes do SHAP; os nomes vêm de `get_feature_names_out()`.
2. **Savitzky-Golay:** janela trailing/causal via `savgol_coeffs(pos=window-1)`; alterar um ponto futuro não altera um ponto passado.
3. **Permutação:** Hold-out, Hold-out 3-way e Teste Separado guardam `X_test/y_test` e a interpretação usa somente esse conjunto.
4. **Excel `.xls`:** `xlrd>=2.0` foi adicionado ao `requirements.txt`.
5. **Tuning:** K-fold/GroupKFold/TimeSeriesSplit avisam e não fazem GridSearch ingênuo; use **CV aninhada** para tuning honesto.
6. **CPU:** modelos internos ficam single-thread quando o GridSearch/cross-validation paraleliza.
7. **Streamlit:** resultados são invalidados quando configurações mudam e diagnósticos/SHAP/permutação têm cache.
8. **Uploads:** múltiplos arquivos precisam ter exatamente as mesmas colunas/ordem.
9. **Detecção de tarefa:** alvo numérico é regressão por padrão; cardinalidade baixa não força classificação.
10. **Warnings:** avisos úteis, como `ConvergenceWarning`, não são mais ocultados globalmente.

## 6. Código completo

> A versão abaixo é a que está no ZIP. As correções críticas são: SHAP no espaço transformado, Savitzky-Golay causal, permutação usando teste OOS real, `xlrd`, paralelismo sem oversubscription, aviso de tuning em K-fold e invalidação/cache no Streamlit.

### `core/__init__.py`

```python

"""Núcleo do ml-learner: seed global, versão e utilidades comuns."""
from __future__ import annotations
import os
import random
import numpy as np

__version__ = "2.0.0"


def set_global_seed(seed: int = 42) -> int:
    """Fixa sementes em numpy/random/python para reprodutibilidade.

    Observação: nem todos os algoritmos do sklearn aceitam `random_state`
    (ex.: alguns otimizadores do scipy). Ainda assim, fixar aqui ajuda.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    return int(seed)

```

### `core/io.py`

```python

"""Leitura de arquivos: CSV (qualquer separador/decimal), Excel, .mat e parquet."""
from __future__ import annotations
import csv
import io
import numpy as np
import pandas as pd


def _sniff_csv(raw: bytes) -> str:
    """Detecta o separador de um CSV (`,` `;` `\\t` `|`) usando as primeiras linhas."""
    text = raw[:20000].decode("utf-8", errors="ignore")
    first_line = text.split("\n", 1)[0] + "\n"
    try:
        return csv.Sniffer().sniff(first_line, delimiters=";,\t|").delimiter
    except csv.Error:
        return ";" if text.count(";") > text.count(",") else ","


def load_table(name: str, raw: bytes) -> pd.DataFrame:
    """Carrega CSV/TXT/TSV/XLSX/XLS/MAT/PARQUET em um DataFrame.

    Estratégia de robustez: para CSV, tenta primeiro decimal `.`, e se houver
    colunas numéricas que viraram texto, tenta decimal `,` (comum em pt-BR).
    """
    lname = name.lower()
    if lname.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if lname.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(raw))
    if lname.endswith((".csv", ".txt", ".tsv")):
        sep = _sniff_csv(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding_errors="replace")
        obj_cols = [c for c in df.columns
                    if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        if obj_cols:
            df2 = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=",",
                              encoding_errors="replace")
            n_num_1 = sum(df[c].dtype.kind in "fi" for c in obj_cols)
            n_num_2 = sum(df2[c].dtype.kind in "fi" for c in obj_cols)
            if n_num_2 > n_num_1:
                df = df2
        return df
    if lname.endswith(".mat"):
        return _load_mat(raw)
    raise ValueError(f"Formato não suportado: {name}")


def _load_mat(raw: bytes) -> pd.DataFrame:
    """Carrega .mat (v7 clássico ou v7.3 HDF5). Ignora objetos `table` do MATLAB."""
    import scipy.io as sio
    try:
        m = sio.loadmat(io.BytesIO(raw))
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(io.BytesIO(raw)) as h:
            m = {k: np.array(v).T for k, v in h.items()
                 if isinstance(v, h5py.Dataset) and v.ndim == 2}
    candidates = {k: v for k, v in m.items()
                  if not k.startswith("__")
                  and isinstance(v, np.ndarray)
                  and v.ndim == 2
                  and v.dtype.kind in "fiu"}
    if not candidates:
        raise ValueError(
            "Este .mat guarda uma 'table' do MATLAB (objeto opaco). "
            "Rode no MATLAB:  writetable(suaTabela, 'saida.csv')  e envie o CSV."
        )
    key = max(candidates, key=lambda k: candidates[k].size)
    arr = candidates[key]
    return pd.DataFrame(arr, columns=[f"x{i+1}" for i in range(arr.shape[1])])

```

### `core/preprocess.py`

```python

"""Pré-processamento robusto e sem vazamento de dados.

Todos os transformers aprendem parâmetros SÓ no treino (fit), e são aplicados
no teste via `transform`. Isso evita data leakage em pipelines do sklearn.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, savgol_filter
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

```

### `core/models.py`

```python

"""Catálogo de algoritmos e grades de hiperparâmetros.

Equivalente aos Regression/Classification Learner do MATLAB.
"""
from __future__ import annotations

from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression, Ridge)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REGRESSION = {
    "Regressão Linear": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=10000), {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 1]}),
    "Árvore de Decisão": (DecisionTreeRegressor(random_state=0),
                          {"max_depth": [3, 5, 8, 12, None]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 9, 15]}),
    "SVR (RBF)": (SVR(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (RandomForestRegressor(random_state=0, n_jobs=1),
                      {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Extra Trees": (ExtraTreesRegressor(random_state=0, n_jobs=1),
                    {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=0),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
    "HistGradient Boosting": (HistGradientBoostingRegressor(random_state=0),
                              {"max_iter": [100, 300], "learning_rate": [0.05, 0.1]}),
    "Rede Neural (MLP)": (MLPRegressor(max_iter=1000, random_state=0),
                          {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}),
}

CLASSIFICATION = {
    "Regressão Logística": (LogisticRegression(max_iter=2000, n_jobs=1),
                            {"C": [0.1, 1, 10]}),
    "Naive Bayes": (GaussianNB(), {}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=0),
                          {"max_depth": [3, 5, 8, 12, None]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 9, 15]}),
    "SVM (RBF)": (SVC(probability=True), {"C": [0.1, 1, 10, 100],
                                          "gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (RandomForestClassifier(random_state=0, n_jobs=1),
                      {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Extra Trees": (ExtraTreesClassifier(random_state=0, n_jobs=1),
                    {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=0),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
    "HistGradient Boosting": (HistGradientBoostingClassifier(random_state=0),
                              {"max_iter": [100, 300], "learning_rate": [0.05, 0.1]}),
    "Rede Neural (MLP)": (MLPClassifier(max_iter=1000, random_state=0),
                          {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}),
}

```

### `core/analysis.py`

```python

"""Estatística descritiva, diagnósticos, VIF, calibração e exportação LaTeX."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve


def detailed_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela descritiva completa (média, DP, mediana, IQR, assimetria, curtose)."""
    num_df = df.select_dtypes(include=[np.number])
    rows = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "Variável": col,
            "Média": s.mean(),
            "Desv. Padrão": s.std(),
            "Mediana": s.median(),
            "IQR": q3 - q1,
            "Mínimo": s.min(),
            "Máximo": s.max(),
            "Assimetria": s.skew(),
            "Curtose": s.kurtosis(),
        })
    return pd.DataFrame(rows).round(4)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor para detectar multicolinearidade.

    Regra prática: VIF > 10 indica multicolinearidade problemática.
    Requer `statsmodels`.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    num = X.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["Variável", "VIF"])
    # Adiciona constante para o cálculo do VIF
    num_const = num.assign(_const=1.0)
    rows = [{"Variável": c,
             "VIF": float(variance_inflation_factor(num_const.values, i))}
            for i, c in enumerate(num_const.columns) if c != "_const"]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).round(3)


def regression_diagnostics(y_true, y_pred, n_features: int):
    """Diagnóstico estatístico para regressão."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    res = y_true - y_pred
    n = len(y_true)
    p = n_features

    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2

    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    mape = float(np.mean(np.abs(res / np.where(y_true == 0, 1e-8, y_true))) * 100)

    sample_res = res[:5000]
    shapiro_p = 0.0
    if len(sample_res) >= 3:
        _, shapiro_p = stats.shapiro(sample_res)

    summary = {
        "Métrica": ["R²", "R² Ajustado", "RMSE", "MAE", "MAPE (%)",
                    "Resíduos: Média", "Resíduos: Desv. Padrão",
                    "Shapiro-Wilk (p-valor)"],
        "Valor": [round(r2, 4), round(r2_adj, 4), round(rmse, 4), round(mae, 4),
                  round(mape, 2), round(float(np.mean(res)), 4),
                  round(float(np.std(res)), 4), round(float(shapiro_p), 5)],
    }
    return pd.DataFrame(summary), res


def classification_detailed_report(y_true, y_pred) -> pd.DataFrame:
    """Relatório classe a classe (precisão, recall, F1, suporte)."""
    report_dict = classification_report(y_true, y_pred, output_dict=True,
                                        zero_division=0)
    df_rep = pd.DataFrame(report_dict).transpose().reset_index()
    df_rep.rename(columns={"index": "Classe/Métrica"}, inplace=True)
    return df_rep.round(4)


def calibration_table(y_true, y_proba, n_bins: int = 10):
    """Curva de calibração (reliability diagram) para classificação binária.

    Só faz sentido para problemas binários. Para multiclasse, use one-vs-rest
    e chame uma vez por classe.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins,
                                             strategy="uniform")
    return pd.DataFrame({"Probabilidade prevista": prob_pred,
                         "Fração de positivos": prob_true})


def learning_curve_table(pipe, X, y, task: str, cv=5, n_points: int = 8,
                         seed: int = 42) -> pd.DataFrame:
    """Dados para curva de aprendizado (diagnóstico de over/underfitting)."""
    scoring = "r2" if task == "regression" else "f1_macro"
    sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=cv, scoring=scoring, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, n_points), random_state=seed,
        shuffle=True,
    )
    return pd.DataFrame({
        "Tamanho do treino": sizes,
        "Treino (média)": train_scores.mean(axis=1),
        "Treino (DP)": train_scores.std(axis=1),
        "Validação (média)": test_scores.mean(axis=1),
        "Validação (DP)": test_scores.std(axis=1),
    })


def to_latex_table(df: pd.DataFrame, caption: str = "Tabela de Resultados") -> str:
    """Exporta DataFrame no formato de tabela LaTeX (booktabs-ready)."""
    return df.to_latex(index=False, caption=caption, float_format="%.4f",
                       column_format="l" + "c" * (len(df.columns) - 1))

```

### `core/evaluate.py`

```python

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

```

### `core/interpret.py`

```python

"""Interpretação honesta: permutação OOS e SHAP no espaço transformado."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_df(model, X_test: pd.DataFrame, y_test, n_repeats: int = 10,
                              seed: int = 42, scoring=None) -> pd.DataFrame:
    """Importância por permutação no conjunto de teste nunca usado no ajuste."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test deve ser um DataFrame para preservar os nomes das variáveis.")
    res = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=seed, scoring=scoring, n_jobs=-1
    )
    return (pd.DataFrame({
        "Variável": X_test.columns,
        "Importância": res.importances_mean,
        "DP": res.importances_std,
    }).sort_values("Importância", ascending=False).reset_index(drop=True))


def _unwrap_estimator(model):
    """Retorna (pipeline/preprocessador, estimador final) de Pipeline ou GridSearchCV."""
    fitted = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    pre = fitted.named_steps.get("pre") if hasattr(fitted, "named_steps") else None
    est = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    return fitted, pre, est


def try_shap(model, X_sample: pd.DataFrame, task: str = "regression",
             max_samples: int = 200, seed: int = 42):
    """Calcula SHAP corretamente no espaço de entrada do estimador final.

    Para pipelines, X bruto é primeiro transformado pelo passo ``pre``. Os nomes
    das features são obtidos do transformador, evitando o bug em que o estimador
    recebia 3 colunas cruas quando esperava, por exemplo, 5 colunas após one-hot.
    Se SHAP não estiver instalado ou o estimador não for suportado, retorna None.
    """
    try:
        import shap
    except ImportError:
        return None

    if not isinstance(X_sample, pd.DataFrame) or X_sample.empty:
        return None

    Xs = X_sample.sample(min(max_samples, len(X_sample)), random_state=seed)
    fitted, pre, est = _unwrap_estimator(model)

    try:
        if pre is not None:
            Xt = pre.transform(Xs)
            try:
                feature_names = list(pre.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(Xt.shape[1])]
        else:
            Xt = Xs.to_numpy()
            feature_names = list(Xs.columns)

        # Mantém formato denso para compatibilidade ampla com SHAP.
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt)

        try:
            explainer = shap.TreeExplainer(est)
            values = explainer.shap_values(Xt)
        except Exception:
            # Fallback genérico, ainda no ESPAÇO TRANSFORMADO.
            predict_fn = (est.predict_proba if task == "classification"
                          and hasattr(est, "predict_proba") else est.predict)
            background = Xt[:min(50, len(Xt))]
            explainer = shap.Explainer(predict_fn, background)
            values = explainer(Xt).values

        # SHAP para classificação multiclasses pode retornar lista ou ndarray 3-D.
        if isinstance(values, list):
            arr = np.stack([np.asarray(v) for v in values], axis=0)
            mean_abs = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
            mean_abs = np.asarray(mean_abs).reshape(-1)
            # arr: classes x samples x features -> média em classes e amostras
            mean_abs = np.mean(np.abs(arr), axis=(0, 1))
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                # versões novas: samples x features x classes
                mean_abs = np.mean(np.abs(arr), axis=(0, 2))
            elif arr.ndim == 2:
                mean_abs = np.mean(np.abs(arr), axis=0)
            else:
                return None

        if len(feature_names) != len(mean_abs):
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        imp = pd.DataFrame({
            "Variável": feature_names,
            "|SHAP| médio": mean_abs
        }).sort_values("|SHAP| médio", ascending=False).reset_index(drop=True)
        return imp, Xt, values

    except Exception:
        # SHAP é diagnóstico opcional; não deve derrubar o app.
        return None

```

### `app.py`

```python

"""Interface web do ml-learner (Streamlit). Substitui Regression/Classification
Learner do MATLAB com técnicas modernas de validação e interpretação.
"""
from __future__ import annotations
import warnings
warnings.simplefilter("default")
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import set_global_seed, __version__
from core.io import load_table
from core.preprocess import (SCALERS, build_full_pipeline, multivariate_outlier_report,
                             outlier_report)
from core.models import CLASSIFICATION, REGRESSION
from core.evaluate import (cross_val, evaluate_test_set, group_cross_val, holdout,
                           holdout_3way, make_pipeline, nested_cv, timeseries_cv)
from core.analysis import (calibration_table, classification_detailed_report,
                           detailed_eda, learning_curve_table,
                           regression_diagnostics, to_latex_table, vif_table)
from core.interpret import permutation_importance_df, try_shap

st.set_page_config(page_title="ml-learner v3", layout="wide")
st.title(f"📊 ml-learner v{__version__} — Regressão e Classificação")
st.caption("Alternativa em Python ao Regression/Classification Learner do MATLAB. "
           "Com validação em grupo, CV aninhada, SHAP e calibração.")

# ---------------------------------------------------------------------------
# 1. DADOS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Dados de treino")
    files = st.file_uploader(
        "CSV, Excel, .mat ou Parquet (múltiplos concatenam)",
        type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls", "mat", "parquet", "pq"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Envie pelo menos 1 arquivo de treino.")
        st.stop()
    frames = []
    for f in files:
        try:
            frames.append(load_table(f.name, f.getvalue()))
        except Exception as e:
            st.error(f"{f.name}: {e}")
    if not frames:
        st.stop()
    if len(frames) > 1:
        reference = list(frames[0].columns)
        incompatible = [f.name for f, frame in zip(files, frames)
                        if list(frame.columns) != reference]
        if incompatible:
            st.error(
                "Os arquivos enviados têm colunas diferentes. "
                "Para evitar NaNs silenciosos por alinhamento do pandas, "
                "use arquivos com as mesmas colunas e na mesma ordem. "
                f"Incompatíveis: {', '.join(incompatible)}"
            )
            st.stop()
    df = pd.concat(frames, ignore_index=True)

    st.header("1b) Teste separado (opcional)")
    use_test_file = st.checkbox("Usar arquivo de teste dedicado", value=False)
    df_test = None
    if use_test_file:
        test_file = st.file_uploader("Arquivo de teste",
                                     type=["csv", "txt", "tsv", "xlsx", "xlsm",
                                           "xls", "mat", "parquet", "pq"],
                                     key="test_upload")
        if test_file:
            try:
                df_test = load_table(test_file.name, test_file.getvalue())
            except Exception as e:
                st.error(f"Erro ao ler teste: {e}")

st.subheader("Visão geral dos dados de treino")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas", len(df))
c2.metric("Colunas", df.shape[1])
c3.metric("Valores ausentes", int(df.isna().sum().sum()))
st.dataframe(df.head(50), use_container_width=True)
if df_test is not None:
    st.caption(f"ℹ️ Teste carregado: {len(df_test)} linhas × {df_test.shape[1]} colunas.")

# ---------------------------------------------------------------------------
# 2. PROBLEMA E FEATURES
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("2) Problema")
    target = st.selectbox("Variável alvo (y)", df.columns, index=len(df.columns) - 1)
    # Numérico é regressão por padrão: cardinalidade baixa não basta para
    # chamar algo de classificação (ex.: Beaufort 0–9 é um alvo de regressão
    # possível). O usuário pode selecionar Classificação manualmente.
    task_guess = ("classification"
                  if not pd.api.types.is_numeric_dtype(df[target])
                  else "regression")
    task = st.radio("Tipo",
                    ["regression", "classification"],
                    index=["regression", "classification"].index(task_guess),
                    format_func=lambda t: "Regressão" if t == "regression" else "Classificação")
    feats = st.multiselect("Características (X)",
                           [c for c in df.columns if c != target],
                           default=[c for c in df.columns if c != target])

if not feats:
    st.warning("Selecione ao menos uma característica.")
    st.stop()

# Aviso de vazamento (correlação alta)
if task == "regression" and pd.api.types.is_numeric_dtype(df[target]):
    corr = (df[feats].select_dtypes("number")
            .corrwith(df[target]).abs().sort_values(ascending=False))
    if len(corr) and corr.iloc[0] > 0.98:
        st.warning(f"⚠️ '{corr.index[0]}' tem correlação {corr.iloc[0]:.3f} "
                   f"com o alvo — possível vazamento de dados.")

# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("3) Pré-processamento")
    num_cols_all = df[feats].select_dtypes("number").columns.tolist()
    winsor = st.checkbox("Limitar outliers (IQR) — no pipeline", value=True)
    k_iqr = st.slider("Fator IQR", 1.0, 3.0, 1.5, 0.1)
    scaler = st.selectbox("Escalonamento", list(SCALERS), index=1)
    imputer = st.selectbox("Imputação",
                           ["Mediana", "MICE (IterativeImputer)"], index=0)
    smooth = st.selectbox("Suavização causal (séries temporais)",
                          ["Nenhuma", "Média móvel", "Mediana móvel", "Savitzky-Golay"])
    win = st.slider("Janela", 3, 51, 5, 2) if smooth != "Nenhuma" else 5
    use_pca = st.checkbox("Aplicar PCA (após o ColumnTransformer)", value=False)
    pca_var = st.slider("Componentes (fração de variância)", 0.50, 0.99, 0.95, 0.01) if use_pca else 0.95

    st.header("4) Validação")
    if use_test_file and df_test is not None:
        strategy = "Teste Separado"
        st.info("Usando arquivo de teste dedicado.")
    else:
        strategy = st.radio(
            "Estratégia",
            ["Hold-out", "Hold-out 3-way", "K-fold", "K-fold repetido",
             "GroupKFold (por grupos)", "TimeSeriesSplit", "CV aninhada"],
        )
        if strategy == "Hold-out":
            test_size = st.slider("% teste", 10, 40, 20, 5) / 100
        elif strategy == "Hold-out 3-way":
            test_size = st.slider("% teste", 10, 30, 20, 5) / 100
            val_size = st.slider("% validação", 5, 20, 10, 5) / 100
        elif strategy in ("K-fold", "K-fold repetido", "GroupKFold (por grupos)",
                          "TimeSeriesSplit"):
            k = st.slider("K / Folds", 3, 10, 5)
            reps = st.slider("Repetições", 2, 10, 5) if strategy == "K-fold repetido" else 1
        elif strategy == "CV aninhada":
            k_outer = st.slider("Folds externos", 3, 10, 5)
            k_inner = st.slider("Folds internos", 2, 5, 3)

    st.header("5) Semente e algoritmos")
    seed = st.number_input("Semente aleatória", 0, 9999, 42)
    zoo = REGRESSION if task == "regression" else CLASSIFICATION
    chosen = st.multiselect("Modelos", list(zoo), default=list(zoo)[:4])
    tune = st.checkbox("Otimizar hiperparâmetros (GridSearch, CV interna)", value=False)
    sampler_name = "Nenhum"
    if task == "classification":
        sampler_name = st.selectbox("Balanceamento (só no treino)",
                                    ["Nenhum", "SMOTE", "ADASYN"])

# Fixa semente global
set_global_seed(int(seed))

# Coluna de grupos (para GroupKFold)
group_col = None
if strategy == "GroupKFold (por grupos)":
    group_col = st.sidebar.selectbox("Coluna de grupos", df.columns.tolist())

# Invalida resultados antigos quando qualquer configuração relevante muda.
import hashlib
_config_payload = repr({
    "shape": df.shape,
    "columns": tuple(map(str, df.columns)),
    "target": target,
    "task": task,
    "feats": tuple(feats),
    "winsor": winsor,
    "k_iqr": k_iqr,
    "scaler": scaler,
    "imputer": imputer,
    "smooth": smooth,
    "win": win,
    "pca": (use_pca, pca_var),
    "strategy": strategy,
    "seed": int(seed),
    "chosen": tuple(chosen),
    "tune": tune,
    "sampler": sampler_name,
    "group_col": group_col,
})
_config_key = hashlib.sha256(_config_payload.encode()).hexdigest()
if st.session_state.get("_config_key") != _config_key:
    st.session_state.pop("res", None)
    st.session_state["_config_key"] = _config_key

# Cache de diagnósticos determinísticos e interpretações caras.
@st.cache_data(show_spinner=False)
def _cached_eda(frame):
    return detailed_eda(frame)

@st.cache_data(show_spinner=False)
def _cached_outliers(frame, k):
    return outlier_report(frame, k)

@st.cache_data(show_spinner=False)
def _cached_shap(model_bytes, frame, task_name, max_samples, seed_value):
    import io as _io
    fitted = joblib.load(_io.BytesIO(model_bytes))
    return try_shap(fitted, frame, task=task_name,
                    max_samples=max_samples, seed=seed_value)

@st.cache_data(show_spinner=False)
def _cached_permutation(model_bytes, frame, y_values, seed_value):
    import io as _io
    fitted = joblib.load(_io.BytesIO(model_bytes))
    y_series = pd.Series(y_values, index=frame.index)
    return permutation_importance_df(fitted, frame, y_series,
                                     n_repeats=8, seed=seed_value)

# ---------------------------------------------------------------------------
# 4. EDA + OUTLIERS MULTIVARIADOS
# ---------------------------------------------------------------------------
work = df.dropna(subset=[target]).copy()
X = work[feats]
y = work[target]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in feats if c not in num_cols]

# Pipeline base (sem sampler, sem tune — o GridSearch é aplicado depois)
pre = build_full_pipeline(num_cols, cat_cols,
                          scaler=scaler, winsor=winsor, k=k_iqr,
                          imputer=imputer, smooth=smooth, window=win,
                          use_pca=use_pca, pca_var=pca_var, seed=int(seed))

with st.expander("🔎 Diagnóstico univariado, multivariado e multicolinearidade"):
    st.markdown("**Outliers univariados (IQR)**")
    st.dataframe(_cached_outliers(work[num_cols], k_iqr), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Outliers multivariados (LOF)**")
        lof_df = multivariate_outlier_report(work[num_cols], method="LOF",
                                             contamination=0.05, seed=int(seed))
        n_lof = int(lof_df["_outlier"].sum()) if "_outlier" in lof_df else 0
        st.metric("Suspeitos (LOF)", n_lof)
    with col_b:
        st.markdown("**Outliers multivariados (Isolation Forest)**")
        iso_df = multivariate_outlier_report(work[num_cols], method="Isolation Forest",
                                             contamination=0.05, seed=int(seed))
        n_iso = int(iso_df["_outlier"].sum()) if "_outlier" in iso_df else 0
        st.metric("Suspeitos (IsoForest)", n_iso)

    if task == "regression" and len(num_cols) >= 2:
        st.markdown("**VIF (multicolinearidade)**")
        st.dataframe(vif_table(work[num_cols]), use_container_width=True)

    if len(num_cols) > 1:
        st.markdown("**Matriz de correlação**")
        st.plotly_chart(
            px.imshow(work[num_cols].corr(), aspect="auto",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# 5. ABAS
# ---------------------------------------------------------------------------
tab_eda, tab_train, tab_diag, tab_export = st.tabs([
    "📈 EDA", "⚙️ Treino & Comparação", "🔍 Diagnóstico", "📄 Exportação"
])

with tab_eda:
    st.subheader("Estatística descritiva")
    st.dataframe(_cached_eda(work[[target] + feats]), use_container_width=True)
    st.subheader("Distribuição do alvo")
    st.plotly_chart(
        px.histogram(work, x=target, marginal="box", title=f"Distribuição de {target}"),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# 6. TREINO
# ---------------------------------------------------------------------------
with tab_train:
    if st.button("▶️ Treinar e comparar", type="primary"):
        sampler = None
        if sampler_name != "Nenhum":
            from imblearn.over_sampling import ADASYN, SMOTE
            sampler = {"SMOTE": SMOTE, "ADASYN": ADASYN}[sampler_name](
                random_state=int(seed))
        rows, store = [], {}
        bar = st.progress(0.0)

        if tune and strategy in ("K-fold", "K-fold repetido",
                                 "GroupKFold (por grupos)", "TimeSeriesSplit"):
            st.warning(
                "Otimização de hiperparâmetros foi desativada nesta estratégia: "
                "GridSearch dentro do mesmo K-fold usado para estimar desempenho "
                "é uma avaliação enviesada. Use **CV aninhada** para tuning honesto."
            )
        if tune and strategy == "Hold-out 3-way":
            st.info(
                "No Hold-out 3-way, o GridSearch já faz CV interna no treino; "
                "o conjunto de validação continua útil para diagnóstico, mas "
                "não participa da seleção dos hiperparâmetros."
            )

        for i, name in enumerate(chosen):
            model, grid = zoo[name]
            try:
                # --- estratégias de avaliação ---
                if strategy == "Teste Separado":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    X_te, y_te = df_test[feats], df_test[target]
                    m, extra = evaluate_test_set(pipe, X, y, X_te, y_te, task)
                    store[name] = extra
                    rows.append({"Modelo": name, **{a: round(b, 4) for a, b in m.items()}})

                elif strategy == "Hold-out":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    m, extra = holdout(pipe, X, y, task, test_size, int(seed))
                    store[name] = extra
                    rows.append({"Modelo": name, **{a: round(b, 4) for a, b in m.items()}})

                elif strategy == "Hold-out 3-way":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    m, extra = holdout_3way(pipe, X, y, task, val_size, test_size, int(seed))
                    store[name] = extra
                    rows.append({"Modelo": name,
                                 **{f"{a} (teste)": round(b, 4) for a, b in m.items()},
                                 **{f"{a} (val)": round(b, 4)
                                    for a, b in extra["m_val"].items()}})

                elif strategy in ("K-fold", "K-fold repetido"):
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = cross_val(pipe, X, y, task, k,
                                  reps if strategy == "K-fold repetido" else 1,
                                  int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "GroupKFold (por grupos)":
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = group_cross_val(pipe, X, y, work[group_col], task, k, int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "TimeSeriesSplit":
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = timeseries_cv(pipe, X, y, task, k)
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "CV aninhada":
                    pipe = make_pipeline(pre, model, grid, tune, cv_inner=k_inner,
                                         task=task, sampler=sampler)
                    m, best = nested_cv(pipe, X, y, task, k_outer, k_inner, int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})
                    if best:
                        st.caption(f"Melhores hiperparâmetros por fold ({name}): {best[:2]}...")
            except Exception as e:
                rows.append({"Modelo": name, "erro": str(e)[:120]})
            bar.progress((i + 1) / len(chosen))

        st.session_state["res"] = (pd.DataFrame(rows), store, task)

    if "res" in st.session_state:
        res_df, _, _ = st.session_state["res"]
        st.subheader("Resultados")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("⬇️ Baixar CSV",
                           res_df.to_csv(index=False).encode(),
                           "resultados.csv")

# ---------------------------------------------------------------------------
# 7. DIAGNÓSTICO E INTERPRETAÇÃO
# ---------------------------------------------------------------------------
with tab_diag:
    if "res" in st.session_state and st.session_state["res"][1]:
        _, store, rtask = st.session_state["res"]
        pick = st.selectbox("Modelo para diagnóstico", list(store))
        e = store[pick]

        if rtask == "regression":
            diag, resid = regression_diagnostics(e["y_true"], e["y_pred"], len(feats))
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Métricas detalhadas")
                st.dataframe(diag, use_container_width=True)
            with c2:
                st.subheader("Resíduos vs Valores Ajustados")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=e["y_pred"], y=resid, mode="markers",
                                         name="Resíduos"))
                fig.add_shape(type="line", x0=float(min(e["y_pred"])), y0=0,
                              x1=float(max(e["y_pred"])), y1=0,
                              line=dict(color="red", dash="dash"))
                fig.update_layout(xaxis_title="Previsto", yaxis_title="Resíduo")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Real × Previsto")
            d = pd.DataFrame({"real": e["y_true"], "previsto": e["y_pred"]})
            fig = px.scatter(d, x="real", y="previsto")
            lo, hi = float(d.min().min()), float(d.max().max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Curva de aprendizado")
            try:
                pipe_for_lc = make_pipeline(pre, zoo[pick][0], None, False, task=task)
                lc = learning_curve_table(pipe_for_lc, X, y, "regression",
                                          cv=min(5, max(2, len(X) // 20)),
                                          n_points=6, seed=int(seed))
                st.plotly_chart(px.line(lc, x="Tamanho do treino",
                                        y=["Treino (média)", "Validação (média)"],
                                        markers=True),
                                use_container_width=True)
            except Exception as err:
                st.caption(f"Curva de aprendizado indisponível: {err}")

        else:  # Classificação
            st.subheader("Relatório por classe")
            st.dataframe(classification_detailed_report(e["y_true"], e["y_pred"]),
                         use_container_width=True)
            st.subheader("Matriz de confusão")
            st.plotly_chart(
                px.imshow(e["cm"], text_auto=True,
                          x=[str(l) for l in e["labels"]],
                          y=[str(l) for l in e["labels"]],
                          labels=dict(x="Previsto", y="Real")),
                use_container_width=True,
            )
            # Curva de calibração (apenas binária)
            if "y_proba" in e and len(e["labels"]) == 2:
                st.subheader("Curva de calibração (binária)")
                proba = e["y_proba"][:, 1]
                cal = calibration_table(e["y_true"], proba, n_bins=10)
                st.plotly_chart(
                    px.line(cal, x="Probabilidade prevista",
                            y="Fração de positivos", markers=True),
                    use_container_width=True,
                )

        # Importância por permutação — medida OUT-OF-SAMPLE
        st.markdown("---")
        st.subheader("Importância por permutação (out-of-sample)")
        try:
            # Use EXCLUSIVAMENTE o teste que ficou fora do ajuste.
            # Para CV sem modelo final único, não exibimos uma permutação
            # artificialmente in-sample.
            if "X_test" not in e or "y_test" not in e:
                st.info(
                    "Importância por permutação OOS disponível para Hold-out "
                    "e Teste Separado. Para K-fold/CV aninhada, use a métrica "
                    "por fold ou ajuste um modelo final em um teste reservado."
                )
                imp_df = None
            else:
                buf_model = io.BytesIO()
                joblib.dump(e["model"], buf_model)
                imp_df = _cached_permutation(
                    buf_model.getvalue(), e["X_test"], e["y_test"], int(seed)
                )
            if imp_df is not None:
                st.plotly_chart(
                    px.bar(imp_df.head(20), x="Importância", y="Variável",
                           orientation="h", error_x="DP"),
                    use_container_width=True,
                )
        except Exception as err:
            st.caption(f"Permutação indisponível: {err}")

        # SHAP (opcional)
        st.subheader("SHAP (opcional)")
        with st.spinner("Calculando SHAP no espaço transformado..."):
            shap_frame = e.get("X_test", X)
            buf_model = io.BytesIO()
            joblib.dump(e["model"], buf_model)
            shap_out = _cached_shap(
                buf_model.getvalue(), shap_frame, rtask, 200, int(seed)
            )
        if shap_out is None:
            st.caption("Pacote `shap` não instalado — apenas permutação disponível.")
        else:
            shap_imp, _, _ = shap_out
            st.plotly_chart(
                px.bar(shap_imp.head(20), x="|SHAP| médio", y="Variável",
                       orientation="h"),
                use_container_width=True,
            )

        if "best_params" in e:
            st.write("Melhores hiperparâmetros:", e["best_params"])

        buf = io.BytesIO()
        joblib.dump(e["model"], buf)
        st.download_button("⬇️ Baixar modelo (.joblib)", buf.getvalue(),
                           f"{pick}.joblib")
    else:
        st.info("Treine modelos na aba 'Treino & Comparação' primeiro.")

# ---------------------------------------------------------------------------
# 8. EXPORTAÇÃO
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader("Tabelas para relatório (LaTeX / Excel)")
    if "res" in st.session_state:
        res_df, store, rtask = st.session_state["res"]

        st.markdown("### 1. Comparação de modelos (LaTeX)")
        st.code(to_latex_table(res_df, "Comparação de algoritmos de ML"),
                language="latex")

        st.markdown("### 2. Estatística descritiva (LaTeX)")
        eda_df = _cached_eda(work[[target] + feats])
        st.code(to_latex_table(eda_df, "Análise exploratória"), language="latex")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            eda_df.to_excel(writer, sheet_name="EDA", index=False)
            res_df.to_excel(writer, sheet_name="Resultados", index=False)
            if store:
                first = list(store.keys())[0]
                if rtask == "regression":
                    diag, _ = regression_diagnostics(store[first]["y_true"],
                                                     store[first]["y_pred"], len(feats))
                    diag.to_excel(writer, sheet_name="Diagnostico", index=False)
                else:
                    classification_detailed_report(store[first]["y_true"],
                                                   store[first]["y_pred"]
                                                   ).to_excel(writer,
                                                              sheet_name="Metricas",
                                                              index=False)
        st.download_button(
            "⬇️ Baixar pacote .xlsx", data=output.getvalue(),
            file_name="relatorio_ml_learner.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Treine os modelos para liberar as tabelas.")

```

### `tests/test_core.py`

```python

"""Testes mínimos dos módulos de core (independentes da UI)."""
import numpy as np
import pandas as pd
import pytest

from core.preprocess import (MADScaler, TanhScaler, Winsorizer,
                             build_full_pipeline, multivariate_outlier_report)
from core.evaluate import holdout, make_pipeline, cross_val
from core.models import REGRESSION


def test_winsorizer_clip():
    X = np.array([[1.0], [2.0], [1000.0], [3.0]])
    w = Winsorizer(k=1.5).fit(X)
    out = w.transform(X)
    assert out.max() < 1000.0


def test_mad_scaler_mediana_zero():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    m = MADScaler().fit(X)
    out = m.transform(X)
    assert abs(np.median(out)) < 1e-9


def test_tanh_scaler_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    t = TanhScaler().fit(X)
    out = t.transform(X)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_multivariate_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[0] = [50, 50, 50]
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    out = multivariate_outlier_report(df, method="LOF", contamination=0.02)
    assert out["_outlier"].sum() >= 1


def test_pipeline_holdout_regression():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = X["a"] * 2 + X["b"] - X["c"] + rng.normal(scale=0.1, size=200)
    pre = build_full_pipeline(list("abc"), [], scaler="Z-score (Standard)",
                              winsor=True, k=1.5, imputer="Mediana")
    model, _ = REGRESSION["Regressão Linear"]
    pipe = make_pipeline(pre, model, None, False, task="regression")
    m, _ = holdout(pipe, X, y, "regression", test_size=0.2, seed=0)
    assert m["R2"] > 0.8


def test_savgol_is_causal():
    from core.preprocess import CausalSmoother
    x1 = pd.DataFrame({"x": np.arange(30, dtype=float)})
    x2 = x1.copy()
    x2.loc[12, "x"] = 10000.0
    s = CausalSmoother("Savitzky-Golay", window=5)
    y1 = s.transform(x1)[:, 0]
    y2 = s.transform(x2)[:, 0]
    # O ponto futuro t=12 não pode alterar t=10.
    assert np.isclose(y1[10], y2[10])


def test_holdout_keeps_true_oos_test_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = X["a"] - 0.5 * X["b"] + rng.normal(scale=.1, size=120)
    pre = build_full_pipeline(list("abc"), [], scaler="Z-score (Standard)")
    model, _ = REGRESSION["Regressão Linear"]
    pipe = make_pipeline(pre, model, None, False, task="regression")
    _, extra = holdout(pipe, X, y, "regression", test_size=.2, seed=42)
    assert "X_test" in extra and "y_test" in extra
    assert len(extra["X_test"]) == len(extra["y_test"]) == 24

```

## 7. Como rodar

```bash
streamlit run app.py
```

Abre em <http://localhost:8501>.

### Debug no VS Code

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Streamlit",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "app.py"]
    }
  ]
}
```

---

## 8. Fluxo sugerido para a disciplina

### Regressão (`Regressão_PSV`)
1. Suba `treino_Regressão.csv` **e** `validacao_Regressão.csv` juntos (concatenam).
2. Alvo: `MDO (m³/h)`. Features: `Speed (kn)`, `Draft Bow (m)`, `BEAUFORT`.
3. **Hold-out 3-way** (20% teste / 10% validação), escalonamento Z-score, winsorização IQR 1,5.
4. Compare R², RMSE, MAE. Ative GridSearch.
5. Rode novamente com **CV aninhada** para reportar métricas honestas.
6. Para teste final: use **Teste Separado** com `teste_Regressão.csv`.

### Classificação (`Classificação_Motor MTU`)
1. Suba o `.xlsx`; alvo `CLASS`.
2. **K-fold repetido 5×5** para base pequena (slide 10).
3. Compare Acurácia, F1 macro e matriz de confusão (hold-out).
4. Veja a **curva de calibração** (se binária) e **SHAP**.

### Roteiro de experimentos (sugestão)

| Experimento | Configuração |
|---|---|
| Baseline | Sem winsorização, sem escalonamento |
| Ruído | Winsor IQR 1,5 + suavização causal |
| Escalonamento | Z-score × Min-Max × Robust × MAD × Tanh |
| Validação | Hold-out × Hold-out 3-way × 5-fold × 5×5-fold × Aninhada |
| Grupos | GroupKFold por regime (velocidade/período) |
| Hiperparâmetros | GridSearch on/off |
| Imputação | Mediana × MICE |
| Interpretação | Permutação × SHAP |

---

## 9. Como as técnicas do paper foram integradas

| Seção do paper | Implementação aqui |
|---|---|
| 3.1.1.2 (outliers multivariados: LOF, k-NN) | `multivariate_outlier_report` (LOF, Isolation Forest) |
| 3.1.2.2 (imputação múltipla / MICE) | `IterativeImputer` como opção |
| 3.2.2 (MAD scaling) | `MADScaler` |
| 3.2.2 (Tanh-estimator de Hampel) | `TanhScaler` |
| 4.2.2 (PCP/SPCP — RPCA) | **Não implementado**. `PCA` clássico disponível. RPCA seria uma extensão futura. |
| 4.3 (BRPCA heavy-tailed) | **Não implementado**. Requer modelagem probabilística própria. |
| 4.4 (R-LSSVM, R-ELM, R-GP) | **Não implementado**. Catálogo cobre versões padrão. |

As técnicas robustas do paper que exigem modelagem probabilística (RPCA, BRPCA, R-GP) foram deliberadamente deixadas como extensão. O guia as menciona para que o relatório as cite como **trabalho futuro**.

---

## 10. Publicação

### Streamlit Community Cloud (grátis)
```bash
git init && git add . && git commit -m "ml-learner v3"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/ml-learner.git
git push -u origin main
```
<https://share.streamlit.io> → New app → repositório → `app.py`.

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ml-learner .
docker run -p 8501:8501 ml-learner
```

> ⚠️ Em hospedagem pública, **não suba dados sensíveis**.

---

## 11. Solução de problemas

| Problema | Solução |
|---|---|
| `ModuleNotFoundError` | Ative o venv e rode `pip install -r requirements.txt` |
| `shap` não instala no Windows | Comente `shap>=0.45` em `requirements.txt`. O app funciona com permutação. |
| `.mat` dá erro de "table" | Exporte para CSV no MATLAB: `writetable(suaTabela,'saida.csv')` |
| CSV com números como texto | O leitor tenta ponto e vírgula como decimal automaticamente |
| Lasso com R² ≈ 0 | Normal com `alpha=1` padrão; ative GridSearch |
| Treino lento | Compare sem tuning primeiro; use tuning só nos melhores modelos e reduza folds |
| `statsmodels` ausente (VIF) | `pip install statsmodels` |
| Resultados diferentes do MATLAB | Esperado (solvers/init). Compare tendências. |

---

## 12. Skill de IA — `CLAUDE.md`

```markdown
# Projeto ml-learner v3
App Streamlit (Python) equivalente ao Regression/Classification Learner do MATLAB,
com validação robusta (grupos, temporal, aninhada) e interpretação (SHAP, VIF).

## Regras
- Lógica em `core/`, UI só em `app.py`. Não duplicar arquivos.
- Pré-processamento SEMPRE dentro de sklearn Pipeline (evitar vazamento).
- Suavização deve ser CAUSAL; o Savitzky-Golay usa janela trailing e nunca `center=True`.
- Classificação: split estratificado; balanceamento (SMOTE/ADASYN) só no treino.
- Métricas regressão: R², RMSE, MAE. Classificação: acurácia, F1 macro, matriz.
- Permutação e SHAP usam o conjunto de teste quando disponível; SHAP transforma X pelo pré-processador antes de explicar.
- Textos em português do Brasil.
- Rodar: `streamlit run app.py`. Testar: `pytest -q`.
```

---

## 13. Extensões futuras (roadmap honesto)

1. **RPCA / BRPCA** (paper, seç. 4.2–4.3) — modelos probabilísticos.
2. **R-LSSVM / R-ELM** (paper, seç. 4.4.1) — robustez a outliers.
3. **Robust ICA** (paper, seç. 4.4.2.1).
4. **Otimização bayesiana** com Optuna.
5. **Ensemble stacking** com meta-modelo.
6. **Experiment tracking** (MLflow-like) em CSV/SQLite.
7. **Curva ROC/AUC** e **PR** para classificação binária.
8. **Relatório PDF automático** com Jinja2 + WeasyPrint.
9. **Suporte a tensores 3-way** (Batch × Variável × Tempo) — paper, seç. 5.4.
