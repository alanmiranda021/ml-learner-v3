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
