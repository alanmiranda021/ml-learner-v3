"""Validação não destrutiva de datasets científicos e de engenharia."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame, target: str | None = None,
                     physical_limits: dict[str, tuple[float, float]] | None = None,
                     rare_class_minimum: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna resumo e alertas, sem alterar os dados enviados pelo usuário."""
    physical_limits = physical_limits or {}
    numeric = df.select_dtypes(include=[np.number])
    infinite = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if not numeric.empty else 0
    summary = pd.DataFrame([{
        "Linhas": len(df), "Colunas": len(df.columns),
        "Células ausentes": int(df.isna().sum().sum()),
        "Linhas duplicadas": int(df.duplicated().sum()),
        "Valores infinitos": infinite,
    }])
    alerts: list[dict[str, object]] = []
    if infinite:
        alerts.append({"Tipo": "Infinito", "Coluna": "—", "Quantidade": infinite,
                       "Mensagem": "Substitua ou trate valores ±inf antes do treinamento."})
    for column, (low, high) in physical_limits.items():
        if column in numeric:
            count = int(((numeric[column] < low) | (numeric[column] > high)).sum())
            if count:
                alerts.append({"Tipo": "Faixa física", "Coluna": column, "Quantidade": count,
                               "Mensagem": f"Fora da faixa informada [{low}, {high}]."})
    if target and target in df and not pd.api.types.is_numeric_dtype(df[target]):
        counts = df[target].value_counts(dropna=False)
        for label, count in counts[counts < rare_class_minimum].items():
            alerts.append({"Tipo": "Classe rara", "Coluna": target, "Quantidade": int(count),
                           "Mensagem": f"Classe {label!r} possui menos de {rare_class_minimum} linhas."})
    return summary, pd.DataFrame(alerts, columns=["Tipo", "Coluna", "Quantidade", "Mensagem"])


def domain_report(train: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Indica entradas numéricas fora do domínio observado no treino."""
    rows = []
    for column in train.select_dtypes(include=[np.number]).columns:
        if column not in candidate:
            continue
        low, high = train[column].min(), train[column].max()
        outside = int(((candidate[column] < low) | (candidate[column] > high)).sum())
        rows.append({"Variável": column, "Mínimo treino": low, "Máximo treino": high,
                     "Fora do domínio": outside})
    return pd.DataFrame(rows)
