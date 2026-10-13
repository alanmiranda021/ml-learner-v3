"""Contrato explícito e serializável da tarefa; não deduz semântica de números."""
from dataclasses import asdict, dataclass, field
import numpy as np
import pandas as pd


def suggest_task(y):
    if pd.api.types.is_numeric_dtype(y):
        return "regression", "Alvo numérico: regressão sugerida. Se os números são códigos de classe, selecione classificação."
    return "classification", "Alvo não numérico: classificação sugerida; confirme o significado dos rótulos."


@dataclass
class TaskConfig:
    task_type: str
    target: str
    features: list[str]
    group_column: str | None = None
    time_column: str | None = None
    entity_column: str | None = None
    sample_id_column: str | None = None
    unit_metadata: dict = field(default_factory=dict)
    split_strategy: str = "random"
    random_state: int = 42
    positive_class: str | None = None
    signal_config: dict | None = None
    augmentation_config: dict | None = None
    reduction_config: dict | None = None
    model_config: dict | None = None

    def validate(self, df):
        if self.task_type not in ("regression", "classification"):
            raise ValueError("Tipo deve ser regression ou classification.")
        if self.split_strategy not in ("random", "group", "time"):
            raise ValueError("Split deve ser random, group ou time.")
        metadata = [c for c in (self.group_column, self.time_column, self.entity_column,
                                self.sample_id_column) if c is not None]
        required = [self.target, *self.features, *metadata]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes: {sorted(missing)}")
        if not df.columns.is_unique or not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Colunas e preditores precisam ser únicos e não vazios.")
        if self.target in self.features or set(metadata) & set(self.features) or self.target in metadata:
            raise ValueError("Alvo, identificadores, grupos e tempo não podem ser preditores.")
        if df[self.target].isna().any():
            raise ValueError("Alvo contém ausentes. Remova-os explicitamente e registre a quantidade.")
        if self.task_type == "regression":
            if not pd.api.types.is_numeric_dtype(df[self.target]) or not np.isfinite(df[self.target]).all():
                raise ValueError("Regressão requer alvo numérico finito.")
            if self.augmentation_config and self.augmentation_config.get("method", "none") != "none":
                raise ValueError("Aumento supervisionado de classificação não é válido para regressão.")
        elif df[self.target].nunique() < 2:
            raise ValueError("Classificação requer pelo menos duas classes.")
        for c in metadata:
            if df[c].isna().any():
                raise ValueError(f"Metadado '{c}' contém ausentes.")
        if self.split_strategy == "group" and not self.group_column:
            raise ValueError("Split agrupado requer coluna de grupo.")
        if self.split_strategy == "time" and not self.time_column:
            raise ValueError("Split temporal requer coluna de tempo.")
        if self.signal_config and (self.split_strategy != "group" or not self.group_column):
            raise ValueError("Atributos de sinais exigem split por aquisição de origem.")
        numeric = df[self.features].select_dtypes("number")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("Preditores contêm infinitos; corrija ou converta explicitamente em ausentes.")
        return self

    def to_dict(self):
        return asdict(self)
