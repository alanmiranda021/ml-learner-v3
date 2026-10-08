"""Registro portátil e reproduzível de experimentos."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import pandas as pd
import sklearn


def dataset_hash(df: pd.DataFrame) -> str:
    """Hash determinístico do conteúdo e esquema da base."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, df.columns)).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


def experiment_record(*, data: pd.DataFrame, config: dict, results: pd.DataFrame) -> dict:
    """Monta um registro JSON que permite identificar e repetir um experimento."""
    now = datetime.now(timezone.utc)
    return {
        "experiment_id": f"ML-{now:%Y%m%d-%H%M%S}",
        "timestamp_utc": now.isoformat(),
        "dataset": {"hash_sha256": dataset_hash(data), "rows": len(data), "columns": list(map(str, data.columns))},
        "configuration": config,
        "results": results.to_dict(orient="records"),
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "scikit_learn": sklearn.__version__},
    }


def experiment_json(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False, indent=2, default=str).encode("utf-8")
