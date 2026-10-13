"""Manifestos sem dados brutos: hashes, ambiente, partições e configuração."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, default=str, allow_nan=False).encode("utf-8")


def manifest(df, config, folds, metrics):
    versions = {}
    for p in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "streamlit", "shap"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=3).decode().strip()
    except Exception:
        commit = None
    try:
        timestamp = datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
    except Exception:
        timestamp = datetime.now().astimezone().isoformat()
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()
                            + json_bytes({"columns": list(df.columns), "dtypes": list(map(str, df.dtypes))})).hexdigest()
    return {"schema_version": 1, "experiment_id": str(uuid.uuid4()), "timestamp": timestamp,
            "config_hash": hashlib.sha256(json_bytes(config)).hexdigest(), "dataset_hash": digest,
            "config": config, "git_commit": commit, "python": platform.python_version(),
            "hardware": {"machine": platform.machine(), "platform": platform.platform()},
            "versions": versions, "folds": folds, "metrics": metrics,
            "warnings": ["Dados sintéticos e benchmarks não comprovam validade física ou generalização no domínio real."]}
