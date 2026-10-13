"""Uso do modelo: treino final com 100% dos dados, pacote portátil e previsão em arquivos novos."""
import io
import pandas as pd
from sklearn.base import clone


def final_model(r):
    """Reajusta o pipeline vencedor (mesmos hiperparâmetros) com TODOS os dados, sem separar teste."""
    est = getattr(r["model"], "best_estimator_", r["model"])
    d = r["data"]
    m = clone(est)
    m.fit(d[r["features"]], d[r["target"]])
    return m


def make_bundle(r, model, trained_on="100% dos dados", schema_data=None):
    d = r["data"] if schema_data is None else schema_data
    bundle = {"bundle_version": 2, "model": model, "features": r["features"], "target": r["target"], "task": r["task"],
              "algorithm": r["best"], "test_metrics": r["test_metrics"], "trained_on": trained_on, "n_rows": len(d),
              "dtypes": {c: str(d[c].dtype) for c in r["features"]},
              "schema": {c: "numeric" if pd.api.types.is_numeric_dtype(d[c]) else "categorical" for c in r["features"]},
              "task_config": r.get("task_config"), "manifest": r.get("manifest"),
              "units": (r.get("task_config") or {}).get("unit_metadata", {})}
    bundle["categories"] = {c: [str(v) for v in d[c].dropna().unique()]
                            for c in r["features"] if bundle["schema"][c] == "categorical"}
    if r["task"] == "classification":
        bundle["classes"] = [str(c) for c in model.classes_]
    return bundle


def to_bytes(bundle) -> bytes:
    import joblib
    b = io.BytesIO(); joblib.dump(bundle, b)
    return b.getvalue()


def predict_table(bundle, df_new: pd.DataFrame, unit_metadata=None) -> pd.DataFrame:
    """Prevê para uma planilha nova. As colunas precisam ter os mesmos nomes do treino."""
    df_new = df_new.copy()
    df_new.columns = [str(c) for c in df_new.columns]
    missing = [c for c in bundle["features"] if c not in df_new.columns]
    if missing:
        raise ValueError(f"Faltam colunas no arquivo novo: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    X = df_new[bundle["features"]]
    if not df_new.columns.is_unique:
        raise ValueError("Arquivo novo contém colunas duplicadas.")
    for c, kind in bundle.get("schema", {}).items():
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(f"Coluna '{c}' deve ser numérica, como no treino.")
    for c, expected in bundle.get("units", {}).items():
        actual = (unit_metadata or df_new.attrs.get("units", {})).get(c)
        if actual is not None and expected and actual != expected:
            raise ValueError(f"Unidade incompatível em '{c}': esperada '{expected}', recebida '{actual}'.")
    out = df_new.copy()
    extras = [c for c in df_new.columns if c not in bundle["features"] and c != bundle["target"]]
    warnings = [f"Colunas extras preservadas, fora do modelo: {extras}"] if extras else []
    for c, known in bundle.get("categories", {}).items():
        unseen = set(df_new[c].dropna().astype(str)) - set(known)
        if unseen:
            warnings.append(f"Categorias novas em '{c}', tratadas pelo pipeline: {sorted(unseen)}")
    out.attrs["inference_warnings"] = warnings
    out[f"previsto_{bundle['target']}"] = bundle["model"].predict(X)
    if bundle["task"] == "classification" and hasattr(bundle["model"], "predict_proba"):
        try:
            pr = bundle["model"].predict_proba(X)
            out["confianca"] = pr.max(axis=1).round(4)
        except Exception:
            pass
    return out


def load_bundle(raw: bytes) -> dict:
    """Carrega um .joblib gerado por este app. ATENÇÃO: joblib/pickle executa código; só abra arquivos de confiança."""
    import joblib
    b = joblib.load(io.BytesIO(raw))
    need = {"model", "features", "target", "task", "algorithm"}
    if not isinstance(b, dict) or not need <= set(b):
        raise ValueError("Este .joblib não foi gerado por este app (faltam model/features/target/task/algorithm).")
    if b.get("bundle_version", 1) not in (1, 2):
        raise ValueError("Versão de bundle não suportada.")
    return b


def retrain_bundle(bundle: dict, df: pd.DataFrame, test_size=0.2, seed=42):
    """Refaz o treino com a MESMA receita (algoritmo + hiperparâmetros) em dados rotulados novos.
    Compara modelo antigo x novo num teste separado dos dados novos e devolve o novo pacote (treinado com 100%)."""
    from core.splitters import holdout_indices
    from core.evaluate import clf_metrics, reg_metrics
    df = df.copy(); df.columns = [str(c) for c in df.columns]
    t = bundle["target"]
    if t not in df.columns:
        raise ValueError(f"A coluna alvo '{t}' não existe no arquivo; para retreinar é preciso ter os valores reais.")
    miss = [c for c in bundle["features"] if c not in df.columns]
    if miss:
        raise ValueError(f"Faltam colunas: {miss[:8]}")
    df = df.dropna(subset=[t])
    X, y = df[bundle["features"]], df[t]
    clf = bundle["task"] == "classification"
    cfg = bundle.get("task_config") or {}
    strategy = cfg.get("split_strategy", "random")
    column = cfg.get("group_column") if strategy == "group" else cfg.get("time_column") if strategy == "time" else None
    if column and column not in df:
        raise ValueError(f"Retreino exige metadado '{column}' para manter o split original.")
    tr, te = holdout_indices(y, bundle["task"], test_size, seed,
                             groups=df[column].to_numpy() if strategy == "group" else None,
                             times=df[column].to_numpy() if strategy == "time" else None)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    metr = clf_metrics if clf else reg_metrics
    old = metr(yte, bundle["model"].predict(Xte))
    cand = clone(bundle["model"]).fit(Xtr, ytr)
    new = metr(yte, cand.predict(Xte))
    final = clone(bundle["model"]).fit(X, y)
    nb = dict(bundle); nb.update(model=final, n_rows=len(df), trained_on=f"retreino com {len(df)} linhas novas",
                                  test_metrics=new, dtypes={c: str(X[c].dtype) for c in bundle["features"]})
    nb["categories"] = {c: [str(v) for v in X[c].dropna().unique()]
                        for c in bundle["features"] if not pd.api.types.is_numeric_dtype(X[c])}
    nb["schema"] = {c: "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical" for c in bundle["features"]}
    from core.experiment_tracking import manifest
    old_id = (bundle.get("manifest") or {}).get("experiment_id")
    nb["manifest"] = manifest(df, {"task": cfg, "retrain_seed": seed, "previous_experiment": old_id,
                                   "test_size": test_size, "final_refit": "all_rows"},
                               {"train": tr.tolist(), "test": te.tolist()}, {"new_holdout": new})
    if clf:
        nb["classes"] = [str(c) for c in final.classes_]
    return nb, {"antigo": old, "novo": new, "n_teste": len(Xte), "n_total": len(df)}
