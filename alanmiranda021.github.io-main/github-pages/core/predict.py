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


def make_bundle(r, model, trained_on="100% dos dados"):
    d = r["data"]
    bundle = {"model": model, "features": r["features"], "target": r["target"], "task": r["task"],
              "algorithm": r["best"], "test_metrics": r["test_metrics"], "trained_on": trained_on, "n_rows": len(d),
              "dtypes": {c: str(d[c].dtype) for c in r["features"]}}
    if r["task"] == "classification":
        bundle["classes"] = [str(c) for c in model.classes_]
    return bundle


def to_bytes(bundle) -> bytes:
    import joblib
    b = io.BytesIO(); joblib.dump(bundle, b)
    return b.getvalue()


def predict_table(bundle, df_new: pd.DataFrame) -> pd.DataFrame:
    """Prevê para uma planilha nova. As colunas precisam ter os mesmos nomes do treino."""
    df_new = df_new.copy()
    df_new.columns = [str(c) for c in df_new.columns]
    missing = [c for c in bundle["features"] if c not in df_new.columns]
    if missing:
        raise ValueError(f"Faltam colunas no arquivo novo: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    X = df_new[bundle["features"]]
    out = df_new.copy()
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
    return b


def retrain_bundle(bundle: dict, df: pd.DataFrame, test_size=0.2, seed=42):
    """Refaz o treino com a MESMA receita (algoritmo + hiperparâmetros) em dados rotulados novos.
    Compara modelo antigo x novo num teste separado dos dados novos e devolve o novo pacote (treinado com 100%)."""
    from sklearn.model_selection import train_test_split
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
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if clf else None)
    metr = clf_metrics if clf else reg_metrics
    old = metr(yte, bundle["model"].predict(Xte))
    cand = clone(bundle["model"]).fit(Xtr, ytr)
    new = metr(yte, cand.predict(Xte))
    final = clone(bundle["model"]).fit(X, y)
    nb = dict(bundle); nb.update(model=final, n_rows=len(df), trained_on=f"retreino com {len(df)} linhas novas",
                                  test_metrics=new, dtypes={c: str(X[c].dtype) for c in bundle["features"]})
    if clf:
        nb["classes"] = [str(c) for c in final.classes_]
    return nb, {"antigo": old, "novo": new, "n_teste": len(Xte), "n_total": len(df)}
