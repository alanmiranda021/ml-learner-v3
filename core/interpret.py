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
            if hasattr(est, "estimators_") and len(est.estimators_) > 0:
                est_for_shap = est.estimators_[0]
            else:
                est_for_shap = est

            explainer = shap.TreeExplainer(est_for_shap)
            values = explainer.shap_values(Xt)
        except Exception:
            # Fallback genérico, ainda no ESPAÇO TRANSFORMADO.
            if hasattr(est, "estimators_") and len(est.estimators_) > 0:
                est_for_shap = est.estimators_[0]
            else:
                est_for_shap = est
            predict_fn = (est_for_shap.predict_proba if task == "classification"
                          and hasattr(est_for_shap, "predict_proba") else est_for_shap.predict)
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
