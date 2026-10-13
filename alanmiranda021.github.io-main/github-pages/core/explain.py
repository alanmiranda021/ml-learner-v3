"""Explicações no espaço efetivamente recebido pelo estimador; sampler omitido."""
import numpy as np


def estimator_input(pipeline, raw):
    pipe = getattr(pipeline, "best_estimator_", pipeline)
    X = raw
    names = np.asarray(raw.columns, dtype=object)
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        X = step.transform(X)
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
        else:
            names = np.asarray([f"{name}_{i+1}" for i in range(X.shape[1])], dtype=object)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pipe.steps[-1][1], np.asarray(X), names


def explain_pipeline(pipeline, training_background, raw_to_explain, max_background=30, max_samples=30, seed=42):
    if not 1 <= max_background <= 100 or not 1 <= max_samples <= 100:
        raise ValueError("SHAP limitado a 100 amostras de fundo e explicação.")
    try:
        import shap
    except ImportError as exc:
        raise ValueError("SHAP exige pacote opcional no Python local; use importância por permutação no relatório.") from exc
    bg = training_background.sample(min(max_background, len(training_background)), random_state=seed)
    raw = raw_to_explain.iloc[:max_samples]
    model, background, names = estimator_input(pipeline, bg)
    _, transformed, names = estimator_input(pipeline, raw)
    predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    # A classifier without scores cannot be explained numerically through labels.
    if hasattr(model, "classes_") and not hasattr(model, "predict_proba"):
        if hasattr(model, "decision_function"):
            predict = model.decision_function
        else:
            raise ValueError("Classificador sem probabilidades ou decision_function não suportado por este explainer.")
    pipe = getattr(pipeline, "best_estimator_", pipeline)
    if hasattr(model, "classes_"):
        np.testing.assert_array_equal(model.predict(transformed), pipe.predict(raw))
    else:
        np.testing.assert_allclose(model.predict(transformed), pipe.predict(raw))
    if len(names) != transformed.shape[1]:
        raise ValueError("Nomes e dimensões transformados não coincidem.")
    explainer = shap.Explainer(predict, background, feature_names=names.tolist(), algorithm="permutation", seed=seed)
    values = explainer(transformed, max_evals=max(2*transformed.shape[1]+1, 101))
    return {"values": values, "feature_names": names.tolist(), "space": "transformed",
            "message": "Valores explicam atributos transformados; PCA/KPCA explicam componentes, sem atribuição causal aos sensores."}
