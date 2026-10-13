"""Adaptadores de aumento: apenas classificação e preditores numéricos."""
def make_sampler(method="none", task="classification", categorical=False, random_state=42, **kwargs):
    if method == "none":
        return None
    if task != "classification":
        raise ValueError("Regressão não utiliza aumento de classes.")
    if categorical:
        raise ValueError("Aumento requer preditores numéricos. One-hot sintético não é categoria válida; SMOTENC ainda não integrado.")
    if method == "sae":
        from core.sparse_autoencoder import SparseAutoencoderSampler
        return SparseAutoencoderSampler(random_state=random_state, **kwargs)
    if method in ("smote", "adasyn"):
        try:
            from imblearn.over_sampling import SMOTE, ADASYN
        except ImportError as exc:
            raise ValueError("Instale imbalanced-learn no Python local para aumento.") from exc
        return {"smote": SMOTE, "adasyn": ADASYN}[method](random_state=random_state, **kwargs)
    raise ValueError("Método de aumento desconhecido.")
