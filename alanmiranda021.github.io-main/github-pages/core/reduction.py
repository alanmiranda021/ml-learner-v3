"""Redução na matriz completa, após aumento; limites validados em cada fit."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils.validation import check_is_fitted

KPCA_GAMMAS = [.01, .1, 1., 10., 100.]


class ValidatedReduction(BaseEstimator, TransformerMixin):
    def __init__(self, method="none", n_components=2, gamma=.1, max_samples=3000, random_state=42):
        self.method = method
        self.n_components = n_components
        self.gamma = gamma
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Redução exige matriz 2D finita.")
        if self.method not in ("none", "pca", "kpca"):
            raise ValueError("Redução deve ser none, pca ou kpca.")
        self.n_features_in_ = X.shape[1]
        self.reducer_ = None
        if self.method == "none":
            self.n_components_ = X.shape[1]
            return self
        n = self.n_components
        fraction = self.method == "pca" and isinstance(n, float) and 0 < n < 1
        limit = min(X.shape) if self.method == "pca" else X.shape[0] - 1
        if not fraction and (not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 1 <= n <= limit):
            raise ValueError(f"n_components={n} inválido neste fold; limite={limit}. Não houve truncamento.")
        if self.method == "kpca":
            if not np.isfinite(self.gamma) or self.gamma <= 0:
                raise ValueError("gamma deve ser positivo e finito.")
            if len(X) > self.max_samples:
                raise ValueError(f"KPCA limitado a {self.max_samples} amostras de treino (kernel quadrático).")
            self.reducer_ = KernelPCA(n_components=n, kernel="rbf", gamma=self.gamma,
                                     eigen_solver="dense", random_state=self.random_state)
        else:
            self.reducer_ = PCA(n_components=n, svd_solver="full")
        self.reducer_.fit(X)
        self.n_components_ = self.reducer_.transform(X[:1]).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "reducer_")
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Número de preditores diferente do ajuste.")
        return X if self.reducer_ is None else self.reducer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_components_")
        if self.method == "none" and input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"{self.method}_component_{i+1}" for i in range(self.n_components_)], dtype=object)
