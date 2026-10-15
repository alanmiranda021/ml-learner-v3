"""Sparse autoencoder naval: NumPy, gradiente analítico e L-BFGS-B.

Escolhas adaptadas: beta=1, rho=.05 e dimensão oculta selecionada no treino.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator


class SparseAutoencoder(BaseEstimator):
    def __init__(self, hidden_dim=4, alpha=.01, beta=1., rho=.05,
                 maxiter=400, random_state=42):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.maxiter = maxiter
        self.random_state = random_state

    def _unpack(self, theta, p):
        h = self.hidden_dim
        a, b, c = p*h, p*h+h, 2*p*h+h
        return theta[:a].reshape(p, h), theta[a:b], theta[b:c].reshape(h, p), theta[c:]

    def loss_gradient(self, theta, X, components=False):
        w, b, v, d = self._unpack(theta, X.shape[1])
        h = expit(X @ w + b)
        residual = h @ v + d - X
        mean = np.clip(h.mean(axis=0), 1e-12, 1-1e-12)
        kl = self.rho*np.log(self.rho/mean)+(1-self.rho)*np.log((1-self.rho)/(1-mean))
        terms = dict(reconstruction=float(np.sum(residual**2)/(2*len(X))),
                     l2=float(self.alpha/2*(np.sum(w*w)+np.sum(v*v))),
                     kl=float(self.beta*kl.sum()))
        total = sum(terms.values())
        dr = residual/len(X)
        dh = (dr @ v.T + self.beta/len(X)*(-self.rho/mean+(1-self.rho)/(1-mean)))*h*(1-h)
        gradient = np.concatenate([(X.T@dh+self.alpha*w).ravel(), dh.sum(axis=0),
                                   (h.T@dr+self.alpha*v).ravel(), dr.sum(axis=0)])
        return ({"total": total, **terms} if components else total), gradient

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("SAE exige matriz 2D finita e não vazia.")
        if (not isinstance(self.hidden_dim, (int, np.integer)) or self.hidden_dim < 1
                or not isinstance(self.maxiter, (int, np.integer)) or self.maxiter < 1
                or not 0 < self.rho < 1 or self.alpha < 0 or self.beta < 0):
            raise ValueError("Hiperparâmetros SAE inválidos.")
        rng = np.random.default_rng(self.random_state)
        p, h = X.shape[1], self.hidden_dim
        theta = np.concatenate([rng.normal(0, 1/np.sqrt(p), p*h), np.zeros(h),
                                rng.normal(0, 1/np.sqrt(h), h*p), np.zeros(p)])
        self.history_ = [self.loss_gradient(theta, X, True)[0]]
        def record(t):
            self.history_.append(self.loss_gradient(t, X, True)[0])
        self.optimization_ = minimize(self.loss_gradient, theta, args=(X,), jac=True,
                                      method="L-BFGS-B", callback=record,
                                      options={"maxiter": self.maxiter})
        self.params_ = self._unpack(self.optimization_.x, p)
        self.n_features_in_ = p
        return self

    def preactivation(self, X):
        w, b, _, _ = self.params_
        return np.asarray(X) @ w + b

    def encode(self, X):
        return expit(self.preactivation(X))

    def decode(self, H):
        return np.asarray(H) @ self.params_[2] + self.params_[3]

    def reconstruct(self, X):
        return self.decode(self.encode(X))
