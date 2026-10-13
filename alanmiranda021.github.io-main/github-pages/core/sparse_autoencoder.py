"""SAE local NumPy: sigmoide, reconstrução linear, penalidade KL e Adam.

Geração por perturbação latente é uma extensão, não reprodução exata do artigo.
Não depende de torch e nunca é executado na inferência do imblearn Pipeline.
"""
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances


class SparseAutoencoderSampler(BaseEstimator):
    def __init__(self, latent_dim=8, epochs=100, batch_size=32, learning_rate=.01,
                 sparsity=.05, sparsity_weight=.1, weight_decay=.0001,
                 patience=15, noise_scale=.05, balance_fraction=1., random_state=42,
                 max_synthetic=10000, mode="sae_extended", physical_bounds=None):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight
        self.weight_decay = weight_decay
        self.patience = patience
        self.noise_scale = noise_scale
        self.balance_fraction = balance_fraction
        self.random_state = random_state
        self.max_synthetic = max_synthetic
        self.mode = mode
        self.physical_bounds = physical_bounds

    def _fit_network(self, data, rng):
        n, p = data.shape
        z = self.latent_dim
        params = [rng.normal(0, 1/np.sqrt(p), (p, z)), np.zeros(z),
                  rng.normal(0, 1/np.sqrt(z), (z, p)), np.zeros(p)]
        m = [np.zeros_like(v) for v in params]
        v = [np.zeros_like(q) for q in params]
        best_loss, best, stale, step = np.inf, None, 0, 0
        history = []
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                x = data[perm[start:start+self.batch_size]]
                w, b, wd, bd = params
                h = expit(x @ w + b)
                rec = h @ wd + bd
                rho = np.clip(h.mean(axis=0), 1e-7, 1-1e-7)
                dr = 2*(rec-x)/x.size
                kl_grad = (-self.sparsity/rho + (1-self.sparsity)/(1-rho))
                dh = (dr @ wd.T + self.sparsity_weight*kl_grad/len(x))*h*(1-h)
                grads = [x.T@dh + self.weight_decay*w, dh.sum(axis=0),
                         h.T@dr + self.weight_decay*wd, dr.sum(axis=0)]
                step += 1
                for j, grad in enumerate(grads):
                    m[j] = .9*m[j] + .1*grad
                    v[j] = .999*v[j] + .001*grad*grad
                    params[j] -= self.learning_rate*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            w, b, wd, bd = params
            h = expit(data@w+b)
            rho = np.clip(h.mean(axis=0), 1e-7, 1-1e-7)
            kl = self.sparsity*np.log(self.sparsity/rho) + (1-self.sparsity)*np.log((1-self.sparsity)/(1-rho))
            loss = float(np.mean((h@wd+bd-data)**2) + self.sparsity_weight*kl.sum()
                         + self.weight_decay/2*(np.sum(w*w)+np.sum(wd*wd)))
            if not np.isfinite(loss):
                raise ValueError("SAE divergiu; reveja escala e taxa de aprendizado.")
            history.append(loss)
            if loss < best_loss - 1e-7:
                best_loss, best, stale = loss, [q.copy() for q in params], 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        return best, history

    def fit_resample(self, X, y):
        from core.env import IN_BROWSER
        if IN_BROWSER:
            raise ValueError("SAE disponível somente no Python local.")
        if self.mode != "sae_extended":
            raise ValueError("article3_reproduction não habilitado: geração e hiperparâmetros não integralmente verificados.")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("SAE requer preditores numéricos finitos e rótulos alinhados.")
        if any(not isinstance(v, int) or v < 1 for v in (self.latent_dim, self.epochs, self.batch_size, self.patience)):
            raise ValueError("Dimensão, épocas, batch e patience devem ser inteiros positivos.")
        if not (0 < self.sparsity < 1 and 0 < self.balance_fraction <= 1 and self.learning_rate > 0
                and self.noise_scale > 0 and self.sparsity_weight >= 0 and self.weight_decay >= 0):
            raise ValueError("Parâmetros SAE fora do domínio permitido.")
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 3:
            raise ValueError("SAE exige duas classes e pelo menos três exemplos por classe no treino.")
        additions = np.ceil((counts.max()-counts)*self.balance_fraction).astype(int)
        if additions.sum() > self.max_synthetic:
            raise ValueError("Quantidade sintética excede o limite configurado.")
        rng = np.random.default_rng(self.random_state)
        generated, labels = [], []
        self.networks_, self.diagnostics_ = {}, []
        self.n_features_in_ = X.shape[1]
        for cls, count, amount in zip(classes, counts, additions):
            data = X[y == cls]
            if not amount:
                self.diagnostics_.append({"class": str(cls), "real": int(count), "synthetic": 0})
                continue
            mean, scale = data.mean(axis=0), data.std(axis=0)
            scale[scale == 0] = 1
            norm = (data-mean)/scale
            params, history = self._fit_network(norm, rng)
            w, b, wd, bd = params
            h = expit(norm@w+b)
            anchors = rng.integers(len(data), size=int(amount))
            latent = np.clip(h[anchors] + rng.normal(0, self.noise_scale, (amount, self.latent_dim)), 0, 1)
            synth = (latent@wd+bd)*scale + mean
            # Empirical envelope is a recorded heuristic, not physical validity.
            lo, hi = data.min(axis=0), data.max(axis=0)
            outside = int(((synth < lo) | (synth > hi)).any(axis=1).sum())
            synth = np.clip(synth, lo, hi)
            if self.physical_bounds is not None:
                bounds = np.asarray(self.physical_bounds, dtype=float)
                if bounds.shape != (2, X.shape[1]) or np.any(bounds[0] > bounds[1]):
                    raise ValueError("physical_bounds deve ser [limites inferiores, superiores] no espaço transformado.")
                if np.any((data < bounds[0]) | (data > bounds[1])):
                    raise ValueError("Treino viola os limites físicos informados.")
                synth = np.clip(synth, bounds[0], bounds[1])
            near = pairwise_distances(synth, data).min(axis=1)
            rec = (h@wd+bd)*scale+mean
            self.diagnostics_.append({"class": str(cls), "real": int(count), "synthetic": int(amount),
                "reconstruction_mse": float(np.mean((rec-data)**2)), "nearest_distance_mean": float(near.mean()),
                "near_copies": int((near < 1e-8).sum()), "outside_envelope_before_clip": outside,
                "mean_shift": (synth.mean(axis=0)-data.mean(axis=0)).tolist(),
                "epochs_run": len(history), "loss": min(history)})
            self.networks_[str(cls)] = {"params": params, "mean": mean, "scale": scale}
            generated.append(synth)
            labels.append(np.full(amount, cls, dtype=y.dtype))
        return (np.vstack([X, *generated]), np.concatenate([y, *labels])) if generated else (X.copy(), y.copy())
