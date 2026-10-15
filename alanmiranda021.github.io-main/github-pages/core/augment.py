"""Aumento naval por célula em DataFrame; nunca aplicado em predict.

Categorias são decodificadas por argmax; nenhum vocabulário vem da validação.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from core.sae import SparseAutoencoder


def cell_labels(X, columns):
    return X[list(columns)].fillna("<ausente>").astype(str).apply(lambda r: repr(tuple(r)), axis=1).to_numpy()


def synthetic_quality(real, synthetic, origins):
    a, b, o = map(lambda v: np.asarray(v, dtype=float), (real, synthetic, origins))
    scale = a.std(axis=0)
    scale[scale == 0] = 1
    near = pairwise_distances((b-a.mean(0))/scale, (a-a.mean(0))/scale).min(axis=1)
    def corr(x):
        x = (x-x.mean(0))/scale
        norms = np.linalg.norm(x, axis=0)
        norms[norms == 0] = 1
        return (x/norms).T @ (x/norms)
    pearson = []
    for i in range(a.shape[1]):
        pearson.append(float(np.corrcoef(b[:, i], o[:, i])[0, 1])
                       if len(b) > 1 and b[:, i].std() > 0 and o[:, i].std() > 0 else None)
    return dict(mse=np.mean((b-o)**2, axis=0).tolist(), pearson=pearson,
                ks=[float(ks_2samp(a[:, i], b[:, i]).statistic) for i in range(a.shape[1])],
                correlation_frobenius=float(np.linalg.norm(corr(a)-corr(b))),
                nearest_distance_mean=float(near.mean()), memory_fraction=float(np.mean(near < 1e-8)))


class SAESampler(BaseEstimator):
    def __init__(self, cell_columns=("ship_type", "route_id"), target="fuel_consumption",
                 fraction=1., hidden_grid=(3, 4, 5, 6), noise_grid=(.25, .5, 1.),
                 maxiter=400, random_state=42, generator="sae", max_attempts=100):
        self.cell_columns = cell_columns
        self.target = target
        self.fraction = fraction
        self.hidden_grid = hidden_grid
        self.noise_grid = noise_grid
        self.maxiter = maxiter
        self.random_state = random_state
        self.generator = generator
        self.max_attempts = max_attempts

    def fit_resample(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SAESampler naval exige DataFrame com nomes de colunas.")
        if not 0 < self.fraction <= 1 or self.generator not in ("sae", "gaussian"):
            raise ValueError("Fração ou gerador inválido.")
        if not self.hidden_grid or not self.noise_grid or any(s <= 0 for s in self.noise_grid):
            raise ValueError("Grades do gerador devem ser positivas e não vazias.")
        frame = X.copy().reset_index(drop=True)
        frame[self.target] = np.asarray(y, dtype=float)
        numeric = ["distance", "engine_efficiency", self.target]
        cats = ["fuel_type", "weather_conditions"]
        required = [*self.cell_columns, *numeric, *cats]
        if any(c not in frame for c in required) or frame[required].isna().any().any():
            raise ValueError("Gerador exige colunas navais completas, sem ausências nas células.")
        if not np.isfinite(frame[numeric].to_numpy(float)).all() or (frame[self.target] <= 0).any():
            raise ValueError("Valores numéricos finitos e alvo estritamente positivo são obrigatórios.")
        labels = cell_labels(frame, self.cell_columns)
        cells, counts = np.unique(labels, return_counts=True)
        rng = np.random.default_rng(self.random_state)
        self.diagnostics_, self.networks_, self.fit_indices_ = [], {}, X.index.tolist()
        generated = []
        for label, count in zip(cells, counts):
            real = frame.loc[labels == label].reset_index(drop=True)
            amount = int(np.floor(count + self.fraction*(counts.max()-count) + .5)) - count
            if not amount:
                self.diagnostics_.append(dict(cell=label, real=int(count), synthetic=0))
                continue
            if count < 6:
                raise ValueError(f"Célula {label} com poucas linhas ({count}); mínimo 6 para aumento.")
            scaler, encoder = StandardScaler(), OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            # Hidden-size validation fits its scaler and vocabulary on its own subtrain.
            it, iv = train_test_split(np.arange(count), test_size=.2, random_state=self.random_state)
            selected, best = None, np.inf
            if self.generator == "sae":
                inner_scaler = StandardScaler().fit(real.iloc[it][numeric])
                inner_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(real.iloc[it][cats])
                def inner_matrix(indices):
                    return np.c_[inner_scaler.transform(real.iloc[indices][numeric]), inner_encoder.transform(real.iloc[indices][cats])]
                for hidden in self.hidden_grid:
                    net = SparseAutoencoder(hidden_dim=hidden, maxiter=self.maxiter, random_state=self.random_state).fit(inner_matrix(it))
                    score = np.mean((net.reconstruct(inner_matrix(iv))-inner_matrix(iv))**2)
                    if score < best:
                        selected, best = hidden, score
            matrix = np.c_[scaler.fit_transform(real[numeric]), encoder.fit_transform(real[cats])]
            net = None
            if self.generator == "sae":
                net = SparseAutoencoder(hidden_dim=selected, maxiter=self.maxiter, random_state=self.random_state).fit(matrix)
                self.networks_[label] = net
            pre = net.preactivation(matrix) if net is not None else matrix
            spread = pre.std(axis=0)
            coef = np.polyfit(real.distance, real[self.target], 1) if real.distance.nunique() > 1 else [0., real[self.target].mean()]
            residual = real[self.target].to_numpy()-np.polyval(coef, real.distance)
            tolerance = max(3*residual.std(), 1e-8)
            real_numeric = real[numeric].to_numpy(float)
            low, high = real_numeric.min(0), real_numeric.max(0)
            def propose(s, size):
                origins = rng.integers(count, size=size)
                noisy = pre[origins] + rng.normal(size=(size, pre.shape[1]))*spread*s
                decoded = net.decode(expit(noisy)) if net is not None else noisy
                numbers = np.clip(scaler.inverse_transform(decoded[:, :3]), low, high)
                out = real.iloc[origins].reset_index(drop=True).copy()
                out[numeric] = numbers
                offset = 3
                for c, categories in zip(cats, encoder.categories_):
                    out[c] = categories[np.argmax(decoded[:, offset:offset+len(categories)], axis=1)]
                    offset += len(categories)
                valid = (numbers[:, 2] > 0) & (np.abs(numbers[:, 2]-np.polyval(coef, numbers[:, 0])) <= tolerance)
                # Reject complete copies, including identical categorical values.
                for j in np.flatnonzero(valid):
                    same = np.isclose(real_numeric, numbers[j], rtol=0, atol=1e-10).all(axis=1)
                    same &= (real[cats].to_numpy() == out.loc[j, cats].to_numpy()).all(axis=1)
                    if same.any():
                        valid[j] = False
                return out.loc[valid].reset_index(drop=True), real.iloc[origins[valid]][numeric].to_numpy(), int((~valid).sum())
            # Noise chosen solely by mean numeric KS on this training cell.
            options = []
            for s in self.noise_grid:
                probe, _, _ = propose(s, max(40, amount*2))
                if len(probe):
                    ks = np.mean([ks_2samp(real[c], probe[c]).statistic for c in numeric])
                    options.append((ks, s))
            if not options:
                raise ValueError(f"Gerador não produziu linhas plausíveis na célula {label}.")
            s = min(options)[1]
            parts, origins, rejected, attempted = [], [], 0, 0
            remaining = amount
            for _ in range(self.max_attempts):
                size = max(remaining*2, 20)
                out, source, bad = propose(s, size)
                attempted += size
                rejected += bad
                take = min(remaining, len(out))
                if take:
                    parts.append(out.iloc[:take]); origins.append(source[:take]); remaining -= take
                if not remaining:
                    break
            if remaining:
                raise ValueError(f"Filtro de plausibilidade impediu balancear {label}: faltam {remaining} linhas.")
            synth = pd.concat(parts, ignore_index=True)
            quality = synthetic_quality(real[numeric], synth[numeric], np.vstack(origins))
            self.diagnostics_.append(dict(cell=label, real=int(count), synthetic=amount,
                hidden_dim=selected, noise_scale=float(s), rejected_fraction=rejected/attempted,
                quality=quality, numeric_columns=numeric,
                history=net.history_ if net is not None else [],
                optimizer_success=bool(net.optimization_.success) if net is not None else None,
                real_numeric=real_numeric.tolist(), synthetic_numeric=synth[numeric].to_numpy().tolist()))
            generated.append(synth)
        result = pd.concat([frame, *generated], ignore_index=True)
        self.synthetic_count_ = len(result)-len(frame)
        return result.drop(columns=self.target), result[self.target].to_numpy()
