"""generic_time18: definições explícitas, sem alegar equivalência ao artigo."""
import numpy as np
from scipy.stats import skew, kurtosis


def time_features(signal, feature_set="generic_time18"):
    if feature_set != "generic_time18":
        raise ValueError("article3_verified indisponível: fórmulas não integralmente verificadas.")
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Sinal deve ser vetor finito com pelo menos 3 amostras.")
    rms = float(np.sqrt(np.mean(x*x)))
    peak, mean_abs = float(np.max(np.abs(x))), float(np.mean(np.abs(x)))
    clearance = float(np.mean(np.sqrt(np.abs(x))) ** 2)
    constant = np.ptp(x) == 0
    energy = float(np.sum(x*x))
    p = x*x / energy if energy else np.zeros(len(x))
    p = p[p > 0]
    ratio = lambda a, b: float(a / b) if b else 0.
    return dict(mean=float(np.mean(x)), rms=rms, std=float(np.std(x, ddof=0)), peak=peak,
                skewness=0. if constant else float(skew(x, bias=True)),
                kurtosis=0. if constant else float(kurtosis(x, fisher=True, bias=True)),
                crest_factor=ratio(peak, rms), clearance_factor=ratio(peak, clearance),
                shape_factor=ratio(rms, mean_abs), impulse_factor=ratio(peak, mean_abs),
                peak_to_peak=float(np.ptp(x)), minimum=float(np.min(x)), total=float(np.sum(x)),
                variance=float(np.var(x, ddof=0)), standard_error=float(np.std(x, ddof=0)/np.sqrt(len(x))),
                median=float(np.median(x)), energy=energy, energy_entropy=float(-np.sum(p*np.log2(p))))
