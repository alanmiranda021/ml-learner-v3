"""Janelas com proveniência, filtro determinístico e grupos obrigatórios."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, firwin, filtfilt
from core.features import time_features


@dataclass
class SignalConfig:
    fs_hz: float
    window_samples: int = 1024
    overlap_samples: int = 0
    min_samples: int = 3
    filter: str = "none"
    low_hz: float = 10.
    high_hz: float = 2000.
    order: int = 4
    unit: str = ""
    label_policy: str = "source"
    feature_set: str = "generic_time18"

    def validate(self):
        if not np.isfinite(self.fs_hz) or self.fs_hz <= 0:
            raise ValueError("fs_hz deve ser confirmado, positivo e finito.")
        if not isinstance(self.window_samples, int) or self.window_samples < max(3, self.min_samples):
            raise ValueError("Janela deve conter pelo menos min_samples >= 3.")
        if not isinstance(self.overlap_samples, int) or not 0 <= self.overlap_samples < self.window_samples:
            raise ValueError("Sobreposição deve ser inteira e menor que a janela.")
        if self.filter not in ("none", "butterworth", "fir"):
            raise ValueError("Filtro não suportado.")
        if self.filter != "none" and not (0 < self.low_hz < self.high_hz < self.fs_hz / 2):
            raise ValueError("Cortes devem satisfazer 0 < low < high < fs/2; não são reduzidos automaticamente.")
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("Ordem deve ser inteira positiva.")
        if self.label_policy != "source":
            raise ValueError("Apenas rótulo confirmado por aquisição é suportado.")
        return self


def filter_signal(signal, config):
    config.validate()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < max(3, config.min_samples) or not np.isfinite(x).all():
        raise ValueError("Sinal curto, não unidimensional ou com NaN/inf.")
    try:
        if config.filter == "butterworth":
            sos = butter(config.order, [config.low_hz, config.high_hz], fs=config.fs_hz, btype="bandpass", output="sos")
            return sosfiltfilt(sos, x)
        if config.filter == "fir":
            taps = firwin(config.order + 1, [config.low_hz, config.high_hz], fs=config.fs_hz,
                          pass_zero=False, window="hamming")
            return filtfilt(taps, [1.], x)
    except ValueError as exc:
        raise ValueError("Sinal insuficiente para padding do filtro; aumente a aquisição ou ajuste a ordem.") from exc
    return x.copy()


def signals_to_table(signals, labels, source_ids, config):
    config.validate()
    if not len(signals) == len(labels) == len(source_ids) or not len(signals):
        raise ValueError("Sinais, rótulos e IDs devem ter o mesmo comprimento não zero.")
    if len(set(source_ids)) != len(source_ids) or any(pd.isna(s) for s in source_ids) or any(pd.isna(y) for y in labels):
        raise ValueError("IDs únicos e rótulos confirmados sem ausentes são obrigatórios.")
    rows, dropped = [], {}
    step = config.window_samples - config.overlap_samples
    for signal, label, source in zip(signals, labels, source_ids):
        x = filter_signal(signal, config)
        if len(x) < config.window_samples:
            raise ValueError(f"Aquisição '{source}' é menor que uma janela completa.")
        starts = list(range(0, len(x) - config.window_samples + 1, step))
        for i, start in enumerate(starts):
            end = start + config.window_samples
            rows.append({**time_features(x[start:end], config.feature_set), "target": label,
                         "source_signal_id": source, "segment_index": i, "start_sample": start,
                         "end_sample": end, "fs_hz": config.fs_hz, "label_source": "source"})
        dropped[str(source)] = len(x) - (starts[-1] + config.window_samples)
    df = pd.DataFrame(rows)
    df.attrs["signal_config"] = asdict(config)
    df.attrs["discarded_tail_samples"] = dropped
    df.attrs["mandatory_group_column"] = "source_signal_id"
    return df
