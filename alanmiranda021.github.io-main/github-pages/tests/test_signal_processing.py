import numpy as np
import pytest
from core.features import time_features
from core.signal_processing import SignalConfig, filter_signal, signals_to_table
from core.splitters import cv_indices


def test_known_features_and_constant_policy():
    f = time_features(np.ones(8))
    assert len(f) == 18
    assert f["rms"] == 1 and f["energy"] == 8 and f["energy_entropy"] == 3
    assert f["kurtosis"] == f["skewness"] == 0
    assert all(v == 0 for v in time_features(np.zeros(8)).values())
    sine = np.sin(2 * np.pi * np.arange(1000) / 100)
    assert time_features(sine)["rms"] == pytest.approx(2**-.5)


@pytest.mark.parametrize("x", [[1, 2], [1, np.nan, 2], [1, np.inf, 2]])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        time_features(x)


def test_filter_rejects_nyquist_and_padding():
    with pytest.raises(ValueError, match="fs/2"):
        filter_signal(np.ones(100), SignalConfig(100, filter="butterworth"))
    with pytest.raises(ValueError, match="padding"):
        filter_signal(np.ones(4), SignalConfig(10000, filter="butterworth"))
    result = filter_signal(np.random.default_rng(42).normal(size=1000), SignalConfig(10000, filter="butterworth"))
    assert np.isfinite(result).all()


def test_overlapping_windows_preserve_groups():
    rng = np.random.default_rng(0)
    table = signals_to_table([rng.normal(size=100) for _ in range(4)], [0, 0, 1, 1],
                             ["a", "b", "c", "d"], SignalConfig(1000, 32, 16))
    assert table.attrs["discarded_tail_samples"]["a"] == 4
    assert len(table) == 20
    for tr, te in cv_indices(table.target, "classification", k=2, groups=table.source_signal_id):
        assert not set(table.source_signal_id.iloc[tr]) & set(table.source_signal_id.iloc[te])
    with pytest.raises(ValueError):
        signals_to_table([np.ones(100)], [], ["a"], SignalConfig(1000, 32))
