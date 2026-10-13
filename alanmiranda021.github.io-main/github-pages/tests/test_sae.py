import numpy as np
import pytest
from core.sparse_autoencoder import SparseAutoencoderSampler


@pytest.mark.parametrize("fraction,expected", [(.5, 12), (1., 16)])
def test_sae_balance_seed_and_quality(fraction, expected):
    X = np.random.default_rng(0).normal(size=(24, 4))
    y = np.array([0]*16 + [1]*8)
    kwargs = dict(epochs=8, latent_dim=3, balance_fraction=fraction, random_state=7)
    a = SparseAutoencoderSampler(**kwargs)
    xa, ya = a.fit_resample(X, y)
    xb, yb = SparseAutoencoderSampler(**kwargs).fit_resample(X, y)
    np.testing.assert_allclose(xa, xb)
    np.testing.assert_array_equal(ya, yb)
    assert (ya == 1).sum() == expected
    np.testing.assert_array_equal(xa[:len(X)], X)
    assert np.isfinite(xa).all()
    assert a.diagnostics_[1]["synthetic"] == expected - 8
    assert np.all(xa[len(X):] <= X[y == 1].max(axis=0))
    with pytest.raises(ValueError):
        SparseAutoencoderSampler(mode="article3_reproduction").fit_resample(X, y)
    with pytest.raises(ValueError):
        SparseAutoencoderSampler().fit_resample(X[:2], y[:2])
