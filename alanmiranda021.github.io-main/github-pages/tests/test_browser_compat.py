import re
import sys
from pathlib import Path
import pytest
from core.augmentation import make_sampler
from core.sparse_autoencoder import SparseAutoencoderSampler


def test_boot_packages_every_python_module():
    root = Path(__file__).resolve().parents[1]
    boot = (root / "boot.js").read_text(encoding="utf-8")
    paths = re.findall(r"'([^']+\.py)'", boot)
    for path in paths:
        if path != "core/__init__.py":
            assert (root / path).is_file(), path
    for module in ["scientific_ui.py", "core/experiments.py", "core/explain.py", "core/signal_processing.py"]:
        assert module in paths


def test_optional_dependencies_not_required_for_generic_browser(monkeypatch):
    import core.env
    import core.data_inspection
    import core.experiments
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "h5py", None)
    monkeypatch.setattr(core.env, "IN_BROWSER", True)
    assert make_sampler("none") is None
    assert core.experiments.scenario_configs("regression")
    with pytest.raises(ValueError, match="local"):
        SparseAutoencoderSampler().fit_resample([[1]], [0])
