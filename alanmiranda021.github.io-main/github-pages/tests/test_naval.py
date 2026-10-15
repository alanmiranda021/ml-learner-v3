import io
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from pypdf import PdfReader
from core.sae import SparseAutoencoder
from core.augment import SAESampler, cell_labels
from core.exp_artigo3 import NavalConfig, NavalReduction, validate_naval, run_naval, rank_results, paired_comparisons
from core.evaluate import reg_metrics
from core.report_pdf import naval_pdf


@pytest.fixture
def naval_data():
    rng = np.random.default_rng(17)
    rows = []
    for ship, count in [("A", 42), ("B", 56)]:
        for i in range(count):
            distance = rng.uniform(10, 100)
            fuel = 10*distance+rng.normal(0, 25)
            rows.append(dict(ship_type=ship, route_id="R"+str(i % 2), distance=distance,
                fuel_type="Diesel" if i % 2 else "HFO", weather_conditions=["sun", "rain", "wind"][i % 3],
                engine_efficiency=rng.uniform(70, 95), fuel_consumption=fuel, CO2_emissions=fuel*2.7))
    return pd.DataFrame(rows)


def test_sae_gradient_and_loss():
    X = np.random.default_rng(2).normal(size=(12, 3))
    net = SparseAutoencoder(hidden_dim=2, maxiter=60).fit(X)
    theta = np.concatenate([p.ravel() for p in net.params_])
    theta += np.random.default_rng(3).normal(0, .1, len(theta))
    loss, gradient = net.loss_gradient(theta, X)
    finite = approx_fprime(theta, lambda t: net.loss_gradient(t, X)[0], 1e-7)
    assert np.linalg.norm(gradient-finite)/max(1., np.linalg.norm(gradient)) < 1e-5
    assert net.history_[-1]["total"] < net.history_[0]["total"]
    assert all(h["kl"] >= -1e-12 for h in net.history_)
    assert all(b["total"] <= a["total"]+1e-8 for a, b in zip(net.history_, net.history_[1:]))


def test_sae_sparsity_and_zero_kl():
    X = np.random.default_rng(2).normal(size=(30, 3))
    sparse = SparseAutoencoder(hidden_dim=2, beta=1, maxiter=150).fit(X)
    plain = SparseAutoencoder(hidden_dim=2, beta=0, maxiter=150).fit(X)
    assert abs(sparse.encode(X).mean()-.05) < abs(plain.encode(X).mean()-.05)
    w = np.zeros((3, 2)); b = np.full(2, np.log(.05/.95))
    theta = np.concatenate([w.ravel(), b, np.zeros(6), np.zeros(3)])
    assert sparse.loss_gradient(theta, X, True)[0]["kl"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("fraction", [.9, 1.])
@pytest.mark.parametrize("generator", ["sae", "gaussian"])
def test_sampler_counts_quality_and_seed(naval_data, fraction, generator):
    X, y = naval_data.drop(columns=["fuel_consumption", "CO2_emissions"]), naval_data.fuel_consumption
    sampler = SAESampler(cell_columns=("ship_type",), fraction=fraction, hidden_grid=(3,), maxiter=12, generator=generator)
    xa, ya = sampler.fit_resample(X, y)
    xb, yb = sampler.fit_resample(X, y)
    pd.testing.assert_frame_equal(xa, xb)
    np.testing.assert_array_equal(ya, yb)
    counts = xa.ship_type.value_counts()
    assert counts["A"] == int(np.floor(42+fraction*14+.5))
    assert counts["B"] == 56
    assert (ya > 0).all()
    assert set(xa.fuel_type) <= set(X.fuel_type)
    assert set(xa.weather_conditions) <= set(X.weather_conditions)
    synth = xa.iloc[len(X):].assign(fuel_consumption=ya[len(X):])
    real = X.assign(fuel_consumption=y)
    for c in ("distance", "engine_efficiency", "fuel_consumption"):
        assert synth[c].between(real.loc[real.ship_type == "A", c].min(), real.loc[real.ship_type == "A", c].max()).all()
    assert not pd.concat([real, synth]).duplicated().any()
    assert sampler.diagnostics_[0]["quality"]["memory_fraction"] == 0


def test_naval_guards_and_metrics(naval_data):
    with pytest.raises(ValueError, match="Vazamento"):
        validate_naval(naval_data, NavalConfig(excluded=()).effective())
    with pytest.raises(ValueError, match="vazia"):
        validate_naval(naval_data.iloc[:0], NavalConfig().effective())
    with pytest.raises(ValueError, match="numérico"):
        validate_naval(naval_data.assign(fuel_consumption="x"), NavalConfig().effective())
    with pytest.raises(ValueError, match="menor"):
        NavalReduction("kpca", 3).fit(np.ones((10, 3)))
    metric = reg_metrics([100, 120, 80], [110, 115, 85])
    assert metric["MAE"] == pytest.approx(20/3)
    assert metric["RMSE"] == pytest.approx(np.sqrt(50))
    assert metric["R2"] == pytest.approx(.8125)
    with pytest.warns(UserWarning, match="poucas"):
        validate_naval(naval_data.iloc[:10], NavalConfig().effective())


def test_optional_kpca_preprocessing_and_repeated_cell_folds(naval_data):
    from core.preprocess import build_preprocessor
    from core.evaluate import cell_cv_indices
    pre = build_preprocessor(["distance", "engine_efficiency"], ["ship_type"],
                             kpca=True, kpca_components=2, winsor=False)
    assert pre.fit_transform(naval_data).shape == (len(naval_data), 2)
    with pytest.raises(ValueError, match="menor"):
        build_preprocessor(["distance"], [], kpca=True, kpca_components=1).fit(naval_data)
    folds = cell_cv_indices(cell_labels(naval_data, ("ship_type",)), k=5, repeats=3)
    assert len(folds) == 15
    assert all(not set(a) & set(b) for a, b in folds)


def test_ranking_excludes_diagnostic_and_80_percent():
    rows = [dict(scenario=s, model="RF", RMSE=r, RMSE_std=.1, seconds=1.) for s, r in [("E1", 3), ("E3", 2), ("E3-orig", .1)]]
    assert rank_results(rows)[0]["scenario"] == "E3"
    details = [dict(scenario=s, model="RF", fold=i, metrics={"RMSE": v}) for s, values in [("E1", [3]*5), ("E3", [2, 2, 2, 2, 4])] for i, v in enumerate(values)]
    assert paired_comparisons(details)[0]["fraction_improved"] == .8
    assert paired_comparisons(details)[0]["conclusion"] != "sem ganho demonstrado"
    details.pop()
    assert paired_comparisons(details)[0]["conclusion"] == "sem ganho demonstrado"


def test_nested_naval_and_pdf(naval_data, monkeypatch):
    import core.exp_artigo3 as module
    from sklearn.linear_model import Ridge
    # Small budgets exercise every scenario, full nested fits and final holdout.
    monkeypatch.setattr(module, "naval_model", lambda name, seed: (Ridge(), {"alpha": [1.]}))
    fitted = []
    original = SAESampler.fit_resample
    def spy(self, X, y):
        fitted.append(set(X.index))
        return original(self, X, y)
    monkeypatch.setattr(SAESampler, "fit_resample", spy)
    cfg = NavalConfig(models=("RF-test", "invalid"), outer_folds=2, inner_folds=2,
        hidden_grid=(3,), gamma_grid=(.1,), components_grid=(2,), draws=1, maxiter=8, sensitivity=True)
    def model(name, seed):
        if name == "invalid":
            raise ValueError("falha de modelo deliberada")
        return Ridge(), {"alpha": [1.]}
    monkeypatch.setattr(module, "naval_model", model)
    result = run_naval(naval_data, cfg)
    assert {r["scenario"] for r in result["validation"]} == set(module.SCENARIOS)
    assert result["failures"] and result["test"]
    assert result["config"]["final_test_evaluations"] == 1
    test = set(result["config"]["final_test_indices"])
    assert all(not indices & test for indices in fitted)
    assert all(not set(d["train_indices"]) & set(d["validation_indices"]) for d in result["folds"])
    assert result["test"]["scenario"] != "E3-orig"
    assert result["sensitivity"]
    data = naval_pdf(result)
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert len(reader.pages) >= 5
    for phrase in ("adaptado", "E3-orig", "Limitações", "Proveniência", result["dataset_sha256"], "Python", "KPCA", "Sensibilidade"):
        assert phrase in text
    assert "reprodução" not in text


def test_naval_ui_and_stale_result(naval_data):
    from streamlit.testing.v1 import AppTest
    code = '''
import pandas as pd
from naval_ui import render_naval
render_naval(pd.DataFrame(DATA))
'''.replace("DATA", repr(naval_data.to_dict("list")))
    app = AppTest.from_string(code).run()
    assert not app.exception
    assert next(b for b in app.button if b.label.startswith("Executar protocolo")).disabled is False
    next(m for m in app.multiselect if m.label == "Colunas excluídas").set_value([]).run()
    assert not app.exception
    assert next(b for b in app.button if b.label.startswith("Executar protocolo")).disabled


def test_training_vocab_and_inner_sampler_scope(naval_data, monkeypatch):
    import core.exp_artigo3 as module
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    cfg = NavalConfig(models=("Ridge-test",), scenarios=("E1", "E3", "E4"),
        outer_folds=2, inner_folds=2, hidden_grid=(3,), gamma_grid=(.1,),
        components_grid=(2,), draws=1, maxiter=5, sensitivity=False)
    lab = cell_labels(naval_data, ("ship_type",))
    train, test = train_test_split(np.arange(len(naval_data)), test_size=.2, stratify=lab, random_state=42)
    data = naval_data.copy()
    data.loc[test, "weather_conditions"] = "ONLY_FINAL_TEST"
    monkeypatch.setattr(module, "naval_model", lambda name, seed: (Ridge(), {"alpha": [1.]}))
    vocabularies, sampler_indices, reduction_sizes = [], [], []
    from sklearn.preprocessing import OneHotEncoder
    original_fit = OneHotEncoder.fit
    def spy_encoder(self, X, y=None):
        fitted = original_fit(self, X, y)
        vocabularies.extend(str(v) for values in self.categories_ for v in values)
        return fitted
    original_sampler = SAESampler.fit_resample
    def spy_sampler(self, X, y):
        sampler_indices.append(set(X.index))
        return original_sampler(self, X, y)
    original_reduce = NavalReduction.fit
    def spy_reduce(self, X, y=None):
        reduction_sizes.append(len(X))
        return original_reduce(self, X, y)
    monkeypatch.setattr(OneHotEncoder, "fit", spy_encoder)
    monkeypatch.setattr(SAESampler, "fit_resample", spy_sampler)
    monkeypatch.setattr(NavalReduction, "fit", spy_reduce)
    result = run_naval(data, cfg)
    assert result["test"] and not result["failures"]
    assert "ONLY_FINAL_TEST" not in vocabularies
    assert reduction_sizes
    assert all(indices <= set(train) for indices in sampler_indices)
    outer = [d for d in result["folds"] if d["scenario"] == "E3"]
    assert all(any(indices < set(d["train_indices"]) for indices in sampler_indices) for d in outer)
    assert all(not set(d["validation_indices"]) & set(d["train_indices"]) for d in outer)
