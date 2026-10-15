"""Smoke artificial com modelos reais; não é evidência sobre os 1440 registros.

python examples/naval_smoke.py [caminho/relatorio.pdf]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core.exp_artigo3 import NavalConfig, run_naval
from core.report_pdf import naval_pdf


def main():
    rng = np.random.default_rng(17)
    n = 98
    distance = rng.uniform(10, 100, n)
    fuel = distance*10+rng.normal(0, 20, n)
    df = pd.DataFrame(dict(ship_type=["A"]*42+["B"]*56, route_id=["R"+str(i%2) for i in range(n)],
        distance=distance, fuel_type=["Diesel" if i%2 else "HFO" for i in range(n)],
        weather_conditions=[["sun", "rain", "wind"][i%3] for i in range(n)],
        engine_efficiency=rng.uniform(70, 95, n), fuel_consumption=fuel, CO2_emissions=fuel*2.7))
    cfg = NavalConfig(filename="fixture artificial naval_smoke", outer_folds=2, inner_folds=2,
        hidden_grid=(3,), gamma_grid=(.1,), components_grid=(2,), draws=1, maxiter=50)
    result = run_naval(df, cfg, lambda p, t: print(f"{p:.0%} {t}", flush=True))
    assert len(result["validation"]) == 16
    assert all(r["complete"] for r in result["validation"])
    assert not result["failures"], result["failures"]
    assert result["test"] and result["config"]["final_test_evaluations"] == 1
    pdf = naval_pdf(result)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_bytes(pdf)
    print(f"Smoke artificial: 8 cenários × 2 modelos, {result['seconds']:.3f}s, PDF {len(pdf)} bytes")


if __name__ == "__main__":
    main()
