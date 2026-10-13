import numpy as np
import pandas as pd
from core.experiments import run_experiment
from core.task_config import TaskConfig
from core.report_pdf import safe_pdf


def test_scientific_pdf_uses_actual_protocol():
    df = pd.DataFrame({"x": np.arange(40, dtype=float), "target": np.arange(40)*.5})
    r = run_experiment(df, TaskConfig("regression", "target", ["x"]), model_names=["Ridge"], k=2)
    pdf, warning = safe_pdf(r)
    assert pdf.startswith(b"%PDF")
    assert not warning
