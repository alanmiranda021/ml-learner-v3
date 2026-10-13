from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_initializes_without_upload():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception


def test_scientific_ui_regression_runs():
    app = AppTest.from_string('''
import pandas as pd
import numpy as np
from scientific_ui import render_scientific
df = pd.DataFrame({"x": np.arange(60, dtype=float), "z": np.tile([0., 1., 2.], 20), "target": np.arange(60)*.5})
render_scientific(df)
''', default_timeout=60).run()
    assert not app.exception
    next(c for c in app.checkbox if c.label.startswith("Confirmo")).check().run()
    next(b for b in app.button if b.label == "Executar experimento").click()
    app.run()
    assert not app.exception
    assert app.success
