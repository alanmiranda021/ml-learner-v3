/* node tests/naval_pyodide.cjs <pyodide module path> */
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const {loadPyodide} = require(process.argv[2] || 'pyodide');
  const py = await loadPyodide();
  await py.loadPackage(['numpy', 'scipy', 'pandas', 'scikit-learn']);
  const root = path.resolve(__dirname, '..');
  py.FS.mkdirTree('/naval/core');
  for (const name of ['__init__.py', 'sae.py', 'augment.py']) {
    py.FS.writeFile('/naval/core/'+name, fs.readFileSync(path.join(root, 'core', name)));
  }
  await py.runPythonAsync(`
import sys, time
sys.path.insert(0, '/naval')
import numpy as np
import pandas as pd
from core.augment import SAESampler
t = time.perf_counter()
rng = np.random.default_rng(8)
n = 60
distance = rng.uniform(10, 90, n)
X = pd.DataFrame(dict(ship_type=['A']*25+['B']*35, route_id=['R']*n,
 distance=distance, engine_efficiency=rng.uniform(70, 95, n),
 fuel_type=['Diesel', 'HFO']*30, weather_conditions=['Sun', 'Rain', 'Wind']*20))
y = distance*10+rng.normal(0, 20, n)
sampler = SAESampler(cell_columns=('ship_type',), hidden_grid=(3,), maxiter=50)
xa, ya = sampler.fit_resample(X, y)
assert len(xa) == 70 and (ya > 0).all()
assert sampler.diagnostics_[0]['quality']['memory_fraction'] == 0
print('Pyodide SAE naval: 60 -> 70 linhas; segundos:', round(time.perf_counter()-t, 3))
`);
})().catch(e => {console.error(e); process.exit(1);});
