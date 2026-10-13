"""Ajustes de ambiente: no navegador (Pyodide) não há multiprocessamento."""
import sys
NJOBS = 1 if sys.platform == "emscripten" else -1
IN_BROWSER = sys.platform == "emscripten"
