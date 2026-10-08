"""Leitura de CSV, Excel, Parquet, HDF5 e MATLAB em DataFrames."""
from __future__ import annotations
import csv
import io
import numpy as np
import pandas as pd


def _sniff_csv(raw: bytes) -> str:
    """Detecta o separador de um CSV (`,` `;` `\\t` `|`) usando as primeiras linhas."""
    text = raw[:20000].decode("utf-8", errors="ignore")
    first_line = text.split("\n", 1)[0] + "\n"
    try:
        return csv.Sniffer().sniff(first_line, delimiters=";,\t|").delimiter
    except csv.Error:
        return ";" if text.count(";") > text.count(",") else ","


def load_table(name: str, raw: bytes) -> pd.DataFrame:
    """Carrega CSV/TXT/TSV/XLSX/XLS/MAT/PARQUET em um DataFrame.

    Estratégia de robustez: para CSV, tenta primeiro decimal `.`, e se houver
    colunas numéricas que viraram texto, tenta decimal `,` (comum em pt-BR).
    """
    lname = name.lower()
    if lname.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if lname.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(raw))
    if lname.endswith((".h5", ".hdf5", ".hdf")):
        return _load_hdf5(raw)
    if lname.endswith((".csv", ".txt", ".tsv")):
        sep = _sniff_csv(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding_errors="replace")
        obj_cols = [c for c in df.columns
                    if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        if obj_cols:
            df2 = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=",",
                              encoding_errors="replace")
            n_num_1 = sum(df[c].dtype.kind in "fi" for c in obj_cols)
            n_num_2 = sum(df2[c].dtype.kind in "fi" for c in obj_cols)
            if n_num_2 > n_num_1:
                df = df2
        return df
    if lname.endswith(".mat"):
        return _load_mat(raw)
    raise ValueError(f"Formato não suportado: {name}")


def _load_hdf5(raw: bytes) -> pd.DataFrame:
    """Carrega o maior dataset tabular numérico de um arquivo HDF5."""
    import h5py

    with h5py.File(io.BytesIO(raw), "r") as handle:
        candidates = []

        def collect(_, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.dtype.kind in "fiu":
                candidates.append(np.array(obj))

        handle.visititems(collect)
    if not candidates:
        raise ValueError("O HDF5 não contém um dataset numérico bidimensional legível.")
    arr = max(candidates, key=lambda value: value.size)
    return pd.DataFrame(arr, columns=[f"x{i + 1}" for i in range(arr.shape[1])])


def validate_test_columns(test_df: pd.DataFrame, features, target: str) -> None:
    """Garante que a base de teste tenha todas as entradas exigidas pelo modelo."""
    required = list(features) + [target]
    missing = [column for column in required if column not in test_df.columns]
    if missing:
        raise ValueError(
            "O arquivo de teste não possui as colunas obrigatórias: "
            + ", ".join(map(str, missing))
        )


def _load_mat(raw: bytes) -> pd.DataFrame:
    """Carrega .mat (v7 clássico ou v7.3 HDF5). Ignora objetos `table` do MATLAB."""
    import scipy.io as sio
    try:
        m = sio.loadmat(io.BytesIO(raw))
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(io.BytesIO(raw)) as h:
            m = {k: np.array(v).T for k, v in h.items()
                 if isinstance(v, h5py.Dataset) and v.ndim == 2}
    candidates = {k: v for k, v in m.items()
                  if not k.startswith("__")
                  and isinstance(v, np.ndarray)
                  and v.ndim == 2
                  and v.dtype.kind in "fiu"}
    if not candidates:
        raise ValueError(
            "Este .mat guarda uma 'table' do MATLAB (objeto opaco). "
            "Rode no MATLAB:  writetable(suaTabela, 'saida.csv')  e envie o CSV."
        )
    key = max(candidates, key=lambda k: candidates[k].size)
    arr = candidates[key]
    return pd.DataFrame(arr, columns=[f"x{i+1}" for i in range(arr.shape[1])])
