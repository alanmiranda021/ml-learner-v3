"""Inspeção sem escolha automática da maior matriz MATLAB."""
import io
import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_MATRIX_ELEMENTS = 10_000_000


def profile_table(df):
    rows = []
    for c in df.columns:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append({"coluna": c, "dtype": str(s.dtype), "ausentes": int(s.isna().sum()),
                     "infinitos": int(np.isinf(s.to_numpy(dtype=float)).sum()) if numeric else 0,
                     "cardinalidade": int(s.nunique()), "constante": s.nunique() <= 1,
                     "possivel_id": s.nunique() == len(s)})
    return {"shape": list(df.shape), "duplicatas": int(df.duplicated().sum()),
            "columns": rows}


def inspect_mat(raw):
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("MAT excede o limite de 100 MiB.")
    # MATLAB HDF5 may carry a 512-byte user block.
    hdf = any(raw[o:o + 8] == b"\x89HDF\r\n\x1a\n" for o in (0, 512))
    if hdf:
        try:
            import h5py
        except ImportError as exc:
            raise ValueError("MAT v7.3 exige h5py no Python local; reexporte como v7 ou CSV.") from exc
        rows = []
        with h5py.File(io.BytesIO(raw)) as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    rows.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype),
                                 "kind": "dataset", "tabular": obj.ndim == 2 and obj.dtype.kind in "fiub"})
                else:
                    rows.append({"name": name, "shape": [], "dtype": "group", "kind": "group", "tabular": False})
            h.visititems(visit)
        return {"format": "v7.3", "variables": rows}
    from scipy.io import whosmat
    try:
        rows = [{"name": n, "shape": list(s), "dtype": t, "kind": t,
                 "tabular": len(s) == 2 and t in ("double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "logical")}
                for n, s, t in whosmat(io.BytesIO(raw))]
    except Exception as exc:
        raise ValueError("MAT inválido/corrompido ou versão não suportada.") from exc
    return {"format": "v4/v5-v7.2", "variables": rows}


def mat_to_table(raw, variable, orientation="rows"):
    info = inspect_mat(raw)
    matches = [v for v in info["variables"] if v["name"] == variable]
    if not matches or not matches[0]["tabular"]:
        raise ValueError("Selecione uma matriz numérica 2D. Structs, células e objetos exigem exportação explícita.")
    if orientation not in ("rows", "columns"):
        raise ValueError("Orientação deve ser rows ou columns.")
    if np.prod(matches[0]["shape"]) > MAX_MATRIX_ELEMENTS:
        raise ValueError("Matriz excede 10 milhões de elementos; exporte um subconjunto.")
    if info["format"] == "v7.3":
        import h5py
        with h5py.File(io.BytesIO(raw)) as h:
            arr = np.asarray(h[variable])
    else:
        from scipy.io import loadmat
        arr = loadmat(io.BytesIO(raw), variable_names=[variable])[variable]
    if orientation == "columns":
        arr = arr.T
    df = pd.DataFrame(arr, columns=[f"x{i+1}" for i in range(arr.shape[1])])
    df.attrs["source"] = {"variable": variable, "orientation": orientation, "format": info["format"]}
    return df
