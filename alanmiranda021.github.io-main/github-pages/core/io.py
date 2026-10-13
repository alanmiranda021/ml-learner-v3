"""Leitura de arquivos: CSV (qualquer separador/decimal), Excel e .mat."""
import io
import csv
import numpy as np
import pandas as pd


def _sniff_csv(raw: bytes):
    text = raw[:20000].decode("utf-8", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(text.split("\n", 1)[0] + "\n", delimiters=";,\t|").delimiter
    except csv.Error:
        sep = ";" if text.count(";") > text.count(",") else ","
    return sep


def _looks_numeric_header(cols) -> bool:
    ok = 0
    for c in cols:
        try:
            float(str(c).replace(",", ".")); ok += 1
        except ValueError:
            pass
    return len(cols) >= 2 and ok == len(cols)


def _read_csv_robust(raw: bytes) -> pd.DataFrame:
    sep = _sniff_csv(raw)
    last = None
    for enc in ("utf-8-sig", "latin-1", "cp1252"):  # tenta codificações até uma funcionar
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last = e
    else:
        raise last
    obj = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    if obj:  # números com vírgula decimal?
        df2 = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=",", encoding=enc)
        if sum(df2[c].dtype.kind in "fi" for c in obj) > sum(df[c].dtype.kind in "fi" for c in obj):
            df = df2
    if _looks_numeric_header(df.columns):  # arquivo sem cabeçalho: a 1ª linha é dado
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, header=None, decimal="," if df is not None and obj else ".")
        df.columns = [f"x{i+1}" for i in range(df.shape[1])]
    return df


def load_table(name: str, raw: bytes) -> pd.DataFrame:
    name = name.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
        if _looks_numeric_header(df.columns):
            df = pd.read_excel(io.BytesIO(raw), header=None)
            df.columns = [f"x{i+1}" for i in range(df.shape[1])]
        return df
    if name.endswith((".csv", ".txt", ".tsv")):
        return _read_csv_robust(raw)
    if name.endswith(".mat"):
        return _load_mat(raw)
    raise ValueError(f"Formato não suportado: {name}")


_MAGIC = 3707764736  # 0xDD000000: marca de referência a objeto MATLAB (MCOS)


def _mcos_cells(m):
    """Abre o bloco interno (__function_workspace__) onde o MATLAB guarda objetos como `table`."""
    import io as _io
    from scipy.io.matlab._mio5 import MatFile5Reader
    fw = m["__function_workspace__"].tobytes()
    raw = b" " * 116 + b"\0" * 8 + fw[:4] + fw[8:]  # remonta um .mat válido a partir do subsistema
    rd = MatFile5Reader(_io.BytesIO(raw), struct_as_record=True, squeeze_me=False)
    rd.mat_stream.seek(0)
    rd.initialize_read()
    rd.mat_stream.seek(128)
    hdr, _ = rd.read_var_header()
    st = rd.read_var_array(hdr, process=False)
    wrapper = st[0, 0]["MCOS"][0]
    # SciPy exposes the MATLAB opaque payload under different field names.
    field = "arr" if "arr" in wrapper.dtype.names else "_ObjectMetadata"
    return list(wrapper[field].ravel())


def _is_ref(x):
    return isinstance(x, np.ndarray) and x.shape == (6, 1) and x.dtype.kind == "u" and int(x.ravel()[0]) == _MAGIC


def _decode_matlab_table(m) -> pd.DataFrame:
    cells = _mcos_cells(m)
    data = names = None
    for c in cells:  # procura a célula 1xN de colunas e a 1xN de nomes das variáveis
        if not (isinstance(c, np.ndarray) and c.dtype == object and c.ndim == 2 and c.shape[0] == 1 and c.shape[1] > 0):
            continue
        items = list(c.ravel())
        if data is None and all(isinstance(x, np.ndarray) and x.ndim == 2 and x.dtype.kind in "fiub" for x in items):
            data = items
        elif names is None and all(isinstance(x, np.ndarray) and x.dtype.kind == "U" for x in items):
            names = [str(x.ravel()[0]) if x.size else "" for x in items]
    if data is None or names is None or len(data) != len(names):
        raise ValueError("estrutura de tabela não reconhecida")
    nrows = max((x.shape[0] for x in data if not _is_ref(x)), default=0)
    cols = {}
    for name, x in zip(names, data):
        if _is_ref(x):  # coluna categórica/texto: códigos + nomes das categorias ficam em células pareadas
            oid = int(x.ravel()[4])
            k = 2 + 2 * (oid - 2)
            try:
                cats, codes = cells[k], cells[k + 1]
                cats = [str(v.ravel()[0]) if isinstance(v, np.ndarray) and v.size else str(v) for v in cats.ravel()]
                codes = np.asarray(codes).ravel().astype(int)
                assert len(codes) == nrows
            except Exception:
                raise ValueError(f"coluna '{name}' é de um tipo (data/texto/string) que não consegui decodificar")
            cols[name] = pd.Categorical([cats[c - 1] if c > 0 else None for c in codes]).astype(object)
        elif x.shape[1] == 1:
            cols[name] = x[:, 0]
        else:
            for j in range(x.shape[1]):
                cols[f"{name}_{j+1}"] = x[:, j]
    return pd.DataFrame(cols)


def _load_mat(raw: bytes) -> pd.DataFrame:
    import scipy.io as sio
    try:
        m = sio.loadmat(io.BytesIO(raw))
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        try:
            import h5py
        except ImportError:
            raise ValueError("Arquivo .mat v7.3 (HDF5): precisa do pacote h5py (disponível no uso local, "
                             "não na versão do navegador). Reexporte com `save(...,'-v7')` ou `writetable`.")
        with h5py.File(io.BytesIO(raw)) as h:
            arrs = {k: np.array(v).T for k, v in h.items() if isinstance(v, h5py.Dataset) and v.ndim == 2}
        if not arrs:
            raise ValueError("Este .mat v7.3 guarda uma `table`. No MATLAB: writetable(T,'saida.csv') e envie o CSV.")
        key = max(arrs, key=lambda k: arrs[k].size)
        return pd.DataFrame(arrs[key], columns=[f"x{i+1}" for i in range(arrs[key].shape[1])])

    if "__function_workspace__" in m:  # arquivo contém objetos MATLAB (ex.: table)
        try:
            return _decode_matlab_table(m)
        except Exception as e:
            raise ValueError(f"Não consegui ler a `table` deste .mat ({e}). "
                             "No MATLAB rode: writetable(suaTabela,'saida.csv') e envie o CSV.")
    cand = {k: v for k, v in m.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fiu"}
    if not cand:
        raise ValueError("Nenhuma matriz numérica 2D encontrada neste .mat.")
    key = max(cand, key=lambda k: cand[k].size)
    arr = cand[key]
    return pd.DataFrame(arr, columns=[f"x{i+1}" for i in range(arr.shape[1])])
