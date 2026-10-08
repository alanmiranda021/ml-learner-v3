"""Interface web do ml-learner (Streamlit). Substitui Regression/Classification
Learner do MATLAB com técnicas modernas de validação e interpretação.
"""
from __future__ import annotations
import warnings
warnings.simplefilter("default")
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import set_global_seed, __version__
from core.io import load_table, validate_test_columns
from core.preprocess import (SCALERS, build_full_pipeline, multivariate_outlier_report,
                             outlier_report)
from core.models import CLASSIFICATION, REGRESSION
from core.evaluate import (cross_val, evaluate_test_set, group_cross_val, holdout,
                           holdout_3way, make_pipeline, nested_cv, timeseries_cv)
from core.analysis import (calibration_table, classification_detailed_report,
                           detailed_eda, learning_curve_table,
                           regression_diagnostics, to_latex_table, vif_table)
from core.interpret import permutation_importance_df, try_shap

st.set_page_config(page_title="ml-learner v3", layout="wide")
st.title(f"📊 ml-learner v{__version__} — Regressão e Classificação")
st.caption("Alternativa em Python ao Regression/Classification Learner do MATLAB. "
           "Com validação em grupo, CV aninhada, SHAP e calibração.")

# ---------------------------------------------------------------------------
# 1. DADOS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Dados de treino")
    files = st.file_uploader(
        "CSV, Excel, HDF5, .mat ou Parquet (múltiplos concatenam)",
        type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls", "mat", "h5", "hdf5", "hdf", "parquet", "pq"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Envie pelo menos 1 arquivo de treino.")
        st.stop()
    frames = []
    for f in files:
        try:
            frames.append(load_table(f.name, f.getvalue()))
        except Exception as e:
            st.error(f"{f.name}: {e}")
    if not frames:
        st.stop()
    if len(frames) > 1:
        reference = list(frames[0].columns)
        incompatible = [f.name for f, frame in zip(files, frames)
                        if list(frame.columns) != reference]
        if incompatible:
            st.error(
                "Os arquivos enviados têm colunas diferentes. "
                "Para evitar NaNs silenciosos por alinhamento do pandas, "
                "use arquivos com as mesmas colunas e na mesma ordem. "
                f"Incompatíveis: {', '.join(incompatible)}"
            )
            st.stop()
    df = pd.concat(frames, ignore_index=True)

    st.header("1b) Teste separado (opcional)")
    use_test_file = st.checkbox("Usar arquivo de teste dedicado", value=False)
    df_test = None
    if use_test_file:
        test_file = st.file_uploader("Arquivo de teste",
                                     type=["csv", "txt", "tsv", "xlsx", "xlsm",
                                           "xls", "mat", "h5", "hdf5", "hdf", "parquet", "pq"],
                                     key="test_upload")
        if test_file:
            try:
                df_test = load_table(test_file.name, test_file.getvalue())
            except Exception as e:
                st.error(f"Erro ao ler teste: {e}")

st.subheader("Visão geral dos dados de treino")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas", len(df))
c2.metric("Colunas", df.shape[1])
c3.metric("Valores ausentes", int(df.isna().sum().sum()))
st.dataframe(df.head(50), use_container_width=True)
if df_test is not None:
    st.caption(f"ℹ️ Teste carregado: {len(df_test)} linhas × {df_test.shape[1]} colunas.")

# ---------------------------------------------------------------------------
# 2. PROBLEMA E FEATURES
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("2) Problema")
    target = st.selectbox("Variável alvo (y)", df.columns, index=len(df.columns) - 1)
    # Numérico é regressão por padrão: cardinalidade baixa não basta para
    # chamar algo de classificação (ex.: Beaufort 0–9 é um alvo de regressão
    # possível). O usuário pode selecionar Classificação manualmente.
    task_guess = ("classification"
                  if not pd.api.types.is_numeric_dtype(df[target])
                  else "regression")
    task = st.radio("Tipo",
                    ["regression", "classification"],
                    index=["regression", "classification"].index(task_guess),
                    format_func=lambda t: "Regressão" if t == "regression" else "Classificação")
    feats = st.multiselect("Características (X)",
                           [c for c in df.columns if c != target],
                           default=[c for c in df.columns if c != target])

if not feats:
    st.warning("Selecione ao menos uma característica.")
    st.stop()

if use_test_file and df_test is not None:
    try:
        validate_test_columns(df_test, feats, target)
    except ValueError as exc:
        st.error(f"Arquivo de teste incompatível: {exc}")
        st.stop()

# Aviso de vazamento (correlação alta)
if task == "regression" and pd.api.types.is_numeric_dtype(df[target]):
    corr = (df[feats].select_dtypes("number")
            .corrwith(df[target]).abs().sort_values(ascending=False))
    if len(corr) and corr.iloc[0] > 0.98:
        st.warning(f"⚠️ '{corr.index[0]}' tem correlação {corr.iloc[0]:.3f} "
                   f"com o alvo — possível vazamento de dados.")

# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("3) Pré-processamento")
    num_cols_all = df[feats].select_dtypes("number").columns.tolist()
    winsor = st.checkbox("Limitar outliers (IQR) — no pipeline", value=True)
    k_iqr = st.slider("Fator IQR", 1.0, 3.0, 1.5, 0.1)
    scaler = st.selectbox("Escalonamento", list(SCALERS), index=1)
    imputer = st.selectbox("Imputação",
                           ["Mediana", "MICE (IterativeImputer)"], index=0)
    smooth = st.selectbox("Suavização causal (séries temporais)",
                          ["Nenhuma", "Média móvel", "Mediana móvel", "Savitzky-Golay"])
    win = st.slider("Janela", 3, 51, 5, 2) if smooth != "Nenhuma" else 5
    use_pca = st.checkbox("Aplicar PCA (após o ColumnTransformer)", value=False)
    pca_var = st.slider("Componentes (fração de variância)", 0.50, 0.99, 0.95, 0.01) if use_pca else 0.95

    st.header("4) Validação")
    if use_test_file and df_test is not None:
        strategy = "Teste Separado"
        st.info("Usando arquivo de teste dedicado.")
    else:
        strategy = st.radio(
            "Estratégia",
            ["Hold-out", "Hold-out 3-way", "K-fold", "K-fold repetido",
             "GroupKFold (por grupos)", "TimeSeriesSplit", "CV aninhada"],
        )
        if strategy == "Hold-out":
            test_size = st.slider("% teste", 10, 40, 20, 5) / 100
        elif strategy == "Hold-out 3-way":
            test_size = st.slider("% teste", 10, 30, 20, 5) / 100
            val_size = st.slider("% validação", 5, 20, 10, 5) / 100
        elif strategy in ("K-fold", "K-fold repetido", "GroupKFold (por grupos)",
                          "TimeSeriesSplit"):
            k = st.slider("K / Folds", 3, 10, 5)
            reps = st.slider("Repetições", 2, 10, 5) if strategy == "K-fold repetido" else 1
        elif strategy == "CV aninhada":
            k_outer = st.slider("Folds externos", 3, 10, 5)
            k_inner = st.slider("Folds internos", 2, 5, 3)

    st.header("5) Semente e algoritmos")
    seed = st.number_input("Semente aleatória", 0, 9999, 42)
    zoo = REGRESSION if task == "regression" else CLASSIFICATION
    chosen = st.multiselect("Modelos", list(zoo), default=list(zoo)[:4])
    tune = st.checkbox("Otimizar hiperparâmetros (GridSearch, CV interna)", value=False)
    sampler_name = "Nenhum"
    if task == "classification":
        sampler_name = st.selectbox("Balanceamento (só no treino)",
                                    ["Nenhum", "SMOTE", "ADASYN"])

# Fixa semente global
set_global_seed(int(seed))

# Coluna de grupos (para GroupKFold)
group_col = None
if strategy == "GroupKFold (por grupos)":
    group_col = st.sidebar.selectbox("Coluna de grupos", df.columns.tolist())

# Invalida resultados antigos quando qualquer configuração relevante muda.
import hashlib


def _frame_signature(frame: pd.DataFrame | None) -> str | None:
    """Identifica o conteúdo para não reutilizar o resultado de outra base."""
    if frame is None:
        return None
    content = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(content).hexdigest()


_config_payload = repr({
    "shape": df.shape,
    "columns": tuple(map(str, df.columns)),
    "data_signature": _frame_signature(df),
    "test_signature": _frame_signature(df_test),
    "target": target,
    "task": task,
    "feats": tuple(feats),
    "winsor": winsor,
    "k_iqr": k_iqr,
    "scaler": scaler,
    "imputer": imputer,
    "smooth": smooth,
    "win": win,
    "pca": (use_pca, pca_var),
    "strategy": strategy,
    "seed": int(seed),
    "chosen": tuple(chosen),
    "tune": tune,
    "sampler": sampler_name,
    "group_col": group_col,
})
_config_key = hashlib.sha256(_config_payload.encode()).hexdigest()
if st.session_state.get("_config_key") != _config_key:
    st.session_state.pop("res", None)
    st.session_state["_config_key"] = _config_key

# Cache de diagnósticos determinísticos e interpretações caras.
@st.cache_data(show_spinner=False)
def _cached_eda(frame):
    return detailed_eda(frame)

@st.cache_data(show_spinner=False)
def _cached_outliers(frame, k):
    return outlier_report(frame, k)

@st.cache_data(show_spinner=False)
def _cached_shap(model_bytes, frame, task_name, max_samples, seed_value):
    import io as _io
    fitted = joblib.load(_io.BytesIO(model_bytes))
    return try_shap(fitted, frame, task=task_name,
                    max_samples=max_samples, seed=seed_value)

@st.cache_data(show_spinner=False)
def _cached_permutation(model_bytes, frame, y_values, seed_value):
    import io as _io
    fitted = joblib.load(_io.BytesIO(model_bytes))
    y_series = pd.Series(y_values, index=frame.index)
    return permutation_importance_df(fitted, frame, y_series,
                                     n_repeats=8, seed=seed_value)

# ---------------------------------------------------------------------------
# 4. EDA + OUTLIERS MULTIVARIADOS
# ---------------------------------------------------------------------------
work = df.dropna(subset=[target]).copy()
X = work[feats]
y = work[target]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = [c for c in feats if c not in num_cols]

# Pipeline base (sem sampler, sem tune — o GridSearch é aplicado depois)
pre = build_full_pipeline(num_cols, cat_cols,
                          scaler=scaler, winsor=winsor, k=k_iqr,
                          imputer=imputer, smooth=smooth, window=win,
                          use_pca=use_pca, pca_var=pca_var, seed=int(seed))

with st.expander("🔎 Diagnóstico univariado, multivariado e multicolinearidade"):
    st.markdown("**Outliers univariados (IQR)**")
    st.dataframe(_cached_outliers(work[num_cols], k_iqr), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Outliers multivariados (LOF)**")
        lof_df = multivariate_outlier_report(work[num_cols], method="LOF",
                                             contamination=0.05, seed=int(seed))
        n_lof = int(lof_df["_outlier"].sum()) if "_outlier" in lof_df else 0
        st.metric("Suspeitos (LOF)", n_lof)
    with col_b:
        st.markdown("**Outliers multivariados (Isolation Forest)**")
        iso_df = multivariate_outlier_report(work[num_cols], method="Isolation Forest",
                                             contamination=0.05, seed=int(seed))
        n_iso = int(iso_df["_outlier"].sum()) if "_outlier" in iso_df else 0
        st.metric("Suspeitos (IsoForest)", n_iso)

    if task == "regression" and len(num_cols) >= 2:
        st.markdown("**VIF (multicolinearidade)**")
        st.dataframe(vif_table(work[num_cols]), use_container_width=True)

    if len(num_cols) > 1:
        st.markdown("**Matriz de correlação**")
        st.plotly_chart(
            px.imshow(work[num_cols].corr(), aspect="auto",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# 5. ABAS
# ---------------------------------------------------------------------------
tab_eda, tab_train, tab_diag, tab_export = st.tabs([
    "📈 EDA", "⚙️ Treino & Comparação", "🔍 Diagnóstico", "📄 Exportação"
])

with tab_eda:
    st.subheader("Estatística descritiva")
    st.dataframe(_cached_eda(work[[target] + feats]), use_container_width=True)
    st.subheader("Distribuição do alvo")
    st.plotly_chart(
        px.histogram(work, x=target, marginal="box", title=f"Distribuição de {target}"),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# 6. TREINO
# ---------------------------------------------------------------------------
with tab_train:
    if st.button("▶️ Treinar e comparar", type="primary"):
        sampler = None
        if sampler_name != "Nenhum":
            from imblearn.over_sampling import ADASYN, SMOTE
            sampler = {"SMOTE": SMOTE, "ADASYN": ADASYN}[sampler_name](
                random_state=int(seed))
        rows, store = [], {}
        bar = st.progress(0.0)

        if tune and strategy in ("K-fold", "K-fold repetido",
                                 "GroupKFold (por grupos)", "TimeSeriesSplit"):
            st.warning(
                "Otimização de hiperparâmetros foi desativada nesta estratégia: "
                "GridSearch dentro do mesmo K-fold usado para estimar desempenho "
                "é uma avaliação enviesada. Use **CV aninhada** para tuning honesto."
            )
        if tune and strategy == "Hold-out 3-way":
            st.info(
                "No Hold-out 3-way, o GridSearch já faz CV interna no treino; "
                "o conjunto de validação continua útil para diagnóstico, mas "
                "não participa da seleção dos hiperparâmetros."
            )

        for i, name in enumerate(chosen):
            model, grid = zoo[name]
            try:
                # --- estratégias de avaliação ---
                if strategy == "Teste Separado":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    X_te, y_te = df_test[feats], df_test[target]
                    m, extra = evaluate_test_set(pipe, X, y, X_te, y_te, task)
                    store[name] = extra
                    rows.append({"Modelo": name, **{a: round(b, 4) for a, b in m.items()}})

                elif strategy == "Hold-out":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    m, extra = holdout(pipe, X, y, task, test_size, int(seed))
                    store[name] = extra
                    rows.append({"Modelo": name, **{a: round(b, 4) for a, b in m.items()}})

                elif strategy == "Hold-out 3-way":
                    pipe = make_pipeline(pre, model, grid, tune, task=task, sampler=sampler)
                    m, extra = holdout_3way(pipe, X, y, task, val_size, test_size, int(seed))
                    store[name] = extra
                    rows.append({"Modelo": name,
                                 **{f"{a} (teste)": round(b, 4) for a, b in m.items()},
                                 **{f"{a} (val)": round(b, 4)
                                    for a, b in extra["m_val"].items()}})

                elif strategy in ("K-fold", "K-fold repetido"):
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = cross_val(pipe, X, y, task, k,
                                  reps if strategy == "K-fold repetido" else 1,
                                  int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "GroupKFold (por grupos)":
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = group_cross_val(pipe, X, y, work[group_col], task, k, int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "TimeSeriesSplit":
                    pipe = make_pipeline(pre, model, grid, False, task=task, sampler=sampler)
                    m = timeseries_cv(pipe, X, y, task, k)
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})

                elif strategy == "CV aninhada":
                    pipe = make_pipeline(pre, model, grid, tune, cv_inner=k_inner,
                                         task=task, sampler=sampler)
                    m, best = nested_cv(pipe, X, y, task, k_outer, k_inner, int(seed))
                    rows.append({"Modelo": name,
                                 **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})
                    if best:
                        st.caption(f"Melhores hiperparâmetros por fold ({name}): {best[:2]}...")
            except Exception as e:
                rows.append({"Modelo": name, "erro": str(e)[:120]})
            bar.progress((i + 1) / len(chosen))

        st.session_state["res"] = (pd.DataFrame(rows), store, task)

    if "res" in st.session_state:
        res_df, _, _ = st.session_state["res"]
        st.subheader("Resultados")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("⬇️ Baixar CSV",
                           res_df.to_csv(index=False).encode(),
                           "resultados.csv")

# ---------------------------------------------------------------------------
# 7. DIAGNÓSTICO E INTERPRETAÇÃO
# ---------------------------------------------------------------------------
with tab_diag:
    if "res" in st.session_state and st.session_state["res"][1]:
        _, store, rtask = st.session_state["res"]
        pick = st.selectbox("Modelo para diagnóstico", list(store))
        e = store[pick]

        if rtask == "regression":
            diag, resid = regression_diagnostics(e["y_true"], e["y_pred"], len(feats))
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Métricas detalhadas")
                st.dataframe(diag, use_container_width=True)
            with c2:
                st.subheader("Resíduos vs Valores Ajustados")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=e["y_pred"], y=resid, mode="markers",
                                         name="Resíduos"))
                fig.add_shape(type="line", x0=float(min(e["y_pred"])), y0=0,
                              x1=float(max(e["y_pred"])), y1=0,
                              line=dict(color="red", dash="dash"))
                fig.update_layout(xaxis_title="Previsto", yaxis_title="Resíduo")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Real × Previsto")
            d = pd.DataFrame({"real": e["y_true"], "previsto": e["y_pred"]})
            fig = px.scatter(d, x="real", y="previsto")
            lo, hi = float(d.min().min()), float(d.max().max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Curva de aprendizado")
            try:
                pipe_for_lc = make_pipeline(pre, zoo[pick][0], None, False, task=task)
                lc = learning_curve_table(pipe_for_lc, X, y, "regression",
                                          cv=min(5, max(2, len(X) // 20)),
                                          n_points=6, seed=int(seed))
                st.plotly_chart(px.line(lc, x="Tamanho do treino",
                                        y=["Treino (média)", "Validação (média)"],
                                        markers=True),
                                use_container_width=True)
            except Exception as err:
                st.caption(f"Curva de aprendizado indisponível: {err}")

        else:  # Classificação
            st.subheader("Relatório por classe")
            st.dataframe(classification_detailed_report(e["y_true"], e["y_pred"]),
                         use_container_width=True)
            st.subheader("Matriz de confusão")
            st.plotly_chart(
                px.imshow(e["cm"], text_auto=True,
                          x=[str(l) for l in e["labels"]],
                          y=[str(l) for l in e["labels"]],
                          labels=dict(x="Previsto", y="Real")),
                use_container_width=True,
            )
            # Curva de calibração (apenas binária)
            if "y_proba" in e and len(e["labels"]) == 2:
                st.subheader("Curva de calibração (binária)")
                proba = e["y_proba"][:, 1]
                cal = calibration_table(e["y_true"], proba, n_bins=10)
                st.plotly_chart(
                    px.line(cal, x="Probabilidade prevista",
                            y="Fração de positivos", markers=True),
                    use_container_width=True,
                )

        # Importância por permutação — medida OUT-OF-SAMPLE
        st.markdown("---")
        st.subheader("Importância por permutação (out-of-sample)")
        try:
            # Use EXCLUSIVAMENTE o teste que ficou fora do ajuste.
            # Para CV sem modelo final único, não exibimos uma permutação
            # artificialmente in-sample.
            if "X_test" not in e or "y_test" not in e:
                st.info(
                    "Importância por permutação OOS disponível para Hold-out "
                    "e Teste Separado. Para K-fold/CV aninhada, use a métrica "
                    "por fold ou ajuste um modelo final em um teste reservado."
                )
                imp_df = None
            else:
                buf_model = io.BytesIO()
                joblib.dump(e["model"], buf_model)
                imp_df = _cached_permutation(
                    buf_model.getvalue(), e["X_test"], e["y_test"], int(seed)
                )
            if imp_df is not None:
                st.plotly_chart(
                    px.bar(imp_df.head(20), x="Importância", y="Variável",
                           orientation="h", error_x="DP"),
                    use_container_width=True,
                )
        except Exception as err:
            st.caption(f"Permutação indisponível: {err}")

        # SHAP (opcional)
        st.subheader("SHAP (opcional)")
        with st.spinner("Calculando SHAP no espaço transformado..."):
            shap_frame = e.get("X_test", X)
            buf_model = io.BytesIO()
            joblib.dump(e["model"], buf_model)
            shap_out = _cached_shap(
                buf_model.getvalue(), shap_frame, rtask, 200, int(seed)
            )
        if shap_out is None:
            st.caption("Pacote `shap` não instalado — apenas permutação disponível.")
        else:
            shap_imp, _, _ = shap_out
            st.plotly_chart(
                px.bar(shap_imp.head(20), x="|SHAP| médio", y="Variável",
                       orientation="h"),
                use_container_width=True,
            )

        if "best_params" in e:
            st.write("Melhores hiperparâmetros:", e["best_params"])

        buf = io.BytesIO()
        joblib.dump(e["model"], buf)
        st.download_button("⬇️ Baixar modelo (.joblib)", buf.getvalue(),
                           f"{pick}.joblib")
    else:
        st.info("Treine modelos na aba 'Treino & Comparação' primeiro.")

# ---------------------------------------------------------------------------
# 8. EXPORTAÇÃO
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader("Tabelas para relatório (LaTeX / Excel)")
    if "res" in st.session_state:
        res_df, store, rtask = st.session_state["res"]

        st.markdown("### 1. Comparação de modelos (LaTeX)")
        st.code(to_latex_table(res_df, "Comparação de algoritmos de ML"),
                language="latex")

        st.markdown("### 2. Estatística descritiva (LaTeX)")
        eda_df = _cached_eda(work[[target] + feats])
        st.code(to_latex_table(eda_df, "Análise exploratória"), language="latex")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            eda_df.to_excel(writer, sheet_name="EDA", index=False)
            res_df.to_excel(writer, sheet_name="Resultados", index=False)
            if store:
                first = list(store.keys())[0]
                if rtask == "regression":
                    diag, _ = regression_diagnostics(store[first]["y_true"],
                                                     store[first]["y_pred"], len(feats))
                    diag.to_excel(writer, sheet_name="Diagnostico", index=False)
                else:
                    classification_detailed_report(store[first]["y_true"],
                                                   store[first]["y_pred"]
                                                   ).to_excel(writer,
                                                              sheet_name="Metricas",
                                                              index=False)
        st.download_button(
            "⬇️ Baixar pacote .xlsx", data=output.getvalue(),
            file_name="relatorio_ml_learner.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Treine os modelos para liberar as tabelas.")
