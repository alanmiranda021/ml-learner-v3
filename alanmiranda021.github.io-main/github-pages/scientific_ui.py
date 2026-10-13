"""Interface científica; treino e validação ficam em core.experiments."""
import hashlib
import importlib.util
import numpy as np
import pandas as pd
import streamlit as st
from core.env import IN_BROWSER
from core.task_config import TaskConfig, suggest_task
from core.data_inspection import profile_table
from core.experiments import run_experiment, scenario_configs, ablation_configs
from core.experiment_tracking import json_bytes
from core.signal_processing import SignalConfig, signals_to_table
from core.predict import make_bundle, to_bytes
from core.auto import report_md


def render_scientific(df):
    st.subheader("Experimento científico")
    st.caption("Escolha o problema e o protocolo. O teste externo fica reservado até a seleção do vencedor na validação.")
    input_mode = st.radio("Formato dos dados", ["Tabela de atributos", "Sinais em formato longo"])
    signal_group = None
    if input_mode.startswith("Sinais"):
        st.info("Cada aquisição deve ter ID e rótulo real constantes. Confirme a frequência de amostragem e ordene as amostras.")
        source = st.selectbox("ID da aquisição", df.columns)
        channel = st.selectbox("Canal numérico", df.select_dtypes("number").columns)
        label = st.selectbox("Alvo por aquisição", df.columns, index=len(df.columns)-1)
        order_column = st.selectbox("Ordem/tempo das amostras", [None, *df.columns], format_func=lambda c: "Ordem das linhas confirmada" if c is None else c)
        fs = st.number_input("Frequência de amostragem confirmada (Hz)", min_value=0., value=0.)
        window = st.number_input("Amostras por janela", min_value=3, value=1024)
        overlap = st.number_input("Amostras de sobreposição", min_value=0, value=0)
        filter_name = st.selectbox("Filtro", ["none", "butterworth", "fir"])
        low, high, order = 10., 2000., 4
        if filter_name != "none":
            low = st.number_input("Corte inferior (Hz)", min_value=0., value=10.)
            high = st.number_input("Corte superior (Hz)", min_value=0., value=2000.)
            order = st.number_input("Ordem do filtro", min_value=1, value=4)
        unit = st.text_input("Unidade do canal", value="")
        signal_request = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()
            + json_bytes(dict(columns=list(df.columns), source=source, channel=channel, label=label,
                              order_column=order_column, fs=fs, window=window, overlap=overlap,
                              filter=filter_name, low=low, high=high, order=order, unit=unit))).hexdigest()
        if st.button("Materializar janelas e inspecionar"):
            try:
                if df[source].isna().any() or df[label].isna().any():
                    raise ValueError("IDs e rótulos não podem conter ausentes.")
                signals, labels, ids = [], [], []
                for source_id, part in df.groupby(source, sort=False):
                    if part[label].nunique() != 1:
                        raise ValueError(f"Aquisição '{source_id}' tem rótulos variáveis; política por aquisição não pode ser aplicada.")
                    if order_column:
                        if part[order_column].isna().any() or part[order_column].duplicated().any():
                            raise ValueError("Ordem deve ser única e completa dentro de cada aquisição.")
                        part = part.sort_values(order_column, kind="stable")
                    signals.append(part[channel].to_numpy())
                    labels.append(part[label].iloc[0])
                    ids.append(source_id)
                cfg = SignalConfig(float(fs), int(window), int(overlap), filter=filter_name,
                                   low_hz=low, high_hz=high, order=int(order), unit=unit)
                st.session_state["signal_table"] = signals_to_table(signals, labels, ids, cfg)
                st.session_state["signal_origin_hash"] = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
                st.session_state["signal_request"] = signal_request
                st.session_state.pop("scientific_result", None)
            except Exception as exc:
                st.error(str(exc))
        origin_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
        if "signal_table" not in st.session_state or st.session_state.get("signal_origin_hash") != origin_hash or st.session_state.get("signal_request") != signal_request:
            return
        df = st.session_state["signal_table"]
        signal_group = "source_signal_id"
        st.write("Cauda descartada por aquisição:", df.attrs["discarded_tail_samples"])
        st.download_button("Exportar atributos das janelas", df.to_csv(index=False).encode(), "signal_features.csv")
        st.download_button("Exportar proveniência", json_bytes(df.attrs), "signal_metadata.json")
    with st.expander("Perfil dos dados", expanded=True):
        p = profile_table(df)
        st.write(f"{p['shape'][0]} linhas × {p['shape'][1]} colunas; {p['duplicatas']} linhas duplicadas.")
        st.dataframe(pd.DataFrame(p["columns"]))
    target = st.selectbox("Alvo confirmado", df.columns, index=list(df.columns).index("target") if signal_group else len(df.columns)-1)
    guess, reason = suggest_task(df[target])
    st.caption(reason + " A última coluna é apenas sugestão de alvo.")
    task = st.radio("Tipo de problema", ["regression", "classification"], index=0 if guess == "regression" else 1)
    id_column = st.selectbox("Identificador (fora dos preditores)", [None, *[c for c in df.columns if c != target]])
    group = signal_group or st.selectbox("Grupo de generalização", [None, *[c for c in df.columns if c != target]])
    time_column = st.selectbox("Tempo para previsão futura", [None, *[c for c in df.columns if c != target]]) if not signal_group else None
    options = ["group"] if signal_group else ["random", *(["group"] if group else []), *(["time"] if time_column else [])]
    split = st.selectbox("Estratégia de divisão", options)
    metadata = {c for c in (id_column, group, time_column) if c}
    if signal_group:
        metadata |= {"segment_index", "start_sample", "end_sample", "fs_hz", "label_source"}
    candidates = [c for c in df.columns if c != target and c not in metadata]
    features = st.multiselect("Preditores", candidates, default=candidates)
    st.write("Distribuição do alvo:", df[target].value_counts(dropna=False).head(30))
    missing_target = int(df[target].isna().sum())
    drop_missing = st.checkbox(f"Excluir explicitamente {missing_target} linhas sem alvo", value=False) if missing_target else False
    work = df.dropna(subset=[target]).copy() if drop_missing else df
    protocol_options = ["generic", "article3", "ablation"] if task == "classification" else ["generic"]
    protocol = st.selectbox("Protocolo", protocol_options,
                            format_func=lambda s: {"generic": "Sem redução / PCA / KPCA", "article3": "E1–E4 inspirados no artigo (SAE extensão)", "ablation": "Ablação de aumento × redução (extensões)"}[s])
    has_imb = importlib.util.find_spec("imblearn") is not None
    incompatible = protocol != "generic" and (IN_BROWSER or not has_imb)
    if incompatible:
        st.info("SAE e ablação exigem Python local com imbalanced-learn. No navegador, use o protocolo genérico.")
    if protocol != "generic" and any(not pd.api.types.is_numeric_dtype(df[c]) for c in features):
        incompatible = True
        st.info("Aumento ainda exige preditores numéricos. Remova categorias dos preditores ou use o protocolo genérico.")
    if protocol != "generic":
        st.caption("SAE NumPy com penalidade KL e perturbação latente. E2 fecha 50% do déficit, E3/E4 100%: escolhas desta extensão. Sem alegação de reprodução exata.")
    n_components = st.number_input("Componentes PCA/KPCA (validado em cada fold)", min_value=1, value=2)
    gamma = st.selectbox("Gamma KPCA", [.01, .1, 1., 10., 100.], index=1)
    epochs = st.slider("Épocas máximas SAE", 5, 300, 100, 5) if protocol != "generic" else 100
    models = ["SVM (RBF)", "Random Forest"] if task == "classification" else ["SVR (RBF)", "Random Forest", "Ridge"]
    names = st.multiselect("Modelos (baseline incluído automaticamente)", models, default=models[:2])
    k = st.slider("Folds", 2, 10, 3)
    test_size = st.slider("Teste externo (%)", 10, 40, 20, 5)/100
    gap = st.number_input("Gap temporal (linhas)", min_value=0, value=0) if split == "time" else 0
    seed = st.number_input("Semente do experimento", min_value=0, value=42)
    tune = st.checkbox("Otimização com CV interna (CV aninhada)")
    unit_target = st.text_input("Unidade do alvo (se aplicável)")
    cfg = TaskConfig(task, target, features, group_column=group, time_column=time_column,
                     sample_id_column=id_column, split_strategy=split, random_state=int(seed),
                     unit_metadata={target: unit_target}, signal_config=df.attrs.get("signal_config"))
    scenarios = ablation_configs(int(n_components), gamma) if protocol == "ablation" else scenario_configs(task, protocol, int(n_components), gamma, epochs)
    signature = hashlib.sha256(pd.util.hash_pandas_object(work, index=True).values.tobytes()
                               + json_bytes({"task": cfg.to_dict(), "scenarios": scenarios, "models": names, "k": k,
                                             "test_size": test_size, "gap": gap, "tune": tune})).hexdigest()
    if st.session_state.get("scientific_signature") != signature:
        st.session_state.pop("scientific_result", None)
    confirm = st.checkbox("Confirmo alvo, significado da tarefa e metadados de divisão")
    if st.button("Executar experimento", type="primary", disabled=incompatible or not confirm or not features or not names):
        bar = st.progress(0., text="Preparando experimento")
        try:
            r = run_experiment(work, cfg, scenarios, names, k, test_size, tune, int(gap),
                               progress=lambda p, t: bar.progress(.95*p, text=t))
            st.session_state["scientific_result"] = r
            st.session_state["scientific_signature"] = signature
            bar.progress(1., text="Experimento concluído")
        except Exception as exc:
            st.error(f"Experimento interrompido: {exc}")
    if "scientific_result" not in st.session_state:
        return
    r = st.session_state["scientific_result"]
    st.success(f"Vencedor pela validação: {r['best']}")
    st.dataframe(r["leaderboard"])
    st.write("Teste externo do vencedor:", r["test_metrics"])
    if "per_class" in r:
        st.json(r["per_class"])
    else:
        st.dataframe(pd.DataFrame({"real": r["y_true"], "previsto": r["y_pred"], "residuo": r["y_true"]-r["y_pred"]}))
    st.download_button("Baixar manifesto", json_bytes(r["manifest"]), "manifest.json")
    st.download_button("Baixar configuração", json_bytes(r["manifest"]["config"]), "config.json")
    st.download_button("Baixar relatório", report_md(r).encode(), "relatorio.md")
    st.download_button("Baixar comparação CSV", r["leaderboard"].to_csv(index=False).encode(), "metrics.csv")
    st.download_button("Baixar previsões do teste", pd.DataFrame({"real": r["y_true"], "previsto": r["y_pred"]}).to_csv(index=False).encode(), "test_predictions.csv")
    if st.button("Gerar relatório PDF científico"):
        from core.report_pdf import safe_pdf
        pdf, warning = safe_pdf(r, "experimento científico")
        if warning:
            st.info(warning)
        st.download_button("Baixar PDF científico", pdf, "scientific_report.pdf", "application/pdf")
    bundle = make_bundle(r, r["model"], "treino externo (teste preservado)")
    bundle["n_rows"] = r["n_train"]
    st.download_button("Baixar pipeline completo", to_bytes(bundle), "scientific_model.joblib")
    if st.button("Explicar vencedor com SHAP", disabled=IN_BROWSER):
        try:
            from core.explain import explain_pipeline
            explained = explain_pipeline(r["model"], r["X_train"], r["X_test"])
            vals = explained["values"].values
            axes = (0, 2) if vals.ndim == 3 else 0
            st.caption(explained["message"])
            st.dataframe(pd.DataFrame({"atributo": explained["feature_names"], "mean_abs_shap": np.abs(vals).mean(axis=axes)}))
        except Exception as exc:
            st.info(f"SHAP indisponível para esta execução: {exc}")
