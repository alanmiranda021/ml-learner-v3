"""Laboratório naval independente dos modos existentes."""
import hashlib
from dataclasses import asdict
import pandas as pd
import streamlit as st
from core.env import IN_BROWSER
from core.exp_artigo3 import NavalConfig, run_naval, validate_naval
from core.experiment_tracking import json_bytes
from core.report_pdf import naval_pdf


def render_naval(df, filename="tabela carregada", source_sha256=""):
    st.subheader("Laboratório — Artigo 3 (naval)")
    st.caption("Protocolo do Artigo 3 adaptado · resultados exploratórios · regressão do consumo de combustível")
    if "fuel_consumption" not in df or "ship_type" not in df:
        st.info("Envie a tabela naval com ship_type, route_id, distance, fuel_type, weather_conditions, engine_efficiency e fuel_consumption.")
        return
    st.info("O modo rápido usa células por ship_type, 1 repetição, 8 sorteios e até 50 iterações SAE. O completo local usa navio × rota, 3 repetições e 20 sorteios. A validação aninhada pode levar bastante tempo.")
    target = st.selectbox("Alvo naval", [c for c in ("fuel_consumption", "CO2_emissions") if c in df], key="naval_target")
    forbidden = "CO2_emissions" if target == "fuel_consumption" else "fuel_consumption"
    excluded = st.multiselect("Colunas excluídas", [c for c in df if c != target], default=[forbidden] if forbidden in df else [], key="naval_excluded_"+target)
    if forbidden not in excluded:
        st.error(f"Execução bloqueada: exclua {forbidden}, que revela o alvo.")
    mode = st.selectbox("Perfil naval", ["rapido"] if IN_BROWSER else ["rapido", "completo"], key="naval_mode")
    cells = st.multiselect("Colunas da célula (completo)", ["ship_type", "route_id"], default=["ship_type", "route_id"], disabled=mode == "rapido", key="naval_cells")
    fraction = st.number_input("Fração do aumento parcial", min_value=.01, max_value=1., value=.9, key="naval_fraction")
    hidden = st.multiselect("Grade de neurônios SAE", [3, 4, 5, 6], default=[3, 4, 5, 6], key="naval_hidden")
    gammas = st.multiselect("Grade gamma KPCA", [.01, .1, 1., 10., 100.], default=[.01, .1, 1., 10., 100.], key="naval_gamma")
    components = st.multiselect("Grade de componentes KPCA", [5, 8, 11], default=[5, 8, 11], key="naval_components")
    repeats = st.number_input("Repetições externas (completo)", min_value=1, max_value=10, value=3, disabled=mode == "rapido", key="naval_repeats")
    models = st.multiselect("Modelos navais", ["SVR (RBF)", "Random Forest", "Rede Neural (MLP)", "KNN"], default=["SVR (RBF)", "Random Forest"], key="naval_models")
    seed = st.number_input("Semente naval", min_value=0, value=42, key="naval_seed")
    units = st.text_input("Unidades e disponibilidade da eficiência no momento da previsão", value="Não confirmadas", key="naval_units")
    cfg = NavalConfig(target=target, excluded=tuple(excluded), cell_columns=tuple(cells), mode=mode,
        partial_fraction=float(fraction), hidden_grid=tuple(hidden), gamma_grid=tuple(gammas),
        components_grid=tuple(components), repeats=int(repeats), seed=int(seed), models=tuple(models),
        units=units, filename=filename, source_sha256=source_sha256)
    signature = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()+json_bytes(asdict(cfg))).hexdigest()
    if st.session_state.get("naval_signature") != signature:
        st.session_state.pop("naval_result", None)
        st.session_state.pop("naval_pdf", None)
        st.session_state["naval_signature"] = signature
    valid = True
    try:
        validate_naval(df, cfg.effective())
    except ValueError as exc:
        valid = False
        st.error(str(exc))
    if st.button("Executar protocolo naval e gerar PDF", disabled=not valid, type="primary"):
        st.session_state.pop("naval_result", None)
        st.session_state.pop("naval_pdf", None)
        bar = st.progress(0., text="Preparando partições navais")
        try:
            result = run_naval(df, cfg, progress=lambda p, text: bar.progress(p*.95, text=text))
            st.session_state["naval_result"] = result
            st.session_state["naval_pdf"] = naval_pdf(result)
            bar.progress(1., text="PDF concluído")
        except Exception as exc:
            st.error(f"Execução naval: {exc}")
    if "naval_result" in st.session_state:
        result = st.session_state["naval_result"]
        st.success(f"Execução exploratória concluída em {result['seconds']:.1f} s")
        st.dataframe(pd.DataFrame(result["ranking"]))
        st.dataframe(pd.DataFrame(result["comparisons"]))
        if result["failures"]:
            st.warning("Ocorreram falhas parciais. Consulte o PDF antes de interpretar o ranking.")
        if "naval_pdf" in st.session_state:
            st.download_button("Baixar relatório naval (.pdf)", st.session_state["naval_pdf"], "protocolo_artigo3_naval.pdf", "application/pdf")
