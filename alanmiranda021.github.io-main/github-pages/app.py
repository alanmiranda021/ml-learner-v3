import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import importlib.util
import streamlit as st
from packaging.version import Version

STRETCH = {"width": "stretch"} if Version(st.__version__.split("+")[0]) >= Version("1.50") else {"use_container_width": True}
HAS_IMB = importlib.util.find_spec("imblearn") is not None

from core.auto import report_md, run_auto
from core.evaluate import cross_val, holdout, make_pipeline
from core.predict import final_model, load_bundle, make_bundle, predict_table, retrain_bundle, to_bytes
from core.report_pdf import safe_pdf, save_pdf
from core.io import load_table
from core.models import CLASSIFICATION, REGRESSION
from core.preprocess import (SCALERS, build_preprocessor, outlier_report,
                             remove_outlier_rows, smooth_columns)

st.set_page_config(page_title="AutoML Acadêmico", layout="wide")
CSS = """
<style>
@keyframes flow {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.hero{padding:1.6rem 1.8rem;border-radius:22px;color:#fff;margin-bottom:1.1rem;
  background:linear-gradient(120deg,#6c5ce7,#00b8a9,#fd79a8,#6c5ce7);background-size:300% 300%;animation:flow 14s ease infinite;
  box-shadow:0 10px 30px rgba(108,92,231,.35)}
.hero h1{margin:0;font-size:2rem;line-height:1.15;color:#fff}
.hero p{margin:.4rem 0 0;opacity:.95}
.hero .chip{display:inline-block;margin:.7rem .4rem 0 0;padding:.2rem .7rem;border-radius:99px;background:rgba(255,255,255,.22);font-size:.8rem}
.hero .orb{float:right;font-size:2.6rem;animation:floaty 4s ease-in-out infinite}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(108,92,231,.14),rgba(0,184,169,.12));border:1px solid rgba(108,92,231,.35);
  border-radius:16px;padding:.8rem 1rem;box-shadow:0 4px 14px rgba(0,0,0,.08)}
.stButton>button[kind="primary"],.stDownloadButton>button{border-radius:14px;border:0;font-weight:600;color:#fff;
  background:linear-gradient(120deg,#6c5ce7,#00b8a9);transition:transform .15s,box-shadow .15s}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(108,92,231,.4);color:#fff}
.podium{border-radius:18px;padding:1rem;text-align:center;border:1px solid rgba(128,128,160,.3);background:rgba(128,128,160,.08)}
.podium .m{font-size:2rem}.podium .n{font-weight:700;margin:.2rem 0}.podium .s{font-size:1.4rem;font-weight:800;color:#6c5ce7}
.podium.g1{border-color:#fdcb6e;box-shadow:0 6px 20px rgba(253,203,110,.35)}
@media (prefers-reduced-motion:reduce){.hero,.hero .orb{animation:none}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""<div class="hero"><span class="orb">🧠</span><h1>AutoML Acadêmico</h1>
<p>Suba a planilha. O sistema limpa, compara, escolhe, avalia e entrega o relatório.</p>
<span class="chip">Regressão</span><span class="chip">Classificação</span><span class="chip">Auto-ajuste</span><span class="chip">Relatório PDF</span></div>""",
            unsafe_allow_html=True)
px.defaults.color_discrete_sequence = ["#6c5ce7", "#00b8a9", "#fd79a8", "#fdcb6e", "#0984e3"]

# ---------- 1. DADOS ----------
with st.sidebar:
    st.header("1) Dados")
    files = st.file_uploader("CSV, Excel ou .mat", type=["csv", "txt", "xlsx", "xls", "mat"],
                             accept_multiple_files=True)
    if not files:
        st.info("Envie 1 arquivo (ou treino + validação para concatenar).")
        st.stop()
    loaded = []
    for f in files:
        try:
            loaded.append((f.name, load_table(f.name, f.getvalue())))
        except Exception as e:
            st.error(f"{f.name}: {e}")
    if not loaded:
        st.stop()
    multi = "juntar"
    if len(loaded) > 1:
        multi = st.radio("Vários arquivos", ["Juntar em uma base (mesmas colunas)", "Processar cada arquivo (lote)"])

if multi.startswith("Processar"):
    st.subheader("🗂️ Modo lote: cada arquivo vira um projeto")
    st.caption("Para cada arquivo: alvo = última coluna, tipo detectado, modelos comparados, relatório PDF. Erros em um arquivo não param os outros.")
    lv = st.radio("Velocidade", ["rápido", "completo"], horizontal=True, key="lv")
    lo = st.slider("Orçamento de tempo por arquivo (s)", 30, 600, 90, 30, key="lo")
    if st.button("▶️ Rodar todos os arquivos", type="primary"):
        import zipfile
        rows, pdfs, bar = [], {}, st.progress(0.0, text="Iniciando o lote...")
        for i, (nome, d) in enumerate(loaded):
            bar.progress(i / len(loaded), text=f"Processando {nome} ({i + 1}/{len(loaded)})")
            try:
                def batch_progress(p, text):
                    value = (i + 0.95 * p) / len(loaded)
                    bar.progress(value, text=f"{value:.0%} · Arquivo {i + 1}/{len(loaded)} · {nome} · {text}")
                r = run_auto(d, None, lv, budget_s=lo, progress=batch_progress)
                k = "R2" if r["task"] == "regression" else "F1 (macro)"
                rows.append({"arquivo": nome, "tipo": "regressão" if r["task"] == "regression" else "classificação", "alvo": r["target"],
                             "melhor modelo": r["best"], "métrica": k, "nota (teste)": round(r["test_metrics"][k], 4), "tempo (s)": r["seconds"]})
                bar.progress((i + 0.95) / len(loaded), text=f"Arquivo {i + 1}/{len(loaded)} · {nome} · Gerando PDF...")
                pdfs[nome], _aviso = safe_pdf(r, nome)
            except Exception as e:
                rows.append({"arquivo": nome, "tipo": "ERRO", "alvo": str(e)[:80]})
        bar.progress(1.0, text="100% · Lote concluído")
        tab = pd.DataFrame(rows)
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("resumo.csv", tab.to_csv(index=False))
            for n, b in pdfs.items():
                z.writestr(f"relatorio_{n.rsplit('.', 1)[0]}.pdf", b)
        st.session_state["lote"] = (tab, zb.getvalue())
    if "lote" in st.session_state:
        tab, zb = st.session_state["lote"]
        st.dataframe(tab, **STRETCH)
        st.download_button("⬇️ Baixar tudo (.zip com resumo + 1 PDF por arquivo)", zb, "lote_automl.zip")
    st.stop()

cols_sets = [set(map(str, d.columns)) for _, d in loaded]
if any(c != cols_sets[0] for c in cols_sets):
    st.error("Os arquivos têm colunas diferentes, então não dá para juntá-los numa base só (misturaria dados de problemas diferentes). "
             "Escolha **Processar cada arquivo (lote)** na barra lateral, ou envie apenas arquivos com as mesmas colunas.")
    for n, d in loaded:
        st.write(f"• **{n}**: {d.shape[1]} colunas, ex.: {', '.join(map(str, d.columns[:4]))}…")
    st.stop()
df = pd.concat([d for _, d in loaded], ignore_index=True)

st.subheader("Visão geral")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas", len(df)); c2.metric("Colunas", df.shape[1]); c3.metric("Valores ausentes", int(df.isna().sum().sum()))
st.dataframe(df.head(50), **STRETCH)

with st.sidebar:
    st.header("Modo")
    modo = st.radio("Como quer trabalhar?", ["🚀 Automático", "🛠️ Manual", "📦 Usar / retreinar .joblib"], label_visibility="collapsed")

if modo.startswith("🚀"):
    st.subheader("🚀 Modo automático")
    st.write("O sistema escolhe o alvo (última coluna, mude se precisar), detecta regressão/classificação, "
             "descarta colunas inúteis, trata outliers, escalona, compara os algoritmos com validação cruzada "
             "repetida, otimiza o melhor e avalia num teste nunca visto.")
    _sig = (tuple(f.name for f in files), df.shape)
    if st.session_state.get("auto_sig") != _sig:  # arquivo mudou: descarta resultados antigos
        for _k in ("auto", "pdf", "pdf_aviso", "pdf_path", "bundle"):
            st.session_state.pop(_k, None)
        st.session_state["auto_sig"] = _sig
    a1, a2 = st.columns(2)
    alvo = a1.selectbox("Variável alvo", df.columns, index=len(df.columns) - 1)
    velocidade = a2.radio("Velocidade", ["rápido", "completo"], horizontal=True,
                          help="Completo inclui rede neural e mais repetições (bem mais lento).")
    orcamento = st.slider("Orçamento de tempo (segundos)", 30, 900, 150 if velocidade == "rápido" else 600, 30,
                          help="O sistema pula modelos lentos ou desliga a otimização para respeitar esse tempo.")
    if st.button("▶️ Rodar tudo automaticamente", type="primary"):
        barra = st.progress(0.0, text="Iniciando...")
        for _k in ("pdf", "pdf_aviso", "pdf_path", "bundle"):
            st.session_state.pop(_k, None)
        try:
            st.session_state["auto"] = run_auto(df, alvo, velocidade,
                progress=lambda p, t: barra.progress(0.95 * p, text=f"{0.95 * p:.0%} · {t}"), budget_s=orcamento)
        except Exception as e:
            barra.empty()
            st.session_state.pop("auto", None)
            st.error(f"A execução automática falhou: {e}")
            st.exception(e)
            st.stop()
        barra.progress(0.96, text="96% · Montando o relatório PDF...")
        try:
            pdf_bytes, aviso = safe_pdf(st.session_state["auto"], ", ".join(f.name for f in files))
            st.session_state["pdf"], st.session_state["pdf_aviso"] = pdf_bytes, aviso
            st.session_state["pdf_path"] = save_pdf(pdf_bytes)
        except Exception as e:
            st.session_state["pdf_aviso"] = f"Não foi possível gerar o PDF: {e}"
        barra.progress(1.0, text="100% · Execução concluída")
    if "auto" in st.session_state:
        r = st.session_state["auto"]
        st.success(f"Concluído em {r['seconds']} s — melhor modelo: **{r['best']}**")
        if st.session_state.get("pdf"):
            st.download_button("📄 Baixar relatório completo (.pdf)", st.session_state["pdf"], "relatorio_automl.pdf",
                               "application/pdf", type="primary", key="dl_pdf_top")
            if st.session_state.get("pdf_path"):
                st.caption(f"Cópia salva automaticamente em: {st.session_state['pdf_path']}")
        if st.session_state.get("pdf_aviso"):
            st.warning(st.session_state["pdf_aviso"])
        cols = st.columns(len(r["test_metrics"]))
        for c, (k, v) in zip(cols, r["test_metrics"].items()):
            c.metric(k + " (teste)", f"{v:.4f}")
        key_ = "R2" if r["task"] == "regression" else "F1 (macro)"
        top = r["leaderboard"].head(3)
        pc = st.columns(len(top))
        for i, (c, row) in enumerate(zip(pc, top.itertuples(index=False))):
            sc = top.iloc[i][key_]
            c.markdown(f'<div class="podium {"g1" if i == 0 else ""}"><div class="m">{["🥇", "🥈", "🥉"][i]}</div>'
                       f'<div class="n">{row.Modelo}</div><div class="s">{sc:.3f}</div><div style="opacity:.7;font-size:.8rem">{key_} (validação)</div></div>',
                       unsafe_allow_html=True)
        st.write("")
        with st.expander("Decisões tomadas automaticamente", expanded=True):
            for x in r["log"]:
                st.write("• " + x)
        with st.expander("🔧 Auto-ajustes (o sistema se adaptou aos seus dados)", expanded=True):
            for x in r["adjustments"]:
                st.write("• " + x)
        st.dataframe(r["leaderboard"].round(4), **STRETCH)
        if r["task"] == "regression":
            d = pd.DataFrame({"real": r["y_true"], "previsto": r["y_pred"]})
            fig = px.scatter(d, x="real", y="previsto", title="Real × Previsto (teste)")
            lo, hi = float(d.min().min()), float(d.max().max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
        else:
            from sklearn.metrics import confusion_matrix
            lab = sorted(pd.unique(r["y_true"]))
            fig = px.imshow(confusion_matrix(r["y_true"], r["y_pred"], labels=lab), text_auto=True,
                            x=[str(l) for l in lab], y=[str(l) for l in lab],
                            labels=dict(x="Previsto", y="Real"), title="Matriz de confusão (teste)")
        st.plotly_chart(fig, **STRETCH)
        st.download_button("⬇️ Resumo (.md)", report_md(r).encode("utf-8"), "relatorio.md")
        st.markdown("#### 🎯 Usar a ML treinada")
        st.caption("O modelo acima foi treinado com 80% dos dados (para medir o desempenho honestamente). "
                   "Para uso real, treine a versão final com 100% dos dados e depois preveja planilhas novas.")
        if st.button("🏋️ Treinar modelo final (100% dos dados)"):
            with st.spinner("Treinando..."):
                st.session_state["bundle"] = make_bundle(r, final_model(r))
        if "bundle" in st.session_state:
            bd = st.session_state["bundle"]
            st.success(f"Modelo final pronto: {bd['algorithm']} treinado com {bd['n_rows']} linhas. "
                       f"Colunas exigidas: {len(bd['features'])}.")
            st.download_button("⬇️ Baixar modelo (.joblib) para usar em qualquer script", to_bytes(bd), "modelo_final.joblib")
            novo = st.file_uploader("🔮 Prever com uma planilha nova (mesmas colunas)", type=["csv", "xlsx", "xls", "mat"], key="novo")
            if novo is not None:
                try:
                    res_new = predict_table(bd, load_table(novo.name, novo.getvalue()))
                    st.dataframe(res_new.head(200), **STRETCH)
                    st.download_button("⬇️ Baixar previsões (.csv)", res_new.to_csv(index=False).encode("utf-8"), "previsoes.csv")
                except Exception as e:
                    st.error(str(e))
        st.markdown("#### 📄 Relatório detalhado em PDF")
        if st.button("Gerar relatório PDF novamente"):
            with st.spinner("Montando o PDF (gráficos, tabelas e importância das variáveis)..."):
                try:
                    st.session_state["pdf"], st.session_state["pdf_aviso"] = safe_pdf(r, ", ".join(f.name for f in files))
                    st.session_state["pdf_path"] = save_pdf(st.session_state["pdf"])
                except Exception as e:
                    st.error(f"Falha ao gerar o PDF: {e}")
        if "pdf" in st.session_state:
            st.download_button("⬇️ Baixar relatório (.pdf)", st.session_state["pdf"], "relatorio_automl.pdf", "application/pdf", key="dl_pdf_bottom")
    st.stop()

if modo.startswith("📦"):
    st.subheader("📦 Usar ou retreinar um modelo salvo (.joblib)")
    st.warning("Só abra arquivos .joblib que você mesmo gerou ou de fonte confiável: esse formato pode executar código ao ser aberto.")
    jb = st.file_uploader("Modelo (.joblib) gerado por este app", type=["joblib"], key="jb")
    if jb is None:
        st.info("Suba o .joblib aqui. A planilha enviada na barra lateral será usada como dados novos.")
        st.stop()
    try:
        bd = load_bundle(jb.getvalue())
    except Exception as e:
        st.error(str(e)); st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritmo", bd["algorithm"]); c2.metric("Alvo", bd["target"]); c3.metric("Treinado com", f"{bd.get('n_rows', '?')} linhas")
    with st.expander("Colunas exigidas e métricas de teste originais"):
        st.write(bd["features"]); st.write(bd.get("test_metrics", {}))
    aba1, aba2 = st.tabs(["🔮 Prever com estes dados", "🏋️ Retreinar com estes dados"])
    with aba1:
        try:
            res_new = predict_table(bd, df)
            st.dataframe(res_new.head(200), **STRETCH)
            st.download_button("⬇️ Baixar previsões (.csv)", res_new.to_csv(index=False).encode("utf-8"), "previsoes.csv")
        except Exception as e:
            st.error(str(e))
    with aba2:
        st.caption(f"Refaz o treino com a mesma receita (algoritmo e hiperparâmetros), usando as linhas da planilha que tenham a coluna real '{bd['target']}'. "
                   "O modelo antigo NÃO é 'continuado': ele esquece o que viu. Para manter o conhecimento antigo, suba na barra lateral "
                   "os dados antigos + novos juntos.")
        if bd["target"] not in df.columns:
            st.error(f"A planilha não tem a coluna '{bd['target']}'.")
        elif st.button("Retreinar e comparar", type="primary"):
            try:
                st.session_state["retrain"] = retrain_bundle(bd, df)
            except Exception as e:
                st.error(str(e))
        if "retrain" in st.session_state:
            nb, rep = st.session_state["retrain"]
            comp = pd.DataFrame({"modelo antigo": rep["antigo"], "modelo retreinado": rep["novo"]}).round(4)
            st.write(f"Comparação em {rep['n_teste']} linhas de teste (20% dos dados enviados):")
            st.dataframe(comp, **STRETCH)
            st.caption("Atenção: se os dados enviados incluem linhas que o modelo antigo já viu no treino, a nota dele fica inflada.")
            st.download_button("⬇️ Baixar modelo retreinado (.joblib)", to_bytes(nb), "modelo_retreinado.joblib")
    st.stop()

with st.sidebar:
    st.header("2) Problema")
    target = st.selectbox("Variável alvo (a prever)", df.columns, index=len(df.columns) - 1)
    task_guess = "classification" if (df[target].dtype == object or df[target].nunique() <= 15) else "regression"
    task = st.radio("Tipo", ["regression", "classification"], index=["regression", "classification"].index(task_guess),
                    format_func=lambda t: "Regressão" if t == "regression" else "Classificação")
    feats = st.multiselect("Características (entradas)", [c for c in df.columns if c != target],
                           default=[c for c in df.columns if c != target])
if not feats:
    st.warning("Selecione ao menos uma característica."); st.stop()

# alerta de vazamento de dados (feature quase idêntica ao alvo)
if task == "regression" and pd.api.types.is_numeric_dtype(df[target]):
    corr = df[feats].select_dtypes("number").corrwith(df[target]).abs().sort_values(ascending=False)
    if len(corr) and corr.iloc[0] > 0.98:
        st.warning(f"⚠️ '{corr.index[0]}' tem correlação {corr.iloc[0]:.3f} com o alvo — possível vazamento de dados.")

# ---------- 2. LIMPEZA ----------
with st.sidebar:
    st.header("3) Ruído / limpeza")
    num_cols_all = df[feats].select_dtypes("number").columns.tolist()
    winsor = st.checkbox("Limitar outliers (IQR) — dentro do pipeline", value=True)
    k_iqr = st.slider("Fator IQR", 1.0, 3.0, 1.5, 0.1)
    rm_rows = st.checkbox("Remover linhas com outliers (antes da divisão)", value=False)
    smooth = st.selectbox("Suavização (dados em ordem temporal)", ["Nenhuma", "Média móvel", "Mediana móvel", "Savitzky-Golay"])
    win = st.slider("Janela", 3, 51, 5, 2) if smooth != "Nenhuma" else 5
    scaler = st.selectbox("Escalonamento", list(SCALERS), index=1)

work = df.dropna(subset=[target]).copy()
if smooth != "Nenhuma":
    work = smooth_columns(work, num_cols_all, smooth, win)
if rm_rows:
    before = len(work); work = remove_outlier_rows(work, num_cols_all, k_iqr)
    st.info(f"Linhas removidas: {before - len(work)}")

with st.expander("Diagnóstico de outliers e correlação"):
    st.dataframe(outlier_report(work[num_cols_all], k_iqr), **STRETCH)
    if len(num_cols_all) > 1:
        st.plotly_chart(px.imshow(work[num_cols_all].corr(), aspect="auto", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1), **STRETCH)

num_cols = work[feats].select_dtypes("number").columns.tolist()
cat_cols = [c for c in feats if c not in num_cols]
X, y = work[feats], work[target]
pre = build_preprocessor(num_cols, cat_cols, scaler, winsor, k_iqr)

# ---------- 3. MODELOS ----------
with st.sidebar:
    st.header("4) Validação")
    strategy = st.radio("Estratégia", ["Hold-out", "K-fold", "K-fold repetido"])
    if strategy == "Hold-out":
        test_size = st.slider("% teste", 10, 40, 20, 5) / 100
    else:
        k = st.slider("K", 3, 10, 5)
        reps = st.slider("Repetições", 2, 10, 5) if strategy == "K-fold repetido" else 1
    seed = st.number_input("Semente", 0, 9999, 42)
    st.header("5) Algoritmos")
    zoo = REGRESSION if task == "regression" else CLASSIFICATION
    chosen = st.multiselect("Modelos", list(zoo), default=list(zoo)[:4])
    tune = st.checkbox("Otimizar hiperparâmetros (GridSearch, CV interna)", value=False)
    sampler_name = "Nenhum"
    if task == "classification":
        sampler_name = st.selectbox("Balanceamento (só no treino)", ["Nenhum", "SMOTE", "ADASYN"] if HAS_IMB else ["Nenhum"])

if st.button("▶️ Treinar e comparar", type="primary"):
    sampler = None
    if sampler_name != "Nenhum":
        from imblearn.over_sampling import ADASYN, SMOTE
        sampler = {"SMOTE": SMOTE, "ADASYN": ADASYN}[sampler_name](random_state=int(seed))
    rows, store = [], {}
    if not chosen:
        st.warning("Selecione pelo menos um algoritmo.")
        st.stop()
    bar = st.progress(0.0, text="Iniciando treino e teste...")
    for i, name in enumerate(chosen):
        bar.progress(i / len(chosen), text=f"Modelo {i + 1}/{len(chosen)} · {name}: treinando e testando")
        model, grid = zoo[name]
        pipe = make_pipeline(pre, model, grid, tune and strategy == "Hold-out", task=task, sampler=sampler)
        try:
            if strategy == "Hold-out":
                m, extra = holdout(pipe, X, y, task, test_size, int(seed))
                store[name] = extra
                rows.append({"Modelo": name, **{a: round(b, 4) for a, b in m.items()}})
            else:
                def manual_progress(p, text):
                    value = (i + p) / len(chosen)
                    bar.progress(value, text=f"{value:.0%} · {name} · {text}")
                m = cross_val(make_pipeline(pre, model, grid, False, task=task, sampler=sampler), X, y, task, k, reps, int(seed), progress=manual_progress)
                rows.append({"Modelo": name, **{a: f"{b[0]:.4f} ± {b[1]:.4f}" for a, b in m.items()}})
        except Exception as e:
            rows.append({"Modelo": name, "erro": str(e)[:80]})
        bar.progress((i + 1) / len(chosen), text=f"Modelo {i + 1}/{len(chosen)} · {name}: concluído")
    bar.progress(1.0, text="100% · Comparação concluída")
    st.session_state["res"] = (pd.DataFrame(rows), store, task)

if "res" in st.session_state:
    res, store, rtask = st.session_state["res"]
    st.subheader("Resultados")
    st.dataframe(res, **STRETCH)
    st.download_button("⬇️ Baixar tabela (CSV)", res.to_csv(index=False).encode(), "resultados.csv")
    if store:
        pick = st.selectbox("Detalhar modelo", list(store))
        e = store[pick]
        if rtask == "regression":
            d = pd.DataFrame({"real": e["y_true"], "previsto": e["y_pred"]})
            fig = px.scatter(d, x="real", y="previsto", title=f"Real × Previsto — {pick}")
            lo, hi = float(d.min().min()), float(d.max().max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
            st.plotly_chart(fig, **STRETCH)
        else:
            st.plotly_chart(px.imshow(e["cm"], text_auto=True, x=[str(l) for l in e["labels"]],
                                      y=[str(l) for l in e["labels"]], labels=dict(x="Previsto", y="Real"),
                                      title=f"Matriz de confusão — {pick}"), **STRETCH)
        if "best_params" in e:
            st.write("Melhores hiperparâmetros:", e["best_params"])
        buf = io.BytesIO(); joblib.dump(e["model"], buf)
        st.download_button("⬇️ Baixar modelo treinado (.joblib)", buf.getvalue(), f"{pick}.joblib")
