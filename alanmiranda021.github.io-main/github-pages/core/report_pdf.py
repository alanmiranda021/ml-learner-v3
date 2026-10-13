"""Relatório PDF detalhado (reportlab puro: funciona no PC e no navegador/Pyodide)."""
import io
import sys
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.lineplots import ScatterPlot
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

INK, ACC, ACC2, SOFT = colors.HexColor("#1b1f3b"), colors.HexColor("#6c5ce7"), colors.HexColor("#00b8a9"), colors.HexColor("#f1f0ff")
SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=ACC, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=INK, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=SS["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#555a7a"))
CELL = ParagraphStyle("C", parent=BODY, fontSize=8, leading=10)


def T(x):  # texto seguro para fontes padrão (Latin-1/cp1252) e para o parser XML do Paragraph
    return escape(str(x).encode("cp1252", "replace").decode("cp1252"))


def P(x, st=BODY):
    return Paragraph(T(x), st)


def _table(df, col_w=None, zebra=True, font=7.5):
    data = [[Paragraph(f"<b>{T(c)}</b>", ParagraphStyle("th", parent=CELL, textColor=colors.white, fontSize=font)) for c in df.columns]]
    for row in df.itertuples(index=False):
        data.append([Paragraph(T(v), ParagraphStyle("td", parent=CELL, fontSize=font)) for v in row])
    t = Table(data, colWidths=col_w, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), ACC), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9d7f5")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT])]
    t.setStyle(TableStyle(st))
    return t


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def _bar_h(labels, values, title, w=15.5 * cm, color=ACC, h=None):
    h = h or max(4.5 * cm, 0.62 * cm * len(labels) + 1.6 * cm)
    d = Drawing(w, h)
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 4.6 * cm, 0.6 * cm, w - 5.4 * cm, h - 1.5 * cm
    ch.data = [list(values)[::-1]]
    ch.categoryAxis.categoryNames = [str(l)[:28] for l in list(labels)[::-1]]
    ch.categoryAxis.labels.fontSize = 7
    ch.valueAxis.labels.fontSize = 7
    lo, hi = min(0, min(values)), max(values)
    ch.valueAxis.valueMin, ch.valueAxis.valueMax = lo, hi * 1.08 if hi > 0 else 1
    ch.bars[0].fillColor = color
    ch.bars.strokeColor = None
    ch.barLabelFormat = "%.3f"
    ch.barLabels.fontSize = 6.5
    ch.barLabels.nudge = 14
    d.add(ch)
    d.add(String(0, h - 9, title, fontSize=9, fontName="Helvetica-Bold", fillColor=INK))
    return d


def _scatter(y, p, w=8.4 * cm, h=7.2 * cm):
    n = len(y)
    idx = np.random.RandomState(0).choice(n, min(n, 600), replace=False)
    y, p = np.asarray(y)[idx], np.asarray(p)[idx]
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    d = Drawing(w, h)
    sp = ScatterPlot()
    sp.x, sp.y, sp.width, sp.height = 1.3 * cm, 1.0 * cm, w - 1.8 * cm, h - 1.9 * cm
    sp.data = [list(zip(map(float, y), map(float, p))), [(lo, lo), (hi, hi)]]
    sp.xValueAxis.valueMin = sp.yValueAxis.valueMin = lo
    sp.xValueAxis.valueMax = sp.yValueAxis.valueMax = hi
    sp.xValueAxis.labels.fontSize = sp.yValueAxis.labels.fontSize = 6.5
    sp.lineLabelFormat = None
    sp.xValueAxis.labelTextFormat = sp.yValueAxis.labelTextFormat = "%.2f"
    sp.xLabel = sp.yLabel = ""
    sp.lines[0].strokeColor = None
    sp.lines[0].symbol = makeMarker("FilledCircle", size=2.6, fillColor=ACC, strokeColor=None)
    sp.lines[1].strokeColor, sp.lines[1].strokeWidth, sp.lines[1].strokeDashArray = ACC2, 1, [3, 2]
    sp.lines[1].symbol = None
    d.add(sp)
    d.add(String(0, h - 9, "Real x Previsto (teste)", fontSize=9, fontName="Helvetica-Bold", fillColor=INK))
    d.add(String(w / 2 - 28, 0.15 * cm, "real (eixo x) x previsto (eixo y)", fontSize=6.5, fillColor=INK))
    return d


def _hist(res, w=7.6 * cm, h=7.2 * cm, bins=12):
    cnt, edges = np.histogram(res, bins=bins)
    d = Drawing(w, h)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 1.1 * cm, 1.0 * cm, w - 1.5 * cm, h - 1.9 * cm
    ch.data = [list(map(int, cnt))]
    ch.categoryAxis.categoryNames = [f"{e:.2g}" for e in edges[:-1]]
    ch.categoryAxis.labels.fontSize = 5.5
    ch.categoryAxis.labels.angle = 45
    ch.categoryAxis.labels.dy = -6
    ch.valueAxis.labels.fontSize = 6.5
    ch.bars[0].fillColor = ACC2
    ch.bars.strokeColor = None
    ch.groupSpacing = 1
    d.add(ch)
    d.add(String(0, h - 9, "Distribuicao dos residuos (teste)", fontSize=9, fontName="Helvetica-Bold", fillColor=INK))
    return d


def _confusion(cm_, labels):
    labels = [str(l)[:10] for l in labels]
    data = [["real \\ prev."] + labels] + [[labels[i]] + [int(v) for v in row] for i, row in enumerate(cm_)]
    mx = max(cm_.max(), 1)
    k = len(labels)
    cw = min(1.3 * cm, 15.5 * cm / (k + 1))
    t = Table(data, colWidths=[cw * 1.4] + [cw] * k, rowHeights=cw * 0.7)
    st = [("FONTSIZE", (0, 0), (-1, -1), 6.5), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("BACKGROUND", (0, 0), (-1, 0), SOFT), ("BACKGROUND", (0, 0), (0, -1), SOFT), ("GRID", (0, 0), (-1, -1), 0.25, colors.white)]
    for i in range(k):
        for j in range(k):
            a = cm_[i, j] / mx
            base = ACC2 if i == j else colors.HexColor("#ff7675")
            c = colors.Color(1 - (1 - base.red) * a, 1 - (1 - base.green) * a, 1 - (1 - base.blue) * a)
            st.append(("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), c))
    t.setStyle(TableStyle(st))
    return t


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACC); canvas.rect(0, A4[1] - 0.5 * cm, A4[0], 0.5 * cm, stroke=0, fill=1)
    canvas.setFillColor(ACC2); canvas.rect(A4[0] * 0.72, A4[1] - 0.5 * cm, A4[0] * 0.28, 0.5 * cm, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#555a7a"))
    canvas.drawString(2 * cm, 1.1 * cm, "AutoML Academico - Relatorio automatico")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def _insights(r):
    task, tm, board = r["task"], r["test_metrics"], r["leaderboard"]
    key = "R2" if task == "regression" else "F1 (macro)"
    out = []
    cv = float(board.loc[board["Modelo"] == r["best"], key].iloc[0])
    te = tm[key]
    out.append(f"O melhor modelo foi <b>{T(r['best'])}</b>, com {key} de {te:.3f} no teste (validacao cruzada: {cv:.3f}).")
    gap = cv - te
    out.append("A diferenca entre validacao cruzada e teste e pequena, sinal de que o modelo generaliza."
               if abs(gap) < 0.05 else f"Atencao: diferenca de {gap:+.3f} entre validacao cruzada e teste; com poucos dados isso e comum, interprete com cautela.")
    if len(board) > 1:
        d = float(board[key].iloc[0] - board[key].iloc[1])
        out.append("O 1o e o 2o colocados estao praticamente empatados (diferenca %.3f); considere o mais simples/rapido." % d if d < 0.01
                   else f"O 1o colocado supera o 2o por {d:.3f} em {key}.")
    if task == "classification" and r["profile"].get("imbalance", 1) < 0.5:
        out.append("As classes sao desbalanceadas; por isso a metrica principal e o F1 macro e foram usados pesos de classe.")
    if r["profile"]["n"] < 300:
        out.append("A base e pequena (menos de 300 linhas): os resultados tem alta variabilidade.")
    return out


def build_pdf(r, filename="dados", importance=True) -> bytes:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import classification_report, confusion_matrix
    task, tm, board, df = r["task"], r["test_metrics"], r["leaderboard"], r["data"]
    key = "R2" if task == "regression" else "F1 (macro)"
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                          title="Relatorio AutoML", author="AutoML Academico")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.6 * cm, id="f")], onPage=_page)])
    S = []
    # ---- capa
    S += [Spacer(1, 3.2 * cm), Paragraph("Relatorio de Modelagem Automatica",
          ParagraphStyle("t", parent=H1, fontSize=28, leading=33, textColor=INK)),
          Paragraph("Regressao e Classificacao com validacao rigorosa", ParagraphStyle("st", parent=BODY, fontSize=13, textColor=ACC)),
          Spacer(1, 1.2 * cm)]
    cover = [["Arquivo", filename], ["Problema", "Regressao" if task == "regression" else "Classificacao"], ["Variavel alvo", r["target"]],
             ["Linhas x colunas", f"{len(df)} x {df.shape[1] - 1} caracteristicas"], ["Gerado em", datetime.now().strftime("%d/%m/%Y %H:%M")],
             ["Tempo de execucao", f"{r['seconds']} s"], ["Semente aleatoria", str(r["seed"])]]
    ct = Table([[P(a, SMALL), P(b)] for a, b in cover], colWidths=[4 * cm, 11 * cm])
    ct.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d7f5")), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    S += [ct, Spacer(1, 1.2 * cm), P("Melhor modelo", SMALL), Paragraph(T(r["best"]), ParagraphStyle("bm", parent=H1, fontSize=22, textColor=ACC2))]
    kp = [[P(k, SMALL) for k in tm], [Paragraph(f"<b>{v:.4f}</b>", ParagraphStyle("kv", parent=BODY, fontSize=14, textColor=ACC)) for v in tm.values()]]
    kt = Table(kp, colWidths=[15.5 * cm / len(tm)] * len(tm)); kt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("TOPPADDING", (0, 0), (-1, -1), 8)]))
    S += [kt, PageBreak()]
    # ---- resumo
    S += [P("1. Resumo executivo", H1)] + [P("- " + i.replace("<b>", "").replace("</b>", ""), BODY) for i in _insights(r)]
    # ---- dados
    S += [P("2. Dados", H1), P(f"Foram usadas {len(r['features'])} caracteristicas e {len(df)} linhas validas. "
          f"Treino: {r['n_train']} linhas. Teste (nunca visto durante a escolha do modelo): {r['n_test']} linhas.")]
    S += [P("Decisoes automaticas sobre as colunas", H2)] + [P("- " + x, SMALL) for x in r["log"]]
    num = df.select_dtypes("number")
    if num.shape[1]:
        ds = num.describe().T[["mean", "std", "min", "50%", "max"]].round(3)
        ds.insert(0, "coluna", ds.index); ds["ausentes"] = df[ds.index].isna().sum().values
        ds = ds.head(25)
        S += [P("Estatisticas descritivas" + (" (25 primeiras colunas numericas)" if num.shape[1] > 25 else ""), H2),
              _table(ds.rename(columns={"mean": "media", "std": "desvio", "50%": "mediana"}), [4.2 * cm] + [1.9 * cm] * 6)]
    # ---- metodologia
    S += [P("3. Metodologia e auto-ajustes", H1),
          P("Os dados foram divididos em treino (80%) e teste (20%, estratificado na classificacao). Todos os algoritmos foram comparados por "
            "validacao cruzada repetida de 5 partes apenas no treino. O pre-processamento (imputacao, limitacao de outliers, escalonamento, PCA) "
            "ocorre dentro de cada particao, evitando vazamento de dados. O modelo vencedor foi otimizado e avaliado uma unica vez no teste."),
          P("Ajustes feitos automaticamente pelo sistema", H2)] + [P("- " + x, SMALL) for x in r["adjustments"]]
    # ---- ranking
    metr = [c for c in board.columns if c not in ("Modelo", "tempo (s)") and "(+-)" not in c and "(±)" not in c]
    show = board[["Modelo"] + metr + ["tempo (s)"]].copy()
    for c in metr:
        std = f"{c} (±)"
        show[c] = [f"{a:.4f} +- {b:.4f}" for a, b in zip(board[c], board[std])] if std in board else show[c].round(4)
    S += [KeepTogether([P("4. Comparacao dos algoritmos", H1), _bar_h(board["Modelo"], board[key], f"{key} medio na validacao cruzada")]),
          Spacer(1, 6), _table(show.round(4), [3.6 * cm] + [(15.5 - 3.6 - 1.6) * cm / max(len(metr), 1)] * len(metr) + [1.6 * cm])]
    # ---- melhor modelo
    S += [P(f"5. Melhor modelo: {r['best']}", H1),
          P("Hiperparametros escolhidos: " + (", ".join(f"{k.replace('model__', '')} = {v}" for k, v in r["best_params"].items()) or "padrao do algoritmo")),
          Spacer(1, 4)]
    yt, yp = np.asarray(r["y_true"]), np.asarray(r["y_pred"])
    if task == "regression":
        res = yt - yp
        pair = Table([[_scatter(yt, yp), _hist(res)]], colWidths=[8.4 * cm, 7.6 * cm])
        S += [pair, Spacer(1, 4), P(f"Residuos: media {res.mean():.4g}, desvio {res.std():.4g}, maior erro absoluto {np.abs(res).max():.4g}. "
              "Pontos proximos da linha tracejada indicam boas previsoes; residuos centrados em zero indicam ausencia de vies.", SMALL)]
    else:
        labels = sorted(pd.unique(yt))
        S += [KeepTogether([P("Matriz de confusao (teste)", H2), _confusion(confusion_matrix(yt, yp, labels=labels), labels)]), Spacer(1, 8)]
        rep = pd.DataFrame(classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)).T
        rep = rep.loc[[str(l) for l in labels] + ["macro avg"]].round(3)
        rep.insert(0, "classe", rep.index); rep["support"] = rep["support"].astype(int)
        S += [P("Desempenho por classe", H2), _table(rep.rename(columns={"precision": "precisao"}), [3 * cm] + [2.4 * cm] * 4)]
    # ---- importancia
    if importance:
        try:
            Xte = r["X_test"]
            if len(Xte) > 500:  # subamostra: importancia por permutacao e cara
                _ix = np.random.RandomState(0).choice(len(Xte), 500, replace=False)
                Xte, yt = Xte.iloc[_ix], yt[_ix]
            pi = permutation_importance(r["model"], Xte, yt, n_repeats=2 if Xte.shape[1] > 50 else 4, random_state=0, n_jobs=1,
                                        scoring="r2" if task == "regression" else "f1_macro")
            imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).head(15)
            S += [KeepTogether([P("6. Importancia das caracteristicas", H1),
                  P("Queda do desempenho no teste quando os valores de cada coluna sao embaralhados (importancia por permutacao). "
                    "Quanto maior, mais o modelo depende da coluna.", SMALL), _bar_h(imp.index, imp.values, "Top 15 caracteristicas", color=ACC2)])]
        except Exception as e:
            S += [P(f"Importancia por permutacao indisponivel ({e}).", SMALL)]
    # ---- conclusoes
    S += [P("7. Conclusoes e limitacoes", H1)] + [P("- " + i.replace("<b>", "").replace("</b>", "")) for i in _insights(r)]
    S += [P("- O resultado vale para a distribuicao dos dados fornecidos; novos dados com outro comportamento exigem reavaliacao."),
          P("- Correlacao nao implica causalidade: a importancia das colunas descreve o modelo, nao o fenomeno fisico."),
          P("- Os resultados podem diferir de outras ferramentas (ex.: MATLAB) por inicializacao e hiperparametros padrao distintos.")]
    # ---- apendice
    import sklearn
    S += [P("Apendice: reprodutibilidade e glossario", H1),
          P(f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}. "
            f"Semente {r['seed']}. Pre-processamento: {r['config']['scaler']}, outliers {'limitados (IQR)' if r['config']['winsor'] else 'mantidos'}"
            f"{', PCA 95%' if r['config']['pca'] else ''}.", SMALL),
          P("Metricas: <b>R2</b> = fracao da variancia explicada (1 e perfeito). <b>RMSE</b> = erro quadratico medio, na unidade do alvo. "
            "<b>MAE</b> = erro absoluto medio. <b>Acuracia</b> = acertos totais. <b>Precisao</b> = dos previstos como classe X, quantos eram X. "
            "<b>Recall</b> = das amostras X, quantas foram achadas. <b>F1</b> = media harmonica de precisao e recall; <b>macro</b> = media simples entre classes.".replace("<b>", "").replace("</b>", ""), SMALL)]
    doc.build(S)
    return buf.getvalue()


def _minimal_pdf(r, filename="dados", motivo="") -> bytes:
    """Último recurso: PDF simples (só texto) para o usuário nunca ficar sem relatório."""
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.6 * cm, id="f")], onPage=_page)])
    S = [P("Relatorio de Modelagem Automatica (versao simplificada)", H1), P(f"Arquivo: {filename}"),
         P(f"Alvo: {r['target']} | Tipo: {r['task']} | Melhor modelo: {r['best']}")]
    S += [P(f"{k}: {v:.4f}") for k, v in r["test_metrics"].items()]
    S += [P("Ranking", H2), _table(r["leaderboard"].round(4))]
    S += [P("Decisoes automaticas", H2)] + [P("- " + x, SMALL) for x in r["log"] + r["adjustments"]]
    if motivo:
        S += [P(f"Observacao: o relatorio completo falhou ({motivo}).", SMALL)]
    doc.build(S)
    return buf.getvalue()


def safe_pdf(r, filename="dados"):
    """Gera o PDF sem nunca levantar erro. Devolve (bytes, aviso). Tenta: completo -> sem importancia -> minimo."""
    try:
        return build_pdf(r, filename), ""
    except Exception as e1:
        try:
            return build_pdf(r, filename, importance=False), f"PDF gerado sem alguns itens ({type(e1).__name__}: {str(e1)[:120]})"
        except Exception as e2:
            try:
                return _minimal_pdf(r, filename, str(e2)[:120]), f"PDF simplificado ({type(e2).__name__}: {str(e2)[:120]})"
            except Exception as e3:
                raise RuntimeError(f"Falha ao gerar o PDF: {e3}") from e3


def save_pdf(data: bytes, base="relatorio_automl"):
    """Grava uma copia em ./relatorios/ quando roda no computador. No navegador nao ha disco: devolve None."""
    import sys
    if sys.platform == "emscripten":
        return None
    try:
        from pathlib import Path
        d = Path("relatorios"); d.mkdir(exist_ok=True)
        f = d / f"{base}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        f.write_bytes(data)
        return str(f.resolve())
    except Exception:
        return None
