"""Modo automático: sobe o arquivo -> o sistema decide, treina, compara, escolhe e avalia."""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from core.env import IN_BROWSER
from core.evaluate import clf_metrics, cross_val, make_pipeline, reg_metrics
from core.models import CLASSIFICATION, REGRESSION
from core.preprocess import build_preprocessor

FAST = {"Regressão Linear", "Ridge", "Lasso", "Árvore de Decisão", "KNN", "SVR (RBF)", "Random Forest",
        "Regressão Logística", "Naive Bayes", "SVM (RBF)", "Gradient Boosting"}  # MLP fica fora do modo rápido


def detect(df: pd.DataFrame, target: str | None = None):
    """Escolhe alvo (última coluna por padrão), tipo de problema e colunas úteis."""
    log = []
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    target = target or df.columns[-1]
    log.append(f"Alvo: '{target}'")
    y = df[target]
    task = "classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15) else "regression"
    log.append("Tipo: " + ("classificação" if task == "classification" else "regressão"))
    feats = []
    for c in df.columns.drop(target):
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            log.append(f"Descartada '{c}' (constante)"); continue
        if s.isna().mean() > 0.5:
            log.append(f"Descartada '{c}' (>50% ausente)"); continue
        if not pd.api.types.is_numeric_dtype(s) and s.nunique() > 0.5 * len(s):
            log.append(f"Descartada '{c}' (texto tipo ID)"); continue
        feats.append(c)
    if task == "regression":
        num = df[feats].select_dtypes("number")
        corr = num.corrwith(y).abs()
        for c in corr[corr > 0.98].index:
            feats.remove(c); log.append(f"Descartada '{c}' (correlação {corr[c]:.3f} com o alvo: vazamento)")
    return df.dropna(subset=[target]), target, task, feats, log


HEAVY = {"Random Forest", "Gradient Boosting", "Rede Neural (MLP)"}
ORDER = ["Regressão Linear", "Ridge", "Lasso", "Regressão Logística", "Naive Bayes", "KNN", "Árvore de Decisão",
         "SVR (RBF)", "SVM (RBF)", "Random Forest", "Gradient Boosting", "Rede Neural (MLP)"]


def profile(X, y, task) -> dict:
    """Mede características dos dados que guiam as decisões automáticas."""
    num = X.select_dtypes("number")
    skew = num.skew().abs() if num.shape[1] else pd.Series(dtype=float)
    out = 0.0
    if num.shape[1]:
        q1, q3 = num.quantile(.25), num.quantile(.75); iqr = (q3 - q1).replace(0, np.nan)
        out = float(((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).mean().mean())
    prof = {"n": len(X), "p": X.shape[1], "skew_frac": float((skew > 1).mean()) if len(skew) else 0.0,
            "outlier_frac": out, "n_classes": int(y.nunique()) if task == "classification" else 0}
    if task == "classification":
        vc = y.value_counts()
        prof["imbalance"] = float(vc.min() / vc.max())
    return prof


def candidate_configs(prof) -> list:
    """Gera configurações de pré-processamento plausíveis para ESTES dados."""
    c = [dict(scaler="Z-score (Standard)", winsor=True, pca=False),
         dict(scaler="Z-score (Standard)", winsor=False, pca=False),
         dict(scaler="Robust (mediana/IQR)", winsor=False, pca=False)]
    if prof["skew_frac"] > 0.3:
        c.append(dict(scaler="Power (Yeo-Johnson)", winsor=False, pca=False))
    if prof["p"] >= 30 and prof["n"] > 100:
        c.append(dict(scaler="Z-score (Standard)", winsor=True, pca=True))
    return c


def _label(cfg):
    return f"{cfg['scaler']}, outliers={'limitados' if cfg['winsor'] else 'mantidos'}" + (", PCA 95%" if cfg["pca"] else "")


def run_auto(df, target=None, mode="rápido", test_size=0.2, seed=42, progress=None, budget_s=None, max_sel_rows=None):
    """budget_s: orçamento de tempo (s). O sistema pula modelos pesados / desliga a otimização se estourar."""
    t0 = time.time()
    notify = progress
    last_progress = 0.0
    def update_progress(value, text):
        nonlocal last_progress
        last_progress = max(last_progress, min(value, 1.0))
        notify(last_progress, text)
    progress = update_progress if notify else None
    if progress: progress(0.01, "Preparando dados e separando treino/teste")
    rapido = mode == "rápido"
    # navegador (Pyodide) roda em 1 thread e é bem mais lento: limites mais enxutos
    max_sel_rows = max_sel_rows or (2000 if IN_BROWSER else (4000 if rapido else 8000))
    budget_s = budget_s or (150 if mode == "rápido" else 900)
    df, target, task, feats, log = detect(df, target)
    adj = []  # decisões de auto-ajuste
    X, y = df[feats], df[target]
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in feats if c not in num]
    zoo = REGRESSION if task == "regression" else CLASSIFICATION
    names = [n for n in ORDER if n in zoo and (mode == "completo" or n in FAST)]

    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    key = "R2" if task == "regression" else "F1 (macro)"
    reps = 1 if (rapido and IN_BROWSER) else (2 if rapido else 3)

    prof = profile(Xtr, ytr, task)
    adj.append(f"Perfil: {prof['n']} linhas × {prof['p']} colunas, {prof['skew_frac']:.0%} das colunas assimétricas, "
               f"{prof['outlier_frac']:.1%} de valores atípicos" + (f", balanceamento {prof['imbalance']:.2f}" if task == 'classification' else ""))

    # adaptação ao tamanho: seleção em subamostra, treino final com tudo
    if len(Xtr) > max_sel_rows:
        idx = Xtr.sample(max_sel_rows, random_state=seed).index
        Xs, ys = Xtr.loc[idx], ytr.loc[idx]
        adj.append(f"Base grande: seleção feita em subamostra de {max_sel_rows} linhas; modelo final treinado com todas.")
    else:
        Xs, ys = Xtr, ytr

    # 1) busca da melhor configuração de pré-processamento (modelos de referência sensíveis à escala)
    ref = ["KNN", "Ridge" if task == "regression" else "Regressão Logística"]
    best_cfg, best_sc, tried = None, -np.inf, []
    cfgs = candidate_configs(prof)
    if rapido:
        cfgs = cfgs[:2] if IN_BROWSER else cfgs[:3]
    Xc, yc = (Xs.iloc[:1500], ys.iloc[:1500]) if rapido and len(Xs) > 1500 else (Xs, ys)
    for cfg_i, cfg in enumerate(cfgs):
        try:
            pre = build_preprocessor(num, cat, cfg["scaler"], cfg["winsor"], 1.5, cfg["pca"])
            scores = []
            for ref_i, model_name in enumerate(ref):
                def prep_progress(p, text):
                    if progress:
                        fraction = (cfg_i + (ref_i + p) / len(ref)) / len(cfgs)
                        progress(0.05 + 0.10 * fraction,
                                 f"Preparação {cfg_i + 1}/{len(cfgs)} · {model_name} · {text}")
                scores.append(cross_val(make_pipeline(pre, zoo[model_name][0], task=task),
                                        Xc, yc, task, 3, 1, seed,
                                        progress=prep_progress if progress else None)[key][0])
            sc = np.mean(scores)
        except Exception as e:  # auto-recuperação: configuração inviável é descartada
            adj.append(f"Configuração descartada ({_label(cfg)}): {str(e)[:60]}"); continue
        tried.append((sc, cfg))
        if sc > best_sc + 1e-4:
            best_sc, best_cfg = sc, cfg
        if progress: progress(0.05 + 0.1 * (cfg_i + 1) / len(cfgs), "Preparação concluída: " + _label(cfg))
    if best_cfg is None:
        best_cfg = dict(scaler="Z-score (Standard)", winsor=True, pca=False)
        adj.append("Nenhuma configuração avaliável; usando o padrão.")
    adj.append("Pré-processamento escolhido: " + _label(best_cfg) + " — " + " | ".join(
        f"{_label(c)}: {sc:.3f}" for sc, c in tried))
    pre = build_preprocessor(num, cat, best_cfg["scaler"], best_cfg["winsor"], 1.5, best_cfg["pca"])

    # 2) desbalanceamento -> pesos de classe (sem criar dados sintéticos)
    balance = task == "classification" and prof["imbalance"] < 0.5
    if balance:
        adj.append(f"Classes desbalanceadas ({prof['imbalance']:.2f}): usando class_weight='balanced' nos modelos que suportam.")

    def get_model(n):
        m, g = zoo[n]
        m = clone(m)
        if balance and "class_weight" in m.get_params():
            m.set_params(class_weight="balanced")
        return m, g

    # 3) comparação com orçamento de tempo (preditivo: mede 1 treino antes de decidir)
    rows, skipped, fit_t = [], [], {}
    for i, n in enumerate(names):
        start = 0.15 + 0.8 * i / len(names)
        if progress: progress(start, f"Modelo {i + 1}/{len(names)} · {n}: iniciando treino")
        m, g = get_model(n)
        remaining = budget_s - (time.time() - t0)
        use_reps = reps
        if n in HEAVY:
            s0 = time.time()
            try:
                make_pipeline(pre, m, task=task).fit(Xs, ys)
            except Exception:
                pass
            t_fit = time.time() - s0
            if t_fit * 5 * reps > 0.5 * remaining:
                if t_fit * 5 <= 0.5 * remaining:
                    use_reps = 1
                    adj.append(f"{n}: lento ({t_fit:.1f}s/treino); validação reduzida para 1×5-fold.")
                else:
                    skipped.append(n); continue
        s_ = time.time()
        try:
            def model_progress(p, text):
                if progress:
                    progress(start + 0.8 * p / len(names),
                             f"Modelo {i + 1}/{len(names)} · {n} · {text}")
            r = cross_val(make_pipeline(pre, m, task=task), Xs, ys, task, 5, use_reps, seed,
                          progress=model_progress if progress else None)
        except Exception as e:
            adj.append(f"Modelo {n} falhou e foi ignorado: {str(e)[:60]}"); continue
        dt = time.time() - s_
        fit_t[n] = dt / (5 * use_reps)
        rows.append({"Modelo": n, **{k: v[0] for k, v in r.items()}, **{f"{k} (±)": v[1] for k, v in r.items()},
                     "tempo (s)": round(dt, 1)})
        if progress: progress(0.15 + 0.8 * (i + 1) / len(names), f"Validação concluída: {n}")
    if skipped:
        adj.append("Pulados por orçamento de tempo: " + ", ".join(skipped))
    board = pd.DataFrame(rows).sort_values(key, ascending=False).reset_index(drop=True)

    best = board.loc[0, "Modelo"]
    m, g = get_model(best)
    combos = int(np.prod([len(v) for v in g.values()])) if g else 0
    est_tune = fit_t.get(best, 1.0) * 5 * combos * (len(Xtr) / max(len(Xs), 1))
    do_tune = bool(g) and (not rapido) and est_tune < 0.8 * (budget_s - (time.time() - t0))
    if g and rapido:
        adj.append("Modo rápido: otimização de hiperparâmetros desligada (use 'completo' para ativá-la).")
    elif g and not do_tune:
        adj.append(f"Otimização de hiperparâmetros desligada (estimada em {est_tune:.0f}s, acima do orçamento restante).")
    if progress: progress(0.97, f"Ajustando {best}")
    final = make_pipeline(pre, m, g, tune=do_tune, task=task)
    final.fit(Xtr, ytr)
    if progress: progress(0.98, "Avaliando o melhor modelo nos dados de teste")
    pred = final.predict(Xte)
    test = reg_metrics(yte, pred) if task == "regression" else clf_metrics(yte, pred)
    if progress: progress(1.0, "Treino e teste concluídos")
    return {"task": task, "target": target, "features": feats, "log": log, "adjustments": adj, "config": best_cfg,
            "leaderboard": board, "best": best, "best_params": getattr(final, "best_params_", {}),
            "test_metrics": test, "model": final, "y_true": np.asarray(yte), "y_pred": np.asarray(pred),
            "n_train": len(Xtr), "n_test": len(Xte), "X_test": Xte, "data": df[feats + [target]], "profile": prof, "seed": seed, "seconds": round(time.time() - t0, 1)}


def _md_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def report_md(r) -> str:
    L = ["# Relatório automático", "",
         f"- Tipo: **{'Regressão' if r['task']=='regression' else 'Classificação'}**  |  Alvo: `{r['target']}`",
         f"- Treino: {r['n_train']} linhas  |  Teste (nunca visto): {r['n_test']} linhas  |  Tempo: {r['seconds']} s",
         f"- Características usadas ({len(r['features'])}): " + ", ".join(f"`{c}`" for c in r['features'][:30]), "",
         "## Decisões automáticas", *[f"- {x}" for x in r["log"]], "",
         "## Auto-ajustes", *[f"- {x}" for x in r["adjustments"]], "",
         "## Ranking (validação cruzada repetida 5-fold, só no treino)", "",
         _md_table(r["leaderboard"].round(4)), "",
         f"## Melhor modelo: **{r['best']}**", f"Hiperparâmetros: `{r['best_params']}`", "",
         "## Desempenho no conjunto de teste", *[f"- {k}: {v:.4f}" for k, v in r["test_metrics"].items()]]
    return "\n".join(L)
