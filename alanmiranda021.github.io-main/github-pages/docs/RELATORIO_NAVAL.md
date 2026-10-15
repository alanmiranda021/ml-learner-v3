# Entrega do laboratório naval — 15/10/2026

## Implementado

Nova aba em **Científico → Laboratório — Artigo 3 (naval)**, carregada também pelo `boot.js` no Stlite. Regressão naval isolada dos experimentos anteriores: SAE NumPy com L-BFGS-B e gradiente analítico, gerador por célula com plausibilidade, controles gaussianos, KPCA, SVR/RF e MLP/KNN opcionais, validação aninhada estratificada, diagnóstico E3-orig separado, ranking, teste final reservado, sensibilidade e PDF único. Há CLI em `naval_cli.py`.

O bloqueio CO2/consumo é obrigatório e simétrico para o alvo secundário. As novas opções de KPCA no pré-processador preservam os padrões anteriores. Os samplers antigos continuam exclusivos de classificação; o executor naval usa `fit_resample` próprio sem exigir imbalanced-learn no navegador. As decisões estão em [SUPOSICOES.md](../SUPOSICOES.md).

## Verificações

- Gradiente SAE por diferenças finitas, erro relativo <1e-5; perda decrescente, KL e efeito da esparsidade.
- Balanceamento parcial/completo, categorias, envelope, positividade, ausência de cópias reais e reprodutibilidade para SAE e controle gaussiano.
- Espiões de ajustes do gerador, vocabulário e KPCA; partições de treino/validação disjuntas, nenhuma linha do holdout final ajusta o gerador, categoria exclusiva do teste ausente do vocabulário. Ajustes internos são subconjuntos do treino externo.
- Teste dos oito cenários com falha deliberada de um modelo: os demais resultados e o PDF são preservados. PDF aberto com pypdf e verificado quanto a páginas, títulos, hash e proveniência.
- AppTest inicializa a interface naval e verifica bloqueio quando a exclusão de CO2 é removida.
- Smoke local `python examples/naval_smoke.py`: **oito cenários × SVR/RF**, CV externa/interna de duas dobras, um sorteio, SAE de até 50 iterações, uma dimensão oculta candidata, com sensibilidade. **12,392 s**, 16 linhas de comparação completas, nenhuma falha, um teste final e PDF válido. Fixture artificial de 98 linhas; componentes KPCA=2 são uma redução explícita do orçamento de teste, não a grade científica padrão.
- Pyodide **0.29.2 real**, via `tests/naval_pyodide.cjs`: SAE naval 60 → 70 linhas, 50 iterações máximas, uma dimensão candidata, consumo positivo, sem cópias praticamente idênticas. Tempo do gerador **0,264 s**, excluído download/inicialização. Não depende de torch/TensorFlow.
- Stlite real no Edge headless: inicialização, upload artificial de 80 linhas, aba naval e controles habilitados, sem exceções. Script reproduzível: `tests/naval_browser.cjs`; `--run` também exercita a execução pelo botão.
- Execução pelo botão no Stlite/Edge com `--run`: **194,0 s até disponibilizar o PDF**, sem exceções de interface. Tabela artificial de 80 linhas, todos os oito cenários, SVR, cinco dobras externas/internas, oito sorteios, SAE até 50 iterações, grade oculta reduzida a `{3}`, gamma a `{0,01; 0,1}`, KPCA a `{5}`. Esse tempo inclui a geração do PDF e exclui a inicialização/download do ambiente; não é medição do conjunto naval original ou da grade completa.
- O script foi ampliado para clicar em baixar, aguardar a transferência e verificar o cabeçalho `%PDF`. A repetição adicional encontrou timeout de 240 s **na inicialização do Stlite, antes do upload**, portanto essa verificação de transferência permanece sem confirmação. A medição de 194 s acima comprova geração e disponibilização do botão, não a transferência efetiva do PDF para disco. A causa do timeout não foi isolada.

As suítes são executadas em processos separados para isolar módulos `core` de projetos diferentes: `github-pages/tests`, `projects` sem os testes arquivados em `baselines`, e `tests` da pasta superior. A coleta indiscriminada na pasta superior falha por cópias com nomes de teste iguais, importação do baseline arquivado e `test_mat.py` dependente de um MAT ausente. Esses problemas não foram alterados para esconder a falha de coleta.

Resultado final: **55 aprovados e um opcional ignorado** em `github-pages/tests`; **48 aprovados e quatro subtestes** em propulsão; **11 aprovados** em `tests` da pasta superior. `tests/test_progress.py` permanece aprovado. `git diff --cached --check` sem erros de whitespace; scripts Node verificados com `node --check`.

## Limites da validação

O MAT original com 1440 registros não está disponível no caminho referido pelo projeto. Portanto não foi executado o estudo científico real nem medido o tempo do perfil completo nesse conjunto. Os números acima são verificações de software com dados artificiais e orçamento identificado, não resultados navais nem prova de melhora de consumo.

O PDF contém métricas de todas as dobras executadas e resumo da qualidade sintética por dobra; curvas/histogramas detalhados usam a primeira dobra externa disponível por experimento/modelo, explicitamente identificada. A interpretação física depende da confirmação de unidades e da disponibilidade da eficiência antes da previsão.
