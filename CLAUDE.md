# Projeto ml-learner v3

- Lógica em `core/`; UI em `app.py`.
- Todo pré-processamento que aprende parâmetros fica dentro do pipeline.
- Suavização temporal deve ser causal; nunca usar janela centrada para inferência temporal.
- SMOTE/ADASYN somente dentro do treino.
- Permutação e SHAP devem usar dados fora do ajuste quando a estratégia disponibilizar teste.
- Em K-fold/GroupKFold/TimeSeriesSplit simples, não fazer GridSearch sobre a mesma CV usada para estimar desempenho; recomendar CV aninhada.
- Evitar oversubscription: CV/GridSearch paraleliza, estimadores internos devem ser single-thread.
- Não ocultar warnings de convergência globalmente.
- Alteração de configurações deve invalidar resultados antigos.
- Rodar `pytest -q` antes de entregar.
