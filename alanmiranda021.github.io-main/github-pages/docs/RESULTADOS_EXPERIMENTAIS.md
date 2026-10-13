# Resultados realmente executados — 13/10/2026

Comando: `python github-pages/examples/benchmark.py`.
Somente bases **sintéticas**, geradas por sklearn; nenhuma conclusão naval,
diagnóstico de falha real ou reprodução das acurácias do artigo.

Configuração comum: 120 amostras, 4 atributos numéricos, 96 em treino externo e
24 em teste, seed do dataset 17, seed das partições/modelos 42, 2 folds, sem tuning.
Classificação: make_classification, 3 informativos, 0 redundantes, pesos [.65,.35].
Regressão: make_regression, ruído 10. Ambos têm Dummy como baseline obrigatório.

| Experimento | Candidatos | Vencedor pela CV | Teste externo |
|---|---|---|---|
| Classificação E1–E4 | SVM/RF/Dummy em 4 cenários, SAE 10 épocas, KPCA 2 componentes/gamma .1 | E1 / Random Forest | accuracy .958333; balanced accuracy .968750; macro-F1 .954459; MCC .912871 |
| Regressão R1–R3 | Ridge/SVR/RF/Dummy, nenhuma redução/PCA/KPCA | R1 / Ridge | R² .983510; RMSE 11.594961; MAE 8.791302; MedAE 6.109247 |

Tempo registrado nesta execução: classificação 3,16 s; regressão 1,56 s.
São tempos do experimento local, sem inicialização do interpretador; variam com
hardware/carga. Sem medição de pico de memória. Versões e plataforma constam nos
manifestos; ambiente de teste Python 3.14.8 Windows x86_64.

Artefatos: `examples/results/*_manifest.json`, `*_metrics.csv`, `*_report.md`,
`*_model.joblib` e `summary.json`. O script recria os dados sintéticos a partir
das seeds; manifests não guardam linhas brutas. Bundles dos exemplos mantêm
o treinamento externo (96 linhas), sem refit no teste.

Os rankings CSV incluem todos os cenários, enquanto o teste externo é apresentado
somente para o vencedor. Neste ensaio **o aumento SAE não venceu o baseline sem
aumento E1**. Este resultado não valida nem invalida o método em outro domínio.

## Verificação automatizada

Linha de base: 4 testes. Suíte expandida inclui inspeção MAT/Excel/CSV, contratos,
signals, Nyquist/padding, PCA/KPCA, samplers espiões, grupos/tempo, CV aninhada,
SAE/seeds, fluxos de classificação/regressão, SHAP/aditividade, bundles,
inicialização Streamlit, execução pela interface, PDF completo e CLI YAML.
Quantidade e resultado finais estão em `RELATORIO_IMPLEMENTACAO.md`.

Não foi realizado benchmark com Dados_navio_nig.mat, teste real em navegador
Pyodide, nem execução do workflow no GitHub. AppTest verifica Streamlit local.
