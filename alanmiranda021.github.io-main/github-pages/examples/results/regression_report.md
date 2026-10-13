# Relatório automático

- Tipo: **Regressão**  |  Alvo: `target`
- Treino: 96 linhas  |  Teste (nunca visto): 24 linhas  |  Tempo: 1.5638773000100628 s
- Características usadas (4): `f1`, `f2`, `f3`, `f4`

## Decisões automáticas
- Teste externo reservado; somente vencedor selecionado na CV é avaliado.

## Auto-ajustes
- Ordem: imputação/IQR/escala/categorias → aumento → redução completa → modelo.

## Ranking (validação cruzada somente no treino externo; consulte configuração/manifesto)

| Modelo | R2 | RMSE | MAE | MedAE | R2 (±) | RMSE (±) | MAE (±) | MedAE (±) | tempo (s) |
|---|---|---|---|---|---|---|---|---|---|
| R1 / Ridge | 0.9805 | 14.0694 | 10.3886 | 7.9794 | 0.0007 | 0.2869 | 0.3129 | 0.3084 | 0.0279 |
| R1 / Random Forest | 0.7182 | 53.4324 | 41.8116 | 33.5752 | 0.0124 | 1.0229 | 3.2629 | 5.7844 | 0.3816 |
| R3 / Ridge | 0.1368 | 93.0474 | 74.5785 | 60.6254 | 0.172 | 9.0962 | 13.1639 | 23.4558 | 0.0345 |
| R2 / Ridge | 0.092 | 95.7341 | 76.9689 | 61.6651 | 0.1134 | 5.7227 | 9.9673 | 20.2023 | 0.0249 |
| R1 / SVR (RBF) | -0.037 | 102.5212 | 82.9528 | 74.2881 | 0.015 | 0.4412 | 3.2306 | 11.4272 | 0.0331 |
| R3 / Random Forest | -0.0452 | 102.2035 | 79.8549 | 64.6577 | 0.2404 | 11.6171 | 14.4255 | 17.3663 | 0.4142 |
| R3 / SVR (RBF) | -0.0452 | 102.8929 | 83.4182 | 74.5414 | 0.0515 | 2.2377 | 4.8887 | 12.925 | 0.0339 |
| R2 / SVR (RBF) | -0.0622 | 103.7415 | 84.2512 | 74.6676 | 0.0414 | 1.7182 | 4.3882 | 12.6236 | 0.0258 |
| R1 / Baseline (média) | -0.1086 | 106.0076 | 85.9476 | 78.3958 | 0.0006 | 0.2797 | 2.7802 | 9.958 | 0.0268 |
| R2 / Baseline (média) | -0.1086 | 106.0076 | 85.9476 | 78.3958 | 0.0006 | 0.2797 | 2.7802 | 9.958 | 0.0268 |
| R3 / Baseline (média) | -0.1086 | 106.0076 | 85.9476 | 78.3958 | 0.0006 | 0.2797 | 2.7802 | 9.958 | 0.0357 |
| R2 / Random Forest | -0.1641 | 107.9812 | 86.5798 | 72.5182 | 0.2463 | 11.2404 | 13.0315 | 22.9337 | 0.4192 |

## Melhor modelo: **R1 / Ridge**
Hiperparâmetros: `{}`

## Desempenho no conjunto de teste
- R2: 0.9835
- RMSE: 11.5950
- MAE: 8.7913
- MedAE: 6.1092