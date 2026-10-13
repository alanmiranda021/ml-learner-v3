# Relatório automático

- Tipo: **Classificação**  |  Alvo: `target`
- Treino: 96 linhas  |  Teste (nunca visto): 24 linhas  |  Tempo: 3.15690979998908 s
- Características usadas (4): `f1`, `f2`, `f3`, `f4`

## Decisões automáticas
- Teste externo reservado; somente vencedor selecionado na CV é avaliado.

## Auto-ajustes
- Ordem: imputação/IQR/escala/categorias → aumento → redução completa → modelo.

## Ranking (validação cruzada somente no treino externo; consulte configuração/manifesto)

| Modelo | Acurácia | Acurácia balanceada | MCC | F1 (macro) | F1 (weighted) | Precisão (macro) | Recall (macro) | Acurácia (±) | Acurácia balanceada (±) | MCC (±) | F1 (macro) (±) | F1 (weighted) (±) | Precisão (macro) (±) | Recall (macro) (±) | tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 / Random Forest | 0.8958 | 0.8795 | 0.7734 | 0.8842 | 0.8948 | 0.8944 | 0.8795 | 0.0 | 0.0133 | 0.0027 | 0.0033 | 0.0016 | 0.0111 | 0.0133 | 0.5151 |
| E2 / Random Forest | 0.8854 | 0.8714 | 0.7537 | 0.8738 | 0.8848 | 0.8829 | 0.8714 | 0.0104 | 0.0052 | 0.017 | 0.0071 | 0.0084 | 0.0226 | 0.0052 | 0.4369 |
| E3 / Random Forest | 0.8854 | 0.8714 | 0.7537 | 0.8738 | 0.8848 | 0.8829 | 0.8714 | 0.0104 | 0.0052 | 0.017 | 0.0071 | 0.0084 | 0.0226 | 0.0052 | 0.3939 |
| E1 / SVM (RBF) | 0.875 | 0.8567 | 0.7255 | 0.8604 | 0.8735 | 0.8692 | 0.8567 | 0.0208 | 0.0361 | 0.0506 | 0.0272 | 0.023 | 0.0141 | 0.0361 | 0.0537 |
| E3 / SVM (RBF) | 0.8646 | 0.8553 | 0.711 | 0.8527 | 0.8647 | 0.856 | 0.8553 | 0.0104 | 0.0052 | 0.0115 | 0.0066 | 0.0083 | 0.0167 | 0.0052 | 0.0538 |
| E2 / SVM (RBF) | 0.8542 | 0.8406 | 0.6872 | 0.8397 | 0.8534 | 0.8472 | 0.8406 | 0.0 | 0.0199 | 0.0123 | 0.0064 | 0.0029 | 0.0079 | 0.0199 | 0.0552 |
| E4 / SVM (RBF) | 0.7604 | 0.7282 | 0.4894 | 0.7342 | 0.7575 | 0.7639 | 0.7282 | 0.0729 | 0.0498 | 0.1433 | 0.0641 | 0.0653 | 0.096 | 0.0498 | 0.1126 |
| E4 / Random Forest | 0.75 | 0.7334 | 0.4815 | 0.7316 | 0.7505 | 0.7491 | 0.7334 | 0.0833 | 0.0579 | 0.1454 | 0.0745 | 0.0767 | 0.0882 | 0.0579 | 0.4822 |
| E1 / Baseline (maioria) | 0.6458 | 0.5 | 0.0 | 0.3924 | 0.5069 | 0.3229 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0551 |
| E2 / Baseline (maioria) | 0.6458 | 0.5 | 0.0 | 0.3924 | 0.5069 | 0.3229 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0519 |
| E3 / Baseline (maioria) | 0.6458 | 0.5 | 0.0 | 0.3924 | 0.5069 | 0.3229 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0503 |
| E4 / Baseline (maioria) | 0.6458 | 0.5 | 0.0 | 0.3924 | 0.5069 | 0.3229 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.184 |

## Melhor modelo: **E1 / Random Forest**
Hiperparâmetros: `{}`

## Desempenho no conjunto de teste
- Acurácia: 0.9583
- Acurácia balanceada: 0.9688
- MCC: 0.9129
- F1 (macro): 0.9545
- F1 (weighted): 0.9589
- Precisão (macro): 0.9444
- Recall (macro): 0.9688