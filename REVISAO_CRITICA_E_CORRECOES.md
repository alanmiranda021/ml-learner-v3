# Revisão crítica e correções — ml-learner v3

## Status

A versão v3 incorpora os problemas confirmados na revisão da v2.

| Item | v2 | v3 |
|---|---|---|
| SHAP | X bruto chegava ao estimador final | X passa pelo `pre`; SHAP usa features transformadas |
| Savitzky-Golay | janela centrada, não causal | janela trailing/causal com `savgol_coeffs(pos=window-1)` e preenchimento apenas com valores passados |
| Permutação | Hold-out podia usar X/y completos | usa `X_test/y_test` guardados no split |
| `.xls` | `xlrd` ausente | `xlrd>=2.0` no requirements |
| Tuning em K-fold | `tune=False` silencioso | aviso explícito + recomendação de CV aninhada |
| CPU | estimador e GridSearch podiam paralelizar simultaneamente | estimadores internos single-thread; CV/GridSearch paralelizam |
| Streamlit | recomputações e resultados antigos | cache em diagnósticos/interpretação + invalidação por configuração |
| Upload múltiplo | concatenação podia criar NaNs silenciosos | exige mesmas colunas e mesma ordem |
| Detecção de tarefa | `nunique() <= 15` podia classificar regressão discreta | alvo numérico é regressão por padrão |
| Warnings | `ignore` global | warnings úteis permanecem visíveis |

## Testes executados

```text
7 passed
```

A suíte cobre:

- Winsorizer;
- MADScaler;
- TanhScaler;
- LOF;
- pipeline + Hold-out;
- causalidade do Savitzky-Golay;
- preservação do teste OOS no Hold-out.

## Observação

A instalação dos pacotes opcionais `streamlit`, `shap`, `imbalanced-learn` etc. não foi necessária para executar a suíte unitária disponível no ambiente de teste. O `requirements.txt` da entrega contém essas dependências para a execução completa da aplicação.
