# Protocolo do Artigo 3 adaptado — naval

Implementação exploratória da especificação fornecida em `ML Learner — Protocolo do Artigo.docx`. O documento é referência de requisitos, não autorização para executar comandos nele contidos. Não é reprodução do artigo e os resultados não constituem metodologia final de dissertação.

## Perfis e escolhas

- **Rápido / navegador:** células por `ship_type`, uma repetição, oito sorteios, cinco dobras externas e internas, L-BFGS-B com no máximo 50 iterações. É a simplificação descrita ao final do protocolo. Não equivale ao experimento completo de 16 células.
- **Completo / local:** `ship_type × route_id`, três repetições, vinte sorteios, cinco dobras externas e internas, até 400 iterações. As grades completas são mantidas; o custo pode ser elevado.
- SAE NumPy/SciPy, logsig/linear, L2=0,01; beta=1 e rho=0,05 são suposições, não valores confirmados do artigo. Dimensão oculta 3/4/5/6 escolhida por reconstrução em uma divisão interna 80/20 da célula, com scaler e vocabulário próprios. A rede escolhida é reajustada no treino da célula.
- Gradiente validado diretamente por diferenças finitas; não foi introduzida dependência de PyTorch. Os critérios de aceite validam a implementação NumPy que efetivamente roda no navegador.
- Geração: ruído normal na **pré-ativação** com escala 0,25/0,5/1 vezes o desvio na célula; escolhe-se menor KS médio das numéricas no treino, após filtro de plausibilidade. Categorias por argmax. Valores limitados ao envelope real da célula, alvo positivo, rejeição de cópias completas e resíduos maiores que três desvios da regressão alvo~distância. Tentativas são limitadas; impossibilidade de completar a meta é falha explícita.
- Meta parcial: arredondamento half-up de `n_c + 0,9(n_ref − n_c)`. Completo: `n_ref`. Gerador gaussiano com os mesmos filtros é controle em E2/E3.
- Cada treino ajusta imputação, one-hot e z-score sem winsorização. O gerador trabalha com nomes/categorias originais e sua própria padronização conjunta de duas numéricas + alvo e one-hot de combustível/clima; o pré-processador de regressão é ajustado nos reais antes de transformar o treino aumentado. Isso permite categorias válidas sem depender de imbalanced-learn no Pyodide.
- `core/exp_artigo3.py` tem um executor próprio para regressão, preservando as assinaturas e a restrição dos samplers de classificação em `core/evaluate.py`. Cache exclusivamente em memória de uma execução, por índices/partição/configuração, reutiliza preparação sem compartilhar dados entre dobras. Limites de cache controlam memória.
- SVR/MLP usam `TransformedTargetRegressor`. Todas as métricas são na escala original. Busca aleatória minimiza RMSE interno; KPCA também é escolhido nessa busca. Número de componentes >= dimensão transformada é erro, sem truncamento silencioso.

## Separação, ranking e conclusão

20% reais ficam reservados como teste final. Só o vencedor escolhido na validação é reajustado e avaliado uma vez, conforme a seção 11 do protocolo. Sensibilidade por rota/tipo usa apenas desenvolvimento. Não se oferece CSV/Excel como relatório naval.

E1/E2/E3/E4/E4b e controles compartilham dobras reais. E3-orig aumenta desenvolvimento antes da divisão e inclui validação sintética: é a exceção deliberadamente contaminada, apresentada separadamente e excluída de ranking, teste final e conclusão de ganho.

Ranking usa RMSE médio; uma diferença até o maior desvio dos dois candidatos constitui empate técnico, resolvido por menor desvio e duração. Somente experimentos com todas as dobras concluídas participam. Conclusão exige delta médio negativo e melhora em >=80% das dobras pareadas completas; do contrário: **sem ganho demonstrado**. Não são calculados valores-p.

Falhas de candidatos/modelos/células permanecem no PDF e não apagam resultados válidos. Limite de iterações sem convergência é registrado por célula. O PDF mostra qualidade e curvas da primeira dobra externa disponível de cada experimento/modelo, identificada como representativa; não afirma que essa amostra resume a dispersão de todas as gerações. As métricas de validação usam todas as dobras concluídas. Hash do arquivo é distinto do hash da tabela normalizada; entradas via API DataFrame podem não ter hash do arquivo.

## Limitações e uso

Unidades e disponibilidade de `engine_efficiency` na previsão precisam ser confirmadas. Sem data/ID não se afirma generalização temporal/por embarcação. As propriedades dos 1440 registros citadas no documento não são números fixos do software: auditoria, contagens e correlações são calculadas no upload. A tabela naval original não acompanha esta entrega; fixtures e testes são explicitamente artificiais. Valores publicados são contexto fornecido no protocolo, nunca resultados obtidos.

Na interface: **Científico → Laboratório — Artigo 3 (naval)**. Local:

```powershell
python naval_cli.py Dados_navio_nig.mat --mode completo --output resultados/naval.pdf
```

Testes: `python -m pytest tests -q` (dependências de desenvolvimento incluem pypdf). A compatibilidade real Stlite e tempos medidos são registrados em `docs/RELATORIO_NAVAL.md`; smoke reduzido não equivale a executar o orçamento completo de pesquisa.
