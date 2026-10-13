# Metodologia implementada

## Contrato e partições

`TaskConfig` declara uma tarefa, alvo único, preditores e metadados. Alvo numérico
sugere regressão mesmo com poucos valores distintos. A UI exige confirmação.
IDs, grupos e tempo ficam fora dos preditores. Ausentes no alvo exigem exclusão
explicitamente confirmada; infinitos são rejeitados. Perfil não altera dados.

O modo científico rejeita duplicatas em divisões aleatórias. O usuário precisa
deduplicar explicitamente ou definir grupos de origem. Grupos de aquisições
ficam juntos. Tempo é normalizado para cronologia UTC quando textual; treinamento
precede validação, timestamps iguais não cruzam a fronteira, e gap é em linhas.
Combinação de grupos e tempo no mesmo splitter não é implementada.

`run_experiment` separa o teste externo primeiro. Todos os cenários compartilham
os folds externos de seleção dentro do treino reservado. Se tuning estiver ativo,
cada fold cria suas partições internas mantendo grupo/tempo. O vencedor é escolhido
por macro-F1 ou R² na validação; somente esse vencedor é avaliado no teste externo.
Nenhum teste externo é usado para selecionar sintéticos ou hiperparâmetros.
O teste não deve ser consultado repetidamente para modificar o protocolo.

Ordem: imputação → winsorização IQR → escala → concatenação de categorias → aumento
(apenas classificação numérica) → redução da matriz completa → estimador.
Samplers têm `fit_resample`, nunca transformação na previsão. Redutores são clonados
em cada fold. Quantidade inválida de componentes provoca erro, sem truncamento.
KPCA RBF usa componentes fixos/gamma declarado, limite de 3.000 linhas de ajuste
e solver denso; seus autovalores não são anunciados como variância PCA.

A implementação antiga `pca=True` permanece no ramo numérico para compatibilidade
com modelos antigos. A redução nova do modo científico atua na matriz inteira.
O automático conserva o fluxo legado, agora com confirmação da tarefa e sugestões
de exclusão de colunas determinadas exclusivamente no treino externo.
Suavização e remoção de outliers pré-split estão bloqueadas no modo manual:
IQR treinável e modo sinais são os caminhos válidos disponíveis.

## Sinais e generic_time18

O modo de sinais recebe CSV/Excel/MAT já tabulado em formato longo: ID de aquisição,
canal numérico, rótulo constante confirmado, ordem das amostras e fs fornecida pelo
usuário. Não estima fs nem inventa rótulos. Butterworth SOS e FIR Hamming usam
filtragem offline de fase zero por aquisição. Filtra a aquisição inteira antes
de segmentar; isso é válido para grupos independentes e não representa previsão
causal online. Nyquist e padding inválidos provocam erro. Janelas completas de
tamanho fixo; cauda curta é descartada e contagem exportada. Sobreposição preserva
ID de origem, índices e limites exclusivos finais. Split por origem é obrigatório.

Para x de comprimento N, μ=mean(x), pico=max(abs(x)), RMS=sqrt(mean(x²)):

| Campo | Definição genérica | Convenção/checagem da tabela publicada |
|---|---|---|
| mean | mean(x) | média |
| rms | sqrt(mean(x²)) | RMS |
| std | std(x, ddof=0) | população |
| peak | max(abs(x)) | pico absoluto |
| skewness | scipy.skew, bias=True | fórmula impressa requer validação visual adicional |
| kurtosis | scipy.kurtosis, fisher=True, bias=True | excesso; difere da curtose não subtraída impressa |
| crest_factor | pico/RMS | pico absoluto explicitado |
| clearance_factor | pico/mean(sqrt(abs(x)))² | difere do denominador impresso; não verificado como réplica |
| shape_factor | RMS/mean(abs(x)) | valor absoluto |
| impulse_factor | pico/mean(abs(x)) | interpretação de max explicitada |
| peak_to_peak | max(x)-min(x) | amplitude |
| minimum | min(x) | mínimo com sinal |
| total | sum(x) | soma com sinal |
| variance | var(x, ddof=0) | população |
| standard_error | std(x, ddof=0)/sqrt(N) | população |
| median | median(x) | mediana |
| energy | sum(x²) | energia discreta sem integrar dt |
| energy_entropy | -sum(p*log2(p)), p=x²/sum(x²) | energia normalizada, base 2 |

Constantes têm assimetria/curtose definidas como zero por convenção operacional.
Denominador zero em fatores gera zero. Sinal todo zero tem entropia zero.
Ausentes/infinitos são rejeitados. Features mínimas: 3 amostras. RMS/pico/média
preservam unidade, energia e variância usam unidade²; soma depende de N.
`article3_verified` é rejeitado para evitar equivalência não demonstrada.

## SAE e cenários inspirados

SAE NumPy local: h=sigmoid(XW+b), reconstrução=hWd+bd. Objetivo:
MSE + λ/2 (||W||²+||Wd||²) + β sum_j KL(ρ || mean(h_j)).
Gradientes analíticos, Adam em minibatches, seed fixa, early stopping pela perda
de treino e restauração dos melhores pesos. Uma rede por classe do fold, mínimo
de três amostras por classe. Padronização interna é aprendida apenas nessa classe.
Perturbação gaussiana dos códigos latentes de âncoras reais e decodificação;
clip no envelope empírico da classe é registrado e **não prova validade física**.
Limites físicos opcionais devem estar no espaço transformado fornecido ao sampler.
Diagnósticos: contagens, MSE reconstrução, distância mínima, quase cópias,
extrapolações antes do clip e deslocamentos de médias. Não há garantia de ganho.

E1 sem aumento/redução; E2 fecha 50% do déficit de cada classe; E3 fecha 100%;
E4 aplica E3 e KPCA. O percentual de E2 é escolha da extensão, ajustável por cenário
na API/YAML. SVM/RF e baseline obrigatório; ablação acrescenta class_weight,
SMOTE/ADASYN e PCA. Regressão compara nenhuma redução/PCA/KPCA sem síntese de y.
Modelos alternativos podem ser escolhidos no catálogo/API. Categorias com aumento
são rejeitadas; ainda não há SMOTENC integrado.

O artigo aumenta sinais antes da extração; este sampler aumenta atributos tabulares.
Adam, síntese latente, envelope e splits protegidos também são decisões desta
implementação. `article3_reproduction` é bloqueado. O artigo pode ser consultado em
[Felix et al. (2024)](https://web.ist.utl.pt/luis.tarrataca/publications/multipleFaultDiagnosisInAWindTurbine.pdf).

## Avaliação e interpretação

Baseline Dummy sempre incluído. Classificação: accuracy, balanced accuracy, macro/
weighted F1, macro precision/recall, MCC e relatório por classe. Regressão: MAE,
RMSE, MedAE e R². Média/desvio dos folds descrevem dispersão, não IC independente.
Não há significância estatística, calibração, ROC/PR-AUC ou MAPE anunciados.

SHAP usa fundo fornecido **do treino**, transforma-o e transforma exemplos de teste
com o pipeline ajustado, omitindo sampler. Explica probabilidades, decision_function
ou regressão no estimador final; verifica identidade da previsão. Nomes PCA/KPCA
são componentes, nunca sensores originais. O caller da API deve fornecer treino
legítimo como fundo. No navegador, permanece importância por permutação do pipeline
completo no PDF. Importância não implica causalidade.

## Artefatos

Manifesto contém configuração/hash, hash de dataset+schema, IDs posicionais dos
folds, seed, commit, ambiente, métricas e diagnósticos. Não contém linhas brutas.
Bundle v2 guarda pipeline, schema, categorias, ordem, unidades declaradas e manifesto;
v1 continua legível. Novas categorias/extras ficam registradas em `inference_warnings`.
Unidades são verificadas quando fornecidas; CSV sem unidades não permite inferi-las.
Previsão usa colunas ordenadas do contrato e não reexecuta aumento. Carregue apenas
joblib confiável. Bundle científico mantém o ajuste do treino externo; o fluxo
automático oferece refit explícito em todos os dados para implantação.
