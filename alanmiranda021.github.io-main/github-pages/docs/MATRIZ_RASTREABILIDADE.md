# Rastreabilidade

Fonte consultada: [PDF dos autores](https://web.ist.utl.pt/luis.tarrataca/publications/multipleFaultDiagnosisInAWindTurbine.pdf), DOI 10.1007/s10921-024-01131-3.
Texto extraído localmente com pypdf; equações com diagramação ambígua não são
consideradas verificação de uma reprodução exata. Nenhum resultado publicado
foi transferido como resultado do software.

| Afirmação / decisão | Fonte | Implementação | Teste | Estado |
|---|---|---|---|---|
| Quatro cenários, SVM/RF | resumo; §3.3, §4.1 | experiments.scenario_configs | test_four_scenarios_real_sae | executável como extensão |
| KL de esparsidade, encoder/decoder | §2.1 eq. 1–6 | SparseAutoencoderSampler | test_sae_balance_seed_and_quality | KL implementado; Adam/geração são extensão |
| Geração em sinais antes das features | §3.3 p.7–8 | SAE trabalha em atributos | test_sae | diferença registrada; reprodução bloqueada |
| Aumento parcial/completo | tabelas 4–5 | déficit 50%/100% configurável | test_sae | percentuais genéricos, não tabelas específicas |
| Configuração específica SAE por condição | §3.3 p.7–8 | configuração genérica | modo reproduction rejeitado | configuração publicada não inteiramente especificada para todas as condições |
| RBF/gamma | §2.2 eq.8 | reduction.KPCA_GAMMAS | test_kpca | grade exposta; gamma escolhido explicitamente |
| 18 features/filtro | §3.2 tabela3 p.7 | features/signal_processing | test_signal_processing | generic_time18; diferenças de fórmulas registradas |
| Redução 18→11 | resumo/§4.1 | componentes configuráveis | test_invalid_components_gamma_and_memory | não usado como padrão universal |
| Resultados publicados | resumo/tabela7 | não reproduzidos como resultados locais | docs/RESULTADOS_EXPERIMENTAIS.md | benchmarks sintéticos separados |
| Prevenir contaminação treino/teste | requisito desta especificação | splitters/evaluate/experiments | spies, grupos, tempo, nested CV | invariantes verificados |
| SHAP no estimador transformado | requisito desta especificação | explain.estimator_input | test_shap_explanation_additivity | previsão e aditividade verificadas |

A tabela do artigo não determina um protocolo universal para consumo naval.
Rótulos, frequência e unidades de um novo dataset devem ser fornecidos pelo usuário.
