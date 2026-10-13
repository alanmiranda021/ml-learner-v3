# Limitações e compatibilidade

| Recurso | Python local | Stlite |
|---|---|---|
| CSV/Excel/MAT numérico v4-v7.2 | sim | pacotes incluídos |
| MAT v7.3 | h5py opcional | reexportar v7/CSV |
| MATLAB structs/células/3D | inspeção, sem conversão automática | mesma restrição |
| Objetos MATLAB table | leitor legado SciPy privado, sem garantia universal | fallback preservado |
| Sinais longos, generic_time18, filtros | sim | módulos empacotados, depende de memória |
| PCA/KPCA/grupos/tempo | sim | limite KPCA 3.000 linhas de ajuste |
| SAE NumPy, E1–E4, ablação com aumento | imbalanced-learn requerido | bloqueado na UI |
| SHAP | dependência opcional, até 100 amostras | botão bloqueado |
| Manifesto/bundle/CSV/Markdown/PDF | sim | módulos empacotados |

Testes locais não substituem execução real em Pyodide. O boot foi verificado
quanto à presença de todos os arquivos Python; não houve teste de navegador real
nem publicação. A versão Stlite foi preservada. Upload limitado a 100 MiB,
matriz MAT a 10 milhões de elementos, folds 2–10, ajustes externos até 600,
sintéticos SAE até 10.000. Esses limites não garantem memória suficiente em
todo navegador/hardware. KPCA usa matriz quadrática e solver denso.

Não há reprodução exata do artigo; diferenças em sinais/atributos, fórmula de
curtose/folga, geração e particionamento estão na metodologia. Somente dados
sintéticos foram usados nos benchmarks documentados. Não há evidência de
validade física, diagnósticos reais ou generalização entre embarcações/máquinas.
Dados_navio_nig.mat não foi fornecido/encontrado; schema, unidades e tarefa
permanecem desconhecidos. Não há CSV/Parquet automático de structs MATLAB.

SAE não é SMOTE. Geração pode produzir quase cópias ou distribuição inadequada;
diagnósticos não são certificação. Não há aumento de regressão, classes inventadas,
SMOTENC, FFT/PSD/envelope, LOF treinável, protocolo físico de regressão nem
combinação de split temporal com grupos. Filtragem é offline, não causal.
Percentuais de aumento e hiperparâmetros devem ser pré-registrados para dados reais.

Seleção entre cenários usa CV interna ao treino externo; tuning cria CV aninhada
por candidato. Métricas de seleção não são estimativa independente do vencedor:
essa função é do teste externo reservado. Reexecutar após consultar teste prejudica
a interpretação. Intervalos de confiança e testes estatísticos não implementados.

Inferência tabular aceita schema de atributos; para dados brutos de sinais é
preciso repetir a extração com SignalConfig exportado antes da previsão.
O bundle não executa sozinho aquisição→janelas. Unidades desconhecidas não são
inferidas; declaração do usuário é necessária. Manifestos do novo modo científico
são completos; relatórios automáticos legados não exportam todas as partições CV.
Extra/categorias novas geram metadados, não rejeição automática; tipos numéricos
incompatíveis e colunas ausentes são rejeitados. Bundles joblib executam código
durante carga e não são apropriados para arquivos de terceiros não confiáveis.

O Git desta pasta tem raiz no diretório pai e ignorava a cópia `*-main/`.
Os commits locais usam caminhos explícitos forçados exclusivamente para o ML Learner.
Não houve push, alteração de remote ou publicação; valide qual checkout corresponde
ao repositório de publicação antes de enviar commits.
