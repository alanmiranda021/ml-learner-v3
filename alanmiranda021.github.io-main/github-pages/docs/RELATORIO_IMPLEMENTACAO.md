# Relatório de implementação

## Entrega

- Auditoria anterior à implementação com 4 testes de referência.
- Contratos e perfil de dados, seleção MAT explícita e leitura de formatos mantida.
- Holdout/CV por grupo e tempo; CV interna compatível; teste externo reservado.
- Removida seleção por correlação com o alvo completo no automático; filtragem de
  atributos agora consulta apenas treino externo. Operações pré-split inseguras
  foram bloqueadas na UI manual, mantendo funções legadas para compatibilidade.
- Redução completa PCA/KPCA; extração generic_time18 com grupos e proveniência.
- SAE NumPy com esparsidade KL, Adam e geração latente; diagnósticos por classe.
- Cenários E1–E4 inspirados, ablação e regressão; Dummy e métricas complementares.
- Interface científica integrada ao app existente e à lista de arquivos Stlite.
- SHAP no espaço transformado, bundle v2/legado v1, manifesto e exportação.
- YAML/CLI, benchmark sintético reproduzível e workflow CI fornecido.

## Verificação e correções

`python -m pytest -q`: **44 passed in 27.12s**, sem warnings na execução final.
`python -m compileall -q github-pages`: compilação bem-sucedida.
`python github-pages/examples/benchmark.py`: dois protocolos executados;
artefatos e métricas em `RESULTADOS_EXPERIMENTAIS.md`.
`python github-pages/experiment_cli.py --help`: interface CLI validada.
AppTest: app inicia sem upload; confirmação habilita experimento; execução
de regressão produz resultados. Não foi iniciado servidor persistente.

Falhas detectadas durante desenvolvimento e corrigidas: ARPACK com matriz
constante (solver denso), teste UI tentando clicar antes do rerun da confirmação,
PDF científico esperando configuração legada e avisos de precisão indefinida
(zero_division explícito). Falhas não foram substituídas por métricas inventadas.

Instalados no ambiente local: reportlab (dependência existente que faltava),
PyYAML (CLI) e pypdf (consulta do artigo, não dependência do app).
`requirements-tested.txt` registra versões diretas observadas; não é lock
transitivo multiplataforma. CI fornece Python 3.12/3.13, ainda não executado no remoto.

## Pendências e limites

- Dados_navio_nig.mat ausente: nenhum schema, unidade ou tarefa inventado.
- article3_reproduction/article3_verified indisponíveis: SAE tabular é extensão,
  com diferenças documentadas frente à síntese de sinais e à tabela de fórmulas.
- Navegador real/compatibilidade completa Pyodide ainda precisa de teste;
  imports opcionais e arquivos do boot são cobertos pelos testes locais.
- Recursos além desta implementação e limitações estão em LIMITACOES.md:
  SMOTENC, conversão MATLAB complexa/Parquet, inferência direta de sinais,
  estatística inferencial, validação física, pico de memória e protocolos de domínio.

Commits locais separados por auditoria, base, contratos, sinais, backend científico,
interface/SHAP e documentação/qualidade. O repositório Git real é o diretório pai;
a cópia do site era ignorada. Foram versionados apenas caminhos do ML Learner.
A alteração preexistente em `.vscode/extensions.json` foi preservada.
Nenhuma publicação ou push foi executado.
