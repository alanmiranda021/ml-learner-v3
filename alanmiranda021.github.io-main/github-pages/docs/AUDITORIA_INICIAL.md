# Auditoria inicial — 13/10/2026

Auditoria anterior às alterações. Branch `main`, commit `927c77b`. O Git tem
raiz na pasta pai do projeto. Havia alteração preexistente em `.vscode/extensions.json`,
fora desta entrega; ela será preservada. Não há AGENTS.md encontrado no projeto.

Comandos: `git status --short`, `git branch --show-current`, `git log -5 --oneline`,
`rg --files github-pages`, `python --version`, `python -m pip freeze`, `python -m pytest -q`.
Linha de base: **4 passed in 7.36s**, Python 3.14.8, sklearn 1.9.1,
SciPy 1.18.1, NumPy 2.5.3, pandas 3.0.6, Streamlit 1.65.0,
imbalanced-learn 0.14.2, h5py 3.16.0, SHAP 0.52.0. PyTorch não instalado.
O navegador usa Stlite 0.85.1 e dependências próprias em `boot.js`.
O início no navegador ainda não foi verificado nesta linha de base.

| Funcionalidade | Arquivo/função | Estado atual | Problema | Ação | Teste |
|---|---|---|---|---|---|
| CSV/Excel | core/io.py/load_table | confirmado em código | MAT escolhe maior matriz sem consulta | inspeção e seleção de variável | loaders |
| MAT objetos | core/io.py/_decode_matlab_table | confirmado em código | API privada SciPy, significado desconhecido | preservar fallback e registrar restrição | MAT |
| Tarefa | core/auto.py/detect | confirmado em código | <=15 valores numéricos vira classe | contrato explícito; heurística reversível | TaskConfig |
| Atributos | core/auto.py/detect | confirmado em código | correlação com alvo completo decide exclusão | alertar sem excluir por correlação | leakage |
| Imputação/IQR/escala | core/preprocess.py | confirmado em código | aprendidos em pipeline | preservar | pipeline spy |
| Remoção de linhas | app.py/remove_outlier_rows | confirmado em código | aprende no dataset inteiro | restringir modo científico | leakage |
| Suavização | core/preprocess.py/smooth_columns | confirmado em código | janela centrada usa futuro e cruza split | impedir uso pré-split científico | temporal |
| PCA | core/preprocess.py/build_preprocessor | confirmado em código | só numéricas, antes de categorias | manter compatibilidade e adicionar redução completa | reduction |
| KPCA/features | core/ | ausentes | patch não foi aplicado | implementar incrementalmente | KPCA/signals |
| Samplers | core/evaluate.py/make_pipeline | confirmado em código | pré→sampler→modelo; sem política categórica | validar combinações | sampler spy |
| Holdout/CV | core/evaluate.py | confirmado em código | somente divisões aleatórias | grupos/tempo e CV interna compatível | split leakage |
| GroupKFold/nested CV/LOF/MAD/Tanh | — | relatado anteriormente, não encontrado | não corresponde a árvore atual | não assumir implementação | novos testes |
| AutoML | core/auto.py/run_auto | confirmado em código | teste reservado para vencedor, seleção pré-split por correlação | corrigir seleção e contrato | classificação/regressão |
| SHAP | core/report_pdf.py | ausente | relatório usa permutation_importance do pipeline completo | adicionar explicação explícita compatível | explain |
| Bundles | core/predict.py | confirmado em código | sem versão/schema completo | versionar mantendo legado | roundtrip |
| SAE/cenários | — | proposto, ausente | não confundir com SMOTE | implementação local com diferenças registradas | SAE/experiments |
| Dados_navio_nig.mat | — | pendente de verificação | arquivo não fornecido/encontrado | não inventar schema | pendente |
| Artigo/patch | anexo textual | especificação disponível | patch e PDF não estavam no repo | consultar artigo; não aplicar patch inexistente | rastreabilidade |

Nenhum resultado do artigo é resultado desta aplicação. A auditoria diferencia
as funções observadas dos recursos apenas citados no plano.
