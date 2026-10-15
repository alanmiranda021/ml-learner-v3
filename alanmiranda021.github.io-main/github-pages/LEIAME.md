# ML Learner — laboratório do portfólio

## Laboratório naval — Artigo 3 adaptado

No modo **Científico**, abra a aba **Laboratório — Artigo 3 (naval)**. A tabela deve conter os seis preditores navais e `fuel_consumption`; `CO2_emissions` é excluída obrigatoriamente. A nova aba roda E1–E4b, E3-orig diagnóstico e controles gaussianos, com validação aninhada, teste final reservado e um PDF por execução. Resultados são exploratórios.

O navegador permite apenas o perfil rápido (células por tipo de navio, SAE limitado a 50 iterações). O completo usa navio × rota e deve ser executado localmente. Consulte [SUPOSICOES.md](SUPOSICOES.md) para escolhas e limites. Para executar sem interface, nesta pasta: `python naval_cli.py Dados_navio_nig.mat --mode completo --output naval.pdf`.

A página de apresentação é `projects/ml-learner.html`. Ela inicia `github-pages/index.html` sob demanda, usando Stlite/Pyodide para executar Python no navegador.

## Visualizar localmente

Na pasta `alanmiranda021.github.io-main`, execute:

```powershell
python -m http.server 8080
```

Abra http://localhost:8080/projects/ml-learner.html. O laboratório precisa de HTTP; abrir por file:// não funciona.

## Publicar no portfólio

Mantenha a estrutura existente. Envie a pasta `github-pages/` inteira, incluindo `boot.js`, `app.py`, `core/` e `examples/`, junto com `projects/ml-learner.html`, `projects/ml-learner.css`, `projects/ml-learner.js` e o index.html do portfólio. Não substitua o index.html da raiz pelo desta pasta.

No GitHub Pages, use a raiz do repositório do portfólio como origem. A página ficará em `/projects/ml-learner.html` e o laboratório em `/github-pages/`.

Mantenha o arquivo `.nojekyll` na raiz do repositório. Ele impede que o processamento Jekyll exclua arquivos como `core/__init__.py`. O carregador também cria esse marcador vazio diretamente no ambiente Python, sem depender do download.

## Usar

1. Baixe um dos CSV demonstrativos na página; são dados sintéticos.
2. Inicie o laboratório e envie a base.
3. Confira o alvo: `consumo_m3_h` para regressão e `estado` para classificação.
4. Selecione o modo automático ou configure o modo manual.
5. Avalie as métricas antes de treinar o modelo final.
6. Baixe relatório, modelo e previsões antes de fechar/recarregar a aba.

O modo automático usa a última coluna como alvo por padrão e interpreta poucos valores numéricos distintos como classes. Para corrigir essa escolha, use o modo manual. Cada experimento usa um alvo. Esta versão não oferece SHAP ou múltiplas saídas.

## Executar Python local

Nesta pasta:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Abra o endereço informado pelo Streamlit, normalmente http://localhost:8501.

## Limites e diagnóstico

A primeira abertura baixa Python e pacotes via CDN; pode levar minutos. Uploads são processados na aba. A versão WebAssembly depende dos pacotes disponíveis: arquivos MAT com table/v7.3, XLS legado e balanceamento podem ter restrições. Prefira CSV ou XLSX no navegador e Python local para bases maiores.

A tela de inicialização mostra falhas de download e permite tentar novamente. Consulte também o Console do navegador. Os resultados do guia são referências externas e não garantem o desempenho de outras bases.

Referência técnica: https://github.com/whitphx/stlite/tree/v0.85.1
# Evolução científica — outubro de 2026

O modo **Científico** é a nova entrada padrão. Mantém os modos automático, manual
e uso de bundles. Inclui contrato explícito de tarefa, inspeção MAT, grupos/tempo,
PCA/KPCA, sinais com proveniência, protocolos de regressão e cenários E1–E4
inspirados no artigo, com SAE local. Consulte [metodologia](docs/METODOLOGIA.md),
[limitações](docs/LIMITACOES.md), [auditoria](docs/AUDITORIA_INICIAL.md) e
[rastreabilidade](docs/MATRIZ_RASTREABILIDADE.md).

```powershell
python -m pip install -r github-pages/requirements-dev.txt
python -m streamlit run github-pages/app.py
python -m pytest -q github-pages/tests
```

No diretório `github-pages`, CLI com contrato obrigatório:

```powershell
python experiment_cli.py dados.csv --config configs/generic_regression.yaml --target potencia --features velocidade temperatura
```

Os nomes acima são exemplos de flags, não schema dos dados do navio.
Edite YAML para grupos/tempo, unidades, modelos e parâmetros; no MAT indique
`--variable` e `--orientation`. A CLI exporta manifesto, configuração, comparação,
relatório Markdown e bundle em uma pasta por experimento. Previsões com alvo real
só são gravadas com `--export-predictions`.

SAE é NumPy com penalidade KL, não depende de PyTorch, mas está restrito ao modo
local. Sua geração em atributos tabulares difere da síntese de sinais do artigo;
`article3_reproduction` está bloqueado. No navegador, E1–E4 com SAE e SHAP
ficam indisponíveis. MAT v7.3 depende de h5py local. Não há promessa de acurácia.
O pipeline científico exportado continua ajustado apenas no treino reservado;
o teste externo é preservado. Nunca carregue joblib de origem desconhecida.
