# ML Learner — laboratório do portfólio

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
