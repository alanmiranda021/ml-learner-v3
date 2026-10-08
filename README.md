# ml-learner v3

Aplicação Streamlit em Python inspirada no Regression/Classification Learner do MATLAB.

## Correções principais

- SHAP explica as **features transformadas** pelo pipeline e usa os nomes de saída do `ColumnTransformer`/PCA.
- Savitzky-Golay é **causal/trailing**, sem olhar o futuro.
- Importação suporta CSV, Excel, Parquet, MATLAB e HDF5 (`.h5`, `.hdf5`, `.hdf`).
- Permutação usa `X_test/y_test` realmente fora do ajuste em Hold-out, Hold-out 3-way e Teste Separado.
- `.xls` é suportado com `xlrd`.
- Tuning é permitido em Hold-out/Teste Separado e CV aninhada; em K-fold/GroupKFold/TimeSeriesSplit simples o app avisa e não faz tuning ingênuo.
- Paralelismo evita oversubscription: modelos internos single-thread quando a CV/GridSearch paraleliza.
- Resultados antigos são invalidados quando a configuração muda.
- Diagnósticos caros têm cache.
- Upload múltiplo exige colunas idênticas.
- Alvo numérico é regressão por padrão.
- `warnings.filterwarnings("ignore")` global foi removido.

## Instalação

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
streamlit run app.py
```

O teste local desta entrega passou: **7 testes**.

## Estrutura

```text
ml-learner-v3/
├── app.py
├── requirements.txt
├── GUIA_ML_Learner_Python_VSCode_v3.md
├── README.md
├── CLAUDE.md
├── Dockerfile
├── .gitignore
├── .vscode/
│   └── launch.json
├── core/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocess.py
│   ├── models.py
│   ├── analysis.py
│   ├── evaluate.py
│   └── interpret.py
└── tests/
    └── test_core.py
```

## Observação metodológica

Para resultados acadêmicos, priorize **CV aninhada** quando houver seleção de hiperparâmetros. No Hold-out, o conjunto de teste é mantido fora do ajuste e é usado para a avaliação OOS e interpretação por permutação/SHAP.
