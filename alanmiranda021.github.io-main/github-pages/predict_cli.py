"""Usar o modelo treinado FORA do app:  python predict_cli.py modelo.joblib dados_novos.xlsx saida.csv"""
import sys
import joblib
from core.io import load_table
from core.predict import predict_table

modelo, entrada, saida = sys.argv[1:4]
bundle = joblib.load(modelo)
df = load_table(entrada, open(entrada, "rb").read())
res = predict_table(bundle, df)
res.to_csv(saida, index=False)
print(f"{len(res)} previsões salvas em {saida}  (modelo: {bundle['algorithm']}, alvo: {bundle['target']})")
