import scipy.io as sio
import pandas as pd
import numpy as np
import h5py

path = r'C:\Users\Alan\Documents\Projeto Analise\alanmiranda021.github.io-main\arquivos\Consumo de Comb_Navio\Dados_navio_nig.mat'
try:
    m = sio.loadmat(path)
    print("Scipy loadmat success")
    for k, v in m.items():
        if not k.startswith('__'):
            print(k, type(v), getattr(v, 'shape', 'no shape'), getattr(v, 'dtype', 'no dtype'))
except Exception as e:
    print("Scipy failed:", e)
    with h5py.File(path, 'r') as h:
        print("H5py success")
        for k, v in h.items():
            print(k, type(v), getattr(v, 'shape', 'no shape'), getattr(v, 'dtype', 'no dtype'))
