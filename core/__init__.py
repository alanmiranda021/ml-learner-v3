"""Núcleo do ml-learner: seed global, versão e utilidades comuns."""
from __future__ import annotations
import os
import random
import numpy as np

__version__ = "2.0.0"


def set_global_seed(seed: int = 42) -> int:
    """Fixa sementes em numpy/random/python para reprodutibilidade.

    Observação: nem todos os algoritmos do sklearn aceitam `random_state`
    (ex.: alguns otimizadores do scipy). Ainda assim, fixar aqui ajuda.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    return int(seed)
