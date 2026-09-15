"""Carrega o pipeline treinado e faz predicoes a partir de dados brutos (ja limpos).

Este modulo e o unico ponto de contato entre "modelo serializado em disco"
e "codigo Python que quer uma predicao" — tanto a API Flask quanto os
testes usam `load_model` + `predict_price` em vez de mexer com joblib
diretamente.
"""

from __future__ import annotations

import functools

import joblib
import pandas as pd

from house_prices.config import MODEL_PATH
from house_prices.features import RAW_INPUT_COLUMNS


@functools.lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    """Carrega o pipeline serializado. Cacheado: o disco so e lido uma vez por processo."""
    return joblib.load(model_path)


def predict_price(payload: dict, model=None) -> float:
    """Recebe um dict com as colunas de `RAW_INPUT_COLUMNS` e retorna o preco previsto."""
    if model is None:
        model = load_model()

    row = {col: payload[col] for col in RAW_INPUT_COLUMNS}
    input_df = pd.DataFrame([row])
    prediction = model.predict(input_df)[0]
    return float(prediction)
