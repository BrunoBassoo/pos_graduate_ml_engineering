"""Engenharia de features e pre-processamento, encapsulados num sklearn Pipeline.

O ponto central deste modulo: TUDO que transforma os dados (features
derivadas, encoding, escala) mora dentro de um unico `sklearn.Pipeline`.
Isso e o que evita o chamado "training-serving skew" — o bug classico em
que o pre-processamento do notebook de treino diverge (sutilmente) do
pre-processamento usado na API em producao. Como o Pipeline inteiro e
serializado com `joblib` em `models/model.joblib`, a API em
`house_prices.api.app` nunca reimplementa essas contas: ela so chama
`pipeline.predict(df)` com os dados brutos (ja limpos) e confia que o
pipeline faz exatamente o que fez no treino.

Features derivadas criadas em `add_engineered_features`:
    - house_age: idade do imovel no momento da venda (sale_year - yr_built).
    - was_renovated: 1 se o imovel ja foi reformado, 0 caso contrario.
    - years_since_renovation: anos desde a ultima reforma (ou house_age,
      se nunca reformado) — combina duas colunas brutas num sinal so.
    - has_basement: 1 se o imovel tem porao (sqft_basement > 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

WATERFRONT_CATEGORIES = ["N", "Y"]
CONDITION_CATEGORIES = ["Poor", "Fair", "Average", "Good", "Very Good"]

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "view",
    "grade",
    "sqft_above",
    "sqft_basement",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "house_age",
    "years_since_renovation",
    "was_renovated",
    "has_basement",
]
ORDINAL_FEATURES = ["waterfront", "condition"]
ONEHOT_FEATURES = ["zipcode"]

# Colunas que o pipeline espera receber ANTES da engenharia de features,
# ou seja, o formato de `data/processed/house_prices_processed.csv` sem o alvo.
RAW_INPUT_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["house_age"] = df["sale_year"] - df["yr_built"]
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["yr_renovated"] > 0, df["sale_year"] - df["yr_renovated"], df["house_age"]
    )
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "ordinal",
                OrdinalEncoder(categories=[WATERFRONT_CATEGORIES, CONDITION_CATEGORIES]),
                ORDINAL_FEATURES,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                ONEHOT_FEATURES,
            ),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Monta o pipeline completo: engenharia de features -> encoding/escala -> modelo.

    `model` e qualquer estimator scikit-learn (ex.: RandomForestRegressor()).
    """
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features)),
            ("preprocessing", build_preprocessor()),
            ("model", model),
        ]
    )
