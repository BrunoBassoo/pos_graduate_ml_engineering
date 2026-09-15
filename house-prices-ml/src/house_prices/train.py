"""Treina o modelo de previsao de preco e salva pipeline + metricas.

Uso:
    python -m house_prices.train
    python -m house_prices.train --n-estimators 300 --max-depth 12
    python -m house_prices.train --data data/sample/house_prices_sample.csv

Por que RandomForestRegressor? E um bom baseline "sem drama" para dados
tabulares: nao exige escolher forma funcional (como uma regressao linear
exigiria), lida bem com a mistura de features numericas e categoricas
que ja preparamos em `features.py`, e da pra comparar a importancia das
features depois. Trocar de modelo e so trocar o estimator passado para
`build_pipeline` — o resto do pipeline (engenharia de features + encoding)
continua igual, essa e a vantagem de ter isso desacoplado.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from house_prices.config import (
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from house_prices.features import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def evaluate(pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "n_test_samples": int(len(y_test)),
    }


def train(
    data_path=PROCESSED_DATA_PATH,
    n_estimators: int = 200,
    max_depth: int | None = 12,
    test_size: float = 0.2,
    model_path=MODEL_PATH,
    metrics_path=METRICS_PATH,
):
    df = load_processed_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = build_pipeline(model)

    logger.info(
        "Treinando RandomForestRegressor (n_estimators=%d, max_depth=%s)...",
        n_estimators,
        max_depth,
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    logger.info("Metricas no conjunto de teste: %s", metrics)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Modelo salvo em %s", model_path)
    logger.info("Metricas salvas em %s", metrics_path)

    return pipeline, metrics


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina o modelo de preco de imoveis.")
    parser.add_argument("--data", type=str, default=str(PROCESSED_DATA_PATH))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
    )
