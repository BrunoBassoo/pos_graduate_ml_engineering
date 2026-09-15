"""Teste de integracao do pipeline de treino, usando a amostra pequena
(data/sample/) para rodar rapido em CI, sem depender do dataset completo."""

from __future__ import annotations

import json
from pathlib import Path

from house_prices.train import train

SAMPLE_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample" / "house_prices_sample.csv"
)


def test_train_produces_model_and_reasonable_metrics(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    pipeline, metrics = train(
        data_path=SAMPLE_DATA_PATH,
        n_estimators=20,
        max_depth=6,
        test_size=0.3,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics == metrics

    # R2 pode ser instavel numa amostra de 300 linhas, mas o modelo deve
    # aprender algo melhor do que "chutar a media" (R2 > 0).
    assert metrics["r2"] > 0
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
