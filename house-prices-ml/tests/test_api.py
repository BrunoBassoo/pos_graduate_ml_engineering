"""Testes da API Flask usando o test client (nao sobe um servidor de verdade)."""

from __future__ import annotations

from pathlib import Path

import pytest

from house_prices.api.app import create_app
from house_prices.train import train

SAMPLE_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample" / "house_prices_sample.csv"
)

VALID_PAYLOAD = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": "N",
    "view": 0,
    "condition": "Average",
    "grade": 7,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": 98115,
    "lat": 47.68,
    "long": -122.29,
    "sqft_living15": 1800,
    "sqft_lot15": 5000,
    "sale_year": 2015,
}


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Treina um modelo pequeno uma unica vez para toda a sessao de testes,
    evitando retreinar a cada teste de API."""
    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"

    train(
        data_path=SAMPLE_DATA_PATH,
        n_estimators=15,
        max_depth=6,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    return model_path


@pytest.fixture
def client(trained_model_path):
    app = create_app(model_path=trained_model_path)
    app.testing = True
    return app.test_client()


@pytest.fixture
def client_without_model(tmp_path):
    app = create_app(model_path=tmp_path / "nao_existe.joblib")
    app.testing = True
    return app.test_client()


def test_health_returns_ok_when_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.get("/health")

    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False


def test_predict_returns_positive_price_for_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.get_json()
    assert "predicted_price" in body
    assert body["predicted_price"] > 0


def test_predict_returns_400_for_missing_fields(client):
    incomplete_payload = {"bedrooms": 3}

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_returns_400_for_invalid_category(client):
    invalid_payload = {**VALID_PAYLOAD, "condition": "Excellent"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400


def test_predict_returns_400_for_non_json_body(client):
    response = client.post("/predict", data="isso nao e json")

    assert response.status_code == 400


def test_predict_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503
