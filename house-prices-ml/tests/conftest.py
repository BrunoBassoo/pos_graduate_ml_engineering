"""Fixtures compartilhadas entre os modulos de teste."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sample" / "house_prices_sample.csv"
)


@pytest.fixture(scope="session")
def sample_processed_df() -> pd.DataFrame:
    """Amostra real (300 linhas) ja limpa, versionada no Git para uso em testes/CI
    — nao depende de credenciais do Kaggle nem do dataset completo (2.6 MB)."""
    return pd.read_csv(SAMPLE_DATA_PATH)


@pytest.fixture
def raw_house_df() -> pd.DataFrame:
    """DataFrame sintetico minimo no formato BRUTO (antes da limpeza), usado para
    testar `house_prices.data.clean.clean_data` de forma isolada e rapida."""
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "date": ["20140101T000000", "20150601T000000", "20140505T000000", "20140101T000000"],
            "price": [500000.0, 520000.0, 300000.0, 900000.0],
            "bedrooms": [3, 3, 2, 33],
            "bathrooms": [2.0, 2.0, 1.0, 1.75],
            "sqft_living": [1800, 1800, 900, 1620],
            "sqft_lot": [5000, 5000, 3000, 4000],
            "floors": [1.0, 1.0, 1.0, 1.0],
            "waterfront": ["N", "N", "N", "N"],
            "view": [0, 0, 0, 0],
            "condition": ["Average", "Average", "Good", "Average"],
            "grade": [7, 7, 6, 7],
            "sqft_above": [1800, 1800, 900, 1620],
            "sqft_basement": [0, 0, 0, 0],
            "yr_built": [1990, 1990, 1985, 1970],
            "yr_renovated": [0, 0, 0, 0],
            "zipcode": [98115, 98115, 98001, 98001],
            "lat": [47.68, 47.68, 47.30, 47.30],
            "long": [-122.29, -122.29, -122.10, -122.10],
            "sqft_living15": [1800, 1800, 1000, 1600],
            "sqft_lot15": [5000, 5000, 3000, 4000],
        }
    )
