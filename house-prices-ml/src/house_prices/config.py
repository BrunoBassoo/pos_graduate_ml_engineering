"""Caminhos e constantes compartilhadas por todo o projeto.

Centralizar isso aqui evita strings de caminho espalhadas pelo codigo
(e repetidas de forma inconsistente entre notebooks, scripts e testes).
"""

from pathlib import Path

# Raiz do projeto = duas pastas acima deste arquivo (src/house_prices/config.py -> raiz)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "house_prices.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "house_prices_processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

KAGGLE_DATASET_SLUG = "alyelbadry/house-pricing-dataset"

TARGET_COLUMN = "price"
RANDOM_STATE = 42
