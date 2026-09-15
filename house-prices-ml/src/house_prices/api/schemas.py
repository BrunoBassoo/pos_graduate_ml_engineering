"""Validacao do payload recebido em POST /predict.

Isso implementa, de forma simples e sem dependencias extras, o mesmo
papel que a `signature` do MLflow (mencionada no material da aula 8)
cumpre em setups mais avancados: um "contrato de interface" que rejeita
de forma explicita requisicoes com esquema incorreto, em vez de deixar o
erro estourar dentro do sklearn com uma mensagem confusa.
"""

from __future__ import annotations

from house_prices.features import CONDITION_CATEGORIES, RAW_INPUT_COLUMNS, WATERFRONT_CATEGORIES

# tipo esperado para cada campo do payload
_FIELD_TYPES: dict[str, type] = {
    "bedrooms": int,
    "bathrooms": (int, float),
    "sqft_living": (int, float),
    "sqft_lot": (int, float),
    "floors": (int, float),
    "waterfront": str,
    "view": int,
    "condition": str,
    "grade": int,
    "sqft_above": (int, float),
    "sqft_basement": (int, float),
    "yr_built": int,
    "yr_renovated": int,
    "zipcode": int,
    "lat": (int, float),
    "long": (int, float),
    "sqft_living15": (int, float),
    "sqft_lot15": (int, float),
    "sale_year": int,
}


class ValidationError(ValueError):
    """Levantado quando o payload de /predict nao segue o esquema esperado."""


def validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValidationError("O corpo da requisicao deve ser um objeto JSON.")

    missing = [col for col in RAW_INPUT_COLUMNS if col not in payload]
    if missing:
        raise ValidationError(f"Campos obrigatorios ausentes: {missing}")

    for field, expected_type in _FIELD_TYPES.items():
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            got = type(value).__name__
            raise ValidationError(f"Campo '{field}' deve ser {expected_type}, recebido {got}.")

    if payload["waterfront"] not in WATERFRONT_CATEGORIES:
        raise ValidationError(f"Campo 'waterfront' deve ser um de {WATERFRONT_CATEGORIES}.")

    if payload["condition"] not in CONDITION_CATEGORIES:
        raise ValidationError(f"Campo 'condition' deve ser um de {CONDITION_CATEGORIES}.")
