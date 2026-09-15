"""Testes da engenharia de features e do pipeline de pre-processamento."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from house_prices.features import add_engineered_features, build_pipeline


def _house(sale_year=2015, yr_built=1990, yr_renovated=0, sqft_basement=0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sale_year": [sale_year],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "sqft_basement": [sqft_basement],
        }
    )


def test_add_engineered_features_computes_house_age():
    df = _house(sale_year=2015, yr_built=1990)

    result = add_engineered_features(df)

    assert result.loc[0, "house_age"] == 25
    assert result.loc[0, "was_renovated"] == 0
    assert result.loc[0, "years_since_renovation"] == 25
    assert result.loc[0, "has_basement"] == 0


def test_add_engineered_features_uses_renovation_year_when_present():
    df = _house(sale_year=2015, yr_built=1990, yr_renovated=2010, sqft_basement=400)

    result = add_engineered_features(df)

    assert result.loc[0, "was_renovated"] == 1
    assert result.loc[0, "years_since_renovation"] == 5
    assert result.loc[0, "has_basement"] == 1


def test_add_engineered_features_does_not_mutate_input():
    df = _house()

    add_engineered_features(df)

    assert "house_age" not in df.columns


def test_pipeline_fits_and_predicts_on_sample_data(sample_processed_df):
    X = sample_processed_df.drop(columns=["price"])
    y = sample_processed_df["price"]

    pipeline = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=0))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert (predictions > 0).all()
