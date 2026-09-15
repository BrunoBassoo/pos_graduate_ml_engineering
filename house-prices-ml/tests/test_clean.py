"""Testes de dados: garantem que as regras de limpeza descritas em
`house_prices.data.clean` continuam valendo. Sao "testes de dados" no
sentido da aula 8 — protegem contra regressao na qualidade dos dados,
nao so em bugs de codigo."""

from __future__ import annotations

from house_prices.data.clean import clean_data


def test_clean_data_drops_duplicate_ids_keeping_latest_sale(raw_house_df):
    result = clean_data(raw_house_df)

    # o id=1 aparecia duas vezes (duas vendas do mesmo imovel); so a mais
    # recente (price=520000, em 2015) deve sobreviver. id=3 e o outlier de
    # bedrooms (33 quartos) e e removido por uma regra separada.
    assert len(result) == 2
    assert 520000.0 in result["price"].values
    assert 500000.0 not in result["price"].values


def test_clean_data_removes_implausible_bedroom_outliers(raw_house_df):
    result = clean_data(raw_house_df)

    assert result["bedrooms"].max() <= 15
    assert 33 not in result["bedrooms"].values


def test_clean_data_drops_id_and_raw_date_columns(raw_house_df):
    result = clean_data(raw_house_df)

    assert "id" not in result.columns
    assert "date" not in result.columns
    assert "sale_year" in result.columns


def test_clean_data_has_no_missing_values(raw_house_df):
    result = clean_data(raw_house_df)

    assert result.isnull().sum().sum() == 0


def test_sample_dataset_is_already_clean(sample_processed_df):
    """A amostra versionada no Git deve refletir o resultado de clean_data
    (sem ids, sem outliers de bedrooms, sem nulos) — se este teste falhar,
    a amostra esta desatualizada em relacao as regras de limpeza."""
    assert "id" not in sample_processed_df.columns
    assert sample_processed_df["bedrooms"].max() <= 15
    assert sample_processed_df.isnull().sum().sum() == 0
