"""Limpeza dos dados brutos (data/raw -> data/processed).

Aqui ficam apenas transformacoes "estruturais", que so fazem sentido
rodar UMA VEZ sobre o dataset inteiro (deduplicar, remover linhas
invalidas, converter tipos). Isso e diferente de "feature engineering",
que mora em `house_prices.features` dentro de um sklearn Pipeline —
porque feature engineering precisa rodar de forma IDENTICA no treino e
na hora de servir uma predicao (senao temos training-serving skew).

Decisoes de limpeza documentadas (o "porque", nao so o "o que"):

1. Duplicatas de `id`: o dataset registra CADA VENDA de um imovel, entao
   o mesmo imovel pode aparecer mais de uma vez (177 casos aqui). Se nao
   deduplicarmos, o mesmo imovel pode cair no treino E no teste ao mesmo
   tempo -> vazamento de dados (data leakage) que infla artificialmente
   a metrica de avaliacao. Mantemos apenas a venda mais recente por id.

2. `bedrooms == 33`: uma unica linha com 33 quartos, 1.75 banheiros e
   apenas 1620 sqft de area util. Fisicamente incoerente (33 quartos
   nesse espaco) — e um erro de digitacao conhecido nesse dataset
   (provavelmente deveria ser 3). Removemos outliers de `bedrooms` acima
   de um limite razoavel em vez de "consertar" o valor, pois nao ha como
   saber o valor correto.

3. Coluna `date`: vem como string `AAAAMMDDT000000`. Convertida para
   datetime e usada para derivar `sale_year` (necessario para calcular a
   idade do imovel na venda). A string original e descartada.

4. Coluna `id`: e apenas um identificador de listagem, sem poder
   preditivo. Usada só durante a deduplicacao e descartada em seguida.
"""

from __future__ import annotations

import logging

import pandas as pd

from house_prices.config import PROCESSED_DATA_PATH, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_PLAUSIBLE_BEDROOMS = 15


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as regras de limpeza descritas no modulo e retorna um df novo."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["sale_year"] = df["date"].dt.year

    n_before = len(df)
    df = df.sort_values("date").drop_duplicates(subset="id", keep="last")
    logger.info("Deduplicacao por id: %d -> %d linhas", n_before, len(df))

    n_before = len(df)
    df = df[df["bedrooms"] <= MAX_PLAUSIBLE_BEDROOMS]
    logger.info(
        "Remocao de outliers de bedrooms (> %d): %d -> %d linhas",
        MAX_PLAUSIBLE_BEDROOMS,
        n_before,
        len(df),
    )

    df = df.drop(columns=["id", "date"])
    df = df.reset_index(drop=True)
    return df


def run(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    df_raw = load_raw_data(input_path)
    df_clean = clean_data(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info("Dados limpos salvos em %s (%d linhas)", output_path, len(df_clean))
    return df_clean


if __name__ == "__main__":
    run()
