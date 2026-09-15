"""Baixa o dataset bruto do Kaggle e o coloca em data/raw/.

Por que um script e nao so o notebook? Porque `data/raw/` fica no
.gitignore (dados brutos nao devem viver no Git), entao qualquer pessoa
que clonar o repositorio precisa de uma forma reprodutivel de obter os
dados de novo. Este script e essa forma.

Uso:
    python -m house_prices.data.download
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from house_prices.config import KAGGLE_DATASET_SLUG, RAW_DATA_DIR, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_raw_dataset() -> Path:
    """Baixa o dataset via kagglehub e copia o CSV para data/raw/house_prices.csv.

    Retorna o caminho final do arquivo.
    """
    import kagglehub

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando dataset '%s' via kagglehub...", KAGGLE_DATASET_SLUG)
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))

    csv_files = list(cache_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {cache_dir}. O dataset mudou de formato?"
        )

    source_csv = csv_files[0]
    shutil.copyfile(source_csv, RAW_DATA_PATH)
    logger.info("Dataset salvo em %s", RAW_DATA_PATH)
    return RAW_DATA_PATH


if __name__ == "__main__":
    download_raw_dataset()
