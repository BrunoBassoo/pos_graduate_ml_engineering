"""
Aula 06 - CI/CD e Automacao de Pipelines
=========================================

Exemplo pratico que simula, localmente e sem depender de GitHub Actions nem
de um servidor MLflow, as mesmas etapas descritas no workflow
`ci_ml_pipeline.yml` do resumo da aula:

    1) Validacao automatizada de dados ("testes unitarios para dados")
    2) Testes automatizados de codigo (sanity check do treino)
    3) Treino do modelo, com parametros e metricas registrados
    4) Comparacao do novo modelo com o modelo em "producao"
       (logica equivalente a `deve_promover_modelo` de src/register.py)
    5) Promocao (ou rejeicao) do novo modelo, atualizando um
       "model registry" local em JSON

Cada etapa imprime um cabecalho como se fosse um step do GitHub Actions e,
se falhar, interrompe o pipeline com um codigo de saida != 0 -- assim como
o job do Actions marcaria o run como falho e pararia antes do proximo step.

Uso:
    python exemplo_pipeline_cicd.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
METRICA_PRINCIPAL = "accuracy"
REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
NOME_MODELO = "heart-disease-model-demo"


def log_step(titulo: str) -> None:
    print(f"\n::step:: {titulo}")


def carregar_dados() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.rename(columns={"target": "target"})
    return df


def validar_dados(df: pd.DataFrame) -> None:
    """Equivalente a tests/test_data_validation.py do resumo: atua como
    "teste unitario" para os dados antes de qualquer treino."""
    assert not df.isnull().any().any(), "Encontrados valores nulos no dataset"
    assert set(df["target"].unique()).issubset({0, 1}), "Target deve ser binario"
    assert len(df) > 0, "Dataset vazio"
    print(f"Dados validados: {len(df)} linhas, {df.shape[1] - 1} features, sem nulos.")


def rodar_testes_unitarios() -> None:
    """Equivalente a tests/test_train.py: garante que o pipeline de treino
    funciona antes de gastar tempo/recursos treinando "de verdade"."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )
    modelo_smoke = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    modelo_smoke.fit(X_train, y_train)
    preds = modelo_smoke.predict(X_test)

    assert len(preds) == len(y_test), "Numero de predicoes diverge do esperado"
    assert not np.isnan(preds).any(), "Predicoes contem NaN"
    print("Testes unitarios de codigo passaram (smoke test do treino).")


def treinar_modelo(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    """Equivalente a src/train.py: treina o modelo e registra parametros e
    metricas (no lugar do MLflow Tracking, guardamos em um dict local)."""
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )

    params = {"n_estimators": 200, "max_depth": 6, "random_state": RANDOM_STATE}
    modelo = RandomForestClassifier(**params)
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    metricas = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
    }

    commit_hash = os.environ.get("GITHUB_SHA", "local-dev")
    print(f"Modelo treinado | commit={commit_hash} | params={params}")
    print(f"Metricas do novo modelo: {metricas}")
    return modelo, metricas


def carregar_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def deve_promover_modelo(nova_metrica: float, metrica_producao: float | None) -> bool:
    """Mesma logica de src/register.py do resumo: so promove se o novo
    modelo igualar ou superar o modelo atualmente em producao."""
    if metrica_producao is None:
        return True
    return nova_metrica >= metrica_producao


def registrar_modelo(metricas: dict, versao_anterior: int) -> None:
    registry = carregar_registry()
    nova_versao = versao_anterior + 1
    registry[NOME_MODELO] = {
        "stage": "Staging",
        "version": nova_versao,
        "metrics": metricas,
    }
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelo promovido a Staging: versao {nova_versao} (registry: {REGISTRY_PATH.name})")


def main() -> int:
    log_step("Checar codigo (checkout)")
    print("Repositorio ja disponivel localmente (equivalente a actions/checkout@v3).")

    log_step("Instalar dependencias")
    print("Dependencias ja instaladas no ambiente atual (equivalente a pip install -r requirements.txt).")

    log_step("Validar dados (testes unitarios para dados)")
    df = carregar_dados()
    try:
        validar_dados(df)
    except AssertionError as erro:
        print(f"FALHA na validacao de dados: {erro}")
        return 1

    log_step("Executar testes automatizados (pytest)")
    try:
        rodar_testes_unitarios()
    except AssertionError as erro:
        print(f"FALHA nos testes: {erro}")
        return 1

    log_step("Treinar modelo e logar metricas")
    modelo, metricas = treinar_modelo(df)

    log_step("Comparar com o modelo em producao e decidir promocao")
    registry = carregar_registry()
    info_atual = registry.get(NOME_MODELO, {})
    metrica_producao = info_atual.get("metrics", {}).get(METRICA_PRINCIPAL)
    versao_anterior = info_atual.get("version", 0)

    if deve_promover_modelo(metricas[METRICA_PRINCIPAL], metrica_producao):
        registrar_modelo(metricas, versao_anterior)
    else:
        print(
            f"Modelo novo ({metricas[METRICA_PRINCIPAL]:.4f}) nao superou o "
            f"modelo em producao ({metrica_producao:.4f}). Promocao abortada."
        )

    print("\nPipeline concluido com sucesso (equivalente a run verde no GitHub Actions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
