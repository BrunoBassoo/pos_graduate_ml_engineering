# Aula 08 — Projeto Final: Ciclo Completo de ML

## Visão Geral

Esta aula encerra a Fase 1 consolidando todas as etapas estudadas anteriormente — entendimento de negócio, engenharia de features, treinamento, deploy, CI/CD, monitoramento e governança — em um único fluxo coeso e integrado. O objetivo é transformar um modelo de notebook (artefato estático e frágil) em um serviço de produção vivo, auditável, escalável e resiliente a falhas. A aula propõe o Projeto Final, uma simulação de ambiente corporativo de alta performance na qual o(a) aluno(a) assume o papel de MLOps Engineer, responsável tanto por dialogar com stakeholders sobre KPIs de negócio quanto por implementar a infraestrutura de código que sustenta o ciclo de vida completo do modelo.

## Tópicos Abordados

- **O que vem por aí?** — Introdução ao Projeto Final como consolidação de todas as aulas anteriores em um fluxo único.
- **Hands On** — Cenário prático: sistema preditivo de doenças cardíacas (Heart Disease, dataset UCI) para a instituição fictícia "FIAP HealthCare Plus", com requisitos de governança, performance técnica e excelência operacional.
- **Saiba Mais**, dividido em seis fases práticas:
  - Fase 1: Estruturação e Versionamento
  - Fase 2: Pipeline de Treinamento com Rastreabilidade
  - Fase 3: Governança e Auditoria de Viés
  - Fase 4: Automação via CI/CD (GitHub Actions)
  - Fase 5: Implantação e Servindo (Deployment)
  - Fase 6: Monitoramento e Fechamento do Ciclo
  - E ainda, dentro do "Saiba Mais": Business Understanding e o ciclo circular e interdependente; Engenharia de Features Avançada e Seleção Estratégica; Automação e Confiabilidade (CI/CD/CT); Governança, Ética e o Ciclo de Auditoria; Monitoramento Proativo e Gestão de Drift.
- **Mercado, Cases e Tendências** — Crescimento da adoção de IA, demanda por profissionais de MLOps, cases em saúde, varejo/mídia (Netflix, Spotify), setor financeiro (detecção de fraude), manutenção preditiva e gestão de risco de crédito; tendências de responsible AI e regulamentação.

## Conceitos-Chave

Esta aula não traz conceitos isolados novos, mas amarra os pilares das aulas 1 a 7 em um sistema único. Cada conceito abaixo é uma "engrenagem" do ciclo completo.

### MLOps como disciplina integradora
MLOps (Machine Learning Operations) une o desenvolvimento de sistemas de ML (Dev) com a operação desses sistemas (Ops), visando padronização, automação e confiabilidade. A aula reforça que o ciclo de vida de ML não deve ser tratado como silos isolados (dados, modelo, deploy, monitoramento), mas como um sistema circular e interdependente: a monitoração valida o negócio, que valida a utilidade do modelo, que depende da qualidade dos dados.

### Business Understanding como fundação de todo o ciclo
O entendimento do negócio não é uma etapa burocrática — as decisões tomadas aqui (ex.: o KPI de reduzir a taxa de readmissão hospitalar em 10%) definem a variável alvo, os limiares de alerta no monitoramento pós-implantação e até a arquitetura de deployment. Se o negócio prioriza minimizar falsos negativos (maximizar Recall), o monitoramento deve focar em drift dessa métrica específica. Da mesma forma, requisitos de tempo real (ex.: detecção de fraude, triagem médica) tornam inviável uma arquitetura batch, independentemente da acurácia do modelo.

### Estruturação e versionamento de projeto
Um notebook monolítico é insustentável em produção. A separação em `src/`, `data/` (raw e processed), `models/`, `tests/` e `.github/workflows/` (inspirada no Cookiecutter Data Science) viabiliza revisão de código, testes e colaboração. Além do Git para código, dados e modelos também precisam de versionamento (via DVC ou MLflow), garantindo rastreabilidade e capacidade de reverter tanto código quanto dados para reproduzir falhas em produção.

### Pipelines reprodutíveis e prevenção de training-serving skew
Encapsular transformações (imputação, normalização, features derivadas como `age_squared` ou `cholesterol_to_age`) em um `sklearn.pipeline.Pipeline` garante que o pré-processamento do treino seja idêntico ao da inferência, evitando o training-serving skew e o vazamento de dados (data leakage) — os parâmetros estatísticos (média, desvio padrão) aprendidos no treino devem ser aplicados sem recálculo em teste e produção.

### Rastreabilidade de experimentos com MLflow
O MLflow deve registrar automaticamente: parâmetros (hiperparâmetros do modelo), métricas (Acurácia, F1-Score, Recall, Precision — com Recall priorizado em contextos de saúde para evitar falsos negativos), artefatos (modelo serializado e script de treino) e a assinatura do modelo (via `infer_signature`), que atua como um contrato de interface rejeitando requisições com esquema incorreto.

### Engenharia e seleção de features no contexto de produção
Feature Engineering (transformações não lineares, features calculadas, binning) continua sendo o fator diferenciador de performance. Mas em produção, cada feature tem um custo computacional (latência de inferência) e um custo de manutenção (mais uma variável a monitorar quanto a drift). Métodos de seleção Filter (ANOVA, Qui-quadrado), Wrapper (RFE) e Embedded (importância de features do Random Forest) ajudam a manter o modelo enxuto.

### Governança como código (Governance as Code)
A governança é um requisito, não uma etapa opcional. Com o Fairlearn, define-se atributos sensíveis (idade, sexo) e calcula-se disparidades via `MetricFrame` (ex.: Taxa de Falso Negativo por grupo). Se a disparidade for inaceitável, o script deve falhar automaticamente e impedir o registro do modelo — implementando fairness diretamente no pipeline automatizado. Técnicas de mitigação podem atuar no pré-processamento (rebalanceamento), treinamento (restrições de otimização) ou pós-processamento (ajuste de limiares por grupo).

### Model Cards, explicabilidade e conformidade regulatória
Model Cards funcionam como uma "bula" do modelo, documentando uso pretendido, limitações, composição dos dados de treino e métricas desagregadas por grupo. Regulações como a LGPD (Brasil) e o AI Act (Europa) exigem explicabilidade e accountability, o que demanda data lineage completo — rastrear exatamente quais dados treinaram a versão do modelo que tomou uma decisão específica.

### CI/CD/CT aplicado a Machine Learning
- **Continuous Integration (CI)**: testa código, dados e pipeline de treinamento (linting, testes unitários e de integração, incluindo casos de dados inesperados).
- **Continuous Delivery (CD)**: empacota o modelo validado (Docker, Model Registry) para Staging/Produção com mínima intervenção manual.
- **Continuous Training (CT)**: automatiza o retreinamento do modelo, acionado por gatilho de desempenho ou por novos dados — diferente de software tradicional, o modelo de ML degrada naturalmente com o data drift, então o CT fecha o ciclo de feedback.

### Estratégias de deployment
A conteinerização com Docker resolve o problema de "funciona na minha máquina", empacotando SO base, dependências (`requirements.txt`) e código da aplicação. A API (Flask/FastAPI) deve expor `/predict` (POST, retorna probabilidade) e `/health` (GET, para monitoramento de disponibilidade). A escolha entre inferência online (API REST) e processamento em batch (scoring em lote) é uma decisão de negócio, não puramente técnica.

### Monitoramento proativo em três frentes
1. **Monitoramento de Serviço (Operacional)**: latência, throughput, taxa de erros HTTP.
2. **Monitoramento de Data Drift**: mudanças na distribuição estatística dos dados de entrada (covariate shift), detectáveis mesmo sem o rótulo (ground truth) ainda disponível.
3. **Monitoramento de Performance (Model Drift)**: acompanha a qualidade da predição quando o rótulo real chega, o que muitas vezes é atrasado (ex.: fraude só é confirmada dias depois) — nesse caso, o data drift funciona como proxy de alerta precoce.

A estratégia de manutenção pode ser retreinamento agendado (simples, porém potencialmente tardio ou desnecessário) ou baseado em gatilhos/triggers (mais eficiente, acionado apenas quando há degradação estatisticamente significativa).

## Exercício Hands-On (do material)

**Cenário:** a instituição de saúde fictícia "FIAP HealthCare Plus" validou o MVP de um modelo de detecção de doenças cardíacas (dataset Heart Disease da UCI) e decidiu colocá-lo em produção para auxiliar médicos(as) em tempo real durante a triagem de pacientes. A diretoria impôs três requisitos estritos:

- **Governança**: o modelo não pode discriminar pacientes por região ou idade.
- **Performance técnica**: a API deve ter baixa latência para não travar o sistema de prontuário.
- **Excelência operacional**: o modelo deve ser retreinado automaticamente se a performance cair abaixo de um limiar aceitável.

A missão do Projeto Final é construir a arquitetura completa que sustenta esse modelo — não mais um arquivo estático, mas um serviço gerenciado, versionado e monitorado — integrando Git, MLflow, Fairlearn, Docker e GitHub Actions, através das seis fases descritas no material:

1. **Estruturação e Versionamento**: organizar o projeto em `src/`, `data/raw` e `data/processed`, `models/`, `tests/` e `.github/workflows/`; versionar dados e modelos com DVC/MLflow.
2. **Pipeline de Treinamento com Rastreabilidade**: refatorar o treino em um `train.py` parametrizável, com features (`age_squared`, `cholesterol_to_age`) dentro de um `sklearn.pipeline.Pipeline`, registrando parâmetros, métricas, artefatos e assinatura do modelo no MLflow.
3. **Governança e Auditoria de Viés**: usar Fairlearn para auditar fairness por faixa etária/sexo com `MetricFrame`, falhando automaticamente o script se houver disparidade inaceitável (ex.: diagnosticar pior mulheres do que homens), e gerar um Model Card com uso pretendido e limitações (ex.: não validado para uso pediátrico).
4. **Automação via CI/CD (GitHub Actions)**: criar `.github/workflows/ml_pipeline.yml` acionado a cada push na `main` ou abertura de PR, com três jobs: (1) Quality Assurance (linting + testes unitários, incluindo testes de dados); (2) Training & Evaluation (treino em ambiente efêmero com gatekeeping — falha se acurácia < 80% do baseline do MVP ou se houver violação de fairness); (3) Model Registration (registro no MLflow Model Registry com tag "Staging" apenas se todos os critérios forem atendidos).
5. **Implantação e Servindo**: criar um Dockerfile e expor o modelo via Flask ou FastAPI com endpoints `/predict` (POST, JSON com dados do paciente, retorna probabilidade) e `/health` (GET); testar localmente subindo o container e simulando requisições `curl` para validar resposta e latência.
6. **Monitoramento e Fechamento do Ciclo**: gerar logs estruturados em JSON (timestamp, input_features, prediction, model_version); criar um script `simulate_drift.py` que envia dados artificialmente modificados (ex.: aumentar a média de idade em 10 anos) e implementar verificação estatística (ex.: teste de Kolmogorov-Smirnov) comparando a distribuição recebida com a de referência (dados de treino); se drift for detectado, emitir alerta ou acionar automaticamente o pipeline de CI/CD para retreinar o modelo, fechando o ciclo de aprendizado contínuo.

O material deixa explícito que o objetivo não é a profundidade em cada tecnologia específica, mas a compreensão dos conceitos fundamentais que integram e norteiam todo o ciclo de vida de desenvolvimento de modelos de ML — sendo permitido adaptar o exercício para outros cenários ou ferramentas (GitLab, Kubeflow etc.).

## Exemplos de Código

Os exemplos abaixo operacionalizam os conceitos apresentados no PDF e seguem a mesma estrutura das seis fases do Hands On (heart disease). Não constam no PDF original — foram elaborados para ilustrar, na prática, o ciclo completo descrito no material.

Estrutura de diretórios ilustrando a Fase 1 (Estruturação e Versionamento), inspirada no Cookiecutter Data Science mencionado no PDF:

```
heart-disease-mlops/
├── data/
│   ├── raw/                  # dados brutos imutáveis
│   └── processed/            # dados transformados
├── src/
│   ├── preprocess.py         # limpeza e engenharia de features
│   ├── train.py               # treinamento parametrizável
│   ├── evaluate.py            # avaliação de métricas e fairness
│   └── serve/
│       ├── app.py              # API FastAPI (/predict, /health)
│       └── simulate_drift.py   # simulação de drift para testes
├── models/                    # artefatos serializados temporários
├── tests/                     # testes unitários e de integração
├── .github/workflows/
│   └── ml_pipeline.yml        # pipeline de CI/CD/CT
├── Dockerfile
├── requirements.txt
└── dvc.yaml                    # versionamento de dados/modelos
```

Pipeline de treino com `sklearn.pipeline.Pipeline` e rastreamento via MLflow, ilustrando a Fase 2 (evita training-serving skew ao encapsular pré-processamento e modelo em um único objeto):

```python
# src/train.py
import argparse
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_squared"] = df["age"] ** 2
    df["cholesterol_to_age"] = df["chol"] / df["age"]
    return df


def main(n_estimators: int, max_depth: int):
    df = pd.read_csv("data/processed/heart_disease.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )),
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Parâmetros (Fase 2)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Métricas de modelo e de negócio — Recall priorizado em saúde
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))

        # Assinatura do modelo: contrato de interface para produção
        signature = infer_signature(X_train, preds)
        mlflow.sklearn.log_model(
            pipeline, "model", signature=signature, registered_model_name=None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
```

Auditoria de fairness com Fairlearn, ilustrando a Fase 3 (Governança como Código — o script falha e impede o registro do modelo se houver disparidade inaceitável entre grupos):

```python
# src/evaluate.py (trecho de auditoria de fairness)
from fairlearn.metrics import MetricFrame, false_negative_rate
import sys

def audit_fairness(y_true, y_pred, sensitive_feature, max_disparity=0.1):
    metric_frame = MetricFrame(
        metrics=false_negative_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )
    disparity = metric_frame.difference(method="between_groups")
    print(f"Taxa de Falso Negativo por grupo:\n{metric_frame.by_group}")
    print(f"Disparidade (max - min): {disparity:.4f}")

    if disparity > max_disparity:
        print("FALHA: disparidade de fairness acima do limiar aceitável.")
        sys.exit(1)  # interrompe o pipeline de CI/CD — Governance as Code

    print("OK: modelo aprovado na auditoria de fairness.")
```

Pipeline de CI/CD/CT no GitHub Actions, ilustrando a Fase 4 com os três jobs descritos no material (Quality Assurance, Training & Evaluation com gatekeeping, e Model Registration):

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline - Heart Disease

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality-assurance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Lint
        run: flake8 src/
      - name: Testes unitários (código e dados)
        run: pytest tests/ -v

  training-and-evaluation:
    needs: quality-assurance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Treinar modelo
        run: python src/train.py --n_estimators 200 --max_depth 8
      - name: Gatekeeping (acurácia mínima e fairness)
        run: python src/evaluate.py --min-accuracy 0.80 --max-fairness-disparity 0.10

  model-registration:
    needs: training-and-evaluation
    runs-on: ubuntu-latest
    steps:
      - name: Registrar modelo no MLflow Model Registry (tag Staging)
        run: python src/register_model.py --stage Staging
```

Dockerfile e API de serving, ilustrando a Fase 5 (elimina o problema "funciona na minha máquina" e expõe os endpoints `/predict` e `/health`):

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# src/serve/app.py
import json
import time
import logging
from datetime import datetime, timezone

import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Risk API")
model = mlflow.pyfunc.load_model("models:/heart_disease_model/Staging")
MODEL_VERSION = "v1.3.0"

logger = logging.getLogger("prediction_logs")
logging.basicConfig(level=logging.INFO)


class PatientData(BaseModel):
    age: float
    chol: float
    trestbps: float
    thalach: float
    # ... demais features do dataset UCI Heart Disease


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientData):
    input_df = patient.model_dump()
    prediction = model.predict([input_df])[0]

    # Log estruturado — matéria-prima para detecção de drift (Fase 6)
    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_features": input_df,
        "prediction": float(prediction),
        "model_version": MODEL_VERSION,
    }))

    return {"disease_probability": float(prediction), "model_version": MODEL_VERSION}
```

Script de simulação e detecção de drift, ilustrando a Fase 6 (fechamento do ciclo com monitoramento proativo via teste estatístico de Kolmogorov-Smirnov):

```python
# src/serve/simulate_drift.py
import requests
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd

API_URL = "http://localhost:8000/predict"


def send_synthetic_patients(n=100, age_shift=10):
    """Simula pacientes com idade artificialmente aumentada (covariate shift)."""
    reference_ages = np.random.normal(loc=54, scale=9, size=n)
    drifted_ages = reference_ages + age_shift

    for age in drifted_ages:
        payload = {"age": age, "chol": 240, "trestbps": 130, "thalach": 150}
        requests.post(API_URL, json=payload)

    return reference_ages, drifted_ages


def detect_drift(reference: np.ndarray, incoming: np.ndarray, alpha: float = 0.05):
    """Teste de Kolmogorov-Smirnov comparando distribuição de referência vs. produção."""
    statistic, p_value = ks_2samp(reference, incoming)
    drift_detected = p_value < alpha

    if drift_detected:
        print(f"ALERTA: data drift detectado (p-value={p_value:.4f}).")
        # Em maturidade mais alta: acionar o pipeline de CT (retreino automático)
    else:
        print(f"Sem drift significativo (p-value={p_value:.4f}).")

    return drift_detected


if __name__ == "__main__":
    train_ages = pd.read_csv("data/processed/heart_disease.csv")["age"].values
    _, drifted_ages = send_synthetic_patients(age_shift=10)
    detect_drift(train_ages, drifted_ages)
```

## Cases e Tendências de Mercado

- 78% das organizações já relatam usar IA em 2024, contra 55% em 2023, evidenciando crescimento exponencial do mercado de IA e ML.
- Crescimento da demanda por profissionais que dominem todo o pipeline de ML — não só algoritmos —, incluindo data scientists, ML/MLOps engineers e especialistas em governança de dados.
- Saúde: modelos de diagnóstico por imagens e dispositivos médicos com IA cada vez mais presentes.
- Varejo e mídia: sistemas de recomendação personalizados (Netflix, Spotify) dependem de pipelines de MLOps para atualizar modelos conforme mudam dados e preferências dos usuários.
- Setor financeiro: bancos e empresas de pagamento usam MLOps para detecção de fraudes adaptativa, monitorando transações em tempo real e atualizando modelos conforme surgem novas técnicas de fraude.
- Outras aplicações comuns: manutenção preditiva na indústria (dados de sensores para antecipar falhas) e gestão de risco de crédito.
- Tendências: consolidação de práticas de MLOps, maior adoção de plataformas em nuvem prontas para ML, arquitetura de microsserviços para deploy e orquestradores de pipeline baseados em código.
- Preocupação crescente com responsible AI e regulamentações globais sobre transparência de IA; discussão sobre uso de LLMs em projetos de ML, integração de dados de IoT e analytics em tempo real.
- O mercado valoriza cada vez mais profissionais que, além de técnicos, entendam de negócio e processo e consigam articular equipes multidisciplinares.

## Checklist de Estudo

- [ ] Sei explicar por que um notebook monolítico é insustentável para produção e como estruturar um projeto de ML em `src/`, `data/`, `models/`, `tests/` e `.github/workflows/`.
- [ ] Sei diferenciar versionamento de código (Git) de versionamento de dados e modelos (DVC/MLflow) e explicar por que ambos são necessários para rastreabilidade e auditoria.
- [ ] Sei explicar o que é training-serving skew e como um `sklearn.pipeline.Pipeline` previne esse problema e o vazamento de dados (data leakage).
- [ ] Sei aplicar o conceito de "Governance as Code" usando Fairlearn/MetricFrame para auditar disparidades entre grupos e bloquear o registro de um modelo que falhe nesse critério.
- [ ] Sei diferenciar Continuous Integration, Continuous Delivery e Continuous Training (CI/CD/CT) no contexto específico de Machine Learning.
- [ ] Sei explicar as três frentes de monitoramento em produção (serviço/operacional, data drift e model drift/performance) e por que o atraso do ground truth torna o data drift um proxy relevante.
- [ ] Sei conectar decisões de Business Understanding (KPI de negócio, métrica prioritária, exigência de tempo real) às escolhas técnicas de arquitetura de deployment e aos limiares de monitoramento.

## Palavras-chave

Machine Learning. Ciclo Completo de ML. MLOps.

## Referências

- DATADOG. **Machine learning model monitoring: Best practices**. 2024. Disponível em: https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/. Acesso em: 28 jan. 2026.
- EVIDENTLY AI. **Model monitoring for ML in production: a comprehensive guide**. 2025. Disponível em: https://www.evidentlyai.com/ml-in-production/model-monitoring. Acesso em: 28 jan. 2026.
- GOOGLE CLOUD ARCHITECTURE CENTER. **MLOps: Continuous delivery and automation pipelines in machine learning**. 2024. Disponível em: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning. Acesso em: 28 jan. 2026.
- MARTIN, M. **Model Deployment: Types, Strategies and Best Practices**. 2022. Disponível em: https://dagshub.com/blog/model-deployment-types-strategies-and-best-practices. Acesso em: 28 jan. 2026.
- MITCHELL, M. et al. **Model Cards for Model Reporting**. 2019. Disponível em: https://arxiv.org/abs/1810.03993. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. **MLflow Model Registry & Serving**. 2025. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- SCIKIT-LEARN. **Pipeline and Feature Union**. 2023. Disponível em: https://scikit-learn.org/stable/modules/compose.html. Acesso em: 28 jan. 2026.
- WEERTS, H. et al. **Fairlearn: Assessing and Improving Fairness of AI Systems**. 2023. Disponível em: https://www.jmlr.org/papers/volume24/23-0389/23-0389.pdf. Acesso em: 28 jan. 2026.
- WIRTH, R.; HIPP, J. **CRISP-DM: Towards a standard process model for data mining**. 2000. Disponível em: https://www.researchgate.net/publication/2424487_CRISP-DM_Towards_a_standard_process_model_for_data_mining. Acesso em: 28 jan. 2026.
