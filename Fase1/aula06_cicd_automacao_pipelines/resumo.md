# Aula 06 — CI/CD e Automação de Pipelines

## Visão Geral

Modelos de Machine Learning não são sistemas estáticos: seu desempenho tende a degradar com o tempo devido a mudanças de dados e de ambiente, e por isso o trabalho não termina quando o modelo vai para produção. Esta aula introduz práticas de CI/CD aplicadas a Machine Learning no contexto de MLOps, explorando como integração e entrega contínua, combinadas a pipelines automatizados, tornam o ciclo de vida de modelos mais confiável, reprodutível e ágil. O foco está em como automatizar testes, treinamento, registro e implantação de modelos, unindo princípios de DevOps ao mundo de ML.

## Tópicos Abordados

- **O que vem por aí?** — contextualização do desafio de manter modelos em produção e introdução do CI/CD aplicado a ML (MLOps).
- **Hands On** — cenário prático de automação de teste e implantação de um modelo de previsão de doenças cardíacas com GitHub Actions e MLFlow.
- **Saiba Mais**
  - Estrutura do repositório e workflow básico no GitHub Actions
  - Exemplo de script de treino com log no MLFlow
  - Estratégias de deploy (Model Registry + Webhooks; continuação do workflow CI/CD)
  - Definições de CI, CD (Entrega Contínua x Implantação Contínua) e MLOps
  - Componentes e Ferramentas de CI/CD para ML
  - Desafios de CI/CD em ML
  - Pipelines Orquestrados com Kubeflow e TFX
  - Infrastructure as Code (IaC) e ambientes reprodutíveis
  - Validação Automatizada de Dados e Modelos
- **Mercado, Cases e Tendências** — adoção de MLOps, demandas de mercado, cases (Uber, Netflix), integração com DevOps, human-in-the-loop, tendências tecnológicas, desafios de adoção e ROI.

## Conceitos-Chave

### CI/CD tradicional versus CI/CD para ML
Em desenvolvimento de software tradicional, CI (Integração Contínua) significa mesclar alterações de código com frequência e testá-las automaticamente, enquanto CD pode ser Entrega Contínua (preparar versões prontas para implantação) ou Implantação Contínua (implantar automaticamente em produção). No contexto de MLOps, a Integração Contínua ganha novos elementos: não apenas código, mas também dados e modelos passam a fazer parte da integração, exigindo que o pipeline de ML seja reexecutado a cada mudança relevante, com versionamento e reprodutibilidade dos resultados.

### MLOps
Abordagem que une DevOps e Data Science para facilitar a entrega confiável de modelos de ML, aplicando os princípios de integração e entrega contínua ao ciclo de vida completo de modelos — do código e dados ao treinamento, validação, deploy e monitoramento.

### Entrega Contínua x Implantação Contínua em ML
A Entrega/Implantação Contínua em ML envolve automatizar não só o deploy de aplicações, mas do próprio modelo treinado. Após treinado e validado automaticamente, o modelo pode ser entregue a um repositório de modelos ou implantado em um serviço de inferência com mínima intervenção humana. Muitas equipes adotam um fluxo de promoção por estágio: Entrega Contínua até um ambiente de staging, seguida de promoção manual ou semiautomática à produção — já que Implantação Contínua totalmente automática nem sempre é adotada por questões de controle, especialmente em cenários críticos (saúde, finanças).

### Continuous Training (CT)
A automação completa do workflow de ML — que envolve retreinar continuamente conforme chegam dados novos — é às vezes chamada de CT (Continuous Training), complementando o CI/CD tradicional.

### Componentes e Ferramentas de um pipeline de CI/CD para ML
- **Controle de versão (Git)** — armazena o código e aciona pipelines quando há mudanças.
- **Serviço de Integração Contínua** — orquestra builds e testes automaticamente (Jenkins, GitHub Actions, GitLab CI, Azure DevOps).
- **Serviço de Deploy/Entrega** — gerencia a implantação dos artefatos aprovados.
- **Registro de Modelos** — repositório central para versionar e gerenciar modelos aprovados (MLFlow Model Registry, SageMaker Model Registry, etc.).
- **Orquestrador de Pipelines de ML** — define e executa pipelines de treinamento e processamento de dados de forma escalável (Kubeflow Pipelines, Apache Airflow, Vertex AI Pipelines).
- **Feature Store** — repositório para armazenar e versionar features, garantindo consistência entre treino e produção (Feast, Feathr).
- **Metadados de ML** — sistemas para armazenar parâmetros, métricas e lineage de dados (MLFlow Tracking, Kubeflow Metadata).

### Ferramentas de orquestração CI/CD e frameworks de MLOps
- **GitHub Actions** — plataforma de CI/CD integrada ao GitHub, com workflows definidos em YAML (`.github/workflows`), acionados por eventos como push ou pull request.
- **Jenkins** — ferramenta open-source clássica de CI, autohospedada, com milhares de plugins; usada em cenários que exigem mais controle do ambiente (ex.: GPUs locais, on-premises).
- **GitLab CI/CD** — pipelines definidos em `.gitlab-ci.yml`, com recursos multi-stage e Model Registry integrado.
- **CML (Continuous Machine Learning)** — ferramenta open-source da Iterative que integra CI a projetos de ML, gerando relatórios e comentários automáticos em Pull Requests com métricas de experimentos.
- **MLFlow** — não é uma ferramenta de orquestração de CI/CD, mas é crucial no pipeline MLOps: o MLFlow Tracking loga parâmetros, métricas e artefatos de cada execução, e o MLFlow Model Registry versiona modelos e controla estágios (Staging, Production).
- **Kubeflow Pipelines** — plataforma open-source para orquestrar pipelines de ML sobre Kubernetes, definindo um DAG de etapas (pré-processamento, treino, validação, deploy) em containers isolados.
- **Infraestrutura como Código (IaC)** — gerenciar servidores, recursos de nuvem e configurações de container via código (Terraform, CloudFormation, Azure ARM), garantindo ambientes consistentes e reprodutíveis.

### Pipelines Orquestrados com Kubeflow e TFX
Em projetos de maior porte, é comum separar a orquestração do workflow de ML em sistemas dedicados como o Kubeflow Pipelines, que permite definir um DAG de etapas de ML executadas em containers isolados e escaláveis. O TensorFlow Extended (TFX), da Google, define pipelines de ponta a ponta com componentes para ingestão, transformação, treino, validação de modelo (TensorFlow Model Analysis) e deploy, podendo rodar sobre Airflow, Kubeflow ou Beam — assegurando que nenhum modelo vá a produção sem passar por verificações de drift e skew entre treino e serviço.

### Infrastructure as Code (IaC) aplicada a MLOps
Com IaC, a infraestrutura necessária para treinamento (ex.: VM com GPU e RAM específicas) é codificada com ferramentas como Terraform, permitindo criar recursos sob demanda (`terraform apply`) antes do treinamento e destruí-los depois, otimizando custos. A IaC contribui para reprodutibilidade e padronização entre dev, staging e prod, e mudanças de infraestrutura passam a ser versionadas e revisadas via pull request, alinhadas ao espírito DevOps. Containerização (Docker) e Kubernetes complementam essa prática, garantindo que o ambiente de execução seja idêntico entre CI, local e produção.

### Validação Automatizada de Dados e Modelos
Ferramentas como Great Expectations, Pandera e Deepchecks atuam como "testes unitários" para dados, permitindo definir expectativas (ex.: coluna sem nulos, valores dentro de um intervalo, distribuição que não diverge do histórico) que, incorporadas ao pipeline de CI, impedem que dados sujos degradem o modelo silenciosamente. Validação de modelo pode incluir comparar o modelo novo com o antigo em métricas de teste e verificar surgimento de viés entre subgrupos demográficos — embora questões complexas de fairness ainda exijam análise humana.

### Desafios de CI/CD em ML
- **Versionamento de Dados e Modelos** — dados de treino podem ser grandes demais para o repositório Git.
- **Tempo de Execução e Recursos** — treinar modelos pode levar horas/dias e exigir GPUs, tornando o pipeline pesado; muitas empresas usam jobs agendados em vez de treinar a cada commit.
- **Testes Automatizados de Modelos** — comparar métricas do novo modelo com o modelo em produção e usar testes de sanity (ex.: detectar NaN).
- **Reprodutibilidade** — exige conteinerização, pin de versões de dependências e fixação de seeds aleatórios.
- **Segurança e Governança** — necessidade de aprovação humana em cenários críticos e armazenamento seguro de secrets.
- **Integração com Monitoramento** — fechar o ciclo CI/CD/CM (Continuous Monitoring), onde alertas de drift podem acionar pipelines de retraining.

## Exercício Hands-On (do material)

Cenário proposto: você possui um projeto de ML em um repositório GitHub — o modelo de previsão de doenças cardíacas treinado em Python — e deseja automatizar o processo de teste e implantação desse modelo sempre que houver alterações no código, além de registrar os resultados (métricas e modelo) em uma ferramenta de tracking (MLFlow) para manter rastreabilidade.

O repositório segue a estrutura:

```
heart-disease-prediction/
├── data/ (dados de treino/validação)
├── src/
│   ├── train.py (treinamento do modelo)
│   ├── register.py (registro do modelo)
│   └── utils.py (funções auxiliares)
├── tests.py (testes unitários)
├── mlflow_runs/ (diretório onde salvar experimentos do mlflow)
├── requirements.txt (dependências Python)
└── README.md
```

O objetivo é configurar um workflow de GitHub Actions (`.github/workflows/ci_ml_pipeline.yml`) que, a cada push no branch principal, execute:

1. Instalar dependências e rodar testes no código (com PyTest) — se algum teste falhar, o job para e o Actions marca falha.
2. Treinar (ou retreinar) o modelo, usando credenciais do MLflow armazenadas como Secrets do GitHub (`MLFLOW_TRACKING_URI`).
3. Opcionalmente, se o modelo for satisfatório, registrar a nova versão no Model Registry do MLFlow ou implantar o modelo.
4. Avaliar o pipeline e reportar os resultados, tudo de forma automática.

`train.py` lê os dados e treina o modelo; `tests.py` executa testes unitários que checam se as funções de `train.py` funcionam adequadamente; `register.py` registraria os artefatos gerados no MLflow caso o novo modelo treinado fosse melhor que o já registrado — garantindo que só modelos com métricas (acurácia, F1-Score etc.) melhores que o modelo em produção avancem.

Após configurar o workflow, um commit no branch `main` alterando um hiperparâmetro aciona o GitHub Actions, visível na aba "Actions" do repositório, com logs em tempo real de cada etapa (checkout, instalação, testes, treino). Se o pipeline for bem-sucedido, o run é marcado como verde, e cada commit fica rastreável a um run no MLFlow (inclusive salvando o commit hash como tag do run). O material sugere ainda que a implantação em produção pode ser automatizada via webhooks do MLFlow Model Registry ou via passos adicionais no próprio workflow YAML, mas recomenda, para maior segurança, Entrega Contínua automatizada até staging combinada com aprovação manual antes da promoção a produção.

## Exemplos de Código

O PDF descreve o workflow de GitHub Actions e o script de treino de forma textual/em figuras; os blocos abaixo operacionalizam esses mesmos conceitos em código executável, servindo como material de apoio ao estudo.

Workflow de CI/CD que ilustra o pipeline descrito no hands-on: checkout, setup do Python, instalação de dependências, testes automatizados e treinamento com log no MLFlow, disparado a cada push na branch `main`.

```yaml
# .github/workflows/ci_ml_pipeline.yml
name: CI Pipeline ML

on:
  push:
    branches: ["main"]

jobs:
  build_and_train:
    runs-on: ubuntu-latest
    steps:
      - name: Checar código
        uses: actions/checkout@v3

      - name: Configurar Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Instalar dependências
        run: pip install -r requirements.txt

      - name: Executar testes
        run: pytest tests/ --maxfail=1 -q

      - name: Treinar modelo e logar no MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: python src/train.py

      - name: Guardar modelo como artefato do Actions
        uses: actions/upload-artifact@v3
        with:
          name: modelo-treinado
          path: mlflow_runs/
```

Script de treino (`src/train.py`) que ilustra o conceito de rastreabilidade via MLFlow Tracking mencionado na aula — parâmetros, métricas e o próprio modelo são logados a cada execução, junto com o commit hash para ligar código a modelo.

```python
# src/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Conectar ao tracking server (URL definida em MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
mlflow.set_experiment("MLPipeline_Exemplo")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with mlflow.start_run():
    params = {"n_estimators": 100, "random_state": 42}

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_metric("f1_score", float(f1))

    # Rastreabilidade: liga o run ao commit que o gerou
    commit_hash = os.environ.get("GITHUB_SHA", "local")
    mlflow.set_tag("git_commit", commit_hash)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Run finalizado. Acurácia={acc:.4f} | F1={f1:.4f}")
```

Testes automatizados de código (unitários), conforme mencionado no hands-on para validar que as funções de `train.py`/`utils.py` funcionam antes de treinar o modelo.

```python
# tests/test_train.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_model_trains_and_predicts():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert set(preds).issubset(set(y))


def test_no_nan_in_predictions():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert not np.isnan(preds).any()
```

Teste de validação de dados, ilustrando o conceito de "testes unitários para dados" (Great Expectations/Pandera) citado na seção de Validação Automatizada de Dados e Modelos.

```python
# tests/test_data_validation.py
import pandas as pd
import pytest


def test_sem_valores_nulos_inesperados():
    df = pd.read_csv("data/heart.csv")
    colunas_obrigatorias = ["age", "sex", "cp", "trestbps", "chol"]
    assert not df[colunas_obrigatorias].isnull().any().any(), (
        "Encontrados valores nulos em colunas obrigatórias"
    )


def test_idade_dentro_do_intervalo_esperado():
    df = pd.read_csv("data/heart.csv")
    assert df["age"].between(0, 120).all(), "Idade fora do intervalo esperado (0-120)"


def test_variavel_alvo_binaria():
    df = pd.read_csv("data/heart.csv")
    assert set(df["target"].unique()).issubset({0, 1}), "Target deve ser binário"
```

Teste de modelo que compara o novo modelo com o modelo atual em produção antes de permitir a promoção automática, conforme discutido nos desafios de CI/CD em ML ("Testes Automatizados de Modelos").

```python
# src/register.py
import mlflow
from mlflow.tracking import MlflowClient


def deve_promover_modelo(novo_run_id: str, nome_modelo: str, metrica: str = "accuracy",
                          limiar_minimo: float = 0.0) -> bool:
    client = MlflowClient()
    novo_run = client.get_run(novo_run_id)
    nova_metrica = novo_run.data.metrics.get(metrica)

    try:
        modelo_producao = client.get_latest_versions(nome_modelo, stages=["Production"])[0]
        run_producao = client.get_run(modelo_producao.run_id)
        metrica_producao = run_producao.data.metrics.get(metrica, 0.0)
    except IndexError:
        # Nenhum modelo em produção ainda: promove o primeiro que passar no limiar
        metrica_producao = limiar_minimo

    return nova_metrica is not None and nova_metrica >= metrica_producao


if __name__ == "__main__":
    import sys
    run_id = sys.argv[1]
    nome_modelo = "heart-disease-model"

    if deve_promover_modelo(run_id, nome_modelo):
        result = mlflow.register_model(f"runs:/{run_id}/model", nome_modelo)
        client = MlflowClient()
        client.transition_model_version_stage(
            name=nome_modelo, version=result.version, stage="Staging"
        )
        print(f"Modelo promovido a Staging: versão {result.version}")
    else:
        print("Modelo novo não superou o modelo em produção. Promoção abortada.")
```

Etapa adicional de deploy condicionado ao sucesso dos passos anteriores, ilustrando a abordagem "via continuação do Workflow CI/CD" descrita na aula para implantar o modelo após o treino e registro.

```yaml
# trecho adicional ao final de ci_ml_pipeline.yml
      - name: Registrar modelo no MLFlow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: python src/register.py ${{ steps.treino.outputs.run_id }}

      - name: Deploy do modelo para staging
        if: success()  # só roda se os steps anteriores tiveram sucesso
        run: python deploy.py --ambiente staging
```

Trecho de Infraestrutura como Código com Terraform, exemplificando como provisionar sob demanda os recursos necessários para o treinamento (ex.: instância com GPU), conceito apresentado na seção de IaC.

```hcl
# infra/main.tf
resource "aws_instance" "ml_training" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "g4dn.xlarge" # instância com GPU

  tags = {
    Name    = "ml-training-pipeline"
    Projeto = "heart-disease-prediction"
  }
}

output "instance_id" {
  value = aws_instance.ml_training.id
}
```

## Cases e Tendências de Mercado

- **Adoção Massiva de MLOps**: em 2022, cerca de 85% das empresas já possuíam orçamento dedicado a iniciativas de MLOps; em 2023, praticamente 98% das empresas planejavam aumentar investimentos, mais da metade em mais de 25%. O mercado global de MLOps é estimado em crescer de ~3,6 bilhões de dólares em 2025 para quase 8,7 bilhões até 2033.
- **Demanda por velocidade e eficiência**: setores como e-commerce (modelos de recomendação) e financeiro (modelos antifraude) dependem de atualização contínua de modelos; sem pipelines automatizados isso levaria meses, com MLOps pode levar dias ou horas.
- **Cases de Sucesso**: a Uber desenvolveu internamente a plataforma Michelangelo, automatizando desde o preparo de dados até deploy e monitoramento, atendendo centenas de casos de uso de ML; a Netflix investiu em pipelines para recomendações e pratica experimentação contínua (A/B) de modelos em produção.
- **Integração com DevOps e Engenharia de Dados**: convergência entre MLOps, DevOps e DataOps; surgimento do papel de Platform Engineer focado em ML; plataformas de dados (Databricks, Dataiku) incorporam pipeline automation e deploy de modelos.
- **Automatização + Human-in-the-loop**: coexistência de automação com revisão humana antes do deploy final, especialmente relevante em setores regulamentados que exigem accountability.
- **Tendências Tecnológicas**: CI/CD declarativo em YAML, serverless ML pipelines, crescimento de Feature Stores integrados, e adaptação de CI/CD para Deep Learning e Large Language Models (Continuous Retraining para LLMs, integrações do Hugging Face Hub).
- **Desafios na Adoção**: falta de mão de obra qualificada em DevOps/MLOps gera alta demanda por MLOps Engineers; necessidade de mudança cultural (code review de notebooks convertidos em scripts, colaboração entre times).
- **ROI e Qualidade**: empresas relatam redução no tempo de deploy (de semanas para dias), maior frequência de atualização de modelos e melhor rastreabilidade para auditoria — como uma empresa de saúde que passou a rastrear precisamente qual versão do modelo gerou cada previsão clínica, e uma empresa de manufatura que reduziu desperdício detectando drifts em horas em vez de meses.

## Checklist de Estudo

- [ ] Sei explicar o que é CI/CD no contexto de MLOps e diferenciar de CI/CD tradicional de software.
- [ ] Sei diferenciar Entrega Contínua de Implantação Contínua aplicadas a modelos de ML.
- [ ] Sei descrever os passos de um workflow de GitHub Actions integrado ao MLFlow (checkout, setup, testes, treino, registro).
- [ ] Sei identificar os principais componentes de um pipeline de CI/CD para ML (controle de versão, CI, deploy, model registry, orquestrador, feature store, metadados).
- [ ] Sei comparar as principais ferramentas de orquestração (GitHub Actions, Jenkins, GitLab CI, CML) e frameworks de MLOps (MLFlow, Kubeflow, TFX).
- [ ] Sei explicar os principais desafios de CI/CD em ML (versionamento de dados, tempo de execução, testes de modelo, reprodutibilidade, segurança, integração com monitoramento).
- [ ] Sei aplicar o conceito de Infraestrutura como Código (IaC) e validação automatizada de dados/modelos em um pipeline de MLOps.

## Palavras-chave

CI/CD. MLOps. Automação.

## Referências

- CLEARML BLOG. MLOps in 2023: What Does the Future Hold? 2023. Disponível em: https://clear.ml/blog/mlops-in-2023. Acesso em: 28 jan. 2026.
- GITHUB ACTIONS DOCUMENTATION. Understanding GitHub Actions. 2025. Disponível em: https://docs.github.com/en/actions/get-started/understand-github-actions. Acesso em: 28 jan. 2026.
- GOOGLE CLOUD ARCHITECTURE CENTER. MLOps: Continuous delivery and automation pipelines in machine learning. 2024. Disponível em: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning. Acesso em: 28 jan. 2026.
- LOGICMONITOR BLOG. Ops Explained: AIOps vs. DevOps vs. MLOps vs. Agentic AIOps. 2025. Disponível em: https://www.logicmonitor.com/blog/aiops-devops-mlops-and-agentic-aiops. Acesso em: 28 jan. 2026.
- MLOPS GUIDE. CI/CD for Machine Learning. 2021. Disponível em: https://mlops-guide.github.io. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. MLflow Model Registry. s.d. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- OJEABULU, G. Stop ML Model Failures: Complete Guide to Data Validation. 2025. Disponível em: https://medium.com/data-science-collective/stop-ml-model-failures-complete-guide-to-data-validation-with-pandera-great-expectations-dbt-d7656eeadfae. Acesso em: 28 jan. 2026.
- RED HAT DEVELOPER. Implement MLOps with Kubeflow Pipelines. 2024. Disponível em: https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines. Acesso em: 28 jan. 2026.
- RYABOKON, D. CI/CD/CT with MLflow and GitHub Actions. 2023. Disponível em: https://medium.com/@d.ryabokon/ci-cd-ct-with-mlflow-and-github-actions-2a6ba5c767dd. Acesso em: 28 jan. 2026.
- STRAITS RESEARCH. MLOps Market Size, Share, Trends & Growth by 2033. 2024. Disponível em: https://straitsresearch.com/report/mlops-market. Acesso em: 28 jan. 2026.
- VAISHNAV. ML CI/CD Pipeline: Architecture, Tools, and Real-World Structure. 2025. Disponível em: https://medium.com/@vaishnav0501/ml-ci-cd-pipeline-architecture-tools-and-real-world-structure-d10233d601c9. Acesso em: 28 jan. 2026.
- ZENML BLOG. Infrastructure-as-Code (IaC) for MLOps with Terraform & ZenML. 2024. Disponível em: https://www.zenml.io/blog/mlops-terraform-zenml. Acesso em: 28 jan. 2026.
