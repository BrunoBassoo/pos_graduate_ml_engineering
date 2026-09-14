# Aula 04 — Implantação de Modelos (Deployment)

## Visão Geral

Esta aula cobre a fase de Implantação (Deployment) do ciclo de vida de Machine Learning, etapa que, segundo a metodologia CRISP-DM, vem logo após a Avaliação e é onde o modelo efetivamente passa a gerar valor de negócio. O foco desloca-se do Jupyter Notebook para uma mentalidade de engenharia de software, infraestrutura e operações (MLOps), assumindo o papel de um Machine Learning Engineer que precisa empacotar o modelo treinado, escolher uma estratégia de deployment adequada e integrá-lo ao fluxo de trabalho de uma aplicação real. O exemplo prático utilizado é o modelo de classificação de doença cardíaca (Random Forest) construído em aulas anteriores, implantado tanto via API REST customizada (Flask) quanto via plataforma gerenciada (MLflow Model Serving no Databricks).

## Tópicos Abordados

- **O que vem por aí?** — Motivação para o deployment: transição do modelo treinado (notebook) para um serviço em produção; papel do Machine Learning Engineer.
- **Hands On** — Deployment do modelo de classificação de doença cardíaca (Random Forest) por duas abordagens: API REST com Flask e MLFlow Model Serving na Databricks.
- **Saiba Mais**
  - Parte 1: Deploy via Flask (API REST) – modelo como microsserviço
  - Parte 2: Deploy via MLflow Model Serving (Databricks)
  - Conceitos e Importância do Deployment de Modelos
  - Estratégias de Implantação de Modelos
    - Predições em Batch (Processamento em Lote)
    - Predições em Tempo Real (Serviço Online)
    - Modelos Embarcados em Aplicações (Edge AI)
    - Streaming de Dados e Inferência Assíncrona
  - Empacotamento e Portabilidade de Modelos
    - Salvando modelos com Pickle ou Joblib
    - Exportação para ONNX (e outros formatos portáteis)
  - Ferramentas de Deployment de Modelos – MLflow e Model Registry
  - Conteinerização com Docker
- **Mercado, Cases e Tendências** — Aplicações por indústria: E-commerce (Netflix/Amazon), Finanças (fraude e risco de crédito), Saúde (diagnóstico assistido), Manufatura (manutenção preditiva), Tecnologia (serviços web escaláveis).
- **O que você viu nesta aula?** — Recapitulação dos conceitos de deployment, estratégias, empacotamento, ferramentas, Docker e desafios de engenharia, escalabilidade e segurança.
- **Referências** — Fontes bibliográficas e documentações consultadas.

## Conceitos-Chave

### Deployment de Modelos
É o processo de colocar em operação um modelo treinado, de forma que ele possa receber dados novos e produzir previsões para usuários ou sistemas — ou seja, encaixar o modelo dentro de uma aplicação maior. É somente após o deployment que o modelo passa a gerar valor real para a organização; sem implantação, mesmo o modelo mais preciso permanece uma prova de conceito isolada.

### MLOps (Machine Learning Operations)
Conjunto de práticas que trata a implantação de modelos como software: versionamento, testes, deploy automatizado, monitoramento contínuo e protocolos de rollback. Garante que o modelo opere 24/7 de forma robusta, eficiente e segura em produção.

### Desafios do Ambiente de Produção
Diferentemente do notebook, em produção o modelo precisa lidar com: dados de entrada imprevisíveis (risco de *drift*), exigências de desempenho e escala (baixa latência ou processamento em massa), integração com sistemas existentes, confiabilidade/observabilidade/manutenção contínua, e segurança e privacidade (LGPD/GDPR, controle de acesso, proteção contra extração do modelo).

### Desafios Técnicos do Deployment
- **Engenharia**: construir APIs e pipelines de dados que conectem o modelo à aplicação.
- **Escalabilidade**: projetar para escalar horizontalmente (mais instâncias) ou verticalmente (mais recursos) conforme a demanda.
- **Segurança**: controle de acesso, criptografia em trânsito e repouso, conformidade regulatória.
- **Manutenção e monitoramento**: detectar drift de dados/conceito e decidir quando retreinar.
- **Trade-off velocidade vs. custo**: menor latência geralmente exige mais recursos computacionais.

### Implantação Batch (Processamento em Lote)
O modelo processa blocos de dados em cadência programada (diária, semanal etc.), em vez de responder a cada requisição individualmente. Vantagens: eficiência, simplicidade, uso de janelas de baixo tráfego, maior facilidade de teste e reprodução. Desvantagem: latência entre a disponibilidade dos dados e a obtenção das predições — não serve para respostas instantâneas. Implementação típica: job scheduler/orquestrador (Apache Airflow, Databricks Jobs) executando um script que carrega o modelo, lê um lote de dados e grava os resultados; ferramentas de big data como Spark ajudam a escalar.

### Implantação em Tempo Real (Serviço Online)
O modelo fica disponível como um serviço (tipicamente API REST) que responde a requisições individuais em milissegundos/segundos. Usado quando há um usuário ou sistema aguardando a resposta para agir (credit scoring, detecção de fraude, chatbots). Exige infraestrutura de serving sempre ativa, baixa latência, tolerância a falhas e escalabilidade para concorrência. REST é a interface mais comum pela simplicidade; gRPC pode ser usado para latência ultra baixa.

### Implantação na Borda (Edge AI / On-device)
O modelo é embutido diretamente na aplicação ou dispositivo (app mobile, navegador, IoT), rodando localmente sem depender de um servidor remoto. Indicado para latência ultra baixa, conectividade instável/inexistente e requisitos de privacidade (dados não saem do dispositivo). Desvantagens: limitação de recursos computacionais/energia, dificuldade de atualização (exige nova versão do app/firmware) e menor visibilidade para monitoramento. Formatos típicos de exportação: ONNX, CoreML, TensorFlow Lite.

### Arquiteturas de Streaming Assíncrono
O modelo processa um fluxo contínuo de eventos via sistema de mensageria (filas, Kafka), sem uma requisição específica de um cliente por evento. Útil para alto throughput (ex.: detecção de fraude em fluxo de transações). Desacopla produção e consumo — se o modelo cair, os eventos ficam na fila e são processados ao retornar. Ferramentas: Apache Kafka, AWS Kinesis, RabbitMQ, Flink, Spark Streaming.

### Empacotamento e Portabilidade de Modelos
Etapa de serializar o modelo para que possa ser carregado em outro ambiente.
- **Pickle/Joblib**: serialização nativa do Python; simples, mas dependente das mesmas versões de bibliotecas usadas no treino (risco de incompatibilidade) e pode gerar arquivos grandes.
- **ONNX (Open Neural Network Exchange)**: formato aberto e padronizado, funciona como "linguagem comum" entre frameworks (PyTorch, scikit-learn) e linguagens (C++, Java, .NET, mobile).
- **Outros formatos**: HDF5/SavedModel (Keras/TensorFlow, servido via TF Serving ou TF Lite) e TorchScript (PyTorch, independente do Python em runtime).

### MLflow e Model Registry
O MLflow é uma plataforma de código aberto para o ciclo de vida de ML. O **Model Registry** é um repositório central e versionado de modelos, permitindo transições controladas de estágio (staging → production → archived), rastreabilidade (metadados, métricas, linhagem) e integração com deployment automatizado (CI/CD). O MLflow também oferece comandos para servir modelos rapidamente (`mlflow models serve`), integração com AWS SageMaker/Azure ML e geração automática de imagens Docker.

### Conteinerização com Docker
Plataforma que empacota a aplicação (código do serviço, modelo e dependências) em uma imagem padronizada, isolada e portátil, garantindo execução consistente em qualquer ambiente ("funciona na minha máquina" deixa de ser problema). Facilita escalabilidade (múltiplas instâncias atrás de um load balancer) e integração com pipelines CI/CD. Ferramentas de ML como MLflow e serviços de nuvem (SageMaker, Azure ML) já incorporam containers nativamente.

### Abordagem Customizada (Flask) vs. Plataforma Gerenciada (Databricks MLflow)
- **Flask/self-hosted**: alta flexibilidade e controle total da arquitetura; ideal para protótipos ou equipes com forte domínio de DevOps; porém exige grande esforço de engenharia para construir e manter rastreamento, registro e monitoramento (MLOps próprio), elevando custos indiretos (TCO).
- **Plataforma gerenciada (Databricks MLflow, SageMaker, Azure ML)**: menos flexível, mas oferece governança, MLOps nativo, autoscaling e segurança integrada; mais eficiente para cenários corporativos com muitos modelos e prioridade de time-to-market; risco de vendor lock-in e custo direto mais elevado.

### Integração com DevOps/CI-CD
Fluxo típico: modelo treinado e registrado (MLflow) → pipeline CI/CD detecta versão marcada "Production" → constrói imagem Docker com modelo e código de inferência → implanta em cluster (Kubernetes/serverless) → executa testes de saúde → modelo entra em uso, com possibilidade de rollback.

## Exercício Hands-On (do material)

Dando continuidade ao cenário aplicado à área da Saúde da aula anterior, o material propõe implantar o modelo de classificação de doença cardíaca (Random Forest) treinado previamente, explorando **duas abordagens de deployment**:

1. **Deploy via Flask (API REST) — modelo como microsserviço**
   - Salvar o modelo treinado com `joblib` (`best_random_forest.joblib`).
   - Criar uma aplicação Flask (`app.py`) com dois endpoints: `"/"` (GET, health check) e `"/heart-disease-predict"` (POST, recebe payload JSON e retorna a predição em JSON).
   - Garantir que a aplicação carregue exatamente o mesmo objeto `Pipeline` do scikit-learn usado no treino (incluindo pré-processamento com One-Hot-Encoding e Feature Engineering), convertendo o JSON de entrada em DataFrame, validando/reordenando colunas e aplicando `.predict()`.
   - Testar localmente com `flask run` ou `python app.py`, enviando requisições de teste (ex.: via `curl`) e verificando se a predição faz sentido.
   - Discutir a evolução para produção: empacotar em container Docker, rodar múltiplas instâncias atrás de um load balancer ou orquestrador (Kubernetes, AWS EKS, Azure AKS), com pipelines de CI/CD e monitoramento (Prometheus, Grafana).

2. **Deploy via MLflow Model Serving (Databricks)**
   - Acessar o Databricks Free Edition (plataforma Lakehouse baseada em Spark, com MLflow integrado para o ciclo de vida de ML).
   - Treinar um Pipeline completo (imputação, encoding categórico, feature engineering, scaler, classificador) registrando parâmetros, métricas e a assinatura do modelo (*signature*) no experimento `heart-disease-prediction` via `mlflow.sklearn.log_model`.
   - Registrar o modelo no Model Registry com o nome `heart-disease-model`, promovendo a versão para "Staging" e depois "Production" usando `MlflowClient().set_registered_model_alias`.
   - Servir o modelo clicando em "Serve this model" no Databricks Model Serving (ou via Databricks SDK), obtendo um endpoint REST (`heart-disease-predict`).
   - Consumir o modelo servido via REST API, de forma análoga ao endpoint Flask.

O objetivo do exercício é comparar as duas estratégias — uma **low-level** (construir o próprio serviço com Flask) e outra **high-level** (usar uma plataforma gerenciada) — discutindo os fatores de decisão: volume de dados, especialização dos times, requisitos de escalabilidade, governança, TCO e vendor lock-in.

## Exemplos de Código

A seguir, exemplos elaborados para ilustrar na prática os conceitos apresentados no PDF (empacotamento, API REST em Flask, containerização, estratégias batch/real-time/streaming e MLflow), mesmo quando o material original não trouxe o código completo correspondente.

Ilustra a persistência do modelo treinado com `joblib`, conforme descrito na seção "Salvar o modelo treinado" do Hands On:

```python
import os
import joblib

# Persistência local com joblib (conforme Código-fonte 1 do material)
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)
joblib_path = os.path.join(models_dir, "best_random_forest.joblib")
joblib.dump(best_rf, joblib_path)
print(f"Modelo salvo via joblib em: {joblib_path}")
```

Ilustra a aplicação Flask com os endpoints de health check e de predição descritos na Parte 1 ("Deploy via Flask"), incluindo carregamento do Pipeline, preparação dos dados e validação de features:

```python
# app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregamento do Pipeline completo (pré-processamento + RandomForest)
pipeline = joblib.load("../models/best_random_forest.joblib")
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "heart-disease-api healthy"}), 200


@app.route("/heart-disease-predict", methods=["POST"])
def predict():
    payload = request.get_json()

    # Preparação dos dados: JSON -> DataFrame tabular
    input_df = pd.DataFrame([payload])

    # Validação de features: reordenar colunas conforme o treino
    missing = set(EXPECTED_COLUMNS) - set(input_df.columns)
    if missing:
        return jsonify({"error": f"Campos ausentes: {missing}"}), 400
    input_df = input_df[EXPECTED_COLUMNS]

    # Predição unificada via Pipeline
    prediction = pipeline.predict(input_df)[0]
    risco = "Alto Risco" if prediction == 1 else "Baixo Risco"

    return jsonify({"prediction": int(prediction), "risco": risco}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

Ilustra o exemplo de requisição via `curl` para o endpoint de predição, reproduzindo o Código-fonte 2 do material:

```bash
curl -X POST http://localhost:5000/heart-disease-predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalch": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

Ilustra a containerização da API Flask com Docker, ponto discutido no material como "Container Docker" e na seção de Conteinerização, garantindo empacotamento padronizado, isolado e portátil:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py preprocessing.py ./
COPY ../models/best_random_forest.joblib ./models/best_random_forest.joblib

EXPOSE 5000

# Em produção, usar um servidor WSGI (ex.: gunicorn) em vez do servidor de desenvolvimento do Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

Ilustra o cenário de produção descrito no material — múltiplas instâncias atrás de um load balancer — usando `docker-compose` para simular réplicas do serviço:

```yaml
# docker-compose.yml
version: "3.9"

services:
  heart-disease-api:
    build: .
    image: heart-disease-api:latest
    deploy:
      replicas: 3
    ports:
      - "5000-5002:5000"
    environment:
      - MODEL_PATH=/app/models/best_random_forest.joblib
    restart: unless-stopped
```

Ilustra a implantação batch descrita no material (script que carrega o modelo, lê um lote de dados e grava os resultados), aplicada ao exemplo de pacientes cadastrados durante a noite:

```python
# batch_predict.py — executado por um scheduler (ex.: Apache Airflow) uma vez por dia
import joblib
import pandas as pd
from datetime import date

pipeline = joblib.load("../models/best_random_forest.joblib")

# Lote de novos pacientes cadastrados no dia (ex.: extraído do banco hospitalar)
novos_pacientes = pd.read_csv("dados/pacientes_novos.csv")

predicoes = pipeline.predict(novos_pacientes)
novos_pacientes["risco_predito"] = predicoes

alto_risco = novos_pacientes[novos_pacientes["risco_predito"] == 1]
saida = f"relatorios/alto_risco_{date.today().isoformat()}.csv"
alto_risco.to_csv(saida, index=False)
print(f"{len(alto_risco)} pacientes de alto risco enviados para revisão médica em {saida}")
```

Ilustra a arquitetura de streaming assíncrono via fila, mencionada como alternativa para alto throughput sem interação direta com o usuário (ex.: detecção de fraude em transações via Kafka):

```python
# stream_consumer.py — consumidor Kafka aplicando o modelo a cada evento recebido
import json
import joblib
from kafka import KafkaConsumer, KafkaProducer

pipeline = joblib.load("../models/fraud_model.joblib")

consumer = KafkaConsumer(
    "transacoes-financeiras",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

for evento in consumer:
    transacao = evento.value
    score = pipeline.predict_proba([list(transacao.values())])[0][1]
    resultado = {**transacao, "fraud_score": score, "bloquear": score > 0.8}
    producer.send("transacoes-avaliadas", resultado)
```

Ilustra o registro e a promoção de um modelo no MLflow Model Registry (Staging → Production), conforme descrito na Parte 2 do Hands On:

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_experiment("/heart-disease-prediction")

with mlflow.start_run(run_name="complete_pipeline_with_best_params"):
    for param_name, param_value in relevant_params.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)

    signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
    )

model_registered = mlflow.register_model(
    model_uri=model_info.model_uri, name="heart-disease-model"
)

# Promover a versão para "Production"
MlflowClient().set_registered_model_alias(
    model_registered.name, "Production", model_registered.version
)
```

## Cases e Tendências de Mercado

- **E-commerce (Recomendações)**: Amazon e Netflix combinam batch e tempo real — a Netflix gera diariamente, em batch, recomendações personalizadas para a homepage, e usa tempo real para sugestões contextuais ("quem viu isso também viu aquilo"), orquestrado com pipelines big data (Spark) e microsserviços.
- **Finanças (Fraude e Risco de Crédito)**: bancos classificam transações suspeitas em milissegundos, exigindo deployment em tempo real com altíssima disponibilidade, baixa latência (proximidade de data centers regionais), forte segurança contra corrupção do modelo e monitoramento constante com protocolos de remoção em caso de queda de performance.
- **Saúde (Diagnóstico Assistido)**: modelos que detectam nódulos em imagens de raio-X começam em batch (lote de imagens processado ao fim do dia) e caminham para tempo real embarcado (edge) nos próprios aparelhos (ex.: Siemens Healthineers, GE), com forte exigência de validação e certificação regulatória (ex.: FDA), o que torna o ciclo de atualização mais lento.
- **Manufatura (Manutenção Preditiva)**: sensores IoT alimentam modelos rodando em edge gateways locais nas máquinas, permitindo ação imediata (parada de máquina) sem depender de conectividade; case citado: General Electric com o sistema Predix em turbinas de avião e locomotivas.
- **Tecnologia (Serviços Web Escaláveis)**: empresas como Google e Facebook operam deployment em escala máxima, com hardware customizado (ex.: Google TPU), canary releases (mudanças testadas em 1% dos usuários) e rollback automático; case citado: Uber Michelangelo, plataforma interna que gerencia todo o ciclo de treino a deploy para casos como estimativa de tempo de chegada e precificação dinâmica.
- Conclusão do material: não existe uma estratégia de deployment única — cada domínio exige avaliação de requisitos e restrições específicas para arquitetar a solução apropriada.

## Checklist de Estudo

- [ ] Sei explicar por que o deployment é a etapa em que o modelo passa a gerar valor de negócio, segundo o CRISP-DM.
- [ ] Sei diferenciar implantação batch, tempo real, edge (on-device) e streaming assíncrono, incluindo quando usar cada uma.
- [ ] Sei descrever o fluxo de deployment de um modelo via Flask (salvar com joblib, carregar Pipeline, endpoints REST).
- [ ] Sei explicar o papel do MLflow Model Registry (staging/production) e do MLflow Model Serving no Databricks.
- [ ] Sei comparar vantagens e desvantagens de uma solução customizada (Flask) versus uma plataforma gerenciada (Databricks MLflow, SageMaker, Azure ML).
- [ ] Sei explicar por que a containerização com Docker é importante para consistência, portabilidade e escalabilidade.
- [ ] Sei identificar os principais formatos de empacotamento de modelos (Pickle/Joblib, ONNX, SavedModel/HDF5, TorchScript) e seus cuidados de compatibilidade de versão.

## Palavras-chave

- Deployment de modelos
- MLFlow
- Model Serving

## Referências

- AWS BLOGS. **Batch vs Real-Time Inference in AWS Machine Learning**. 2020. Disponível em: https://aws.amazon.com/blogs. Acesso em: 28 jan. 2026.
- DATABRICKS DOCUMENTATION. **Mosaic ML Model Serving**. 2025. Disponível em: https://docs.databricks.com. Acesso em: 28 jan. 2026.
- DATASCIENCE-PM. **What is CRISP DM?** 2024. Disponível em: https://www.datascience-pm.com/crisp-dm-2. Acesso em: 28 jan. 2026.
- EMMANUEL, C. **Predicting Heart Disease Flask – A Practical Deployment Guide**. 2025. Disponível em: https://chris-emmanuel.medium.com/predicting-heart-disease-with-machine-learning-and-flask-a-practical-deployment-guide-39ba5a388bf5. Acesso em: 28 jan. 2026.
- HAPKE, H.; NELSON, C. **Building Machine Learning Pipelines**. 2020. Sebastopol: O'Reilly.
- MARTIN, M. **Model Deployment: Types, Strategies and Best Practices**. 2022. Disponível em: https://dagshub.com/blog/model-deployment-types-strategies-and-best-practices. Acesso em: 28 jan. 2026.
- MEHREEN, K. **Step-by-Step Guide to Deploying ML Models with Docker**. 2024. Disponível em: https://www.kdnuggets.com/step-by-step-guide-to-deploying-ml-models-with-docker. Acesso em: 28 jan. 2026.
- MIRANTIS. **Model Deployment and Orchestration: The Definitive Guide**. 2021. Disponível em: https://www.mirantis.com/blog/model-deployment-and-orchestration-the-definitive-guide/. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. **MLflow Model Registry**. 2025. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. **MLflow Model Serving**. 2025. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- SCIKIT-LEARN DOCUMENTATION. **Model Persistence**. 2023. Disponível em: https://scikit-learn.org. Acesso em: 28 jan. 2026.
- SHELF.IO. **In-depth Guide to ML Model Deployment**. 2023. Disponível em: https://shelf.io/blog/machine-learning-deployment. Acesso em: 28 jan. 2026.
- STACKADEMIC BLOG. **MLOps: Model Deployment Strategies — Batch vs. Real-Time vs. Edge**. 2024. Disponível em: https://blog.stackademic.com/mlops-model-deployment-strategies-batch-vs-real-time-vs-edge-c7b1ac642e8f. Acesso em: 28 jan. 2026.
- TERA BLOG. **Entenda o que é deploy de modelos em Machine Learning**. s.d. Disponível em: https://blog.somostera.com/data-science/deploy-o-que-e. Acesso em: 28 jan. 2026.
