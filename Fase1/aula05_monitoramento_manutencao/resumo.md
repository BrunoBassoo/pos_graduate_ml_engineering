# Aula 05 — Monitoramento e Manutenção

## Visão Geral

Esta aula trata do que acontece depois que um modelo de Machine Learning é colocado em produção: como monitorar continuamente seu desempenho e realizar manutenção proativa para garantir que ele continue confiável. São discutidos os sinais de degradação de performance (data drift e model drift), as métricas de acompanhamento (desde acurácia preditiva até latência e uso do modelo) e as ferramentas e práticas usadas para observar, diagnosticar e adaptar modelos em produção. O fio condutor prático é o modelo de detecção de doença cardíaca implantado via REST API nas aulas anteriores, que agora é instrumentado para coleta de métricas e definição de critérios de retreino.

## Tópicos Abordados

- **O que vem por aí?** — introdução ao problema da degradação de modelos em produção e ao cenário motivador do modelo de detecção de doença cardíaca.
- **Hands On** — continuidade do caso prático do endpoint REST de detecção de doença cardíaca, agora com foco em instrumentar o monitoramento de métricas e definir critérios de retreino.
- **Saiba Mais**
  - Monitoramento de requisições via logs estruturados (Flask + biblioteca `logging`)
  - Estrutura típica dos dados capturados (Tabela 1)
  - Inference Tables no Databricks como alternativa gerenciada
  - Data Drift: conceito e exemplo de análise da variável idade (age)
  - A importância de monitorar o Data Drift
  - Monitoramento da Latência do Modelo (percentis P50, P95, P99)
  - Outros itens importantes no monitoramento (Model Drift, KPIs de negócio, viés e equidade/fairness)
  - Por que monitorar modelos em produção (training-serving skew, drift)
  - Que métricas acompanhar (qualidade preditiva, data drift, negócio, serviço)
  - Ferramentas de monitoramento e alertas (Prometheus & Grafana, MLflow, plataformas de terceiros, logging estruturado, alertas e automação)
  - Model Registry e gestão de versões
  - Estratégias de manutenção contínua (agendamento periódico, gatilho por performance, gatilho por drift, atualização contínua/online, retraining on demand)
- **Mercado, Cases e Tendências** — lições práticas de casos reais (pandemia, drift lento em ML médico, falhas de dados/pipelines, estratégias de defesa da Netflix e Uber, ferramentas emergentes, COVID-19 como catalisador, tendências futuras).
- **O que você viu nesta aula?** — síntese do conteúdo.
- **Referências** — bibliografia consultada.
- **Palavras-chave** — termos centrais da aula.

## Conceitos-Chave

### Data Drift
Mudança nas propriedades estatísticas dos dados de entrada (variáveis preditoras) ou na relação entre entrada e alvo, ao longo do tempo, após o modelo ter sido treinado e implantado. Quando ocorre, o ambiente de produção deixa de se assemelhar ao ambiente de treino, invalidando as premissas do modelo e levando à degradação de desempenho. É subdividido em **drift de atributos/features**: a distribuição dos dados de entrada muda, mas a relação entre entrada e saída permanece a mesma (ex.: exames vindos de um novo equipamento com características diferentes).

### Model Drift (Performance Drift)
Ocorre quando a relação fundamental entre entradas e alvo predito muda — ou seja, a própria definição do que se está tentando prever evolui (ex.: fraudadores adotando novas táticas). Mesmo sem mudança na distribuição das features, o significado dos rótulos ou a função-alvo mudou, invalidando parcialmente o aprendizado original do modelo. A chave para monitorar a qualidade preditiva é a coleta de **ground truth** (dado real) após a inferência, que em muitos cenários chega com atraso (horas, dias ou semanas).

### Training-Serving Skew
Diferenças significativas entre as condições dos dados de treinamento e os dados reais que o modelo vê em produção, levando a resultados inesperados. É um problema correlacionado ao drift e reforça a necessidade de monitoramento contínuo.

### Métricas de Qualidade Preditiva
Métricas que refletem diretamente o desempenho do modelo na tarefa (acurácia, precisão, recall, F1-score). Uma queda significativa nesses indicadores ao longo do tempo é o sinal mais direto de model drift. Requerem dados com rótulo real (ground truth), que às vezes chegam com atraso.

### Métricas de Data Drift
Indicadores estatísticos que apontam mudança de distribuição nos dados de entrada ou saída, mesmo sem rótulos disponíveis. Incluem a comparação de distribuições via testes estatísticos como **Kolmogorov-Smirnov**, **testes qui-quadrado** ou métricas de distância como **Wasserstein** e **PSI (Population Stability Index)**. Também cobrem o **prediction drift** — mudança na distribuição das predições do modelo (ex.: proporção de "alto risco" saltando de 10% para 30%).

### Métricas de Negócio (uso/engajamento)
Indicadores de uso que, embora não sejam performance técnica direta, correlacionam-se com a qualidade do modelo — por exemplo, taxa de cliques/conversão em recomendações, proporção de positivos previstos, ou aparição de categorias inéditas nas variáveis de entrada.

### Métricas de Serviço (latência, throughput, erros)
Métricas de observabilidade tradicionais de software aplicadas ao modelo em produção: tempo de resposta médio e por percentil, throughput (número de requisições atendidas), taxas de erro HTTP (500, timeout) e uso de memória/CPU. Complementam as métricas de acurácia garantindo disponibilidade e estabilidade do serviço.

### Percentis de Latência (P50, P95, P99)
O **P50** (mediana) mostra o tempo de resposta típico, sendo menos sensível a outliers que a média. As **métricas de cauda P95 e P99** são indispensáveis: o P95 indica o tempo máximo que 95% das requisições atingem, e o P99, o "pior caso" para 99% das requisições. Um aumento repentino no P99 pode sinalizar problema de alocação de recursos e ajuda a garantir SLAs (Acordos de Nível de Serviço).

### Monitoramento de Viés e Equidade (Fairness)
Em contexto de regulamentação e responsabilidade ética, é obrigatório monitorar se o modelo toma decisões justas e imparciais, analisando métricas de desempenho (acurácia, F1-score) desagregadas por grupos sensíveis (gênero, etnia, localização) e detectando desigualdade nas predições entre subgrupos.

### Logging Estruturado
Registro detalhado das predições e dados de entrada de cada requisição de inferência (payload de entrada, resposta, latência, status HTTP), essencial para monitoramento de desempenho e qualidade, debugging/auditoria e refinamento/retreinamento de modelos. Serve como registro histórico imutável e permite investigar a fundo problemas específicos (ex.: erros concentrados em um hospital ou intervalo de datas).

### Prometheus & Grafana
Dupla amplamente usada em observabilidade. O **Prometheus** é um toolkit open-source de monitoramento e time-series database que funciona por modelo *pull*, consultando periodicamente endpoints expostos pelos serviços (ex.: `/metrics` via `prometheus_client` em Flask). O **Grafana** conecta-se ao Prometheus como front-end de visualização e alerta, permitindo dashboards, gráficos e notificações (e-mail, Slack) quando métricas ultrapassam limites definidos.

### MLflow Tracking e Model Registry
O **MLflow Tracking** loga parâmetros, métricas e artefatos, podendo ser adaptado para jobs periódicos de avaliação pós-deployment. O **Model Registry** é um repositório central de modelos com controle de versões e estágios ("Staging", "Production", "Archived"), permitindo comparar versões, promover modelos com governança, realizar A/B testing/shadow deployment (alias Champion/Challenger) e fazer rollback rápido em caso de problemas.

### Inference Tables (Databricks)
Recurso que automatiza a coleta de dados de entrada, saída e metadados de latência em endpoints de Model Serving, persistindo-os no Delta Lake com esquema padronizado, eliminando a necessidade de instrumentação manual de logging.

### Alertas e Automação
Configuração de gatilhos automáticos para notificar responsáveis quando métricas ultrapassam limites (ex.: acurácia < 0,8, ou desvio estatístico significativo por dias seguidos). É importante calibrar bem os thresholds para evitar fadiga de alertas, começando por um período de observação manual da variabilidade normal.

### Estratégias de Manutenção Contínua
- **Agendamento periódico**: retreinar em intervalos fixos (semanal, mensal, trimestral), ajustado à dinâmica do problema.
- **Gatilho por degradação de performance**: retreinar quando a métrica de qualidade cai abaixo de um limiar.
- **Gatilho por data drift**: retreinar quando a distribuição dos dados/predições muda significativamente, mesmo sem confirmação de queda de acurácia.
- **Atualização contínua/online**: incorporar novos dados incrementalmente, de forma quase contínua.
- **Retraining on demand**: retreino manual disparado por eventos extraordinários ou descoberta de problemas (ex.: viés identificado).

## Exercício Hands-On (do material)

O hands on desta aula dá sequência ao caso da aula anterior, em que foi implantado um modelo de detecção de doenças cardíacas usando um endpoint via REST API. O foco agora é implementar um processo de monitoramento de métricas desse modelo em produção e definir critérios para retreino quando necessário — ou seja, instrumentar o modelo implantado para coletar indicadores de performance ao longo do tempo e reagir quando esses indicadores apontarem degradação.

A prática envolve adaptar a aplicação Flask para registrar métricas em tempo real durante as inferências: a cada requisição, são capturados o payload de entrada, a latência (tempo de resposta) e métricas personalizadas sobre a distribuição dos valores de entrada, permitindo identificar proativamente alterações nos padrões dos dados que chegam ao modelo. Isso é feito através de logs estruturados coletados diretamente da aplicação Flask API, usando a biblioteca padrão `logging` do Python configurada com um `FileHandler` que grava os registros em formato JSON (campos: `timestamp`, `request_id`, `request_payload`, `response_payload`, `status_code`, `latency_ms`). Com esses dados coletados, o material demonstra em seguida como analisar se a distribuição das features de produção (ex.: a variável `age`) se desviou da distribuição observada no treinamento (baseline), configurando um cenário concreto de Data Drift.

## Exemplos de Código

A seguir, exemplos ilustrativos (não extraídos literalmente do PDF, exceto onde indicado) que operacionalizam os conceitos apresentados na aula.

Configuração de logging estruturado na aplicação Flask, conforme o Código-fonte 1 do material, que grava cada requisição de inferência em formato JSON para posterior análise de drift, debugging e retreinamento:

```python
import os
import logging

LOG_ENV_VAR = "REQUEST_LOG_PATH"
DEFAULT_LOG_PATH = "logs/requests.log"

# -------------------------------------------------
# Configuração de logging estruturado
# -------------------------------------------------
logger = logging.getLogger("request_logger")
logger.setLevel(logging.INFO)
_log_path = os.getenv(LOG_ENV_VAR, DEFAULT_LOG_PATH)
if not logger.handlers:
    fh = logging.FileHandler(_log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
```

Registro de uma requisição de inferência em formato JSON dentro do endpoint Flask, capturando payload, resposta, status e latência — a estrutura de dados descrita na Tabela 1 do material:

```python
import json
import time
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    request_id = str(uuid.uuid4())
    payload = request.get_json()

    # (chamada real ao modelo de detecção de doença cardíaca)
    prediction = model.predict([list(payload.values())])
    probability = model.predict_proba([list(payload.values())])[0][1]

    response_payload = {
        "predictions": prediction.tolist(),
        "probabilities": [probability],
    }
    latency_ms = (time.time() - start) * 1000

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "request_payload": payload,
        "response_payload": response_payload,
        "status_code": 200,
        "latency_ms": round(latency_ms, 3),
    }
    logger.info(json.dumps(log_entry))

    return jsonify(response_payload), 200
```

Cálculo de Data Drift comparando a distribuição de uma feature em produção contra o baseline de treino, usando o teste de Kolmogorov-Smirnov e o PSI (Population Stability Index), ambos citados no material como métodos estatísticos para detectar mudança de distribuição:

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def calcular_ks(baseline: pd.Series, producao: pd.Series, alpha: float = 0.05):
    """Teste de Kolmogorov-Smirnov entre distribuição de treino e de produção."""
    estatistica, p_valor = ks_2samp(baseline.dropna(), producao.dropna())
    drift_detectado = p_valor < alpha
    return {"ks_statistic": estatistica, "p_valor": p_valor, "drift_detectado": drift_detectado}


def calcular_psi(baseline: pd.Series, producao: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (PSI) entre duas distribuições."""
    quantis = np.linspace(0, 1, bins + 1)
    limites = np.unique(baseline.quantile(quantis))

    freq_baseline = np.histogram(baseline, bins=limites)[0] / len(baseline)
    freq_producao = np.histogram(producao, bins=limites)[0] / len(producao)

    # evita divisão por zero / log(0)
    freq_baseline = np.where(freq_baseline == 0, 1e-6, freq_baseline)
    freq_producao = np.where(freq_producao == 0, 1e-6, freq_producao)

    psi = np.sum((freq_producao - freq_baseline) * np.log(freq_producao / freq_baseline))
    return float(psi)


# Exemplo: variável "age" (idade), citada na Figura 2 do material
resultado_ks = calcular_ks(dados_treino["age"], dados_producao["age"])
psi_age = calcular_psi(dados_treino["age"], dados_producao["age"])

print(resultado_ks)
print(f"PSI (age): {psi_age:.3f}")
# Regra prática de mercado: PSI < 0.1 sem drift relevante;
# 0.1-0.25 drift moderado; > 0.25 drift significativo (retreino recomendado)
```

Exposição de métricas de inferência no formato Prometheus a partir da aplicação Flask, como descrito na seção sobre Prometheus & Grafana (contador `total_predictions` e gauge `model_accuracy`):

```python
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from flask import Response

total_predictions = Counter(
    "total_predictions", "Número total de predições realizadas"
)
model_accuracy = Gauge(
    "model_accuracy", "Acurácia mais recente do modelo em produção"
)
latency_histogram = Histogram(
    "inference_latency_ms", "Latência das inferências em milissegundos"
)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# Dentro do endpoint /predict, após cada inferência:
total_predictions.inc()
latency_histogram.observe(latency_ms)

# Em um job periódico que reconcilia predições com ground truth:
model_accuracy.set(acuracia_calculada)
```

Job periódico que calcula métricas de qualidade preditiva a partir de ground truth reconciliado, conforme descrito na seção "Que métricas acompanhar", e dispara um gatilho de retreinamento quando a acurácia cai abaixo do limiar definido:

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LIMIAR_ACURACIA = 0.85

def avaliar_e_decidir_retreino(y_true, y_pred) -> dict:
    metricas = {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    metricas["acionar_retreino"] = metricas["acuracia"] < LIMIAR_ACURACIA
    return metricas


# Exemplo de uso em um job diário (ex.: agendado via Airflow)
resultado = avaliar_e_decidir_retreino(y_true=ground_truth_recente, y_pred=predicoes_recentes)

if resultado["acionar_retreino"]:
    print("Acurácia abaixo do limiar — disparando pipeline de retreinamento.")
    # trigger_retraining_pipeline()
```

Configuração de alertas no Prometheus (regra de alerta) e Alertmanager, ilustrando os exemplos de threshold citados no material ("se a acurácia calculada no job diário for menor que 0.8, enviar e-mail" e "se o desvio ultrapassar significativamente por dias seguidos, gerar alerta"):

```yaml
# prometheus_alert_rules.yml
groups:
  - name: ml_model_monitoring
    rules:
      - alert: AcuraciaModeloAbaixoDoLimiar
        expr: model_accuracy < 0.80
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Acurácia do modelo de doença cardíaca abaixo de 80%"
          description: "model_accuracy = {{ $value }} nos últimos 15 minutos."

      - alert: LatenciaP99Alta
        expr: histogram_quantile(0.99, rate(inference_latency_ms_bucket[5m])) > 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 de latência acima do SLA (50ms)"

      - alert: DataDriftDetectado
        expr: data_drift_psi_score > 0.25
        for: 3d
        labels:
          severity: warning
        annotations:
          summary: "PSI indica drift significativo persistente há 3 dias"
```

Estrutura de um pipeline de retreinamento automatizado (orquestração conceitual, como as ferramentas Airflow/Kubeflow Pipelines citadas no material), incluindo etapa de validação antes de promover o modelo no Model Registry:

```python
import mlflow
from mlflow.tracking import MlflowClient

def pipeline_retreinamento(nome_modelo: str = "HeartDiseaseModel"):
    # 1. Ingestão e pré-processamento de dados novos
    X_train, y_train, X_val, y_val = carregar_dados_atualizados()

    # 2. Treinamento
    with mlflow.start_run(run_name="retrain_scheduled") as run:
        modelo = treinar_modelo(X_train, y_train)

        # 3. Avaliação comparativa com o modelo em produção
        metricas_novo = avaliar_e_decidir_retreino(y_val, modelo.predict(X_val))
        mlflow.log_metrics({k: v for k, v in metricas_novo.items() if isinstance(v, float)})

        client = MlflowClient()
        versao_producao = client.get_model_version_by_alias(nome_modelo, "champion")
        acuracia_producao = float(versao_producao.tags.get("accuracy", 0))

        # 4. Validação: só promove se o novo modelo for melhor
        if metricas_novo["acuracia"] > acuracia_producao:
            mlflow.sklearn.log_model(modelo, "model", registered_model_name=nome_modelo)
            nova_versao = client.get_latest_versions(nome_modelo, stages=["None"])[0]
            client.set_registered_model_alias(nome_modelo, "challenger", nova_versao.version)
            print(f"Nova versão {nova_versao.version} registrada como challenger.")
        else:
            print("Modelo novo não superou o modelo em produção — retreino descartado.")

pipeline_retreinamento()
```

## Cases e Tendências de Mercado

- **Pandemia e mudanças bruscas**: modelos de demanda em companhias aéreas, risco de crédito em bancos e manutenção preditiva em fábricas tiveram que ser rapidamente refeitos durante a COVID-19, quando padrões históricos perderam relevância. Lição: projetar pipelines de retreino ágeis e manter reservas de capacidade para retreinamento emergencial.
- **Drift lento e imperceptível**: estudo de 2023 na área de saúde mostrou que modelos de diagnóstico por imagem perdem acurácia com o tempo devido a mudanças sutis em equipamentos e população de pacientes — reforça a necessidade de monitoramento contínuo mesmo quando "tudo parece bem", já que quedas graduais (ex.: 0,1% por semana) se acumulam.
- **Falhas por falha nos dados ou pipelines**: nem toda queda de performance é drift genuíno — pode ser bug. Empresas como Uber e Airbnb usam camadas de validação de dados em tempo real (ex.: Great Expectations) para evitar que o modelo consuma dados ruins.
- **Estratégias de defesa**: a Netflix monitora métricas online (cliques, retenção), feature drift e qualidade de recomendação, além de praticar canary releases (lançamento gradual com comparação de métricas antes de liberar 100% do tráfego). A Uber mantém camadas de redundância nos pipelines de ML (ex.: fórmula básica de distância como fallback se o modelo de ETA falhar).
- **Ferramentas e plataformas emergentes**: Arize AI e Fiddler AI oferecem monitoramento de data drift, performance e explicabilidade para setores financeiro e de tecnologia; Evidently AI (open source) tem adoção crescente; a Microsoft anunciou em 2023 o Azure ML Model Monitor, unificando drift e métricas customizadas com alertas via Application Insights.
- **COVID-19 como catalisador**: o artigo "How Instacart's Item Availability Evolved Over the Pandemic" descreve como o time lidou com mudanças abruptas de demanda e estoque via novas arquiteturas e limiares adaptativos — pós-2020 houve maior exigência de planos de contingência para modelos críticos.
- **Tendências futuras**: maior integração entre monitoramento automatizado e ação autônoma (auto-reconfiguração de modelos, pipelines de retreino/validação com aprovação humana mínima, alinhado ao conceito de AutoML contínuo); expansão do escopo de monitoramento para LLMs (data drift e alucinações), com ferramentas emergentes como LangChain e MLflow 3.0; consolidação da Observabilidade de ML como disciplina própria; pressão regulatória (ex.: AI Act da União Europeia) exigindo monitoramento, log de decisões e planos de atualização para sistemas de IA de alto risco.

## Checklist de Estudo

- [ ] Sei diferenciar Data Drift de Model Drift (drift de atributos vs. mudança na relação entre entrada e alvo).
- [ ] Sei explicar o que é training-serving skew e por que ele é um problema correlacionado ao drift.
- [ ] Sei descrever os quatro grupos de métricas de monitoramento: qualidade preditiva, data drift, negócio/uso e serviço.
- [ ] Sei explicar por que os percentis P95 e P99 de latência são mais informativos que a média para SLAs.
- [ ] Sei aplicar testes estatísticos (Kolmogorov-Smirnov) e métricas de distância (PSI, Wasserstein) para detectar drift.
- [ ] Sei descrever o papel do Prometheus, do Grafana e do MLflow Model Registry no monitoramento e na gestão de versões de modelos.
- [ ] Sei comparar as diferentes estratégias de retreinamento (periódico, por gatilho de performance, por gatilho de drift, online, on demand) e quando aplicar cada uma.

## Palavras-chave

Monitoramento. Manutenção. Data Drift.

## Referências

- AZURE ML DOCUMENTATION. Monitor data drift. 2025. Disponível em: https://learn.microsoft.com. Acesso em: 28 jan. 2026.
- BRAINFOGE. How Netflix Uses ML to Create Recommendations. 2023. Disponível em: https://www.brainforge.ai/blog/how-netflix-uses-machine-learning-ml-to-create-perfect-recommendations. Acesso em: 28 jan. 2026.
- DATACAMP. Grafana Tutorial: A Beginner's Guide to Monitoring Machine Learning Models. 2023. Disponível em: https://www.datacamp.com/tutorial/grafana-tutorial-monitoring-machine-learning-models. Acesso em: 28 jan. 2026.
- DATADOG. Machine learning model monitoring: Best practices. 2024. Disponível em: https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/. Acesso em: 28 jan. 2026.
- EVIDENTLY AI. Model monitoring for ML in production: a comprehensive guide. 2025. Disponível em: https://www.evidentlyai.com/ml-in-production/model-monitoring. Acesso em: 28 jan. 2026.
- GOUABOU, A. Monitoring ML Model Performance with Prometheus & Grafana. 2025. Disponível em: https://medium.com/@cartelgouabou/step-by-step-guide-monitoring-ml-model-performance-with-prometheus-grafana-88195f741365. Acesso em: 28 jan. 2026.
- IBM DOCUMENTATION. Collecting telemetry data using Prometheus – Grafana. 2025. Disponível em: https://www.ibm.com/docs/en/watsonxdata/standard/2.1.x?topic=v211-collecting-telemetry-data-using-prometheus-grafana. Acesso em: 28 jan. 2026.
- INSTACART – TECH BLOG. How Instacart's Item Availability Evolved Over the Pandemic. 2023. Disponível em: https://www.instacart.com/company/tech-innovation/how-instacarts-item-availability-evolved-over-the-pandemic. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. MLflow Model Registry. 2025. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- NEPTUNE.AI. Retraining Model During Deployment: Continuous Training and Continuous Testing. 2022. Disponível em: https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing. Acesso em: 28 jan. 2026.
- NETFLIX TECH BLOG. ML Observability: Bring Transparency to Payments and Beyond. s.d. Disponível em: https://netflixtechblog.com/ml-observability-bring-transparency-to-payments-and-beyond-33073e260a38. Acesso em: 28 jan. 2026.
- QWAK. Top 7 ML Model Monitoring Tools in 2024. 2024. Disponível em: https://www.qwak.com/post/top-ml-model-monitoring-tools. Acesso em: 28 jan. 2026.
- SAHINER, B.; CHEN, W.; SAMALA, R. K.; PETRICK, N. Data drift in medical machine learning: implications and potential remedies. 2023. Disponível em: https://pmc.ncbi.nlm.nih.gov. Acesso em: 28 jan. 2026.
