# Aula 07 — Deploy e Monitoramento de Código

## Visão Geral
Aula da disciplina de Engenharia de Software para Cientista de Dados (Fase 3) focada na transição do modelo do notebook para produção. O fio condutor é a automação: **CI** valida e empacota o código, **CD** promove versões com segurança, e a **observabilidade** (traces, métricas, logs) conecta o comportamento do serviço às métricas de qualidade do modelo. A aula constrói uma esteira mínima viável com GitHub Actions, Docker e FastAPI, discute estratégias de release de baixo risco (blue-green e canary) e fecha o ciclo com instrumentação via OpenTelemetry, Prometheus e detecção de drift com Evidently AI.

## Tópicos Abordados
- Transição de protótipo para produção guiada por pipelines de CI/CD reprodutíveis (testes, lint, build, rastreabilidade)
- Modos de deploy: batch (alto throughput) vs. online via API (baixa latência)
- Estratégias de lançamento de baixo risco: blue-green e canary releases
- Hands On: API FastAPI servindo modelo scikit-learn, Dockerfile e workflow `ci-cd.yml` com GitHub Actions + Azure
- Saiba Mais: pipelines de CI/CD para ML, frameworks de serving, arquitetura de referência, SLOs e Golden Signals
- Observabilidade para ML: instrumentação com OpenTelemetry, métricas com Prometheus, integração com Azure Monitor/App Insights
- Data drift vs. concept drift: definições formais e testes (PSI, KL, KS)
- Mapeamento dos Golden Signals de SRE para KPIs de ML
- Mercado, cases e tendências em MLOps/CI-CD

## Conceitos-Chave

### Pipelines de CI/CD para ML
O CI/CD em ML amplia o ciclo tradicional de software com testes estatísticos, controle de dependências e versionamento de artifacts (dados, modelos, imagens). No **CI**, além de lint, testes unitários e checagens de segurança, incluem-se validações de **acurácia mínima** e comparação com baseline (ex.: ΔAUC ≤ 1 p.p. ou ΔF1 ≤ 0,02), de modo que apenas execuções que cumprem o "contrato" de qualidade são promovidas. No **CD**, a infraestrutura é tratada como código e os deployments para staging/produção passam por *gates* automáticos. No ecossistema Azure, há workflows prontos de GitHub Actions para treinar/publicar modelos no Azure Machine Learning, incluindo autenticação OIDC (Entra) e promoção por ambientes.

### Modos de Deploy: Batch vs. Online
- **Batch**: maximiza throughput sob latência relaxada (ETL + scoring em lote); adequado para cenários como scoring diário de milhões de registros.
- **Online (API)**: prioriza p95/p99 baixo, com autoscaling horizontal e warm-up de containers; adequado para sistemas interativos e recomendações em tempo real.
- Frameworks de serving variam da simplicidade do FastAPI (faça-você-mesmo) a servidores especializados com multiversionamento, autoscaling, mTLS e split de tráfego (TensorFlow Serving, TorchServe, BentoML, KServe, Seldon Core), geralmente nativos de Kubernetes.

### Estratégias de Lançamento (Blue-Green e Canary)
- **Blue-green**: mantém dois ambientes idênticos e comuta o roteamento quando o ambiente "green" passa nos testes de fumaça; rollback é instantâneo, mas exige duplicação de infraestrutura.
- **Canary**: envia uma fração progressiva do tráfego (por peso ou headers) para a nova versão, monitorando latência, taxa de erro e saturação antes de chegar a 100%; reduz o impacto ao limitar a exposição inicial, mas exige roteamento dinâmico e maior sofisticação de observabilidade.
- **Trade-off central**: objetivo (throughput vs. baixa latência vs. zero-downtime vs. rollout gradual com métricas), risco (baixo no batch, médio em APIs, mitigado por rollback no blue-green, reduzido por exposição parcial no canary) e complexidade (baixa no batch, crescente até o canary, que exige roteamento dinâmico e automação). Recomendação do material: iniciar com batch para MVPs e evoluir para APIs com canary quando a criticidade do negócio justificar.

### SRE Golden Signals e SLOs aplicados a ML
Em produção, adotam-se os **Golden Signals** de SRE (latência, tráfego, erros, saturação) como SLIs operacionais, com SLOs explícitos (ex.: p99 < 300 ms; error rate < 0,5%). Esses sinais guiam decisões de canary e go/no-go no CD. A aula propõe mapear cada Golden Signal a um KPI de ML e a um exemplo de alerta:

| Golden Signal | KPI de ML associado | Exemplo de alerta |
|---|---|---|
| Latência | p95/p99 do endpoint `/predict` | p99 > 300 ms por 5 min |
| Erros | taxa de 5xx / timeout | error rate > 1% por 10 min |
| Tráfego | RPS/QPS | pico > p95 histórico |
| Saturação | uso de CPU/GPU, memória | GPU mem > 90% por 10 min |
| Qualidade (extra) | AUC/F1 online, PSI | PSI > 0,2 (drift), ΔF1 > 0,03 |

A integração de Golden Signals operacionais com KPIs de qualidade do modelo reduz o MTTD/MTTR (tempo de detecção/recuperação) e dá base objetiva para decisões de canary e rollback.

### Observabilidade: Traces, Métricas e Logs
A tríade traces-métricas-logs descreve o comportamento do serviço; métricas de modelo refletem a qualidade do aprendizado sob a distribuição real de produção — observabilidade para ML não é apenas monitoramento de infraestrutura. **OpenTelemetry (OTel)** padroniza a coleta de forma *vendor-agnostic*; em FastAPI, a instrumentação oficial captura spans HTTP automaticamente e exporta via OTLP para coletores/serviços como Azure Monitor. **Prometheus** deve usar histogramas com buckets calibrados à SLO (em vez de médias) para calcular percentis verdadeiros de latência; bibliotecas como `prometheus-fastapi-instrumentator` aceleram a adoção.

### Data Drift vs. Concept Drift
- **Data drift**: mudança da distribuição de entrada P(X). Detectado com **PSI** — ∑(Aᵢ − Eᵢ)·ln(Aᵢ/Eᵢ) —, **KL** (divergência de Kullback-Leibler, D_KL(P‖Q) = ∑P(x)·log(P(x)/Q(x))) e **KS** (máxima diferença entre as CDFs — teste de Kolmogorov-Smirnov).
- **Concept drift**: mudança da relação P(Y|X), isto é, a própria função que o modelo tenta aprender muda. Requer ground truth atrasado ou proxies, como *prediction drift* ou variação de *feature importance*.
- A ferramenta **Evidently AI** gera relatórios comparando distribuição de produção vs. referência (batch ou near-real-time) e se integra a pipelines de orquestração.

## Exercício Hands-On (do material)
Construção de uma esteira mínima viável que empacota um modelo scikit-learn em uma API FastAPI, containerizada com Docker, testada com Pytest e publicada automaticamente via GitHub Actions até o Azure Container Apps:

1. **API de serving** (`app/main.py`) — endpoint `/predict` que recebe três features via Pydantic e retorna a predição do modelo carregado com `joblib`.
2. **Dockerfile** — imagem baseada em `python:3.11-slim`, instala dependências, copia a API e o modelo serializado, expõe a porta 8000 e sobe com `uvicorn`.
3. **Workflow `ci-cd.yml`** — job `build-test` (checkout, setup Python 3.11, `pytest`, build e push da imagem para o Azure Container Registry) seguido do job `deploy` (login no Azure via OIDC e `az containerapp update` com a nova tag de imagem baseada no SHA do commit).
4. **Instrumentação de observabilidade** — módulo `obs/otel_setup.py` configurando um `TracerProvider` do OpenTelemetry com `FastAPIInstrumentor`; módulo `obs/metrics.py` expondo métricas Prometheus (contador de requisições e histograma de latência) via middleware montado em `/metrics`.
5. **Detecção de drift** — script `drift/check_drift.py` usando Evidently (`Report` com `DataDriftPreset`) para comparar dados de referência vs. dados atuais e gerar um relatório HTML + um JSON de status.

## Exemplos de Código

API FastAPI servindo o modelo:

```python
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Modelo Demo - FastAPI")

class Features(BaseModel):
    x1: float; x2: float; x3: float

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(f: Features):
    X = np.array([[f.x1, f.x2, f.x3]])
    y = model.predict(X)[0]
    return {"prediction": float(y)}
```

Dockerfile da aplicação:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY model.pkl .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Workflow de CI/CD com GitHub Actions (build, teste, push da imagem e deploy no Azure):

```yaml
# .github/workflows/ci-cd.yml
name: ci-cd
on: [push]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt && pytest -q
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ secrets.ACR_LOGIN_SERVER }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ secrets.ACR_LOGIN_SERVER }}/fastapi-ml:$(echo $GITHUB_SHA | cut -c1-7)
  deploy:
    needs: build-test
    runs-on: ubuntu-latest
    steps:
      - uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
      - uses: azure/cli@v2
        with:
          inlineScript: |
            az containerapp update \
              --name ml-api --resource-group rg-ml \
              --image ${{ secrets.ACR_LOGIN_SERVER }}/fastapi-ml:$(echo $GITHUB_SHA | cut -c1-7)
```

Instrumentação com OpenTelemetry:

```python
# obs/otel_setup.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_otel(app, service_name="fastapi-ml"):
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
```

Métricas Prometheus (contador de requisições + latência por rota):

```python
# obs/metrics.py
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app

REQS = Counter("http_requests_total", "Total de requisições", ["method", "path", "code"])
LAT = Histogram("http_request_latency_seconds", "Latência por rota", ["path"])

def setup_metrics(app: FastAPI):
    @app.middleware("http")
    async def _mw(request: Request, call_next):
        path = request.url.path
        with LAT.labels(path).time():
            resp = await call_next(request)
        REQS.labels(request.method, path, resp.status_code).inc()
        return resp

    app.mount("/metrics", make_asgi_app())
```

Detecção de drift com Evidently:

```python
# drift/check_drift.py
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

ref = pd.read_parquet("data/ref.parquet")
cur = pd.read_parquet("data/current.parquet")

r = Report(metrics=[DataDriftPreset()])
r.run(reference_data=ref, current_data=cur)
r.save_html("reports/drift.html")
open("reports/drift.json", "w").write(r.as_dict().__repr__())
```

## Cases e Tendências de Mercado
- **Tendências em MLOps e CI/CD para 2025**: artigos do setor reportam ganhos concretos ao adotar pipelines de CI/CD com estratégias blue-green/canary, monitoramento via Prometheus/Grafana e detecção de drift com Evidently AI — redução de downtime em até 90% e aceleração de deploys em 60–80%.
- **Case prático: CI/CD com Azure Databricks** — documentação oficial detalha a integração de pipelines de CI/CD em projetos de ML combinando DataOps, ModelOps e DevOps, com automação de testes, monitoramento de qualidade de dados e infraestrutura como código.
- **Blue-Green e Canary Releases** — comparativos de mercado reforçam os benefícios de rollback imediato (blue-green) e rollout gradual com métricas (canary), com guias práticos de implementação em Kubernetes/Istio.
- Comparativos entre servidores de modelo especializados (BentoML, KServe, Seldon Core) mostram diferentes níveis de maturidade para cenários Kubernetes-nativos e abordagens Python-first.

## Checklist de Estudo
- [ ] Sei explicar a diferença entre CI e CD aplicados especificamente a Machine Learning (testes estatísticos, baseline, artifacts)
- [ ] Sei diferenciar deploy batch de deploy online e escolher o modo adequado ao caso de negócio
- [ ] Sei explicar as estratégias blue-green e canary, seus riscos e níveis de complexidade
- [ ] Sei montar uma esteira mínima de CI/CD com GitHub Actions, Docker e FastAPI para servir um modelo
- [ ] Sei instrumentar uma API com OpenTelemetry e expor métricas Prometheus (incluindo histogramas de latência)
- [ ] Sei mapear os Golden Signals de SRE (latência, erros, tráfego, saturação) para KPIs de ML
- [ ] Sei diferenciar data drift de concept drift e conheço as métricas PSI, KL e KS
- [ ] Sei usar o Evidently AI para gerar relatórios de drift comparando dados de referência e produção
- [ ] Sei definir SLOs práticos (ex.: p99 < 300 ms, error rate < 0,5%) e usá-los como gate de promoção em canary

## Palavras-chave
MLOps · Continuous Delivery · SRE

## Referências
- CODEZUP. *Implementing Canaries and Blue-Green Deployments in Kubernetes*. 2024. Disponível em: https://codezup.com/implementing-canaries-and-blue-green-deployments-in-kubernetes/. Acesso em: 16 dez. 2025.
- EVIDENTLY AI. *Model monitoring for ML in production: a comprehensive guide*. 2025. Disponível em: https://www.evidentlyai.com/ml-in-production/model-monitoring. Acesso em: 16 dez. 2025.
- EVIDENTLY AI. *Overview*. 2025. Disponível em: https://docs.evidentlyai.com/docs/platform/monitoring_overview. Acesso em: 16 dez. 2025.
- EWASCHUK, R. *Monitoring Distributed Systems*. 2025. Disponível em: https://sre.google/sre-book/monitoring-distributed-systems/. Acesso em: 16 dez. 2025.
- GITHUB. *Azure-Samples /opencensus-with-fastapi-and-azure-monitor*. 2025. Disponível em: https://github.com/Azure-Samples/opencensus-with-fastapi-and-azure-monitor. Acesso em: 16 dez. 2025.
- GITHUB. *FastAPI + Gunicorn*. 2024. Disponível em: http://prometheus.github.io/client_python/exporting/http/fastapi-gunicorn/. Acesso em: 16 dez. 2025.
- GITHUB. *trallnag /prometheus-fastapi-instrumentator*. 2025. Disponível em: https://github.com/trallnag/prometheus-fastapi-instrumentator. Acesso em: 16 dez. 2025.
- LEARN. *MLOps (operações de machine learning) de ponta a ponta com o Azure Machine Learning*. 2025. Disponível em: https://learn.microsoft.com/pt-br/training/paths/build-first-machine-operations-workflow/. Acesso em: 16 dez. 2025.
- MICROSOFT. *Use GitHub Actions with Azure Machine Learning*. 2025. Disponível em: https://learn.microsoft.com/.../how-to-github-actions-machine-learning. Acesso em: 16 dez. 2025.
- OPEN TELEMETRY. *Getting Started*. 2025. Disponível em: https://opentelemetry.io/docs/languages/python/getting-started/. Acesso em: 16 dez. 2025.
- PRIYA, B. *Step-by-Step Guide to Deploying Machine Learning Models with FastAPI and Docker*. 2025. Disponível em: https://machinelearningmastery.com/step-by-step-guide-to-deploying-machine-learning-models-with-fastapi-and-docker/. Acesso em: 16 dez. 2025.
- SOURCE FORGE. *BentoML vs. KServe vs. Seldon Comparison Chart*. 2025. Disponível em: https://sourceforge.net/software/compare/BentoML-vs-KServe-vs-Seldon/. Acesso em: 16 dez. 2025.
- VITHLANI, N. *Implementing Blue-Green Deployments with Kubernetes and Istio*. 2025. Disponível em: https://talent500.com/blog/blue-green-deployments-kubernetes-istio/. Acesso em: 16 dez. 2025.
