# house-prices-ml

Projeto de estudo que implementa um **ciclo de vida completo de Machine Learning**
— da ciência de dados até uma API em produção — para prever o preço de venda de
imóveis em King County (região de Seattle, EUA), usando o
[House Pricing Dataset](https://www.kaggle.com/datasets/alyelbadry/house-pricing-dataset)
do Kaggle (21.613 vendas registradas entre 2014 e 2015).

Este projeto acompanha o conteúdo da Fase 1 do curso (ver `../Fase1/aula08_projeto_final_ciclo_completo/`),
implementando de forma enxuta as etapas centrais do ciclo: estruturação de projeto,
limpeza de dados, pipeline de treino reprodutível, API de serving e testes
automatizados em CI. Itens mais avançados do material da aula 8 (MLflow, Fairlearn,
Docker, simulação de drift) ficam descritos como próximos passos ao final deste
README, mas não fazem parte do escopo implementado aqui.

## Índice

- [Por que esse projeto existe (o problema de negócio)](#por-que-esse-projeto-existe-o-problema-de-negócio)
- [Estrutura do repositório](#estrutura-do-repositório)
- [O ciclo de vida de ML, passo a passo](#o-ciclo-de-vida-de-ml-passo-a-passo)
- [Setup](#setup)
- [Como executar cada etapa](#como-executar-cada-etapa)
- [Testes automatizados e CI](#testes-automatizados-e-ci)
- [Decisões de projeto (o "porquê")](#decisões-de-projeto-o-porquê)
- [Próximos passos possíveis](#próximos-passos-possíveis)

## Por que esse projeto existe (o problema de negócio)

Um notebook que baixa dados, limpa, treina um modelo e imprime uma métrica é um bom
ponto de partida, mas é um artefato frágil: ninguém mais consegue rodá-lo de forma
confiável, não há como saber se uma mudança quebrou algo, e não existe forma de
disponibilizar o modelo para outro sistema consumir. Este projeto pega exatamente
esse fluxo e o transforma em um **serviço**: código versionado e testado,
separado em módulos com responsabilidade única, com uma API HTTP na frente e um
pipeline de CI que impede que uma quebra chegue à branch principal sem ser notada.

## Estrutura do repositório

```
house-prices-ml/
├── data/
│   ├── raw/            # dataset bruto baixado do Kaggle (gitignored)
│   ├── processed/       # dataset limpo, pronto para treino (gitignored)
│   └── sample/           # amostra pequena (300 linhas) VERSIONADA no Git,
│                          # usada por testes e pelo CI (não depende do Kaggle)
├── docs/
│   ├── pipeline.md       # decisões de arquitetura do pipeline de ML
│   └── api.md            # contrato da API + exemplos de curl
├── models/               # modelo treinado (.joblib) + metrics.json (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb       # análise exploratória (já executado, com gráficos)
│   └── 02_modeling.ipynb  # comparação de modelos + importância de features
├── src/house_prices/
│   ├── config.py          # caminhos e constantes centralizadas
│   ├── data/
│   │   ├── download.py    # baixa o dataset via kagglehub -> data/raw/
│   │   └── clean.py       # limpeza estrutural -> data/processed/
│   ├── features.py        # engenharia de features + sklearn Pipeline
│   ├── train.py            # treina o modelo, salva model.joblib + metrics.json
│   ├── predict.py          # carrega o modelo e faz predições
│   └── api/
│       ├── app.py          # API Flask (/health, /predict)
│       └── schemas.py      # validação do payload de entrada
├── tests/                 # testes unitários e de integração (pytest)
├── pyproject.toml         # dependências e configuração de lint/testes
└── README.md
```

O workflow de CI vive em `../.github/workflows/ci.yml` (na raiz do repositório
Git — é onde o GitHub Actions espera encontrá-lo).

## O ciclo de vida de ML, passo a passo

| Etapa | Onde mora | O que faz |
|---|---|---|
| 1. Entendimento dos dados | `notebooks/01_eda.ipynb` | Explora distribuições, outliers, nulos, correlações |
| 2. Limpeza de dados | `src/house_prices/data/clean.py` | Deduplica, remove outliers, ajusta tipos |
| 3. Engenharia de features | `src/house_prices/features.py` | Features derivadas + encoding, dentro de um `sklearn.Pipeline` |
| 4. Treino e avaliação | `src/house_prices/train.py` | Treina `RandomForestRegressor`, calcula MAE/RMSE/R², salva artefatos |
| 5. Comparação de modelos | `notebooks/02_modeling.ipynb` | Baseline vs. regressão linear vs. random forest |
| 6. Serving (API) | `src/house_prices/api/app.py` | Expõe `/predict` e `/health` via Flask |
| 7. Testes automatizados | `tests/` | Testes de dados, de features, de treino e da API |
| 8. CI/CD | `.github/workflows/ci.yml` | Lint + testes + treino ponta-a-ponta a cada push/PR |

## Setup

Pré-requisitos: Python 3.11+.

```bash
cd house-prices-ml

# 1. Criar e ativar um ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash) / Linux / macOS:
source .venv/Scripts/activate   # ou .venv/bin/activate no Linux/macOS

# 2. Instalar o projeto em modo editável, com as dependências de desenvolvimento
pip install -e ".[dev]"

# (opcional) dependências extras só para rodar os notebooks
pip install -e ".[notebooks]"
```

Isso instala o pacote `house_prices` (de `src/house_prices`) de forma editável —
ou seja, qualquer alteração no código em `src/` é refletida imediatamente sem
reinstalar, e os comandos `python -m house_prices.<algo>` funcionam de qualquer
diretório.

## Como executar cada etapa

### 1. Baixar os dados brutos

O dataset **não** está versionado no Git (ver [.gitignore](../.gitignore) — dados
são grandes e reproduzíveis, não pertencem ao histórico do código). Baixe-o via
[kagglehub](https://github.com/Kaggle/kagglehub) (é necessário ter uma conta
Kaggle e as credenciais configuradas — veja a documentação do kagglehub caso seja
a primeira vez):

```bash
python -m house_prices.data.download
```

Isso salva o CSV em `data/raw/house_prices.csv`.

### 2. Limpar os dados

```bash
python -m house_prices.data.clean
```

Gera `data/processed/house_prices_processed.csv` — remove duplicatas de imóveis
vendidos mais de uma vez (mantendo a venda mais recente), remove um outlier
conhecido (uma linha com 33 quartos, erro de digitação) e converte a data de venda
em `sale_year`. Detalhes e justificativas em `docs/pipeline.md` e nos comentários
de `src/house_prices/data/clean.py`.

### 3. Treinar o modelo

```bash
python -m house_prices.train
# ou, para ajustar hiperparâmetros:
python -m house_prices.train --n-estimators 300 --max-depth 15
```

Treina um `RandomForestRegressor` dentro do pipeline de features, avalia num
conjunto de teste (20% dos dados, `random_state` fixo para reprodutibilidade) e
salva:
- `models/model.joblib` — o pipeline completo (features + modelo), pronto para
  ser carregado pela API.
- `models/metrics.json` — MAE, RMSE, R² e tamanho do conjunto de teste.

### 4. Subir a API

```bash
flask --app house_prices.api.app run --debug
```

A API sobe em `http://127.0.0.1:5000`. Ver `docs/api.md` para o contrato completo
e exemplos de `curl`. Resumo rápido:

Git Bash / Linux / macOS (`curl` de verdade):

```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4, "bathrooms": 2.5, "sqft_living": 2400, "sqft_lot": 6000,
    "floors": 2.0, "waterfront": "N", "view": 0, "condition": "Good",
    "grade": 8, "sqft_above": 2400, "sqft_basement": 0, "yr_built": 2005,
    "yr_renovated": 0, "zipcode": 98052, "lat": 47.65, "long": -122.14,
    "sqft_living15": 2200, "sqft_lot15": 6200, "sale_year": 2015
  }'
```

PowerShell (`curl` é um alias de `Invoke-WebRequest`, que usa parâmetros diferentes do curl real):

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:5000/health"

Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "bedrooms": 4, "bathrooms": 2.5, "sqft_living": 2400, "sqft_lot": 6000,
    "floors": 2.0, "waterfront": "N", "view": 0, "condition": "Good",
    "grade": 8, "sqft_above": 2400, "sqft_basement": 0, "yr_built": 2005,
    "yr_renovated": 0, "zipcode": 98052, "lat": 47.65, "long": -122.14,
    "sqft_living15": 2200, "sqft_lot15": 6200, "sale_year": 2015
  }'
```

### 5. Rodar os notebooks (opcional)

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/
```

Os notebooks já estão salvos com as saídas de uma execução real (gráficos e
tabelas incluídos) — não é necessário reexecutá-los para lê-los, mas rodar você
mesmo é a melhor forma de aprender: mude uma célula, veja o que muda.

## Testes automatizados e CI

```bash
pytest                                    # roda todos os testes
pytest --cov=house_prices --cov-report=term-missing   # com cobertura
ruff check src tests                      # lint
```

Os testes **não dependem de credenciais do Kaggle nem do dataset completo**: eles
usam `data/sample/house_prices_sample.csv`, uma amostra de 300 linhas já limpa e
versionada no Git só para esse fim. Isso é o que permite o job de treino
ponta-a-ponta rodar no GitHub Actions sem segredos configurados.

O workflow `.github/workflows/ci.yml` roda a cada push/PR na `main` que toque
nesta pasta, em três jobs sequenciais:

1. **lint** — `ruff check`.
2. **test** — `pytest` com cobertura.
3. **train-smoke-test** — treina o modelo na amostra versionada e confere que
   `model.joblib` e `metrics.json` foram gerados corretamente, validando o
   pipeline de treino de ponta a ponta (não só unidades isoladas).

## Decisões de projeto (o "porquê")

As decisões de limpeza e de features estão documentadas com o "porquê" (não só o
"o quê") diretamente nos docstrings de `src/house_prices/data/clean.py` e
`src/house_prices/features.py`, e resumidas em `docs/pipeline.md` — vale a pena
ler antes de mexer no código, principalmente os pontos sobre vazamento de dados
(data leakage) e o motivo de a engenharia de features estar dentro de um
`sklearn.Pipeline` em vez de solta em um notebook.

## Próximos passos possíveis

Fora do escopo implementado aqui, mas natural de evoluir (e coberto na Aula 08 do
curso, `../Fase1/aula08_projeto_final_ciclo_completo/resumo.md`):

- **Rastreabilidade de experimentos** com MLflow (parâmetros, métricas, artefatos
  e `model signature` versionados a cada treino).
- **Conteinerização** com Docker, para eliminar "funciona na minha máquina".
- **Monitoramento de data drift** em produção (ex.: teste de
  Kolmogorov-Smirnov comparando distribuição de entrada em produção vs. dados de
  treino).
- **Continuous Training (CT)**: disparar retreino automático quando a
  performance cair abaixo de um limiar.
- **Governança**: neste dataset não há atributos sensíveis de pessoas (é sobre
  imóveis), então auditoria de fairness (Fairlearn) não se aplica da mesma forma
  que no exemplo de saúde da Aula 08 — mas o princípio de "gatekeeping automático
  antes de liberar um modelo" (ex.: falhar o CI se R² cair abaixo do baseline)
  seria o próximo passo natural aqui.
