# Aula 02 — Experimentação e MVP de Modelos

## Visão Geral

Esta aula cobre as etapas iniciais e cruciais de um projeto de Machine Learning: o mapeamento e entendimento dos dados brutos, a Análise Exploratória de Dados (EDA) e a construção de um MVP (Produto Mínimo Viável) de modelo. Usando o dataset clássico de Doença Cardíaca (Heart Disease, UCI), a aula demonstra na prática como inspecionar dados, identificar outliers e dados faltantes, aplicar técnicas iniciais de preparação e treinar um modelo baseline com Regressão Logística. O fio condutor é o princípio "dados ruins levam a análises ruins" (garbage in, garbage out) e a ideia de validar rapidamente uma ideia de modelagem antes de investir em soluções complexas.

## Tópicos Abordados

- **O que vem por aí?** — introdução ao problema de dados inconsistentes/implausíveis e apresentação do dataset de Doença Cardíaca (UCI) como fio condutor da aula.
- **Hands On** — aplicação prática de EDA e construção de MVP de modelo sobre o dataset Heart Disease.
- **Saiba Mais**
  - Uso do Python e Bibliotecas (Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn)
  - Passo 1: Carregamento e Entendimento Inicial dos Dados (`df.info()`, `df.describe()`, dicionário de dados)
  - Passo 2: Análise Exploratória de Dados (EDA) — missing values, distribuição da variável target, identificação de outliers, anomalias e valores inválidos, análise de distribuições, correlações e análise bivariada
  - Passo 3: Preparação e Limpeza dos Dados — tratamento de missing values (mediana/moda), One-Hot Encoding
  - Passo 4: Construção do Modelo MVP (Minimum Viable Product) — Regressão Logística como baseline, train/test split, MLflow Tracking, métricas de avaliação
  - Data Understanding no contexto CRISP-DM
  - Análise Exploratória de Dados (EDA) e variáveis dos dados (univariada, bivariada, multivariada)
  - Construção de um MVP de modelo e experimentação rápida (lean startup, MVI, MLflow, Databricks)
- **Mercado, Cases e Tendências** — case Airbnb de predição de Lifetime Value (LTV) de anúncios.

## Conceitos-Chave

### CRISP-DM e Data Understanding
O CRISP-DM (Cross-Industry Standard Process for Data Mining) é um framework tradicional de ciência de dados com fases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation e Deployment. A aula foca na fase de Data Understanding, cujo propósito é obter uma visão geral das fontes de dados, suas características e qualidade. Essa fase se organiza em quatro tarefas: (1) coletar dados iniciais e documentar seu conteúdo (dicionário de dados), (2) descrever os dados via estatísticas descritivas, (3) explorar os dados (EDA) e (4) verificar a qualidade dos dados (aplicando o princípio GIGO — Garbage In, Garbage Out). O Data Understanding é iterativo: descobertas na modelagem frequentemente levam a revisitar o entendimento dos dados.

### Análise Exploratória de Dados (EDA)
A EDA é o coração do Data Understanding — permite que os dados "contem sua história" antes da modelagem preditiva. Classifica-se conforme o número de variáveis analisadas:
- **Univariada**: análise de uma variável por vez (estatísticas ou gráficos como histogramas e box plots).
- **Bivariada**: análise de duas variáveis simultaneamente (tabelas de contingência, coeficientes de correlação, gráficos de dispersão).
- **Multivariada**: três ou mais variáveis simultaneamente (heatmaps de correlação, pair plots, PCA).

Na aula, a EDA sobre o dataset de Doença Cardíaca foi estruturada em quatro pilares: Análise de Dados Faltantes, Distribuição da Variável Target, Identificação de Outliers e Valores Inválidos/Inconsistentes.

### `df.info()` e `df.describe()`
São os primeiros comandos executados em qualquer projeto de ciência de dados. `df.info()` fornece tipos de dados (dtypes), contagem de valores não-nulos (revelando missing data) e uso de memória. `df.describe()` fornece estatísticas descritivas: tendência central (média, mediana), dispersão (desvio-padrão), range/limites (min/max, úteis para detectar valores implausíveis) e quartis (base para boxplots e regra do IQR). Para categóricas (`include='all'`), fornece contagem, categoria mais frequente (top), frequência (freq) e cardinalidade (unique).

### Missing Values (Dados Faltantes)
Valores ausentes (NaN, Null, "?") são um desafio comum, pois a maioria dos algoritmos de ML não processa entradas incompletas. As estratégias principais são:
- **Remoção** dos registros incompletos — simples, mas arriscada se a proporção de ausentes for alta.
- **Imputação simples** — preencher com média, mediana (mais robusta a outliers) ou moda.
- **Imputação por modelo (avançada)** — k-NN ou modelos de regressão para estimar o valor ausente.
- **Análise de sensibilidade** — testar o impacto de diferentes estratégias de imputação no resultado do modelo.

No dataset de Heart Disease, as colunas `ca`, `thal` e `slope` apresentaram os maiores percentuais de missing values.

### Outliers (Valores Atípicos)
Outliers são valores que se desviam significativamente do restante dos dados, podendo representar erros de medição/registro ou casos legítimos raros. As técnicas de identificação incluem:
- **Z-Score**: marca como outlier valores a mais de 3 desvios-padrão da média.
- **Boxplots**: inspeção visual rápida via quartis e IQR (Intervalo Interquartil).

O tratamento pode envolver remoção (com cautela, pois pode eliminar informação legítima), transformação (ex.: log para reduzir assimetria) ou substituição por valor menos extremo (ex.: mediana/capping).

### Variável Target e Balanceamento de Classes
Antes de modelar, é essencial entender a distribuição da variável alvo. Um dataset **desbalanceado** (onde uma classe domina) pode levar a modelos enviesados — por exemplo, um modelo que sempre prevê a classe majoritária pode ter alta acurácia sem utilidade prática. O grau de balanceamento pode ser medido pelo ratio entre a classe minoritária e a majoritária; ratios abaixo de 0,5 sugerem considerar técnicas como SMOTE ou `class_weight`.

### One-Hot Encoding
Técnica de codificação de variáveis categóricas nominais (sem ordem intrínseca, como `sex`, `cp`, `restecg`, `slope`, `thal`) em colunas binárias (0/1), uma para cada categoria. Evita que o modelo interprete uma ordem artificial entre categorias que não possuem hierarquia.

### MVP (Minimum Viable Product) de Modelo
Conceito originado no desenvolvimento de startups (Eric Ries), adaptado à ciência de dados: construir a versão mais simples de um modelo que já entregue valor e aprendizado, antes de investir em soluções sofisticadas. Um MVP de modelo serve a dois propósitos: (1) **validação rápida da viabilidade** — verificar se os dados contêm sinal preditivo útil; e (2) **feedback e iteração** — colher retorno de stakeholders e ajustar o rumo rapidamente ("fail fast" ou "succeed fast"). O conceito se estende à noção de **Minimum Viable Intelligence (MVI)**: começar a gerar inteligência a partir dos dados sem esperar pelo algoritmo perfeito.

### Regressão Logística como Baseline
A Regressão Logística foi escolhida como modelo MVP por três razões: (1) **simplicidade e interpretabilidade**, permitindo que a equipe técnica e stakeholders entendam como as features influenciam o resultado; (2) serve como **referencial (baseline) de desempenho** que modelos mais complexos (Random Forest, Gradient Boosting) precisam superar para justificar seu custo; (3) permite **ciclos de feedback rápidos**, com treinamento e avaliação ágeis.

### Métricas de Avaliação
- **Acurácia**: proporção de previsões corretas sobre o total.
- **Precision**: quão confiável é o modelo ao prever a classe positiva (minimiza falsos positivos).
- **Recall**: capacidade de encontrar todas as amostras positivas (minimiza falsos negativos).
- **F1-Score**: média harmônica entre precision e recall, útil para equilibrar as duas métricas.
- **Overfitting**: diferença entre acurácia de treino e de teste; uma diferença pequena indica boa generalização.

### MLflow Tracking
Framework open-source (iniciado pela Databricks) para gerenciar o ciclo de vida de ML, com quatro componentes: **Tracking** (registrar parâmetros e métricas de experimentos), **Projects** (empacotar código de forma reprodutível), **Models** (padronizar modelos) e **Model Registry** (gerenciar modelos em produção). Na aula, o foco foi o MLflow Tracking, usado para registrar métricas do MVP em um "Run", possibilitando comparação objetiva entre iterações futuras.

## Exercício Hands-On (do material)

O exercício prático da aula consiste em aplicar os conceitos de EDA e construção de MVP de modelo usando o dataset clássico de Doença Cardíaca (Heart Disease) do repositório UCI, disponível no Kaggle como `heart_disease_uci.csv`, para prever a presença ou ausência de doença cardíaca (coluna `num`, renomeada para `target`). O fluxo proposto segue quatro passos:

1. **Carregamento e Entendimento Inicial dos Dados**: importar Pandas, NumPy, Matplotlib e Seaborn; carregar o CSV com `pd.read_csv`; inspecionar com `df.shape`, `df.head()`, `df.info()` e `df.describe()`. O dataset contém atributos clínicos (idade, sexo, tipo de dor no peito, pressão arterial, colesterol, resultados de exames, etc.) de pacientes das bases Cleveland, Hungarian, Switzerland e VA Long Beach.
2. **Análise Exploratória de Dados (EDA)**: quantificar e visualizar missing values por coluna (tabela + gráfico de barras); renomear `num` para `target` e binarizar (0 = sem doença, 1 = com doença); visualizar a distribuição da classe (gráfico de barras e pizza) e calcular o ratio de balanceamento; identificar outliers via boxplots e Z-Score para as variáveis numéricas (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`).
3. **Preparação e Limpeza dos Dados**: imputar valores ausentes — mediana para variáveis numéricas (`trestbps`, `chol`, `thalch`, `oldpeak`, `ca`) e moda para variáveis categóricas (`thal`, `slope`, `fbs`, `exang`, `restecg`); remover a coluna `dataset`; aplicar One-Hot Encoding nas variáveis categóricas nominais (`sex`, `cp`, `restecg`, `slope`, `thal`) com `pd.get_dummies(..., drop_first=True)`.
4. **Construção do Modelo MVP**: dividir os dados em treino (80%) e teste (20%) com `train_test_split` (usando `random_state=42` para reprodutibilidade); treinar uma Regressão Logística com `scikit-learn`; registrar parâmetros e métricas no MLflow Tracking; avaliar o modelo com acurácia, F1-score, precision e recall.

**Resultado obtido no material (baseline)**: Acurácia de Treino = 82,47%; Acurácia de Teste = 80,43%; F1-Score de Teste = 83,02%; Precision = 85,44%; Recall = 80,73%. A diferença de apenas ~2 pontos percentuais entre treino e teste indicou baixo overfitting, e a acurácia de teste superou amplamente o baseline de chance (55,3%, correspondente a sempre prever a classe majoritária), validando que o dataset possui sinal preditivo robusto. Os próximos passos sugeridos foram aprimoramento de features (Feature Engineering/Selection) e teste de algoritmos mais complexos (Random Forest, Gradient Boosting) que devem superar esse baseline.

## Exemplos de Código

O conceito de **inspeção inicial com `df.info()` e `df.describe()`**, apresentado no Passo 1 da aula, pode ser reproduzido assim:

```python
import pandas as pd

df = pd.read_csv("data/heart_disease_uci.csv")

print(f"Shape: {df.shape}")
print(df.info())

# Estatísticas descritivas para numéricas e categóricas
print(df.describe())
print(df.describe(include='all'))
```

O conceito de **análise e imputação de missing values** (mediana para numéricas, moda para categóricas), discutido no Passo 3, pode ser generalizado em uma função reutilizável:

```python
import pandas as pd

def relatorio_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Gera um relatório de valores ausentes por coluna, ordenado do maior para o menor."""
    missing = pd.DataFrame({
        "coluna": df.columns,
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
    })
    return missing[missing["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def imputar_valores_faltantes(df: pd.DataFrame, cols_numericas: list, cols_categoricas: list) -> pd.DataFrame:
    """Imputa mediana em colunas numéricas e moda em colunas categóricas (menos sensível a outliers)."""
    df_clean = df.copy()
    for col in cols_numericas:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in cols_categoricas:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    return df_clean
```

O conceito de **identificação de outliers via Z-Score e IQR**, apresentado na análise exploratória, pode ser implementado assim:

```python
import numpy as np
from scipy import stats

def outliers_zscore(serie: pd.Series, limiar: float = 3.0) -> pd.Series:
    """Retorna uma máscara booleana marcando outliers (|Z-Score| > limiar)."""
    z = np.abs(stats.zscore(serie.dropna()))
    return z > limiar

def outliers_iqr(serie: pd.Series) -> pd.Series:
    """Identifica outliers pela regra do Intervalo Interquartil (IQR)."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return (serie < limite_inferior) | (serie > limite_superior)

def capping_outliers(serie: pd.Series) -> pd.Series:
    """Limita (capping) valores extremos aos limites definidos pelo IQR."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return serie.clip(lower=limite_inferior, upper=limite_superior)
```

O conceito de **One-Hot Encoding de variáveis categóricas nominais**, usado para preparar os dados para a Regressão Logística, pode ser reproduzido assim:

```python
categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
```

O conceito de **train/test split reprodutível**, usado para preparar o MVP, é ilustrado a seguir:

```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['id', 'target'])
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

O conceito de **MVP de modelo com Regressão Logística como baseline**, comparado a um "chute" ingênuo pela classe majoritária, pode ser exemplificado assim:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Baseline "ingênuo": sempre prevê a classe majoritária
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
print(f"Acurácia do baseline ingênuo: {accuracy_score(y_test, dummy.predict(X_test)):.4f}")

# MVP real: Regressão Logística
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
```

O conceito de **experiment tracking com MLflow**, introduzido para tornar o MVP rastreável e reprodutível, é ilustrado a seguir:

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_experiment("heart_disease_classification")

with mlflow.start_run(run_name="logistic_regression_baseline"):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Parâmetros do modelo (para comparação entre runs futuros)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Métricas de desempenho
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
    mlflow.log_metric("overfitting_gap", train_accuracy - test_accuracy)

    mlflow.sklearn.log_model(model, "model")
```

O conceito de **comparação de modelos** — o MVP como baseline que modelos mais complexos precisam superar — pode ser ilustrado com um loop simples de experimentação:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

modelos = {
    "logistic_regression_baseline": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

resultados = []
for nome, modelo in modelos.items():
    with mlflow.start_run(run_name=nome):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        resultados.append({"modelo": nome, "acuracia": acc, "f1_score": f1})

df_resultados = pd.DataFrame(resultados).sort_values("f1_score", ascending=False)
print(df_resultados)
```

## Cases e Tendências de Mercado

- **Airbnb — predição de Lifetime Value (LTV) de novos anúncios**: a equipe de ciência de dados da Airbnb seguiu etapas semelhantes às estudadas na aula, divulgadas no Airbnb Tech Blog.
  - **Entendimento dos dados e EDA**: analisaram dados históricos de reservas e avaliaram features como localização, tipo de acomodação, preço relativo a similares e taxa de ocupação; identificaram, por exemplo, que a porcentagem de noites disponíveis nos próximos 6 meses é um indicador relevante de LTV.
  - **Qualidade e preparação**: validaram valores ausentes e consistência, e transformaram variáveis categóricas e numéricas (ex.: one-hot encoding para tipo de quarto ou bairro).
  - **MVP de modelo (prototipagem)**: treinaram inicialmente modelos simples em Python para avaliar rapidamente quais features mais contribuíam e a performance obtida, iterando localmente com agilidade.
  - **Validação e iteração**: colocaram o modelo candidato em produção experimental graças à infraestrutura de ML interna, sem reescrever todo o código; um modelo de deep learning de recomendação lançado apresentou problemas que só foram percebidos em produção, reforçando que o aprendizado real ocorre ao expor o modelo ao mundo real.
- **Tendência de mercado**: uso crescente de plataformas e ferramentas de MLOps (como MLflow, nativamente integrado à Databricks) para encurtar o ciclo entre ideia e modelo implantado, reduzindo o custo de colocar novos modelos em produção e permitindo criar e testar MVPs de forma mais rápida e escalável.

## Checklist de Estudo

- [ ] Sei explicar as quatro tarefas da fase de Data Understanding no CRISP-DM.
- [ ] Sei diferenciar análise univariada, bivariada e multivariada em uma EDA.
- [ ] Sei aplicar e interpretar `df.info()` e `df.describe()` para diagnosticar tipos, missing values e distribuições.
- [ ] Sei identificar outliers usando Z-Score e boxplots, e explicar as opções de tratamento (remoção, transformação, capping).
- [ ] Sei explicar por que a mediana é preferida à média para imputação em variáveis numéricas com outliers.
- [ ] Sei justificar a escolha da Regressão Logística como modelo MVP/baseline e o papel do MLflow Tracking na experimentação.
- [ ] Sei explicar o conceito de MVP de modelo (e MVI) e por que validar sinal preditivo rapidamente é preferível a buscar a solução perfeita de início.

## Palavras-chave

EDA. MVP. CRISP-DM.

## Referências

- CHANG, R. **Using Machine Learning to Predict Value of Homes On Airbnb**. 2017. Disponível em: https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d. Acesso em: 28 jan. 2026.
- CONCEITOS TECH. **Entenda os Outliers: Identificação e Tratamento em Dados**. 2023. Disponível em: https://conceitos.tech/tutoriais/inteligencia-artificial/machine-learning/o-que-sao-outliers-e-como-identifica-los-nos-dados/. Acesso em: 28 jan. 2026.
- CONCEITOS TECH. **Estratégias para Gerenciar Dados Faltantes em Machine Learning**. 2023. Disponível em: https://conceitos.tech/tutoriais/inteligencia-artificial/machine-learning/como-lidar-com-dados-faltantes-em-machine-learning. Acesso em: 28 jan. 2026.
- GARCIA, J. P. **What Airbnb discovered after launching its first AI product?** 2023. Disponível em: https://joseparreogarcia.substack.com/p/what-airbnb-discovered-after-launching. Acesso em: 28 jan. 2026.
- IBM ANALYTICS. **O que é análise exploratória de dados (EDA)?** 2026. Disponível em: https://www.ibm.com/br-pt/think/topics/exploratory-data-analysis. Acesso em: 28 jan. 2026.
- ILUMEO. **O Produto Mínimo Viável – MVP – para Dados**. 2021. Disponível em: https://ilumeo.com.br/categorias/2021-02-09-o-produto-minimo-viavel-mvp-para-dados. Acesso em: 28 jan. 2026.
- KANOKI; DATA SCIENCE PM. **What is a Data Science MVP?** 2020. Disponível em: https://www.datascience-pm.com/data-science-mvp. Acesso em: 28 jan. 2026.
- MIOT, H. A. **Valores anômalos e dados faltantes em estudos clínicos e experimentais**. Jornal Vascular Brasileiro, v. 18, n. 3, 2019. Disponível em: https://www.researchgate.net/publication/333487764_Valores_anomalos_e_dados_faltantes_em_estudos_clinicos_e_experimentais. Acesso em: 28 jan. 2026.
- RICHARDSON. **A Fase de Business Data Understanding (Compreensão dos Dados do Negócio)**. 2023. Disponível em: https://dev.to/_richardson_/a-fase-de-business-data-understanding-compreensao-dos-dados-do-negocio-2bab. Acesso em: 28 jan. 2026.
- SONY, R. K. **Heart Disease Data**. 2023. Disponível em: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data. Acesso em: 28 jan. 2026.
- UCI MACHINE LEARNING REPOSITORY. **Heart Disease**. 1988. Disponível em: https://archive.ics.uci.edu/dataset/45/heart+disease. Acesso em: 28 jan. 2026.
