# Aula 03 — Desenvolvimento e Engenharia de Modelos

## Visão Geral

Esta aula parte de um MVP de modelo (Regressão Logística, ~80% de acurácia) para detecção de doença cardíaca e ensina como evoluí-lo para uma solução mais robusta. O foco está em quatro frentes complementares: engenharia e seleção de atributos, comparação entre algoritmos de classificação, ajuste (tuning) de hiperparâmetros e construção de pipelines reproduzíveis. Ao final, o objetivo é sair de um protótipo básico para um modelo bem calibrado, documentado, versionado e organizado com boas práticas de engenharia de software, pronto para os próximos passos de implantação (deployment).

## Tópicos Abordados

- **O que vem por aí?** — contextualização do objetivo da aula: evoluir o MVP de detecção de doença cardíaca (dataset Heart Disease da UCI) superando o baseline de ~80% de acurácia.
- **Hands On** — atividade prática de evolução do projeto de detecção de doenças cardíacas, reaproveitando o dataset pré-processado da aula anterior.
- **Saiba Mais**
  - Passo 1: Feature Engineering e Seleção de Atributos
  - Passo 2: Treinamento de diferentes modelos
  - Passo 3: Tuning de Hiperparâmetros
  - Passo 4: Construção de um Pipeline Reproduzível
  - Passo 5: Boas Práticas de Código e Integração
  - Entendendo o Problema de Negócio e os Dados
  - Seleção de Features (métodos filtro, wrapper e embedded)
  - Engenharia de Atributos (Feature Engineering)
  - Comparação entre Algoritmos de Modelagem
  - Tuning de Hiperparâmetros (Grid Search, Random Search, Halving, Otimização Bayesiana)
  - Pipelines Reproduzíveis e Validação
- **Mercado, Cases e Tendências** — MLOps, AutoML, cases da Uber (Michelangelo) e Airbnb, regulamentação em saúde (FDA) e o papel do "engenheiro de ML full-stack".
- **O que você viu nesta aula?** — síntese do ciclo de vida do modelo pós-MVP.
- **Referências**

## Conceitos-Chave

### Feature Engineering (Engenharia de Atributos)
Processo de selecionar, modificar e criar características a partir de dados brutos para melhorar o desempenho dos modelos. Inclui transformações matemáticas (log, quadrática, escalonamento com StandardScaler/MinMaxScaler), criação de features combinadas (razões, produtos, binning), encoding de variáveis categóricas (one-hot, ordinal, target encoding) e tratamento de valores ausentes/outliers. É um processo iterativo e dependente do algoritmo alvo: uma feature útil para um modelo linear pode ser redundante para uma árvore de decisão, que já capta pontos de corte naturalmente.

### Seleção de Features
Processo de identificar e manter apenas os atributos mais relevantes, eliminando os redundantes ou pouco informativos, para reduzir dimensionalidade, diminuir risco de overfitting, ganhar eficiência computacional e melhorar a interpretabilidade. Divide-se em três abordagens:
- **Filtro**: análises estatísticas simples e independentes do modelo (variância, qui-quadrado, ANOVA F-value) — ex.: `SelectKBest`. Rápido, mas ignora interações entre variáveis.
- **Wrapper**: busca guiada por um modelo, como o RFE (Recursive Feature Elimination), que treina, remove a feature menos importante e retreina iterativamente. Mais custoso, porém considera interações.
- **Embedded**: a seleção ocorre durante o próprio treinamento (árvores, Random Forest com importância de features, regularização L1/Lasso). Eficiente, mas depende da escolha do modelo base.

Um ponto crítico é **evitar leakage de dados**: a seleção de features deve ser feita somente com dados de treino, nunca olhando o conjunto de teste, sob risco de um viés otimista que não generaliza.

### Comparação entre Algoritmos de Modelagem
Não existe um modelo universalmente superior (espírito do **No Free Lunch Theorem**): o desempenho depende dos dados e do contexto do problema. A aula recapitula famílias de modelos:
- **Modelos Lineares** (Regressão Logística): rápidos, simples, interpretáveis, mas limitados para capturar relações não lineares sem feature engineering adequada.
- **Árvores de Decisão**: intuitivas, capturam interações e não linearidades, não exigem normalização, mas tendem a overfitting se crescerem demais.
- **Random Forest**: ensemble de árvores treinadas em amostras e subconjuntos de features aleatórios; maior poder preditivo e estabilidade, porém menos interpretável.
- **k-Nearest Neighbors (k-NN)**: baseado em instâncias e distância; sofre com a "curse of dimensionality" e é custoso em bases grandes.
- **Outros**: Naive Bayes, SVM e Redes Neurais, cada um com prós/contras dependendo do volume e natureza dos dados.

A escolha do algoritmo deve considerar não só a métrica (acurácia, F1, recall), mas também tempo de treinamento/predição, interpretabilidade e o contexto de negócio (por exemplo, em saúde, recall alto pode ser mais importante que acurácia para não perder casos positivos).

### Tuning de Hiperparâmetros
Hiperparâmetros são configurações definidas antes do treinamento (ex.: `n_estimators` e `max_depth` no Random Forest, `C` na regressão logística, `k` no k-NN). O ajuste busca o ponto ótimo de generalização, evitando tanto subaproveitamento quanto overfitting. Principais estratégias:
- **Grid Search (busca exaustiva)**: testa todas as combinações de um grid definido, usando validação cruzada — pode ser computacionalmente caro.
- **Random Search (busca aleatória)**: sorteia combinações, sendo mais eficiente quando poucos hiperparâmetros realmente importam.
- **Halving Grid/Random Search**: refina progressivamente as candidatas mais promissoras com mais recursos (Successive Halving, Hyperband).
- **Otimização Bayesiana** (ex.: Tree Parzen Estimator via Hyperopt, Optuna): modela a função de desempenho para escolher inteligentemente as próximas combinações.

Cuidados importantes: usar um conjunto holdout final não usado no tuning (ou nested cross-validation) para evitar overfitting no processo de tuning; priorizar hiperparâmetros de maior impacto; escolher a métrica de otimização alinhada ao objetivo de negócio (accuracy, F1, recall etc.); e reconhecer retornos decrescentes — ganhos marginais podem não justificar o custo computacional adicional.

### Pipelines Reproduzíveis
Um pipeline de Machine Learning encadeia de forma automatizada e repetível as etapas de pré-processamento, seleção de features e treinamento/predição (ex.: `Pipeline` do Scikit-learn). Ao chamar `fit`, todas as transformações são ajustadas com dados de treino; ao chamar `predict`, as mesmas transformações (parâmetros aprendidos no treino) são aplicadas ao conjunto de teste ou a novos dados. Isso evita erros comuns como normalizar o treino e esquecer o teste, e previne vazamento de informação entre folds de validação cruzada. Pipelines também facilitam a integração com `GridSearchCV`/`cross_val_score` e são a base para levar um modelo à produção de forma confiável.

### Reprodutibilidade e Boas Práticas de Código
Envolve fixar sementes aleatórias (`random_state`), versionar dados e código (Git), modularizar o projeto (por exemplo, seguindo o template do Cookiecutter Data Science, com módulos como `preprocessing.py`, `features.py`, `train.py`) e registrar experimentos com ferramentas de rastreamento como o **MLflow**, que permite logar métricas de treino/teste de cada execução (run) e comparar experimentos objetivamente. Essas práticas são essenciais para auditoria, colaboração em equipe e, em setores regulados (saúde, financeiro), para conformidade regulatória.

### Data Leakage (Vazamento de Dados)
Ocorre quando informação do conjunto de teste (ou informação que não estaria disponível no momento real de uso do modelo) influencia o treinamento ou a seleção de features, inflando artificialmente a avaliação de desempenho. A regra prática apresentada: sempre perguntar "esta informação estaria disponível quando eu for usar o modelo de verdade?" — se a resposta for não, ela não deve ser usada.

### Overfitting
Ocorre quando o modelo memoriza o conjunto de treino, incluindo ruído, em vez de aprender o padrão geral, prejudicando a generalização para novos dados. Pode acontecer tanto no treinamento do modelo quanto no próprio processo de tuning de hiperparâmetros, sendo mitigado por validação cruzada e conjuntos holdout.

## Exercício Hands-On (do material)

O exercício prático propõe evoluir o projeto de detecção de doenças cardíacas construído na aula anterior (MVP com Regressão Logística, ~80% de acurácia), seguindo estes passos:

1. **Preparação**: reutilizar o dataset pré-processado da aula anterior (variáveis codificadas e dados limpos), separar em variáveis preditoras `X` e alvo `y`, e dividir em treino/teste (ex.: 80%/20%) com `random_seed` fixo no `train_test_split`, garantindo comparação justa com o baseline.

2. **Feature Engineering e Seleção de Atributos** (Passo 1): criar novas features a partir de conhecimento de domínio médico — transformação quadrática da idade, razão colesterol/idade, percentual da frequência cardíaca máxima prevista, razão pressão/colesterol — e aplicar padronização (`StandardScaler`). Em seguida, realizar seleção de atributos (por exemplo, `SelectKBest` com ANOVA F-value, limitando a 25 variáveis) usando **apenas** dados de treino, gerando `X_train_sel` e `X_test_sel`.

3. **Treinamento de diferentes modelos** (Passo 2): treinar Árvore de Decisão, Random Forest e SVM (além da Regressão Logística já conhecida) na mesma partição treino/teste e mesmo `random_state`, comparando acurácia, F1-score, precisão e recall. Cenário hipotético do material: o Random Forest atinge ~85% de acurácia no teste, superando os ~80% da regressão logística.

4. **Tuning de Hiperparâmetros** (Passo 3): aplicar `GridSearchCV` no modelo mais promissor (Random Forest), testando uma grade como `{'n_estimators': [50, 100, 200], 'max_depth': [None, 4, 8]}` com 5 folds de validação cruzada (45 treinos no total). Cenário hipotético: `max_depth=4` e `n_estimators=100` otimizam a performance, elevando a acurácia hipoteticamente para ~86%.

5. **Construção de um Pipeline Reproduzível** (Passo 4): montar um `Pipeline` do Scikit-learn encadeando `StandardScaler`/`SelectKBest` e o modelo final otimizado, de forma que `pipeline.fit()` e `pipeline.predict()` apliquem consistentemente as mesmas transformações em treino e teste/produção. Resultado ilustrativo final do material: ~88% de acurácia e 0.88 de F1 no teste. Salvar o pipeline treinado com `pickle` ou `joblib` para reuso.

6. **Boas Práticas de Código e Integração** (Passo 5): organizar o projeto de forma modular (ex.: `preprocessing.py`, `features.py`, `train.py`), evitando notebooks monolíticos, fixando sementes aleatórias (randomstate=42) e registrando experimentos com MLflow para rastreabilidade.

## Exemplos de Código

A seguir, exemplos ilustrativos em Python que operacionalizam os conceitos apresentados no PDF. Eles vão além do que está literalmente no material, servindo como aplicação prática dos conceitos discutidos.

Feature engineering aplicada ao domínio de saúde cardíaca, seguindo a lógica de criação de variáveis derivadas descrita no Passo 1 do material (transformação quadrática, razões e binning):

```python
import pandas as pd
import numpy as np

eps = 1e-6

def engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas a partir de conhecimento de domínio médico,
    conforme descrito no Passo 1 (Feature Engineering) do material da Aula 03."""
    df_eng = df.copy()

    # Peso não linear para a idade (risco acelerado)
    df_eng["age_squared"] = df_eng["age"] ** 2

    # Risco relativo: colesterol em relação à idade
    df_eng["cholesterol_to_age"] = df_eng["chol"] / (df_eng["age"] + eps)

    # Percentual da frequência cardíaca máxima alcançada (regra 220 - idade)
    if {"thalch", "age"}.issubset(df_eng.columns):
        predicted_max_hr = (220 - df_eng["age"]).clip(lower=1)
        df_eng["max_hr_pct"] = df_eng["thalch"] / (predicted_max_hr + eps)

    # Perfil de risco vascular combinado
    if {"trestbps", "chol"}.issubset(df_eng.columns):
        df_eng["bp_chol_ratio"] = df_eng["trestbps"] / (df_eng["chol"] + 1)

    # Faixas etárias (bins) para capturar padrões não lineares de idade x risco
    df_eng["age_bin"] = pd.cut(
        df_eng["age"], bins=[0, 40, 55, 70, 120],
        labels=["jovem", "meia_idade", "idoso", "muito_idoso"]
    )

    # Categorização da depressão do segmento ST (oldpeak)
    df_eng["oldpeak_level"] = pd.cut(
        df_eng["oldpeak"], bins=[-0.01, 1, 2, 10],
        labels=["baixo", "moderado", "alto"]
    )

    return df_eng
```

Seleção de features com o método de filtro (`SelectKBest`) descrito no Passo 1, aplicada corretamente apenas sobre o conjunto de treino para evitar data leakage:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

selector = SelectKBest(score_func=f_classif, k=min(25, X_train.shape[1]))
X_train_sel = selector.fit_transform(X_train, y_train)  # fit apenas no treino
X_test_sel = selector.transform(X_test)                 # apenas transform no teste

selected_columns = X_train.columns[selector.get_support()]
print("Features selecionadas:", list(selected_columns))
```

Comparação entre algoritmos de classificação (Regressão Logística, Árvore de Decisão, Random Forest e SVM), ilustrando o Passo 2 e a seção "Comparação entre Algoritmos de Modelagem":

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    results.append({
        "modelo": name,
        "acuracia": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),   # crítico em saúde: minimizar falsos negativos
        "precisao": precision_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print(results_df)
```

Tuning de hiperparâmetros com `GridSearchCV`, conforme o Passo 3, priorizando F1-score como métrica alinhada ao objetivo clínico de equilibrar precisão e recall:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 4, 8],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",   # métrica alinhada ao objetivo de negócio, não apenas acurácia
    n_jobs=-1,
)
grid_search.fit(X_train_sel, y_train)

print("Melhores parâmetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_
```

Pipeline completo e reprodutível (Passo 4), encadeando escalonamento, seleção de features e modelo final em um único objeto — o que evita erros de vazamento entre treino e teste, conforme discutido em "Pipelines Reproduzíveis e Validação":

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
    ("model", RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )),
])

full_pipeline.fit(X_train, y_train)
y_pred_final = full_pipeline.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
print(f"Acurácia final: {final_accuracy:.3f} | F1 final: {final_f1:.3f}")
```

Versionamento do pipeline treinado com `joblib`, incluindo metadados do experimento — prática de reprodutibilidade discutida no Passo 5 (fixar sementes, registrar resultados, versionar código e dados):

```python
import joblib
import json
from datetime import datetime, timezone

model_metadata = {
    "modelo": "RandomForestClassifier",
    "hiperparametros": full_pipeline.named_steps["model"].get_params(),
    "features_utilizadas": list(selected_columns),
    "random_state": 42,
    "acuracia_teste": final_accuracy,
    "f1_teste": final_f1,
    "dataset": "Heart Disease UCI v1",
    "data_treinamento": datetime.now(timezone.utc).isoformat(),
}

joblib.dump(full_pipeline, "heart_disease_pipeline_v1.joblib")
with open("heart_disease_pipeline_v1_metadata.json", "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False)
```

Rastreamento de experimentos com MLflow, mencionado no Passo 5 e na seção de Mercado como prática recorrente para comparar execuções (runs) de forma objetiva:

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("heart_disease_model_development")

with mlflow.start_run(run_name="random_forest_tuned"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy_test", final_accuracy)
    mlflow.log_metric("f1_test", final_f1)
    mlflow.sklearn.log_model(full_pipeline, artifact_path="model")
```

Estrutura de projeto modular inspirada no template Cookiecutter Data Science, mencionado no Passo 5 como boa prática de organização de código:

```text
heart_disease_project/
├── data/
│   ├── raw/                # dados brutos originais
│   └── processed/          # dados pré-processados
├── notebooks/               # notebooks exploratórios (EDA)
├── src/
│   ├── preprocessing.py     # normalização, encoding, limpeza
│   ├── features.py          # criação e seleção de features
│   ├── train.py             # treinamento do modelo dado um conjunto de hiperparâmetros
│   └── evaluate.py          # cálculo de métricas no holdout
├── models/                  # pipelines/modelos serializados (joblib/pickle)
└── tests/
    ├── test_features.py     # testes unitários de feature engineering
    └── test_preprocessing.py
```

Teste unitário simples para uma função de feature engineering, ilustrando a preocupação com qualidade e testabilidade de código mencionada na seção de Mercado (convergência entre Data Science e Engenharia de Software):

```python
import pandas as pd
from src.features import engineer_heart_features

def test_age_squared_feature():
    df = pd.DataFrame({"age": [40, 60], "chol": [200, 250], "thalch": [150, 130]})
    df_result = engineer_heart_features(df)
    assert "age_squared" in df_result.columns
    assert df_result.loc[0, "age_squared"] == 1600

def test_bp_chol_ratio_avoids_division_by_zero():
    df = pd.DataFrame({"age": [50], "chol": [0], "trestbps": [120], "thalch": [140]})
    df_result = engineer_heart_features(df)
    assert df_result.loc[0, "bp_chol_ratio"] == 120.0  # 120 / (0 + 1)
```

## Cases e Tendências de Mercado

- **MLOps em crescimento**: convergência entre Data Science e Engenharia de Software; não basta ter alta acurácia, é preciso deployment, monitoramento e manutenção de qualidade ao longo do tempo. O MLflow é citado como ferramenta amplamente adotada para rastrear experimentos e gerenciar modelos em todas as fases.
- **Plataformas de nuvem** (AWS, GCP, Azure) oferecem serviços integrados de pipeline, da preparação de dados ao deployment, sinalizando um futuro cada vez mais automatizado e orientado a pipelines.
- **AutoML (Automated Machine Learning)**: automatiza seleção de features, escolha de algoritmos e tuning de hiperparâmetros. Citados players como Google AutoML, H2O.ai e DataRobot, que podem permitir a profissionais com conhecimento básico de dados (ex.: um médico) treinar e validar modelos.
- **Case Uber — Michelangelo**: plataforma para gerenciar todo o ciclo de vida de modelos, da extração de features em tempo real ao treinamento e deployment.
- **Case Airbnb**: investimento em frameworks de experimentação rápida, combinando rigor na análise exploratória com protótipos iterativos.
- **Regulamentação em saúde**: exigências como as da FDA (EUA) para softwares médicos com IA demandam processos de desenvolvimento auditáveis e modelos capazes de explicar suas predições — versionamento, documentação de features e monitoramento de performance tornam-se requisitos regulatórios, não apenas boas práticas.
- **Data drift**: à medida que a população de pacientes muda ao longo do tempo, modelos podem precisar de recalibração — ponto citado no contexto de MLOps em saúde.
- **Tendência de carreira**: a fronteira entre "cientista de dados" e "engenheiro de ML" está cada vez mais tênue; profissionais full-stack de ML são muito demandados. Com o avanço do AutoML, o profissional tende a se concentrar em definir problemas, garantir qualidade dos dados e interpretar resultados, enquanto partes mais mecânicas da otimização são delegadas a plataformas.

## Checklist de Estudo

- [ ] Sei diferenciar seleção de features (reduzir variáveis) de engenharia de atributos (criar/transformar variáveis).
- [ ] Sei explicar os três tipos de métodos de seleção de features (filtro, wrapper e embedded) e dar um exemplo de cada.
- [ ] Sei explicar por que a seleção/engenharia de features deve ser feita apenas com dados de treino, evitando data leakage.
- [ ] Sei comparar as vantagens e limitações de modelos lineares, árvores de decisão, Random Forest e k-NN para um problema tabular.
- [ ] Sei diferenciar Grid Search de Random Search e explicar quando cada um é mais indicado.
- [ ] Sei explicar por que um Pipeline do Scikit-learn evita erros de vazamento de dados entre treino e teste/produção.
- [ ] Sei justificar, com um exemplo prático, por que a métrica de otimização de um modelo deve estar alinhada ao objetivo de negócio (ex.: recall em diagnóstico médico).

## Palavras-chave

Feature Engineering. Tuning de Hiperparâmetros. Pipelines de ML.

## Referências

- LABEL YOUR DATA TEAM. **Pattern Recognition and Machine Learning: 2025 Industry Applications**. 2025. Disponível em: https://labelyourdata.com/articles/pattern-recognition-in-machine-learning. Acesso em: 28 jan. 2026.
- LE, T. et al. **The tree-based pipeline optimization tool: Tackling biomedical research problems with genetic programming and automated machine learning**. 2025. Disponível em: https://www.sciencedirect.com/. Acesso em: 28 jan. 2026.
- MLFLOW. **MLflow**. 2025. Disponível em: https://mlflow.org/. Acesso em: 28 jan. 2026.
- PATI, S. K. **Best Practices in Data Science — Part 1 (Organizing and Coding)**. 2025. Disponível em: https://medium.com/thedeephub/best-practices-for-organizing-and-coding-data-science-projects-part-1-72539e14a7a0. Acesso em: 28 jan. 2026.
- SCIKIT-LEARN. **Common pitfalls and recommended practices**. 2023. Disponível em: https://scikit-learn.org/stable/common_pitfalls.html. Acesso em: 28 jan. 2026.
- SCIKIT-LEARN. **Feature selection**. 2023. Disponível em: https://scikit-learn.org/stable/modules/feature_selection.html. Acesso em: 28 jan. 2026.
- SCIKIT-LEARN. **Tuning the hyper-parameters of an estimator**. 2023. Disponível em: https://scikit-learn.org/stable/modules/grid_search.html. Acesso em: 28 jan. 2026.
- SONY, R. K. **Heart Disease Data – Kaggle**. 2023. Disponível em: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data. Acesso em: 28 jan. 2026.
- UCI MACHINE LEARNING REPOSITORY. **Heart Disease**. 1988. Disponível em: https://archive.ics.uci.edu/dataset/45/heart+disease. Acesso em: 28 jan. 2026.
- WHITFIELD, B. **Feature Engineering Explained**. 2025. Disponível em: https://builtin.com/articles/feature-engineering. Acesso em: 28 jan. 2026.
