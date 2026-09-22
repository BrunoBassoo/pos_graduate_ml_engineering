# Aula 02 — Regressão Linear e Regularização

## Visão Geral
Aula sobre como manter modelos de regressão linear confiáveis em produção usando práticas de observabilidade inspiradas em sistemas distribuídos (vitalidade, prontidão e inicialização), combinadas a um controle explícito da "capacidade estatística" do modelo via regularização (Ridge/L2, Lasso/L1 e Elastic Net). A meta é detectar rapidamente degradação de modelo do ponto de vista do negócio e agir de forma objetiva, em vez de depender de intuição ou intervenções tardias.

## Tópicos Abordados
- Checagens de vitalidade, prontidão e inicialização em modelos lineares em produção
- Hands On: pipeline completo de regressão linear e regularização com scikit-learn e statsmodels sobre dados sintéticos (renda, quartos, latitude, longitude, bairro, condição)
- Pré-processamento com `ColumnTransformer` (StandardScaler + OneHotEncoder) dentro de `Pipeline`
- Baseline OLS (`LinearRegression`) e regularização com Ridge, Lasso e Elastic Net
- Ajuste de hiperparâmetros com `KFold` e `GridSearchCV`
- Expansão de complexidade com `PolynomialFeatures` e risco de overfitting
- "Caminho de regularização" do Lasso (comportamento dos coeficientes ao variar alpha)
- Métricas RMSE, MAE e R² e sua leitura operacional
- Diagnóstico estatístico com `statsmodels`: resíduos, testes de heterocedasticidade/normalidade, multicolinearidade e pontos influentes
- Operação em produção: prontidão, vitalidade e inicialização de pipelines de regressão
- Mercado, cases e tendências: data leakage, multicolinearidade, escalabilidade (OLS vs. solvers iterativos) e inferência on-device

## Conceitos-Chave

### Observabilidade de modelos lineares em produção
- **Vitalidade**: avalia se o modelo permanece estatisticamente saudável, monitorando RMSE/MAE/R², resíduos e diagnósticos (heterocedasticidade, normalidade, multicolinearidade, influência). Quando falha de forma sustentada, dispara correções automatizadas (re-treino com Ridge/Lasso/Elastic Net ou rollback para um baseline mais simples).
- **Prontidão**: decide se o modelo pode receber tráfego, checando alinhamento entre pipeline de pré-processamento, esquema de features, versões de artefatos e resultados de shadow testing frente à versão em produção.
- **Inicialização**: reconhece que alguns pipelines (muitas dummies, carregamento de transformadores e coeficientes) precisam de tempo para "aquecer", evitando falsos positivos de falha logo após o start do serviço.

### Pipeline de pré-processamento
Separar colunas numéricas (StandardScaler) das categóricas (OneHotEncoder com `handle_unknown='ignore'`) dentro de um `ColumnTransformer`, encadeado a um `Pipeline` com o estimador. Isso fixa a ordem das transformações, evita data leakage (o pipeline não "aprende" com o conjunto de teste) e garante reprodutibilidade entre notebook, API e execução em lote.

### Métricas: RMSE, MAE e R²
- **RMSE** (raiz do erro quadrático médio): mesma unidade do alvo, penaliza erros grandes.
- **MAE** (erro absoluto médio): mais robusto a outliers, leitura direta do erro típico.
- **R²**: fração da variância do alvo explicada pelas features.
Comparar essas métricas em treino e teste revela overfitting (erro de teste maior, R² menor) e underfitting (erros altos em ambos). Em produção, faixas de tolerância dessas métricas viram os próprios limites de prontidão e vitalidade.

### Regularização: controle da complexidade
- **Ridge (L2)**: adiciona penalidade λ‖β‖², encolhe coeficientes e melhora o condicionamento quando XᵀX é mal-condicionado; raramente zera termos — útil sob alta correlação.
- **Lasso (L1)**: adiciona penalidade λ‖β‖, promove sparsidade (zera coeficientes) e atua como seleção de variáveis.
- **Elastic Net**: combina L1 e L2 via `alpha` e `l1_ratio`, equilibrando seleção e estabilidade em grupos de preditores correlacionados.
Pouca penalização aumenta variância (overfitting); penalização excessiva eleva viés (perda de sinal) — daí a necessidade de padronizar as variáveis antes de Lasso/Elastic Net e escolher λ/α por validação cruzada.

### Diagnóstico com statsmodels
Um ajuste OLS via `statsmodels` fornece coeficientes, erros-padrão e significância, além de possibilitar inspeção de resíduos. Testes e indicadores citados:
- **Breusch–Pagan**: heterocedasticidade.
- **Shapiro–Wilk**: normalidade dos resíduos.
- **VIF (Variance Inflation Factor)**, **leverage** e **distância de Cook**: multicolinearidade e pontos influentes.
Esses sinais orientam ações concretas (transformações como log, tratamento de outliers, revisão de engenharia de variáveis ou adoção de regularização).

### Seleção de hiperparâmetros e validação
`KFold` + `GridSearchCV` para escolher λ/α, usando `scoring='neg_root_mean_squared_error'` (valor negativo por convenção do scikit-learn, invertido na leitura). Em problemas temporais, `TimeSeriesSplit` substitui `KFold` para não vazar informação do futuro. Para avaliações mais rigorosas, validação aninhada (nested CV) separa tuning de aferição de desempenho.

### Complexidade com PolynomialFeatures
Adicionar termos polinomiais (ex.: grau 3) amplia a capacidade do modelo de capturar não linearidades, mas eleva o risco de overfitting — especialmente combinado a OneHotEncoder, que já aumenta o número de colunas. A prática recomendada é comparar RMSE/MAE em treino, teste e CV, aplicando Ridge/Lasso/Elastic Net quando necessário para conter a variância.

### Escalabilidade: OLS fechado vs. solvers iterativos
Em bases pequenas, a solução fechada do OLS (β̂ = (XᵀX)⁻¹Xᵀy) é direta. Em bases grandes (milhões de linhas, milhares de features), inverter XᵀX fica caro em tempo e memória — o caminho prático é usar solvers iterativos como Gradient Descent/SGD (mini-batch, regularização, escala de features), trocando "exatidão única" por convergência rápida e escalável. Na prática: `Ridge(solver='sag'/'saga'/'lsqr'/'sparse_cg')` ou `SGDRegressor` para 10⁵+ exemplos/features.

## Exercício Hands-On (do material)
Cenário: dataset tabular sintético com variáveis numéricas (renda, quartos, latitude, longitude) e categóricas (bairro, condição), com o objetivo de construir um serviço de previsão baseado em regressão.

Passos do hands-on:
1. Geração de dados sintéticos com `numpy`/`pandas` e split treino/teste (80/20).
2. Pipeline de baseline: `ColumnTransformer` (StandardScaler + OneHotEncoder) + `LinearRegression`, avaliado com RMSE_train, RMSE_test, MAE_test e R2_test.
3. Três pipelines alternativos (Ridge, Lasso, ElasticNet) reaproveitando o mesmo pré-processamento, ajustados via `GridSearchCV` com `KFold(n_splits=5)` sobre grades de `alpha` (e `l1_ratio` no Elastic Net).
4. Pipeline com `PolynomialFeatures(degree=3, include_bias=False)` nas colunas numéricas para testar ganho de complexidade vs. overfitting.
5. Varredura de 30 valores de `alpha` em escala logarítmica para o Lasso, guardando o vetor de coeficientes a cada passo — o "caminho de regularização", que mostra como a penalização L1 zera coeficientes progressivamente (sparsidade).

## Exemplos de Código

Geração dos dados sintéticos e split treino/teste.

```python
# Setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import statsmodels.api as sm

np.random.seed(42)
N = 3000
X_num = pd.DataFrame({
    'renda': np.random.normal(50000, 10000, N),
    'quartos': np.random.randint(1, 6, N),
    'latitude': np.random.uniform(32, 42, N),
    'longitude': np.random.uniform(-124, -114, N)
})
X_cat = pd.DataFrame({
    'bairro': np.random.choice(['A','B','C','D'], size=N),
    'condicao': np.random.choice(['nova','usada'], size=N)
})
X = pd.concat([X_num, X_cat], axis=1)
y = 100000 + 3*X_num['renda'] + 10000*X_num['quartos'] - 5000*(X_num['latitude']-37)**2 + np.random.normal(0, 50000, N)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Fonte: Elaborado pelo autor (2026)

Pipeline de baseline com pré-processamento e regressão linear (OLS).

```python
num_cols = ['renda','quartos','latitude','longitude']
cat_cols = ['bairro','condicao']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

ols = Pipeline([
    ('prep', preprocess),
    ('model', LinearRegression())
])

ols.fit(X_train, y_train)
pred_tr = ols.predict(X_train)
pred_te = ols.predict(X_test)
print({
    'RMSE_train': mean_squared_error(y_train, pred_tr, squared=False),
    'RMSE_test': mean_squared_error(y_test, pred_te, squared=False),
    'MAE_test': mean_absolute_error(y_test, pred_te),
    'R2_test': r2_score(y_test, pred_te)
})
```
Fonte: Elaborado pelo autor (2026)

Pipelines regularizados (Ridge, Lasso, Elastic Net) com busca de hiperparâmetros via validação cruzada.

```python
ridge = Pipeline([('prep', preprocess), ('model', Ridge())])
lasso = Pipeline([('prep', preprocess), ('model', Lasso(max_iter=10000))])
elastic = Pipeline([('prep', preprocess), ('model', ElasticNet(max_iter=10000))])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
pr = {'model__alpha':[0.1,1,10,100]}
pl = {'model__alpha':[0.001,0.01,0.1,1]}
pe = {'model__alpha':[0.001,0.01,0.1,1], 'model__l1_ratio':[0.2,0.5,0.8]}

GR = GridSearchCV(ridge, pr, cv=cv, scoring='neg_root_mean_squared_error')
GL = GridSearchCV(lasso, pl, cv=cv, scoring='neg_root_mean_squared_error')
GE = GridSearchCV(elastic, pe, cv=cv, scoring='neg_root_mean_squared_error')

for g in [GR, GL, GE]:
    g.fit(X_train, y_train)

print('Ridge:', GR.best_params_, -GR.best_score_)
print('Lasso:', GL.best_params_, -GL.best_score_)
print('Elastic:', GE.best_params_, -GE.best_score_)
```
Fonte: Elaborado pelo autor (2026)

Expansão polinomial de grau 3 nas features numéricas, mantendo one-hot nas categóricas.

```python
poly_prep = ColumnTransformer([
    ('num', Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3, include_bias=False))]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
poly_ols = Pipeline([('prep', poly_prep), ('model', LinearRegression())])
poly_ols.fit(X_train, y_train)
print('RMSE_test (poly grau 3):', mean_squared_error(y_test, poly_ols.predict(X_test), squared=False))
```
Fonte: Elaborado pelo autor (2026)

Caminho de regularização do Lasso: como os coeficientes encolhem (e zeram) à medida que `alpha` cresce.

```python
alphas = np.logspace(-3, 0, 30)
coefs = []
for a in alphas:
    model = Pipeline([('prep', preprocess), ('model', Lasso(alpha=a, max_iter=10000))])
    model.fit(X_train, y_train)
    coefs.append(model.named_steps['model'].coef_)
coefs = np.array(coefs)
print('Shape dos coeficientes ao longo do caminho:', coefs.shape)
```
Fonte: Elaborado pelo autor (2026)

## Cases e Tendências de Mercado
- **Data leakage em regressão de risco financeiro (2025)**: um time de ciência de dados descobriu que normalizações e seleção de variáveis eram feitas antes do split treino/teste, e havia features com informação futura (target leakage). Isso inflava métricas offline e derrubava a performance em produção. A correção envolveu isolar features sensíveis em pipelines dedicados (fit apenas no treino), bloquear variáveis derivadas do alvo e adotar inspeções automáticas de leakage — resultando em métricas offline mais conservadoras, porém generalização estável em produção.
- **Redução de dimensionalidade em e-commerce**: um e-commerce com centenas de preditores tabulares migrou de OLS para Lasso e depois Elastic Net, reduzindo a dimensionalidade efetiva (coeficientes zerados via L1), deixando o scoring mais leve e melhorando a interpretabilidade sem perda de acurácia — menos features significa menos I/O, menos cache churn e menor custo de manutenção de ETLs.
- **Multicolinearidade em regressão de preços (varejo)**: uma startup de varejo viu coeficientes explodirem e inverterem sinal após um update de features, causado por multicolinearidade. A recuperação envolveu diagnóstico via VIF, remoção/combinação de preditores redundantes (ou PCA) e adoção de Ridge como baseline regularizado.
- **Inferência on-device (Core ML)**: equipes de apps de áudio/vídeo relatam ganhos ao rodar inferência local com Core ML, eliminando latência de rede e mantendo privacidade, com respostas de baixa latência (sub-100ms) mesmo offline.
- **Escalabilidade de treino**: em bases grandes, a solução fechada de OLS fica cara para inverter XᵀX; a prática de mercado é usar solvers iterativos (Ridge com `solver='sag'/'saga'`, ou `SGDRegressor`) para 10⁵+ exemplos/features.

## Checklist de Estudo
- [ ] Sei explicar as checagens de vitalidade, prontidão e inicialização de um modelo linear em produção
- [ ] Sei montar um pipeline reprodutível com `ColumnTransformer` (StandardScaler + OneHotEncoder) e `Pipeline`
- [ ] Sei diferenciar Ridge (L2), Lasso (L1) e Elastic Net e quando usar cada um
- [ ] Sei interpretar RMSE, MAE e R² em treino/teste para identificar overfitting e underfitting
- [ ] Sei descrever os testes de diagnóstico do statsmodels (Breusch-Pagan, Shapiro-Wilk, VIF, leverage, distância de Cook)
- [ ] Sei explicar por que PolynomialFeatures aumenta o risco de overfitting e como mitigar com regularização
- [ ] Sei por que bases grandes preferem solvers iterativos (SGD) em vez da solução fechada do OLS

## Palavras-chave
Regressão Linear. Regularização. Pipeline.

## Referências
- BREUSCH, T. S.; PAGAN, A. R. *A Simple Test for Heteroscedasticity and Random Coefficient Variation*. Econometrica, v. 47, n. 5, p. 1287–1294, 1979.
- EFRON, B.; HASTIE, T.; JOHNSTONE, I.; TIBSHIRANI, R. *Least Angle Regression*. Annals of Statistics, v. 32, n. 2, p. 407–499, 2004.
- FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. *Regularization Paths for Generalized Linear Models via Coordinate Descent*. Journal of Statistical Software, v. 33, n. 1, p. 1–22, 2010.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. New York: Springer, 2009.
- HOERL, A. E.; KENNARD, R. W. *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics, v. 12, n. 1, p. 55–67, 1970.
- JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. *An Introduction to Statistical Learning: with Applications in R*. 2. ed. New York: Springer, 2021.
- KUTNER, M. H.; NACHTSHEIM, C. J.; NETER, J. *Applied Linear Regression Models*. 4. ed. New York: McGraw-Hill/Irwin, 2004.
- MONTGOMERY, D. C.; PECK, E. A.; VINING, G. G. *Introduction to Linear Regression Analysis*. 6. ed. Hoboken: Wiley, 2021.
- SCIKIT-LEARN. *User Guide & API Reference*. scikit-learn Documentation. Disponível em: https://scikit-learn.org/. Acesso em: 23 jan. 2026.
- STATSMODELS. *Documentation*. statsmodels.org. Disponível em: https://www.statsmodels.org/. Acesso em: 23 jan. 2026.
- TIBSHIRANI, R. *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society: Series B, v. 58, n. 1, p. 267–288, 1996.
- ZOU, H.; HASTIE, T. *Regularization and Variable Selection via the Elastic Net*. Journal of the Royal Statistical Society: Series B, v. 67, n. 2, p. 301–320, 2005.
