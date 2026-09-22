# Aula 04 — Árvores de Decisão e Ensembles

## Visão Geral
Aula que revisita os fundamentos das árvores de decisão (critérios de impureza, overfitting/underfitting, poda) e avança para os métodos **Ensemble**, que combinam múltiplos modelos para produzir previsões mais estáveis e precisas. São explorados dois paradigmas centrais: **Bagging** (com destaque para **Random Forest**) e **Boosting** (com foco em **Gradient Boosting**, base de XGBoost e LightGBM). O fio condutor é o cenário de uma operadora de telecomunicações prevendo churn de clientes.

## Tópicos Abordados
- Dilema overfitting x underfitting em árvores de decisão
- Hands On: árvore de decisão simples, Random Forest e Gradient Boosting no dataset Iris (scikit-learn)
- Critérios de divisão: Índice de Gini e Entropia/Ganho de Informação
- Controle de complexidade: parada antecipada (max_depth, min_samples_split/leaf) e poda (pruning, cost-complexity pruning)
- Dilema Viés–Variância e como Bagging e Boosting o atacam de formas diferentes
- Bagging e Random Forest em detalhes (bootstrap, subespaço aleatório de atributos, erro Out-of-Bag)
- Boosting: AdaBoost e Gradient Boosting (ajuste de resíduos via gradiente), XGBoost/LightGBM/CatBoost
- Hiperparâmetros e tuning (GridSearchCV, curvas de treino/validação)
- Mercado, Cases e Tendências: churn em telecom, detecção de fraudes, competições Kaggle, explicabilidade (SHAP, EBM)

## Conceitos-Chave

### Critérios de Impureza
- **Índice de Gini**: $Gini = 1 - \sum_i p_i^2$. Vale 0 quando o nó é puro e é máximo (0,5 no caso binário) quando há mistura máxima (p=0,5). Tende a isolar a classe mais frequente em um dos ramos.
- **Entropia**: $Entropia = -\sum_i p_i \log_2 p_i$. Derivada da Teoria da Informação de Shannon; 0 quando o nó é puro, máxima (1 bit) quando p=0,5. Tende a criar divisões mais balanceadas entre os lados.
- **Ganho de Informação**: redução na impureza média ponderada dos filhos em relação ao pai; a divisão escolhida é a que maximiza esse ganho. Algoritmos como CART, ID3/C4.5 usam esse princípio para escolher a "melhor pergunta" em cada nó.

### Complexidade da Árvore: Parada e Poda
- **Parada antecipada**: `max_depth`, `min_samples_split`, `min_samples_leaf` — limitam o crescimento e atuam como regularização.
- **Poda posterior (pruning)**: gera a árvore completa e remove nós/folhas pouco relevantes; técnica de referência é o *cost-complexity pruning* do CART.
- Limitar/podar aumenta viés e reduz variância; árvore totalmente crescida tem variância altíssima e viés baixo nos dados de treino.

### Dilema Viés–Variância e Papel dos Ensembles
- **Bagging** (bootstrap aggregating) reduz **variância** de modelos complexos e de alta variância (árvores profundas), treinando modelos independentes em paralelo e agregando (votação/média).
- **Boosting** reduz **viés** de modelos simples ("fracos"), treinando modelos sequencialmente, cada um focando nos erros do anterior; resultado final é combinação ponderada, com maior risco de sobreajuste se exagerado.

### Bagging e Random Forest
- Bagging cria $m$ conjuntos por bootstrap (amostragem com reposição); em média ~36,8% dos exemplos ficam fora de cada bootstrap (amostra *out-of-bag*).
- **Random Forest** (Breiman, 2001) soma ao bagging a seleção aleatória de um subconjunto de atributos ($k$) em cada nó, reduzindo a correlação entre árvores.
- Regra prática de `max_features`: $\sqrt{d}$ em classificação, $d/3$ em regressão.
- Fornece **erro Out-of-Bag (OOB)** como estimativa de generalização sem conjunto de validação separado, e **feature importance** para interpretabilidade parcial.
- Costuma ser um baseline robusto: pouco tuning crítico, boa performance sem muito risco de overfitting ao aumentar `n_estimators` (retorno decrescente após algumas centenas de árvores).

### Boosting e Gradient Boosting
- **AdaBoost**: treina sequência de "stumps" (árvores de 1 nível), aumentando o peso das observações mal classificadas a cada iteração; combina classificadores ponderando pela acurácia.
- **Gradient Boosting**: generaliza o boosting como otimização de função de perda via descida de gradiente funcional — cada nova árvore aprende a prever o **resíduo** do modelo atual, somado ao conjunto com peso `learning_rate` (shrinkage).
- **XGBoost** (2016), **LightGBM** e **CatBoost**: implementações state-of-the-art, com regularização L1/L2 nos pesos das folhas, subsampling de instâncias/atributos, e alta performance computacional.
- Modelos-base do boosting costumam ser árvores rasas (3–8 níveis), diferente do bagging que usa árvores profundas.

### Hiperparâmetros e Tuning
- **Árvore simples**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `max_features`.
- **Random Forest**: os anteriores + `n_estimators`, `max_features`, `bootstrap` (True/False — se False vira *pasting*), `oob_score`.
- **Gradient Boosting/XGBoost/LightGBM**: `n_estimators`, `learning_rate`, `max_depth`/`num_leaves`, `subsample`, `colsample_bytree`, penalizações L1/L2, `min_child_weight`/`min_data_in_leaf`. Trade-off central entre `n_estimators` e `learning_rate` (λ pequeno, tipicamente 0,01–0,2, exige mais árvores mas generaliza melhor).
- Tuning feito via validação cruzada e busca em grade (`GridSearchCV`); curvas de treino/validação ajudam a identificar o "sweet spot" de iterações (erro de validação do boosting cai e depois sobe; no Random Forest tende a estabilizar).

## Exercício Hands-On (do material)
Sequência prática em Python com o dataset Iris (scikit-learn), comparando três modelos:
1. **Árvore de decisão simples** sem restrição de profundidade (`DecisionTreeClassifier`) — acurácia ~0,98, discussão sobre risco de overfitting e técnicas de pruning/critérios de divisão (Gini vs. Entropia).
2. **Random Forest** com 100 árvores (`RandomForestClassifier`, `n_estimators=100`) — acurácia 1,00, discussão sobre redução de variância via bagging, amostra out-of-bag e trade-off precisão x interpretabilidade.
3. **Gradient Boosting** (`GradientBoostingClassifier`, `n_estimators=100`, `learning_rate=0.1`) — desempenho comparável, discussão sobre redução de viés e hiperparâmetros de regularização.
4. Exemplo de **busca em grade** (`GridSearchCV`) para encontrar `max_depth` e `min_samples_leaf` ótimos de uma árvore de decisão via validação cruzada (cv=5).

## Exemplos de Código

Árvore de decisão simples sobre o dataset Iris, com avaliação de acurácia e matriz de confusão.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carrega dados de exemplo (flores Iris)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=42
)

# Cria e treina uma Árvore de Decisão sem restrições de profundidade
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)

# Realiza previsões no conjunto de teste
y_pred = modelo_arvore.predict(X_test)

# Avalia a acurácia e gera a matriz de confusão
acuracia = accuracy_score(y_test, y_pred)
mat_confusao = confusion_matrix(y_test, y_pred)
print(f"Acurácia da Árvore de Decisão: {acuracia:.2f}")
print("Matriz de Confusão:\n", mat_confusao)
```

Random Forest com 100 árvores sobre o mesmo conjunto de dados.

```python
from sklearn.ensemble import RandomForestClassifier

# Instancia e treina uma Random Forest com 100 árvores (estimadores)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Avalia o desempenho no conjunto de teste
y_pred_rf = modelo_rf.predict(X_test)
acuracia_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acurácia da Random Forest: {acuracia_rf:.2f}")
```

Gradient Boosting Classifier no mesmo conjunto de dados.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Treina um modelo de Gradient Boosting (100 árvores de profundidade 3 padrão)
modelo_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
modelo_gb.fit(X_train, y_train)

# Avalia a performance no conjunto de teste
y_pred_gb = modelo_gb.predict(X_test)
acuracia_gb = accuracy_score(y_test, y_pred_gb)
print(f"Acurácia do Gradient Boosting: {acuracia_gb:.2f}")
```

Busca em grade (Grid Search) para encontrar os melhores hiperparâmetros de uma árvore de decisão via validação cruzada.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'max_depth': [None, 2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 5, 10]
}
modelo = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(modelo, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)
print("Acurácia (validação cruzada):", f"{grid.best_score_:.3f}")
```

## Cases e Tendências de Mercado
- **Caso 1 — Churn em Telecom (TeleSim, fictício)**: árvore de decisão isolada obtinha ~70% de acurácia e era instável entre amostras de treino. Uma Random Forest com 200 árvores elevou a acurácia para 80%, com maior estabilidade e importância de variáveis consistente (reclamações, tempo de contrato, excesso de franquia). Resultado: identificação antecipada de 15% mais clientes insatisfeitos e 10% menos cancelamentos mensais em seis meses.
- **Caso 2 — Detecção de Fraudes Financeiras**: sistema de duas camadas — Random Forest rápido como *screener* em tempo real, seguido de XGBoost mais pesado para refinar a decisão em casos suspeitos. Detecção de fraudes subiu de ~60% para ~85%, com falsos positivos controlados em ~3%. Uso de importância de variáveis revelou novos indicadores de fraude (sequência de declínios em curto intervalo).
- **Caso 3 — Competições e Tendências (Kaggle)**: XGBoost dominou competições como o "Otto Group Product Classification" (2015), consolidando ensembles de árvores como padrão em dados tabulares estruturados, inclusive superando redes neurais nesse domínio. Frameworks de AutoML (H2O, auto-sklearn) costumam entregar ensembles como solução final.
- **Tendências**: surgimento de algoritmos de *explainable boosting* (ex.: EBM da Microsoft Research); crescente uso de técnicas de explicabilidade pós-treino (SHAP, LIME) em setores regulados (financeiro, saúde) para justificar decisões de modelos caixa-preta; preferência por modelos simplificados em cenários de compliance; pesquisas sobre como ensembles podem amplificar ou mascarar vieses presentes nos dados.

## Checklist de Estudo
- [ ] Sei explicar os critérios de impureza Gini e Entropia e calcular o ganho de informação
- [ ] Sei diferenciar parada antecipada de poda (pruning) e seu efeito no trade-off viés–variância
- [ ] Sei explicar a diferença conceitual entre Bagging (reduz variância, paralelo) e Boosting (reduz viés, sequencial)
- [ ] Sei descrever como a Random Forest adiciona aleatoriedade além do bagging simples e o que é o erro Out-of-Bag
- [ ] Sei explicar como o Gradient Boosting ajusta resíduos via descida de gradiente e cito XGBoost/LightGBM como implementações populares
- [ ] Sei listar os principais hiperparâmetros de árvores, Random Forest e Gradient Boosting e como fazer tuning via validação cruzada/grid search
- [ ] Consigo relacionar os conceitos aos casos de churn em telecom e detecção de fraudes

## Palavras-chave
Árvores de decisão. Floresta Aleatória. Gradient Boosting.

## Referências
- BREIMAN, L. *Random Forests*. Machine Learning, v. 45, n. 1, p. 5–32, 2001. Disponível em: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf. Acesso em: 23 fev. 2026.
- GOMES, P. C. T. *Métodos de Ensemble Learning: Bagging, Boosting e Stacking*. 2024. Disponível em: https://www.datageeks.com.br/metodos-de-ensemble-learning/. Acesso em: 23 fev. 2026.
- GOMES, P. C. T. *O que é Floresta Aleatória e Como Aplicar esse Algoritmo?* 2024. Disponível em: https://www.datageeks.com.br/floresta-aleatoria/. Acesso em: 23 fev. 2026.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. New York: Springer, 2009. Disponível em: https://hastie.su.domains/ElemStatLearn/. Acesso em: 23 fev. 2026.
- HONÓRIO, R. *O que é Random Forest e como usar em Machine Learning*. 2025. Disponível em: https://hub.asimov.academy/blog/o-que-e-random-forest/. Acesso em: 23 fev. 2026.
- JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. *An Introduction to Statistical Learning*. New York: Springer, 2013. Disponível em: https://www.statlearning.com/. Acesso em: 23 fev. 2026.
- SHEIKH, C. *Medindo Entropia e Gini: Compreendendo Árvores de Decisão*. 2025. Disponível em: https://studyeasy.org/medindo-entropia-e-gini/. Acesso em: 23 fev. 2026.
- WIKIPÉDIA. *Floresta Aleatória*. 2024. Disponível em: https://pt.wikipedia.org/wiki/Floresta_aleat%C3%B3ria. Acesso em: 23 fev. 2026.
