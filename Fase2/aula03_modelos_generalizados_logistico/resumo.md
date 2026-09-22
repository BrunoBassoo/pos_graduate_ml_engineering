# Aula 03 — Modelos Lineares Generalizados e Regressão Logística

## Visão Geral
Aula focada em Modelos Lineares Generalizados (GLM), com ênfase na Regressão Logística como técnica de classificação binária. Parte da função sigmoide para transformar uma combinação linear de variáveis em uma probabilidade entre 0 e 1, apresenta os conceitos de odds e log-odds, treina um modelo logístico em Python/scikit-learn e conecta tudo à estrutura mais ampla dos GLMs (componente sistemático + componente aleatório + função de ligação), incluindo a estimação por Máxima Verossimilhança (MLE) e cuidados de modelagem como regularização, multicolinearidade e outliers.

## Tópicos Abordados
- Função sigmoide: definição, limites, valor central e interpretação como probabilidade
- Regressão logística binária: P(Y=1|X), limiar de decisão (0,5) para classificar
- Odds e log-odds (logit) e a relação linear do logit com as variáveis de entrada
- Treinamento e avaliação de um modelo de regressão logística com scikit-learn (acurácia, matriz de confusão, precisão, recall, F1-score)
- Modelos Lineares Generalizados (GLM): componente sistemático, componente aleatório, função de ligação
- Estimação de parâmetros por Máxima Verossimilhança (MLE) e métodos iterativos (Newton-Raphson, IRLS, gradiente descendente)
- Regularização L1 (Lasso) e L2 (Ridge) para evitar overfitting
- Multicolinearidade e outliers como riscos de modelagem
- Aplicações de mercado: saúde, marketing digital, crédito/fintech, detecção de fraude, previsão eleitoral

## Conceitos-Chave

### Função Sigmoide
σ(z) = 1 / (1 + e⁻ᶻ), mapeando qualquer número real para o intervalo (0, 1).
- lim_{z→−∞} σ(z) = 0 e lim_{z→+∞} σ(z) = 1
- σ(0) = 0,5
- Interpretação: z ≪ 0 → evento muito improvável; z ≫ 0 → evento muito provável; z = 0 → 50% de chance.

### Regressão Logística (GLM binomial com link logit)
Modela g(μ) = logit(μ) = log(μ/(1-μ)) = Xβ ⇒ μ = σ(Xβ), onde μ é a probabilidade estimada de Y=1 dado o vetor de características X. Define-se um limiar (tipicamente 0,5) para converter a probabilidade em classe prevista (0 ou 1).

### Odds e Log-Odds
- Odds = P(Y=1) / (1 − P(Y=1)) — razão entre a chance de ocorrer e de não ocorrer o evento.
- Log-odds (logit) = log(odds), modelado como combinação linear: log(P(Y=1)/(1-P(Y=1))) = β₀ + β₁X₁ + ... + βₙXₙ.
- A transformação logarítmica converte efeitos multiplicativos no odds em efeitos aditivos no log-odds, capturados pelos coeficientes β: um β₁ = 0,5 implica que aumentar X₁ em uma unidade multiplica o odds por e^0,5 ≈ 1,65 (aumento de 65% na chance do evento), mantendo as demais variáveis constantes.

### Modelos Lineares Generalizados (GLM)
A regressão logística é um caso particular de GLM. Um GLM tem:
- **Componente sistemático**: combinação linear das variáveis independentes (preditor linear).
- **Componente aleatório**: distribuição de probabilidade da variável resposta (Bernoulli/Binomial para eventos binários, Poisson para contagens, Gaussiana para contínuos, etc.).
- **Função de ligação (link function)**: conecta o valor esperado da resposta ao preditor linear — no caso logístico, a função logit.
Estrutura formalizada por Nelder & Wedderburn (1972). Outros exemplos de GLM: regressão de Poisson (link log, para contagens) e regressão Gamma (dados positivos assimétricos).

### Estimação por Máxima Verossimilhança (MLE)
Não há fórmula fechada para os coeficientes da regressão logística (diferente da regressão linear/OLS). O ajuste é feito maximizando a função de log-verossimilhança:
ℓ(β) = Σ [yᵢ log P(yᵢ) + (1-yᵢ) log(1-P(yᵢ))]
via métodos iterativos: Newton-Raphson (equivalente ao algoritmo IRLS — Iteratively Reweighted Least Squares — em GLMs) ou métodos de primeira ordem como Gradiente Descendente/SGD. Na prática, bibliotecas como scikit-learn usam solvers como LIBLINEAR ou LBFGS internamente.

### Regularização e Cuidados de Modelagem
- **Regularização L2 (Ridge)** e **L1 (Lasso)**: penalizam coeficientes grandes (||β||² ou ||β||₁) para reduzir overfitting; no scikit-learn, L2 é aplicada por padrão via o parâmetro `C` (inverso da intensidade da regularização).
- **Multicolinearidade**: variáveis independentes muito correlacionadas tornam os coeficientes instáveis e difíceis de interpretar; mitigação via remoção/combinação de variáveis ou PCA.
- **Limitações**: pressupõe relação monotônica/linear no logit entre preditores e resposta; sensível a outliers; deve sempre ser validada em dados não vistos (conjunto de teste ou validação cruzada) para evitar sub/sobreajuste.

## Exemplos de Código

Cálculo da função sigmoide para diferentes valores de z:
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

valores = np.array([-4, -1, 0, 1, 4])
probs = sigmoid(valores)
print("Valores de z: ", valores)
print("Probabilidades σ(z):", np.round(probs, 4))
# Probabilidades σ(z): [0.0179 0.2689 0.5 0.7311 0.9820]
```

Cálculo de odds e log-odds a partir de uma probabilidade:
```python
import math

p = 0.8  # 80% de chance do evento ocorrer
odds = p / (1 - p)
log_odds = math.log(odds)
print(f"Probabilidade = {p}")
print(f"Odds = {odds:.2f}")        # Odds = 4.00
print(f"Log-odds = {log_odds:.3f}") # Log-odds = 1.386
```

Geração de dados sintéticos (idade e renda prevendo probabilidade de compra):
```python
import numpy as np
import pandas as pd

np.random.seed(42)
idades = np.random.randint(18, 61, size=100)
rendas = np.random.randint(1000, 10001, size=100)
prob_compra = 1 / (1 + np.exp(-(-8 + 0.0003*rendas + 0.05*idades)))
y = (prob_compra >= 0.5).astype(int)
dados = pd.DataFrame({'idade': idades, 'renda': rendas, 'comprou': y})
```

Treino e avaliação de um modelo de regressão logística com scikit-learn:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = dados[['idade', 'renda']]
y = dados['comprou']

modelo = LogisticRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)

acc = accuracy_score(y, y_pred)
mat_conf = confusion_matrix(y, y_pred)
relatorio = classification_report(y, y_pred, output_dict=False)

print(f"Acurácia: {acc:.3f}")   # Acurácia: 0.90
print("Matriz de Confusão:")
print(mat_conf)                  # [[45 5] [5 45]]
print("Relatório de Classificação:")
print(relatorio)                 # precision/recall/F1 = 0.90 em ambas as classes
```

## Cases e Tendências de Mercado
- **Saúde**: regressão logística usada para estimar risco de doenças (ex.: probabilidade de diabetes a partir de idade, IMC e glicose), auxiliando decisões clínicas com fatores de risco interpretáveis.
- **Marketing digital**: previsão de conversão e churn (ex.: probabilidade de clique em anúncio, assinatura de serviço, compra em e-commerce), permitindo ações segmentadas como cupons direcionados.
- **Bancos e fintechs**: credit scoring (probabilidade de inadimplência) e detecção de fraude (probabilidade de uma transação ser fraudulenta), com forte valorização da interpretabilidade dos coeficientes por conta da regulação do setor.
- **Política**: previsão de resultados eleitorais e comportamento de voto por grupo demográfico, orientando estratégias de campanha.
- Mesmo com a ascensão de modelos mais complexos (redes neurais, ensembles), a regressão logística permanece relevante como **baseline** e ganha destaque no movimento de **Explainable AI**, graças à interpretabilidade dos coeficientes via odds ratio — especialmente valiosa em saúde, finanças e contextos regulatórios.

## Checklist de Estudo
- [ ] Sei explicar a função sigmoide, seus limites e como ela vira uma probabilidade
- [ ] Sei diferenciar odds de log-odds e explicar por que a regressão logística modela o logit linearmente
- [ ] Sei treinar e avaliar um modelo de regressão logística no scikit-learn (acurácia, matriz de confusão, precisão, recall, F1-score)
- [ ] Sei descrever os três componentes de um GLM (sistemático, aleatório, função de ligação) e situar a regressão logística nessa família
- [ ] Sei explicar por que não há fórmula fechada para os coeficientes da regressão logística e como MLE/Newton-Raphson/IRLS resolvem isso
- [ ] Sei explicar o papel da regularização L1/L2 e os riscos de multicolinearidade e outliers
- [ ] Consigo citar aplicações da regressão logística em pelo menos três setores (saúde, marketing, finanças/política)

## Palavras-chave
Regressão Logística. Odds. Logit Link.

## Referências
- BARROS, L. *Regressão Logística: Entendendo um dos Modelos de Classificação Mais Utilizados*. 2024. Disponível em: https://alemdosdados.tech/regressao-logistica-entendendo-um-dos-modelos-de-classificacao-mais-utilizados/. Acesso em: 23 fev. 2026.
- EBAC. *Regressão Logística: O que é, tipos, vantagens, desvantagens e casos de uso*. 2025. Disponível em: https://ebaconline.com.br/blog/o-que-e-regressao-logistica. Acesso em: 23 fev. 2026.
- GOMES, P. C. T. *Regressão Logística: Um Guia Detalhado para Machine Learning*. 2024. Disponível em: https://www.datageeks.com.br/regressao-logistica/. Acesso em: 23 fev. 2026.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. New York: Springer, 2009. Disponível em: https://hastie.su.domains/ElemStatLearn/. Acesso em: 23 fev. 2026.
- HONÓRIO, R. *Entendendo a regressão logística: conceitos, aplicações e exemplos práticos*. 2024. Disponível em: https://hub.asimov.academy/blog/entendendo-a-regressao-logistica/. Acesso em: 23 fev. 2026.
- HOSMER, D. W.; LEMESHOW, S.; STURDIVANT, R. X. *Applied Logistic Regression*. 3. ed. Hoboken: Wiley, 2013. DOI: 10.1002/9781118548387. Disponível em: https://www.wiley.com/en-us/Applied+Logistic+Regression%2C+3rd+Edition-p-9780470582473. Acesso em: 23 fev. 2026.
- JAMES, G.; WITTEN, D.; HASTIE, T.; TIBSHIRANI, R. *An Introduction to Statistical Learning*. New York: Springer, 2013. Disponível em: https://www.statlearning.com/. Acesso em: 23 fev. 2026.
- NELDER, J. A.; WEDDERBURN, R. W. M. Generalized Linear Models. *Journal of the Royal Statistical Society: Series A*, v. 135, n. 3, p. 370–384, 1972. DOI: 10.2307/2344614. Disponível em: https://academic.oup.com/jrsssa/article/135/3/370/7110572. Acesso em: 23 fev. 2026.
- WIKIPEDIA. *Iteratively Reweighted Least Squares*. 2023. Disponível em: https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares. Acesso em: 23 fev. 2026.
