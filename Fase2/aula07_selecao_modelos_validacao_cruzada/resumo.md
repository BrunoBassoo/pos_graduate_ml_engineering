# Aula 07 — Seleção de Modelos e Validação Cruzada

## Visão Geral
Um modelo que acerta quase tudo no conjunto de treino não garante bom desempenho em dados novos. Esta aula introduz a **validação cruzada (cross-validation)**, conjunto de técnicas estatísticas para estimar de forma confiável a performance de um modelo em dados não vistos, evitando conclusões enganosas causadas por uma única divisão treino/teste. A aula formaliza o problema de seleção de modelos como um problema de otimização, situa isso no equilíbrio **viés-variância**, e percorre variações práticas de validação (K-fold, LOO, estratificada, séries temporais, nested CV), critérios estatísticos de comparação, escolha de hiperparâmetros e boas práticas de governança em MLOps.

## Tópicos Abordados
- O que vem por aí — por que uma única divisão treino/teste é arriscada e como a validação cruzada resolve isso
- Hands On — seleção do grau de um polinômio via 5-fold cross-validation, comparando MSE de validação entre graus 1 a 7
- Saiba Mais
  - Dilema underfitting vs overfitting e decomposição bias-variance
  - Seleção de modelos como problema de otimização (AIC, BIC, grid search, random search, busca bayesiana)
  - Variações de validação cruzada: hold-out, Leave-One-Out (LOO), estratificada, séries temporais, repetição/média, nested CV
  - Testes estatísticos de significância entre modelos (teste t pareado, Wilcoxon, 5x2cv de Dietterich, McNemar, regra 1-SE)
  - Hiperparâmetros e validação: early stopping, regularização, arquitetura como hiperparâmetro
  - Custos computacionais da validação cruzada e estratégias de mitigação
  - Governança e boas práticas (data leakage, comitês de modelo, reprodutibilidade)
- Mercado, Cases e Tendências — Netflix Prize, cultura de validação no Kaggle, fiasco hipotético por falta de validação, automação de MLOps (Google Vizier), AutoML, monitoramento contínuo e revalidação em produção

## Conceitos-Chave

### Underfitting vs Overfitting e decomposição Bias-Variance
Modelos simples demais **subajustam** (não capturam padrões relevantes); modelos complexos demais **sobreajustam** (memorizam ruído do treino e falham em generalizar). O erro esperado de um modelo se decompõe em três termos:

E[(y − f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²

Modelos simples têm viés alto; modelos complexos têm variância alta; σ² é ruído irredutível. Existe um ponto ótimo de complexidade (compromisso viés-variância), e o papel da validação é identificar esse ponto na prática.

### Seleção de modelos como otimização
A seleção de hiperparâmetros/modelo pode ser formulada como minimizar o erro de teste real, aproximado pelo erro de validação (cross-validation ou hold-out), eventualmente penalizado por uma medida de complexidade Ω(h) — filosofia por trás de critérios como **AIC** (AIC = −2ln(L) + 2p) e **BIC**. Como o espaço de hiperparâmetros costuma ser grande (o número de configurações de um grid cresce como produtório dos valores por dimensão, podendo ser NP-difícil encontrar o ótimo), usam-se heurísticas de busca: **grid search** (exaustivo), **random search** (amostragem aleatória, muitas vezes mais eficiente que grid) e **busca bayesiana** (usa processos Gaussianos/florestas aleatórias para guiar a próxima tentativa).

### Variações de validação cruzada
- **Hold-out simples**: divisão única treino/teste (ex.: 80/20); barato, mas com maior variância na estimativa.
- **K-fold** (tipicamente K=5 ou 10): divide os dados em K partes, treina em K-1 e valida na restante, repetindo K vezes; aproveita todos os exemplos para treino e validação.
- **Leave-One-Out (LOO)**: caso extremo de K-fold com K=N; baixo viés, mas alta variância e alto custo computacional; útil com poucos dados.
- **Validação Estratificada**: preserva a proporção de classes em cada fold, essencial em datasets desbalanceados.
- **Validação em Séries Temporais**: não embaralha os dados — respeita a cronologia (blocked cross-validation, blocos deslizantes, gap validation), nunca usando dados futuros para treinar e validar no passado.
- **Repetição e média (ex.: 10x10-fold CV)**: repete a validação cruzada com partições diferentes para reduzir a variância da estimativa, a um custo computacional maior.
- **Nested Cross-Validation**: usa uma camada externa de CV para avaliar generalização e uma camada interna para selecionar hiperparâmetros, evitando otimismo no erro reportado quando o mesmo CV usado para escolher o modelo também é usado para reportar sua performance.

### Testes estatísticos e significância de diferenças
Diferenças pequenas de desempenho entre modelos podem não ser confiáveis. Métodos citados: teste t pareado, teste de Wilcoxon, teste **5x2cv** de Dietterich (1998) e teste de **McNemar** para comparação de acertos/erros pareados. A **regra do 1 desvio-padrão (1-SE rule)**, mencionada por Hastie et al. (2009), propõe escolher o modelo mais simples cujo desempenho esteja a no máximo 1 desvio-padrão do melhor modelo observado.

### Hiperparâmetros e validação
A linha entre "modelo" e "hiperparâmetro" é tênue: cada valor de um hiperparâmetro (ex.: λ de regularização Ridge) define de fato um modelo diferente, escolhido via validação. O **early stopping** usa um hold-out interno ao treinamento para parar quando o erro de validação começa a piorar, sendo amplamente usado em deep learning. Regularização (L1, L2, dropout), data augmentation e até a arquitetura de uma rede (número de camadas/neurônios) também são tratados como hiperparâmetros validados — o que hoje é levado ao extremo por abordagens de **AutoML** e **Neural Architecture Search**.

### Custos computacionais
Uma K-fold CV multiplica o custo de treino por K; buscas em hiperparâmetros multiplicam ainda mais. Estratégias de mitigação: treinar em amostras menores para filtrar candidatos ruins antes de treinar no dataset completo, paralelizar folds, e usar warm-start entre folds quando o algoritmo permite. Apesar do custo, a validação cruzada é considerada padrão-ouro de benchmarking porque evita apostar em um modelo errado em produção.

### Governança e boas práticas
Times de ML profissionais tratam a seleção de modelos com controle formal: conjuntos de teste separados e inacessíveis a quem desenvolve o modelo, comitês de revisão de experimentos, e atenção a **data leakage** (ex.: normalizar com estatísticas do dataset inteiro antes de dividir em folds contamina o treino com informação do teste — todo pré-processamento deve ocorrer dentro do ciclo de validação). Ferramentas como MLflow, Azure ML, SageMaker e Vertex AI registram experimentos, métricas e artefatos para reprodutibilidade e auditoria — relevante inclusive para compliance em setores regulados.

## Exercício Hands-On (do material)

Ajuste do grau de um polinômio a dados sintéticos de um polinômio cúbico com ruído, usando **5-fold cross-validation** para estimar o erro de generalização (MSE) de cada grau candidato (1 a 7), em vez de escolher pelo erro no próprio treino.

```python
import numpy as np

# 1. Dados sintéticos: y = 1 + 0.5x - 2x^2 + 0.3x^3 + ruído
np.random.seed(42)
N = 60
X = np.linspace(-3, 3, N).reshape(-1, 1)
y_true_fn = lambda x: 1 + 0.5*x - 2*x**2 + 0.3*x**3
y = y_true_fn(X) + np.random.normal(scale=5.0, size=X.shape)

indices = np.random.permutation(N)
K = 5
folds = np.array_split(indices, K)

# 2. K-fold cross-val: MSE médio de um polinômio de certo grau
def cross_val_mse(degree, K=5):
    errors = []
    for k in range(K):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        coeffs = np.polyfit(X_train.ravel(), y_train.ravel(), degree)
        y_pred = np.polyval(coeffs, X_val.ravel())
        mse_val = np.mean((y_val.ravel() - y_pred)**2)
        errors.append(mse_val)
    return np.mean(errors)

# 3. Avalia graus candidatos e escolhe o melhor
degrees = [1, 2, 3, 4, 5, 6, 7]
best_deg, best_mse = None, float("inf")
for deg in degrees:
    mse = cross_val_mse(deg, K=5)
    print(f"Modelo grau {deg}: MSE médio = {mse:.2f}")
    if mse < best_mse:
        best_mse, best_deg = mse, deg

print(f"\nMelhor modelo: grau {best_deg} (MSE médio {best_mse:.2f})")
```

Resultado obtido no material:

```
Modelo grau 1: MSE médio = 171.29
Modelo grau 2: MSE médio = 26.07
Modelo grau 3: MSE médio = 25.52
Modelo grau 4: MSE médio = 25.60
Modelo grau 5: MSE médio = 25.74
Modelo grau 6: MSE médio = 26.80
Modelo grau 7: MSE médio = 28.45

Melhor modelo: grau 3 (MSE médio 25.52)
```

O grau 3 (a ordem real do polinômio gerador) venceu; graus 1 subajustaram (erro alto) e graus 6-7 começaram a piorar por sobreajuste — exatamente o padrão que a validação cruzada existe para detectar, evitando que graus altos (que teriam MSE de treino quase zero) parecessem enganosamente melhores.

## Cases e Tendências de Mercado

- **Netflix Prize (2006–2009)**: o time vencedor obteve +10,06% de RMSE com um ensemble de centenas de modelos, mas a Netflix nunca colocou esse modelo em produção — a complexidade de engenharia não justificava o ganho marginal (~0,03 absoluto em RMSE). Mostra que a validação técnica confirma qual modelo é melhor, mas a decisão de adoção também pesa simplicidade, custo e manutenção.
- **Kaggle**: competidores de alto nível usam CV local rigorosa (ex.: 5-fold) para não serem enganados pelo public leaderboard; casos de "shake-up" (queda no ranking privado) evidenciam overfitting ao teste público. No Zillow Prize, o vencedor usou validação cruzada estratificada por região geográfica.
- **Startup X (fiasco hipotético)**: uma rede neural complexa atingiu 98% de acurácia no treino (vs. ~85% de modelos simples) e foi para produção sem validação adequada; três meses depois a retenção não melhorou — o modelo havia memorizado padrões específicos do histórico (overfitting clássico).
- **MLOps e automação (Google, Microsoft, AWS, Facebook)**: Google Vizier / Vertex AI Vizier automatiza otimização de hiperparâmetros com métodos bayesianos; Facebook dispara treinamento + validação cruzada automaticamente a cada novo dado/feature, promovendo o modelo apenas se supera o atual por uma margem definida (gating, similar a canary release/blue-green do DevOps).
- **AutoML e democratização**: frameworks como AutoKeras, H2O Driverless AI, TPOT, Auto-sklearn e plataformas como DataRobot usam validação cruzada interna (inclusive nested CV) para comparar algoritmos e hiperparâmetros automaticamente, permitindo que equipes sem expertise profunda em ML produzam modelos competitivos.
- **Monitoramento contínuo e revalidação**: após o deploy, práticas de MLOps captam feedback real e revalidam o modelo periodicamente, monitorando **drift** na distribuição dos dados. Técnicas como **shadow deployment** rodam o modelo novo em paralelo com tráfego real (sem impactar o usuário) para validar antes de promovê-lo.

## Checklist de Estudo
- [ ] Sei explicar por que uma única divisão treino/teste é arriscada e como a validação cruzada mitiga isso
- [ ] Sei descrever a decomposição bias-variance e relacioná-la a underfitting/overfitting
- [ ] Sei diferenciar hold-out, K-fold, Leave-One-Out, validação estratificada, validação em séries temporais e nested CV
- [ ] Sei explicar grid search, random search e busca bayesiana para tuning de hiperparâmetros
- [ ] Sei citar pelo menos um teste estatístico para comparar modelos e explicar a regra do 1 desvio-padrão (1-SE)
- [ ] Sei explicar o que é data leakage e como evitá-lo (pré-processamento dentro do ciclo de validação)
- [ ] Consigo implementar K-fold cross-validation do zero em Python para escolher um hiperparâmetro
- [ ] Sei relacionar validação cruzada com práticas de MLOps (governança, reprodutibilidade, shadow deployment)

## Palavras-chave
Overfitting · Validação cruzada · Hiperparâmetros

## Referências
- ARLOT, S.; CELISSE, A. *A survey of cross-validation procedures for model selection*. Statistics Surveys, 4: 40–79. 2010.
- BERGSTRA, J.; BENGIO, Y. *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research, 13(10): 281–305. 2012.
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. 2006.
- BROWNLEE, J. *A Gentle Introduction to k-fold Cross-Validation*. 2023.
- DIETTERICH, T. G. *Approximate statistical tests for comparing supervised classification learning algorithms*. 1998.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn & TensorFlow*. 2. ed. O'Reilly, 2019.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009.
- KOHAVI, R. *A study of cross-validation and bootstrap for accuracy estimation and model selection*. In: Proc. of IJCAI. 1995.
- RASCHKA, S. *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. 2018.
- YAU, N. *Why $1M Netflix algorithm never went to production*. 2012.
