# Aula 06 — Métricas de Avaliação

## Visão Geral
Aula sobre como avaliar corretamente o desempenho de modelos de Machine Learning, mostrando por que confiar em uma única métrica (como acurácia) é arriscado — analogia com um "deploy big bang" de software. Métricas atuam como "canários" que alertam sobre falhas do modelo antes que causem estrago em produção. A aula combina fundamentos práticos (cálculo manual de métricas de classificação e regressão) com uma revisão teórica mais ampla sobre redes neurais (formulação do problema de aprendizado, separabilidade linear, backpropagation, custo computacional, arquiteturas e MLOps/reprodutibilidade), traçando paralelos constantes com práticas de deploy de software (Blue/Green, canary release, rollback, RBAC).

## Tópicos Abordados
- Por que uma métrica única (ex.: acurácia) pode mascarar falhas graves do modelo
- Hands On: cálculo manual em Python de métricas de classificação (matriz de confusão, acurácia, precisão, revocação, F1-score) e de regressão (MAE, MSE, RMSE, R²)
- Saiba Mais: formulação matemática do aprendizado em redes neurais, separabilidade linear (perceptron vs. MLP), algoritmo de backpropagation, custo computacional de redes profundas, modelagem matemática (função de custo e otimização), arquiteturas/versões de redes neurais, controle de experimentos e reprodutibilidade (MLOps)
- Mercado, Cases e Tendências: Netflix Prize, uso de AUC-ROC/curvas PR no Kaggle, caso hipotético de um detector de intrusão com acurácia enganosa, práticas do Google e Facebook, MLOps em tempo real (Uber, Airbnb), futuro com AutoML e métricas de fairness/explicabilidade

## Conceitos-Chave

### Métricas de Classificação
A partir da matriz de confusão (VP, FP, VN, FN):
- **Acurácia**: `(VP + VN) / (VP + VN + FP + FN)` — proporção geral de acertos.
- **Precisão (Precision)**: `VP / (VP + FP)` — fração de predições positivas que realmente eram positivas.
- **Revocação (Recall/Sensibilidade)**: `VP / (VP + FN)` — fração de positivos reais que o modelo capturou.
- **F1-score**: média harmônica entre precisão e revocação, `2 · (P·R)/(P+R)`.

O exemplo clássico do classificador de spam/fraude com alta acurácia mas baixa revocação ilustra por que essas métricas devem ser analisadas em conjunto.

### Métricas de Regressão
- **MAE** (Erro Médio Absoluto): erro médio em unidades do problema.
- **MSE** (Erro Quadrático Médio) e **RMSE** (sua raiz): penalizam mais os erros grandes.
- **R²** (Coeficiente de Determinação): proporção da variância explicada pelo modelo (1.0 = previsão perfeita); um R² alto não garante ausência de erros significativos, por isso deve ser lido junto com MAE/RMSE.

### Redes Neurais — Separabilidade Linear e Backpropagation
- Um perceptron simples só separa dados **linearmente separáveis** (traça um hiperplano); problemas como o XOR não são linearmente separáveis (Minsky e Papert, 1969 — gerou a "primeira AI winter").
- Redes multicamadas (MLPs) com ativação não-linear (ex.: sigmoide) são **aproximadores universais**: com neurônios suficientes na camada oculta, aproximam qualquer função contínua.
- **Backpropagation**: ciclo de inicialização de pesos → forward pass → cálculo do erro (função de custo) → retropropagação do gradiente via regra da cadeia → ajuste dos pesos (`Δw = -η·∂E/∂w`) → repetição por várias épocas (descida do gradiente).
- **Early stopping**: interrompe o treino quando o erro de validação piora, evitando overfitting — comparado a um "rollback" de deploy.

### Custo Computacional e Otimização
- Treinar redes profundas escala aproximadamente como `O(N · P)` (N = exemplos, P = parâmetros); há retornos decrescentes na paralelização em clusters grandes.
- Modelos superdimensionados sem dados suficientes geram overfitting; modelos pequenos demais geram underfitting.
- **Learning rate warm-up** e decaimento gradual evitam instabilidade na convergência (analogia com "aquecer" um ambiente novo com tráfego sombra).
- A função de custo `E(w)` (ex.: erro quadrático médio) define uma superfície de otimização com múltiplos mínimos locais; regularização penaliza pesos grandes para melhorar a generalização.

### Arquiteturas e Governança de Modelos (MLOps)
- Migração entre arquiteturas (ex.: SVMs pré-2012 → redes profundas como AlexNet) é comparada a uma migração de versão de software, exigindo testes paralelos e validação em sombra antes do cutover.
- **Canary release de modelos**: lançar gradualmente um novo modelo para uma fração dos usuários antes de expandir para 100%.
- Ferramentas de MLOps (MLflow, Kubeflow, TFX) permitem versionar modelos, registrar experimentos e automatizar a promoção controlada (staging → produção), com aprovação humana em sistemas críticos (análogo a RBAC).
- **Reprodutibilidade**: fixar random seeds, versionar bibliotecas e datasets para garantir que o mesmo pipeline gere o mesmo modelo.

### Curvas ROC vs. Precisão-Revocação
- A curva **ROC** (True Positive Rate vs. False Positive Rate) dá uma visão geral do desempenho do classificador variando o limiar de decisão.
- A curva **Precisão-Revocação (PR)** é mais informativa em datasets desbalanceados, evidenciando o trade-off: aumentar revocação tende a reduzir precisão e vice-versa.
- A escolha do ponto de operação depende do negócio: fintechs tendem a priorizar revocação (não perder fraudes); sistemas de recomendação tendem a priorizar precisão (mostrar só itens relevantes).

## Exercício Hands-On (do material)
Cálculo de métricas em Python para dois cenários:
1. **Classificação binária** (detecção de fraude): a partir de `y_true` e `y_pred`, calcula-se manualmente VP, FP, VN, FN e, em seguida, acurácia (~0,70), precisão (~0,67), revocação (~0,50) e F1 (~0,57) — evidenciando que a acurácia sozinha esconderia o problema de metade das fraudes não detectadas.
2. **Regressão** (previsão de preço de imóvel): cálculo manual de MAE, MSE, RMSE e R² a partir de `y_true_reg` e `y_pred_reg`, obtendo MAE = RMSE = 16.0 e R² ≈ 0,90.

## Exemplos de Código

Cálculo da matriz de confusão a partir de listas de valores verdadeiros e previstos.

```python
# Dados de exemplo: 1 = fraude, 0 = nao fraude
y_true = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

# Inicializar contadores da matriz de confusão
VP = FP = VN = FN = 0
for verdadeiro, previsto in zip(y_true, y_pred):
    if previsto == 1 and verdadeiro == 1:
        VP += 1  # Verdadeiro Positivo: previu 1, era 1
    elif previsto == 1 and verdadeiro == 0:
        FP += 1  # Falso Positivo: previu 1, era 0
    elif previsto == 0 and verdadeiro == 0:
        VN += 1  # Verdadeiro Negativo: previu 0, era 0
    elif previsto == 0 and verdadeiro == 1:
        FN += 1  # Falso Negativo: previu 0, era 1

print(f"VP={VP}, FP={FP}, VN={VN}, FN={FN}")
# Saída esperada (por exemplo): VP=2, FP=1, VN=5, FN=2
```

Cálculo de acurácia, precisão, revocação e F1-score a partir da matriz de confusão.

```python
# (Continuando do código anterior: já temos VP, FP, VN, FN definidos)
accuracy = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP) if (VP + FP) > 0 else 0.0
recall = VP / (VP + FN) if (VP + FN) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Acurácia = {accuracy:.2f}")
print(f"Precisão = {precision:.2f}")
print(f"Revocação = {recall:.2f}")
print(f"F1-score = {f1:.2f}")
```

Cálculo de métricas de regressão (MAE, MSE, RMSE, R²) para um problema de previsão de preços.

```python
import math

# Exemplo de valores reais (em milhares de dólares) vs preditos pelo modelo
y_true_reg = [150, 200, 210, 180, 240]  # precos reais
y_pred_reg = [180, 195, 205, 170, 250]  # previsoes do modelo

n = len(y_true_reg)
# Erro Medio Absoluto (MAE)
mae = sum(abs(v - p) for v, p in zip(y_true_reg, y_pred_reg)) / n
# Erro Quadratico Medio (MSE)
mse = sum((v - p)**2 for v, p in zip(y_true_reg, y_pred_reg)) / n
# Raiz do Erro Quadratico Medio (RMSE)
rmse = math.sqrt(mse)
# Coeficiente de Determinacao (R2)
var_real = sum((v - sum(y_true_reg)/n)**2 for v in y_true_reg)
r2 = 1 - (sum((v - p)**2 for v, p in zip(y_true_reg, y_pred_reg)) / var_real)

print(f"MAE = {mae:.2f}")
print(f"MSE = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R² = {r2:.3f}")
```

## Cases e Tendências de Mercado
- **Netflix Prize (2006–2009)**: prêmio de US$ 1 milhão para melhorar em 10% o RMSE de previsão de ratings; o time vencedor alcançou ~10,05% de melhoria, mas a Netflix nunca implantou o modelo na íntegra — o ganho em RMSE não se traduziu em ganho relevante de engajamento, ilustrando que "melhorar uma métrica não garante sucesso do produto".
- **Kaggle**: popularizou métricas como AUC-ROC e evidenciou que otimizar AUC pode não corresponder a bom desempenho em precisão/recall no ponto operacional de interesse; em ranking, métricas como NDCG e MAP importam mais que taxa bruta de acerto.
- **Caso hipotético — detector de intrusão**: uma startup divulga 99,9% de acurácia, mas o modelo ignorava quase todos os ataques (classe minoritária) — mascarado pelo desbalanceamento dos dados; setores de detecção passaram a exigir SLAs como "sensibilidade ≥ 0,95 e especificidade ≥ 0,99".
- **Google**: avalia buscadores tanto em métricas offline (precisão de resultados) quanto online (satisfação via testes A/B).
- **Facebook**: treina modelos de detecção de conteúdo abusivo otimizando AUC-ROC, mas calibra o threshold operacional usando a curva PR.
- **Uber e Airbnb**: monitoramento em tempo real de métricas em produção, comparando com referências históricas e disparando retraining automático quando há degradação — análogo a rollback automatizado.
- **Futuro**: AutoML com otimização multiobjetivo (ex.: maximizar AUC-ROC e minimizar tempo de inferência), bandits/aprendizado por reforço para ajuste dinâmico de limiares pós-deploy, incorporação de metamétricas de confiança e calibração, e métricas de fairness/explicabilidade (ex.: diferença de revocação entre grupos demográficos em modelos de crédito) como parte do "painel multifacetado" de avaliação de modelos.

## Checklist de Estudo
- [ ] Sei calcular manualmente VP, FP, VN, FN e derivar acurácia, precisão, revocação e F1-score
- [ ] Sei calcular MAE, MSE, RMSE e R² e explicar a diferença entre eles
- [ ] Sei explicar por que um único número de acurácia pode mascarar falhas em dados desbalanceados
- [ ] Sei diferenciar a curva ROC da curva Precisão-Revocação e quando usar cada uma
- [ ] Sei explicar por que um perceptron simples não resolve o XOR e como o MLP supera essa limitação
- [ ] Sei descrever o ciclo de backpropagation (forward pass, cálculo do erro, retropropagação, ajuste de pesos)
- [ ] Sei relacionar boas práticas de MLOps (versionamento, reprodutibilidade, canary release de modelos) com o monitoramento contínuo de métricas
- [ ] Consigo explicar o caso do Netflix Prize e a lição sobre alinhar métricas técnicas a métricas de negócio

## Palavras-chave
Acurácia · Precisão · RMSE

## Referências
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. 2006.
- BROWNLEE, J. *Classification Accuracy is Not Enough: More Performance Measures You Can Use*. 2019. Disponível em: https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/. Acesso em: 18 jan. 2026.
- CHAI, T.; DRAXLER, R. *Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE*. Geoscientific Model Development, v. 7, n. 3, p. 1247–1250, 2014. Disponível em: https://gmd.copernicus.org/articles/7/1247/2014/gmd-7-1247-2014.html. Acesso em: 18 jan. 2026.
- CZAKON, J. *F1 Score vs ROC AUC vs Accuracy vs PR AUC: Which Evaluation Metric Should You Choose?* 2025. Disponível em: https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc. Acesso em: 18 jan. 2026.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2. ed. Sebastopol, CA: O'Reilly Media, 2019.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge, MA: MIT Press, 2016.
- POWERS, D. M. *Evaluation: From Precision, Recall to ROC, Informedness, Markedness & Correlation*. Journal of Machine Learning Technologies, v. 2, n. 1, p. 37–63, 2011. Disponível em: https://arxiv.org/abs/2010.16061. Acesso em: 18 jan. 2026.
- SAITO, T.; REHMSMEIER, M. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE, v. 10, n. 3, e0118432, 2015. Disponível em: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432. Acesso em: 18 jan. 2026.
