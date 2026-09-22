# Aula 08 — Projeto de Comparação de Modelos

## Visão Geral
Aula final do módulo, dedicada a como conduzir de forma rigorosa a comparação entre modelos de Machine Learning candidatos. Argumenta que escolher o modelo com melhor métrica isolada em um único experimento é insuficiente: uma comparação séria precisa avaliar desempenho em dados nunca vistos, custo computacional, reprodutibilidade e significância estatística das diferenças observadas. O material usa analogias recorrentes com autoscaling de sistemas (HPA, KEDA, histerese, sistemas de controle) para explicar por que comparações de modelos precisam de "amortecimento" e critérios de decisão bem definidos.

## Tópicos Abordados
- Por que escolher modelos apenas pela acurácia de treino é uma armadilha
- Hands On: comparação reprodutível entre regressão logística e uma pequena rede neural (MLP) em PyTorch, medindo acurácia, número de parâmetros e tempo de treino
- Saiba Mais:
  - Equilíbrio entre complexidade do modelo e volume de dados (razão p/N)
  - Decomposição do erro em bias, variância e ruído irredutível
  - Comparação estatisticamente significativa (intervalo de confiança, teste t pareado, teste de Wilcoxon, teste de McNemar)
  - Reprodutibilidade e controle de variáveis
  - Controle de custo computacional e eficiência (multi-fidelidade, Hyperband)
  - Oscilações e histerese na escolha de modelos (regra do 1-SE)
  - Buffering e atualizações periódicas vs. contínuas
  - Considerações multiobjetivo (SLOs) além da acurácia
  - Arquiteturas e padrões operacionais (Champion-Challenger, shadow deployment)
  - Tendências: AutoML, meta-learning e personalização de métricas
- Mercado, Cases e Tendências: Netflix Prize, cultura Kaggle, caso fictício de triagem de currículos, Facebook/FBLearner Flow, Google Vizier, Gmail, especialização de modelos, governança e auditabilidade

## Conceitos-Chave

### Equilíbrio entre complexidade e volume de dados
Define-se a razão de utilização de dados ρ = p/N, em que p é o número de parâmetros (ou outra medida de complexidade, como dimensão VC) e N é o número de exemplos de treino. Se ρ ≫ 1 (parâmetros em excesso para poucos dados), o modelo tende a sobreajustar (overfitting); a condição ideal de estabilidade é ρ < 1, idealmente ρ ≪ 1. Modelos simples sofrem mais de viés (bias) alto; modelos complexos, de variância alta.

### Decomposição do erro de generalização
```
Erro_total = Bias² + Variância + Ruído irredutível (σ²)
```
Modelos simples têm bias intrínseco que não diminui com mais dados; modelos complexos podem ter bias baixo, mas pagam com variância maior. Aumentar N reduz a variância (beneficia modelos complexos) mas não altera o bias. Em regimes de muitos dados, modelos complexos tendem a se sobressair; com poucos dados, tendem a sobreajustar.

### Comparação estatisticamente significativa
- **Intervalo de confiança da acurácia**: p̂ ± 1.96·√(p̂(1-p̂)/n) para amostras grandes (95% de confiança). Se os intervalos de dois modelos se sobrepõem, não há evidência de superioridade.
- **Teste t pareado** sobre diferenças de desempenho fold a fold (ex.: 5-fold CV), testando H₀: diferença média = 0.
- **Teste de Wilcoxon** (não paramétrico) como alternativa ao teste t pareado.
- **Teste de McNemar**, usado em classificação binária, foca nas instâncias em que dois classificadores discordam.
- Em contextos críticos (ex.: medicina), costuma-se exigir folga estatística de 95% ou 99% antes de promover um novo modelo.

### Reprodutibilidade e controle de variáveis
Comparar modelos exige controlar todas as variáveis exceto o modelo em si: mesmo split de treino/teste (ou mesmas dobras de CV), mesmo pré-processamento, condições computacionais equivalentes, versões de biblioteca documentadas e seeds aleatórias fixadas. Ferramentas como MLflow e Weights & Biases automatizam esse registro. A aula cita um caso relatado (2018) em que um modelo de detecção de fraudes parecia superior, mas a diferença era causada por seed e embaralhamento de dados não controlados — reforçando a prática de rodar cada experimento várias vezes com inicializações diferentes.

### Controle de custo computacional e eficiência
Estratégia de avaliação progressiva ("scale-to-zero" aplicado a experimentos): treinar rapidamente todos os candidatos com poucos recursos (poucas épocas ou subconjunto pequeno de dados) para eliminar os claramente inferiores, e só então investir treino completo nos mais promissores. Algoritmos multi-fidelidade como **Hyperband** implementam essa ideia, avaliando parcialmente muitos modelos e interrompendo cedo os de baixo desempenho.

### Oscilações e histerese na escolha de modelos
Trocar de modelo em produção a cada pequena oscilação de métrica gera instabilidade (análogo a um HPA sem histerese). A **regra do 1 desvio-padrão (1-SE)**: se um modelo mais simples fica dentro de 1 desvio-padrão do resultado do modelo mais complexo, prefira o mais simples — só promova o complexo se ele superar com folga além da variação esperada. Também se recomenda período mínimo de teste A/B antes de efetivar uma troca definitiva.

### Buffering e atualizações periódicas vs. contínuas
Tolerar um "backlog de desempenho" antes de agir evita reagir a flutuações de curto prazo. Atualizações programadas (ex.: retreinar semanalmente) amortecem ruído, ao custo de alguma defasagem do modelo — a escolha depende dos SLOs da aplicação. Gatilhos com histerese para retreinamento (ex.: "retreine apenas se o erro ultrapassar 5% por 3 dias consecutivos") evitam o efeito de "retrain on/off".

### Considerações multiobjetivo além da acurácia
Formulação de uma função objetivo combinando desempenho e custo:
```
J(modelo) = Acurácia − λ × Custo
```
onde λ converte custo (tempo de inferência, $/hora de GPU) na mesma escala de valor da acurácia. Projetos devem definir SLOs claros (latência, memória, interpretabilidade) e podem montar uma tabela comparativa multicritério entre os modelos candidatos.

### Padrões operacionais: Champion-Challenger e shadow deployment
O modelo em produção é o **champion**; novos modelos treinados são **challengers**. Em vez de substituir o champion assim que um challenger parece melhor offline, usa-se **shadow deployment**: o challenger roda em paralelo, recebendo as mesmas requisições sem afetar usuários, e só assume produção se superar consistentemente o champion nas métricas de negócio. Empresas como Google, Amazon e Meta usam esse padrão combinado a canary deployment. Recomenda-se definir previamente: (1) quais modelos e hiperparâmetros comparar, (2) metodologia de avaliação (ex.: hold-out final + CV), (3) critérios objetivos de seleção, e (4) plano de implantação/monitoramento com rollback preparado.

## Exercício Hands-On (do material)
Cenário: comparar dois classificadores no dataset sintético `make_moons` (classes não linearmente separáveis) — **Modelo A** (regressão logística, `nn.Linear(2,1)` + Sigmoid) e **Modelo B** (MLP com uma camada oculta de 10 neurônios). Ambos treinados com a mesma seed (42), mesmo split treino/teste (67/33) e mesmos hiperparâmetros (SGD, lr=0.1, 1000 épocas), medindo acurácia no teste, número de parâmetros e tempo de treino via `torch.cuda.Event`.

Resultado obtido no material: Modelo A — 85,00% de acurácia, 3 parâmetros, ~23,5ms de treino; Modelo B — 96,00% de acurácia, 31 parâmetros, ~27,4ms de treino. Conclusão do material: o ganho de ~11 pontos percentuais do Modelo B compensa o custo computacional adicional (irrelevante nesse exemplo em pequena escala), tornando justificada a escolha do modelo mais complexo — decisão que em cenários reais também consideraria latência de inferência, uso de memória, interpretabilidade e consumo energético.

## Exemplos de Código

Experimento reprodutível comparando os dois modelos (Modelo A: regressão logística; Modelo B: MLP), conforme apresentado no Hands On.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Reprodutibilidade: fixa seed para numpy e PyTorch
np.random.seed(42)
torch.manual_seed(42)

# 2. Geração de dados sintéticos (não linearmente separáveis) e divisão treino/teste
# Aqui usamos um dataset "moons" para ter classes em meia-lua
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
X, y = make_moons(n_samples=300, noise=0.10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Converte para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 3. Definição dos dois modelos a comparar:
modelA = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
modelB = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Hiperparâmetros de treinamento
num_epochs = 1000
lr = 0.1

# 4. Função auxiliar para treinar e avaliar um modelo
def treinar_e_avaliar(model, nome):
    criterion = nn.BCELoss()  # perda para classificação binária
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Treinamento
    inicio = torch.cuda.Event(enable_timing=True); fim = torch.cuda.Event(enable_timing=True)
    inicio.record()  # marca tempo inicial (usando eventos CUDA para alta precisão)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    fim.record(); torch.cuda.synchronize()  # marca tempo final e sincroniza
    tempo_ms = inicio.elapsed_time(fim)  # tempo em milissegundos
    # Avaliação
    with torch.no_grad():  # desliga gradiente para teste
        preds = model(X_test)
        preds_classes = (preds >= 0.5).float()
        acertos = (preds_classes == y_test).sum().item()
        acc = acertos / y_test.shape[0]
    # Coleta métricas do modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{nome}: Acurácia = {acc*100:.2f}% | Parâmetros = {total_params} | Tempo treino = {tempo_ms:.1f}ms")

# 5. Executa experimento para cada modelo
treinar_e_avaliar(modelA, "Modelo A (Logístico)")
treinar_e_avaliar(modelB, "Modelo B (MLP)")
```

Saída obtida ao rodar o experimento:

```
Modelo A (Logístico): Acurácia = 85.00% | Parâmetros = 3 | Tempo treino = 23.5ms
Modelo B (MLP): Acurácia = 96.00% | Parâmetros = 31 | Tempo treino = 27.4ms
```

## Cases e Tendências de Mercado
- **Netflix Prize (2009)**: ensina que ganhos algorítmicos marginais podem não se justificar na prática — ainda que a competição tenha gerado ideias (ensembles, fatoração de matrizes) que a Netflix aproveitou indiretamente. Lição central: alinhar comparações de modelos a métricas de negócio antes de investir em ganhos incrementais.
- **Cultura Kaggle**: popularizou validação cruzada extensiva, ensembles, model stacking e o cuidado estatístico para não se enganar com variação aleatória — mas o modelo campeão de competição costuma ser um "Frankenstein" difícil de reproduzir e colocar em produção, levando empresas a destilar ensembles complexos em modelos mais simples antes do deploy.
- **Caso fictício — triagem de currículos**: uma startup adotou uma rede neural com 2% mais acurácia que uma árvore de decisão sem considerar latência (5s vs 0,1s por currículo) nem interpretabilidade (exigida por auditoria externa), tendo que reverter a decisão. Lição: comparação de modelos precisa incluir latência e interpretabilidade, não só acurácia.
- **Facebook/Meta — FBLearner Flow**: plataforma que roda milhares de experimentos de modelo em paralelo (ex.: previsão de cliques em anúncios), com job schedulers inteligentes para enfileirar e pausar experimentos de baixa prioridade conforme disponibilidade de GPU — um análogo de autoscaling/fila aplicado a experimentos de ML.
- **Google (Vizier)**, **Microsoft (Azure AutoML)** e **Uber (Michelangelo)**: investem em sistemas para automatizar e escalar a comparação de modelos, democratizando o processo para engenheiros com menos experiência.
- **Gmail**: exemplo de modelo de filtragem de spam com pipeline de aprendizado contínuo e auto-tuning, com intervenção humana apenas se a performance cair abaixo de um patamar crítico.
- **Especialização de modelos**: tendência de manter múltiplos modelos especializados por subgrupo (ex.: um modelo de inadimplência geral e outro para clientes autônomos), com uma lógica de alto nível decidindo qual usar por contexto — deslocando a pergunta de "qual modelo é melhor?" para "qual conjunto de modelos e em que condições cada um é melhor?".
- **Governança e auditabilidade**: setores financeiro e médico exigem documentação da escolha do modelo (ex.: por que o modelo B foi escolhido e não o A). Empresas adotam "registros de decisão" anexados a cada modelo em produção, resumindo os experimentos comparativos que o justificaram.

## Checklist de Estudo
- [ ] Sei explicar por que uma única métrica de treino não é suficiente para comparar modelos
- [ ] Sei calcular e interpretar um intervalo de confiança para a acurácia de um modelo
- [ ] Sei diferenciar teste t pareado, teste de Wilcoxon e teste de McNemar e quando usar cada um
- [ ] Sei explicar a razão ρ = p/N e sua relação com overfitting/underfitting e a decomposição bias-variância-ruído
- [ ] Sei descrever boas práticas de reprodutibilidade em comparação de modelos (seeds, versionamento, mesmas condições)
- [ ] Sei explicar a regra do 1-SE e por que ela evita oscilações na escolha de modelos
- [ ] Sei descrever o padrão Champion-Challenger e o conceito de shadow deployment
- [ ] Sei formular uma comparação multiobjetivo considerando custo, latência e outros SLOs além da acurácia

## Palavras-chave
Modelos. Reprodutibilidade. MLOps.

## Referências
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. New York: Springer, 2006.
- BROWNLEE, J. *Statistical Significance Tests for Comparing Machine Learning Algorithms*. 2020. Disponível em: https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/. Acesso em: 23 fev. 2026.
- DEMŠAR, J. Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, v. 7, n. 1, p. 1–30, 2006. Disponível em: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf. Acesso em: 23 fev. 2026.
- DIETTERICH, T. G. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, v. 10, n. 7, p. 1895–1923, 1998. Disponível em: https://doi.org/10.1162/089976698300017197. Acesso em: 23 fev. 2026.
- GÉRON, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2. ed. Sebastopol: O'Reilly Media, 2019.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. 2016. Disponível em: https://www.deeplearningbook.org/. Acesso em: 23 fev. 2026.
- HESTNESS, J.; NARANG, S.; et al. *Deep Learning Scaling is Predictable, Empirically*. 2017. Disponível em: https://arxiv.org/abs/1712.00409. Acesso em: 23 fev. 2026.
- IBM RESEARCH. *AI Fairness 360: An Open Source Toolkit*. 2018. Disponível em: https://ibm.biz/AI-Fairness-360. Acesso em: 23 fev. 2026.
- RASCHKA, S. *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. 2018. Disponível em: https://arxiv.org/abs/1811.12808. Acesso em: 23 fev. 2026.
- YAU, N. *Why the Netflix Prize Never Materialized*. 2012. Disponível em: https://flowingdata.com/2012/04/17/why-the-netflix-prize-never-materialized/. Acesso em: 23 fev. 2026.
