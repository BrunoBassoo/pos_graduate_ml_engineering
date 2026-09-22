# Aula 01 — Paradigmas de Aprendizado de Máquina

## Visão Geral
Aula introdutória da disciplina de Machine Learning (Fase 2). Apresenta os **três paradigmas fundamentais** do aprendizado de máquina — **supervisionado**, **não supervisionado** e **por reforço** — a partir de três cenários de negócio distintos (previsão de churn, segmentação de clientes e treino de um agente para jogar um videogame). O critério central de diferenciação entre os paradigmas é o **tipo de sinal/feedback disponível** para treinar o modelo: rótulos conhecidos, ausência de rótulos, ou recompensas obtidas por interação com um ambiente.

## Tópicos Abordados
- Os três desafios de negócio que ilustram cada paradigma (churn, segmentação, agente de jogo)
- Hands On: exemplo prático em Python de aprendizado supervisionado (árvore de decisão no dataset Iris)
- Hands On: exemplo prático de aprendizado não supervisionado (K-Means em dados sintéticos)
- Hands On: exemplo prático de aprendizado por reforço (multi-armed bandit com estratégia epsilon-greedy)
- Saiba Mais: fundamentos teóricos de cada paradigma (função objetivo, generalização, formalização matemática)
- Aprendizado semi-supervisionado e auto-supervisionado
- Aprendizado por Reforço Profundo (Deep RL) e Processos de Decisão de Markov (MDP)
- Mercado, cases e tendências de cada paradigma

## Conceitos-Chave

### Aprendizado Supervisionado
O algoritmo recebe dados de entrada **e** as respostas esperadas (rótulos), aprendendo a mapear entradas em saídas. O objetivo formal é minimizar uma função de perda (erro quadrático médio para regressão, entropia cruzada para classificação) sobre o conjunto de treino — estratégia conhecida como **minimização do risco empírico**. Métodos comuns: regressão linear/logística, árvores de decisão, Random Forest, Support Vector Machines, redes neurais.

**Generalização** é o aspecto crucial: um bom modelo precisa acertar em dados novos, não apenas nos dados de treino. Técnicas de treino/validação/teste, validação cruzada e regularização ajudam a evitar:
- **Overfitting** — modelo especializado demais nos dados de treino, perde desempenho em dados novos.
- **Underfitting** — modelo simples demais, não captura os padrões dos dados.

### Aprendizado Não Supervisionado
Sem rótulos, o algoritmo busca estruturas ocultas nos dados. Formalmente, pode ser visto como estimação da distribuição **P(x)** (em vez de P(y|x) do caso supervisionado). Técnicas principais: agrupamento/clustering (K-Means, DBSCAN, clustering hierárquico), redução de dimensionalidade (PCA), detecção de anomalias.

Como não há rótulos "verdadeiros", a avaliação é mais subjetiva e depende de métricas indiretas, como **silhouette score** (coesão/separação de clusters) e **variância explicada** (redução de dimensionalidade).

Variações importantes:
- **Aprendizado semi-supervisionado** — parte dos dados rotulada, parte não.
- **Aprendizado auto-supervisionado** — o próprio dado gera rótulos artificiais; é a estratégia usada no pré-treino de Large Language Models (LLMs) a partir de grandes volumes de texto não rotulado.

### Aprendizado por Reforço (RL)
Um agente interage com um ambiente em uma sequência de estados e ações, buscando **maximizar recompensas** ao longo do tempo — aprendizado por tentativa e erro. Formalizado como **Processo de Decisão Markoviano (MDP)**: conjunto de estados S, ações A, função de recompensa R(s,a) e (opcionalmente) modelo de transição T(s,a→s'). O objetivo é encontrar a **política ótima π(s)**.

Conceito central: **trade-off exploração vs. exploração** — explorar novas ações para descobrir seus retornos vs. explorar (usar) o conhecimento já obtido para maximizar ganho.

Abordagens de solução:
- **Model-free**: Q-Learning, SARSA — estimam valores Q de estado/ação.
- **Gradiente de política**: aprendem a política diretamente, sem estimar valores de estado.
- **Deep Reinforcement Learning (Deep RL)**: combina redes neurais com algoritmos de RL para lidar com espaços de estados/ações muito grandes (ex.: AlphaGo/DQN).

Extensões citadas: aprendizado por reforço multi-agente, representation learning (uso de modelos não supervisionados para extrair características do ambiente) e imitation learning (iniciar o agente imitando dados de especialistas).

## Exercício Hands-On (do material)
Três exemplos práticos e funcionais em Python:

1. **Supervisionado** — classificação das flores do dataset Iris com uma `DecisionTreeClassifier` (scikit-learn), usando `train_test_split` e `accuracy_score`. Resultado típico: acurácia ≈ 0.97.
2. **Não supervisionado** — geração de 100 pontos sintéticos em torno de dois centros (0,0) e (3,3) e aplicação de `KMeans(n_clusters=2)` para recuperar os agrupamentos sem rótulos. O algoritmo estimou centroides muito próximos dos centros reais.
3. **Por reforço** — simulação de um problema de *multi-armed bandit* com 3 "máquinas" com probabilidades reais de recompensa de 20%, 50% e 70%. Um agente com estratégia **epsilon-greedy** (ε = 0.1) roda 2.000 iterações, atualizando estimativas de valor Q por média incremental das recompensas, e ao final identifica corretamente a máquina mais lucrativa (valores Q estimados ≈ [0.18, 0.46, 0.71]).

## Exemplos de Código

Exemplo de aprendizado supervisionado com árvore de decisão no dataset Iris:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {acuracia:.2f}")
```

Exemplo de aprendizado não supervisionado com K-Means:

```python
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2))
])

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
rotulos = kmeans.fit_predict(X)

print("Quantidade de clusters encontrados:", len(np.unique(rotulos)))
print("Centroides:", kmeans.cluster_centers_)
```

Exemplo de aprendizado por reforço — multi-armed bandit com epsilon-greedy:

```python
import random
import numpy as np

probs = [0.2, 0.5, 0.7]   # probabilidades de recompensa de cada máquina
Q = [0.0, 0.0, 0.0]       # estimativas iniciais de valor (qualidade) de cada ação
N = [0, 0, 0]             # contagem de tentativas de cada ação

epsilon = 0.1
for t in range(2000):
    if random.random() < epsilon:
        acao = random.randrange(len(Q))     # explora (escolha aleatória)
    else:
        acao = int(np.argmax(Q))            # explora o melhor conhecido

    recompensa = 1 if random.random() < probs[acao] else 0
    N[acao] += 1
    Q[acao] += (recompensa - Q[acao]) / N[acao]

print("Valores Q estimados:", [round(q, 2) for q in Q])
print("Melhor ação encontrada:", int(np.argmax(Q)))
```

## Cases e Tendências de Mercado
- **Supervisionado**: diagnóstico médico por imagem, detecção de fraude em transações financeiras, previsão de churn e de cliques em anúncios (CTR).
- **Não supervisionado**: segmentação de clientes no varejo/e-commerce, detecção de anomalias em segurança da informação, compressão de dados via PCA/autoencoders em imagens e sinais.
- **Auto-supervisionado como pré-treino de LLMs**: modelos como o GPT-3 e sucessores são pré-treinados de forma não supervisionada em grandes volumes de texto, e depois refinados com aprendizado supervisionado e/ou reforço com feedback humano (**RLHF**, usado no ChatGPT).
- **Aprendizado por Reforço**: marcos históricos como o **AlphaGo** (DeepMind, 2016, venceu o campeão mundial de Go) e sua evolução **AlphaZero** (aprende xadrez, shogi e Go só a partir das regras); agentes que superaram profissionais em StarCraft II e Dota 2.
- Aplicações de RL fora de jogos: controle de processos industriais (resfriamento de data centers, plantas químicas), logística/supply chain, sistemas de recomendação (maximização de valor de vida do cliente), finanças (alocação sequencial, market making) e mobilidade/autonomia.
- Uso prático de **bandits/contextual bandits** para decisões com feedback imediato e de **RL offline/sim-to-real** quando não é seguro explorar diretamente em produção.
- Desafios citados: definição de função de recompensa alinhada ao negócio, exploração segura, observabilidade parcial, restrições operacionais (SLA, segurança, compliance) e generalização.

## Checklist de Estudo
- [ ] Sei diferenciar os três paradigmas pelo tipo de sinal/feedback disponível
- [ ] Sei explicar overfitting e underfitting e as técnicas para mitigá-los (validação cruzada, regularização)
- [ ] Sei implementar um exemplo simples de classificação supervisionada com scikit-learn
- [ ] Sei explicar como o K-Means agrupa dados sem rótulos e como avaliar clusters (silhouette score)
- [ ] Sei diferenciar aprendizado semi-supervisionado de auto-supervisionado e seu papel no pré-treino de LLMs
- [ ] Sei formalizar um problema de RL como MDP (estados, ações, recompensa, política)
- [ ] Sei explicar o dilema exploração vs. exploração e implementar uma estratégia epsilon-greedy
- [ ] Conheço cases de mercado que combinam paradigmas (ex.: pré-treino não supervisionado + RLHF)

## Palavras-chave
Aprendizado Supervisionado · Aprendizado Não Supervisionado · Aprendizado por Reforço

## Referências
- ARXIV. *Deep reinforcement learning from human preferences*. 2017. https://arxiv.org/abs/1706.03741
- ARXIV. *Training language models to follow instructions with human feedback*. 2022. https://arxiv.org/abs/2203.02155
- JMLR. *What Regularized Auto-Encoders Learn from the Data-Generating Distribution*. 2014. https://jmlr.org/papers/volume15/alain14a/alain14a.pdf
- JOLLIFFE, I. T. *Principal Component Analysis*. 2. ed. New York: Springer, 2002.
- NATURE. *Human-level control through deep reinforcement learning*. 2015. https://www.nature.com/articles/nature14236.pdf
- NATURE. *Mastering the game of Go with deep neural networks and tree search*. 2016. https://www.nature.com/articles/nature16961.pdf
- NATURE. *Mastering the game of Go without human knowledge*. 2017. https://www.nature.com/articles/nature24270.pdf
- OPENAI. *Aligning language models to follow instructions*. 2022. https://openai.com/index/instruction-following/
- SUTTON, R. S.; BARTO, A. G. *Reinforcement Learning: An Introduction*. 2. ed. Cambridge: MIT Press, 2018.
- TAYLOR & FRANCIS. *Zero-inflated models — Knowledge and References*. 2026.
- WIKIPEDIA. *Principal component analysis*. 2026.
- WIKIPEDIA. *Zero-inflated model*. 2026.
