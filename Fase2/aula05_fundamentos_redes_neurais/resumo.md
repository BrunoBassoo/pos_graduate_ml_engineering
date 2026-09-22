# Aula 05 — Fundamentos de Redes Neurais

## Visão Geral
Aula introdutória sobre redes neurais artificiais dentro da disciplina de Machine Learning Engineering. Parte da limitação de algoritmos tradicionais (programados "na mão") para tarefas de reconhecimento de padrões complexos e apresenta as redes neurais como abordagem em que o modelo aprende a partir de exemplos, generalizando para casos novos. O fio condutor teórico da aula é uma extensa analogia entre o treinamento/deploy de redes neurais e práticas de engenharia de software (Blue/Green deployment, canary release, feature flags, GitOps, IaC), usada para explicar riscos, custos e boas práticas de colocar modelos de Deep Learning em produção.

## Tópicos Abordados
- Limitações de algoritmos tradicionais e motivação para redes neurais
- Hands On: resolução do problema XOR com Perceptron simples vs. MLP em PyTorch
- Saiba Mais (teoria): formulação matemática do aprendizado, separabilidade linear, backpropagation, complexidade computacional, modelagem matemática (função de custo), arquiteturas e versões de redes neurais, controle de experimentos e reprodutibilidade
- Mercado, Cases e Tendências: caso AlexNet/ImageNet (2012), canary releases de modelos (Twitter, Netflix), feature flags aplicados a IA (Uber Michelangelo, LaunchDarkly), casos de fracasso por adoção não-incremental de IA, AutoML, integração neuro-simbólica

## Conceitos-Chave

### Perceptron simples vs. MLP
Um único neurônio (perceptron simples) só consegue classificar dados **linearmente separáveis** — traça um único hiperplano. O problema XOR não é linearmente separável, limitação destacada por Minsky e Papert em 1969 (gerando a "primeira AI winter"). A solução é usar um **perceptron de múltiplas camadas (MLP)** com pelo menos uma camada oculta e função de ativação não-linear (ex.: sigmoide), capaz de combinar múltiplas fronteiras lineares em regiões de decisão não-lineares.

### Teorema do aproximador universal
Redes neurais com pelo menos uma camada oculta e ativação não-linear são aproximadoras universais: dado número suficiente de neurônios, conseguem aproximar qualquer função contínua com a precisão desejada. O desafio real está em encontrar os pesos adequados (treinar a rede).

### Backpropagation e ciclo de treinamento
Ciclo de treinamento supervisionado:
1. **Inicialização** dos pesos (aleatória) e definição da taxa de aprendizagem (η).
2. **Forward pass** — calcula a saída da rede camada a camada.
3. **Cálculo do erro** — compara saída prevista com a esperada via uma função de custo (ex.: erro quadrático médio, entropia cruzada).
4. **Retropropagação + ajuste** — calcula os gradientes do erro em relação a cada peso (via regra da cadeia) e atualiza os pesos na direção oposta ao gradiente (Δw = −η·∂E/∂w).
5. **Repetição** por várias épocas até o erro atingir um mínimo aceitável.

Esse processo (descida do gradiente) não garante o mínimo global — a superfície de erro tem múltiplos mínimos locais — mas, na prática, otimizadores estocásticos encontram soluções suficientemente boas.

### Early stopping e warm-up
**Early stopping** interrompe o treinamento quando o erro de validação começa a piorar, prevenindo overfitting. **Learning rate warm-up** (taxa de aprendizagem alta no início, reduzida gradualmente) evita mudanças bruscas que desestabilizariam a convergência.

### Complexidade computacional
O custo de treinar redes profundas escala aproximadamente de forma linear com o volume de dados e o número de parâmetros (O(N·P)). Redes muito grandes sem dados suficientes tendem a overfitting e desperdício de recursos; redes pequenas demais sofrem underfitting — encontrar esse equilíbrio é parte do trabalho de quem projeta redes neurais.

### Função de custo e otimização
O treinamento é formulado como minimização de uma função de perda E(w) — por exemplo, o erro quadrático médio sobre os exemplos de treino. Técnicas de **regularização** adicionam termos à função de custo penalizando pesos muito grandes, equilibrando ajuste aos dados de treino com capacidade de generalização (viés vs. variância).

### Arquiteturas de redes neurais
Além do MLP, existem redes convolucionais (CNNs, para visão), redes recorrentes (RNNs, para sequências) e redes Transformer (com mecanismos de atenção). A evolução de arquiteturas é comparada a versionamento de software: mudanças grandes exigem testes paralelos entre modelo antigo e novo antes do cutover completo, garantindo compatibilidade de entradas/saídas.

### Reprodutibilidade e MLOps
Ferramentas de MLOps (MLflow, Kubeflow, TensorFlow Extended/TFX) permitem versionar modelos, gerenciar experimentos e automatizar implantação de forma controlada e auditável. Boas práticas de reprodutibilidade incluem fixar random seeds, documentar versões de bibliotecas e armazenar o dataset de treino usado, de forma que retreinar na mesma configuração leve a resultados equivalentes. É comum haver aprovação manual antes de promover um modelo de staging para produção em sistemas críticos.

## Exercício Hands-On (do material)
Construção de duas redes neurais em Python/PyTorch para resolver o problema lógico do XOR:

1. **Perceptron de camada única** (`nn.Linear(2,1)` + `Sigmoid`), treinado por 1000 épocas com `BCELoss` e `SGD` — converge para saídas próximas de 0.5 em todas as entradas, ou seja, **falha** em aprender o XOR (não é linearmente separável).
2. **MLP com uma camada oculta** (`nn.Linear(2,2)` → `Sigmoid` → `nn.Linear(2,1)` → `Sigmoid`), treinado por 10000 épocas — converge para saídas próximas da tabela verdade correta ([0,1,1,0]), demonstrando que a camada oculta adiciona poder de representação suficiente para resolver o problema.

## Exemplos de Código

Definição do dataset XOR e do Perceptron de camada única, que não consegue aprender a função:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dados de treino para XOR (entradas X e saída y)
X = torch.tensor([[0.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 0.0],
                   [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Modelo 1: Perceptron de camada única (2 entradas -> 1 saída com função sigmoide)
model1 = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()  # Função de custo: erro quadrático médio binário
optimizer = optim.SGD(model1.parameters(), lr=1.0)

# Treinamento do modelo 1
for epoch in range(1000):
    optimizer.zero_grad()              # Zera gradientes acumulados
    outputs = model1(X)                # Forward: calcula saída para todas as entradas
    loss = criterion(outputs, y)       # Calcula o erro (loss) em relação ao esperado
    loss.backward()                    # Backpropagation: gradientes dos pesos
    optimizer.step()                   # Atualiza pesos (descida do gradiente)

# Saídas finais do modelo 1
print("Saídas do Perceptron:", model1(X).detach().numpy())
# (Esperado aproximadamente: [0.5, 0.5, 0.5, 0.5] -> não consegue aprender XOR)
```
*Fonte: Elaborado pelo autor (2026)*

MLP com uma camada oculta, que consegue aprender a função XOR:

```python
# Modelo 2: Rede Neural Multicamadas (2 entradas -> 2 neurônios ocultos -> 1 saída)
model2 = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),                # Função de ativação sigmoide na camada oculta
    nn.Linear(2, 1),
    nn.Sigmoid()                 # Sigmoide na saída para obter valor entre 0 e 1
)
criterion2 = nn.BCELoss()
optimizer2 = optim.SGD(model2.parameters(), lr=1.0)

# Treinamento do modelo 2 (mais extenso, rede mais complexa)
for epoch in range(10000):
    optimizer2.zero_grad()
    outputs = model2(X)
    loss = criterion2(outputs, y)
    loss.backward()
    optimizer2.step()

# Saídas finais do modelo 2
print("Saídas do MLP:", model2(X).detach().numpy())
# (Esperado: valores próximos a [[0.0], [1.0], [1.0], [0.0]] – modelo aprende XOR!)
```
*Fonte: Elaborado pelo autor (2026)*

## Cases e Tendências de Mercado
- **AlexNet / ImageNet (2012)** — equipe de Geoff Hinton (Universidade de Toronto) venceu a competição ImageNet com uma CNN de 8 camadas treinada em duas GPUs, saltando a acurácia de ~75% (métodos tradicionais) para ~85%. Marco que reacendeu o interesse e a credibilidade em redes neurais, iniciando a era moderna do Deep Learning.
- **Canary releases de modelos** — Twitter e Netflix passaram a lançar novos modelos de recomendação (rede neural) inicialmente para uma pequena porcentagem de usuários, expandindo gradualmente conforme as métricas se mostrassem boas.
- **Feature flags + IA** — ferramentas como Uber Michelangelo e LaunchDarkly permitem desacoplar o deploy do modelo do lançamento de funcionalidades, possibilitando desativar granularmente comportamentos específicos de um modelo sem reverter o modelo inteiro.
- **Caso de fracasso (instituição financeira monolítica, não identificada)** — tentou substituir simultaneamente as lógicas de vários sistemas interconectados por redes neurais, gerando um "pesadelo logístico"; a lição foi desacoplar mudanças e migrar cada componente no seu próprio ritmo, mantendo compatibilidade entre versões.
- **Caso de fracasso (startup não identificada)** — tentou colocar "IA em tudo" mantendo ambientes Blue/Green duplicados com modelos neurais, dobrando o custo de infraestrutura e acumulando muitas mudanças de uma vez por medo de liberar continuamente; corrigiu o rumo passando a treinar e implantar o modelo novo continuamente, promovendo-o apenas quando havia confiança suficiente.
- **Ferramentas de MLOps e AutoML** — TensorFlow Extended (TFX) e Kubeflow suportam estratégias avançadas de deploy (sombreamento de tráfego); ferramentas de AutoML (Google AutoML, AutoKeras) buscam automaticamente a melhor arquitetura de rede neural para um problema.
- **Terminologias correlatas** — "cognitive computing" (termo popularizado pela IBM/Watson nos anos 2010) e "neural symbolic integration" (combinação de redes neurais com regras simbólicas) são citados como variações e tendências do campo.

## Checklist de Estudo
- [ ] Sei explicar por que um Perceptron simples não resolve o problema XOR
- [ ] Sei descrever o ciclo forward pass → cálculo do erro → backpropagation → atualização de pesos
- [ ] Sei diferenciar dados linearmente separáveis de não separáveis e o papel da camada oculta
- [ ] Sei explicar o conceito de aproximador universal e suas limitações práticas
- [ ] Sei citar técnicas de mitigação de overfitting e instabilidade de treino (early stopping, learning rate warm-up, regularização)
- [ ] Sei relacionar boas práticas de MLOps (versionamento, reprodutibilidade, canary release, feature flags) com o deploy seguro de modelos de redes neurais

## Palavras-chave
Perceptron · Backpropagação · Deep Learning

## Referências
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. New York: Springer, 2006. Disponível em: https://link.springer.com/book/9780387310732. Acesso em: 23 fev. 2026.
- BROWNLEE, J. *Crash Course on Multi-Layer Perceptron Neural Networks*. 2016. Disponível em: https://machinelearningmastery.com/neural-networks-crash-course/. Acesso em: 23 fev. 2026.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge: MIT Press, 2016. Disponível em: https://www.deeplearningbook.org/. Acesso em: 23 fev. 2026.
- LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. *Nature*, v. 521, n. 7553, p. 436–444, 2015. Disponível em: https://www.nature.com/articles/nature14539. Acesso em: 23 fev. 2026.
- LUIS. *Redes Neurais: Guia Completo com Tipos, Aplicações e Melhores Recursos*. 2023. Disponível em: https://www.etechpt.com/redes-neurais-guia-completo/. Acesso em: 23 fev. 2026.
- MINSKY, M.; PAPERT, S. *Perceptrons: An Introduction to Computational Geometry*. Cambridge: MIT Press, 1969. Disponível em: https://mitpress.mit.edu/9780262630221/perceptrons/. Acesso em: 23 fev. 2026.
- NIELSEN, M. *Neural Networks and Deep Learning*. 2015. Disponível em: https://neuralnetworksanddeeplearning.com/. Acesso em: 23 fev. 2026.
- SCHMIDHUBER, J. Deep Learning in Neural Networks: An Overview. *Neural Networks*, v. 61, p. 85–117, 2015. Disponível em: https://www.sciencedirect.com/science/article/pii/S0893608014002135. Acesso em: 23 fev. 2026.
- SPADINI, A. *Redes neurais: o que são e como funcionam*. 2024. Disponível em: https://www.alura.com.br/artigos/redes-neurais. Acesso em: 23 fev. 2026.
- WIKIPÉDIA. *Artificial neural network*. 2026. Disponível em: https://en.wikipedia.org/wiki/Artificial_neural_network. Acesso em: 23 fev. 2026.
