# Aula 06 — Padrões de Código e Estilo

## Visão Geral
Aula da disciplina "Engenharia de Software para Cientista de Dados" (Fase 3) dedicada a **padrões de código, convenções de estilo e design patterns** aplicados ao dia a dia de quem faz ciência de dados. A premissa central é que "código é lido muito mais vezes do que é escrito" — por isso, escrever de forma legível, padronizada e bem estruturada não é frescura estética, mas um requisito de manutenibilidade e colaboração. A aula conecta três camadas: (1) convenções de estilo (PEP 8), (2) princípios de código limpo (DRY, KISS, Single Responsibility, YAGNI) e (3) padrões de projeto (Strategy, Observer, Factory), sempre trazendo o paralelo com pipelines e scripts de ciência de dados.

## Tópicos Abordados
- Convenções de estilo em Python (PEP 8): nomenclatura, indentação, legibilidade
- Hands On: refatoração de funções duplicadas de avaliação de modelos aplicando DRY
- Princípios de Clean Code: DRY, KISS, Single Responsibility Principle (SOLID)
- YAGNI e o risco de over-engineering
- Refatoração contínua e "bad smells" (maus cheiros de código)
- Padrões de Projeto (Design Patterns): Strategy, Observer, Factory
- Risco de "patternitis" — aplicar padrões sem necessidade real
- Mercado, cases e tendências ligados a qualidade de código e DataOps

## Conceitos-Chave

### Convenções de Estilo (PEP 8)
O **PEP 8 — Style Guide for Python Code** (criado em 2001 por Guido van Rossum) define regras como: nomes de variáveis e funções em **snake_case**, nomes de classes em **CamelCase**, indentação consistente de 4 espaços e limite de 79 caracteres por linha. A ideia-chave, citando Martin Fowler: "código limpo é código que humanos entendem" — não basta funcionar, precisa ser compreensível para outras pessoas (e para o próprio autor no futuro).

Benefícios práticos:
- Reduz a "carga cognitiva" de quem lê, deixando o foco na lógica e não na formatação.
- Permite verificação automática via **linters** (ex.: flake8) e **formatadores** (ex.: black), frequentemente integrados ao pipeline de CI — código fora do padrão nem chega à produção.
- Em notebooks Jupyter, aplicar convenções desde o início (nomes consistentes como `X_train`/`X_test`, organização em seções) facilita a conversão de experimentos exploratórios em scripts de produção.

### DRY (Don't Repeat Yourself)
"Cada pedaço de informação deve ter uma representação única, não ambígua e definitiva no sistema." Código duplicado é sinal de que deveria ser refatorado em função ou módulo reutilizável. A motivação é quase matemática: com **N trechos duplicados**, um bug exige correção em N lugares, e a chance de esquecer algum cresce com N. O princípio se aplica também a configurações e caminhos de dados — centralizar em um único local evita inconsistências quando algo muda.

### KISS (Keep It Simple, Stupid)
Mantenha o design o mais simples possível: não complicar o que pode ser resolvido de forma direta. Evitar excesso de funcionalidades e camadas desnecessárias traz dois ganhos concretos:
- **Performance** — menos passos significam menos operações de CPU e memória (ex.: operações vetorizadas do NumPy em vez de loops Python puro).
- **Verificabilidade** — lógica concentrada em um bloco coeso é mais fácil de validar do que espalhada em várias funções interligadas.

### Single Responsibility Principle (SRP)
Parte do acrônimo **SOLID**: cada módulo ou classe deve ter uma — e apenas uma — razão para mudar. Um script que baixa dados, limpa, treina modelo e envia e-mail de resultado viola o SRP; o ideal é separar em componentes (acesso a dados, processamento, modelagem, notificação). Benefícios: mudanças ficam confinadas ao módulo responsável, testes unitários ficam mais simples, e componentes isolados favorecem reuso — resultando em **alta coesão e baixo acoplamento**.

### YAGNI e Over-Engineering
**YAGNI (You Aren't Gonna Need It)** alerta contra implementar funcionalidades "para o caso de precisar no futuro" quando isso só adiciona complexidade sem valor imediato. Andando lado a lado com KISS, evita o **over-engineering** (superengenharia): código complexo demais por tentar abarcar cenários hipotéticos ou usar abstrações sofisticadas sem necessidade real.

### Refatoração Contínua e Bad Smells
Refatorar é melhorar a estrutura interna do código **sem mudar seu comportamento externo**. A aula cita a história de "Joe e Jane" (Feifke, 2023): Jane parava periodicamente para refatorar e eliminar duplicações, enquanto Joe só acumulava funcionalidades — resultado: Joe travou em bugs, Jane acelerou com uma base de código limpa. Frase-chave (atribuída a Martin Fowler): **"a única maneira de ir rápido é indo bem"**.

**Bad smells** (maus cheiros) a observar: funções muito longas, condicionais aninhadas em excesso, flags para alterar comportamento de função, variáveis com nomes genéricos (`temp`, `data`, `value`). Técnicas de correção: extrair função, renomear para clareza, mover função para módulo mais coerente, reduzir aninhamento (ex.: retornos antecipados em vez de if/else encadeados).

### Padrões de Projeto (Design Patterns)
Soluções arquiteturais reutilizáveis para problemas recorrentes, catalogadas pelo "Gang of Four" (Gamma, Helm, Johnson, Vlissides, 1994) — 23 padrões clássicos. A aula aprofunda três:

**Strategy (Estratégia)** — padrão comportamental que encapsula algoritmos intercambiáveis em classes separadas com interface comum, permitindo trocar o algoritmo em tempo de execução sem alterar o código cliente. Exemplo em data science: uma função genérica de treino que aceita diferentes "estratégias de otimização" (Gradiente Descendente, Adam, LBFGS) por trás da mesma interface. O próprio `avaliar_modelo` do hands on já é uma aplicação embrionária desse conceito.

**Observer (Observador)** — relação um-para-muitos: mudanças em um objeto "sujeito" disparam notificações a vários "observadores" (padrão base de sistemas de eventos, pub-sub e arquitetura MVC). Exemplo em ciência de dados: um loop de treino notifica "terminei a época 5, acurácia 0.8" e diferentes observadores reagem de forma independente (logar em arquivo, atualizar dashboard, ajustar parâmetro).

**Factory (Fábrica)** — padrão criacional que abstrai a criação de objetos, centralizando a lógica de decisão de qual classe concreta instanciar (ex.: `cria_modelo(tipo)` ou `carrega_dados(fonte)` que decide internamente qual implementação usar). O código cliente não precisa conhecer os detalhes, apenas a interface comum — favorece o **princípio aberto/fechado (OCP)**.

Alerta importante: evitar **patternitis** — aplicar padrões de projeto a todo custo, mesmo quando uma solução simples (um `if` ou um callback) resolveria. Padrões devem ser usados quando claramente beneficiam qualidade e flexibilidade, nunca para "mostrar erudição".

## Exercício Hands-On (do material)
Cenário: avaliar dois algoritmos de classificação (AdaBoost e Random Forest) no dataset do Titanic. Uma implementação apressada criaria duas funções quase idênticas — uma por modelo —, violando o DRY. A refatoração demonstrada unifica a lógica em uma única função genérica `avaliar_modelo`, que recebe a classe do modelo como parâmetro, seguindo as convenções da PEP 8 (nomes em snake_case, indentação de 4 espaços).

## Exemplos de Código

Função genérica aplicando o princípio DRY para avaliar diferentes modelos de classificação sem duplicar lógica:

```python
def avaliar_modelo(dados, ClasseModelo):
    X = dados[["is_male", "SibSp", "Pclass", "Fare"]]
    y = dados["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = ClasseModelo()
    modelo.fit(X_train, y_train)
    return modelo.score(X_test, y_test)

# Exemplo de uso: basta trocar a classe do modelo, sem duplicar código
acuracia_rf = avaliar_modelo(df, RandomForestClassifier)
acuracia_ada = avaliar_modelo(df, AdaBoostClassifier)
```

A função encapsula treino e teste; para testar um terceiro algoritmo, basta reutilizá-la passando outra classe — nenhuma lógica nova precisa ser escrita, o que ilustra tanto o DRY quanto, em espírito, o padrão Strategy (o "algoritmo" de modelo é passado como parâmetro intercambiável).

Ilustração conceitual (adaptada do material) de uma *Factory* simples para centralizar a criação de modelos, evitando condicionais espalhadas pelo código:

```python
def cria_modelo(tipo):
    """Centraliza a decisão de qual modelo instanciar."""
    if tipo == "arvore":
        return RandomForestClassifier()
    elif tipo == "rede":
        return MLPClassifier()
    elif tipo == "boost":
        return AdaBoostClassifier()
    raise ValueError(f"Tipo de modelo desconhecido: {tipo}")

# O código chamador não precisa saber os detalhes de cada classe
modelo = cria_modelo("arvore")
```

## Cases e Tendências de Mercado
- **IA Generativa e Automação de Código**: ferramentas como ChatGPT e Claude vêm sendo usadas para gerar insights e otimizar código, acelerando ciclos de desenvolvimento em projetos de dados (MIT Sloan Review).
- **Real-Time Data Engineering e Zero-ETL**: arquiteturas orientadas a eventos e pipelines em tempo real em destaque, com ferramentas como Azure Stream Analytics, Databricks Structured Streaming e Microsoft Fabric Real-Time Analytics (DataEX).
- **DataOps e Governança Avançada**: aplicação de princípios DevOps à gestão de dados, garantindo qualidade e rastreabilidade em ambientes complexos (Coffee and Tips).
- **Target (Varejo)**: uso de algoritmos preditivos para personalização de ofertas, aumentando vendas e fidelização (Intercompany).
- **Hospital Mount Sinai (Saúde)**: projeto "Deep Patient" aplicando machine learning para prever doenças a partir de prontuários, melhorando diagnósticos.
- **Fracasso emblemático — Knight Capital (2012)**: perda de US$ 440 milhões em 45 minutos por falhas de código legado e ausência de testes — alerta sobre a importância de boas práticas e governança.
- **"Best Practices for Scientific Computing"** (Wilson et al., PLOS Biology): reforça princípios como DRY e modularidade para garantir reprodutibilidade científica.
- Artigos e vídeos complementares sobre código limpo e sustentável (Datanovia), legibilidade e refatoração (conversa com Fabio Akita, Alura) e padrões de projeto na prática com Strategy, Factory e SOLID (Dialogando TI).

## Checklist de Estudo
- [ ] Sei explicar por que convenções de estilo (PEP 8) importam além da estética
- [ ] Sei nomear variáveis, funções e classes seguindo snake_case/CamelCase
- [ ] Sei identificar código duplicado e refatorá-lo aplicando DRY
- [ ] Sei explicar o princípio KISS e seu impacto em performance e verificabilidade
- [ ] Sei aplicar o Single Responsibility Principle para separar responsabilidades em um pipeline
- [ ] Sei diferenciar YAGNI de over-engineering e reconhecer quando estou "superengenheirando"
- [ ] Sei reconhecer "bad smells" de código (funções longas, aninhamento excessivo, nomes genéricos)
- [ ] Sei descrever os padrões Strategy, Observer e Factory e dar um exemplo de cada em ciência de dados
- [ ] Sei evitar "patternitis" — aplicar padrões apenas quando realmente beneficiam o código
- [ ] Conheço o case Knight Capital como exemplo de risco de código mal gerenciado

## Palavras-chave
Código Limpo · Padrões de Projeto · Manutenibilidade de Software

## Referências
- FEIFKE, B. *Refactoring For Data Scientists: A Beginner's Guide*. 2023. Disponível em: https://benfeifke.com/posts/refactoring-for-data-scientists-a-beginners-guide/. Acesso em: 15 dez. 2025.
- FINER, J. *How to Write Beautiful Python Code With PEP 8*. 2025. Disponível em: https://realpython.com/python-pep8/. Acesso em: 15 dez. 2025.
- FREECODECAMP. *Clean Code explicado: um guia prático para iniciantes*. 2024. Disponível em: https://www.freecodecamp.org/portuguese/news/clean-code-explicado-um-guia-pratico-para-iniciantes/. Acesso em: 15 dez. 2025.
- MARTIN, R. C. *Clean Code: A Handbook of Agile Software Craftsmanship*. Upper Saddle River: Prentice Hall, 2008. ("Código Limpo", tradução em português por Alta Books, 2009.)
- TORRES NETO, F. S. *Design Patterns: Os Padrões de Projeto mais Usados*. 2025. Disponível em: https://manual-si-ufc.blogspot.com/2025/03/design-patterns-os-padroes-de-projeto.html. Acesso em: 15 dez. 2025.
- WATI. *O caso: Knight Capital Group*. 2025. Disponível em: https://www.wati.com.br/post/o-caso-knight-capital-group. Acesso em: 15 dez. 2025.
- WILSON, G. et al. *Best Practices for Scientific Computing*. PLoS Biology, v. 12, n. 1, e1001745, 2014. Disponível em: https://doi.org/10.1371/journal.pbio.1001745. Acesso em: 15 dez. 2025.
