# Aula 02 — Git e Controle de Versão

## Visão Geral
Segunda aula da disciplina "Engenharia de Software para Cientista de Dados" (Fase 3). O material parte de um problema comum em projetos de ciência de dados — perder trabalho por sobrescrever código ou nomear arquivos como "analise_final_v2_definitivo" — para apresentar o **Git** como a solução padrão de **controle de versão distribuído**. A aula combina um Hands On prático (inicializar um repositório, versionar um script Python e simular commits) com uma seção teórica densa sobre boas práticas de commits, estratégias de branching (Git Flow, GitHub Flow, Trunk-Based Development), revisão de código via Pull Requests, resolução de conflitos de merge e a relação entre Git/GitHub e os fluxos de trabalho de ciência de dados e MLOps (incluindo DVC e versionamento de dados/modelos).

## Tópicos Abordados
- O que vem por aí: motivação para controle de versão em projetos de ciência de dados
- Hands On: inicializar repositório, criar e versionar um script Python (`analise.py`) com commits sequenciais
- Saiba Mais: modelo distribuído do Git, commits atômicos, mensagens de commit (regra 50/72, Conventional Commits)
- Branches leves do Git; `git revert` vs. `git reset`; modelo de snapshot e hashing (SHA-1/SHA-256); commit DAG
- Git aplicado à ciência de dados: limitações com arquivos grandes, Git LFS e DVC
- Estratégias de branching: Git Flow, GitHub Flow, Trunk-Based Development (TBD) e GitLab Flow
- Pull Requests e code review: tamanho de PR, descrição, revisores, automação (linters/CI), SLA de revisão, merge e pós-merge
- Conflitos de merge: causas, prevenção e estratégias de resolução
- Boas práticas de commits em equipe, `.gitignore`, branches protegidas e secret scanning
- CI/CD aplicado a pipelines de dados e modelos de ML (introdução a MLOps)
- Mercado, Cases e Tendências: falhas históricas de versionamento (Knight Capital), benefícios do Git/GitHub em ciência de dados (Pew Research Center), tendências de branching, estatísticas de uso do GitHub e ferramentas de versionamento de dados/ML (DVC, MLflow)

## Conceitos-Chave

### Controle de Versão Distribuído
O Git registra cada alteração como um **commit** com mensagem descritiva, construindo um histórico completo e permitindo reverter mudanças com segurança. Diferente de sistemas centralizados antigos (como SVN), no modelo **distribuído** cada colaborador(a) possui uma cópia completa do repositório (todos os arquivos e todo o histórico de commits) em sua própria máquina, permitindo operações rápidas e offline. O Git é usado por cerca de 93% dos especialistas em desenvolvimento (pesquisa de 2023) por sua flexibilidade, velocidade e modelo distribuído.

### Commits Atômicos e Mensagens de Qualidade
Um **commit atômico** representa uma única alteração lógica (ex.: corrigir um bug específico, adicionar uma função). Esse hábito facilita identificar quando um problema foi introduzido e torna viável reverter apenas uma mudança isolada. Commits "monolíticos" (muitas alterações distintas de uma vez) dificultam a análise de histórico e a reversão parcial.

A **mensagem de commit** funciona como documentação do histórico do projeto. Boas práticas incluem:
- Regra dos **50/72 caracteres**: linha de assunto com até 50 caracteres, corpo explicativo (se necessário) com quebras em 72 caracteres.
- Começar a mensagem com verbo no imperativo ou seguir a convenção **Conventional Commits** (prefixos `feat`, `fix`, `docs`, `refactor`, `test`), que também permite gerar changelogs automaticamente.
- Referenciar issues rastreáveis (ex.: "resolves #23").

### Branches, Merge, Rebase e Reversão Segura
Criar uma **branch** no Git é uma operação leve e quase instantânea, o que incentiva isolar o desenvolvimento de funcionalidades e experimentos sem afetar o código estável. É recomendado atualizar regularmente a branch de feature com as mudanças da `main` (via **merge**, que cria um commit de junção e preserva o histórico divergente, ou **rebase**, que reaplica os commits sobre a nova base, criando histórico linear), para evitar divergências grandes e conflitos extensos.

Um mandamento do controle de versão é **não apagar histórico compartilhado**. Para desfazer mudanças já publicadas, usa-se `git revert` (cria um commit inverso que anula uma mudança, preservando o histórico) em vez de `git reset` (que remove commits do histórico e só deve ser usado em commits locais ainda não enviados ao repositório remoto). O Git armazena dados como **snapshots** (não diffs lineares) e cada commit tem um identificador hash (SHA-1/SHA-256) derivado do conteúdo e dos commits antecessores, formando um **grafo acíclico dirigido (DAG)** que garante integridade do histórico.

### Git em Projetos de Ciência de Dados
Datasets grandes e arquivos binários de modelo não versionam bem com Git, que foi projetado para código-texto. Ferramentas como **Git LFS** e, principalmente, o **DVC (Data Version Control)** resolvem isso rastreando metadados de arquivos pesados nos commits enquanto armazenam o conteúdo externamente — uma separação de responsabilidades: Git para código, DVC para dados/artefatos volumosos. Isso preserva a reprodutibilidade científica (saber exatamente qual versão de dados e código gerou um resultado).

### Estratégias de Branching
- **Git Flow**: branches com papéis específicos (`main`/produção, `develop`/integração, `feature/*`, `release/*`, `hotfix/*`). Traz estrutura e clareza, mas é mais pesado e lento para lançar; adequado a projetos com versões formais (X.Y.Z) ou ambientes críticos.
- **GitHub Flow**: workflow simples centrado na `main`, sempre pronta para deploy. Cada mudança é feita em branch curta, integrada via Pull Request após revisão. Exige maturidade em testes automatizados e CI/CD.
- **Trunk-Based Development (TBD)**: branches de vida muito curta (horas/dias), merges extremamente frequentes na `main` (trunk), muitas vezes com merge queues e feature flags. É quase sinônimo de Continuous Integration estrita; associado a métricas DORA mais altas, mas exige forte cultura de testes e automação.
- **GitLab Flow**: variação híbrida que adiciona branches de ambiente (ex.: `main` espelhando produção, `staging` espelhando homologação).

Nenhum modelo é intrinsecamente melhor — a escolha depende do tamanho da equipe, frequência de releases e criticidade do software.

### Pull Requests e Code Review
Boas práticas: PRs pequenos e focados (uma feature/correção por PR); descrição clara do que foi feito e por quê, referenciando issues; pelo menos um revisor (four eyes principle) com comentários construtivos; automação de linters/formatadores (flake8, black) e análise estática no CI antes da revisão humana; SLA informal de revisão (ex.: até 1 dia útil); e, no merge, decisão consciente sobre squash de commits. Revisão de código reduz custo de correção de defeitos e promove compartilhamento de conhecimento — inclusive detectando problemas específicos de ciência de dados, como um gráfico sem rótulo de eixo ou um cálculo estatístico ineficiente.

### Conflitos de Merge
Estratégias para minimizar e resolver conflitos: comunicação e coordenação entre quem edita os mesmos arquivos; atualizar branches de longa duração frequentemente; usar ferramentas de diff/merge de três vias (KDiff3, Meld, VS Code); ao resolver, considerar que as duas mudanças podem precisar coexistir (não é sempre "uma ganha, outra perde"); e evitar conflitos estruturalmente modularizando o código em arquivos/funções menores. Notebooks Jupyter são especialmente propensos a conflitos (JSON extenso) — recomenda-se evitar edição simultânea do mesmo notebook ou convertê-lo em scripts `.py` para trabalho colaborativo.

### CI/CD e MLOps
**CI (Continuous Integration)** executa builds e testes automatizados a cada mudança integrada; **CD (Continuous Delivery/Deployment)** implanta automaticamente mudanças aprovadas. Em projetos de ciência de dados, isso se traduz em testes unitários de funções, testes de performance/acurácia de modelos e pipelines de ETL que rodam a cada Pull Request, alertando se uma métrica piorou. Ferramentas como GitHub Actions viabilizam esse fluxo, aproximando a ciência de dados das práticas tradicionais de engenharia de software — tendência conhecida como **MLOps**.

## Exercício Hands-On (do material)
Criação do projeto fictício **AnaliseClima**, com os seguintes passos práticos em terminal:

1. Criar a pasta do projeto e inicializar o repositório (`git init`), verificando que a branch padrão é `main` e ainda não há commits (`git status`).
2. Criar o script `analise.py`, que calcula a média de uma lista de temperaturas.
3. Registrar o arquivo para acompanhamento (`git add`) e criar o primeiro commit com mensagem seguindo o padrão Conventional Commits (`feat: ...`).
4. Confirmar que o Git identificou a alteração (`1 file changed, 5 insertions(+)`) e que o diretório de trabalho está limpo (`working tree clean`).

## Exemplos de Código

Inicialização do repositório e primeiro commit:

```bash
# Cria a pasta do projeto e inicializa o repositório Git
$ mkdir AnaliseClima && cd AnaliseClima
$ git init
Initialized empty Git repository in /AnaliseClima/.git/

$ git status
On branch main
No commits yet
```

Script Python versionado (versão inicial):

```python
# analise.py - Versão 1
temperaturas = [22.4, 21.8, 23.1, 20.0, 19.5]
media = sum(temperaturas) / len(temperaturas)
print(f"Média das temperaturas = {media:.2f}°C")
```

Adicionando o arquivo e criando o commit inicial com mensagem descritiva:

```bash
$ git add analise.py
$ git commit -m "feat: calcula média de temperatura inicial"
[main (root-commit) ea8321d] feat: calcula média de temperatura inicial
 1 file changed, 5 insertions(+)
 create mode 100644 analise.py

$ git status
On branch main
nothing to commit, working tree clean
```

## Cases e Tendências de Mercado
- **Falhas de versionamento/deploy**: o caso **Knight Capital Group (2012)** é citado como exemplo clássico de um bug que, sem testes de regressão e controle de versão rigoroso, enviou ordens de compra/venda indevidas e gerou um prejuízo de US$ 7 bilhões em ordens. Casos mais recentes (2025) reforçam o padrão: deploy com flag de teste ativo sem revisão dupla causando US$ 440 milhões em prejuízo, e ativação de flag legado sem testes de aceitação (UAT) e regressão gerando perdas rápidas.
- **Benefícios do Git/GitHub em ciência de dados**: o **Pew Research Center** é citado como case de adoção de Git/GitHub em projetos de análise de dados, trazendo consistência entre projetos, transparência e revisão por pares obrigatória via Pull Requests antes de qualquer decisão. Projetos acadêmicos (arXiv, PLoS Biology) também adotam issues, project boards e containers para reprodutibilidade científica.
- **Tendências de branching**: fontes de 2025 (Mergify, estudo da UFPE com desenvolvedores brasileiros, Aviator) indicam que Trunk-Based Development com merge queues favorece integração rápida e é mais indicado para equipes ágeis e pipelines de dados, enquanto Git Flow permanece relevante para releases regulados.
- **Escala do GitHub**: estatísticas de 2025 apontam mais de 150-180 milhões de desenvolvedores(as), mais de 1 bilhão de repositórios, centenas de milhões de Pull Requests e dezenas de milhões de usuários do GitHub Copilot — evidenciando a centralidade do Git/GitHub no desenvolvimento de software moderno.
- **Versionamento de dados e ML**: ferramentas como **DVC** e **MLflow** (e comparativos com W&B) são apontadas como tendência para versionar datasets e modelos de forma integrada ao Git, apoiando práticas de MLOps e rastreabilidade via model registry.
- **Materiais complementares**: cursos gratuitos (freeCodeCamp, Kevin Stratvert), o livro **Pro Git** (git-scm.com) e listas de recursos (GitHub Docs, MLTut) são indicados para aprofundamento.

## Checklist de Estudo
- [ ] Sei explicar por que controle de versão é importante em projetos de ciência de dados
- [ ] Sei inicializar um repositório Git, criar commits e checar o status do repositório
- [ ] Sei escrever mensagens de commit seguindo boas práticas (50/72 caracteres, Conventional Commits)
- [ ] Entendo a diferença entre commits atômicos e commits monolíticos
- [ ] Sei diferenciar `git revert` de `git reset` e sei quando usar cada um com segurança
- [ ] Entendo o modelo de snapshot do Git e por que o histórico é baseado em hashes (commit DAG)
- [ ] Sei por que datasets e modelos grandes não versionam bem no Git puro e como Git LFS/DVC resolvem isso
- [ ] Sei diferenciar Git Flow, GitHub Flow e Trunk-Based Development e quando usar cada um
- [ ] Conheço boas práticas de Pull Request e code review (tamanho, descrição, automação, SLA)
- [ ] Sei aplicar estratégias para prevenir e resolver conflitos de merge
- [ ] Entendo como CI/CD se conecta a pipelines de dados e modelos de ML (MLOps)
- [ ] Conheço cases de mercado que ilustram os riscos de versionamento mal conduzido (ex.: Knight Capital) e os benefícios de boas práticas (ex.: Pew Research Center)

## Palavras-chave
Colaboração em Desenvolvimento de Software · Controle de Versão Distribuído · DevOps e MLOps

## Referências
- CHACON, S.; STRAUB, B. *Pro Git*. 2. ed. Apress, 2014. Disponível em: https://git-scm.com/book/en/v2. Acesso em: 12 dez. 2025.
- DANJOU, J. *Trunk-Based Development vs Gitflow: Which Branching Model Actually Works?* 2025. Disponível em: https://mergify.com/blog/trunk-based-development-vs-gitflow-which-branching-model-actually-works. Acesso em: 12 dez. 2025.
- NILSEN, P. *Version Control Best Practices – Mastering Git, Branching Strategies and Collaborative Workflows*. DEV Community, 17 set. 2024. Disponível em: https://dev.to/pellenilsen/version-control-best-practices-mastering-git-branching-strategies-and-collaborative-workflows-5gal. Acesso em: 12 dez. 2025.
- REMY, E. *How Pew Research Center uses git and GitHub for version control*. 2022. Disponível em: https://www.pewresearch.org/decoded/2022/08/01/how-pew-research-center-uses-git-and-github-for-version-control/. Acesso em: 12 dez. 2025.
- GITHUB. *100 million developers and counting*. 2023. Disponível em: https://github.blog/2023-01-25-100-million-developers/. Acesso em: 12 dez. 2025.
- MARTINS, L. *Git: Boas práticas para escrita das mensagens de commits*. 2023. Disponível em: https://www.dio.me/articles/git-boas-praticas-para-escrita-das-mensagens-de-commits. Acesso em: 12 dez. 2025.
- PEW RESEARCH CENTER. *How we review code at Pew Research Center*. 2022. Disponível em: https://www.pewresearch.org/decoded/2022/09/15/how-we-review-code-at-pew-research-center/. Acesso em: 12 dez. 2025.
- QEEDIO. *When Software Goes Unchecked*. 2025. Disponível em: https://www.qeedio.com/posts-en/when-software-goes-unchecked-financial-giant-knight-capital-nearly-ruined. Acesso em: 12 dez. 2025.
- TANRIKULU, O. *Git Flow vs. Trunk-Based Development: Choosing the Right Branching Strategy*. 2025. Disponível em: https://medium.com/@ozlemtanrikulu/git-flow-vs-trunk-based-development-choosing-the-right-branching-strategy-38d9c2d23b78/. Acesso em: 12 dez. 2025.
