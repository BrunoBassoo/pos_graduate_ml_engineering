# Aula 08 — Refatoração de Projeto DS

## Visão Geral
Aula final da disciplina de Engenharia de Software para Cientista de Dados. Trata de **refatoração** — melhorar o design interno do código sem alterar seu comportamento observável — aplicada especificamente a projetos de ciência de dados. O fio condutor é sair de notebooks longos, com estados implícitos e "funções Deus", para módulos coesos, testáveis e tipados; sair de data wrangling duplicado para pipelines reutilizáveis; e transformar "provas de conceito" em artefatos evolutivos. A aula conecta teoria (code smells, técnicas clássicas de refatoração) e prática (testes com pytest, pipelines do scikit-learn, medição de desempenho, empacotamento com `pyproject.toml`), fechando o módulo com uma visão de que refatoração em ML vai além do código — envolve também dados, infraestrutura e governança.

## Tópicos Abordados
- O que é refatoração e por que ela é essencial para projetos de DS sobreviverem além do primeiro resultado
- Hands On: transformação de um projeto legado "bagunçado" (`train_and_plot`, função que mistura leitura de dados, treino e plot) em um projeto modular com pipeline scikit-learn e testes pytest
- Saiba Mais:
  - Bad smells em código de Data Science e refatorações clássicas (Extract Function, Rename, Split Function, Encapsulate Variable, Delete Dead Code)
  - Refatoração segura com testes automatizados (testes de caracterização) e pipelines
  - Desempenho, memória e boas práticas computacionais (broadcasting, views vs. copies, complexidade algorítmica)
  - Reestruturação da arquitetura do projeto: de notebooks soltos a pacote Python instalável (`pyproject.toml`)
  - Pipelines e separação de responsabilidades (Split Phase, prevenção de data leakage)
  - O dilema dos notebooks: como "domar" e integrar à engenharia
  - Dívida técnica em sistemas de ML — refatoração além do código (dados, monitoramento, deploy)
- Mercado, Cases e Tendências: Metaflow (Netflix), Knowledge Repo (Airbnb), caso Zillow Offers, ferramentas e materiais abertos para aprofundamento

## Conceitos-Chave

### Code smells em projetos de Data Science
Indícios superficiais de problemas de design. Os mais recorrentes em DS:
- **Código duplicado** — mesma lógica copiada em vários notebooks/scripts → sugere falta de abstração.
- **Funções "Deus" (God Function)** — uma função que faz limpeza, treino e plot ao mesmo tempo → viola o princípio da responsabilidade única.
- **Variáveis globais excessivas** — dependências implícitas difíceis de rastrear → indicam acoplamento oculto.
- **Nomes crípticos** (`df2`, `temp`) — dificultam a leitura e aumentam a propensão a erros.
- **Comentários excessivos** — muitas vezes mascaram código confuso; Fowler defende refatorar até o comentário se tornar desnecessário.

### Refatorações fundamentais (catálogo de Martin Fowler)
Duas características centrais: (a) é feita em **pequenos passos**, rodando testes a cada passo; (b) foca em **qualidade interna**, não em adicionar funcionalidades. Técnicas citadas: Extract Function/Module (isolar lógica duplicada), Split Function/Phase (separar uma função gigante em etapas menores), Rename Variable/Method (nomes significativos), Encapsulate Variable (substituir globais por parâmetros/atributos de classe), Delete Dead Code.

### Testes automatizados como guarda-chuva da refatoração
Refatorar só é seguro com testes cobrindo o comportamento atual. Prática central: o **teste de caracterização** — captura o output atual (ex.: acurácia de um modelo) dado um input conhecido, servindo de contrato antes de qualquer mudança. Se o número monitorado mudar sem motivo após a refatoração, o teste falha e sinaliza problema. O framework **pytest** permite parametrizar testes, criar fixtures reutilizáveis e medir cobertura de código (>80% é uma referência útil, não uma meta absoluta). Integração com CI (ex.: GitHub Actions rodando testes a cada push) automatiza essa rede de segurança.

### Pipelines do scikit-learn contra vazamento de dados
O `Pipeline` do scikit-learn encadeia pré-processamento e modelo em um único objeto, garantindo que transformações (ex.: `StandardScaler`) sejam ajustadas (`fit`) somente com dados de treino e aplicadas de forma idêntica no teste — evitando **data leakage**. Isso substitui código manual espalhado, reduz erro, simplifica testes (uma linha testa o pipeline inteiro) e facilita grid search conjunto sobre hiperparâmetros do modelo e das etapas de pré-processamento.

### Desempenho, memória e boas práticas computacionais
Ao refatorar, oportunidades de otimização costumam aparecer — mas a ordem correta é **primeiro garantir correção, depois otimizar**, validando que o resultado final não mudou (teste de regressão de performance). Dois conceitos centrais em Python:
- **Broadcasting (NumPy)** — aplica operações vetorizadas em arrays de tamanhos diferentes sem loops Python explícitos, executando em C (muito mais rápido); porém pode expandir dados implicitamente e consumir memória em excesso se não for testado com volumes maiores.
- **Views vs. Copies (pandas)** — fatiar um DataFrame pode retornar uma *view* (mesma memória, risco de efeito colateral no original) ou uma *copy* (mais RAM, mais segura). O `copy-on-write` (pandas ≥ 2.1) reduz a ambiguidade; usar `df.copy()` explicitamente e ficar atento a `SettingWithCopyWarning` evita bugs sutis.
- **Complexidade algorítmica** — ao modularizar, vale perguntar a ordem de complexidade de cada função (ex.: uma busca linear dentro de um loop pode introduzir O(n²) sem perceber); trocar listas por sets/dicts pode levar buscas de O(n) para O(1) amortizado.

### Reestruturação arquitetural: de notebooks soltos a pacote Python
Projetos de DS costumam começar como sequências de notebooks (`analise1.ipynb`, `analise2.ipynb`...) que viram uma "colcha de retalhos". A prática recomendada é adotar uma estrutura de pacote instalável:
```
project_name/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── __init__.py
├── tests/
├── pyproject.toml
└── README.md
```
O `pyproject.toml` define dependências, versão de Python e *entry points* (scripts de linha de comando), permitindo `pip install -e .` e integração com CI/CD. Ganhos: manutenibilidade (localizar código fica trivial), modularidade (baixo acoplamento entre `data` e `models`), testabilidade (testes organizados por módulo) e integração contínua simplificada.

### Split Phase e separação temporal de responsabilidades
Um projeto de DS pode ser dividido em etapas independentes: ingestão, limpeza, feature engineering, treinamento, validação, deploy. A refatoração **Split Phase** separa fases que antes estavam misturadas em um único script/notebook (ex.: de consultas SQL até avaliação do modelo) em módulos distintos (`data_ingestion`, `model_training`), permitindo inclusive rodá-las em ambientes diferentes.

### O dilema dos notebooks
Notebooks são valorizados pela interatividade, mas propensos a más práticas: execução fora de ordem, estados implícitos, dificuldade de versionamento. Disciplinas recomendadas: modularizar dentro do próprio notebook (funções/classes em vez de células soltas); sempre fazer *Restart and Run All* antes de considerar o resultado válido; orquestrar notebooks externamente com ferramentas como **Papermill** ou `nbconvert`; versionar com cuidado (ex.: `nbstripout` para remover outputs/metadados ruidosos); e migrar lógica estável do notebook para módulos `.py` de produção. Exemplos de mercado: Databricks (notebooks em produção com boas práticas de Git, testes via `%run`) e Netflix Metaflow (transforma experimentos exploratórios em flows reproduzíveis e versionados).

### Dívida técnica em sistemas de ML além do código
Citando o artigo "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., Google), a maior parte de um sistema de ML é infraestrutura — dados, configuração, monitoramento — não apenas o código do modelo. Smells organizacionais incluem datasets duplicados, ausência de monitoramento de performance e scripts de deploy manuais não reproduzíveis. Refatorar nesse nível significa normalizar pipelines de dados, consolidar fontes de verdade de features e automatizar implantação via infraestrutura como código.

## Exercício Hands-On (do material)
Refatoração guiada, em três etapas, de um projeto legado de classificação:

1. **Código legado** (`src/legacy_train.py`) — função `train_and_plot(csv_path)` que lê CSV, remove nulos, treina uma `LogisticRegression` e ainda teria efeitos colaterais de plot/gravação, tudo em uma única função (code smell de função "Deus" + responsabilidade múltipla).
2. **Extração do pipeline** (`src/pipeline/model.py`) — criação de `build_pipeline()`, que monta um `Pipeline` scikit-learn com `StandardScaler` + `LogisticRegression`, separando pré-processamento de modelagem.
3. **Teste com pytest** (`tests/test_pipeline.py`) — cria um DataFrame mínimo, treina o pipeline e valida que a acurácia no próprio conjunto é ≥ 0,5, funcionando como rede de segurança para futuras mudanças (reforçando a importância do teste de caracterização antes de refatorar).

Em paralelo, o material orienta reorganizar a estrutura de pastas criando um pacote `pipeline/` com módulos e configurar um `pyproject.toml` para instalação em modo editável (`pip install -e .`).

## Exemplos de Código

Função legada antes da refatoração — mistura leitura, pré-processamento, treino e efeitos colaterais:

```python
# arquivo: src/legacy_train.py (antes da refatoração)
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_and_plot(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df[["x1", "x2", "x3"]]; y = df["target"]
    model = LogisticRegression(max_iter=200).fit(X, y)
    # ... vários efeitos colaterais (plots, gravação etc.)
    return model.score(X, y)
```

Depois da refatoração — pipeline isolado e testável:

```python
# arquivo: src/pipeline/model.py (depois)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
```

Teste de caracterização / regressão com pytest:

```python
# arquivo: tests/test_pipeline.py
import pandas as pd
from pipeline.model import build_pipeline

def test_pipeline_runs(tmp_path):
    df = pd.DataFrame({"x1": [0, 1], "x2": [1, 0], "x3": [0.5, 0.1], "target": [0, 1]})
    X, y = df[["x1", "x2", "x3"]], df["target"]
    pipe = build_pipeline()
    pipe.fit(X, y)
    assert pipe.score(X, y) >= 0.5
```

Trecho mínimo de `pyproject.toml` configurando o projeto como pacote instalável:

```toml
[project]
name = "ds_refatorado"
version = "0.1.0"
dependencies = [
    "pandas>=2.1",
    "scikit-learn>=1.4",
    "pytest>=8.0"
]
[project.scripts]
treinar-modelo = "src.models.train:main"
```

## Cases e Tendências de Mercado
- **Metaflow (Netflix)** — plataforma criada para aproximar pesquisa e produção, formalizando *flows* e versionamento de artefatos; exemplo de "refatoração de processo": padronizar estrutura, declarar dependências e permitir scale-out sem reescrever tudo.
- **Airbnb Knowledge Repo** — institucionaliza "posts reprodutíveis" (notebooks + metadados), reduzindo a duplicação de análises — um exemplo de refatoração organizacional, não só de código.
- **Zillow Offers (2021)** — caso de fracasso citado como alerta: o serviço de precificação imobiliária via ML colapsou, gerando grandes prejuízos, em parte por não ter sistemas robustos de monitoramento e ajuste do modelo conforme o mercado mudava (acúmulo de dívida técnica de ML). Quando os dados mudaram drasticamente (pandemia), o modelo errou e a empresa não reagiu a tempo — reforça a importância de planejar feedback loops e atualizações desde o início.
- **Databricks** — recomenda boas práticas para notebooks em produção: desenvolvimento em repositórios Git integrados, testes com `%run`, separação de configuração.
- Recursos abertos recomendados para aprofundamento: catálogo de refatorações de Martin Fowler, guia de empacotamento moderno da PyPA/Real Python, a palestra "I Don't Like Notebooks" (Joel Grus) e o guia de boas práticas de notebooks do Databricks.

## Checklist de Estudo
- [ ] Sei identificar code smells comuns em projetos de DS (duplicação, função "Deus", globais, nomes crípticos)
- [ ] Sei aplicar refatorações clássicas (Extract Function, Rename, Split Phase, Encapsulate Variable)
- [ ] Sei explicar o papel de um teste de caracterização antes de refatorar código crítico
- [ ] Sei montar um `Pipeline` scikit-learn para evitar vazamento de dados entre treino e teste
- [ ] Sei diferenciar broadcasting, views e copies e seus impactos em desempenho/memória
- [ ] Sei estruturar um projeto de DS como pacote Python instalável com `pyproject.toml`
- [ ] Sei descrever boas práticas para "domar" notebooks (modularização, Restart and Run All, orquestração externa)
- [ ] Sei explicar por que dívida técnica em ML vai além do código (dados, monitoramento, deploy) e citar o caso Zillow Offers

## Palavras-chave
Refatoração incremental; Pipelines de ML; Arquitetura modular em Data Science.

## Referências
- AIRBNB. *Airbnb – Knowledge Repo*. 2025. Disponível em: https://airbnb.io/projects/knowledge-repo/. Acesso em: 16 dez. 2025.
- AIRBNB. *Knowledge Repo v0.9.3 documentation*. 2025. Disponível em: https://knowledge-repo.readthedocs.io/en/latest/. Acesso em: 16 dez. 2025.
- DATABRICKS. *Software engineering best practices for notebooks*. 2025. Disponível em: https://docs.databricks.com/aws/en/notebooks/best-practices. Acesso em: 16 dez. 2025.
- FEIFKE, B. *Refactoring for Data Scientists: A Beginner's Guide*. 2023. Disponível em: https://benfeifke.com/posts/refactoring-for-data-scientists-a-beginners-guide/. Acesso em: 16 dez. 2025.
- FOWLER, M. *Refactoring – Catalog of Refactorings*. 2025. Disponível em: https://refactoring.com/catalog/. Acesso em: 16 dez. 2025.
- GRUS, J. *I Don't Like Notebooks*. 2018. Disponível em: https://www.youtube.com/watch?v=7jiPeIFXb6U. Acesso em: 16 dez. 2025.
- NETFLIX. *Metaflow – Docs*. s.d. Disponível em: https://docs.metaflow.org/. Acesso em: 16 dez. 2025.
- NUMPY. *NumPy — Broadcasting*. 2025. Disponível em: https://numpy.org/doc/stable/user/basics.broadcasting.html. Acesso em: 16 dez. 2025.
- PANDAS. *pandas docs (2.3.x)*. 2025. Disponível em: https://pandas.pydata.org/docs/. Acesso em: 16 dez. 2025.
- PYPA. *Writing your pyproject.toml*. 2025. Disponível em: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/. Acesso em: 16 dez. 2025.
- PYTEST. *Get Started – pytest*. 2025. Disponível em: https://docs.pytest.org/en/stable/getting-started.html. Acesso em: 16 dez. 2025.
- SCIKIT-LEARN. *Pipeline — scikit-learn*. 2025. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html. Acesso em: 16 dez. 2025.
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. 2015. Disponível em: https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf. Acesso em: 16 dez. 2025.
