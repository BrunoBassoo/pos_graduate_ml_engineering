# Aula 05 — Gerenciamento de Dependências

## Visão Geral
Aula da disciplina Engenharia de Software para Cientista de Dados (Fase 3) dedicada ao "inferno das dependências" — o problema clássico de bibliotecas incompatíveis ou versões conflitantes que quebram um projeto ao rodar em outra máquina ou após uma atualização. A aula cobre três frentes complementares: **isolar** o ambiente de cada projeto (ambientes virtuais), **documentar e fixar** as versões das bibliotecas (requirements, lockfiles, versionamento semântico) e **empacotar** código reutilizável como biblioteca instalável. O fio condutor é a reprodutibilidade: garantir que um projeto rode "hoje, amanhã ou em qualquer lugar, sempre da mesma forma".

## Tópicos Abordados
- Hands On: criação de um projeto Python com o gerenciador de pacotes **UV**, adicionando dependências e gerando lockfile
- Isolamento de dependências com ambientes virtuais (venv/virtualenv e Conda)
- Documentação e fixação de requisitos (requirements.txt, environment.yml)
- Versionamento semântico (SemVer) e estratégias de pinagem de versões
- Dependências diretas vs. transitivas e ferramentas como pip-tools
- Ferramentas modernas de gerenciamento: Pipenv, Poetry e UV (comparativo)
- Algoritmos de resolução de dependências (dependency resolution, SAT)
- Dependências em MLOps (MLflow e o "snapshot" de ambiente de um modelo)
- Empacotamento e distribuição de código reutilizável (pyproject.toml, wheels, PyPI)
- Mercado, cases e tendências em ciência/engenharia de dados para 2025

## Conceitos-Chave

### Isolamento de dependências: ambientes virtuais
Em vez de instalar pacotes globalmente no sistema (fonte de conflitos entre projetos — ex.: um projeto exige NumPy 1.19 e outro NumPy 1.25), a prática recomendada é usar **ambientes virtuais**: "bolhas" isoladas com uma instalação própria do Python e dos pacotes de cada projeto. O que é instalado no ambiente do Projeto A não afeta o Projeto B.

- **venv/virtualenv**: módulo nativo do Python (`python3 -m venv .env`, ativado com `source .env/bin/activate`); qualquer `pip install` subsequente instala apenas dentro do ambiente.
- **Conda**: gerencia não só pacotes Python mas também dependências nativas/SO (bibliotecas C, MKL, CUDA), o que o torna popular em setups de ML com TensorFlow/PyTorch.
- Isolamento também traz ganhos (sutis) de eficiência — menos bibliotecas carregadas em memória — e facilita replicar a configuração exata em múltiplos nós de um cluster.

### Documentação e fixação de requisitos
Listar dependências em `requirements.txt` (pip) ou `environment.yml` (Conda) permite que qualquer pessoa, ou um servidor de CI/CD, recrie o ambiente com `pip install -r requirements.txt`. Mas **listar sem fixar versão não garante reprodutibilidade temporal**: sem pinagem, uma instalação hoje pode trazer uma versão diferente da de amanhã.

A solução é o **versionamento semântico (SemVer)**: `MAJOR.MINOR.PATCH` (ex.: 2.4.1).
- **PATCH** (2.4.1 → 2.4.2): correção de bug, sem alterar interface.
- **MINOR** (2.4 → 2.5): funcionalidades novas, compatíveis.
- **MAJOR** (2.x → 3.0): mudanças incompatíveis (*breaking changes*).

Notações como `X>=1.0,<2.0` aceitam qualquer versão 1.x, equilibrando flexibilidade e estabilidade — mas exigem confiar que o pacote segue SemVer corretamente. Por isso, muitas equipes preferem **pinar tudo exatamente** (`pandas==1.5.3`) e atualizar manualmente após testar.

Ferramentas como **pip-tools** (`pip-compile`) separam **dependências diretas** (arquivo `requirements.in`, de alto nível) das **transitivas** (subdependências resolvidas automaticamente e gravadas em `requirements.txt`). Bots como o **Dependabot** automatizam PRs de atualização, sinalizando mudanças MAJOR/MINOR e riscos.

### Ferramentas modernas: Poetry, Pipenv e UV
- **Pipenv**: introduz `Pipfile` e `Pipfile.lock`, unindo gerenciamento de pacotes e ambiente virtual.
- **Poetry**: declara dependências em `pyproject.toml` (ex.: `numpy = "^1.21.0"`), resolve versões compatíveis, instala em ambiente isolado e gera `poetry.lock`; também facilita publicar o projeto como pacote.
- **UV (Astral)**: gerenciador "tudo em um" (2024), escrito em Rust, substitui pip/virtualenv/pipenv/poetry/pyenv; cria ambiente virtual automaticamente, instala pacotes com resolução moderna de dependências e mantém `pyproject.toml` + `uv.lock`. Destaca-se pela **velocidade** (instalações até 10× mais rápidas que pip tradicional), menor consumo de memória e suporte a workspaces e múltiplas versões de Python.

Comparativo resumido do material (Tabela 1):

| Característica | pip + virtualenv | Conda | Poetry | UV (Astral) |
|---|---|---|---|---|
| Ambiente virtual | Manual (venv) | Integrado | Integrado | Integrado (auto) |
| Gerência de pacotes | pip install (sem lock) | conda install | poetry add (gera poetry.lock) | uv add (gera uv.lock) |
| Velocidade | Base (serial) | Moderada | Boa | Muito alta (paralelo em Rust) |
| Lockfile | Manual (pip freeze) | environment.yml | Sim | Sim |
| Dependências não-Python | Não gerencia | Sim | Não (usa pip internamente) | Não (foca só em pacotes Py) |
| Publicar pacote | Via setup.py/twine | N/A | Sim (poetry publish) | Sim (uv publish) |

Em termos de ciência da computação, esses gerenciadores resolvem o **problema de satisfação de dependências** (encontrar versões compatíveis que atendam a todas as restrições), usando resolvedores inspirados em algoritmos de satisfação booleana (SAT) e, no caso do UV, resolução paralela em múltiplas threads.

### Dependências em MLOps
O conceito de dependência se estende a pipelines e modelos de ML: ferramentas como o **MLflow** empacotam um modelo treinado junto com um "snapshot" do ambiente (um `requirements.txt` ou `conda.yaml` dentro do artefato), garantindo que se saiba exatamente com quais bibliotecas aquele modelo foi criado — essencial para reprodutibilidade de experimentos.

### Empacotamento de código reutilizável
Para evitar copiar e colar utilitários entre projetos (ferindo o princípio **DRY — Don't Repeat Yourself**), funções e módulos comuns devem ser transformados em um **pacote instalável** (via pip ou Conda).

Fluxo típico:
1. Organizar o código em `src/nome_do_pacote/` com um `__init__.py`.
2. Descrever metadados (nome, versão, autor, dependências) em `pyproject.toml` (PEP 518/621), usando um *backend* de build como `setuptools`, `hatch` ou o próprio Poetry.
3. Gerar um artefato distribuível (`.whl` ou *source distribution*) com `pip build`/`poetry build`.
4. Publicar no **PyPI** (pacote público) ou em um índice/registry interno.

Vantagens: manutenção centralizada (corrigir um bug em um só lugar e versionar com SemVer, ex.: 1.2.1 → 1.2.2), modularização e design explícito de API pública. Empresas como Spotify e Airbnb mantêm ecossistemas de pacotes internos para reduzir retrabalho.

Uma distinção importante de filosofia de pinagem: **projetos/aplicações** buscam determinismo (pin total, ex. `numpy==1.23.4`), enquanto **bibliotecas/pacotes reutilizáveis** preferem compatibilidade (pin parcial, ex. `numpy>=1.21,<1.25`), para não forçar todos os consumidores a uma versão exata.

## Exercício Hands-On (do material)
A videoaula demonstra, passo a passo, como criar um projeto de ciência de dados com o **UV**: inicializar o projeto, adicionar bibliotecas (pandas, numpy), e verificar que o `uv.lock` gerado garante que qualquer pessoa que rode os mesmos comandos obtenha o mesmo ambiente (mesmas versões exatas das dependências).

## Exemplos de Código

Demonstração de uso do UV para criar um ambiente virtual e gerenciar dependências de um projeto Python:

```bash
# Criando um novo projeto Python e entrando no diretório
$ uv init meu_projeto_ds
Initialized project meu_projeto_ds at ./meu_projeto_ds
$ cd meu_projeto_ds

# Adicionando dependências ao projeto (ex.: pandas e numpy)
$ uv add pandas numpy
Creating virtual environment at: .venv
Resolved 5 packages in 2.1s
Installed 5 packages in 0.5s
+ pandas 1.5.3
+ numpy 1.21.6

# Executando um código dentro do ambiente para verificar
$ uv run python -c "import pandas; print(pandas.__version__)"
1.5.3
```

O UV inicializa o projeto com ambiente virtual automático, instala as bibliotecas com versões compatíveis e gera `pyproject.toml` + `uv.lock` (lockfile com versões exatas), garantindo que outro desenvolvedor obtenha o mesmo ambiente ao rodar os mesmos comandos.

Exemplo hipotético de empacotamento de código reutilizável (estrutura e uso de uma biblioteca interna):

```python
# minha_lib_ml/metrics.py
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o MAPE entre valores reais e previstos."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

```toml
# pyproject.toml
[project]
name = "minha-lib-ml"
version = "0.1.0"
dependencies = ["numpy>=1.21"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python
# uso em outro projeto, após `pip install minha_lib_ml==0.1.0`
from minha_lib_ml.metrics import mean_absolute_percentage_error

erro = mean_absolute_percentage_error([100, 150, 200], [110, 140, 190])
print(f"MAPE: {erro:.2f}%")
```

## Cases e Tendências de Mercado
- **Tendências em Ciência de Dados para 2025**: ascensão do Data Engineering, integração de IA generativa em plataformas de BI, crescimento do Edge Computing para análises em tempo real e descentralização via Data Mesh — tendências que exigem ambientes complexos e dependências bem gerenciadas (artigo de Gabriel Beran Ribeiro).
- **Desafios de Dados e Analytics segundo o Gartner**: nove desafios para 2025, incluindo produtos de dados altamente consumíveis, gerenciamento de metadados e data fabric multimodal, reforçando a importância de produtos de dados reutilizáveis e ambientes reprodutíveis.
- **Estudo bibliométrico sobre gestão de dados**: crescimento acelerado nas publicações sobre o tema, destacando autores e periódicos influentes em tecnologia e bancos de dados.
- **Curso de Fundamentos da Ciência de Dados (FM2S)**: vídeo introdutório sobre transformar dados em decisões estratégicas, com ênfase em ambientes organizados.
- **Podcast "Pizza de Dados"**: primeiro podcast brasileiro sobre ciência de dados, com episódios sobre arquitetura de dados, data lakes, governança e qualidade de dados.
- **Lista de podcasts da DataCamp**: compilação dos melhores podcasts sobre IA, machine learning, governança e tendências de mercado.

## Checklist de Estudo
- [ ] Sei explicar o "inferno das dependências" e por que ambientes virtuais o resolvem
- [ ] Sei criar e ativar um ambiente virtual com venv e sei quando usar Conda em vez disso
- [ ] Sei diferenciar requirements.txt sem versão de um requirements.txt com pinagem (`==`) e seus efeitos na reprodutibilidade
- [ ] Sei explicar o versionamento semântico (MAJOR.MINOR.PATCH) e interpretar notações como `>=1.0,<2.0`
- [ ] Sei diferenciar dependências diretas de transitivas e o papel de ferramentas como pip-tools
- [ ] Conheço as diferenças práticas entre Pipenv, Poetry e UV, e sei quando cada um se encaixa melhor
- [ ] Sei usar o UV para iniciar um projeto, adicionar dependências e gerar um lockfile
- [ ] Entendo como o MLflow estende o conceito de dependências para modelos de ML (snapshot de ambiente)
- [ ] Sei os passos básicos para empacotar código Python reutilizável (pyproject.toml, build, publicação)
- [ ] Sei explicar a diferença de filosofia de pinagem entre aplicações (pin total) e bibliotecas (pin parcial)

## Palavras-chave
Gerenciamento de dependências. Reprodutibilidade em ciência de dados. Ambientes virtuais e versionamento.

## Referências
- IBM DATA SCIENCE COMMUNITY. *Dependency Management – Best Practices.* 2019. Disponível em: <https://github.com/IBM/data-science-best-practices/blob/main/dependency_management.md>. Acesso em: 15 dez. 2025.
- KIILI, J. *What every data scientist should know about Python dependencies.* 2022. Disponível em: <https://valohai.com/blog/dependency-management-for-data-science/>. Acesso em: 14 nov. 2025.
- LARANJA, Emerson. *Versionamento Semântico (SemVer): uma breve introdução.* 2022. Disponível em: <https://www.alura.com.br/artigos/versionamento-semantico-breve-introducao>. Acesso em: 15 dez. 2025.
- MEHER, S. *A Beginner's Guide to Packaging Python Code.* 2025. Disponível em: <https://shibu778.github.io/posts/packaging_python/>. Acesso em: 15 dez. 2025.
- TREADWAY, A. *Software Engineering for Data Scientists.* Manning (MEAP), 2024. Disponível em: <https://www.manning.com/books/software-engineering-for-data-scientists>. Acesso em: 15 dez. 2025.
- ULILI, S. *A Deep Dive into UV: The Fast Python Package Manager.* 2025. Disponível em: <https://betterstack.com/community/guides/scaling-python/uv-explained/>. Acesso em: 15 dez. 2025.
