---
name: ML_lifecicle_1
description: Estrutura um projeto completo de ciclo de vida de ML a partir de um dataset tabular — ciência de dados (limpeza/EDA), pipeline de treino em sklearn, API Flask de serving, testes automatizados e CI/CD no GitHub Actions. Use quando o usuário pedir para montar/estruturar um projeto de ML do zero, "ciclo de vida de ML", "pipeline de dados até API", ou mencionar explicitamente esta skill (ML_lifecicle_1).
user-invocable: true
---

# ML_lifecicle_1 — Ciclo de vida de ML completo (dados → modelo → API → CI)

Playbook para montar, a partir de um dataset tabular, um projeto de estudo de
ML de ponta a ponta: limpeza de dados, EDA, pipeline de features/treino,
API Flask, testes automatizados e CI no GitHub Actions. É a mesma estrutura
usada em `house-prices-ml/` neste repositório (Fase1/aula08 do curso de
MLOps) — trate aquele diretório como referência viva de implementação, mas
esta skill deve funcionar para **qualquer** dataset tabular novo, não só
aquele.

Arguments passed: `$ARGUMENTS` (pode conter caminho/URL do dataset, nome do
projeto, coluna alvo — parseie o que vier; o que faltar, pergunte).

**Princípio geral: este projeto é para o usuário aprender de verdade.**
Explique o "porquê" de cada decisão não-óbvia enquanto constrói (em
comentários no código E em mensagens curtas para o usuário nos pontos-chave),
não só execute em silêncio. Vá construindo e mostrando resultados reais
(rodar comandos, mostrar métricas, testar a API) em vez de só descrever o que
"seria" feito.

## 0. Descobrir o dataset e alinhar escopo antes de codar

1. Se o usuário já apontou um notebook/arquivo/URL de dataset, leia-o e rode
   uma inspeção real dos dados (shape, dtypes, nulos, describe, valores
   únicos de categóricas, possíveis duplicatas/outliers) antes de desenhar a
   limpeza — não assuma o schema, confirme rodando código.
2. Se faltar informação essencial e não for óbvia pelo contexto, pergunte
   objetivamente (via AskUserQuestion se a decisão for arquitetural, ex.: "os
   testes/CI devem rodar sem depender de baixar o dataset de novo?" —
   normalmente a resposta é sim, ver seção 5). Não pergunte o que já dá pra
   inferir do dataset ou do pedido do usuário.
3. Decida e confirme silenciosamente (ou pergunte se ambíguo):
   - Nome do projeto/diretório.
   - Coluna alvo e tipo de problema (regressão vs. classificação) — isso
     muda a escolha de modelo/métricas na seção 3.
   - Onde este projeto vive: subpasta dedicada dentro do repo atual (padrão,
     se o repo já tem outro conteúdo não relacionado) ou raiz do repo (se o
     repo é dedicado a este projeto).
4. Verifique se já existe um repositório Git e onde fica sua raiz —
   `.github/workflows/` **precisa** estar na raiz do repositório Git para o
   GitHub Actions encontrar o workflow, mesmo que o projeto em si viva numa
   subpasta (`<projeto>/`). Não cometa o erro de colocar
   `<projeto>/.github/workflows/`.

## 1. Estrutura de diretórios

Dentro da pasta do projeto (`<nome-projeto>/`):

```
<nome-projeto>/
├── data/
│   ├── raw/            # dataset bruto (gitignored)
│   ├── processed/       # dataset limpo (gitignored)
│   └── sample/           # amostra pequena, VERSIONADA no Git (ver secao 5)
├── docs/
│   ├── pipeline.md       # decisoes de arquitetura, o "porque"
│   └── api.md            # contrato da API + exemplos de curl
├── models/               # modelo treinado + metricas (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/<pacote>/
│   ├── config.py          # caminhos/constantes centralizados
│   ├── data/
│   │   ├── download.py    # ou instrucoes se o dataset vier de arquivo local
│   │   └── clean.py
│   ├── features.py        # engenharia de features + sklearn Pipeline
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── app.py
│       └── schemas.py
├── tests/
├── pyproject.toml
└── README.md
```

E na raiz do repositório Git: `.gitignore` e `.github/workflows/ci.yml`.

## 2. Ciência de dados: limpeza separada de features

Duas camadas distintas, não misturar:

- **Limpeza estrutural** (`data/clean.py`): roda uma vez sobre o dataset
  inteiro. Deduplicação, remoção de outliers/linhas inválidas, correção de
  tipos, parsing de datas. Documente cada regra com o *porquê* (docstring do
  módulo), não só o *o quê* — inclua números reais encontrados na inspeção
  (ex.: "177 ids duplicados", "1 linha com valor impossível").
  - **Sempre cheque vazamento de dados (data leakage)**: se o dataset tem uma
    chave que pode se repetir (mesmo item vendido/medido mais de uma vez),
    deduplique mantendo o registro mais recente ANTES do `train_test_split`,
    senão o mesmo item pode cair em treino e teste ao mesmo tempo e inflar a
    métrica de forma enganosa.
- **Engenharia de features** (`features.py`): roda a cada predição (treino
  E serving), por isso vive **dentro de um único `sklearn.Pipeline`**
  (`FunctionTransformer` para features derivadas + `ColumnTransformer` para
  encoding/escala + o estimator). Isso evita training-serving skew: a API
  nunca reimplementa a lógica de features, só chama `pipeline.predict()`
  no artefato serializado (`joblib`).
  - Para categóricas: use `OrdinalEncoder` com `categories=[...]` explícito
    quando a categoria tem ordem real (ex.: condição Ruim < Regular < Boa);
    use `OneHotEncoder(handle_unknown="ignore")` quando não tem ordem (ex.:
    região/código postal).

Produza `notebooks/01_eda.ipynb` com análise real (distribuição do alvo,
outliers, correlações, relação de features-chave com o alvo, e se houver
lat/long ou algo geográfico, um scatter colorido pelo alvo) — **execute o
notebook de verdade** (ver seção 6) para que as saídas fiquem salvas; não
entregue um notebook com células vazias.

## 3. Modelo e pipeline de treino

`train.py`: carrega dados processados, faz `train_test_split` com
`random_state` fixo, monta o pipeline (`build_pipeline(model)` de
`features.py`), treina, avalia e salva `models/model.joblib` +
`models/metrics.json`. Parametrizável via `argparse` (ao menos hiperparâmetros
principais e `--data` para apontar outro CSV — isso é o que permite treinar
tanto no dataset completo quanto na amostra pequena dos testes).

- Escolha de modelo: comece com um baseline "ingênuo" (`DummyRegressor`/
  `DummyClassifier`) como piso de comparação, depois um modelo linear
  simples, depois um ensemble baseado em árvore (RandomForest é um bom
  default "sem drama" para dados tabulares mistos). Compare os três em
  `notebooks/02_modeling.ipynb` com números reais, não só afirme qual é
  melhor.
- Métricas: regressão → MAE, RMSE, R². Classificação → accuracy, precision,
  recall, F1 (e Recall priorizado se falso-negativo for o erro caro no
  domínio, ex.: saúde/fraude).
- Depois de treinar, extraia e mostre `feature_importances_` (ou
  coeficientes) no notebook de modelagem — conecta de volta com o que a EDA
  sugeriu.
- Ao arredondar uma tabela de métricas para exibição, cuidado com métricas
  em escalas diferentes (ex.: `.round(0)` num R² entre 0-1 vira tudo `1.0` e
  esconde a diferença real — arredonde cada coluna com a casa decimal
  apropriada).

`predict.py`: único ponto de carregamento do modelo (`joblib.load`, cacheado)
e de predição — API e testes devem reusar isso, não reimplementar.

## 4. API Flask

`api/app.py` com **application factory** (`create_app(model_path=...)`), não
uma instância global rígida — isso é o que permite os testes criarem uma app
apontando para um modelo de teste sem tocar no modelo de produção.

- `GET /health`: retorna se o modelo está carregado (200 se sim, 503 se não —
  não deixe o serviço mentir que está saudável sem modelo).
- `POST /predict`: valida o payload (`api/schemas.py` — campos obrigatórios,
  tipos, categorias válidas) ANTES de chamar o pipeline; erro de validação →
  400 com mensagem clara do que está errado, nunca deixe a exceção crua do
  sklearn vazar pro cliente. Se o modelo não está carregado → 503, não 500.
- Trate a ausência do modelo em `create_app` como aviso (log), não como
  crash — a API deve subir mesmo sem modelo treinado ainda, só recusar
  `/predict` até que exista.

Documente o contrato em `docs/api.md` (tabela de campos + exemplos `curl` de
sucesso e de cada tipo de erro).

## 5. Testes automatizados — nunca dependa de rede/credenciais em CI

Isso é o ponto que mais gente esquece e quebra o CI depois:

1. Depois de rodar a limpeza no dataset completo, **gere uma amostra pequena
   (algumas centenas de linhas) já limpa e versione-a em `data/sample/`**.
   É ela que testes e o job de treino do CI vão usar — nunca o dataset
   completo nem uma chamada de download (Kaggle, S3, API externa) durante os
   testes.
2. Escreva testes cobrindo pelo menos:
   - **Dados**: regras de limpeza (dedupe, outliers removidos, sem nulos) —
     use um DataFrame sintético mínimo construído no teste para casos de
     borda, não só a amostra real.
   - **Features**: cada feature derivada calculada corretamente; a função de
     engenharia não muta o input; o pipeline treina e prediz na amostra sem
     erro.
   - **Treino**: roda na amostra pequena (rápido), valida que os artefatos
     são salvos e que a métrica supera um piso trivial (ex. R² > 0).
   - **API**: test client do Flask — sucesso, campos faltando, categoria
     inválida, corpo não-JSON, modelo ausente (503). Treine o modelo de
     teste uma vez por sessão de testes (fixture `scope="session"`), não a
     cada teste.
3. Rode `pytest` e o linter (`ruff check`, se disponível no ambiente; instale
   como dependência de dev se não estiver) localmente e confirme que passam
   **antes** de escrever o workflow de CI — não escreva o workflow "no
   escuro".

## 6. Ambiente, execução real e notebooks executados

- Crie um venv dedicado ao projeto (`python -m venv .venv` dentro da pasta do
  projeto) e instale com `pip install -e ".[dev]"` a partir de um
  `pyproject.toml` com `[tool.setuptools.packages.find] where = ["src"]` e
  `[tool.pytest.ini_options] pythonpath = ["src"]`.
- Rode de verdade, nesta ordem, e mostre os resultados reais ao usuário (não
  descreva o que "deveria" acontecer): download/leitura do dataset →
  limpeza → geração da amostra em `data/sample/` → treino completo → teste
  manual da API (`app.test_client()` ou `curl`) → `pytest` → `ruff check`.
- Para os notebooks, registre um kernel Jupyter apontando pro venv do
  projeto (`python -m ipykernel install --user --name <projeto> --display-name
  <projeto>`) e execute de verdade com
  `python -m jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.kernel_name=<projeto> notebooks/0X_*.ipynb` — entregue
  os notebooks já com saídas/gráficos reais, não células vazias que o usuário
  teria que rodar antes de conseguir ler.
- Matplotlib/seaborn/jupyter só entram como optional-dependency
  (`[project.optional-dependencies] notebooks = [...]`), não como dependência
  obrigatória do pacote.

## 7. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` **na raiz do repositório Git** (ver seção 0.4).
Estrutura mínima, 3 jobs sequenciais (`needs:`):

1. **lint** — `ruff check src tests`.
2. **test** — `pytest` (com `--cov` se `pytest-cov` estiver instalado).
3. **train-smoke-test** — roda `python -m <pacote>.train --data
   data/sample/<amostra>.csv` de verdade no runner e confere que
   `models/model.joblib` e `models/metrics.json` foram gerados. Isso valida o
   pipeline ponta a ponta em CI, não só unidades isoladas.

Se o projeto vive numa subpasta do repo, use `defaults.run.working-directory`
e um filtro `paths:` no `on.push`/`on.pull_request` apontando para essa
subpasta, para o workflow não rodar à toa em mudanças não relacionadas.

## 8. `.gitignore` e o que versionar

Nunca versionar: `.venv/`, dataset bruto/processado completo
(`data/raw/*`, `data/processed/*`, com exceção de `.gitkeep`), artefatos de
modelo (`models/*`, exceto `.gitkeep`), caches de ferramentas
(`.pytest_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`), `.env`.

Sempre versionar: código-fonte, `data/sample/` (a amostra pequena da seção
5), notebooks executados, `pyproject.toml`, `.gitignore`, `README.md`,
`docs/`, `.github/workflows/ci.yml`.

## 9. Documentação

- `docs/pipeline.md`: decisões de arquitetura com o *porquê* (não repita o
  que já está no docstring do código — sintetize a intenção geral: por que
  separar limpeza de features, como o projeto evita vazamento de dados, por
  que esse encoding e não outro, por que esse modelo).
- `docs/api.md`: contrato completo da API + exemplos `curl` de sucesso e de
  erro.
- `README.md` do projeto: o que é, por que existe (problema de negócio em 2-3
  frases), estrutura de pastas, tabela "etapa do ciclo de vida → onde mora
  o código", instruções de setup, comando por comando de cada etapa
  (download → limpeza → treino → subir API → testar → rodar notebooks),
  como rodar testes/CI localmente, e uma seção curta de "próximos passos
  possíveis" (MLflow, Docker, monitoramento de drift, governança/fairness se
  fizer sentido pro domínio) deixando claro que é escopo futuro, não
  implementado.

## 10. Commits

Não invente múltiplos autores/colaboradores fictícios para simular "trabalho
em grupo" — isso é enganoso. Em vez disso, faça commits atômicos e
significativos, um por etapa do ciclo de vida (estrutura inicial → dados →
features/treino → API → testes → CI → docs/notebooks), cada um explicando o
"porquê" no corpo da mensagem, não só o "o quê". Confirme com o usuário antes
de dar `git push` (ação visível/compartilhada) — commits locais não precisam
de confirmação extra além da já dada ao acionar esta skill, mas push sim.

## Checklist final antes de encerrar

- [ ] `pytest` passa localmente, sem tocar rede/credenciais.
- [ ] `ruff check` (ou linter equivalente) limpo.
- [ ] Treino ponta-a-ponta rodado de verdade (dataset completo, se
      disponível) com métricas reais reportadas ao usuário.
- [ ] API testada de verdade (`/health` e `/predict`, sucesso e pelo menos um
      caso de erro).
- [ ] Notebooks executados com saídas reais salvas.
- [ ] `.github/workflows/ci.yml` está na raiz do repositório Git.
- [ ] `.gitignore` cobre venv/dados brutos/processados/modelos, mas
      `data/sample/` está versionada.
- [ ] README explica setup e cada comando de execução.
- [ ] Resumo final ao usuário: o que foi construído, métricas reais
      obtidas, e o que só ele pode decidir (push, revisão).
