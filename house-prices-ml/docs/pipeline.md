# Decisões de arquitetura do pipeline de ML

Este documento resume as decisões de projeto que não são óbvias só de ler o
código — o "porquê" por trás de cada escolha. Os detalhes específicos de cada
regra de limpeza/feature ficam nos docstrings dos módulos correspondentes; aqui
o foco é a arquitetura geral.

## 1. Por que separar "limpeza" (`data/clean.py`) de "features" (`features.py`)?

São dois tipos de transformação bem diferentes:

- **Limpeza** (`clean_data`): roda **uma única vez**, sobre o dataset inteiro,
  fora de qualquer modelo. Deduplicar linhas, remover outliers, corrigir tipos —
  nada disso depende de "o que o modelo vai fazer com os dados depois". O
  resultado é salvo em `data/processed/`.
- **Engenharia de features** (`add_engineered_features`, `build_pipeline`): roda
  **toda vez que alguém pede uma predição** — seja durante o treino (milhares de
  vezes, uma por linha) ou na API, com uma única linha vinda de uma requisição
  HTTP. Por isso ela vive dentro de um `sklearn.Pipeline`, não como uma função
  solta chamada manualmente antes de salvar um CSV.

Misturar as duas seria um erro sutil: se a "feature engineering" rodasse só uma
vez sobre o CSV de treino, a API precisaria reimplementar exatamente a mesma
lógica para uma predição nova — e qualquer divergência entre as duas
implementações (um parâmetro esquecido, uma ordem de colunas diferente) é o
chamado **training-serving skew**: o modelo se comporta diferente em produção do
que se comportou no treino, silenciosamente. Ter um único `Pipeline`
serializado (`models/model.joblib`) que já inclui a engenharia de features
elimina essa classe inteira de bug por construção.

## 2. Por que evitar vazamento de dados (data leakage) na deduplicação?

O dataset registra **vendas**, não imóveis únicos — 177 imóveis aparecem mais de
uma vez (foram revendidos no período coberto). Se não deduplicássemos por `id`,
o `train_test_split` poderia colocar duas vendas do MESMO imóvel em treino e
teste ao mesmo tempo. O modelo "veria" esse imóvel durante o treino (ainda que
com uma data e preço um pouco diferentes) e teria uma vantagem artificial ao
prevê-lo no teste — inflando a métrica de avaliação de um jeito que não se
sustenta em produção, onde todo imóvel avaliado é, por definição, inédito para o
modelo. Por isso mantemos só a venda mais recente de cada `id`.

## 3. Por que `condition` é `OrdinalEncoder` mas `zipcode` é `OneHotEncoder`?

Ambas são colunas categóricas (texto ou código), mas com naturezas diferentes:

- `condition` tem uma ordem real: `Poor < Fair < Average < Good < Very Good`.
  Codificá-la como números ordenados (`OrdinalEncoder` com a ordem explícita)
  preserva essa relação — o modelo pode aprender "quanto melhor a condição,
  maior o preço" como uma tendência, em vez de tratar cada categoria como
  totalmente desconectada das outras.
- `zipcode` não tem ordem: `98004` não é "maior" que `98001` no sentido de
  preço — são só rótulos de região. Codificar como número cru enganaria o
  modelo (sugeriria uma relação de ordem/distância que não existe). Por isso
  vai para `OneHotEncoder`, que cria uma coluna binária por zipcode (70 no
  total — cardinalidade alta, mas administrável).

## 4. Por que `RandomForestRegressor` em vez de regressão linear?

O notebook `02_modeling.ipynb` compara as duas de forma objetiva: a regressão
linear atinge R² ≈ 0.80 no conjunto de teste, o Random Forest chega a R² ≈ 0.88,
com MAE (erro médio absoluto) cerca de 30% menor. A hipótese razoável é que a
relação entre localização/qualidade do imóvel e preço não é puramente linear —
tem interações (ex.: o efeito de `grade` no preço provavelmente muda dependendo
da região) que uma árvore captura sem exigir que a gente as modele manualmente
como termos de interação explícitos.

## 5. Por que uma amostra separada em `data/sample/` para os testes?

CI (GitHub Actions) não tem credenciais do Kaggle configuradas, e não deveria
precisar — testes automatizados não podem depender de um serviço externo estar
disponível e autenticado para passar. `data/sample/house_prices_sample.csv` é
uma amostra de 300 linhas do dataset já limpo, versionada no Git, grande o
suficiente para treinar um modelo de verdade (ainda que pequeno) em segundos, e
pequena o suficiente para não pesar no repositório.
