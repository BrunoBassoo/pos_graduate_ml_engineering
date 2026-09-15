# API — contrato e exemplos

Base URL local: `http://127.0.0.1:5000` (subida com
`flask --app house_prices.api.app run --debug`).

## `GET /health`

Healthcheck — usado por orquestradores/monitoramento para saber se o serviço
está no ar E se o modelo foi carregado com sucesso.

```bash
curl http://127.0.0.1:5000/health
```

Resposta (200, modelo carregado):

```json
{"status": "ok", "model_loaded": true, "version": "0.1.0"}
```

Resposta (503, modelo ausente — rode `python -m house_prices.train` antes):

```json
{"status": "model_not_loaded", "model_loaded": false, "version": "0.1.0"}
```

## `POST /predict`

Recebe as características de um imóvel e retorna o preço previsto.

### Campos obrigatórios

| Campo | Tipo | Descrição |
|---|---|---|
| `bedrooms` | int | Número de quartos |
| `bathrooms` | float | Número de banheiros (aceita frações, ex.: 1.75) |
| `sqft_living` | número | Área útil interna (pés²) |
| `sqft_lot` | número | Área do terreno (pés²) |
| `floors` | float | Número de andares |
| `waterfront` | string | `"N"` ou `"Y"` — vista para água |
| `view` | int | Nota de qualidade da vista (0–4) |
| `condition` | string | Um de: `Poor`, `Fair`, `Average`, `Good`, `Very Good` |
| `grade` | int | Nota de construção/design (1–13) |
| `sqft_above` | número | Área útil acima do nível do solo |
| `sqft_basement` | número | Área do porão (0 se não houver) |
| `yr_built` | int | Ano de construção |
| `yr_renovated` | int | Ano da última reforma (0 se nunca reformado) |
| `zipcode` | int | CEP (King County, WA) |
| `lat` | float | Latitude |
| `long` | float | Longitude |
| `sqft_living15` | número | Área útil média dos 15 vizinhos mais próximos |
| `sqft_lot15` | número | Área de terreno média dos 15 vizinhos mais próximos |
| `sale_year` | int | Ano da venda (usado para calcular a idade do imóvel) |

### Exemplo de requisição

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000,
    "floors": 1.0, "waterfront": "N", "view": 0, "condition": "Average",
    "grade": 7, "sqft_above": 1800, "sqft_basement": 0, "yr_built": 1990,
    "yr_renovated": 0, "zipcode": 98115, "lat": 47.68, "long": -122.29,
    "sqft_living15": 1800, "sqft_lot15": 5000, "sale_year": 2015
  }'
```

Resposta (200):

```json
{"predicted_price": 636235.95, "model_version": "0.1.0"}
```

### Erros

- **400** — payload não é JSON válido, tem campo faltando, campo do tipo
  errado, ou `waterfront`/`condition` fora dos valores aceitos. Corpo da
  resposta: `{"error": "<mensagem explicando o problema>"}`.
- **503** — modelo ainda não foi treinado/carregado (`models/model.joblib` não
  existe). Rode `python -m house_prices.train` primeiro.

```bash
# exemplo de erro 400 (campo faltando)
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"bedrooms": 3}'
# {"error": "Campos obrigatorios ausentes: ['bathrooms', 'sqft_living', ...]"}
```
