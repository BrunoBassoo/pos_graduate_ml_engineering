"""API Flask que expoe o modelo de previsao de precos.

Endpoints:
    GET  /health   -> healthcheck (usado por orquestradores/monitoramento)
    POST /predict  -> recebe as features de um imovel, retorna o preco previsto

Uso local:
    flask --app house_prices.api.app run --debug
    # ou
    python -m house_prices.api.app
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from house_prices import __version__
from house_prices.api.schemas import ValidationError, validate_payload
from house_prices.config import MODEL_PATH
from house_prices.predict import load_model, predict_price

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_app(model_path=MODEL_PATH) -> Flask:
    """Application factory: permite criar instancias isoladas do app nos testes,
    cada uma apontando para o modelo que quiser (ex.: um modelo de teste)."""
    app = Flask(__name__)

    try:
        app.config["MODEL"] = load_model(model_path)
    except FileNotFoundError:
        logger.warning(
            "Nenhum modelo encontrado em %s. Rode 'python -m house_prices.train' primeiro. "
            "A API vai subir, mas /predict retornara 503 ate que o modelo exista.",
            model_path,
        )
        app.config["MODEL"] = None

    @app.get("/health")
    def health():
        model_loaded = app.config["MODEL"] is not None
        status_code = 200 if model_loaded else 503
        return jsonify(
            {
                "status": "ok" if model_loaded else "model_not_loaded",
                "model_loaded": model_loaded,
                "version": __version__,
            }
        ), status_code

    @app.post("/predict")
    def predict():
        if app.config["MODEL"] is None:
            error = "Modelo nao carregado. Treine o modelo antes de usar /predict."
            return jsonify({"error": error}), 503

        payload = request.get_json(silent=True)
        try:
            validate_payload(payload)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        predicted_price = predict_price(payload, model=app.config["MODEL"])
        return jsonify({"predicted_price": predicted_price, "model_version": __version__}), 200

    return app


# instancia usada por `flask run` / servidores WSGI (gunicorn, waitress, etc.)
app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
