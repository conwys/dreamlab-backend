import os
import threading
import logging

from dotenv import load_dotenv
from flask import Flask, current_app
from flask_cors import CORS

from config import Config, DevelopmentConfig, TestingConfig, ProductionConfig
from utils.session_helpers import cleanup_expired_sessions


def create_app(config_class=Config):
    """
    Creates and configures the Flask application instance
    """
    load_dotenv()  # Load .env variables when app is created (e.g., for local dev)

    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, origins=["http://localhost:4200"]) # TODO

    logging.basicConfig(level=logging.INFO)

    from api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app


if __name__ == "__main__":
    app_instance = create_app(DevelopmentConfig)

    if app_instance.config.get("SESSIONS_DIR"):
        os.makedirs(app_instance.config["SESSIONS_DIR"], exist_ok=True)

        cleanup_thread = threading.Thread(
            target=cleanup_expired_sessions,
            args=(
                app_instance.config["SESSIONS_DIR"],
                app_instance.config["SESSION_EXPIRE_REMOVE_SECONDS"],
                app_instance.config["SESSION_CLEANUP_INTERVAL_SECONDS"]
            ),
            daemon=True
        )
        cleanup_thread.start()
        app_instance.logger.info("Started background session cleanup thread")
    else:
        app_instance.logger.warning("SESSIONS_DIR is not configured. Session cleanup will not run")

    app_instance.run(debug=app_instance.config["DEBUG"], host="0.0.0.0", port=5000)
