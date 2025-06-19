import logging
import os
import threading

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig
from utils.session_helpers import cleanup_expired_sessions


def create_app():
    """
    Creates and configures the Flask application instance
    """
    load_dotenv()  # Load .env variables when app is created (e.g., for local dev)

    env = os.environ.get("FLASK_ENV")

    if env == "production":
        config_class = ProductionConfig
    elif env == "testing":
        config_class = TestingConfig
    else:
        config_class = DevelopmentConfig

    app = Flask(__name__)
    app.config.from_object(config_class)

    cors_origins = app.config.get("CORS_ORIGINS")

    CORS(app, origins=cors_origins)  # is support credentials needed

    logging.basicConfig(
        level=getattr(logging, app.config.get("LOG_LEVEL", "INFO").upper())
    )
    app.logger.info(
        f"App running in {app.config['ENV']} environment with log level {app.config['LOG_LEVEL']}"
    )

    from api import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app_instance = create_app()

    if app_instance.config.get("SESSIONS_DIR"):
        os.makedirs(app_instance.config["SESSIONS_DIR"], exist_ok=True)
        app_instance.logger.info(
            f"Session directory: {app_instance.config['SESSIONS_DIR']}"
        )

        cleanup_thread = threading.Thread(
            target=cleanup_expired_sessions,
            args=(
                app_instance,
                app_instance.config["SESSIONS_DIR"],
                app_instance.config["SESSION_EXPIRE_REMOVE_SECONDS"],
                app_instance.config["SESSION_CLEANUP_INTERVAL_SECONDS"],
            ),
            daemon=True,
        )
        cleanup_thread.start()
        app_instance.logger.info("Started background session cleanup thread")
    else:
        app_instance.logger.warning(
            "SESSIONS_DIR is not configured. Session cleanup will not run"
        )

    port = int(os.environ.get("PORT", 5000))
    app_instance.run(
        debug=app_instance.config["DEBUG"], host="0.0.0.0", port=port, load_dotenv=False
    )
