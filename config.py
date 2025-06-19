import os


class Config:
    """Base configuration class"""
    FLASK_ENV = os.environ.get("FLASK_ENV", "development")
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:4200")
    LOG_LEVEL = "INFO"

    # Session management
    SESSIONS_DIR = os.path.join(os.getcwd(), "sessions")
    SESSION_EXPIRE_REMOVE_SECONDS = 3600
    SESSION_CLEANUP_INTERVAL_SECONDS = 600

    # Hunyuan service configuration
    HUNYUAN_SPACE_ID = os.environ.get("HUNYUAN_SPACE_ID")
    HUNYUAN_API_NAME = os.environ.get("HUNYUAN_API_NAME")
    APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")

    # Allowed views for images
    ALLOWED_VIEWS = ["front", "back", "left", "right"]


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = "development"
    LOG_LEVEL = "DEBUG"


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    ENV = "testing"
    LOG_LEVEL = "DEBUG"

    # Session management
    SESSIONS_DIR = os.path.join(os.getcwd(), "test_sessions")
    # Ensure background cleanup is not started
    SESSION_CLEANUP_INTERVAL_SECONDS = 0
    SESSION_EXPIRE_REMOVE_SECONDS = 0


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = "production"
    LOG_LEVEL = "ERROR"
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS")
