import os


class Config:
    """Base configuration class"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "your_development_secret_key")

    # Session management
    SESSIONS_DIR = os.environ.get("SESSIONS_DIR", "sessions")
    SESSION_EXPIRE_REMOVE_SECONDS = int(os.environ.get('SESSION_EXPIRE_REMOVE_TIME', 3600))
    SESSION_CLEANUP_INTERVAL_SECONDS = int(os.environ.get('SESSION_EXPIRE_SLEEP_TIME', 300))

    # Hunyuan service configuration
    HUNYUAN_SPACE_ID = os.environ.get("HUNYUAN_SPACE_ID")
    HUNYUAN_API_NAME = os.environ.get("HUNYUAN_API_NAME")
    APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")

    # Allowed views for images
    ALLOWED_VIEWS = ["front", "back", "left", "right"]


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = True


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SESSIONS_DIR = "test_sessions"
    # Ensure background cleanup is not started
    SESSION_CLEANUP_INTERVAL_SECONDS = 0
    SESSION_EXPIRE_REMOVE_SECONDS = 0

# TODO
class ProductionConfig(Config):
    """Production configuration"""
    pass
    # DEBUG = False
    # TESTING = False
    # Further specific production settings
