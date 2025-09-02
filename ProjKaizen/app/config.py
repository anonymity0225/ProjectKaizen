# YAML loader for application configuration 
"""
Kaizen Application Configuration Module

Production-ready configuration management using Pydantic BaseSettings
with support for environment variables, .env files, and multiple deployment profiles.
"""

import os
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseSettings, Field, validator


class Environment(str, Enum):
    """Supported environment profiles"""
    DEVELOPMENT = "dev"
    TESTING = "test"
    PRODUCTION = "prod"


class MLSettings(BaseSettings):
    """Machine Learning specific configuration"""
    model_path: str = Field(default="./models", description="Path to ML models directory")
    batch_size: int = Field(default=32, description="Default batch size for ML operations")
    max_model_memory: int = Field(default=2048, description="Max memory per model in MB")
    inference_timeout: int = Field(default=30, description="Inference timeout in seconds")
    cache_predictions: bool = Field(default=True, description="Enable prediction caching")

    class Config:
        env_prefix = "ML_"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: Optional[str] = Field(default=None, description="Database URL")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="kaizen", description="Database name")
    user: str = Field(default="kaizen_user", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    @validator('url', pre=True, always=True)
    def assemble_db_url(cls, v, values):
        if isinstance(v, str) and v:
            return v
        return f"postgresql://{values.get('user')}:{values.get('password')}@{values.get('host')}:{values.get('port')}/{values.get('name')}"

    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and sessions"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=20, description="Max Redis connections")
    
    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security and authentication settings"""
    secret_key: str = Field(..., description="Application secret key")
    jwt_secret: str = Field(..., description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, description="JWT expiration in seconds")
    bcrypt_rounds: int = Field(default=12, description="Bcrypt hashing rounds")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    
    class Config:
        env_prefix = "SECURITY_"


class AppSettings(BaseSettings):
    """Main application configuration"""
    
    # Application basics
    app_name: str = Field(default="Kaizen", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    description: str = Field(default="AI-Powered Development Assistant", description="App description")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    # CORS settings
    allowed_origins: List[str] = Field(default_factory=list, description="Allowed CORS origins")
    allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="Allowed HTTP methods")
    allowed_headers: List[str] = Field(default=["*"], description="Allowed headers")
    
    # API configuration
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    docs_url: Optional[str] = Field(default="/docs", description="OpenAPI docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", description="OpenAPI JSON URL")
    
    # File handling
    upload_dir: str = Field(default="./uploads", description="File upload directory")
    max_file_size: int = Field(default=10485760, description="Max file size in bytes (10MB)")
    allowed_file_types: List[str] = Field(default=["txt", "pdf", "doc", "docx", "py", "js", "html", "css"], description="Allowed file extensions")
    
    # Nested settings
    ml: MLSettings = Field(default_factory=MLSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    @validator('allowed_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('debug', pre=True, always=True)
    def set_debug_mode(cls, v, values):
        """Auto-set debug mode based on environment"""
        env = values.get('environment', Environment.DEVELOPMENT)
        if env == Environment.DEVELOPMENT:
            return True
        return v if v is not None else False
    
    @validator('docs_url', 'redoc_url', 'openapi_url', pre=True, always=True)
    def disable_docs_in_prod(cls, v, values):
        """Disable API docs in production"""
        env = values.get('environment', Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v is not None:
            return None
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True


def get_logging_config(environment: Environment = Environment.DEVELOPMENT) -> Dict[str, Any]:
    """
    Get logging configuration based on environment
    
    Args:
        environment: Current environment setting
        
    Returns:
        Dictionary configuration for Python logging
    """
    
    base_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "[%(asctime)s] %(levelname)s in %(name)s [%(pathname)s:%(lineno)d]: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "logs/kaizen.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": "logs/kaizen_error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file", "error_file"]
        },
        "loggers": {
            "kaizen": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Environment-specific adjustments
    if environment == Environment.DEVELOPMENT:
        base_config["root"]["level"] = "DEBUG"
        base_config["loggers"]["kaizen"]["level"] = "DEBUG"
        base_config["handlers"]["console"]["level"] = "DEBUG"
        
    elif environment == Environment.TESTING:
        base_config["handlers"]["console"]["level"] = "WARNING"
        base_config["handlers"]["file"]["filename"] = "logs/kaizen_test.log"
        
    elif environment == Environment.PRODUCTION:
        base_config["formatters"]["default"] = base_config["formatters"]["json"]
        base_config["handlers"]["console"]["formatter"] = "json"
        base_config["handlers"]["file"]["formatter"] = "json"
        
        # Add structured logging for production
        base_config["handlers"]["audit"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "logs/kaizen_audit.log",
            "maxBytes": 52428800,  # 50MB
            "backupCount": 10
        }
        
        base_config["loggers"]["kaizen.audit"] = {
            "level": "INFO",
            "handlers": ["audit"],
            "propagate": False
        }
    
    return base_config


# Environment-specific configuration factories
def get_development_settings() -> AppSettings:
    """Get development environment settings"""
    return AppSettings(
        environment=Environment.DEVELOPMENT,
        debug=True,
        reload=True,
        allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        docs_url="/docs",
        redoc_url="/redoc"
    )


def get_testing_settings() -> AppSettings:
    """Get testing environment settings"""
    return AppSettings(
        environment=Environment.TESTING,
        debug=False,
        database=DatabaseSettings(name="kaizen_test"),
        redis=RedisSettings(db=1)
    )


def get_production_settings() -> AppSettings:
    """Get production environment settings"""
    return AppSettings(
        environment=Environment.PRODUCTION,
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        workers=4
    )


# Global settings instance
settings = AppSettings()

# Logging configuration
LOGGING_CONFIG = get_logging_config(settings.environment)


def get_settings() -> AppSettings:
    """
    Dependency function to get current settings
    Can be used with FastAPI Depends() for dependency injection
    """
    return settings


# Environment-specific setting getters
_settings_map = {
    Environment.DEVELOPMENT: get_development_settings,
    Environment.TESTING: get_testing_settings,
    Environment.PRODUCTION: get_production_settings,
}


def get_settings_for_environment(env: Environment) -> AppSettings:
    """Get settings for a specific environment"""
    return _settings_map[env]()


# Export commonly used settings
__all__ = [
    "AppSettings",
    "MLSettings", 
    "DatabaseSettings",
    "RedisSettings",
    "SecuritySettings",
    "Environment",
    "settings",
    "LOGGING_CONFIG",
    "get_settings",
    "get_logging_config",
    "get_settings_for_environment"
]