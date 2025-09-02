from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MLSettings(BaseModel):
    """Machine Learning specific settings."""
    model_path: str = Field(default="./models", description="Path to ML models directory")


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="ALLOWED_ORIGINS",
        description="List of allowed CORS origins"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Application log level"
    )
    
    # API settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    # Debug mode
    debug: bool = Field(default=False, alias="DEBUG")
    
    # ML settings
    ml: MLSettings = Field(default_factory=MLSettings)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()