# config/settings.py - Enterprise configuration management
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field
from typing import Set, Dict, Any
from functools import lru_cache
import os


class PreprocessingSettings(BaseSettings):
    """
    Enterprise-grade configuration settings for data preprocessing service.
    Uses Pydantic BaseSettings for environment variable support and validation.
    """
    
    # File processing limits
    MAX_FILE_SIZE_MB: int = Field(default=100, description="Maximum file size in MB")
    MAX_ROWS: int = Field(default=1_000_000, description="Maximum number of rows")
    MAX_COLUMNS: int = Field(default=1000, description="Maximum number of columns")
    MAX_MEMORY_USAGE_MB: int = Field(default=2048, description="Maximum memory usage in MB")
    
    # Supported file formats
    SUPPORTED_FILE_EXTENSIONS: Set[str] = Field(
        default={'.csv', '.xlsx', '.xls', '.parquet', '.json'},
        description="Supported file extensions"
    )
    
    # Data quality thresholds
    MISSING_VALUE_THRESHOLD: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Threshold for missing values (0.5 = 50%)"
    )
    DUPLICATE_THRESHOLD: float = Field(
        default=0.9, 
        ge=0.0, 
        le=1.0,
        description="Threshold for duplicate detection"
    )
    
    # Encoding settings
    MAX_CATEGORIES_FOR_ONEHOT: int = Field(
        default=10, 
        description="Maximum categories for one-hot encoding"
    )
    DEFAULT_SCALING_METHOD: str = Field(
        default="standard", 
        description="Default scaling method"
    )
    
    # Processing settings
    PROCESSING_TIMEOUT_SECONDS: int = Field(
        default=300, 
        description="Processing timeout in seconds"
    )
    TEMP_DIR: str = Field(
        default="/tmp/preprocessing", 
        description="Temporary directory for file processing"
    )
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    ENABLE_STRUCTURED_LOGGING: bool = Field(
        default=True, 
        description="Enable structured logging with JSON format"
    )
    
    # Performance settings
    ENABLE_PARALLEL_PROCESSING: bool = Field(
        default=True, 
        description="Enable parallel processing where possible"
    )
    MAX_WORKERS: int = Field(
        default=4, 
        description="Maximum number of worker threads"
    )
    
    # Feature flags
    ENABLE_ADVANCED_VALIDATION: bool = Field(
        default=True, 
        description="Enable advanced data validation"
    )
    ENABLE_DATA_PROFILING: bool = Field(
        default=True, 
        description="Enable detailed data profiling"
    )
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True, 
        description="Enable comprehensive audit logging"
    )
    
    # Security settings
    ALLOWED_MIME_TYPES: Set[str] = Field(
        default={
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json'
        },
        description="Allowed MIME types for uploads"
    )
    
    # Database/Cache settings (if needed)
    ENABLE_RESULT_CACHING: bool = Field(
        default=False, 
        description="Enable result caching"
    )
    CACHE_TTL_SECONDS: int = Field(
        default=3600, 
        description="Cache TTL in seconds"
    )
    
    # Monitoring settings
    ENABLE_METRICS: bool = Field(
        default=True, 
        description="Enable metrics collection"
    )
    METRICS_ENDPOINT: str = Field(
        default="/metrics", 
        description="Metrics endpoint path"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "PREPROCESSING_"
        case_sensitive = True


class DevelopmentSettings(PreprocessingSettings):
    """Development environment specific settings."""
    
    LOG_LEVEL: str = "DEBUG"
    MAX_FILE_SIZE_MB: int = 50
    MAX_ROWS: int = 100_000
    PROCESSING_TIMEOUT_SECONDS: int = 60
    ENABLE_RESULT_CACHING: bool = False


class ProductionSettings(PreprocessingSettings):
    """Production environment specific settings."""
    
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 500
    MAX_ROWS: int = 10_000_000
    MAX_MEMORY_USAGE_MB: int = 8192
    PROCESSING_TIMEOUT_SECONDS: int = 600
    ENABLE_RESULT_CACHING: bool = True
    ENABLE_METRICS: bool = True


class TestingSettings(PreprocessingSettings):
    """Testing environment specific settings."""
    
    LOG_LEVEL: str = "WARNING"
    MAX_FILE_SIZE_MB: int = 10
    MAX_ROWS: int = 1000
    PROCESSING_TIMEOUT_SECONDS: int = 30
    ENABLE_AUDIT_LOGGING: bool = False
    ENABLE_METRICS: bool = False


@lru_cache()
def get_settings() -> PreprocessingSettings:
    """
    Get settings instance based on environment.
    Uses LRU cache to ensure singleton behavior.
    
    Returns:
        PreprocessingSettings: Configuration settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export commonly used settings
settings = get_settings()

# Configuration validation
def validate_settings() -> Dict[str, Any]:
    """
    Validate current settings and return validation report.
    
    Returns:
        Dict containing validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "settings_summary": {}
    }
    
    try:
        current_settings = get_settings()
        
        # Validate critical settings
        if current_settings.MAX_FILE_SIZE_MB <= 0:
            validation_results["errors"].append("MAX_FILE_SIZE_MB must be positive")
            validation_results["valid"] = False
        
        if current_settings.MAX_ROWS <= 0:
            validation_results["errors"].append("MAX_ROWS must be positive")
            validation_results["valid"] = False
        
        if not (0.0 <= current_settings.MISSING_VALUE_THRESHOLD <= 1.0):
            validation_results["errors"].append("MISSING_VALUE_THRESHOLD must be between 0.0 and 1.0")
            validation_results["valid"] = False
        
        # Check if temp directory is writable
        import tempfile
        import os
        try:
            temp_dir = current_settings.TEMP_DIR
            os.makedirs(temp_dir, exist_ok=True)
            test_file = os.path.join(temp_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            validation_results["warnings"].append(f"Temp directory not writable: {str(e)}")
        
        # Generate settings summary
        validation_results["settings_summary"] = {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "max_file_size_mb": current_settings.MAX_FILE_SIZE_MB,
            "max_rows": current_settings.MAX_ROWS,
            "supported_extensions": list(current_settings.SUPPORTED_FILE_EXTENSIONS),
            "log_level": current_settings.LOG_LEVEL,
            "features_enabled": {
                "advanced_validation": current_settings.ENABLE_ADVANCED_VALIDATION,
                "data_profiling": current_settings.ENABLE_DATA_PROFILING,
                "audit_logging": current_settings.ENABLE_AUDIT_LOGGING,
                "metrics": current_settings.ENABLE_METRICS,
                "caching": current_settings.ENABLE_RESULT_CACHING
            }
        }
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Settings validation failed: {str(e)}")
    
    return validation_results


# Constants for backwards compatibility
SUPPORTED_FILE_EXTENSIONS = settings.SUPPORTED_FILE_EXTENSIONS
DEFAULT_MISSING_VALUE_THRESHOLD = settings.MISSING_VALUE_THRESHOLD
DEFAULT_DUPLICATE_THRESHOLD = settings.DUPLICATE_THRESHOLD
MAX_CATEGORIES_FOR_ONEHOT = settings.MAX_CATEGORIES_FOR_ONEHOT