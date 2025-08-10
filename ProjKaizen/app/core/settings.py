"""
Core settings module that provides a clean interface to application settings.
"""
from app.config.settings import get_settings, PreprocessingSettings

# Export the settings instance
settings = get_settings()

# Export commonly used settings for convenience
SUPPORTED_FILE_EXTENSIONS = settings.SUPPORTED_FILE_EXTENSIONS
MAX_ROWS = settings.MAX_ROWS
MAX_COLUMNS = settings.MAX_COLUMNS
MAX_MEMORY_USAGE_MB = settings.MAX_MEMORY_USAGE_MB
MAX_CATEGORIES_FOR_ONEHOT = settings.MAX_CATEGORIES_FOR_ONEHOT
MISSING_VALUE_THRESHOLD = settings.MISSING_VALUE_THRESHOLD
MAX_FILE_SIZE_MB = settings.MAX_FILE_SIZE_MB
TEMP_DIR = settings.TEMP_DIR 