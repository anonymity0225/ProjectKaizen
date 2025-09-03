# Utility functions for loading and managing config 
# Configuration conversion utilities
from typing import Optional, Tuple
from ..schemas.preprocess import PreprocessingRequest, CleaningConfig, EncodingConfig

def convert_preprocessing_config(config: PreprocessingRequest) -> Tuple[Optional[CleaningConfig], Optional[EncodingConfig]]:
    """
    Convert PreprocessingRequest to CleaningConfig and EncodingConfig objects.
    
    Args:
        config: The preprocessing request configuration
        
    Returns:
        Tuple of (CleaningConfig, EncodingConfig) where either can be None
    """
    cleaning_config = None
    encoding_config = None
    
    # Create CleaningConfig if any cleaning operations are requested
    if config.drop_duplicates or config.handle_missing:
        cleaning_config = CleaningConfig(
            remove_duplicates=config.drop_duplicates,
            handle_missing=config.handle_missing,
            numeric_columns=config.numeric_columns,
            categorical_columns=config.categorical_columns
        )
    
    # Create EncodingConfig if encoding is requested
    if config.encoding_method and config.encoding_columns:
        encoding_config = EncodingConfig(
            method=config.encoding_method,
            columns=config.encoding_columns
        )
    
    return cleaning_config, encoding_config