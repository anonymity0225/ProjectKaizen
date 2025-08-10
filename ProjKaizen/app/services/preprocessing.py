import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import UploadFile
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from app.core.settings import settings
from app.schemas.preprocess import (
    CleaningConfig,
    EncodingConfig,
    CleanedDataResponse,
    EncodedDataResponse,
    ValidationReport
)
from app.utils.file_io import save_temp_file, load_csv, load_excel, load_json, load_parquet, cleanup_temp_file

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessingError(Exception):
    """Raised when preprocessing validation or operations fail."""
    pass


class DataPreprocessingService:
    """Enterprise-grade data preprocessing service with comprehensive cleaning and encoding capabilities."""
    
    def __init__(self) -> None:
        """Initialize the preprocessing service with empty label encoder storage."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def generate_cleanliness_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive cleanliness report for the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dictionary containing cleanliness metrics including row/column counts,
            missing value percentages, duplicate counts, and data type summary
        """
        logger.debug(f"Generating cleanliness report for DataFrame with shape {df.shape}")
        
        try:
            # Total rows & columns
            total_rows, total_cols = df.shape
            
            # Missing value percentage per column
            missing_per_column = (df.isna().sum() / total_rows).to_dict()
            
            # Duplicate count
            duplicate_rows = df.duplicated().sum()
            
            # Data type summary
            column_types = df.dtypes.astype(str).to_dict()
            
            # Additional cleanliness metrics
            total_missing_values = df.isna().sum().sum()
            overall_missing_percentage = (total_missing_values / (total_rows * total_cols)) * 100 if total_rows * total_cols > 0 else 0
            
            # Columns with all null values
            completely_null_columns = df.columns[df.isnull().all()].tolist()
            
            # Memory usage
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1024**2
            
            report = {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "missing_per_column": missing_per_column,
                "duplicate_rows": int(duplicate_rows),
                "column_types": column_types,
                "total_missing_values": int(total_missing_values),
                "overall_missing_percentage": round(overall_missing_percentage, 2),
                "completely_null_columns": completely_null_columns,
                "memory_usage_mb": round(memory_usage_mb, 2),
                "report_timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.debug(f"Cleanliness report generated successfully with {total_missing_values} missing values and {duplicate_rows} duplicates")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate cleanliness report: {str(e)}")
            raise DataPreprocessingError(f"Cleanliness report generation failed: {str(e)}")
        
    def validate_data(self, df: pd.DataFrame) -> ValidationReport:
        """Validate DataFrame against enterprise data quality standards and system limits."""
        logger.debug(f"Starting data validation for DataFrame with shape {df.shape}")
        start_time = time.time()
        issues = {}
        warnings = []
        
        try:
            # Enforce system limits
            if len(df) > settings.MAX_ROWS:
                raise DataPreprocessingError(
                    f"Dataset exceeds maximum allowed rows: {len(df)} > {settings.MAX_ROWS}"
                )
            
            if df.shape[1] > settings.MAX_COLUMNS:
                raise DataPreprocessingError(
                    f"Dataset exceeds maximum allowed columns: {df.shape[1]} > {settings.MAX_COLUMNS}"
                )
            
            # Check missing value threshold
            missing_pct = df.isna().mean().max()
            if missing_pct > settings.MISSING_VALUE_THRESHOLD:
                raise DataPreprocessingError(
                    f"Dataset exceeds missing value threshold: {missing_pct:.2%} > {settings.MISSING_VALUE_THRESHOLD:.2%}"
                )
            
            # Check duplicate threshold if configured
            if hasattr(settings, 'DUPLICATE_THRESHOLD'):
                duplicate_pct = df.duplicated().sum() / len(df)
                if duplicate_pct > settings.DUPLICATE_THRESHOLD:
                    raise DataPreprocessingError(
                        f"Dataset exceeds duplicate threshold: {duplicate_pct:.2%} > {settings.DUPLICATE_THRESHOLD:.2%}"
                    )
            
            # Check for completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                issues["null_columns"] = null_columns
                warnings.append(f"Found {len(null_columns)} columns with 100% null values")
            
            # Check for inconsistent data types
            mixed_type_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        numeric_count = pd.to_numeric(non_null_values, errors='coerce').notna().sum()
                        if 0 < numeric_count < len(non_null_values):
                            mixed_type_columns.append(col)
            
            if mixed_type_columns:
                issues["mixed_type_columns"] = mixed_type_columns
                warnings.append(f"Found columns with mixed data types: {mixed_type_columns}")
            
            # Check for high cardinality categorical columns
            high_cardinality_columns = []
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                if total_count > 0 and unique_count > min(1000, total_count * 0.8):
                    high_cardinality_columns.append({
                        "column": col,
                        "unique_values": unique_count,
                        "total_values": total_count
                    })
            
            if high_cardinality_columns:
                issues["high_cardinality_columns"] = high_cardinality_columns
                warnings.append(f"Found {len(high_cardinality_columns)} high cardinality categorical columns")
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                issues["duplicate_rows"] = duplicate_count
                warnings.append(f"Found {duplicate_count} duplicate rows")
            
            # Memory usage analysis
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            if memory_usage > 100:  # Warn if > 100MB
                warnings.append(f"Large dataset detected: {memory_usage:.1f}MB")
            
            duration = time.time() - start_time
            logger.debug(f"Data validation completed in {duration:.3f}s with {len(issues)} issue types")
            
            return ValidationReport(
                is_valid=len(issues) == 0,
                issues=issues,
                warnings=warnings,
                row_count=len(df),
                column_count=len(df.columns),
                memory_usage_mb=round(memory_usage, 2),
                validation_duration=round(duration, 3)
            )
            
        except DataPreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Data validation failed with unexpected error: {str(e)}")
            raise DataPreprocessingError(f"Data validation failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame, config: CleaningConfig) -> CleanedDataResponse:
        """Clean DataFrame using enterprise-grade parallel processing where applicable."""
        logger.debug(f"Starting data cleaning for DataFrame with shape {df.shape}, config: {config.dict()}")
        start_time = time.time()
        original_shape = df.shape
        cleaning_actions = []
        
        try:
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Remove duplicates
            if config.remove_duplicates:
                duplicate_count = cleaned_df.duplicated().sum()
                if duplicate_count > 0:
                    before_shape = cleaned_df.shape
                    cleaned_df = cleaned_df.drop_duplicates()
                    after_shape = cleaned_df.shape
                    
                    cleaning_actions.append({
                        "column": "all",
                        "action": "remove_duplicates",
                        "params": {"method": "drop_duplicates"},
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "before_shape": before_shape,
                        "after_shape": after_shape
                    })
                    logger.info(f"Removed {duplicate_count} duplicate rows")
            
            # Handle missing values with parallel processing if enabled
            if config.handle_missing_values:
                # Process numeric columns
                if config.numeric_columns:
                    if settings.ENABLE_PARALLEL_PROCESSING and len(config.numeric_columns) > 2:
                        # Parallel processing for multiple columns
                        def process_numeric_column(col: str) -> Tuple[str, pd.Series, Dict]:
                            if col not in cleaned_df.columns:
                                raise DataPreprocessingError(f"Numeric column '{col}' not found in DataFrame")
                            
                            null_count = cleaned_df[col].isnull().sum()
                            action_info = None
                            
                            if null_count > 0:
                                before_shape = cleaned_df.shape
                                
                                if config.missing_value_strategy == "fill_mean":
                                    mean_val = cleaned_df[col].mean()
                                    filled_series = cleaned_df[col].fillna(mean_val)
                                    action_info = {
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "mean", "fill_value": round(mean_val, 4)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": before_shape  # Shape doesn't change for fillna
                                    }
                                    return col, filled_series, action_info
                                elif config.missing_value_strategy == "fill_median":
                                    median_val = cleaned_df[col].median()
                                    filled_series = cleaned_df[col].fillna(median_val)
                                    action_info = {
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "median", "fill_value": round(median_val, 4)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": before_shape
                                    }
                                    return col, filled_series, action_info
                            
                            return col, cleaned_df[col], action_info
                        
                        # Execute parallel processing
                        results = Parallel(n_jobs=min(settings.MAX_WORKERS, len(config.numeric_columns)))(
                            delayed(process_numeric_column)(col) for col in config.numeric_columns
                        )
                        
                        # Apply results
                        for col, series, action_info in results:
                            cleaned_df[col] = series
                            if action_info:
                                cleaning_actions.append(action_info)
                    else:
                        # Sequential processing
                        for col in config.numeric_columns:
                            if col not in cleaned_df.columns:
                                raise DataPreprocessingError(f"Numeric column '{col}' not found in DataFrame")
                            
                            null_count = cleaned_df[col].isnull().sum()
                            if null_count > 0:
                                before_shape = cleaned_df.shape
                                
                                if config.missing_value_strategy == "fill_mean":
                                    mean_val = cleaned_df[col].mean()
                                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                                    cleaning_actions.append({
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "mean", "fill_value": round(mean_val, 4)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": cleaned_df.shape
                                    })
                                elif config.missing_value_strategy == "fill_median":
                                    median_val = cleaned_df[col].median()
                                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                                    cleaning_actions.append({
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "median", "fill_value": round(median_val, 4)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": cleaned_df.shape
                                    })
                                elif config.missing_value_strategy == "drop_rows":
                                    before_drop = len(cleaned_df)
                                    cleaned_df = cleaned_df.dropna(subset=[col])
                                    rows_dropped = before_drop - len(cleaned_df)
                                    cleaning_actions.append({
                                        "column": col,
                                        "action": "dropna",
                                        "params": {"method": "drop"},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": cleaned_df.shape
                                    })
                
                # Process categorical columns
                if config.categorical_columns:
                    for col in config.categorical_columns:
                        if col not in cleaned_df.columns:
                            raise DataPreprocessingError(f"Categorical column '{col}' not found in DataFrame")
                        
                        null_count = cleaned_df[col].isnull().sum()
                        if null_count > 0:
                            before_shape = cleaned_df.shape
                            
                            if config.missing_value_strategy == "fill_mode":
                                mode_values = cleaned_df[col].mode()
                                if len(mode_values) > 0:
                                    mode_val = mode_values.iloc[0]
                                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                                    cleaning_actions.append({
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "mode", "fill_value": str(mode_val)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "before_shape": before_shape,
                                        "after_shape": cleaned_df.shape
                                    })
                            elif config.missing_value_strategy == "drop_rows":
                                before_drop = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                rows_dropped = before_drop - len(cleaned_df)
                                cleaning_actions.append({
                                    "column": col,
                                    "action": "dropna",
                                    "params": {"method": "drop"},
                                    "timestamp": pd.Timestamp.now().isoformat(),
                                    "before_shape": before_shape,
                                    "after_shape": cleaned_df.shape
                                })
            
            final_shape = cleaned_df.shape
            duration = time.time() - start_time
            
            logger.debug(f"Data cleaning completed in {duration:.3f}s - Shape: {original_shape} → {final_shape}")
            
            return CleanedDataResponse(
                data=cleaned_df,
                original_shape=original_shape,
                final_shape=final_shape,
                cleaning_actions=cleaning_actions,
                num_rows_removed=original_shape[0] - final_shape[0],
                num_columns_removed=original_shape[1] - final_shape[1]
            )
            
        except DataPreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Data cleaning failed with unexpected error: {str(e)}")
            raise DataPreprocessingError(f"Data cleaning failed: {str(e)}")
    
    def encode_data(self, df: pd.DataFrame, config: EncodingConfig) -> EncodedDataResponse:
        """Encode categorical variables using enterprise-grade parallel processing where applicable."""
        logger.debug(f"Starting data encoding for DataFrame with shape {df.shape}, method: {config.categorical_encoding_method}")
        start_time = time.time()
        original_shape = df.shape
        encoding_actions = []
        
        try:
            # Create a copy to avoid modifying original
            encoded_df = df.copy()
            
            # Validate columns exist
            for col in config.categorical_columns:
                if col not in encoded_df.columns:
                    raise DataPreprocessingError(f"Column '{col}' not found in DataFrame")
            
            if config.categorical_encoding_method == "onehot":
                # One-hot encoding
                original_columns = set(encoded_df.columns)
                before_shape = encoded_df.shape
                encoded_df = pd.get_dummies(encoded_df, columns=config.categorical_columns, prefix=config.categorical_columns)
                after_shape = encoded_df.shape
                new_columns = list(set(encoded_df.columns) - original_columns)
                
                for col in config.categorical_columns:
                    unique_values = df[col].nunique()
                    col_new_columns = [c for c in new_columns if c.startswith(f"{col}_")]
                    encoding_actions.append({
                        "column": col,
                        "action": "onehot_encode",
                        "params": {
                            "method": "onehot",
                            "unique_values": unique_values,
                            "new_columns": col_new_columns,
                            "columns_created": len(col_new_columns)
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "before_shape": before_shape,
                        "after_shape": after_shape
                    })
                
            elif config.categorical_encoding_method == "label":
                # Label encoding with parallel processing if enabled
                if settings.ENABLE_PARALLEL_PROCESSING and len(config.categorical_columns) > 2:
                    def process_label_encoding(col: str) -> Tuple[str, pd.Series, Dict, LabelEncoder]:
                        unique_values = encoded_df[col].nunique()
                        le = LabelEncoder()
                        before_shape = encoded_df.shape
                        
                        # Handle NaN values by creating a mask
                        mask = encoded_df[col].notna()
                        encoded_series = encoded_df[col].copy()
                        
                        if mask.sum() > 0:  # Only encode if there are non-null values
                            encoded_series.loc[mask] = le.fit_transform(encoded_df.loc[mask, col])
                            
                            action_info = {
                                "column": col,
                                "action": "label_encode",
                                "params": {
                                    "method": "label",
                                    "unique_values": unique_values,
                                    "encoded_range": f"0-{len(le.classes_)-1}",
                                    "classes": le.classes_.tolist() if len(le.classes_) <= 20 else f"{len(le.classes_)} classes"
                                },
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "before_shape": before_shape,
                                "after_shape": before_shape  # Shape doesn't change for label encoding
                            }
                            return col, encoded_series, action_info, le
                        
                        return col, encoded_series, None, None
                    
                    # Execute parallel processing
                    results = Parallel(n_jobs=min(settings.MAX_WORKERS, len(config.categorical_columns)))(
                        delayed(process_label_encoding)(col) for col in config.categorical_columns
                    )
                    
                    # Apply results
                    for col, series, action_info, le in results:
                        encoded_df[col] = series
                        if action_info and le:
                            encoding_actions.append(action_info)
                            self.label_encoders[col] = le
                else:
                    # Sequential processing
                    for col in config.categorical_columns:
                        unique_values = encoded_df[col].nunique()
                        le = LabelEncoder()
                        before_shape = encoded_df.shape
                        
                        # Handle NaN values by creating a mask
                        mask = encoded_df[col].notna()
                        if mask.sum() > 0:  # Only encode if there are non-null values
                            encoded_df.loc[mask, col] = le.fit_transform(encoded_df.loc[mask, col])
                            
                            # Store encoder for potential inverse transform
                            self.label_encoders[col] = le
                            
                            encoding_actions.append({
                                "column": col,
                                "action": "label_encode",
                                "params": {
                                    "method": "label",
                                    "unique_values": unique_values,
                                    "encoded_range": f"0-{len(le.classes_)-1}",
                                    "classes": le.classes_.tolist() if len(le.classes_) <= 20 else f"{len(le.classes_)} classes"
                                },
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "before_shape": before_shape,
                                "after_shape": encoded_df.shape
                            })
                    
            elif config.categorical_encoding_method == "target":
                # Target encoding
                for col in config.categorical_columns:
                    before_shape = encoded_df.shape
                    value_counts = encoded_df[col].value_counts()
                    encoded_df[col] = encoded_df[col].map(value_counts)
                    
                    encoding_actions.append({
                        "column": col,
                        "action": "target_encode",
                        "params": {
                            "method": "target",
                            "unique_values": len(value_counts),
                            "frequency_range": f"{value_counts.min()}-{value_counts.max()}"
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "before_shape": before_shape,
                        "after_shape": encoded_df.shape
                    })
                    
            else:
                raise DataPreprocessingError(f"Unsupported encoding method: {config.categorical_encoding_method}")
            
            final_shape = encoded_df.shape
            duration = time.time() - start_time
            
            logger.debug(f"Data encoding completed in {duration:.3f}s - Shape: {original_shape} → {final_shape}")
            
            return EncodedDataResponse(
                data=encoded_df,
                original_shape=original_shape,
                final_shape=final_shape,
                encoding_actions=encoding_actions,
                encoders_used=config.categorical_columns,
                scaler_used=None
            )
            
        except DataPreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Data encoding failed with unexpected error: {str(e)}")
            raise DataPreprocessingError(f"Data encoding failed: {str(e)}")


# FastAPI wrapper functions (keeping FastAPI concerns separate)
async def handle_file_upload(file: UploadFile) -> Tuple[pd.DataFrame, str]:
    """Handle file upload and return DataFrame with temporary file path for cleanup."""
    logger.debug(f"Starting file upload handling for: {file.filename}")
    
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.SUPPORTED_FILE_EXTENSIONS:
            raise DataPreprocessingError(f"Unsupported file format: {file_extension}. Supported formats: {settings.SUPPORTED_FILE_EXTENSIONS}")
        
        # Validate file size before saving
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise DataPreprocessingError(f"File size ({file_size_mb:.1f}MB) exceeds maximum limit ({settings.MAX_FILE_SIZE_MB}MB)")
        
        # Reset file pointer for saving
        await file.seek(0)
        
        # Save uploaded file temporarily
        temp_path = await save_temp_file(file)
        
        # Determine file type and read accordingly
        if file_extension == '.csv':
            df = load_csv(temp_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = load_excel(temp_path)
        elif file_extension == '.json':
            df = load_json(temp_path)
        elif file_extension == '.parquet':
            df = load_parquet(temp_path)
        else:
            raise DataPreprocessingError(f"Unsupported file format: {file_extension}")
        
        if df.empty:
            raise DataPreprocessingError("Uploaded file contains no data or is empty")
        
        logger.debug(f"Successfully loaded file: {file.filename} - Shape: {df.shape}")
        return df, temp_path
        
    except DataPreprocessingError:
        raise
    except Exception as e:
        logger.error(f"File upload handling failed with unexpected error: {str(e)}")
        raise DataPreprocessingError(f"File could not be processed: {str(e)}")


# Main preprocessing pipeline
def preprocess_pipeline(
    df: pd.DataFrame, 
    cleaning_config: Optional[CleaningConfig] = None,
    encoding_config: Optional[EncodingConfig] = None,
    validate_first: bool = True,
    temp_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete enterprise preprocessing pipeline with validation, cleaning, and encoding.
    
    Args:
        df: Input DataFrame to preprocess
        cleaning_config: Optional configuration for data cleaning operations
        encoding_config: Optional configuration for data encoding operations
        validate_first: Whether to run validation before processing
        temp_path: Optional temporary file path for cleanup
    
    Returns:
        Dictionary containing processing results, cleanliness report, and audit information
    
    Example:
        >>> from app.services.preprocessing import preprocess_pipeline
        >>> from app.schemas.preprocess import CleaningConfig
        >>> config = CleaningConfig(remove_duplicates=True, handle_missing_values=True)
        >>> result = preprocess_pipeline(df, cleaning_config=config)
    """
    logger.debug(f"Starting preprocessing pipeline for DataFrame with shape {df.shape}")
    service = DataPreprocessingService()
    audit_log = []
    
    try:
        # Track pipeline start
        pipeline_start = time.time()
        
        # Generate initial cleanliness report
        initial_cleanliness_report = service.generate_cleanliness_report(df)
        logger.info(f"Initial data cleanliness: {initial_cleanliness_report['total_missing_values']} missing values, {initial_cleanliness_report['duplicate_rows']} duplicates")
        
        # Validation
        validation_result = None
        if validate_first:
            validation_result = service.validate_data(df)
            if not validation_result.is_valid:
                logger.warning(f"Data validation found issues: {validation_result.issues}")
        
        # Cleaning
        cleaning_result = None
        if cleaning_config:
            cleaning_result = service.clean_data(df, cleaning_config)
            df = cleaning_result.data
            audit_log.extend(cleaning_result.cleaning_actions)
        
        # Encoding
        encoding_result = None
        if encoding_config:
            encoding_result = service.encode_data(df, encoding_config)
            df = encoding_result.data
            audit_log.extend(encoding_result.encoding_actions)
        
        # Generate final cleanliness report
        final_cleanliness_report = service.generate_cleanliness_report(df)
        
        # Calculate processing summary
        total_duration = time.time() - pipeline_start
        
        # Log audit information if enabled
        if settings.ENABLE_AUDIT_LOGGING and audit_log:
            logger.info(json.dumps({
    "pipeline_audit": audit_log,
    "initial_shape": {
        "rows": initial_cleanliness_report["total_rows"],
        "columns": initial_cleanliness_report["total_columns"]
    },
    "final_shape": df.shape,
    "total_duration": round(total_duration, 3)
}))
        
        logger.debug(f"Preprocessing pipeline completed successfully with final shape {df.shape}")
        
        result = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "audit_log": audit_log,
            "preview": df.head(5).to_dict(orient="records"),
            "initial_cleanliness_report": initial_cleanliness_report,
            "final_cleanliness_report": final_cleanliness_report,
            "validation_result": validation_result.dict() if validation_result else None,
            "processing_duration": round(total_duration, 3),
            "pipeline_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return result
        
    except DataPreprocessingError as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed with unexpected error: {str(e)}")
        raise DataPreprocessingError(f"Preprocessing pipeline failed: {str(e)}")
    finally:
        # Clean up temporary file if provided
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
                logger.debug(f"Successfully cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {str(e)}")


# Legacy function for backward compatibility
async def preprocess_data(file: UploadFile, config: Optional[CleaningConfig] = None) -> dict:
    """
    Legacy function for backward compatibility - handles file upload and basic cleaning.
    
    Args:
        file: Uploaded file to process
        config: Optional cleaning configuration
        
    Returns:
        Dictionary containing processing results and cleaned data
        
    Raises:
        DataPreprocessingError: If file processing or cleaning fails
    """
    logger.debug(f"Using legacy preprocess_data function for file: {file.filename}")
    
    try:
        df, temp_path = await handle_file_upload(file)
        
        try:
            if config:
                result = preprocess_pipeline(
                    df, 
                    cleaning_config=config, 
                    temp_path=temp_path
                )
                return {
                    "processed_rows": result["num_rows"],
                    "cleaned_columns": list(df.columns),
                    "data": df,
                    "cleanliness_report": result.get("final_cleanliness_report"),
                    "processing_summary": {
                        "duration": result.get("processing_duration"),
                        "actions_performed": len(result.get("audit_log", []))
                    }
                }
            else:
                # Generate cleanliness report even without cleaning config
                service = DataPreprocessingService()
                cleanliness_report = service.generate_cleanliness_report(df)
                
                return {
                    "processed_rows": len(df),
                    "cleaned_columns": list(df.columns),
                    "data": df,
                    "cleanliness_report": cleanliness_report
                }
        finally:
            # Ensure cleanup happens even if processing fails
            if temp_path:
                try:
                    cleanup_temp_file(temp_path)
                    logger.debug(f"Successfully cleaned up temporary file in legacy function: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file in legacy function: {str(e)}")
            
    except DataPreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Legacy preprocess_data failed with unexpected error: {str(e)}")
        raise DataPreprocessingError(f"Data preprocessing failed: {str(e)}")