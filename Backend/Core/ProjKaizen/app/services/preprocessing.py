import json
import logging
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from app.config.settings import settings
from app.schemas.preprocess import (
    CleaningConfig,
    EncodingConfig,
    CleanlinessReport,
    ValidationReport,
    CleanedDataResponse,
    EncodedDataResponse,
    PreprocessingResponse
)
from app.utils.file_io import save_temp_file, load_csv, load_excel, load_json, load_parquet

# Configure logging
logger = logging.getLogger(__name__)

# Thread executor for CPU-bound operations
_thread_pool: Optional[ThreadPoolExecutor] = None
_thread_pool_lock = threading.Lock()


def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create thread pool for CPU-bound operations."""
    global _thread_pool
    if _thread_pool is None:
        with _thread_pool_lock:
            if _thread_pool is None:
                max_workers = getattr(settings, 'MAX_WORKERS', 4)
                _thread_pool = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="preprocessing"
                )
    return _thread_pool


# Metrics stubs for future implementation
class MetricsStub:
    """Placeholder for metrics implementation."""
    
    def observe_duration(self, operation: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record operation duration."""
        pass
    
    def count(self, metric: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        pass
    
    def gauge(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        pass


metrics = MetricsStub()


class DataPreprocessingError(Exception):
    """Raised when preprocessing validation or operations fail."""
    pass


class DataPreprocessingService:
    """Enterprise-grade data preprocessing service with comprehensive cleaning and encoding capabilities."""
    
    def __init__(self) -> None:
        """Initialize the preprocessing service with encoder persistence support."""
        self.encoders_path = Path("models/encoders")
        self.scalers_path = Path("models/scalers")
        self.encoders_path.mkdir(parents=True, exist_ok=True)
        self.scalers_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_content_hash(
        self,
        df: pd.DataFrame,
        columns: List[str],
        config: Union[CleaningConfig, EncodingConfig],
        operation: str
    ) -> str:
        """Generate deterministic content hash for encoder/scaler persistence."""
        # Create schema fingerprint
        schema_info = {
            "columns": sorted(columns),
            "dtypes": {col: str(df[col].dtype) for col in columns if col in df.columns},
            "operation": operation,
            "config": config.dict() if hasattr(config, 'dict') else str(config)
        }
        
        # Add unique values for categorical columns (sorted for determinism)
        for col in columns:
            if col in df.columns and df[col].dtype == 'object':
                unique_vals = sorted(df[col].dropna().unique().astype(str))
                schema_info[f"unique_{col}"] = unique_vals[:100]  # Limit for hash stability
        
        content_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _save_encoder(self, encoder_id: str, encoder: LabelEncoder) -> None:
        """Persist encoder to disk."""
        encoder_path = self.encoders_path / f"{encoder_id}.joblib"
        try:
            dump(encoder, encoder_path)
            logger.debug(f"Saved encoder {encoder_id}")
        except Exception as e:
            logger.error(f"Failed to save encoder {encoder_id}: {str(e)}")
            raise DataPreprocessingError(f"Encoder persistence failed: {str(e)}")
    
    def _load_encoder(self, encoder_id: str) -> Optional[LabelEncoder]:
        """Load encoder from disk if it exists."""
        encoder_path = self.encoders_path / f"{encoder_id}.joblib"
        if not encoder_path.exists():
            return None
        
        try:
            encoder = load(encoder_path)
            logger.debug(f"Loaded encoder {encoder_id}")
            return encoder
        except Exception as e:
            logger.warning(f"Failed to load encoder {encoder_id}: {str(e)}")
            return None
    
    def _save_scaler(self, scaler_id: str, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        """Persist scaler to disk."""
        scaler_path = self.scalers_path / f"{scaler_id}.joblib"
        try:
            dump(scaler, scaler_path)
            logger.debug(f"Saved scaler {scaler_id}")
        except Exception as e:
            logger.error(f"Failed to save scaler {scaler_id}: {str(e)}")
            raise DataPreprocessingError(f"Scaler persistence failed: {str(e)}")
    
    def _load_scaler(self, scaler_id: str) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        """Load scaler from disk if it exists."""
        scaler_path = self.scalers_path / f"{scaler_id}.joblib"
        if not scaler_path.exists():
            return None
        
        try:
            scaler = load(scaler_path)
            logger.debug(f"Loaded scaler {scaler_id}")
            return scaler
        except Exception as e:
            logger.warning(f"Failed to load scaler {scaler_id}: {str(e)}")
            return None
    
    def generate_cleanliness_report(self, df: pd.DataFrame, request_id: str) -> CleanlinessReport:
        """Generate a comprehensive cleanliness report for the DataFrame."""
        logger.info(f"Generating cleanliness report | request_id={request_id} | shape={df.shape}")
        start_time = time.time()
        
        try:
            total_rows, total_cols = df.shape
            
            # Missing value percentage per column
            missing_per_column = (df.isna().sum() / total_rows).to_dict()
            
            # Duplicate count
            duplicate_rows = int(df.duplicated().sum())
            
            # Data type summary
            column_types = df.dtypes.astype(str).to_dict()
            
            # Categorical cardinality
            categorical_cardinality = {}
            for col in df.select_dtypes(include=['object']).columns:
                categorical_cardinality[col] = int(df[col].nunique())
            
            duration_ms = (time.time() - start_time) * 1000
            metrics.observe_duration("cleanliness_report", duration_ms, {"request_id": request_id})
            
            report = CleanlinessReport(
                total_rows=total_rows,
                total_columns=total_cols,
                missing_per_column=missing_per_column,
                duplicate_rows=duplicate_rows,
                column_types=column_types,
                categorical_cardinality=categorical_cardinality
            )
            
            logger.info(f"Cleanliness report completed | request_id={request_id} | duration_ms={duration_ms:.1f}")
            return report
            
        except Exception as e:
            metrics.count("errors", {"operation": "cleanliness_report", "request_id": request_id})
            logger.error(f"Failed to generate cleanliness report | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Cleanliness report generation failed: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, request_id: str) -> ValidationReport:
        """Validate DataFrame against enterprise data quality standards."""
        logger.info(f"Starting data validation | request_id={request_id} | shape={df.shape}")
        start_time = time.time()
        issues = {}
        warnings = []
        
        try:
            # Enforce system limits
            if len(df) > settings.MAX_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset exceeds maximum allowed rows: {len(df)} > {settings.MAX_ROWS}"
                )
            
            if df.shape[1] > settings.MAX_COLUMNS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset exceeds maximum allowed columns: {df.shape[1]} > {settings.MAX_COLUMNS}"
                )
            
            # Check missing value threshold
            missing_pct = df.isna().mean().max()
            if missing_pct > settings.MISSING_VALUE_THRESHOLD:
                raise HTTPException(
                    status_code=422,
                    detail=f"Dataset exceeds missing value threshold: {missing_pct:.2%} > {settings.MISSING_VALUE_THRESHOLD:.2%}"
                )
            
            # Check duplicate threshold if configured
            if hasattr(settings, 'DUPLICATE_THRESHOLD'):
                duplicate_pct = df.duplicated().sum() / len(df)
                if duplicate_pct > settings.DUPLICATE_THRESHOLD:
                    issues["high_duplicate_rate"] = {
                        "rate": round(duplicate_pct, 4),
                        "threshold": settings.DUPLICATE_THRESHOLD
                    }
                    warnings.append(f"High duplicate rate: {duplicate_pct:.2%}")
            
            # Check for completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                issues["null_columns"] = null_columns
                warnings.append(f"Found {len(null_columns)} columns with 100% null values")
            
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
            
            duration_ms = (time.time() - start_time) * 1000
            metrics.observe_duration("validation", duration_ms, {"request_id": request_id})
            
            validation_report = ValidationReport(
                is_valid=len(issues) == 0,
                issues=issues,
                warnings=warnings,
                row_count=len(df),
                column_count=len(df.columns)
            )
            
            logger.info(f"Data validation completed | request_id={request_id} | valid={validation_report.is_valid} | duration_ms={duration_ms:.1f}")
            return validation_report
            
        except HTTPException:
            # Re-raise HTTPExceptions unchanged
            metrics.count("errors", {"operation": "validation", "type": "http", "request_id": request_id})
            raise
        except Exception as e:
            metrics.count("errors", {"operation": "validation", "type": "unexpected", "request_id": request_id})
            logger.error(f"Data validation failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data validation failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame, config: CleaningConfig, request_id: str) -> CleanedDataResponse:
        """Clean DataFrame using enterprise-grade processing."""
        logger.info(f"Starting data cleaning | request_id={request_id} | shape={df.shape} | config={config.dict()}")
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
                    before_count = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates()
                    after_count = len(cleaned_df)
                    
                    action = {
                        "column": "all",
                        "action": "remove_duplicates",
                        "params": {"rows_removed": before_count - after_count},
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "request_id": request_id
                    }
                    cleaning_actions.append(action)
                    logger.info(f"Removed {before_count - after_count} duplicate rows | request_id={request_id}")
            
            # Handle missing values
            if config.handle_missing_values:
                # Process numeric columns
                if config.numeric_columns:
                    for col in config.numeric_columns:
                        if col not in cleaned_df.columns:
                            raise DataPreprocessingError(f"Numeric column '{col}' not found in DataFrame")
                        
                        null_count = cleaned_df[col].isnull().sum()
                        if null_count > 0:
                            if config.missing_value_strategy == "fill_mean":
                                mean_val = cleaned_df[col].mean()
                                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                                action = {
                                    "column": col,
                                    "action": "fillna",
                                    "params": {"method": "mean", "fill_value": round(mean_val, 4)},
                                    "timestamp": pd.Timestamp.now().isoformat(),
                                    "request_id": request_id
                                }
                                cleaning_actions.append(action)
                            elif config.missing_value_strategy == "fill_median":
                                median_val = cleaned_df[col].median()
                                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                                action = {
                                    "column": col,
                                    "action": "fillna",
                                    "params": {"method": "median", "fill_value": round(median_val, 4)},
                                    "timestamp": pd.Timestamp.now().isoformat(),
                                    "request_id": request_id
                                }
                                cleaning_actions.append(action)
                            elif config.missing_value_strategy == "drop_rows":
                                before_count = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                rows_dropped = before_count - len(cleaned_df)
                                action = {
                                    "column": col,
                                    "action": "dropna",
                                    "params": {"rows_dropped": rows_dropped},
                                    "timestamp": pd.Timestamp.now().isoformat(),
                                    "request_id": request_id
                                }
                                cleaning_actions.append(action)
                
                # Process categorical columns
                if config.categorical_columns:
                    for col in config.categorical_columns:
                        if col not in cleaned_df.columns:
                            raise DataPreprocessingError(f"Categorical column '{col}' not found in DataFrame")
                        
                        null_count = cleaned_df[col].isnull().sum()
                        if null_count > 0:
                            if config.missing_value_strategy == "fill_mode":
                                mode_values = cleaned_df[col].mode()
                                if len(mode_values) > 0:
                                    mode_val = mode_values.iloc[0]
                                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                                    action = {
                                        "column": col,
                                        "action": "fillna",
                                        "params": {"method": "mode", "fill_value": str(mode_val)},
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "request_id": request_id
                                    }
                                    cleaning_actions.append(action)
                            elif config.missing_value_strategy == "drop_rows":
                                before_count = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                rows_dropped = before_count - len(cleaned_df)
                                action = {
                                    "column": col,
                                    "action": "dropna",
                                    "params": {"rows_dropped": rows_dropped},
                                    "timestamp": pd.Timestamp.now().isoformat(),
                                    "request_id": request_id
                                }
                                cleaning_actions.append(action)
            
            final_shape = cleaned_df.shape
            duration_ms = (time.time() - start_time) * 1000
            metrics.observe_duration("cleaning", duration_ms, {"request_id": request_id})
            
            response = CleanedDataResponse(
                data=cleaned_df,
                original_shape=original_shape,
                final_shape=final_shape,
                cleaning_actions=cleaning_actions,
                num_rows_removed=original_shape[0] - final_shape[0],
                num_columns_removed=original_shape[1] - final_shape[1]
            )
            
            logger.info(f"Data cleaning completed | request_id={request_id} | shape={original_shape}→{final_shape} | duration_ms={duration_ms:.1f}")
            return response
            
        except DataPreprocessingError:
            metrics.count("errors", {"operation": "cleaning", "type": "domain", "request_id": request_id})
            raise
        except Exception as e:
            metrics.count("errors", {"operation": "cleaning", "type": "unexpected", "request_id": request_id})
            logger.error(f"Data cleaning failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data cleaning failed: {str(e)}")
    
    def encode_data(self, df: pd.DataFrame, config: EncodingConfig, request_id: str) -> EncodedDataResponse:
        """Encode categorical variables with deterministic, persisted encoders."""
        logger.info(f"Starting data encoding | request_id={request_id} | shape={df.shape} | method={config.categorical_encoding_method}")
        start_time = time.time()
        original_shape = df.shape
        encoding_actions = []
        encoder_ids = {}
        
        try:
            # Create a copy to avoid modifying original
            encoded_df = df.copy()
            
            # Validate columns exist
            for col in config.categorical_columns:
                if col not in encoded_df.columns:
                    raise DataPreprocessingError(f"Column '{col}' not found in DataFrame")
            
            if config.categorical_encoding_method == "onehot":
                # One-hot encoding with deterministic column ordering
                original_columns = set(encoded_df.columns)
                
                # Get unique categories for each column and sort deterministically
                category_mappings = {}
                for col in config.categorical_columns:
                    # Replace NaN with special token for consistent handling
                    unique_cats = encoded_df[col].fillna("__NA__").unique()
                    unique_cats_sorted = sorted(str(cat) for cat in unique_cats)
                    category_mappings[col] = unique_cats_sorted
                
                # Generate content hash for persistence
                content_hash = self._generate_content_hash(df, config.categorical_columns, config, "onehot")
                
                # Create one-hot columns deterministically
                for col in config.categorical_columns:
                    col_data = encoded_df[col].fillna("__NA__")
                    categories = category_mappings[col]
                    
                    # Create one-hot columns
                    for category in categories:
                        new_col_name = f"{col}_{category}"
                        encoded_df[new_col_name] = (col_data.astype(str) == category).astype(int)
                    
                    # Drop original column
                    encoded_df = encoded_df.drop(columns=[col])
                    
                    action = {
                        "column": col,
                        "action": "onehot_encode",
                        "params": {
                            "method": "onehot",
                            "categories": categories,
                            "columns_created": len(categories),
                            "content_hash": content_hash
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "request_id": request_id
                    }
                    encoding_actions.append(action)
                
            elif config.categorical_encoding_method == "label":
                # Label encoding with deterministic mappings
                for col in config.categorical_columns:
                    # Generate content hash for this column
                    content_hash = self._generate_content_hash(df, [col], config, "label")
                    encoder_id = f"label_{col}_{content_hash}"
                    
                    # Try to load existing encoder
                    encoder = self._load_encoder(encoder_id)
                    
                    if encoder is None:
                        # Create new encoder with deterministic categories
                        encoder = LabelEncoder()
                        
                        # Get unique values and sort deterministically (NaN → "__NA__")
                        unique_values = encoded_df[col].fillna("__NA__").unique()
                        categories = sorted(str(val) for val in unique_values)
                        
                        # Fit encoder
                        encoder.fit(categories)
                        
                        # Persist encoder
                        self._save_encoder(encoder_id, encoder)
                        logger.info(f"Created and saved new label encoder | request_id={request_id} | column={col} | encoder_id={encoder_id}")
                    else:
                        logger.info(f"Loaded existing label encoder | request_id={request_id} | column={col} | encoder_id={encoder_id}")
                    
                    # Apply encoding with unseen category handling
                    col_data = encoded_df[col].fillna("__NA__").astype(str)
                    
                    # Handle unseen categories
                    known_categories = set(encoder.classes_)
                    unseen_categories = set(col_data.unique()) - known_categories
                    
                    if unseen_categories:
                        logger.warning(f"Found unseen categories | request_id={request_id} | column={col} | unseen={list(unseen_categories)}")
                        # Map unseen to "__UNSEEN__" or first category
                        if "__UNSEEN__" in encoder.classes_:
                            col_data = col_data.replace(list(unseen_categories), "__UNSEEN__")
                        else:
                            # Map to first category as fallback
                            col_data = col_data.replace(list(unseen_categories), encoder.classes_[0])
                    
                    # Transform
                    encoded_df[col] = encoder.transform(col_data)
                    encoder_ids[col] = encoder_id
                    
                    action = {
                        "column": col,
                        "action": "label_encode",
                        "params": {
                            "method": "label",
                            "categories": encoder.classes_.tolist(),
                            "unseen_categories": list(unseen_categories) if unseen_categories else [],
                            "encoder_id": encoder_id
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "request_id": request_id
                    }
                    encoding_actions.append(action)
            
            elif config.categorical_encoding_method == "target":
                # Target encoding (frequency-based)
                for col in config.categorical_columns:
                    value_counts = encoded_df[col].value_counts()
                    encoded_df[col] = encoded_df[col].map(value_counts).fillna(0)
                    
                    action = {
                        "column": col,
                        "action": "target_encode",
                        "params": {
                            "method": "target",
                            "frequency_mapping": value_counts.to_dict()
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "request_id": request_id
                    }
                    encoding_actions.append(action)
            
            else:
                raise DataPreprocessingError(f"Unsupported encoding method: {config.categorical_encoding_method}")
            
            # Handle numeric scaling if configured
            scaler_id = None
            if config.numeric_scaling_method and config.numeric_columns:
                # Generate content hash for scaler
                content_hash = self._generate_content_hash(df, config.numeric_columns, config, "scaling")
                scaler_id = f"scaler_{config.numeric_scaling_method}_{content_hash}"
                
                # Try to load existing scaler
                scaler = self._load_scaler(scaler_id)
                
                if scaler is None:
                    # Create new scaler
                    if config.numeric_scaling_method == "standard":
                        scaler = StandardScaler()
                    elif config.numeric_scaling_method == "minmax":
                        scaler = MinMaxScaler()
                    elif config.numeric_scaling_method == "robust":
                        scaler = RobustScaler()
                    else:
                        raise DataPreprocessingError(f"Unsupported scaling method: {config.numeric_scaling_method}")
                    
                    # Fit and save scaler
                    scaler.fit(encoded_df[config.numeric_columns])
                    self._save_scaler(scaler_id, scaler)
                    logger.info(f"Created and saved new scaler | request_id={request_id} | method={config.numeric_scaling_method} | scaler_id={scaler_id}")
                else:
                    logger.info(f"Loaded existing scaler | request_id={request_id} | method={config.numeric_scaling_method} | scaler_id={scaler_id}")
                
                # Apply scaling
                encoded_df[config.numeric_columns] = scaler.transform(encoded_df[config.numeric_columns])
                
                action = {
                    "column": "numeric_columns",
                    "action": "scale",
                    "params": {
                        "method": config.numeric_scaling_method,
                        "columns": config.numeric_columns,
                        "scaler_id": scaler_id
                    },
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "request_id": request_id
                }
                encoding_actions.append(action)
            
            final_shape = encoded_df.shape
            duration_ms = (time.time() - start_time) * 1000
            metrics.observe_duration("encoding", duration_ms, {"request_id": request_id})
            
            response = EncodedDataResponse(
                data=encoded_df,
                original_shape=original_shape,
                final_shape=final_shape,
                encoding_actions=encoding_actions,
                encoders_used=config.categorical_columns,
                encoder_ids=encoder_ids,
                scaler_used=config.numeric_scaling_method,
                scaler_id=scaler_id
            )
            
            logger.info(f"Data encoding completed | request_id={request_id} | shape={original_shape}→{final_shape} | duration_ms={duration_ms:.1f}")
            return response
            
        except DataPreprocessingError:
            metrics.count("errors", {"operation": "encoding", "type": "domain", "request_id": request_id})
            raise
        except Exception as e:
            metrics.count("errors", {"operation": "encoding", "type": "unexpected", "request_id": request_id})
            logger.error(f"Data encoding failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data encoding failed: {str(e)}")


async def handle_file_upload(file: UploadFile, request_id: str) -> Tuple[pd.DataFrame, Path]:
    """
    Handle file upload by delegating to file_io service and return DataFrame with temp file path.
    This function does NOT read file content - it delegates to save_temp_file for streaming.
    """
    logger.info(f"Starting file upload handling | request_id={request_id} | filename={file.filename}")
    start_time = time.time()
    
    try:
        # Validate file extension
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.SUPPORTED_FILE_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file format: {file_extension}. Supported: {settings.SUPPORTED_FILE_EXTENSIONS}"
                )
        
        # Delegate file saving to file_io service (handles streaming and validation)
        # This will raise HTTPException (413/415) if issues occur - we don't catch these
        temp_path = await save_temp_file(file, request_id)
        
        # Determine file type and load using thread executor to avoid blocking
        loop = asyncio.get_running_loop()
        
        def _load_file() -> pd.DataFrame:
            """Load file in thread executor."""
            file_extension = Path(file.filename).suffix.lower() if file.filename else ""
            
            if file_extension == '.csv':
                return load_csv(temp_path)
            elif file_extension in ['.xlsx', '.xls']:
                return load_excel(temp_path)
            elif file_extension == '.json':
                return load_json(temp_path)
            elif file_extension == '.parquet':
                return load_parquet(temp_path)
            else:
                raise DataPreprocessingError(f"Unsupported file format: {file_extension}")
        
        try:
            df = await loop.run_in_executor(get_thread_pool(), _load_file)
            
            if df.empty:
                # Clean up temp file
                try:
                    temp_path.unlink()
                except:
                    pass
                raise DataPreprocessingError("Uploaded file contains no data or is empty")
            
            duration_ms = (time.time() - start_time) * 1000
            metrics.observe_duration("file_upload", duration_ms, {"request_id": request_id})
            
            logger.info(f"File upload completed | request_id={request_id} | shape={df.shape} | duration_ms={duration_ms:.1f}")
            return df, temp_path
            
        except Exception as load_error:
            # Clean up temp file if loading fails
            try:
                temp_path.unlink()
            except:
                pass
            raise DataPreprocessingError(f"Failed to load file content: {str(load_error)}")
        
    except HTTPException:
        # Re-raise HTTPExceptions (413, 415) unchanged to preserve status codes
        metrics.count("errors", {"operation": "file_upload", "type": "http", "request_id": request_id})
        raise
    except DataPreprocessingError:
        metrics.count("errors", {"operation": "file_upload", "type": "domain", "request_id": request_id})
        raise
    except Exception as e:
        metrics.count("errors", {"operation": "file_upload", "type": "unexpected", "request_id": request_id})
        logger.error(f"File upload handling failed | request_id={request_id} | error={str(e)}")
        raise DataPreprocessingError(f"File could not be processed: {str(e)}")


def cleanup_temp_file(temp_path: Path, request_id: str) -> None:
    """Safely clean up temporary file."""
    try:
        if temp_path and temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Cleaned up temp file | request_id={request_id} | path={temp_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file | request_id={request_id} | path={temp_path} | error={str(e)}")


async def preprocess_pipeline(
    df: pd.DataFrame,
    cleaning_config: Optional[CleaningConfig] = None,
    encoding_config: Optional[EncodingConfig] = None,
    validate_first: bool = True,
    temp_path: Optional[Path] = None,
    request_id: str = "unknown"
) -> PreprocessingResponse:
    """
    Complete preprocessing pipeline with timeout enforcement and structured responses.
    Runs CPU-bound operations in thread executor to avoid blocking the event loop.
    """
    logger.info(f"Starting preprocessing pipeline | request_id={request_id} | shape={df.shape}")
    pipeline_start = time.time()
    
    async def _run_preprocessing() -> PreprocessingResponse:
        """Inner function to run preprocessing pipeline in thread executor."""
        def _cpu_intensive_pipeline() -> PreprocessingResponse:
            """CPU-intensive pipeline operations."""
            service = DataPreprocessingService()
            audit_log = []
            
            try:
                # Generate initial cleanliness report
                initial_cleanliness_report = service.generate_cleanliness_report(df, request_id)
                logger.info(f"Initial cleanliness | request_id={request_id} | missing={sum(initial_cleanliness_report.missing_per_column.values())} | duplicates={initial_cleanliness_report.duplicate_rows}")
                
                # Validation
                validation_report = None
                if validate_first:
                    validation_report = service.validate_data(df, request_id)
                    if not validation_report.is_valid:
                        logger.warning(f"Data validation issues | request_id={request_id} | issues={validation_report.issues}")
                
                current_df = df.copy()
                
                # Cleaning
                cleaning_summary = None
                if cleaning_config:
                    cleaning_result = service.clean_data(current_df, cleaning_config, request_id)
                    current_df = cleaning_result.data
                    cleaning_summary = cleaning_result
                    audit_log.extend(cleaning_result.cleaning_actions)
                
                # Encoding
                encoding_summary = None
                if encoding_config:
                    encoding_result = service.encode_data(current_df, encoding_config, request_id)
                    current_df = encoding_result.data
                    encoding_summary = encoding_result
                    audit_log.extend(encoding_result.encoding_actions)
                
                # Generate final cleanliness report
                final_cleanliness_report = service.generate_cleanliness_report(current_df, request_id)
                
                # Generate preview
                preview_rows = current_df.head(50).to_dict(orient="records")
                
                # Calculate total duration
                total_duration = time.time() - pipeline_start
                
                logger.info(f"Preprocessing pipeline completed | request_id={request_id} | final_shape={current_df.shape} | duration_ms={total_duration*1000:.1f}")
                
                return PreprocessingResponse(
                    data=current_df,
                    num_rows=current_df.shape[0],
                    num_columns=current_df.shape[1],
                    initial_cleanliness_report=initial_cleanliness_report,
                    final_cleanliness_report=final_cleanliness_report,
                    validation_report=validation_report,
                    audit_log=audit_log,
                    preview_rows=preview_rows,
                    processing_duration_ms=round(total_duration * 1000, 1),
                    cleaning_summary=cleaning_summary,
                    encoding_summary=encoding_summary,
                    request_id=request_id
                )
                
            except (DataPreprocessingError, HTTPException):
                raise
            except Exception as e:
                logger.error(f"Preprocessing pipeline failed | request_id={request_id} | error={str(e)}")
                raise DataPreprocessingError(f"Preprocessing pipeline failed: {str(e)}")
        
        # Run CPU-intensive work in thread executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_thread_pool(), _cpu_intensive_pipeline)
    
    try:
        # Apply timeout
        timeout_seconds = getattr(settings, 'PROCESSING_TIMEOUT_SECONDS', 300)
        result = await asyncio.wait_for(_run_preprocessing(), timeout=timeout_seconds)
        
        duration_ms = (time.time() - pipeline_start) * 1000
        metrics.observe_duration("preprocessing_pipeline", duration_ms, {"request_id": request_id})
        
        return result
        
    except asyncio.TimeoutError:
        metrics.count("errors", {"operation": "preprocessing_pipeline", "type": "timeout", "request_id": request_id})
        logger.error(f"Preprocessing pipeline timed out | request_id={request_id} | timeout_seconds={timeout_seconds}")
        raise HTTPException(
            status_code=504,
            detail=f"Processing timed out after {timeout_seconds} seconds. Please try with a smaller dataset."
        )
    except (HTTPException, DataPreprocessingError):
        metrics.count("errors", {"operation": "preprocessing_pipeline", "type": "expected", "request_id": request_id})
        raise
    except Exception as e:
        metrics.count("errors", {"operation": "preprocessing_pipeline", "type": "unexpected", "request_id": request_id})
        logger.error(f"Preprocessing pipeline failed with unexpected error | request_id={request_id} | error={str(e)}")
        raise DataPreprocessingError(f"Preprocessing pipeline failed: {str(e)}")
    finally:
        # Clean up temporary file if provided
        if temp_path:
            cleanup_temp_file(temp_path, request_id)


async def preprocess_data_from_file(
    file: UploadFile,
    cleaning_config: Optional[CleaningConfig] = None,
    encoding_config: Optional[EncodingConfig] = None,
    validate_first: bool = True,
    request_id: str = "unknown"
) -> PreprocessingResponse:
    """
    Complete preprocessing pipeline starting from file upload.
    Handles file upload, then delegates to preprocessing pipeline.
    """
    logger.info(f"Starting file-based preprocessing | request_id={request_id} | filename={file.filename}")
    
    try:
        # Handle file upload (this may raise HTTPException 413/415)
        df, temp_path = await handle_file_upload(file, request_id)
        
        # Run preprocessing pipeline (temp_path will be cleaned up automatically)
        return await preprocess_pipeline(
            df=df,
            cleaning_config=cleaning_config,
            encoding_config=encoding_config,
            validate_first=validate_first,
            temp_path=temp_path,
            request_id=request_id
        )
        
    except (HTTPException, DataPreprocessingError):
        # Re-raise expected exceptions unchanged
        raise
    except Exception as e:
        logger.error(f"File-based preprocessing failed | request_id={request_id} | error={str(e)}")
        raise DataPreprocessingError(f"File processing failed: {str(e)}")


# Encoder and scaler management
class ArtifactRegistry:
    """Registry for managing persisted encoders and scalers."""
    
    def __init__(self, service: DataPreprocessingService):
        self.service = service
    
    def list_encoders(self) -> List[Dict[str, Any]]:
        """List all available encoders."""
        encoder_files = list(self.service.encoders_path.glob("*.joblib"))
        return [
            {
                "encoder_id": f.stem,
                "created": pd.Timestamp.fromtimestamp(f.stat().st_mtime).isoformat(),
                "size_kb": round(f.stat().st_size / 1024, 2),
                "type": "encoder"
            }
            for f in encoder_files
        ]
    
    def list_scalers(self) -> List[Dict[str, Any]]:
        """List all available scalers."""
        scaler_files = list(self.service.scalers_path.glob("*.joblib"))
        return [
            {
                "scaler_id": f.stem,
                "created": pd.Timestamp.fromtimestamp(f.stat().st_mtime).isoformat(),
                "size_kb": round(f.stat().st_size / 1024, 2),
                "type": "scaler"
            }
            for f in scaler_files
        ]
    
    def get_encoder_info(self, encoder_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific encoder."""
        try:
            encoder = self.service._load_encoder(encoder_id)
            if encoder is None:
                raise DataPreprocessingError(f"Encoder not found: {encoder_id}")
            
            encoder_path = self.service.encoders_path / f"{encoder_id}.joblib"
            
            return {
                "encoder_id": encoder_id,
                "classes": encoder.classes_.tolist(),
                "n_classes": len(encoder.classes_),
                "created": pd.Timestamp.fromtimestamp(encoder_path.stat().st_mtime).isoformat(),
                "size_kb": round(encoder_path.stat().st_size / 1024, 2),
                "type": "encoder"
            }
        except Exception as e:
            raise DataPreprocessingError(f"Failed to get encoder info: {str(e)}")
    
    def get_scaler_info(self, scaler_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific scaler."""
        try:
            scaler = self.service._load_scaler(scaler_id)
            if scaler is None:
                raise DataPreprocessingError(f"Scaler not found: {scaler_id}")
            
            scaler_path = self.service.scalers_path / f"{scaler_id}.joblib"
            scaler_info = {
                "scaler_id": scaler_id,
                "type": "scaler",
                "scaler_type": type(scaler).__name__,
                "created": pd.Timestamp.fromtimestamp(scaler_path.stat().st_mtime).isoformat(),
                "size_kb": round(scaler_path.stat().st_size / 1024, 2)
            }
            
            # Add scaler-specific attributes
            if hasattr(scaler, 'mean_'):
                scaler_info["mean"] = scaler.mean_.tolist()
            if hasattr(scaler, 'scale_'):
                scaler_info["scale"] = scaler.scale_.tolist()
            if hasattr(scaler, 'data_min_'):
                scaler_info["data_min"] = scaler.data_min_.tolist()
            if hasattr(scaler, 'data_max_'):
                scaler_info["data_max"] = scaler.data_max_.tolist()
            
            return scaler_info
        except Exception as e:
            raise DataPreprocessingError(f"Failed to get scaler info: {str(e)}")
    
    def delete_encoder(self, encoder_id: str) -> bool:
        """Delete a persisted encoder."""
        try:
            encoder_path = self.service.encoders_path / f"{encoder_id}.joblib"
            if encoder_path.exists():
                encoder_path.unlink()
                logger.info(f"Deleted encoder: {encoder_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete encoder {encoder_id}: {str(e)}")
            raise DataPreprocessingError(f"Failed to delete encoder: {str(e)}")
    
    def delete_scaler(self, scaler_id: str) -> bool:
        """Delete a persisted scaler."""
        try:
            scaler_path = self.service.scalers_path / f"{scaler_id}.joblib"
            if scaler_path.exists():
                scaler_path.unlink()
                logger.info(f"Deleted scaler: {scaler_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete scaler {scaler_id}: {str(e)}")
            raise DataPreprocessingError(f"Failed to delete scaler: {str(e)}")
    
    def cleanup_old_artifacts(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up artifacts older than specified days."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
        
        deleted_encoders = 0
        deleted_scalers = 0
        
        try:
            # Clean up old encoders
            for encoder_file in self.service.encoders_path.glob("*.joblib"):
                if encoder_file.stat().st_mtime < cutoff_time:
                    encoder_file.unlink()
                    deleted_encoders += 1
            
            # Clean up old scalers
            for scaler_file in self.service.scalers_path.glob("*.joblib"):
                if scaler_file.stat().st_mtime < cutoff_time:
                    scaler_file.unlink()
                    deleted_scalers += 1
            
            logger.info(f"Cleaned up old artifacts: {deleted_encoders} encoders, {deleted_scalers} scalers")
            return {"deleted_encoders": deleted_encoders, "deleted_scalers": deleted_scalers}
            
        except Exception as e:
            logger.error(f"Failed to cleanup old artifacts: {str(e)}")
            raise DataPreprocessingError(f"Artifact cleanup failed: {str(e)}")


def get_processing_stats() -> Dict[str, Any]:
    """Get current processing statistics and system health."""
    try:
        service = DataPreprocessingService()
        encoder_count = len(list(service.encoders_path.glob("*.joblib"))) if service.encoders_path.exists() else 0
        scaler_count = len(list(service.scalers_path.glob("*.joblib"))) if service.scalers_path.exists() else 0
        
        # Calculate storage usage
        encoder_storage_mb = 0
        if service.encoders_path.exists():
            encoder_storage_mb = sum(f.stat().st_size for f in service.encoders_path.glob("*.joblib")) / (1024 * 1024)
        
        scaler_storage_mb = 0
        if service.scalers_path.exists():
            scaler_storage_mb = sum(f.stat().st_size for f in service.scalers_path.glob("*.joblib")) / (1024 * 1024)
        
        return {
            "system_status": "healthy",
            "artifacts": {
                "total_encoders": encoder_count,
                "total_scalers": scaler_count,
                "encoder_storage_mb": round(encoder_storage_mb, 2),
                "scaler_storage_mb": round(scaler_storage_mb, 2)
            },
            "limits": {
                "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
                "max_rows": settings.MAX_ROWS,
                "max_columns": settings.MAX_COLUMNS,
                "processing_timeout_seconds": getattr(settings, 'PROCESSING_TIMEOUT_SECONDS', 300)
            },
            "capabilities": {
                "parallel_processing_enabled": getattr(settings, 'ENABLE_PARALLEL_PROCESSING', False),
                "max_workers": getattr(settings, 'MAX_WORKERS', 4),
                "supported_extensions": settings.SUPPORTED_FILE_EXTENSIONS
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get processing stats: {str(e)}")
        return {
            "system_status": "error",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }


# Legacy compatibility functions
async def preprocess_data(
    file: UploadFile,
    config: Optional[CleaningConfig] = None,
    request_id: str = "legacy"
) -> dict:
    """
    Legacy function for backward compatibility.
    Returns simplified dict format instead of Pydantic model.
    """
    logger.info(f"Using legacy preprocess_data function | request_id={request_id}")
    
    try:
        result = await preprocess_data_from_file(
            file=file,
            cleaning_config=config,
            request_id=request_id
        )
        
        # Convert to legacy format
        return {
            "processed_rows": result.num_rows,
            "cleaned_columns": list(result.data.columns),
            "data": result.data,
            "cleanliness_report": result.final_cleanliness_report.dict(),
            "processing_summary": {
                "duration_ms": result.processing_duration_ms,
                "actions_performed": len(result.audit_log)
            }
        }
        
    except (HTTPException, DataPreprocessingError):
        raise
    except Exception as e:
        logger.error(f"Legacy preprocess_data failed | request_id={request_id} | error={str(e)}")
        raise DataPreprocessingError(f"Legacy preprocessing failed: {str(e)}")


# Graceful shutdown
def shutdown_thread_pool() -> None:
    """Shutdown the thread pool gracefully."""
    global _thread_pool
    if _thread_pool:
        logger.info("Shutting down preprocessing thread pool")
        _thread_pool.shutdown(wait=True)
        _thread_pool = None