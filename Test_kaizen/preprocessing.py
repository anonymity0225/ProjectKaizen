import json
import logging
import time
import hashlib
import tempfile
import shutil
import uuid
import os
import atexit
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import HTTPException
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder

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

# Configure logging
logger = logging.getLogger(__name__)

# Schema version for encoder/scaler compatibility
ENCODER_SCHEMA_VERSION = "1.0.0"
SCALER_SCHEMA_VERSION = "1.0.0"

# ==========================================
# SIMPLIFIED COMPONENTS FOR MVP 1
# ==========================================

# Simple thread pool management
_thread_pool = None
_thread_pool_lock = threading.RLock()

def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create simple thread pool."""
    global _thread_pool
    if _thread_pool is None:
        with _thread_pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(
                    max_workers=min(4, (os.cpu_count() or 1) + 1)
                )
    return _thread_pool

def check_memory_availability(required_mb: float = 500) -> bool:
    """Simple memory check."""
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 * 1024)
        return available > required_mb
    except ImportError:
        return True  # Skip check if psutil not available

class SimpleEncoderCache:
    """Simple dict-based encoder cache with LRU eviction."""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value

# Global simple cache
_simple_cache = SimpleEncoderCache()

# Thread pool cleanup
def cleanup_resources():
    """Clean up resources on shutdown."""
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None

atexit.register(cleanup_resources)

class DataPreprocessingError(Exception):
    """Raised when preprocessing validation or operations fail."""
    pass

class DataPreprocessingService:
    """Enterprise-grade data preprocessing service with comprehensive cleaning and encoding capabilities."""
    
    def __init__(self) -> None:
        """Initialize the preprocessing service with encoder persistence support."""
        self.encoders_path = Path("models/encoders")
        self.scalers_path = Path("models/scalers")
        self.metadata_path = Path("models/metadata")
        
        # Create directories
        for path in [self.encoders_path, self.scalers_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def detect_semantic_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect semantic types for each column."""
        semantic_types = {}
        for col in df.columns:
            series = df[col]
            
            # ID detection
            if (col.lower() in ['id', 'uuid', 'guid', 'key'] or 
                (series.dtype == 'object' and series.nunique() == len(series))):
                semantic_types[col] = 'identifier'
            
            # Date/time detection
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(series.dropna().head(100))
                    semantic_types[col] = 'datetime'
                except:
                    semantic_types[col] = 'text'
            
            # Categorical detection
            elif series.dtype == 'object' and series.nunique() < len(series) * 0.5:
                semantic_types[col] = 'categorical'
            
            # Numeric detection
            elif pd.api.types.is_numeric_dtype(series):
                semantic_types[col] = 'numeric'
            
            else:
                semantic_types[col] = 'text'
        
        return semantic_types

    def compare_cleaning_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare different cleaning strategies and their impact."""
        strategies_impact = {}
        
        # Drop rows strategy
        drop_result = df.dropna()
        strategies_impact['drop_rows'] = {
            'rows_retained': len(drop_result),
            'data_loss_pct': (1 - len(drop_result)/len(df)) * 100,
            'columns_affected': 0
        }
        
        # Mean imputation strategy
        mean_result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_result[col].fillna(mean_result[col].mean(), inplace=True)
        strategies_impact['mean_impute'] = {
            'rows_retained': len(mean_result),
            'data_loss_pct': 0,
            'columns_affected': len(numeric_cols)
        }
        
        # Median imputation strategy
        median_result = df.copy()
        for col in numeric_cols:
            median_result[col].fillna(median_result[col].median(), inplace=True)
        strategies_impact['median_impute'] = {
            'rows_retained': len(median_result),
            'data_loss_pct': 0,
            'columns_affected': len(numeric_cols)
        }
        
        # Add recommendation
        if strategies_impact['drop_rows']['data_loss_pct'] < 5:
            recommendation = 'drop_rows'
        else:
            recommendation = 'median_impute'
        
        return {
            'strategies': strategies_impact,
            'recommendation': recommendation,
            'reasoning': f"Based on data loss vs quality trade-off"
        }

    def append_to_dataset(self, dataset_id: str, new_df: pd.DataFrame, request_id: str) -> pd.DataFrame:
        """Support incremental uploads to existing dataset."""
        # Load existing dataset (from cache or storage)
        existing_path = Path(f"datasets/{dataset_id}.parquet")
        
        if existing_path.exists():
            existing_df = pd.read_parquet(existing_path)
            
            # Validate schema compatibility
            if set(new_df.columns) != set(existing_df.columns):
                raise DataPreprocessingError("Column mismatch for incremental upload")
            
            # Append new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save updated dataset
            combined_df.to_parquet(existing_path)
            
            logger.info(f"Incremental upload | dataset_id={dataset_id} | added_rows={len(new_df)} | total_rows={len(combined_df)}")
            return combined_df
        else:
            # First upload, create new dataset
            os.makedirs("datasets", exist_ok=True)
            new_df.to_parquet(existing_path)
            return new_df

    def generate_cleanliness_report(self, df: pd.DataFrame, request_id: str) -> CleanlinessReport:
        """Enhanced cleanliness report with strategy comparison."""
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
            
            # Add strategy comparison
            strategy_comparison = self.compare_cleaning_strategies(df)
            
            # Add semantic type detection
            semantic_types = self.detect_semantic_types(df)
            
            # Generate dataset hash for versioning
            dataset_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df, index=False).values
            ).hexdigest()[:16]
            
            # Add warnings
            warnings = []
            if duplicate_rows > total_rows * 0.1:
                warnings.append(f"High duplicate rate: {duplicate_rows/total_rows:.1%}")
            if any(ratio > 0.5 for ratio in missing_per_column.values()):
                warnings.append("High missing value ratio detected")
            
            report = CleanlinessReport(
                total_rows=total_rows,
                total_columns=total_cols,
                missing_per_column=missing_per_column,
                duplicate_rows=duplicate_rows,
                column_types=column_types,
                categorical_cardinality=categorical_cardinality,
                semantic_types=semantic_types,
                strategy_comparison=strategy_comparison,
                dataset_hash=dataset_hash,
                warnings=warnings
            )
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Cleanliness report completed | request_id={request_id} | duration_ms={duration_ms:.1f}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate cleanliness report | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Cleanliness report generation failed: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, request_id: str) -> ValidationReport:
        """Validate DataFrame against enterprise data quality standards."""
        logger.info(f"Starting data validation | request_id={request_id} | shape={df.shape}")
        start_time = time.time()
        errors = []
        warnings = []
        details = {}
        
        try:
            # Enforce system limits
            max_rows = getattr(settings, 'MAX_ROWS', 1000000)
            max_columns = getattr(settings, 'MAX_COLUMNS', 1000)
            missing_threshold = getattr(settings, 'MISSING_VALUE_THRESHOLD', 0.8)
            
            if len(df) > max_rows:
                errors.append(f"Dataset exceeds maximum allowed rows: {len(df)} > {max_rows}")
            
            if df.shape[1] > max_columns:
                errors.append(f"Dataset exceeds maximum allowed columns: {df.shape[1]} > {max_columns}")
            
            # Check missing value threshold
            missing_pct = df.isna().mean().max()
            if missing_pct > missing_threshold:
                errors.append(f"Dataset exceeds missing value threshold: {missing_pct:.2%} > {missing_threshold:.2%}")
            
            # Check for completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                errors.append(f"Found {len(null_columns)} columns with 100% null values")
                details["null_only_columns"] = null_columns
            
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
                warnings.append(f"Found {len(high_cardinality_columns)} high cardinality categorical columns")
                details["high_cardinality_columns"] = high_cardinality_columns
            
            duration_ms = (time.time() - start_time) * 1000
            
            validation_report = ValidationReport(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
            logger.info(f"Data validation completed | request_id={request_id} | valid={validation_report.valid} | duration_ms={duration_ms:.1f}")
            return validation_report
            
        except Exception as e:
            logger.error(f"Data validation failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data validation failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame, config: CleaningConfig, request_id: str) -> CleanedDataResponse:
        """Clean DataFrame using enterprise-grade processing."""
        logger.info(f"Starting data cleaning | request_id={request_id} | shape={df.shape}")
        start_time = time.time()
        original_shape = df.shape
        cleaning_actions = []
        
        try:
            # Create a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Remove duplicates
            if config.drop_duplicates:
                duplicate_count = cleaned_df.duplicated().sum()
                if duplicate_count > 0:
                    before_count = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates()
                    after_count = len(cleaned_df)
                    
                    action = {
                        "action": "remove_duplicates",
                        "params": {"rows_removed": before_count - after_count},
                        "timestamp": datetime.now().isoformat()
                    }
                    cleaning_actions.append(action)
                    logger.info(f"Removed {before_count - after_count} duplicate rows | request_id={request_id}")
            
            # Handle missing values
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
            
            if config.missing_strategy == "mean" and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        mean_val = cleaned_df[col].mean()
                        cleaned_df[col].fillna(mean_val, inplace=True)
                        cleaning_actions.append({
                            "action": "fillna_mean",
                            "column": col,
                            "value": float(mean_val),
                            "timestamp": datetime.now().isoformat()
                        })
            
            elif config.missing_strategy == "median" and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        cleaned_df[col].fillna(median_val, inplace=True)
                        cleaning_actions.append({
                            "action": "fillna_median",
                            "column": col,
                            "value": float(median_val),
                            "timestamp": datetime.now().isoformat()
                        })
            
            elif config.missing_strategy == "mode":
                for col in categorical_cols:
                    if cleaned_df[col].isnull().any():
                        mode_values = cleaned_df[col].mode()
                        if len(mode_values) > 0:
                            mode_val = mode_values.iloc[0]
                            cleaned_df[col].fillna(mode_val, inplace=True)
                            cleaning_actions.append({
                                "action": "fillna_mode",
                                "column": col,
                                "value": str(mode_val),
                                "timestamp": datetime.now().isoformat()
                            })
            
            elif config.missing_strategy == "drop":
                before_count = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                rows_dropped = before_count - len(cleaned_df)
                if rows_dropped > 0:
                    cleaning_actions.append({
                        "action": "dropna",
                        "params": {"rows_dropped": rows_dropped},
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Handle outliers
            if config.outlier_handling != "none":
                for col in numeric_cols:
                    if config.outlier_handling == "iqr":
                        Q1 = cleaned_df[col].quantile(0.25)
                        Q3 = cleaned_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - config.outlier_threshold * IQR
                        upper = Q3 + config.outlier_threshold * IQR
                        
                        outliers = cleaned_df[(cleaned_df[col] < lower) | (cleaned_df[col] > upper)]
                        if len(outliers) > 0:
                            if config.outlier_action == "remove":
                                cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
                            elif config.outlier_action == "cap":
                                cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                            
                            cleaning_actions.append({
                                "action": f"outlier_{config.outlier_action}",
                                "column": col,
                                "method": "iqr",
                                "outliers_found": len(outliers),
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Normalize column names
            if config.normalize_column_names:
                old_columns = cleaned_df.columns.tolist()
                new_columns = [col.lower().replace(' ', '_') for col in old_columns]
                if old_columns != new_columns:
                    cleaned_df.columns = new_columns
                    cleaning_actions.append({
                        "action": "normalize_column_names",
                        "old_columns": old_columns,
                        "new_columns": new_columns,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Remove empty columns
            if config.remove_empty_columns:
                empty_cols = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
                if empty_cols:
                    cleaned_df = cleaned_df.drop(columns=empty_cols)
                    cleaning_actions.append({
                        "action": "remove_empty_columns",
                        "columns_removed": empty_cols,
                        "timestamp": datetime.now().isoformat()
                    })
            
            final_shape = cleaned_df.shape
            
            return CleanedDataResponse(
                data=cleaned_df.to_dict('records'),
                original_shape=original_shape,
                final_shape=final_shape,
                cleaning_actions=cleaning_actions,
                num_rows_removed=original_shape[0] - final_shape[0],
                num_columns_removed=original_shape[1] - final_shape[1]
            )
            
        except Exception as e:
            logger.error(f"Data cleaning failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data cleaning failed: {str(e)}")
    
    def encode_data(self, data: Dict[str, Any], config: EncodingConfig, request_id: str) -> EncodedDataResponse:
        """Encode categorical data using various methods."""
        logger.info(f"Starting data encoding | request_id={request_id}")
        start_time = time.time()
        
        try:
            # Convert back to DataFrame
            df = pd.DataFrame(data)
            original_shape = df.shape
            encoding_actions = []
            encoders_used = []
            encoder_ids = {}
            
            # Auto-detect categorical columns if not specified
            categorical_columns = config.categorical_columns
            if not categorical_columns:
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Apply encoding based on method
            if config.method == "label":
                for col in categorical_columns:
                    if col in df.columns:
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        df[col] = encoder.fit_transform(df[[col]]).flatten()
                        
                        # Cache encoder
                        encoder_id = f"label_{col}_{hashlib.md5(str(df[col].unique()).encode()).hexdigest()[:8]}"
                        _simple_cache.put(encoder_id, encoder)
                        encoder_ids[col] = encoder_id
                        
                        encoders_used.append("OrdinalEncoder")
                        encoding_actions.append({
                            "action": "label_encode",
                            "column": col,
                            "encoder_id": encoder_id,
                            "timestamp": datetime.now().isoformat()
                        })
            
            elif config.method == "onehot":
                for col in categorical_columns:
                    if col in df.columns and df[col].nunique() <= config.max_categories_for_onehot:
                        # Create one-hot encoding
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = df.drop(columns=[col])
                        df = pd.concat([df, dummies], axis=1)
                        
                        encoders_used.append("OneHotEncoder")
                        encoding_actions.append({
                            "action": "onehot_encode",
                            "column": col,
                            "new_columns": dummies.columns.tolist(),
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Apply scaling if requested
            scaler_used = None
            scaler_id = None
            if config.scale_numerical:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    if config.scaling_method == "standard":
                        scaler = StandardScaler()
                        scaler_used = "StandardScaler"
                    elif config.scaling_method == "minmax":
                        scaler = MinMaxScaler()
                        scaler_used = "MinMaxScaler"
                    elif config.scaling_method == "robust":
                        scaler = RobustScaler()
                        scaler_used = "RobustScaler"
                    
                    if scaler:
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        
                        # Cache scaler
                        scaler_id = f"scaler_{config.scaling_method}_{hashlib.md5(str(numeric_cols).encode()).hexdigest()[:8]}"
                        _simple_cache.put(scaler_id, scaler)
                        
                        encoding_actions.append({
                            "action": "scale_features",
                            "method": config.scaling_method,
                            "columns": numeric_cols,
                            "scaler_id": scaler_id,
                            "timestamp": datetime.now().isoformat()
                        })
            
            final_shape = df.shape
            
            return EncodedDataResponse(
                data=df.to_dict('records'),
                original_shape=original_shape,
                final_shape=final_shape,
                encoding_actions=encoding_actions,
                encoders_used=list(set(encoders_used)),
                encoder_ids=encoder_ids,
                scaler_used=scaler_used,
                scaler_id=scaler_id
            )
            
        except Exception as e:
            logger.error(f"Data encoding failed | request_id={request_id} | error={str(e)}")
            raise DataPreprocessingError(f"Data encoding failed: {str(e)}")


# Helper functions for backward compatibility
def handle_file_upload(file, request_id: str = None):
    """Handle file upload and return DataFrame."""
    service = DataPreprocessingService()
    # This would contain file loading logic
    pass

def preprocess_pipeline(df: pd.DataFrame, cleaning_config: CleaningConfig, 
                       encoding_config: EncodingConfig, validate_first: bool = True, 
                       request_id: str = None) -> Dict[str, Any]:
    """Run complete preprocessing pipeline."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    service = DataPreprocessingService()
    
    # Initial report
    initial_report = service.generate_cleanliness_report(df, request_id)
    
    # Validation
    validation_report = None
    if validate_first:
        validation_report = service.validate_data(df, request_id)
        if not validation_report.valid:
            raise DataPreprocessingError(f"Validation failed: {validation_report.errors}")
    
    # Cleaning
    cleaned_response = service.clean_data(df, cleaning_config, request_id)
    
    # Encoding
    encoded_response = service.encode_data(cleaned_response.data, encoding_config, request_id)
    
    # Final report
    final_df = pd.DataFrame(encoded_response.data)
    final_report = service.generate_cleanliness_report(final_df, request_id)
    
    # Build audit log
    audit_log = []
    audit_log.extend(cleaned_response.cleaning_actions)
    audit_log.extend(encoded_response.encoding_actions)
    
    return {
        "initial_cleanliness_report": initial_report.dict(),
        "final_cleanliness_report": final_report.dict(),
        "validation_report": validation_report.dict() if validation_report else None,
        "audit_log": audit_log,
        "preview_rows": final_df.head(50).to_dict('records'),
        "num_rows": final_df.shape[0],
        "num_columns": final_df.shape[1],
        "dataset_hash": final_report.dataset_hash,
        "cleaning_summary": cleaned_response.dict(),
        "encoding_summary": encoded_response.dict()
    }
