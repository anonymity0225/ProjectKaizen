import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from fastapi import UploadFile
from sklearn.preprocessing import LabelEncoder

from app.utils.file_io import save_temp_file
from app.schemas.preprocess import (
    CleaningConfig, 
    EncodingConfig, 
    CleanedDataResponse, 
    EncodedDataResponse,
    ValidationReport
)

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessingService:
    """Production-ready data preprocessing service with comprehensive cleaning and encoding capabilities."""
    
    def __init__(self):
        self.label_encoders = {}
        
    def validate_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate DataFrame for common data quality issues.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            ValidationReport with identified issues
        """
        start_time = time.time()
        issues = {}
        warnings = []
        
        try:
            # Check for completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                issues["null_columns"] = null_columns
                warnings.append(f"Found {len(null_columns)} columns with 100% null values")
            
            # Check for inconsistent data types
            mixed_type_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains mixed numeric/string data
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
            logger.info(f"Data validation completed in {duration:.3f}s - Found {len(issues)} issue types")
            
            return ValidationReport(
                is_valid=len(issues) == 0,
                issues=issues,
                warnings=warnings,
                row_count=len(df),
                column_count=len(df.columns),
                memory_usage_mb=round(memory_usage, 2),
                validation_duration=round(duration, 3)
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise ValueError(f"Data validation failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame, config: CleaningConfig) -> CleanedDataResponse:
        """
        Clean DataFrame based on provided configuration.
        
        Args:
            df: Input DataFrame to clean
            config: Cleaning configuration
            
        Returns:
            CleanedDataResponse with cleaning results and metadata
        """
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
                    cleaned_df = cleaned_df.drop_duplicates()
                    cleaning_actions.append({
                        "step": "remove_duplicates",
                        "affected_rows": duplicate_count,
                        "method": "drop_duplicates"
                    })
                    logger.info(f"Removed {duplicate_count} duplicate rows")
            
            # Handle missing values
            if config.handle_missing:
                # Process numeric columns
                if config.numeric_columns:
                    for col in config.numeric_columns:
                        if col not in cleaned_df.columns:
                            raise ValueError(f"Numeric column '{col}' not found in DataFrame")
                        
                        null_count = cleaned_df[col].isnull().sum()
                        if null_count > 0:
                            if config.handle_missing == "mean":
                                mean_val = cleaned_df[col].mean()
                                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                                cleaning_actions.append({
                                    "step": "fillna",
                                    "column": col,
                                    "method": "mean",
                                    "fill_value": round(mean_val, 4),
                                    "affected_rows": null_count
                                })
                            elif config.handle_missing == "median":
                                median_val = cleaned_df[col].median()
                                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                                cleaning_actions.append({
                                    "step": "fillna",
                                    "column": col,
                                    "method": "median",
                                    "fill_value": round(median_val, 4),
                                    "affected_rows": null_count
                                })
                            elif config.handle_missing == "drop":
                                before_drop = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                rows_dropped = before_drop - len(cleaned_df)
                                cleaning_actions.append({
                                    "step": "dropna",
                                    "column": col,
                                    "method": "drop",
                                    "affected_rows": rows_dropped
                                })
                
                # Process categorical columns
                if config.categorical_columns:
                    for col in config.categorical_columns:
                        if col not in cleaned_df.columns:
                            raise ValueError(f"Categorical column '{col}' not found in DataFrame")
                        
                        null_count = cleaned_df[col].isnull().sum()
                        if null_count > 0:
                            if config.handle_missing == "mode":
                                mode_values = cleaned_df[col].mode()
                                if len(mode_values) > 0:
                                    mode_val = mode_values.iloc[0]
                                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                                    cleaning_actions.append({
                                        "step": "fillna",
                                        "column": col,
                                        "method": "mode",
                                        "fill_value": str(mode_val),
                                        "affected_rows": null_count
                                    })
                            elif config.handle_missing == "drop":
                                before_drop = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(subset=[col])
                                rows_dropped = before_drop - len(cleaned_df)
                                cleaning_actions.append({
                                    "step": "dropna",
                                    "column": col,
                                    "method": "drop",
                                    "affected_rows": rows_dropped
                                })
            
            final_shape = cleaned_df.shape
            duration = time.time() - start_time
            
            logger.info(f"Data cleaning completed in {duration:.3f}s - Shape: {original_shape} → {final_shape}")
            
            return CleanedDataResponse(
                processed_rows=final_shape[0],
                original_rows=original_shape[0],
                cleaned_columns=list(cleaned_df.columns),
                rows_removed=original_shape[0] - final_shape[0],
                cleaning_actions=cleaning_actions,
                processing_duration=round(duration, 3),
                data=cleaned_df
            )
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise ValueError(f"Data cleaning failed: {str(e)}")
    
    def encode_data(self, df: pd.DataFrame, config: EncodingConfig) -> EncodedDataResponse:
        """
        Encode categorical variables using specified method.
        
        Args:
            df: Input DataFrame to encode
            config: Encoding configuration
            
        Returns:
            EncodedDataResponse with encoding results and metadata
        """
        start_time = time.time()
        original_shape = df.shape
        encoding_actions = []
        
        try:
            # Create a copy to avoid modifying original
            encoded_df = df.copy()
            
            # Validate columns exist
            for col in config.columns:
                if col not in encoded_df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
            
            if config.method == "onehot":
                # One-hot encoding
                original_columns = set(encoded_df.columns)
                encoded_df = pd.get_dummies(encoded_df, columns=config.columns, prefix=config.columns)
                new_columns = list(set(encoded_df.columns) - original_columns)
                
                for col in config.columns:
                    unique_values = df[col].nunique()
                    col_new_columns = [c for c in new_columns if c.startswith(f"{col}_")]
                    encoding_actions.append({
                        "step": "onehot_encode",
                        "column": col,
                        "method": "onehot",
                        "unique_values": unique_values,
                        "new_columns": col_new_columns,
                        "columns_created": len(col_new_columns)
                    })
                
            elif config.method == "label":
                # Label encoding
                for col in config.columns:
                    unique_values = encoded_df[col].nunique()
                    le = LabelEncoder()
                    
                    # Handle NaN values by creating a mask
                    mask = encoded_df[col].notna()
                    if mask.sum() > 0:  # Only encode if there are non-null values
                        encoded_df.loc[mask, col] = le.fit_transform(encoded_df.loc[mask, col])
                        
                        # Store encoder for potential inverse transform
                        self.label_encoders[col] = le
                        
                        encoding_actions.append({
                            "step": "label_encode",
                            "column": col,
                            "method": "label",
                            "unique_values": unique_values,
                            "encoded_range": f"0-{len(le.classes_)-1}",
                            "classes": le.classes_.tolist() if len(le.classes_) <= 20 else f"{len(le.classes_)} classes"
                        })
                    
            elif config.method == "frequency":
                # Frequency encoding
                for col in config.columns:
                    value_counts = encoded_df[col].value_counts()
                    encoded_df[col] = encoded_df[col].map(value_counts)
                    
                    encoding_actions.append({
                        "step": "frequency_encode",
                        "column": col,
                        "method": "frequency",
                        "unique_values": len(value_counts),
                        "frequency_range": f"{value_counts.min()}-{value_counts.max()}"
                    })
                    
            else:
                raise ValueError(f"Unsupported encoding method: {config.method}")
            
            final_shape = encoded_df.shape
            duration = time.time() - start_time
            
            logger.info(f"Data encoding completed in {duration:.3f}s - Shape: {original_shape} → {final_shape}")
            
            return EncodedDataResponse(
                original_shape=original_shape,
                encoded_shape=final_shape,
                columns_encoded=config.columns,
                encoding_method=config.method,
                encoding_actions=encoding_actions,
                processing_duration=round(duration, 3),
                data=encoded_df
            )
            
        except Exception as e:
            logger.error(f"Data encoding failed: {str(e)}")
            raise ValueError(f"Data encoding failed: {str(e)}")
    
    def audit_cleaning_actions(self, cleaning_actions: List[Dict], encoding_actions: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit log for UI step tracking.
        
        Args:
            cleaning_actions: List of cleaning actions performed
            encoding_actions: Optional list of encoding actions performed
            
        Returns:
            Dictionary with audit information for frontend
        """
        try:
            all_actions = cleaning_actions.copy()
            if encoding_actions:
                all_actions.extend(encoding_actions)
            
            # Summarize actions
            action_summary = {}
            for action in all_actions:
                step_type = action["step"]
                if step_type not in action_summary:
                    action_summary[step_type] = {
                        "count": 0,
                        "total_affected_rows": 0,
                        "columns": []
                    }
                
                action_summary[step_type]["count"] += 1
                if "affected_rows" in action:
                    action_summary[step_type]["total_affected_rows"] += action["affected_rows"]
                if "column" in action:
                    action_summary[step_type]["columns"].append(action["column"])
            
            return {
                "total_steps": len(all_actions),
                "step_details": all_actions,
                "action_summary": action_summary,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Audit generation failed: {str(e)}")
            return {
                "error": f"Audit generation failed: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }


# FastAPI wrapper functions (keeping FastAPI concerns separate)
async def handle_file_upload(file: UploadFile) -> pd.DataFrame:
    """
    Handle file upload and return DataFrame.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Loaded DataFrame
    """
    try:
        # Save uploaded file temporarily
        temp_path = await save_temp_file(file)
        
        # Determine file type and read accordingly
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(temp_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if df.empty:
            raise ValueError("Uploaded file is empty")
        
        logger.info(f"Successfully loaded file: {file.filename} - Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"File upload handling failed: {str(e)}")
        raise ValueError(f"File could not be read: {str(e)}")


# Convenience functions for direct usage
def preprocess_pipeline(
    df: pd.DataFrame, 
    cleaning_config: Optional[CleaningConfig] = None,
    encoding_config: Optional[EncodingConfig] = None,
    validate_first: bool = True
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline combining validation, cleaning, and encoding.
    
    Args:
        df: Input DataFrame
        cleaning_config: Optional cleaning configuration
        encoding_config: Optional encoding configuration
        validate_first: Whether to validate data first
        
    Returns:
        Dictionary with all processing results
    """
    service = DataPreprocessingService()
    results = {}
    
    try:
        # Validation
        if validate_first:
            results["validation"] = service.validate_data(df)
        
        # Cleaning
        if cleaning_config:
            results["cleaning"] = service.clean_data(df, cleaning_config)
            df = results["cleaning"].data
        
        # Encoding
        if encoding_config:
            results["encoding"] = service.encode_data(df, encoding_config)
            df = results["encoding"].data
        
        # Generate audit
        cleaning_actions = results.get("cleaning", {}).get("cleaning_actions", []) if "cleaning" in results else []
        encoding_actions = results.get("encoding", {}).get("encoding_actions", []) if "encoding" in results else []
        results["audit"] = service.audit_cleaning_actions(cleaning_actions, encoding_actions)
        
        results["final_data"] = df
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise ValueError(f"Preprocessing pipeline failed: {str(e)}")


# Legacy function for backward compatibility
async def preprocess_data(file: UploadFile, config: CleaningConfig | None = None) -> dict:
    """
    Legacy function for backward compatibility.
    """
    try:
        df = await handle_file_upload(file)
        service = DataPreprocessingService()
        
        if config:
            result = service.clean_data(df, config)
            return {
                "processed_rows": result.processed_rows,
                "cleaned_columns": result.cleaned_columns,
                "data": result.data
            }
        else:
            return {
                "processed_rows": len(df),
                "cleaned_columns": list(df.columns),
                "data": df
            }
            
    except Exception as e:
        logger.error(f"Legacy preprocess_data failed: {str(e)}")
        raise ValueError(f"Data preprocessing failed: {str(e)}")