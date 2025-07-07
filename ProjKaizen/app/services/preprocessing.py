# Preprocessing service logic 
from fastapi import UploadFile, HTTPException
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from ..utils.file_io import save_temp_file, load_csv, load_excel
from ..schemas.preprocess import (
    CleaningConfig, EncodingConfig, CleanedDataResponse, EncodedDataResponse, ValidationReport,
    PreprocessingRequest, PreprocessingResponse, FileUploadResponse
)

logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FILE_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
DEFAULT_MISSING_VALUE_THRESHOLD = 0.5  # 50% missing values threshold
DEFAULT_DUPLICATE_THRESHOLD = 0.9  # 90% duplicate threshold
MAX_CATEGORIES_FOR_ONEHOT = 10  # Maximum categories for one-hot encoding


class DataPreprocessingService:
    """
    Service class for data preprocessing operations including validation,
    cleaning, and encoding of datasets.
    """

    def __init__(self):
        """Initialize the DataPreprocessingService."""
        self.scaler = None
        self.encoders = {}
        self.cleaning_log = []
        self.encoding_log = []

    def validate_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate the input DataFrame and return a validation report.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
            
        Returns:
            ValidationReport: Validation report containing issues and recommendations
        """
        try:
            issues = []
            recommendations = []
            
            # Check for empty DataFrame
            if df.empty:
                issues.append("Dataset is empty")
                return ValidationReport(
                    is_valid=False,
                    issues=issues,
                    recommendations=["Upload a valid dataset with data"],
                    num_rows=0,
                    num_columns=0,
                    missing_values_summary={},
                    duplicate_rows=0
                )
            
            # Basic statistics
            num_rows, num_columns = df.shape
            duplicate_rows = df.duplicated().sum()
            
            # Missing values analysis
            missing_values_summary = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                if missing_count > 0:
                    missing_values_summary[col] = {
                        'count': int(missing_count),
                        'percentage': round(missing_pct, 2)
                    }
                    
                    if missing_pct > DEFAULT_MISSING_VALUE_THRESHOLD * 100:
                        issues.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                        recommendations.append(f"Consider dropping column '{col}' or imputing missing values")
            
            # Duplicate rows check
            if duplicate_rows > 0:
                duplicate_pct = (duplicate_rows / len(df)) * 100
                issues.append(f"Found {duplicate_rows} duplicate rows ({duplicate_pct:.1f}%)")
                recommendations.append("Remove duplicate rows to improve data quality")
            
            # Data type consistency check
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for mixed types in object columns
                    sample_values = df[col].dropna().head(100)
                    if len(sample_values) > 0:
                        value_types = set(type(v).__name__ for v in sample_values)
                        if len(value_types) > 1:
                            issues.append(f"Column '{col}' contains mixed data types: {value_types}")
                            recommendations.append(f"Standardize data types in column '{col}'")
            
            # Check for columns with single unique value
            for col in df.columns:
                if df[col].nunique() <= 1:
                    issues.append(f"Column '{col}' has only one unique value")
                    recommendations.append(f"Consider removing column '{col}' as it provides no information")
            
            is_valid = len(issues) == 0
            
            return ValidationReport(
                is_valid=is_valid,
                issues=issues,
                recommendations=recommendations,
                num_rows=num_rows,
                num_columns=num_columns,
                missing_values_summary=missing_values_summary,
                duplicate_rows=duplicate_rows
            )
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data validation failed: {str(e)}")

    def clean_data(self, df: pd.DataFrame, config: CleaningConfig) -> CleanedDataResponse:
        """
        Clean the DataFrame based on the provided configuration.
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean
            config (CleaningConfig): Cleaning configuration parameters
            
        Returns:
            CleanedDataResponse: Response containing cleaned data and cleaning actions
        """
        try:
            cleaned_df = df.copy()
            cleaning_actions = []
            original_shape = cleaned_df.shape
            
            # Remove duplicates
            if config.remove_duplicates:
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_duplicates = initial_rows - len(cleaned_df)
                if removed_duplicates > 0:
                    cleaning_actions.append(f"Removed {removed_duplicates} duplicate rows")
                    logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            if config.handle_missing_values:
                for col in cleaned_df.columns:
                    missing_count = cleaned_df[col].isnull().sum()
                    if missing_count > 0:
                        missing_pct = (missing_count / len(cleaned_df)) * 100
                        
                        if config.missing_value_strategy == "drop_rows":
                            initial_rows = len(cleaned_df)
                            cleaned_df = cleaned_df.dropna(subset=[col])
                            dropped_rows = initial_rows - len(cleaned_df)
                            if dropped_rows > 0:
                                cleaning_actions.append(f"Dropped {dropped_rows} rows with missing values in '{col}'")
                        
                        elif config.missing_value_strategy == "drop_columns":
                            if missing_pct > (config.missing_threshold * 100):
                                cleaned_df = cleaned_df.drop(columns=[col])
                                cleaning_actions.append(f"Dropped column '{col}' (missing: {missing_pct:.1f}%)")
                        
                        elif config.missing_value_strategy == "fill_mean":
                            if cleaned_df[col].dtype in ['int64', 'float64']:
                                mean_value = cleaned_df[col].mean()
                                cleaned_df[col] = cleaned_df[col].fillna(mean_value)
                                cleaning_actions.append(f"Filled {missing_count} missing values in '{col}' with mean ({mean_value:.2f})")
                        
                        elif config.missing_value_strategy == "fill_median":
                            if cleaned_df[col].dtype in ['int64', 'float64']:
                                median_value = cleaned_df[col].median()
                                cleaned_df[col] = cleaned_df[col].fillna(median_value)
                                cleaning_actions.append(f"Filled {missing_count} missing values in '{col}' with median ({median_value:.2f})")
                        
                        elif config.missing_value_strategy == "fill_mode":
                            mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "Unknown"
                            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                            cleaning_actions.append(f"Filled {missing_count} missing values in '{col}' with mode ('{mode_value}')")
                        
                        elif config.missing_value_strategy == "fill_forward":
                            # Use pandas 2.0+ method
                            cleaned_df[col] = cleaned_df[col].fillna(method='ffill') if hasattr(cleaned_df[col], 'fillna') else cleaned_df[col].ffill()
                            cleaning_actions.append(f"Forward filled {missing_count} missing values in '{col}'")
                        
                        elif config.missing_value_strategy == "fill_backward":
                            # Use pandas 2.0+ method
                            cleaned_df[col] = cleaned_df[col].fillna(method='bfill') if hasattr(cleaned_df[col], 'fillna') else cleaned_df[col].bfill()
                            cleaning_actions.append(f"Backward filled {missing_count} missing values in '{col}'")
            
            # Remove outliers (using IQR method for numeric columns)
            if config.remove_outliers:
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    initial_rows = len(cleaned_df)
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
                    removed_outliers = initial_rows - len(cleaned_df)
                    
                    if removed_outliers > 0:
                        cleaning_actions.append(f"Removed {removed_outliers} outliers from '{col}'")
            
            # Normalize column names
            if config.normalize_column_names:
                original_columns = cleaned_df.columns.tolist()
                cleaned_df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in cleaned_df.columns]
                new_columns = cleaned_df.columns.tolist()
                
                renamed_columns = [(orig, new) for orig, new in zip(original_columns, new_columns) if orig != new]
                if renamed_columns:
                    cleaning_actions.append(f"Normalized {len(renamed_columns)} column names")
            
            final_shape = cleaned_df.shape
            self.cleaning_log = cleaning_actions
            
            return CleanedDataResponse(
                data=cleaned_df,
                original_shape=original_shape,
                final_shape=final_shape,
                cleaning_actions=cleaning_actions,
                num_rows_removed=original_shape[0] - final_shape[0],
                num_columns_removed=original_shape[1] - final_shape[1]
            )
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")

    def encode_data(self, df: pd.DataFrame, config: EncodingConfig) -> EncodedDataResponse:
        """
        Encode categorical variables and scale numerical features based on configuration.
        
        Args:
            df (pd.DataFrame): Input DataFrame to encode
            config (EncodingConfig): Encoding configuration parameters
            
        Returns:
            EncodedDataResponse: Response containing encoded data and encoding actions
        """
        try:
            encoded_df = df.copy()
            encoding_actions = []
            original_shape = encoded_df.shape
            
            # Categorical encoding
            categorical_columns = encoded_df.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_columns:
                if col in config.categorical_columns or not config.categorical_columns:
                    unique_values = encoded_df[col].nunique()
                    
                    if config.categorical_encoding_method == "label":
                        le = LabelEncoder()
                        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                        self.encoders[col] = le
                        encoding_actions.append(f"Label encoded '{col}' ({unique_values} categories)")
                    
                    elif config.categorical_encoding_method == "onehot":
                        if unique_values <= MAX_CATEGORIES_FOR_ONEHOT:
                            # Use pandas get_dummies for one-hot encoding
                            dummies = pd.get_dummies(encoded_df[col], prefix=col, drop_first=config.drop_first)
                            encoded_df = pd.concat([encoded_df.drop(columns=[col]), dummies], axis=1)
                            encoding_actions.append(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")
                        else:
                            # Fall back to label encoding for high cardinality
                            le = LabelEncoder()
                            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                            self.encoders[col] = le
                            encoding_actions.append(f"Label encoded '{col}' (too many categories for one-hot: {unique_values})")
                    
                    elif config.categorical_encoding_method == "target":
                        # Simple target encoding (mean of target for each category)
                        if config.target_column and config.target_column in encoded_df.columns:
                            target_means = encoded_df.groupby(col)[config.target_column].mean()
                            encoded_df[col] = encoded_df[col].map(target_means)
                            encoding_actions.append(f"Target encoded '{col}' using '{config.target_column}'")
                        else:
                            # Fall back to label encoding if no target specified
                            le = LabelEncoder()
                            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                            self.encoders[col] = le
                            encoding_actions.append(f"Label encoded '{col}' (no target column for target encoding)")
            
            # Numerical scaling
            numerical_columns = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if config.scale_numerical and numerical_columns:
                columns_to_scale = [col for col in numerical_columns 
                                 if col in config.numerical_columns or not config.numerical_columns]
                
                if config.target_column and config.target_column in columns_to_scale:
                    columns_to_scale.remove(config.target_column)
                
                if columns_to_scale:
                    if config.scaling_method == "standard":
                        scaler = StandardScaler()
                        encoded_df[columns_to_scale] = scaler.fit_transform(encoded_df[columns_to_scale])
                        self.scaler = scaler
                        encoding_actions.append(f"Standard scaled {len(columns_to_scale)} numerical columns")
                    
                    elif config.scaling_method == "minmax":
                        scaler = MinMaxScaler()
                        encoded_df[columns_to_scale] = scaler.fit_transform(encoded_df[columns_to_scale])
                        self.scaler = scaler
                        encoding_actions.append(f"MinMax scaled {len(columns_to_scale)} numerical columns")
                    
                    elif config.scaling_method == "robust":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        encoded_df[columns_to_scale] = scaler.fit_transform(encoded_df[columns_to_scale])
                        self.scaler = scaler
                        encoding_actions.append(f"Robust scaled {len(columns_to_scale)} numerical columns")
            
            final_shape = encoded_df.shape
            self.encoding_log = encoding_actions
            
            return EncodedDataResponse(
                data=encoded_df,
                original_shape=original_shape,
                final_shape=final_shape,
                encoding_actions=encoding_actions,
                encoders_used=list(self.encoders.keys()),
                scaler_used=config.scaling_method if config.scale_numerical else None
            )
            
        except Exception as e:
            logger.error(f"Data encoding failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data encoding failed: {str(e)}")

    def audit_cleaning_actions(self, cleaning_actions: List[str], encoding_actions: List[str]) -> Dict[str, Any]:
        """
        Create an audit log of all preprocessing actions performed.
        
        Args:
            cleaning_actions (List[str]): List of cleaning actions performed
            encoding_actions (List[str]): List of encoding actions performed
            
        Returns:
            Dict[str, Any]: Comprehensive audit log with timestamps and action details
        """
        audit_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preprocessing_summary": {
                "total_cleaning_actions": len(cleaning_actions),
                "total_encoding_actions": len(encoding_actions),
                "total_actions": len(cleaning_actions) + len(encoding_actions)
            },
            "cleaning_actions": cleaning_actions,
            "encoding_actions": encoding_actions,
            "transformers_used": {
                "encoders": list(self.encoders.keys()) if self.encoders else [],
                "scaler": type(self.scaler).__name__ if self.scaler else None
            }
        }
        
        return audit_log

    def create_snapshot(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create a JSON-serializable snapshot of the DataFrame for frontend preview.
        
        Args:
            df (pd.DataFrame): DataFrame to snapshot
            
        Returns:
            List[Dict[str, Any]]: First 5 rows as JSON-serializable records
        """
        try:
            # Create a copy and handle potential non-serializable data types
            snapshot_df = df.head(5).copy()
            
            # Convert any datetime columns to strings
            for col in snapshot_df.columns:
                if pd.api.types.is_datetime64_any_dtype(snapshot_df[col]):
                    snapshot_df[col] = snapshot_df[col].astype(str)
                # Convert numpy types to native Python types for JSON serialization
                elif snapshot_df[col].dtype == 'object':
                    snapshot_df[col] = snapshot_df[col].astype(str)
                elif 'int' in str(snapshot_df[col].dtype):
                    snapshot_df[col] = snapshot_df[col].astype(int)
                elif 'float' in str(snapshot_df[col].dtype):
                    snapshot_df[col] = snapshot_df[col].astype(float)
            
            # Replace NaN values with None for JSON compatibility
            snapshot_df = snapshot_df.where(pd.notnull(snapshot_df), None)
            
            return snapshot_df.to_dict(orient="records")
            
        except Exception as e:
            logger.warning(f"Failed to create snapshot on df shape {df.shape}, columns: {list(df.columns)}: {str(e)}")
            return []


async def handle_file_upload(file: UploadFile) -> pd.DataFrame:
    """
    Handle file upload and return DataFrame.
    
    Args:
        file (UploadFile): Uploaded file object
        
    Returns:
        pd.DataFrame: Loaded DataFrame from the uploaded file
        
    Raises:
        HTTPException: If file format is unsupported or file cannot be read
    """
    try:
        temp_path = await save_temp_file(file)
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension == '.csv':
            df = load_csv(temp_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = load_excel(temp_path)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported formats: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info(f"Successfully loaded file: {file.filename} - Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"File upload handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File could not be read: {str(e)}")


def preprocess_pipeline(
    df: pd.DataFrame,
    cleaning_config: Optional[CleaningConfig] = None,
    encoding_config: Optional[EncodingConfig] = None,
    validate_first: bool = True
) -> PreprocessingResponse:
    """
    Complete preprocessing pipeline combining validation, cleaning, and encoding.
    
    Runs full preprocessing pipeline: validation → cleaning → encoding. 
    Returns structured results, audit log, and snapshot previews.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess
        cleaning_config (Optional[CleaningConfig]): Configuration for data cleaning
        encoding_config (Optional[EncodingConfig]): Configuration for data encoding
        validate_first (bool): Whether to validate data before processing
        
    Returns:
        PreprocessingResponse: Complete preprocessing response with results and audit log
        
    Raises:
        HTTPException: If any step in the pipeline fails
    """
    service = DataPreprocessingService()
    results = {}
    intermediate_states = {}  # Fixed: renamed from intermediate_snapshots
    
    # Validation step
    if validate_first:
        try:
            results["validation"] = service.validate_data(df)
            # Take snapshot after validation (original data)
            intermediate_states["after_validation"] = service.create_snapshot(df)
            logger.info("Data validation completed successfully")
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data validation failed: {str(e)}")
    else:
        # If validation is skipped, still take initial snapshot
        intermediate_states["after_validation"] = service.create_snapshot(df)
    
    # Cleaning step
    if cleaning_config:
        try:
            results["cleaning"] = service.clean_data(df, cleaning_config)
            df = results["cleaning"].data
            # Take snapshot after cleaning
            intermediate_states["after_cleaning"] = service.create_snapshot(df)
            logger.info(f"Data cleaning completed successfully. Shape changed from {results['cleaning'].original_shape} to {results['cleaning'].final_shape}")
        except Exception as e:
            logger.error(f"Data cleaning step failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data cleaning step failed: {str(e)}")
    else:
        # If cleaning is skipped, set snapshot to None
        intermediate_states["after_cleaning"] = None
    
    # Encoding step
    if encoding_config:
        try:
            results["encoding"] = service.encode_data(df, encoding_config)
            df = results["encoding"].data
            # Take snapshot after encoding
            intermediate_states["after_encoding"] = service.create_snapshot(df)
            logger.info(f"Data encoding completed successfully. Shape changed from {results['encoding'].original_shape} to {results['encoding'].final_shape}")
        except Exception as e:
            logger.error(f"Data encoding step failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Data encoding step failed: {str(e)}")
    else:
        # If encoding is skipped, set snapshot to None
        intermediate_states["after_encoding"] = None
    
    # Generate audit log
    try:
        cleaning_actions = results.get("cleaning", {}).get("cleaning_actions", []) if "cleaning" in results else []
        encoding_actions = results.get("encoding", {}).get("encoding_actions", []) if "encoding" in results else []
        audit = service.audit_cleaning_actions(cleaning_actions, encoding_actions)
        
        # Generate final preview
        preview = service.create_snapshot(df)
        
        return PreprocessingResponse(
            num_rows=len(df),
            num_columns=len(df.columns),
            audit_log=audit,
            preview=preview,
            intermediate_states=intermediate_states  # Fixed: using correct variable name
        )
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline finalization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing pipeline finalization failed: {str(e)}")