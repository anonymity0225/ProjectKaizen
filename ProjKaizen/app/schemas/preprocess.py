# Pydantic schemas for preprocessing service
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple, Literal


class PreviewRow(BaseModel):
    """Single row of data for preview purposes."""
    
    class Config:
        extra = "allow"  # Allow dynamic fields based on dataset columns
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "age": 30,
                "salary": 50000.0,
                "department": "Engineering"
            }
        }


class CleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""
    
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows from the dataset")
    handle_missing_values: bool = Field(default=True, description="Apply missing value handling strategy")
    missing_value_strategy: Literal[
        "drop_rows", "drop_columns", "fill_mean", "fill_median", 
        "fill_mode", "fill_forward", "fill_backward"
    ] = Field(
        default="drop_rows",
        description="Strategy for handling missing values: drop_rows removes rows with any missing values, drop_columns removes columns exceeding missing threshold, fill_* methods impute missing values"
    )
    missing_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for dropping columns with missing values (0.0 = drop if any missing, 1.0 = drop only if all missing)"
    )
    remove_outliers: bool = Field(default=False, description="Remove statistical outliers using Interquartile Range (IQR) method")
    normalize_column_names: bool = Field(default=True, description="Normalize column names to lowercase with underscores for consistency")
    
    class Config:
        schema_extra = {
            "example": {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "missing_value_strategy": "fill_mean",
                "missing_threshold": 0.3,
                "remove_outliers": True,
                "normalize_column_names": True
            }
        }


class EncodingConfig(BaseModel):
    """Configuration for data encoding operations."""
    
    categorical_encoding_method: Literal["label", "onehot", "target"] = Field(
        default="label",
        description="Method for encoding categorical variables: label (ordinal encoding), onehot (one-hot encoding), target (target mean encoding)"
    )
    categorical_columns: List[str] = Field(
        default=[],
        description="Specific categorical columns to encode. If empty, all detected categorical columns will be encoded"
    )
    scale_numerical: bool = Field(default=True, description="Apply feature scaling to numerical columns")
    scaling_method: Literal["standard", "minmax", "robust"] = Field(
        default="standard",
        description="Method for scaling numerical features: standard (z-score normalization), minmax (0-1 scaling), robust (median and IQR scaling)"
    )
    numerical_columns: List[str] = Field(
        default=[],
        description="Specific numerical columns to scale. If empty, all detected numerical columns will be scaled"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name for supervised learning. This column will be excluded from scaling and used for target encoding"
    )
    drop_first: bool = Field(
        default=True,
        description="Drop first category in one-hot encoding to avoid multicollinearity (n-1 encoding)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "categorical_encoding_method": "onehot",
                "categorical_columns": ["department", "category"],
                "scale_numerical": True,
                "scaling_method": "standard",
                "numerical_columns": ["age", "salary", "experience"],
                "target_column": "performance_score",
                "drop_first": True
            }
        }


class ValidationReport(BaseModel):
    """Report from data validation step."""
    
    is_valid: bool = Field(description="Whether the dataset passed all validation checks")
    issues: List[str] = Field(description="List of specific validation issues found in the dataset")
    recommendations: List[str] = Field(description="Actionable recommendations to resolve identified issues")
    num_rows: int = Field(description="Total number of rows in the dataset")
    num_columns: int = Field(description="Total number of columns in the dataset")
    missing_values_summary: Dict[str, Dict[str, Union[int, float]]] = Field(
        description="Per-column summary of missing values. Each column contains count and percentage of missing values"
    )
    duplicate_rows: int = Field(description="Number of duplicate rows detected in the dataset")


class CleanedDataResponse(BaseModel):
    """Response from data cleaning step."""
    
    data: Any = Field(description="Cleaned dataset as DataFrame (serialized to appropriate format for API response)")
    original_shape: Tuple[int, int] = Field(description="Original dataset dimensions as (rows, columns) before cleaning")
    final_shape: Tuple[int, int] = Field(description="Final dataset dimensions as (rows, columns) after cleaning")
    cleaning_actions: List[str] = Field(description="Chronological list of cleaning operations performed on the dataset")
    num_rows_removed: int = Field(description="Total number of rows removed during cleaning process")
    num_columns_removed: int = Field(description="Total number of columns removed during cleaning process")
    
    class Config:
        arbitrary_types_allowed = True


class EncodedDataResponse(BaseModel):
    """Response from data encoding step."""
    
    data: Any = Field(description="Encoded dataset as DataFrame (serialized to appropriate format for API response)")
    original_shape: Tuple[int, int] = Field(description="Dataset dimensions as (rows, columns) before encoding")
    final_shape: Tuple[int, int] = Field(description="Dataset dimensions as (rows, columns) after encoding")
    encoding_actions: List[str] = Field(description="Chronological list of encoding operations performed")
    encoders_used: List[str] = Field(description="List of column names that had encoding transformations applied")
    scaler_used: Optional[str] = Field(description="Type of scaler applied to numerical features (e.g., 'StandardScaler', 'MinMaxScaler')")
    
    class Config:
        arbitrary_types_allowed = True


class PreprocessingRequest(BaseModel):
    """Request for preprocessing pipeline."""
    
    cleaning_config: Optional[CleaningConfig] = Field(default=None, description="Data cleaning configuration. If null, cleaning step is skipped")
    encoding_config: Optional[EncodingConfig] = Field(default=None, description="Data encoding configuration. If null, encoding step is skipped")
    validate_first: bool = Field(default=True, description="Whether to perform data validation before preprocessing steps")
    
    class Config:
        schema_extra = {
            "example": {
                "cleaning_config": {
                    "remove_duplicates": True,
                    "handle_missing_values": True,
                    "missing_value_strategy": "fill_mean",
                    "missing_threshold": 0.3,
                    "remove_outliers": False,
                    "normalize_column_names": True
                },
                "encoding_config": {
                    "categorical_encoding_method": "onehot",
                    "categorical_columns": [],
                    "scale_numerical": True,
                    "scaling_method": "standard",
                    "numerical_columns": [],
                    "target_column": "target",
                    "drop_first": True
                },
                "validate_first": True
            }
        }


class PreprocessingResponse(BaseModel):
    """Response from complete preprocessing pipeline."""
    
    num_rows: int = Field(description="Final number of rows after all preprocessing steps")
    num_columns: int = Field(description="Final number of columns after all preprocessing steps")
    audit_log: Dict[str, Any] = Field(
        description="Comprehensive audit trail containing details of all preprocessing operations, timestamps, and parameters used"
    )
    preview: List[PreviewRow] = Field(description="Preview of final processed dataset showing first 5 rows")
    intermediate_states: Optional[Dict[str, List[PreviewRow]]] = Field(
        default=None,
        description="Preview data after each processing step. Keys include 'after_cleaning', 'after_encoding', etc."
    )


class FileUploadResponse(BaseModel):
    """Response from file upload."""
    
    filename: str = Field(description="Original name of the uploaded file")
    file_size: int = Field(description="Size of the uploaded file in bytes")
    num_rows: int = Field(description="Number of data rows in the uploaded dataset")
    num_columns: int = Field(description="Number of columns in the uploaded dataset")
    column_names: List[str] = Field(description="List of all column names found in the dataset")
    data_types: Dict[str, str] = Field(
        description="Mapping of column names to their detected data types (e.g., 'int64', 'float64', 'object')"
    )
    preview: List[PreviewRow] = Field(description="Preview showing first 5 rows of the uploaded dataset")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "employee_data.csv",
                "file_size": 2048576,
                "num_rows": 1000,
                "num_columns": 8,
                "column_names": ["id", "name", "age", "department", "salary", "hire_date", "performance_score", "active"],
                "data_types": {
                    "id": "int64",
                    "name": "object",
                    "age": "int64",
                    "department": "object",
                    "salary": "float64",
                    "hire_date": "object",
                    "performance_score": "float64",
                    "active": "bool"
                },
                "preview": [
                    {
                        "id": 1,
                        "name": "John Doe",
                        "age": 30,
                        "department": "Engineering",
                        "salary": 75000.0,
                        "hire_date": "2020-01-15",
                        "performance_score": 4.2,
                        "active": True
                    }
                ]
            }
        }


class DataQualityReport(BaseModel):
    """Comprehensive data quality report."""
    
    validation_report: ValidationReport = Field(description="Results from data validation checks")
    column_statistics: Dict[str, Dict[str, Any]] = Field(
        description="Statistical summary for each column including mean, median, std, min, max for numerical columns and value counts for categorical columns"
    )
    data_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall data quality score from 0.0 (poor) to 1.0 (excellent) based on completeness, consistency, and validity"
    )
    recommendations: List[str] = Field(description="Prioritized list of actionable recommendations to improve data quality")


class TransformationHistory(BaseModel):
    """History of transformations applied to the dataset."""
    
    transformation_id: str = Field(description="Unique identifier for this transformation session")
    timestamp: str = Field(description="ISO 8601 timestamp when the transformation was applied")
    transformation_type: str = Field(description="Type of transformation performed (e.g., 'cleaning', 'encoding', 'validation')")
    parameters: Dict[str, Any] = Field(
        description="Complete set of parameters and configuration used for this transformation"
    )
    input_shape: Tuple[int, int] = Field(description="Dataset dimensions as (rows, columns) before this transformation")
    output_shape: Tuple[int, int] = Field(description="Dataset dimensions as (rows, columns) after this transformation")
    actions_performed: List[str] = Field(description="Detailed chronological list of all actions performed during this transformation")


class PreprocessingPipeline(BaseModel):
    """Complete preprocessing pipeline configuration."""
    
    pipeline_name: str = Field(description="Human-readable name for this preprocessing pipeline configuration")
    steps: List[str] = Field(description="Ordered list of preprocessing steps to execute (e.g., ['validation', 'cleaning', 'encoding'])")
    cleaning_config: Optional[CleaningConfig] = Field(default=None, description="Configuration for data cleaning step")
    encoding_config: Optional[EncodingConfig] = Field(default=None, description="Configuration for data encoding step")
    validation_enabled: bool = Field(default=True, description="Whether to include data validation in the pipeline")
    save_intermediates: bool = Field(default=True, description="Whether to preserve intermediate results after each step for debugging")


class BatchProcessingRequest(BaseModel):
    """Request for batch processing multiple files."""
    
    pipeline_config: PreprocessingPipeline = Field(description="Pipeline configuration to apply to all files in the batch")
    output_format: Literal["csv", "xlsx", "json", "parquet"] = Field(
        default="csv", 
        description="Output file format for all processed files"
    )
    merge_results: bool = Field(default=False, description="Whether to combine all processed files into a single output file")
    
    class Config:
        schema_extra = {
            "example": {
                "pipeline_config": {
                    "pipeline_name": "Standard Data Prep",
                    "steps": ["validation", "cleaning", "encoding"],
                    "cleaning_config": {
                        "remove_duplicates": True,
                        "handle_missing_values": True,
                        "missing_value_strategy": "fill_mean",
                        "missing_threshold": 0.3,
                        "remove_outliers": False,
                        "normalize_column_names": True
                    },
                    "encoding_config": {
                        "categorical_encoding_method": "onehot",
                        "scale_numerical": True,
                        "scaling_method": "standard",
                        "target_column": "target",
                        "drop_first": True
                    },
                    "validation_enabled": True,
                    "save_intermediates": False
                },
                "output_format": "csv",
                "merge_results": False
            }
        }


class BatchProcessingResponse(BaseModel):
    """Response from batch processing."""
    
    total_files_processed: int = Field(description="Total number of files attempted to process")
    successful_files: int = Field(description="Number of files that were successfully processed without errors")
    failed_files: int = Field(description="Number of files that failed during processing")
    processing_time: float = Field(description="Total elapsed processing time in seconds for the entire batch")
    output_files: List[str] = Field(description="List of file paths for all successfully generated output files")
    error_log: List[Dict[str, str]] = Field(
        description="Detailed error information for failed files. Each entry contains 'filename' and 'error' keys"
    )
    summary_statistics: Dict[str, Any] = Field(
        description="Aggregated statistics across all successfully processed files including total rows, columns, and processing metrics"
    )