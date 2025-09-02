from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import pandas as pd


class CleanlinessReport(BaseModel):
    """Report on data quality and cleanliness metrics."""
    model_config = ConfigDict(extra="forbid")
    
    total_rows: int = Field(..., description="Total number of rows in the dataset")
    total_columns: int = Field(..., description="Total number of columns in the dataset")
    missing_per_column: Dict[str, float] = Field(
        ..., 
        description="Missing value ratio per column (0.0 to 1.0)"
    )
    duplicate_rows: int = Field(..., description="Number of duplicate rows in the dataset")
    column_types: Dict[str, str] = Field(..., description="Data types for each column")
    categorical_cardinality: Dict[str, int] = Field(
        ..., 
        description="Number of unique values for categorical columns"
    )

    @field_validator('missing_per_column')
    @classmethod
    def validate_missing_ratios(cls, v):
        """Validate that all missing ratios are between 0 and 1."""
        for column, ratio in v.items():
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"Missing ratio for column '{column}' must be between 0.0 and 1.0, got {ratio}")
        return v

    @field_validator('duplicate_rows')
    @classmethod
    def validate_duplicates_non_negative(cls, v):
        """Validate that duplicate rows count is non-negative."""
        if v < 0:
            raise ValueError("Duplicate rows count cannot be negative")
        return v

    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate consistency between reported metrics."""
        if self.duplicate_rows > self.total_rows:
            raise ValueError("Duplicate rows cannot exceed total rows")
        return self

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "total_rows": 1000,
                "total_columns": 8,
                "missing_per_column": {
                    "name": 0.0,
                    "age": 0.08,
                    "department": 0.0,
                    "salary": 0.12,
                    "city": 0.05
                },
                "duplicate_rows": 15,
                "column_types": {
                    "name": "object",
                    "age": "int64",
                    "department": "object",
                    "salary": "float64",
                    "city": "object"
                },
                "categorical_cardinality": {
                    "department": 5,
                    "city": 12
                }
            }
        }
    )


class ValidationReport(BaseModel):
    """Report from data validation process."""
    model_config = ConfigDict(extra="forbid")
    
    valid: bool = Field(..., description="Whether the data passes all validation checks")
    errors: List[str] = Field(
        default_factory=list, 
        description="List of validation errors that prevent processing"
    )
    warnings: List[str] = Field(
        default_factory=list, 
        description="List of non-critical warnings about the data"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional validation details and metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "valid": False,
                "errors": [
                    "Dataset exceeds maximum row limit (1,000,000)",
                    "Column 'invalid_col' contains only null values"
                ],
                "warnings": [
                    "High cardinality detected in column 'id' (10,000 unique values)",
                    "Large file size: 150.3MB may impact performance"
                ],
                "details": {
                    "max_rows_exceeded_by": 50000,
                    "null_only_columns": ["invalid_col"],
                    "high_cardinality_columns": {
                        "id": {"unique_count": 10000, "total_count": 10000}
                    }
                }
            }
        }
    )


class CleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""
    model_config = ConfigDict(extra="forbid")
    
    drop_duplicates: bool = Field(
        default=True, 
        description="Whether to remove duplicate rows from the dataset"
    )
    
    missing_strategy: Literal["mean", "median", "mode", "constant", "drop"] = Field(
        default="mean",
        description="Strategy for handling missing values"
    )
    
    missing_constant_value: Optional[str] = Field(
        default=None,
        description="Constant value to use when missing_strategy is 'constant'"
    )
    
    outlier_handling: Literal["none", "iqr", "zscore"] = Field(
        default="none",
        description="Method for detecting and handling outliers"
    )
    
    outlier_threshold: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Threshold for outlier detection (Z-score or IQR multiplier)"
    )
    
    outlier_action: Literal["remove", "cap", "flag"] = Field(
        default="cap",
        description="Action to take when outliers are detected"
    )
    
    normalize_column_names: bool = Field(
        default=True,
        description="Whether to normalize column names (lowercase, underscore)"
    )
    
    remove_empty_columns: bool = Field(
        default=True,
        description="Whether to remove columns that are entirely empty"
    )
    
    @model_validator(mode='after')
    def validate_missing_constant(self):
        """Validate that constant value is provided when strategy is 'constant'."""
        if self.missing_strategy == "constant" and self.missing_constant_value is None:
            raise ValueError("missing_constant_value must be provided when missing_strategy is 'constant'")
        return self

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "drop_duplicates": True,
                "missing_strategy": "mean",
                "missing_constant_value": None,
                "outlier_handling": "iqr",
                "outlier_threshold": 1.5,
                "outlier_action": "cap",
                "normalize_column_names": True,
                "remove_empty_columns": True
            }
        }
    )


class EncodingConfig(BaseModel):
    """Configuration for data encoding operations."""
    model_config = ConfigDict(extra="forbid")
    
    categorical_columns: List[str] = Field(
        default_factory=list,
        description="List of categorical columns to encode (empty list means auto-detect)"
    )
    
    method: Literal["none", "label", "onehot", "target"] = Field(
        default="none",
        description="Categorical encoding method to use"
    )
    
    handle_unseen: Literal["ignore", "token"] = Field(
        default="token",
        description="How to handle unseen categories during encoding"
    )
    
    max_categories_for_onehot: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Maximum number of categories for one-hot encoding"
    )
    
    scale_numerical: bool = Field(
        default=False,
        description="Whether to scale numerical columns"
    )
    
    scaling_method: Literal["standard", "minmax", "robust", "none"] = Field(
        default="standard",
        description="Method for scaling numerical columns"
    )

    @model_validator(mode='after')
    def validate_encoding_consistency(self):
        """Validate encoding configuration consistency."""
        if self.method in {"label", "onehot", "target"} and not self.categorical_columns:
            # Allow empty categorical_columns - will auto-detect
            pass
        
        if self.method == "onehot":
            for col in self.categorical_columns:
                # Note: Actual cardinality check will happen at runtime
                pass
        
        return self

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "categorical_columns": ["department", "city", "status"],
                "method": "label",
                "handle_unseen": "token",
                "max_categories_for_onehot": 10,
                "scale_numerical": True,
                "scaling_method": "standard"
            }
        }
    )


class PreprocessingRequest(BaseModel):
    """Request configuration for data preprocessing pipeline."""
    model_config = ConfigDict(extra="forbid")
    
    cleaning: CleaningConfig = Field(
        default_factory=CleaningConfig,
        description="Configuration for data cleaning operations"
    )
    
    encoding: EncodingConfig = Field(
        default_factory=EncodingConfig,
        description="Configuration for data encoding operations"
    )
    
    validate_first: bool = Field(
        default=True,
        description="Whether to validate data quality before processing"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "cleaning": {
                    "drop_duplicates": True,
                    "missing_strategy": "mean",
                    "outlier_handling": "iqr",
                    "normalize_column_names": True
                },
                "encoding": {
                    "categorical_columns": ["department", "city"],
                    "method": "label",
                    "scale_numerical": True,
                    "scaling_method": "standard"
                },
                "validate_first": True
            }
        }
    )


class FileUploadResponse(BaseModel):
    """Response from file upload and initial analysis."""
    model_config = ConfigDict(extra="forbid")
    
    filename: str = Field(..., description="Original filename that was uploaded")
    file_size: int = Field(..., description="File size in bytes")
    row_count: int = Field(..., description="Number of rows in the dataset")
    column_count: int = Field(..., description="Number of columns in the dataset")
    preview_rows: List[Dict[str, Any]] = Field(
        ..., 
        description="Preview of first few rows as list of dictionaries"
    )
    initial_cleanliness_report: CleanlinessReport = Field(
        ..., 
        description="Initial data quality assessment"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "filename": "sales_data.csv",
                "file_size": 2621440,
                "row_count": 1000,
                "column_count": 8,
                "preview_rows": [
                    {
                        "name": "John Doe",
                        "age": 30,
                        "department": "IT",
                        "salary": 75000.0,
                        "city": "New York"
                    },
                    {
                        "name": "Jane Smith",
                        "age": 25,
                        "department": "HR",
                        "salary": 55000.0,
                        "city": "Chicago"
                    }
                ],
                "initial_cleanliness_report": {
                    "total_rows": 1000,
                    "total_columns": 8,
                    "missing_per_column": {
                        "name": 0.0,
                        "age": 0.08,
                        "department": 0.0,
                        "salary": 0.12
                    },
                    "duplicate_rows": 15,
                    "column_types": {
                        "name": "object",
                        "age": "int64",
                        "department": "object",
                        "salary": "float64"
                    },
                    "categorical_cardinality": {
                        "department": 5,
                        "city": 12
                    }
                }
            }
        }
    )


class PreprocessingResponse(BaseModel):
    """Response from data preprocessing pipeline."""
    model_config = ConfigDict(extra="forbid")
    
    initial_cleanliness_report: CleanlinessReport = Field(
        ..., 
        description="Data quality report before processing"
    )
    
    final_cleanliness_report: CleanlinessReport = Field(
        ..., 
        description="Data quality report after processing"
    )
    
    validation_report: ValidationReport = Field(
        ..., 
        description="Results from data validation checks"
    )
    
    audit_log: List[Dict[str, Any]] = Field(
        ..., 
        description="Complete audit trail of all operations performed"
    )
    
    preview_rows: List[Dict[str, Any]] = Field(
        ..., 
        description="Preview of processed data as list of dictionaries"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "initial_cleanliness_report": {
                    "total_rows": 1000,
                    "total_columns": 8,
                    "missing_per_column": {"age": 0.08, "salary": 0.12},
                    "duplicate_rows": 15,
                    "column_types": {"age": "int64", "salary": "float64"},
                    "categorical_cardinality": {"department": 5}
                },
                "final_cleanliness_report": {
                    "total_rows": 985,
                    "total_columns": 10,
                    "missing_per_column": {"age": 0.0, "salary": 0.0},
                    "duplicate_rows": 0,
                    "column_types": {"age": "int64", "salary": "float64"},
                    "categorical_cardinality": {"department": 5}
                },
                "validation_report": {
                    "valid": True,
                    "errors": [],
                    "warnings": ["Large dataset detected"],
                    "details": {}
                },
                "audit_log": [
                    {
                        "operation": "remove_duplicates",
                        "timestamp": "2024-01-15T10:30:45.123456",
                        "before_shape": [1000, 8],
                        "after_shape": [985, 8],
                        "details": {"rows_removed": 15}
                    },
                    {
                        "operation": "encode_categorical",
                        "column": "department",
                        "timestamp": "2024-01-15T10:32:15.789012",
                        "before_shape": [985, 8],
                        "after_shape": [985, 8],
                        "details": {"method": "label", "unique_values": 5}
                    }
                ],
                "preview_rows": [
                    {
                        "name": "John Doe",
                        "age": 30,
                        "department": 1,
                        "salary": 75000.0,
                        "city": 0
                    },
                    {
                        "name": "Jane Smith",
                        "age": 25,
                        "department": 2,
                        "salary": 55000.0,
                        "city": 1
                    }
                ]
            }
        }
    )


# Additional utility models for comprehensive API coverage
class ErrorResponse(BaseModel):
    """Standardized error response format."""
    model_config = ConfigDict(extra="forbid")
    
    error: bool = Field(default=True, description="Always true to indicate this is an error response")
    error_type: str = Field(..., description="Classification of the error that occurred")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional error details and context"
    )
    timestamp: str = Field(..., description="ISO timestamp when the error occurred")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "error": True,
                "error_type": "ValidationError",
                "message": "File size exceeds maximum limit (100MB)",
                "details": {
                    "file_size_mb": 150.0,
                    "max_allowed_mb": 100.0,
                    "filename": "large_dataset.csv"
                },
                "timestamp": "2024-01-15T10:30:00.000000"
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """Health check response for system monitoring."""
    model_config = ConfigDict(extra="forbid")
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., 
        description="Current system health status"
    )
    version: str = Field(..., description="Current service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    timestamp: str = Field(..., description="ISO timestamp of this health check")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 86400.0,
                "memory_usage_mb": 512.3,
                "timestamp": "2024-01-15T10:00:00.000000"
            }
        }
    )


# Validation helper functions
def validate_column_exists(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Validate that specified columns exist in DataFrame."""
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    return columns


def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Validate that specified columns are numeric."""
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns specified for numeric operations: {non_numeric}")
    return columns


def validate_categorical_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Validate that specified columns are suitable for categorical encoding."""
    unsuitable = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            if unique_ratio > 0.5:  # More than 50% unique values
                unsuitable.append(col)
    
    if unsuitable:
        raise ValueError(f"Columns may not be suitable for categorical encoding (high cardinality): {unsuitable}")
    return columns