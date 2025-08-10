from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class BaseSchema(BaseModel):
    """Base schema class with common configuration."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelType(str, Enum):
    """Machine learning problem types."""
    classification = "classification"
    regression = "regression"


class ModelAlgorithm(str, Enum):
    """Supported machine learning algorithms."""
    # Classification algorithms
    random_forest_classifier = "random_forest_classifier"
    logistic_regression = "logistic_regression"
    gradient_boosting_classifier = "gradient_boosting_classifier"
    svc = "svc"
    knn_classifier = "knn_classifier"
    decision_tree_classifier = "decision_tree_classifier"
    
    # Regression algorithms
    random_forest_regressor = "random_forest_regressor"
    linear_regression = "linear_regression"
    ridge_regression = "ridge_regression"
    lasso_regression = "lasso_regression"
    gradient_boosting_regressor = "gradient_boosting_regressor"
    svr = "svr"
    knn_regressor = "knn_regressor"
    decision_tree_regressor = "decision_tree_regressor"


class ModelConfig(BaseSchema):
    """Configuration for machine learning model training."""
    model_type: Optional[ModelType] = Field(
        None, 
        description="Type of ML problem. If not specified, will be auto-detected"
    )
    feature_columns: List[str] = Field(
        ..., 
        description="List of feature column names",
        min_items=1
    )
    target_column: str = Field(
        ..., 
        description="Target column name"
    )
    algorithm: Optional[ModelAlgorithm] = Field(
        None,
        description="Specific algorithm to use. If not specified, will try multiple algorithms and select the best"
    )
    hyperparameters: Optional[Dict[str, Union[int, float, str, bool, list, dict]]] = Field(
        None,
        description="Hyperparameters for the algorithm. Will be merged with defaults"
    )
    test_size: Optional[float] = Field(
        0.2, 
        description="Proportion of data to use for testing",
        ge=0.1,
        le=0.5
    )
    random_state: Optional[int] = Field(
        42,
        description="Random seed for reproducibility"
    )
    auto_select_algorithm: Optional[bool] = Field(
        True,
        description="Whether to automatically select the best performing algorithm"
    )
    
    @validator('feature_columns')
    def validate_feature_columns(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one feature column must be specified")
        return v
    
    @validator('target_column')
    def validate_target_column(cls, v):
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()


class EvaluationMetrics(BaseSchema):
    """Comprehensive model evaluation metrics."""
    # Classification metrics
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Regression metrics
    mse: Optional[float] = Field(None, ge=0.0)
    rmse: Optional[float] = Field(None, ge=0.0)
    r2_score: Optional[float] = None
    mae: Optional[float] = Field(None, ge=0.0, description="Mean Absolute Error")
    
    # Additional metrics
    training_score: Optional[float] = Field(None, description="Score on training data")
    validation_score: Optional[float] = Field(None, description="Score on validation data")


class AlgorithmResult(BaseSchema):
    """Results for a single algorithm during training."""
    algorithm: ModelAlgorithm
    score: float
    training_time: float
    metrics: EvaluationMetrics
    status: str = Field(default="success", description="Training status: success, failed, skipped")
    error_message: Optional[str] = None


class DataInfo(BaseSchema):
    """Information about the training data."""
    total_samples: int = Field(..., ge=1)
    clean_samples: int = Field(..., ge=1)
    features_count: int = Field(..., ge=1)
    target_unique_values: int = Field(..., ge=1)
    missing_values_count: Optional[int] = 0
    feature_types: Optional[Dict[str, str]] = None
    target_type: Optional[str] = None


class TrainingResponse(BaseSchema):
    """Response from model training operation."""
    model_id: str = Field(..., description="Unique identifier for the trained model")
    model_type: ModelType
    algorithm: Optional[ModelAlgorithm] = Field(None, description="Algorithm used for the best model")
    feature_columns: List[str]
    target_column: str
    model_path: str = Field(..., description="Path to saved model file")
    
    # Training information
    training_time: float = Field(..., ge=0.0, description="Total training time in seconds")
    train_size: int = Field(..., ge=1)
    test_size: int = Field(..., ge=1)
    best_score: Optional[float] = Field(None, description="Best score achieved")
    
    # Performance metrics (populated based on model type)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    mae: Optional[float] = None
    
    # Additional information
    algorithm_results: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Results from all algorithms tried during auto-selection"
    )
    data_info: Optional[DataInfo] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class PredictionRequest(BaseSchema):
    """Request for making predictions."""
    model_id: Optional[str] = Field(None, description="Model ID to use for prediction")
    model_path: Optional[str] = Field(None, description="Direct path to model file")
    input_data: List[Dict[str, Any]] = Field(
        ...,
        description="List of dictionaries containing input features",
        min_items=1
    )
    validate_input: Optional[bool] = Field(
        True,
        description="Whether to perform strict input validation"
    )
    return_confidence: Optional[bool] = Field(
        True,
        description="Whether to return confidence scores (classification only)"
    )
    return_feature_importance: Optional[bool] = Field(
        False,
        description="Whether to return feature importance"
    )
    
    @validator('input_data')
    def validate_input_data(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        return v


class PredictionResponse(BaseSchema):
    """Response from prediction operation."""
    predictions: List[Any] = Field(..., description="Model predictions")
    model_id: str = Field(..., description="ID of the model used")
    model_type: ModelType
    algorithm: Optional[ModelAlgorithm] = Field(None, description="Algorithm of the model used")
    num_predictions: int = Field(..., ge=1)
    
    # Optional additional information
    confidence_scores: Optional[List[float]] = Field(
        None,
        description="Confidence scores for predictions (classification only)"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )
    prediction_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the predictions"
    )
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken for prediction")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ModelInfo(BaseSchema):
    """Comprehensive model information."""
    model_id: str
    model_type: ModelType
    algorithm: ModelAlgorithm
    feature_columns: List[str]
    target_column: str
    model_path: str
    
    # Training metadata
    hyperparameters: Optional[Dict[str, Union[int, float, str, bool, list, dict]]] = None
    train_size: int
    test_size: int
    training_time: float
    timestamp: datetime
    
    # Performance metrics
    best_score: float
    metrics: EvaluationMetrics
    all_algorithm_results: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Data information
    data_info: DataInfo
    
    # File information
    model_file_exists: bool
    model_file_size: int = Field(..., ge=0, description="Model file size in bytes")


class ModelListResponse(BaseSchema):
    """Response for listing models."""
    models: Dict[str, ModelInfo]
    summary: Optional[Dict[str, Any]] = None


class ModelHealthStatus(BaseSchema):
    """Model health check status."""
    model_id: str
    healthy: bool
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    file_size: Optional[int] = None
    last_modified: Optional[float] = None
    check_timestamp: datetime = Field(default_factory=datetime.now)


class ModelComparisonResponse(BaseSchema):
    """Response for model performance comparison."""
    classification_models: List[Dict[str, Any]] = Field(default_factory=list)
    regression_models: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any]


class ModelConfigExport(BaseSchema):
    """Exported model configuration for reproducibility."""
    model_id: str
    model_type: ModelType
    algorithm: ModelAlgorithm
    feature_columns: List[str]
    target_column: str
    hyperparameters: Optional[Dict[str, Union[int, float, str, bool, list, dict]]]
    training_config: Dict[str, Any]
    performance: EvaluationMetrics
    data_info: DataInfo
    timestamp: datetime
    export_timestamp: datetime = Field(default_factory=datetime.now)


class CleanupResult(BaseSchema):
    """Result of model cleanup operation."""
    cleaned_models: List[str] = Field(default_factory=list)
    failed_cleanups: List[Dict[str, str]] = Field(default_factory=list)
    total_cleaned: int = Field(..., ge=0)
    cleanup_timestamp: datetime = Field(default_factory=datetime.now)


class ModelDeleteResponse(BaseSchema):
    """Response for model deletion."""
    model_id: str
    deleted: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


# Error response schemas
class ErrorDetail(BaseSchema):
    """Detailed error information."""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FieldValidationError(BaseSchema):
    """Validation error details."""
    field: str
    message: str
    invalid_value: Optional[Any] = None


class ModelingError(BaseSchema):
    """General modeling operation error."""
    operation: str
    error_details: ErrorDetail
    suggestions: Optional[List[str]] = None