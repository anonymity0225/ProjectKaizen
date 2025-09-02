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
    logistic_regression = "logistic_regression"
    random_forest_classifier = "random_forest_classifier"
    gradient_boosting_classifier = "gradient_boosting_classifier"
    svc = "svc"
    knn_classifier = "knn_classifier"
    decision_tree_classifier = "decision_tree_classifier"
    
    # Regression algorithms
    linear_regression = "linear_regression"
    random_forest_regressor = "random_forest_regressor"
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
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification accuracy")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification precision")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification recall")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    
    # Regression metrics
    mse: Optional[float] = Field(None, ge=0.0, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, ge=0.0, description="Root Mean Squared Error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    mae: Optional[float] = Field(None, ge=0.0, description="Mean Absolute Error")
    
    # Additional metrics
    training_score: Optional[float] = Field(None, description="Score on training data")
    validation_score: Optional[float] = Field(None, description="Score on validation data")


class AlgorithmResult(BaseSchema):
    """Results for a single algorithm during training."""
    algorithm: ModelAlgorithm = Field(..., description="Algorithm name")
    score: float = Field(..., description="Algorithm performance score")
    training_time: float = Field(..., description="Time taken to train in seconds")
    metrics: EvaluationMetrics = Field(..., description="Detailed evaluation metrics")
    status: str = Field(default="success", description="Training status: success, failed, skipped")
    error_message: Optional[str] = Field(None, description="Error message if training failed")


class DataInfo(BaseSchema):
    """Information about the training data."""
    total_samples: int = Field(..., ge=1, description="Total number of samples")
    clean_samples: int = Field(..., ge=1, description="Number of clean samples used")
    features_count: int = Field(..., ge=1, description="Number of features")
    target_unique_values: int = Field(..., ge=1, description="Number of unique target values")
    missing_values_count: Optional[int] = Field(0, description="Number of missing values")
    feature_types: Optional[Dict[str, str]] = Field(None, description="Feature data types")
    target_type: Optional[str] = Field(None, description="Target column data type")


class TrainingResponse(BaseSchema):
    """Response from model training operation."""
    model_id: str = Field(..., description="Unique identifier for the trained model")
    model_type: ModelType = Field(..., description="Type of ML problem")
    algorithm: Optional[ModelAlgorithm] = Field(None, description="Algorithm used for the best model")
    feature_columns: List[str] = Field(..., description="List of feature columns used")
    target_column: str = Field(..., description="Target column name")
    model_path: str = Field(..., description="Path to saved model file")
    
    # Training information
    training_time: float = Field(..., ge=0.0, description="Total training time in seconds")
    train_size: int = Field(..., ge=1, description="Number of training samples")
    test_size: int = Field(..., ge=1, description="Number of test samples")
    best_score: Optional[float] = Field(None, description="Best score achieved")
    
    # Performance metrics (populated based on model type)
    accuracy: Optional[float] = Field(None, description="Classification accuracy")
    precision: Optional[float] = Field(None, description="Classification precision")
    recall: Optional[float] = Field(None, description="Classification recall")
    f1_score: Optional[float] = Field(None, description="F1 score")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    
    # Additional information
    algorithm_results: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Results from all algorithms tried during auto-selection"
    )
    data_info: Optional[DataInfo] = Field(None, description="Information about training data")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Training timestamp")


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
    model_type: ModelType = Field(..., description="Type of ML problem")
    algorithm: Optional[ModelAlgorithm] = Field(None, description="Algorithm of the model used")
    num_predictions: int = Field(..., ge=1, description="Number of predictions made")
    
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
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Prediction timestamp")


class ModelInfo(BaseSchema):
    """Comprehensive model information."""
    model_id: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Type of ML problem")
    algorithm: ModelAlgorithm = Field(..., description="Algorithm used")
    feature_columns: List[str] = Field(..., description="List of feature columns")
    target_column: str = Field(..., description="Target column name")
    model_path: str = Field(..., description="Path to model file")
    
    # Training metadata
    hyperparameters: Optional[Dict[str, Union[int, float, str, bool, list, dict]]] = Field(
        None, description="Model hyperparameters"
    )
    train_size: int = Field(..., description="Number of training samples")
    test_size: int = Field(..., description="Number of test samples")
    training_time: float = Field(..., description="Training time in seconds")
    timestamp: datetime = Field(..., description="Training timestamp")
    
    # Performance metrics
    best_score: float = Field(..., description="Best performance score")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    all_algorithm_results: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Results from all algorithms tested"
    )
    
    # Data information
    data_info: DataInfo = Field(..., description="Training data information")
    
    # File information
    model_file_exists: bool = Field(..., description="Whether model file exists")
    model_file_size: int = Field(..., ge=0, description="Model file size in bytes")


class ModelListResponse(BaseSchema):
    """Response for listing models."""
    models: List[ModelInfo] = Field(..., description="List of model information")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary statistics")


class ModelHealthStatus(BaseSchema):
    """Model health check status."""
    model_id: str = Field(..., description="Model identifier")
    healthy: bool = Field(..., description="Whether model is healthy")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    file_size: Optional[int] = Field(None, description="Model file size in bytes")
    last_modified: Optional[float] = Field(None, description="Last modification timestamp")
    check_timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class ModelComparisonResponse(BaseSchema):
    """Response for model performance comparison."""
    classification_models: List[Dict[str, Any]] = Field(
        default_factory=list, description="Classification models comparison"
    )
    regression_models: List[Dict[str, Any]] = Field(
        default_factory=list, description="Regression models comparison"
    )
    summary: Dict[str, Any] = Field(..., description="Comparison summary")


class ModelConfigExport(BaseSchema):
    """Exported model configuration for reproducibility."""
    model_id: str = Field(..., description="Model identifier")
    model_type: ModelType = Field(..., description="Type of ML problem")
    algorithm: ModelAlgorithm = Field(..., description="Algorithm used")
    feature_columns: List[str] = Field(..., description="List of feature columns")
    target_column: str = Field(..., description="Target column name")
    hyperparameters: Optional[Dict[str, Union[int, float, str, bool, list, dict]]] = Field(
        None, description="Model hyperparameters"
    )
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    performance: EvaluationMetrics = Field(..., description="Performance metrics")
    data_info: DataInfo = Field(..., description="Training data information")
    timestamp: datetime = Field(..., description="Original training timestamp")
    export_timestamp: datetime = Field(default_factory=datetime.now, description="Export timestamp")


class CleanupResult(BaseSchema):
    """Result of model cleanup operation."""
    cleaned_models: List[str] = Field(default_factory=list, description="List of cleaned model IDs")
    failed_cleanups: List[Dict[str, str]] = Field(default_factory=list, description="Failed cleanup attempts")
    total_cleaned: int = Field(..., ge=0, description="Total number of models cleaned")
    cleanup_timestamp: datetime = Field(default_factory=datetime.now, description="Cleanup timestamp")


class ModelDeleteResponse(BaseSchema):
    """Response for model deletion."""
    model_id: str = Field(..., description="Deleted model identifier")
    deleted: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Deletion result message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Deletion timestamp")


# Error response schemas
class ErrorDetail(BaseSchema):
    """Detailed error information."""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class FieldValidationError(BaseSchema):
    """Validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="Value that caused validation failure")


class ModelingError(BaseSchema):
    """General modeling operation error."""
    operation: str = Field(..., description="Operation that failed")
    error_details: ErrorDetail = Field(..., description="Detailed error information")
    suggestions: Optional[List[str]] = Field(None, description="Suggested solutions")


# Alias classes for router compatibility
ExportResponse = ModelConfigExport
CleanupResponse = CleanupResult
HealthResponse = ModelHealthStatus