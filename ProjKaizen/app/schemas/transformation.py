# Schemas for transformation
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Dict, List, Any, Optional, Tuple, Literal, Union
from enum import Enum


class ScalingMethod(str, Enum):
    """Scaling method options"""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    LOG = "log"
    ROBUST = "robust"
    CUSTOM = "custom"


class EncodingMethod(str, Enum):
    """Categorical encoding method options"""
    ONEHOT = "onehot"
    LABEL = "label"
    FREQUENCY = "frequency"
    CUSTOM = "custom"


class TextMethod(str, Enum):
    """Text preprocessing method options"""
    STEM = "stem"
    LEMMATIZE = "lemmatize"


class BaseTransformationResponse(BaseModel):
    """Base response model that all transformation responses inherit from"""
    model_config = ConfigDict(populate_by_name=True)
    
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BaseTransformationRequest(BaseModel):
    """Base request model with common fields and validation"""
    model_config = ConfigDict(populate_by_name=True)
    
    data: List[Dict[str, Any]] = Field(alias="dataframe", description="Data as list of row dictionaries")
    
    @field_validator('data')
    @classmethod
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class ColumnBasedRequest(BaseTransformationRequest):
    """Base for requests that operate on columns"""
    columns: List[str] = Field(alias="column", description="Column names to transform")
    
    @field_validator('columns', mode='before')
    @classmethod
    def normalize_columns(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError("columns must be a string or list of strings")
    
    @field_validator('columns')
    @classmethod
    def validate_columns_not_empty(cls, v):
        if not v:
            raise ValueError("At least one column must be specified")
        return v


# Date Extraction Models
class DateExtractionRequest(ColumnBasedRequest):
    """Request model for date extraction transformation"""
    components: List[str] = Field(
        default=["year", "month", "day"],
        description="List of date components to extract (year, month, day, hour, minute, second, weekday)"
    )
    
    @field_validator('components')
    @classmethod
    def validate_components(cls, v):
        valid_components = {"year", "month", "day", "hour", "minute", "second", "weekday"}
        invalid = set(v) - valid_components
        if invalid:
            raise ValueError(f"Invalid date components: {invalid}. Valid options: {valid_components}")
        return v


class DateExtractionResponse(BaseTransformationResponse):
    """Response model for date extraction transformation"""
    added_columns: List[str]


# Scaling Models
class ScalingRequest(ColumnBasedRequest):
    """Request model for scaling transformation"""
    method: ScalingMethod
    custom_range: Optional[Tuple[float, float]] = Field(default=None, description="Custom range for minmax scaling")
    
    @model_validator(mode='after')
    def validate_custom_range(self):
        if self.method == ScalingMethod.CUSTOM and self.custom_range is None:
            raise ValueError("custom_range must be provided when method is 'custom'")
        if self.custom_range is not None and len(self.custom_range) != 2:
            raise ValueError("custom_range must be a tuple of exactly 2 values")
        if self.custom_range is not None and self.custom_range[0] >= self.custom_range[1]:
            raise ValueError("custom_range first value must be less than second value")
        return self


class ScalingResponse(BaseTransformationResponse):
    """Response model for scaling transformation"""
    scaled_columns: List[str]
    scaling_parameters: Dict[str, Any] = Field(default_factory=dict)


# Categorical Encoding Models
class CategoricalEncodingRequest(ColumnBasedRequest):
    """Request model for categorical encoding transformation"""
    method: EncodingMethod
    custom_mapping: Optional[Dict[str, Any]] = Field(default=None, description="Custom mapping for categorical values")
    
    @model_validator(mode='after')
    def validate_custom_mapping(self):
        if self.method == EncodingMethod.CUSTOM and self.custom_mapping is None:
            raise ValueError("custom_mapping must be provided when method is 'custom'")
        return self


class CategoricalEncodingResponse(BaseTransformationResponse):
    """Response model for categorical encoding transformation"""
    encoded_columns: List[str]
    encoding_mappings: Dict[str, Any] = Field(default_factory=dict)


# Text Preprocessing Models
class TextPreprocessingRequest(ColumnBasedRequest):
    """Request model for text preprocessing transformation"""
    method: TextMethod


class TextPreprocessingResponse(BaseTransformationResponse):
    """Response model for text preprocessing transformation"""
    processed_columns: List[str]


# TF-IDF Models
class TFIDFRequest(BaseTransformationRequest):
    """Request model for TF-IDF transformation"""
    column: str = Field(description="Text column to transform")
    max_features: Optional[int] = Field(default=500, description="Maximum number of features")
    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    
    @field_validator('max_features')
    @classmethod
    def validate_max_features(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_features must be a positive integer")
        return v


class TFIDFResponse(BaseTransformationResponse):
    """Response model for TF-IDF transformation"""
    tfidf_columns: List[str]
    feature_names: List[str] = Field(default_factory=list)


# PCA Transform Models
class PCATransformRequest(ColumnBasedRequest):
    """Request model for PCA transformation"""
    n_components: int = Field(description="Number of principal components")
    whiten: bool = Field(default=False, description="Whiten the components")
    
    @field_validator('n_components')
    @classmethod
    def validate_n_components(cls, v):
        if v <= 0:
            raise ValueError("n_components must be a positive integer")
        return v


class PCATransformResponse(BaseTransformationResponse):
    """Response model for PCA transformation"""
    principal_components: List[str]
    explained_variance_ratio: List[float] = Field(default_factory=list)


# Transformation Step for Batch Processing
class TransformationStep(BaseModel):
    """Individual transformation step for batch processing"""
    model_config = ConfigDict(populate_by_name=True)
    
    # Support both action/params and function/args patterns
    action: Optional[str] = Field(default=None, alias="function", description="Transformation action/function name")
    params: Optional[Dict[str, Any]] = Field(default=None, alias="args", description="Parameters/arguments for transformation")
    
    @model_validator(mode='after')
    def validate_step(self):
        if self.action is None:
            raise ValueError("Either 'action' or 'function' must be provided")
        if self.params is None:
            self.params = {}
        return self


# Batch Transformation Models
class BatchTransformationRequest(BaseTransformationRequest):
    """Request model for batch transformation"""
    transformations: List[TransformationStep] = Field(description="List of transformations to apply")
    
    @field_validator('transformations')
    @classmethod
    def validate_transformations_not_empty(cls, v):
        if not v:
            raise ValueError("At least one transformation must be specified")
        return v


class BatchTransformationResponse(BaseTransformationResponse):
    """Response model for batch transformation"""
    applied_transformations: List[str]
    transformation_results: Dict[str, Any] = Field(default_factory=dict)


# Custom Transformation Models
class CustomTransformationRequest(BaseTransformationRequest):
    """Request model for custom transformation"""
    function_name: str = Field(description="Name of the custom function")
    function_code: str = Field(description="Python code for the custom function")
    
    @field_validator('function_name')
    @classmethod
    def validate_function_name(cls, v):
        if not v.strip():
            raise ValueError("function_name cannot be empty")
        # Basic validation for valid Python identifier
        if not v.replace('_', 'a').isalnum() or v[0].isdigit():
            raise ValueError("function_name must be a valid Python identifier")
        return v
    
    @field_validator('function_code')
    @classmethod
    def validate_function_code(cls, v):
        if not v.strip():
            raise ValueError("function_code cannot be empty")
        return v


class CustomTransformationResponse(BaseTransformationResponse):
    """Response model for custom transformation"""
    function_name: str
    execution_details: Dict[str, Any] = Field(default_factory=dict)