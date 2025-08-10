# Schemas for transformation
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple, Literal


class BaseTransformationResponse(BaseModel):
    """Base response model that all transformation responses inherit from"""
    status: str
    message: Optional[str] = None


# Date Extraction Models
class DateExtractionRequest(BaseModel):
    """Request model for date extraction transformation"""
    dataframe: Dict[str, List[Any]]
    column: str
    components: List[str] = Field(description="List of date components to extract (year, month, day, hour, minute, second, weekday)")


class DateExtractionResponse(BaseTransformationResponse):
    """Response model for date extraction transformation"""
    transformed_df: Dict[str, List[Any]]
    added_columns: List[str]


# Scaling Models
class ScalingRequest(BaseModel):
    """Request model for scaling transformation"""
    dataframe: Dict[str, List[Any]]
    columns: List[str]
    method: Literal["minmax", "zscore", "log", "robust", "custom"]
    custom_range: Optional[Tuple[float, float]] = None


class ScalingResponse(BaseTransformationResponse):
    """Response model for scaling transformation"""
    transformed_df: Dict[str, List[Any]]
    scaled_columns: List[str]


# Categorical Encoding Models
class CategoricalEncodingRequest(BaseModel):
    """Request model for categorical encoding transformation"""
    dataframe: Dict[str, List[Any]]
    columns: List[str]
    method: Literal["onehot", "label", "frequency", "custom"]
    custom_mapping: Optional[Dict[str, Any]] = None


class CategoricalEncodingResponse(BaseTransformationResponse):
    """Response model for categorical encoding transformation"""
    transformed_df: Dict[str, List[Any]]
    encoded_columns: List[str]


# Text Preprocessing Models
class TextPreprocessingRequest(BaseModel):
    """Request model for text preprocessing transformation"""
    dataframe: Dict[str, List[Any]]
    column: str
    method: Literal["stem", "lemmatize"]


class TextPreprocessingResponse(BaseTransformationResponse):
    """Response model for text preprocessing transformation"""
    transformed_df: Dict[str, List[Any]]
    processed_column: str


# TF-IDF Models
class TFIDFRequest(BaseModel):
    """Request model for TF-IDF transformation"""
    dataframe: Dict[str, List[Any]]
    column: str
    max_features: Optional[int] = 500
    lowercase: bool = True
    remove_stopwords: bool = True


class TFIDFResponse(BaseTransformationResponse):
    """Response model for TF-IDF transformation"""
    transformed_df: Dict[str, List[Any]]
    tfidf_columns: List[str]


# PCA Transform Models
class PCATransformRequest(BaseModel):
    """Request model for PCA transformation"""
    dataframe: Dict[str, List[Any]]
    columns: List[str]
    n_components: int
    whiten: Optional[bool] = False


class PCATransformResponse(BaseTransformationResponse):
    """Response model for PCA transformation"""
    transformed_df: Dict[str, List[Any]]
    principal_components: List[str]


# Custom Transformation Models
class CustomTransformationResponse(BaseTransformationResponse):
    """Response model for custom transformation"""
    transformed_df: Dict[str, List[Any]]