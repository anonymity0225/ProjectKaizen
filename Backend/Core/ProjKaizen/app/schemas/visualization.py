# Schemas for visualization 
"""
Pydantic schemas for visualization requests and responses in Kaizen.

This module defines all data models used for visualization API endpoints,
including request validation, response formatting, and configuration options.
"""

from typing import Optional, Tuple, Dict, Any, Union, List, Literal
from pathlib import Path

from pydantic import BaseModel, validator, Field
from fastapi import UploadFile


class PlotConfig(BaseModel):
    """Configuration options for plot generation."""
    
    title: Optional[str] = Field(
        None,
        description="Plot title. If not provided, a default title will be generated.",
        example="Distribution of Customer Ages"
    )
    
    figsize: Optional[Tuple[int, int]] = Field(
        (10, 6),
        description="Figure size as (width, height) in inches. Both values must be positive.",
        example=(12, 8)
    )
    
    dpi: Optional[int] = Field(
        100,
        description="Image resolution in dots per inch. Must be between 50 and 300.",
        example=150
    )
    
    style: Optional[str] = Field(
        "whitegrid",
        description="Seaborn style for the plot.",
        example="darkgrid"
    )
    
    return_base64: bool = Field(
        True,
        description="Whether to return base64 encoded image data in the response."
    )
    
    skip_base64: bool = Field(
        False,
        description="Skip base64 generation to avoid memory issues with large images."
    )
    
    kde: bool = Field(
        False,
        description="Add KDE (kernel density estimation) overlay for numeric histograms."
    )
    
    output_path: Optional[str] = Field(
        None,
        description="Custom output path for saved plot. Must be relative path within storage root.",
        example="plots/customer_analysis/age_distribution.png"
    )
    
    @validator('figsize')
    def validate_figsize(cls, v):
        """Validate that figsize contains positive integers."""
        if v is not None:
            width, height = v
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError("Figure dimensions must be integers")
            if width <= 0 or height <= 0:
                raise ValueError("Figure dimensions must be positive")
            if width > 50 or height > 50:
                raise ValueError("Figure dimensions must not exceed 50 inches")
        return v
    
    @validator('dpi')
    def validate_dpi(cls, v):
        """Validate DPI is within acceptable range."""
        if v is not None:
            if not isinstance(v, int):
                raise ValueError("DPI must be an integer")
            if v < 50 or v > 300:
                raise ValueError("DPI must be between 50 and 300")
        return v
    
    @validator('style')
    def validate_style(cls, v):
        """Validate seaborn style is in allowed set."""
        allowed_styles = {
            'darkgrid', 'whitegrid', 'dark', 'white', 'ticks',
            'default', 'classic', 'seaborn', 'ggplot', 'bmh',
            'fivethirtyeight', 'grayscale', 'seaborn-bright',
            'seaborn-colorblind', 'seaborn-dark', 'seaborn-darkgrid',
            'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
            'seaborn-paper', 'seaborn-pastel', 'seaborn-poster',
            'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
            'seaborn-whitegrid'
        }
        
        if v is not None and v not in allowed_styles:
            raise ValueError(f"Style must be one of: {', '.join(sorted(allowed_styles))}")
        return v
    
    @validator('output_path')
    def validate_output_path(cls, v):
        """Validate output path is safe (no path traversal)."""
        if v is not None:
            # Check for path traversal attempts
            if '..' in v:
                raise ValueError("Path traversal (..) not allowed in output_path")
            
            # Convert to Path object for validation
            path = Path(v)
            
            # Check if it's an absolute path
            if path.is_absolute():
                raise ValueError("Absolute paths not allowed in output_path")
            
            # Check for suspicious path components
            parts = path.parts
            for part in parts:
                if part.startswith('.'):
                    raise ValueError("Hidden directories/files not allowed in output_path")
        
        return v
    
    @validator('skip_base64')
    def validate_skip_base64_with_return_base64(cls, v, values):
        """Validate skip_base64 logic with return_base64."""
        if v and values.get('return_base64', True):
            # This is allowed - skip_base64 overrides return_base64
            pass
        return v

    class Config:
        schema_extra = {
            "example": {
                "title": "Customer Age Distribution",
                "figsize": (12, 8),
                "dpi": 150,
                "style": "whitegrid",
                "return_base64": True,
                "skip_base64": False,
                "kde": False,
                "output_path": "plots/analysis/customer_age_hist.png"
            }
        }


class VisualizationResponse(BaseModel):
    """Response model for visualization operations."""
    
    status: Literal['success', 'error'] = Field(
        description="Status of the visualization operation"
    )
    
    message: str = Field(
        description="Human-readable message describing the result"
    )
    
    plot_path: Optional[str] = Field(
        None,
        description="File path where the plot was saved"
    )
    
    base64_image: Optional[str] = Field(
        None,
        description="Base64 encoded image data (if return_base64=True and skip_base64=False)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the visualization operation"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Histogram generated successfully for column 'age'",
                "plot_path": "/storage/plots/histogram_20241215_143022.png",
                "base64_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "metadata": {
                    "column": "age",
                    "data_type": "int64",
                    "null_count": 5,
                    "unique_values": 45,
                    "total_values": 1000,
                    "sampled": False,
                    "original_size": 1000
                }
            }
        }


class HistogramRequest(BaseModel):
    """Request model for histogram generation."""
    
    file_or_path: Union[str, UploadFile] = Field(
        description="File path or UploadFile object containing the data"
    )
    
    column: str = Field(
        description="Column name for which to generate the histogram"
    )
    
    bins: Optional[int] = Field(
        None,
        description="Number of bins for numeric histograms. If not specified, automatic binning is used.",
        gt=1,
        le=200
    )
    
    config: Optional[PlotConfig] = Field(
        default_factory=PlotConfig,
        description="Plot configuration options"
    )

    class Config:
        schema_extra = {
            "example": {
                "column": "age",
                "bins": 20,
                "config": {
                    "title": "Age Distribution",
                    "figsize": (10, 6),
                    "style": "whitegrid",
                    "kde": True
                }
            }
        }


class CorrelationHeatmapRequest(BaseModel):
    """Request model for correlation heatmap generation."""
    
    file_or_path: Union[str, UploadFile] = Field(
        description="File path or UploadFile object containing the data"
    )
    
    columns: Optional[List[str]] = Field(
        None,
        description="List of column names to include in correlation analysis. If not specified, all numeric columns are used."
    )
    
    correlation_method: str = Field(
        "pearson",
        description="Correlation method to use"
    )
    
    config: Optional[PlotConfig] = Field(
        default_factory=PlotConfig,
        description="Plot configuration options"
    )
    
    @validator('correlation_method')
    def validate_correlation_method(cls, v):
        """Validate correlation method is supported."""
        allowed_methods = {'pearson', 'spearman', 'kendall'}
        if v not in allowed_methods:
            raise ValueError(f"Correlation method must be one of: {', '.join(allowed_methods)}")
        return v
    
    @validator('columns')
    def validate_columns_list(cls, v):
        """Validate columns list is not empty if provided."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Columns list cannot be empty if provided")
            if len(v) > 100:
                raise ValueError("Too many columns specified (maximum 100)")
            # Check for duplicates
            if len(v) != len(set(v)):
                raise ValueError("Duplicate column names not allowed")
        return v

    class Config:
        schema_extra = {
            "example": {
                "columns": ["age", "income", "score", "rating"],
                "correlation_method": "pearson",
                "config": {
                    "title": "Feature Correlation Matrix",
                    "figsize": (10, 8),
                    "style": "white"
                }
            }
        }


class ScatterPlotRequest(BaseModel):
    """Request model for scatter plot generation."""
    
    file_or_path: Union[str, UploadFile] = Field(
        description="File path or UploadFile object containing the data"
    )
    
    x_column: str = Field(
        description="Column name for x-axis (must be numeric)"
    )
    
    y_column: str = Field(
        description="Column name for y-axis (must be numeric)"
    )
    
    color_column: Optional[str] = Field(
        None,
        description="Optional column name for color grouping/gradient"
    )
    
    config: Optional[PlotConfig] = Field(
        default_factory=PlotConfig,
        description="Plot configuration options"
    )
    
    @validator('y_column')
    def validate_different_columns(cls, v, values):
        """Validate that x and y columns are different."""
        if 'x_column' in values and v == values['x_column']:
            raise ValueError("x_column and y_column must be different")
        return v

    class Config:
        schema_extra = {
            "example": {
                "x_column": "height",
                "y_column": "weight",
                "color_column": "gender",
                "config": {
                    "title": "Height vs Weight by Gender",
                    "figsize": (10, 6),
                    "style": "whitegrid"
                }
            }
        }


class BoxPlotRequest(BaseModel):
    """Request model for box plot generation."""
    
    file_or_path: Union[str, UploadFile] = Field(
        description="File path or UploadFile object containing the data"
    )
    
    numeric_column: str = Field(
        description="Numeric column name for the box plot values"
    )
    
    categorical_column: str = Field(
        description="Categorical column name for grouping"
    )
    
    config: Optional[PlotConfig] = Field(
        default_factory=PlotConfig,
        description="Plot configuration options"
    )
    
    @validator('categorical_column')
    def validate_different_columns(cls, v, values):
        """Validate that numeric and categorical columns are different."""
        if 'numeric_column' in values and v == values['numeric_column']:
            raise ValueError("numeric_column and categorical_column must be different")
        return v

    class Config:
        schema_extra = {
            "example": {
                "numeric_column": "salary",
                "categorical_column": "department",
                "config": {
                    "title": "Salary Distribution by Department",
                    "figsize": (12, 6),
                    "style": "whitegrid"
                }
            }
        }


class TimeSeriesRequest(BaseModel):
    """Request model for time series plot generation."""
    
    file_or_path: Union[str, UploadFile] = Field(
        description="File path or UploadFile object containing the data"
    )
    
    datetime_column: str = Field(
        description="Column name containing datetime values for the time axis"
    )
    
    value_column: str = Field(
        description="Numeric column name for the values to plot over time"
    )
    
    group_column: Optional[str] = Field(
        None,
        description="Optional categorical column for creating multiple time series lines"
    )
    
    config: Optional[PlotConfig] = Field(
        default_factory=PlotConfig,
        description="Plot configuration options"
    )
    
    @validator('value_column')
    def validate_different_datetime_value(cls, v, values):
        """Validate that datetime and value columns are different."""
        if 'datetime_column' in values and v == values['datetime_column']:
            raise ValueError("datetime_column and value_column must be different")
        return v
    
    @validator('group_column')
    def validate_group_column_different(cls, v, values):
        """Validate that group column is different from datetime and value columns."""
        if v is not None:
            if 'datetime_column' in values and v == values['datetime_column']:
                raise ValueError("group_column must be different from datetime_column")
            if 'value_column' in values and v == values['value_column']:
                raise ValueError("group_column must be different from value_column")
        return v

    class Config:
        schema_extra = {
            "example": {
                "datetime_column": "date",
                "value_column": "sales",
                "group_column": "region",
                "config": {
                    "title": "Sales Trends by Region",
                    "figsize": (14, 8),
                    "style": "whitegrid"
                }
            }
        }


# Additional utility schemas for enhanced functionality

class DatasetInfo(BaseModel):
    """Schema for dataset information and recommendations."""
    
    shape: Tuple[int, int] = Field(
        description="Dataset shape as (rows, columns)"
    )
    
    columns: List[str] = Field(
        description="List of all column names"
    )
    
    dtypes: Dict[str, str] = Field(
        description="Data types for each column"
    )
    
    null_counts: Dict[str, int] = Field(
        description="Number of null values per column"
    )
    
    numeric_columns: List[str] = Field(
        description="List of numeric column names"
    )
    
    categorical_columns: List[str] = Field(
        description="List of categorical column names"
    )
    
    datetime_columns: List[str] = Field(
        description="List of datetime column names"
    )
    
    recommended_plots: List[Dict[str, Any]] = Field(
        description="List of recommended plot types and configurations"
    )
    
    memory_usage: int = Field(
        description="Memory usage of the dataset in bytes"
    )

    class Config:
        schema_extra = {
            "example": {
                "shape": (1000, 8),
                "columns": ["id", "name", "age", "salary", "department", "hire_date", "rating", "active"],
                "dtypes": {
                    "id": "int64",
                    "name": "object", 
                    "age": "int64",
                    "salary": "float64",
                    "department": "object",
                    "hire_date": "datetime64[ns]",
                    "rating": "float64",
                    "active": "bool"
                },
                "null_counts": {
                    "id": 0,
                    "name": 2,
                    "age": 5,
                    "salary": 10,
                    "department": 0,
                    "hire_date": 3,
                    "rating": 15,
                    "active": 0
                },
                "numeric_columns": ["age", "salary", "rating"],
                "categorical_columns": ["name", "department"],
                "datetime_columns": ["hire_date"],
                "recommended_plots": [
                    {
                        "plot_type": "histogram",
                        "description": "Distribution analysis of age",
                        "parameters": {"column": "age"}
                    },
                    {
                        "plot_type": "correlation_heatmap",
                        "description": "Correlation analysis of numeric variables",
                        "parameters": {"columns": ["age", "salary", "rating"]}
                    }
                ],
                "memory_usage": 64000
            }
        }


class PlotRecommendation(BaseModel):
    """Schema for individual plot recommendations."""
    
    plot_type: str = Field(
        description="Type of recommended plot"
    )
    
    description: str = Field(
        description="Human-readable description of the recommendation"
    )
    
    parameters: Dict[str, Any] = Field(
        description="Suggested parameters for the plot"
    )
    
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the recommendation (0-1)",
        ge=0.0,
        le=1.0
    )

    class Config:
        schema_extra = {
            "example": {
                "plot_type": "scatter_plot",
                "description": "Explore relationship between height and weight",
                "parameters": {
                    "x_column": "height",
                    "y_column": "weight",
                    "color_column": "gender"
                },
                "confidence": 0.9
            }
        }