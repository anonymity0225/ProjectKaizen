"""
Visualization service logic for Kaizen enterprise data platform.

This module provides service-layer functionality for generating various types of
data visualizations including histograms, correlation heatmaps, scatter plots,
box plots, and time-series plots.
"""

import base64
import io
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Tuple
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import UploadFile

from config import settings
from schemas.visualization import (
    PlotConfig,
    VisualizationResponse,
    HistogramRequest,
    CorrelationHeatmapRequest,
    ScatterPlotRequest,
    BoxPlotRequest,
    TimeSeriesRequest
)

# Suppress matplotlib warnings in production
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
BASE_STORAGE_PATH = Path(settings.STORAGE_BASE_PATH) if hasattr(settings, 'STORAGE_BASE_PATH') else Path("/tmp")
PLOTS_DIR = BASE_STORAGE_PATH / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_FULL_PROCESSING = 50000
SAMPLE_SIZE_LARGE_DATASET = 10000
MAX_CATEGORIES_HISTOGRAM = 50
MAX_CORRELATION_COLUMNS = 100


class ValidationError(ValueError):
    """Custom exception for data validation errors."""
    pass


class ProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


def _log_operation(operation_type: str, columns: List[str], duration: float, 
                  file_path: Optional[str] = None, sample_size: Optional[int] = None,
                  metadata: Optional[dict] = None) -> None:
    """Log visualization operation with structured JSON format."""
    log_data = {
        "operation": "visualization",
        "type": operation_type,
        "columns": columns,
        "duration_seconds": round(duration, 3),
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if file_path:
        log_data["output_path"] = str(file_path)
    if sample_size:
        log_data["sample_size"] = sample_size
    if metadata:
        log_data["metadata"] = metadata
    
    logger.info(json.dumps(log_data))


def _validate_path_safety(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file path is safe and within allowed boundaries.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Path: Validated and resolved path
        
    Raises:
        ValidationError: If path is unsafe
    """
    path = Path(file_path).resolve()
    
    # Check for path traversal attempts
    if ".." in str(file_path):
        raise ValidationError("Path traversal not allowed")
    
    # Check if path is absolute and outside base storage
    if path.is_absolute() and not str(path).startswith(str(BASE_STORAGE_PATH.resolve())):
        raise ValidationError(f"Absolute paths outside {BASE_STORAGE_PATH} not allowed")
    
    return path


def _generate_filename(prefix: str, extension: str = "png") -> str:
    """Generate timestamped filename for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    return f"{prefix}_{timestamp}.{extension}"


def load_dataset(file_or_path: Union[str, Path, UploadFile]) -> pd.DataFrame:
    """
    Load dataset from file path or uploaded file.
    
    Args:
        file_or_path: File path string, Path object, or FastAPI UploadFile
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        ValidationError: If file cannot be loaded or is invalid
    """
    start_time = time.time()
    
    try:
        if isinstance(file_or_path, UploadFile):
            # Handle uploaded file
            return _load_from_upload(file_or_path)
        else:
            # Handle file path
            file_path = _validate_path_safety(file_or_path)
            return _load_from_path(file_path)
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to load dataset: {str(e)} (duration: {duration:.3f}s)")
        raise ValidationError(f"Failed to load dataset: {str(e)}")


def _load_from_upload(file: UploadFile) -> pd.DataFrame:
    """Load dataset from uploaded file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Determine file type and load accordingly
        filename = file.filename.lower() if file.filename else ""
        
        if filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_path)
        else:
            # Try CSV first, then Excel
            try:
                df = pd.read_csv(temp_path)
            except:
                df = pd.read_excel(temp_path)
        
        if df.empty:
            raise ValidationError("Dataset contains no data")
        
        return df
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def _load_from_path(file_path: Path) -> pd.DataFrame:
    """Load dataset from file path."""
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValidationError(f"Unsupported file format: {file_extension}")
        
        if df.empty:
            raise ValidationError("Dataset contains no data")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValidationError("Dataset contains no data")
    except Exception as e:
        raise ValidationError(f"Failed to read file: {str(e)}")


def validate_columns(df: pd.DataFrame, columns: List[str], 
                    expected_type: str = 'any') -> None:
    """
    Validate that columns exist and have expected data types.
    
    Args:
        df: DataFrame to validate
        columns: List of column names to check
        expected_type: 'numeric', 'categorical', 'datetime', or 'any'
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        available = list(df.columns)
        raise ValidationError(
            f"Columns not found: {missing_columns}. Available columns: {available}"
        )
    
    # Check data types if specified
    if expected_type == 'numeric':
        non_numeric = [
            col for col in columns 
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValidationError(
                f"Columns must be numeric: {non_numeric}"
            )
    
    elif expected_type == 'categorical':
        # Categorical can be object, category, or low-cardinality numeric
        invalid_categorical = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > MAX_CATEGORIES_HISTOGRAM:
                invalid_categorical.append(col)
        if invalid_categorical:
            raise ValidationError(
                f"Columns have too many unique values for categorical analysis: {invalid_categorical}"
            )
    
    elif expected_type == 'datetime':
        non_datetime = []
        for col in columns:
            try:
                pd.to_datetime(df[col])
            except:
                non_datetime.append(col)
        if non_datetime:
            raise ValidationError(
                f"Columns cannot be converted to datetime: {non_datetime}"
            )
    
    # Check for all null columns
    all_null_columns = [
        col for col in columns 
        if df[col].isna().all()
    ]
    if all_null_columns:
        raise ValidationError(
            f"Columns contain only null values: {all_null_columns}"
        )


def _sample_large_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Sample large datasets for performance and return sampling metadata.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (sampled_df, metadata_dict)
    """
    if len(df) <= MAX_ROWS_FULL_PROCESSING:
        return df, {"sampled": False, "original_size": len(df)}
    
    # Use stratified sampling if possible, otherwise random
    sample_size = min(SAMPLE_SIZE_LARGE_DATASET, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    metadata = {
        "sampled": True,
        "original_size": len(df),
        "sample_size": sample_size,
        "sampling_method": "random"
    }
    
    return sampled_df, metadata


def render_and_save_plot(fig: plt.Figure, output_path: str, 
                        return_base64: bool = True, skip_base64: bool = False,
                        dpi: int = 100) -> dict:
    """
    Render plot to file and optionally return base64 string.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save plot
        return_base64: Whether to generate base64 string
        skip_base64: Whether to skip base64 generation (overrides return_base64)
        dpi: DPI for rendering
        
    Returns:
        dict: Contains 'file_path' and optionally 'base64_image'
    """
    result = {}
    
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply tight layout
        fig.tight_layout()
        
        # Save to file
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        result['file_path'] = str(output_path)
        
        # Generate base64 if requested and not skipped
        if return_base64 and not skip_base64:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            result['base64_image'] = f"data:image/png;base64,{image_base64}"
            buffer.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error rendering plot: {str(e)}")
        raise ProcessingError(f"Failed to render plot: {str(e)}")
    
    finally:
        plt.close(fig)


def generate_histogram(request: HistogramRequest) -> VisualizationResponse:
    """
    Generate histogram visualization.
    
    Args:
        request: Histogram request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.column])
        
        # Sample large datasets
        df, sampling_metadata = _sample_large_dataset(df)
        
        # Configure plot style
        sns.set_style(request.config.style)
        plt.rcParams.update({'font.size': 10})
        
        # Create figure
        fig, ax = plt.subplots(figsize=request.config.figsize)
        
        # Determine plot type based on data
        column_data = df[request.column]
        
        if pd.api.types.is_numeric_dtype(column_data):
            # Numeric histogram
            sns.histplot(
                data=df, 
                x=request.column, 
                kde=request.config.kde,
                bins=request.bins or 'auto',
                ax=ax
            )
        else:
            # Categorical count plot
            if column_data.nunique() > MAX_CATEGORIES_HISTOGRAM:
                raise ValidationError(
                    f"Column '{request.column}' has {column_data.nunique()} unique values, "
                    f"which exceeds the maximum of {MAX_CATEGORIES_HISTOGRAM} for histogram display"
                )
            
            sns.countplot(data=df, x=request.column, ax=ax)
            
            # Rotate labels if many categories
            if column_data.nunique() > 10:
                plt.xticks(rotation=45, ha='right')
        
        # Customize labels and title
        ax.set_xlabel(request.column.replace('_', ' ').title())
        ax.set_ylabel('Count')
        
        if request.config.title:
            ax.set_title(request.config.title)
        else:
            ax.set_title(f"Distribution of {request.column.replace('_', ' ').title()}")
        
        # Generate output path
        output_filename = _generate_filename("histogram")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64, 
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "column": request.column,
            "data_type": str(df[request.column].dtype),
            "null_count": int(df[request.column].isna().sum()),
            "unique_values": int(df[request.column].nunique()),
            "total_values": len(df),
            **sampling_metadata
        }
        
        # Log operation
        _log_operation("histogram", [request.column], duration, 
                      render_result['file_path'], 
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Histogram generated successfully for column '{request.column}'",
            metadata=metadata
        )
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        _log_operation("histogram", [request.column], duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"column": request.column}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in histogram generation: {str(e)}")
        _log_operation("histogram", [request.column], duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during histogram generation",
            metadata={"column": request.column}
        )


def generate_correlation_heatmap(request: CorrelationHeatmapRequest) -> VisualizationResponse:
    """
    Generate correlation heatmap visualization.
    
    Args:
        request: Correlation heatmap request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load data
        df = load_dataset(request.file_or_path)
        
        # Select numeric columns
        if request.columns:
            validate_columns(df, request.columns, 'numeric')
            numeric_df = df[request.columns]
        else:
            numeric_df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
        
        if numeric_df.empty:
            raise ValidationError("No numeric columns found for correlation analysis")
        
        if numeric_df.shape[1] < 2:
            raise ValidationError("At least 2 numeric columns required for correlation analysis")
        
        if numeric_df.shape[1] > MAX_CORRELATION_COLUMNS:
            raise ValidationError(
                f"Too many columns ({numeric_df.shape[1]}) for correlation heatmap. "
                f"Maximum allowed: {MAX_CORRELATION_COLUMNS}"
            )
        
        # Sample large datasets
        numeric_df, sampling_metadata = _sample_large_dataset(numeric_df)
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=request.correlation_method)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Calculate figure size based on number of columns
        n_cols = len(corr_matrix.columns)
        figsize = (max(8, n_cols * 0.6), max(6, n_cols * 0.5))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": 0.8},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        # Customize layout
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if request.config.title:
            ax.set_title(request.config.title)
        else:
            ax.set_title("Correlation Matrix")
        
        # Generate output path
        output_filename = _generate_filename("correlation_heatmap")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "numeric_columns": list(numeric_df.columns),
            "correlation_method": request.correlation_method,
            "matrix_shape": corr_matrix.shape,
            **sampling_metadata
        }
        
        # Log operation
        _log_operation("correlation_heatmap", list(numeric_df.columns), 
                      duration, render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Correlation heatmap generated successfully for {len(numeric_df.columns)} columns",
            metadata=metadata
        )
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = request.columns or []
        _log_operation("correlation_heatmap", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"columns": columns}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in correlation heatmap generation: {str(e)}")
        columns = request.columns or []
        _log_operation("correlation_heatmap", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during correlation heatmap generation",
            metadata={"columns": columns}
        )


def generate_scatter_plot(request: ScatterPlotRequest) -> VisualizationResponse:
    """
    Generate scatter plot visualization.
    
    Args:
        request: Scatter plot request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.x_column, request.y_column], 'numeric')
        
        # Check for sufficient valid data
        valid_data = df[[request.x_column, request.y_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.x_column}' and '{request.y_column}'"
            )
        
        # Sample large datasets
        valid_data, sampling_metadata = _sample_large_dataset(valid_data)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=request.config.figsize)
        
        # Create scatter plot
        scatter_kwargs = {'alpha': 0.6, 's': 30}
        
        if request.color_column and request.color_column in df.columns:
            # Color by category or continuous variable
            sns.scatterplot(
                data=valid_data, 
                x=request.x_column, 
                y=request.y_column,
                hue=request.color_column,
                ax=ax,
                **scatter_kwargs
            )
        else:
            sns.scatterplot(
                data=valid_data,
                x=request.x_column,
                y=request.y_column,
                ax=ax,
                **scatter_kwargs
            )
        
        # Customize labels
        ax.set_xlabel(request.x_column.replace('_', ' ').title())
        ax.set_ylabel(request.y_column.replace('_', ' ').title())
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        if request.config.title:
            ax.set_title(request.config.title)
        else:
            title = f"{request.x_column.replace('_', ' ').title()} vs {request.y_column.replace('_', ' ').title()}"
            ax.set_title(title)
        
        # Generate output path
        output_filename = _generate_filename("scatter_plot")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate correlation
        correlation = valid_data[request.x_column].corr(valid_data[request.y_column])
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "x_column": request.x_column,
            "y_column": request.y_column,
            "color_column": request.color_column,
            "correlation": round(correlation, 4) if not pd.isna(correlation) else None,
            "valid_points": len(valid_data),
            "original_points": len(df),
            **sampling_metadata
        }
        
        # Log operation
        columns = [request.x_column, request.y_column]
        if request.color_column:
            columns.append(request.color_column)
        
        _log_operation("scatter_plot", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Scatter plot generated successfully for '{request.x_column}' vs '{request.y_column}'",
            metadata=metadata
        )
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.x_column, request.y_column]
        _log_operation("scatter_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"x_column": request.x_column, "y_column": request.y_column}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in scatter plot generation: {str(e)}")
        columns = [request.x_column, request.y_column]
        _log_operation("scatter_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during scatter plot generation",
            metadata={"x_column": request.x_column, "y_column": request.y_column}
        )


def generate_box_plot(request: BoxPlotRequest) -> VisualizationResponse:
    """
    Generate box plot visualization.
    
    Args:
        request: Box plot request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.numeric_column], 'numeric')
        validate_columns(df, [request.categorical_column], 'categorical')
        
        # Check for sufficient data
        valid_data = df[[request.numeric_column, request.categorical_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.numeric_column}' and '{request.categorical_column}'"
            )
        
        # Check category count
        unique_categories = valid_data[request.categorical_column].nunique()
        if unique_categories > MAX_CATEGORIES_HISTOGRAM:
            raise ValidationError(
                f"Too many categories ({unique_categories}) in '{request.categorical_column}'. "
                f"Maximum allowed: {MAX_CATEGORIES_HISTOGRAM}"
            )
        
        # Sample large datasets
        valid_data, sampling_metadata = _sample_large_dataset(valid_data)
        
        # Configure plot style
        sns.set_style(request.config.style)
        
        # Create figure with adjusted width for many categories
        width_factor = max(1, unique_categories * 0.8)
        figsize = (min(16, request.config.figsize[0] * width_factor), request.config.figsize[1])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        sns.boxplot(
            data=valid_data,
            x=request.categorical_column,
            y=request.numeric_column,
            ax=ax
        )
        
        # Customize labels
        ax.set_xlabel(request.categorical_column.replace('_', ' ').title())
        ax.set_ylabel(request.numeric_column.replace('_', ' ').title())
        
        # Rotate x-axis labels if many categories
        if unique_categories > 5:
            plt.xticks(rotation=45, ha='right')
        
        if request.config.title:
            ax.set_title(request.config.title)
        else:
            title = f"Distribution of {request.numeric_column.replace('_', ' ').title()} by {request.categorical_column.replace('_', ' ').title()}"
            ax.set_title(title)
        
        # Generate output path
        output_filename = _generate_filename("box_plot")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate metadata
        duration = time.time() - start_time
        metadata = {
            "numeric_column": request.numeric_column,
            "categorical_column": request.categorical_column,
            "unique_categories": unique_categories,
            "valid_points": len(valid_data),
            "original_points": len(df),
            **sampling_metadata
        }
        
        # Log operation
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Box plot generated successfully for '{request.numeric_column}' by '{request.categorical_column}'",
            metadata=metadata
        )
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={
                "numeric_column": request.numeric_column,
                "categorical_column": request.categorical_column
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in box plot generation: {str(e)}")
        columns = [request.numeric_column, request.categorical_column]
        _log_operation("box_plot", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during box plot generation",
            metadata={
                "numeric_column": request.numeric_column,
                "categorical_column": request.categorical_column
            }
        )


def generate_time_series_plot(request: TimeSeriesRequest) -> VisualizationResponse:
    """
    Generate time series line plot visualization.
    
    Args:
        request: Time series request configuration
        
    Returns:
        VisualizationResponse: Generated plot response
    """
    start_time = time.time()
    
    try:
        # Load and validate data
        df = load_dataset(request.file_or_path)
        validate_columns(df, [request.datetime_column], 'datetime')
        validate_columns(df, [request.value_column], 'numeric')
        
        # Convert datetime column
        df[request.datetime_column] = pd.to_datetime(df[request.datetime_column])
        
        # Check for sufficient valid data
        valid_data = df[[request.datetime_column, request.value_column]].dropna()
        if valid_data.empty:
            raise ValidationError(
                f"No valid data points found for columns '{request.datetime_column}' and '{request.value_column}'"
            )
        
        # Sort by datetime
        valid_data = valid_data.sort_values(request.datetime_column)
        
        # Sample large datasets while maintaining time order
        if len(valid_data) > MAX_ROWS_FULL_PROCESSING:
            # For time series, use systematic sampling to maintain temporal structure
            step_size = len(valid_data) // SAMPLE_SIZE_LARGE_DATASET
            sample_indices = range(0, len(valid_data), max(1, step_size))
            valid_data = valid_data.iloc[list(sample_indices)]
            
            sampling_metadata = {
                "sampled": True,
                "original_size": len(df),
                "sample_size": len(valid_data),
                "sampling_method": "systematic_temporal"
            }
        else:
            sampling_metadata = {"sampled": False, "original_size": len(valid_data)}
        
        # Configure plot style
        sns.set_style(request.config.style)
        plt.rcParams.update({'axes.grid': True})
        
        # Create figure
        fig, ax = plt.subplots(figsize=request.config.figsize)
        
        # Create time series plot
        if request.group_column and request.group_column in df.columns:
            # Multiple series grouped by category
            for group_name, group_data in valid_data.groupby(request.group_column):
                ax.plot(
                    group_data[request.datetime_column],
                    group_data[request.value_column],
                    label=str(group_name),
                    linewidth=1.5,
                    alpha=0.8
                )
            ax.legend(title=request.group_column.replace('_', ' ').title())
        else:
            # Single time series
            ax.plot(
                valid_data[request.datetime_column],
                valid_data[request.value_column],
                linewidth=1.5,
                color='steelblue'
            )
        
        # Customize labels and formatting
        ax.set_xlabel(request.datetime_column.replace('_', ' ').title())
        ax.set_ylabel(request.value_column.replace('_', ' ').title())
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if request.config.title:
            ax.set_title(request.config.title)
        else:
            title = f"Time Series: {request.value_column.replace('_', ' ').title()}"
            if request.group_column:
                title += f" by {request.group_column.replace('_', ' ').title()}"
            ax.set_title(title)
        
        # Generate output path
        output_filename = _generate_filename("time_series")
        output_path = PLOTS_DIR / output_filename
        
        # Render plot
        render_result = render_and_save_plot(
            fig, output_path, request.config.return_base64,
            request.config.skip_base64, request.config.dpi
        )
        
        # Calculate time range and metadata
        time_range = {
            "start": valid_data[request.datetime_column].min().isoformat(),
            "end": valid_data[request.datetime_column].max().isoformat(),
            "duration_days": (valid_data[request.datetime_column].max() - 
                             valid_data[request.datetime_column].min()).days
        }
        
        duration = time.time() - start_time
        metadata = {
            "datetime_column": request.datetime_column,
            "value_column": request.value_column,
            "group_column": request.group_column,
            "time_range": time_range,
            "data_points": len(valid_data),
            "original_points": len(df),
            **sampling_metadata
        }
        
        # Log operation
        columns = [request.datetime_column, request.value_column]
        if request.group_column:
            columns.append(request.group_column)
        
        _log_operation("time_series", columns, duration,
                      render_result['file_path'],
                      sampling_metadata.get('sample_size'), metadata)
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Time series plot generated successfully for '{request.value_column}' over time",
            metadata=metadata
        )
        
    except (ValidationError, ProcessingError) as e:
        duration = time.time() - start_time
        columns = [request.datetime_column, request.value_column]
        _log_operation("time_series", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={
                "datetime_column": request.datetime_column,
                "value_column": request.value_column
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error in time series plot generation: {str(e)}")
        columns = [request.datetime_column, request.value_column]
        _log_operation("time_series", columns, duration, metadata={"error": str(e)})
        return VisualizationResponse(
            status="error",
            message="An unexpected error occurred during time series plot generation",
            metadata={
                "datetime_column": request.datetime_column,
                "value_column": request.value_column
            }
        )


# Utility functions for backward compatibility and additional functionality

def get_dataset_info(file_or_path: Union[str, Path, UploadFile]) -> dict:
    """
    Get basic information about a dataset without generating visualizations.
    
    Args:
        file_or_path: File path or UploadFile object
        
    Returns:
        dict: Dataset information including columns, types, and basic stats
    """
    try:
        df = load_dataset(file_or_path)
        
        # Basic info
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Categorize columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        
        # Try to identify datetime columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except:
                pass
        
        info.update({
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "recommended_plots": _get_plot_recommendations(df, numeric_cols, categorical_cols, datetime_cols)
        })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise ValidationError(f"Failed to analyze dataset: {str(e)}")


def _get_plot_recommendations(df: pd.DataFrame, numeric_cols: List[str], 
                             categorical_cols: List[str], datetime_cols: List[str]) -> List[dict]:
    """Generate plot recommendations based on dataset characteristics."""
    recommendations = []
    
    # Histogram recommendations
    for col in numeric_cols[:5]:  # Limit recommendations
        recommendations.append({
            "plot_type": "histogram",
            "description": f"Distribution analysis of {col}",
            "parameters": {"column": col}
        })
    
    for col in categorical_cols[:3]:
        if df[col].nunique() <= MAX_CATEGORIES_HISTOGRAM:
            recommendations.append({
                "plot_type": "histogram",
                "description": f"Count distribution of {col}",
                "parameters": {"column": col}
            })
    
    # Correlation heatmap
    if len(numeric_cols) >= 2:
        recommendations.append({
            "plot_type": "correlation_heatmap",
            "description": f"Correlation analysis of {len(numeric_cols)} numeric variables",
            "parameters": {"columns": numeric_cols}
        })
    
    # Scatter plot recommendations
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i+1:4]:
            recommendations.append({
                "plot_type": "scatter_plot",
                "description": f"Relationship between {col1} and {col2}",
                "parameters": {"x_column": col1, "y_column": col2}
            })
    
    # Box plot recommendations
    for num_col in numeric_cols[:2]:
        for cat_col in categorical_cols[:2]:
            if df[cat_col].nunique() <= MAX_CATEGORIES_HISTOGRAM:
                recommendations.append({
                    "plot_type": "box_plot",
                    "description": f"Distribution of {num_col} across {cat_col} categories",
                    "parameters": {"numeric_column": num_col, "categorical_column": cat_col}
                })
    
    # Time series recommendations
    for datetime_col in datetime_cols[:2]:
        for value_col in numeric_cols[:2]:
            recommendations.append({
                "plot_type": "time_series",
                "description": f"Temporal trend of {value_col}",
                "parameters": {"datetime_column": datetime_col, "value_column": value_col}
            })
    
    return recommendations[:10]  # Limit total recommendations


def cleanup_old_plots(days_old: int = 7) -> int:
    """
    Clean up old plot files to manage disk space.
    
    Args:
        days_old: Remove plots older than this many days
        
    Returns:
        int: Number of files removed
    """
    try:
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for plot_file in PLOTS_DIR.glob("*.png"):
            if plot_file.stat().st_mtime < cutoff_time:
                plot_file.unlink()
                removed_count += 1
                logger.info(f"Removed old plot file: {plot_file}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error during plot cleanup: {str(e)}")
        return 0


# Initialize matplotlib backend for headless operation
plt.switch_backend('Agg')