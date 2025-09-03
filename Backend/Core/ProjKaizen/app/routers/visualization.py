 #routers/visualization.py
"""
Visualization router for Kaizen enterprise data platform.

This module provides FastAPI endpoints for generating various types of
data visualizations with JWT authentication and comprehensive error handling.
"""

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

from auth.dependencies import get_current_user
from schemas.user import User
from schemas.visualization import (
    VisualizationResponse,
    HistogramRequest,
    CorrelationHeatmapRequest,
    ScatterPlotRequest,
    BoxPlotRequest,
    TimeSeriesRequest,
    PlotConfig
)
from services.visualization import (
    generate_histogram,
    generate_correlation_heatmap,
    generate_scatter_plot,
    generate_box_plot,
    generate_time_series_plot,
    ValidationError,
    ProcessingError
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI grouping
router = APIRouter(prefix="/visualization", tags=["Visualization"])


def _log_request(user_id: str, plot_type: str, columns: list, 
                dataset_ref: Optional[str], duration: float, 
                success: bool = True, error: Optional[str] = None) -> None:
    """Log visualization request with structured metadata."""
    log_data = {
        "event": "visualization_request",
        "user_id": user_id,
        "plot_type": plot_type,
        "columns": columns,
        "dataset_ref": dataset_ref,
        "duration_seconds": round(duration, 3),
        "success": success,
        "timestamp": time.time()
    }
    
    if error:
        log_data["error"] = error
    
    if success:
        logger.info(json.dumps(log_data))
    else:
        logger.error(json.dumps(log_data))


def _handle_file_or_path(upload_file: Optional[UploadFile], data_ref: Optional[str]):
    """
    Validate and return either upload file or data reference.
    
    Args:
        upload_file: Optional uploaded file
        data_ref: Optional file path or data reference
        
    Returns:
        The file or path to process
        
    Raises:
        HTTPException: If neither or both are provided
    """
    if upload_file and data_ref:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either upload_file or data_ref, not both"
        )
    
    if not upload_file and not data_ref:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either upload_file or data_ref must be provided"
        )
    
    return upload_file if upload_file else data_ref


def _handle_service_exceptions(e: Exception, plot_type: str, user_id: str, 
                              columns: list, dataset_ref: Optional[str], 
                              start_time: float) -> HTTPException:
    """
    Handle service exceptions and convert to appropriate HTTP responses.
    
    Args:
        e: Exception raised by service
        plot_type: Type of plot being generated
        user_id: User ID for logging
        columns: Columns involved in the request
        dataset_ref: Dataset reference for logging
        start_time: Request start time for duration calculation
        
    Returns:
        HTTPException: Appropriate HTTP exception
    """
    duration = time.time() - start_time
    
    if isinstance(e, (ValidationError, ValueError)):
        _log_request(user_id, plot_type, columns, dataset_ref, duration, 
                    success=False, error=str(e))
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    elif isinstance(e, ProcessingError):
        _log_request(user_id, plot_type, columns, dataset_ref, duration, 
                    success=False, error=str(e))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )
    else:
        logger.exception(f"Unexpected error in {plot_type} generation")
        _log_request(user_id, plot_type, columns, dataset_ref, duration, 
                    success=False, error="Internal server error")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during visualization generation"
        )


@router.post("/histogram", response_model=VisualizationResponse)
async def create_histogram(
    column: str = Form(..., description="Column name for histogram"),
    bins: Optional[int] = Form(None, description="Number of bins for histogram"),
    title: Optional[str] = Form(None, description="Plot title"),
    figsize_width: int = Form(10, description="Figure width"),
    figsize_height: int = Form(6, description="Figure height"),
    return_base64: bool = Form(True, description="Return base64 encoded image"),
    skip_base64: bool = Form(False, description="Skip base64 generation for large images"),
    kde: bool = Form(False, description="Add KDE overlay for numeric histograms"),
    style: str = Form("whitegrid", description="Seaborn style"),
    dpi: int = Form(100, description="Image DPI"),
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> VisualizationResponse:
    """
    Generate histogram visualization for a specified column.
    
    Creates a histogram plot showing the distribution of values in the specified column.
    Supports both numeric and categorical data with automatic plot type selection.
    """
    start_time = time.time()
    plot_type = "histogram"
    columns = [column]
    
    try:
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Create plot configuration
        config = PlotConfig(
            title=title,
            figsize=(figsize_width, figsize_height),
            return_base64=return_base64,
            skip_base64=skip_base64,
            kde=kde,
            style=style,
            dpi=dpi
        )
        
        # Create request object
        request = HistogramRequest(
            file_or_path=file_or_path,
            column=column,
            bins=bins,
            config=config
        )
        
        # Generate histogram
        result = generate_histogram(request)
        
        # Log successful request
        duration = time.time() - start_time
        _log_request(current_user.id, plot_type, columns, dataset_ref, duration)
        
        return result
        
    except Exception as e:
        raise _handle_service_exceptions(e, plot_type, current_user.id, columns, 
                                       data_ref, start_time)


@router.post("/correlation", response_model=VisualizationResponse)
async def create_correlation_heatmap(
    columns: Optional[str] = Form(None, description="Comma-separated list of columns (all numeric if not specified)"),
    correlation_method: str = Form("pearson", description="Correlation method: pearson, spearman, or kendall"),
    title: Optional[str] = Form(None, description="Plot title"),
    figsize_width: int = Form(10, description="Figure width"),
    figsize_height: int = Form(8, description="Figure height"),
    return_base64: bool = Form(True, description="Return base64 encoded image"),
    skip_base64: bool = Form(False, description="Skip base64 generation for large images"),
    style: str = Form("whitegrid", description="Seaborn style"),
    dpi: int = Form(100, description="Image DPI"),
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> VisualizationResponse:
    """
    Generate correlation heatmap for numeric columns.
    
    Creates a correlation matrix heatmap showing relationships between numeric variables.
    Supports different correlation methods and automatic column selection.
    """
    start_time = time.time()
    plot_type = "correlation_heatmap"
    
    try:
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Parse columns list
        column_list = [col.strip() for col in columns.split(",")] if columns else None
        log_columns = column_list or ["all_numeric"]
        
        # Create plot configuration
        config = PlotConfig(
            title=title,
            figsize=(figsize_width, figsize_height),
            return_base64=return_base64,
            skip_base64=skip_base64,
            style=style,
            dpi=dpi
        )
        
        # Create request object
        request = CorrelationHeatmapRequest(
            file_or_path=file_or_path,
            columns=column_list,
            correlation_method=correlation_method,
            config=config
        )
        
        # Generate correlation heatmap
        result = generate_correlation_heatmap(request)
        
        # Log successful request
        duration = time.time() - start_time
        _log_request(current_user.id, plot_type, log_columns, dataset_ref, duration)
        
        return result
        
    except Exception as e:
        raise _handle_service_exceptions(e, plot_type, current_user.id, 
                                       log_columns if 'log_columns' in locals() else [],
                                       data_ref, start_time)


@router.post("/scatter", response_model=VisualizationResponse)
async def create_scatter_plot(
    x_column: str = Form(..., description="X-axis column name"),
    y_column: str = Form(..., description="Y-axis column name"),
    color_column: Optional[str] = Form(None, description="Optional column for color grouping"),
    title: Optional[str] = Form(None, description="Plot title"),
    figsize_width: int = Form(10, description="Figure width"),
    figsize_height: int = Form(6, description="Figure height"),
    return_base64: bool = Form(True, description="Return base64 encoded image"),
    skip_base64: bool = Form(False, description="Skip base64 generation for large images"),
    style: str = Form("whitegrid", description="Seaborn style"),
    dpi: int = Form(100, description="Image DPI"),
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> VisualizationResponse:
    """
    Generate scatter plot for two numeric columns.
    
    Creates a scatter plot showing the relationship between two numeric variables.
    Optionally supports color grouping by a third categorical variable.
    """
    start_time = time.time()
    plot_type = "scatter_plot"
    columns = [x_column, y_column]
    if color_column:
        columns.append(color_column)
    
    try:
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Create plot configuration
        config = PlotConfig(
            title=title,
            figsize=(figsize_width, figsize_height),
            return_base64=return_base64,
            skip_base64=skip_base64,
            style=style,
            dpi=dpi
        )
        
        # Create request object
        request = ScatterPlotRequest(
            file_or_path=file_or_path,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            config=config
        )
        
        # Generate scatter plot
        result = generate_scatter_plot(request)
        
        # Log successful request
        duration = time.time() - start_time
        _log_request(current_user.id, plot_type, columns, dataset_ref, duration)
        
        return result
        
    except Exception as e:
        raise _handle_service_exceptions(e, plot_type, current_user.id, columns, 
                                       data_ref, start_time)


@router.post("/boxplot", response_model=VisualizationResponse)
async def create_box_plot(
    numeric_column: str = Form(..., description="Numeric column for box plot"),
    categorical_column: str = Form(..., description="Categorical column for grouping"),
    title: Optional[str] = Form(None, description="Plot title"),
    figsize_width: int = Form(12, description="Figure width"),
    figsize_height: int = Form(6, description="Figure height"),
    return_base64: bool = Form(True, description="Return base64 encoded image"),
    skip_base64: bool = Form(False, description="Skip base64 generation for large images"),
    style: str = Form("whitegrid", description="Seaborn style"),
    dpi: int = Form(100, description="Image DPI"),
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> VisualizationResponse:
    """
    Generate box plot showing distribution of numeric column by categories.
    
    Creates box plots showing the distribution of a numeric variable across
    different categories of a categorical variable.
    """
    start_time = time.time()
    plot_type = "box_plot"
    columns = [numeric_column, categorical_column]
    
    try:
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Create plot configuration
        config = PlotConfig(
            title=title,
            figsize=(figsize_width, figsize_height),
            return_base64=return_base64,
            skip_base64=skip_base64,
            style=style,
            dpi=dpi
        )
        
        # Create request object
        request = BoxPlotRequest(
            file_or_path=file_or_path,
            numeric_column=numeric_column,
            categorical_column=categorical_column,
            config=config
        )
        
        # Generate box plot
        result = generate_box_plot(request)
        
        # Log successful request
        duration = time.time() - start_time
        _log_request(current_user.id, plot_type, columns, dataset_ref, duration)
        
        return result
        
    except Exception as e:
        raise _handle_service_exceptions(e, plot_type, current_user.id, columns, 
                                       data_ref, start_time)


@router.post("/timeseries", response_model=VisualizationResponse)
async def create_time_series_plot(
    datetime_column: str = Form(..., description="Datetime column for time axis"),
    value_column: str = Form(..., description="Numeric column for values"),
    group_column: Optional[str] = Form(None, description="Optional column for grouping multiple series"),
    title: Optional[str] = Form(None, description="Plot title"),
    figsize_width: int = Form(12, description="Figure width"),
    figsize_height: int = Form(6, description="Figure height"),
    return_base64: bool = Form(True, description="Return base64 encoded image"),
    skip_base64: bool = Form(False, description="Skip base64 generation for large images"),
    style: str = Form("whitegrid", description="Seaborn style"),
    dpi: int = Form(100, description="Image DPI"),
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> VisualizationResponse:
    """
    Generate time series line plot for temporal data.
    
    Creates a line plot showing how a numeric variable changes over time.
    Optionally supports multiple series grouped by a categorical variable.
    """
    start_time = time.time()
    plot_type = "time_series"
    columns = [datetime_column, value_column]
    if group_column:
        columns.append(group_column)
    
    try:
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Create plot configuration
        config = PlotConfig(
            title=title,
            figsize=(figsize_width, figsize_height),
            return_base64=return_base64,
            skip_base64=skip_base64,
            style=style,
            dpi=dpi
        )
        
        # Create request object
        request = TimeSeriesRequest(
            file_or_path=file_or_path,
            datetime_column=datetime_column,
            value_column=value_column,
            group_column=group_column,
            config=config
        )
        
        # Generate time series plot
        result = generate_time_series_plot(request)
        
        # Log successful request
        duration = time.time() - start_time
        _log_request(current_user.id, plot_type, columns, dataset_ref, duration)
        
        return result
        
    except Exception as e:
        raise _handle_service_exceptions(e, plot_type, current_user.id, columns, 
                                       data_ref, start_time)


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for visualization service."""
    return {"status": "healthy", "service": "visualization"}


# Additional utility endpoints for enhanced functionality

@router.post("/dataset-info", tags=["Utility"])
async def get_dataset_info(
    data_ref: Optional[str] = Form(None, description="Data file path or reference"),
    upload_file: Optional[UploadFile] = File(None, description="Upload data file"),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Get dataset information and visualization recommendations.
    
    Analyzes the dataset structure and provides information about columns,
    data types, and recommended visualization types.
    """
    start_time = time.time()
    
    try:
        from services.visualization import get_dataset_info
        
        # Validate file/path input
        file_or_path = _handle_file_or_path(upload_file, data_ref)
        dataset_ref = data_ref or (upload_file.filename if upload_file else None)
        
        # Get dataset information
        info = get_dataset_info(file_or_path)
        
        # Log request
        duration = time.time() - start_time
        _log_request(current_user.id, "dataset_info", [], dataset_ref, duration)
        
        return info
        
    except Exception as e:
        duration = time.time() - start_time
        if isinstance(e, (ValidationError, ValueError)):
            _log_request(current_user.id, "dataset_info", [], data_ref, duration, 
                        success=False, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
        else:
            logger.exception("Unexpected error in dataset info")
            _log_request(current_user.id, "dataset_info", [], data_ref, duration, 
                        success=False, error="Internal server error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while analyzing dataset"
            )
