import base64
import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import UploadFile
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path("/tmp/visuals")
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)


class PlotConfig(BaseModel):
    """Configuration for plot generation."""
    title: Optional[str] = None
    figsize: tuple[int, int] = (10, 6)
    output_path: Optional[str] = None
    return_base64: bool = True
    kde: bool = False  # For histogram KDE smoothing
    dpi: int = 100
    style: str = "whitegrid"  # seaborn style


class VisualizationResponse(BaseModel):
    """Response model for visualization operations."""
    status: str  # "success" | "error"
    plot_path: Optional[str] = None
    base64_image: Optional[str] = None
    message: str
    metadata: dict = {}


def validate_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists and has appropriate data.
    
    Args:
        df: DataFrame to check
        column: Column name to validate
        
    Raises:
        ValueError: If column doesn't exist or has issues
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}")
    
    if df[column].isna().all():
        raise ValueError(f"Column '{column}' contains only null values")


def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists and is numeric.
    
    Args:
        df: DataFrame to check
        column: Column name to validate
        
    Raises:
        ValueError: If column doesn't exist or isn't numeric
    """
    validate_column(df, column)
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric and cannot be used for this plot type")


def generate_filename(prefix: str = "plot", extension: str = "png") -> str:
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


async def handle_upload_file(file: UploadFile) -> pd.DataFrame:
    """
    Handle uploaded file and convert to DataFrame.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file cannot be processed
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Read CSV
            df = pd.read_csv(temp_path)
            
            if df.empty:
                raise ValueError("Uploaded file contains no data")
                
            return df
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise ValueError(f"Failed to process uploaded file: {str(e)}")


def render_plot(
    plt_obj: plt.Figure,
    title: str,
    output_path: Optional[str] = None,
    return_base64: bool = True,
    dpi: int = 100
) -> dict:
    """
    Render plot to file and/or base64 string.
    
    Args:
        plt_obj: Matplotlib figure object
        title: Plot title
        output_path: Optional file path to save
        return_base64: Whether to return base64 encoded image
        dpi: DPI for image rendering
        
    Returns:
        dict: Contains 'file_path' and/or 'base64_image'
    """
    result = {}
    
    try:
        # Set title if provided
        if title:
            plt_obj.suptitle(title)
        
        # Apply tight layout
        plt_obj.tight_layout()
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt_obj.savefig(output_path, dpi=dpi, bbox_inches='tight')
            result['file_path'] = str(output_path)
        
        # Generate base64 if requested
        if return_base64:
            buffer = io.BytesIO()
            plt_obj.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            result['base64_image'] = f"data:image/png;base64,{image_base64}"
            buffer.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error rendering plot: {str(e)}")
        raise ValueError(f"Failed to render plot: {str(e)}")
    
    finally:
        plt.close(plt_obj)


def create_histogram(
    df: pd.DataFrame,
    column: str,
    config: PlotConfig
) -> VisualizationResponse:
    """
    Create histogram for specified column.
    
    Args:
        df: Input DataFrame
        column: Column name for histogram
        config: Plot configuration
        
    Returns:
        VisualizationResponse: Result with plot data
    """
    try:
        # Validate column
        validate_column(df, column)
        
        # Check if column can be used for histogram
        if not (pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
            if df[column].dtype == 'object' and df[column].nunique() > 50:
                raise ValueError(f"Column '{column}' has too many unique values for histogram. Consider using a different plot type.")
        
        # Set seaborn style
        sns.set_style(config.style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=config.figsize)
        
        # Create histogram
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(data=df, x=column, kde=config.kde, ax=ax)
        else:
            # For categorical data
            sns.countplot(data=df, x=column, ax=ax)
            # Rotate labels if many categories
            if df[column].nunique() > 10:
                plt.xticks(rotation=45, ha='right')
        
        # Set labels
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_ylabel('Count')
        
        # Generate output path if not provided
        output_path = config.output_path
        if not output_path and not config.return_base64:
            output_path = str(DEFAULT_OUTPUT_DIR / generate_filename("histogram"))
        
        # Render plot
        render_result = render_plot(
            fig,
            config.title or f"Histogram of {column}",
            output_path,
            config.return_base64,
            config.dpi
        )
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Histogram created successfully for column '{column}'",
            metadata={
                "column": column,
                "data_type": str(df[column].dtype),
                "null_count": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique()),
                "figsize": config.figsize
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating histogram for column '{column}': {str(e)}")
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"column": column}
        )


def create_correlation_heatmap(
    df: pd.DataFrame,
    config: PlotConfig
) -> VisualizationResponse:
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        config: Plot configuration
        
    Returns:
        VisualizationResponse: Result with plot data
    """
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found in dataset for correlation analysis")
        
        if numeric_df.shape[1] < 2:
            raise ValueError("At least 2 numeric columns required for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Set seaborn style
        sns.set_style(config.style)
        
        # Create figure with adjusted size for readability
        n_cols = len(corr_matrix.columns)
        figsize = (max(8, n_cols * 0.8), max(6, n_cols * 0.6))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        # Improve layout
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Generate output path if not provided
        output_path = config.output_path
        if not output_path and not config.return_base64:
            output_path = str(DEFAULT_OUTPUT_DIR / generate_filename("correlation_heatmap"))
        
        # Render plot
        render_result = render_plot(
            fig,
            config.title or "Correlation Heatmap",
            output_path,
            config.return_base64,
            config.dpi
        )
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Correlation heatmap created successfully for {len(numeric_df.columns)} numeric columns",
            metadata={
                "numeric_columns": list(numeric_df.columns),
                "correlation_shape": corr_matrix.shape,
                "figsize": figsize
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"available_columns": list(df.columns)}
        )


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    config: PlotConfig
) -> VisualizationResponse:
    """
    Create scatter plot for two columns.
    
    Args:
        df: Input DataFrame
        x_column: X-axis column name
        y_column: Y-axis column name
        config: Plot configuration
        
    Returns:
        VisualizationResponse: Result with plot data
    """
    try:
        # Validate both columns are numeric
        validate_numeric_column(df, x_column)
        validate_numeric_column(df, y_column)
        
        # Check for sufficient data
        valid_data = df[[x_column, y_column]].dropna()
        if valid_data.empty:
            raise ValueError(f"No valid data points found for columns '{x_column}' and '{y_column}'")
        
        # Set seaborn style
        sns.set_style(config.style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=config.figsize)
        
        # Create scatter plot
        sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax, alpha=0.7)
        
        # Set labels
        ax.set_xlabel(x_column.replace('_', ' ').title())
        ax.set_ylabel(y_column.replace('_', ' ').title())
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Generate output path if not provided
        output_path = config.output_path
        if not output_path and not config.return_base64:
            output_path = str(DEFAULT_OUTPUT_DIR / generate_filename("scatter_plot"))
        
        # Render plot
        render_result = render_plot(
            fig,
            config.title or f"Scatter Plot: {x_column} vs {y_column}",
            output_path,
            config.return_base64,
            config.dpi
        )
        
        # Calculate correlation coefficient
        correlation = df[x_column].corr(df[y_column])
        
        return VisualizationResponse(
            status="success",
            plot_path=render_result.get('file_path'),
            base64_image=render_result.get('base64_image'),
            message=f"Scatter plot created successfully for '{x_column}' vs '{y_column}'",
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "correlation": round(correlation, 3) if not pd.isna(correlation) else None,
                "valid_points": len(valid_data),
                "total_points": len(df),
                "figsize": config.figsize
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating scatter plot for '{x_column}' vs '{y_column}': {str(e)}")
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"x_column": x_column, "y_column": y_column}
        )


# Convenience functions for backward compatibility and FastAPI integration
async def create_histogram_from_upload(
    file: UploadFile,
    column: str,
    config: Optional[PlotConfig] = None
) -> VisualizationResponse:
    """Create histogram from uploaded file."""
    try:
        df = await handle_upload_file(file)
        config = config or PlotConfig()
        return create_histogram(df, column, config)
    except Exception as e:
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"column": column}
        )


async def create_correlation_heatmap_from_upload(
    file: UploadFile,
    config: Optional[PlotConfig] = None
) -> VisualizationResponse:
    """Create correlation heatmap from uploaded file."""
    try:
        df = await handle_upload_file(file)
        config = config or PlotConfig()
        return create_correlation_heatmap(df, config)
    except Exception as e:
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={}
        )


async def create_scatter_plot_from_upload(
    file: UploadFile,
    x_column: str,
    y_column: str,
    config: Optional[PlotConfig] = None
) -> VisualizationResponse:
    """Create scatter plot from uploaded file."""
    try:
        df = await handle_upload_file(file)
        config = config or PlotConfig()
        return create_scatter_plot(df, x_column, y_column, config)
    except Exception as e:
        return VisualizationResponse(
            status="error",
            message=str(e),
            metadata={"x_column": x_column, "y_column": y_column}
        )