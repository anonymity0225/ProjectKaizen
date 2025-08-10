# File I/O utility functions 
import pandas as pd
import os
import tempfile
import logging
import json
from pathlib import Path
from fastapi import UploadFile, HTTPException
from typing import Optional
from app.core.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file to a temporary directory with collision prevention and validation."""
    # Validate file size before processing
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413, 
            detail=f"File size ({file_size_mb:.1f}MB) exceeds limit ({settings.MAX_FILE_SIZE_MB}MB)"
        )
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: {file_extension}. Supported formats: {list(settings.SUPPORTED_FILE_EXTENSIONS)}"
        )
    
    # Reset file pointer for saving
    await file.seek(0)
    
    temp_dir = make_temp_dir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    logger.info(f"Saving uploaded file to: {temp_path} ({file_size_mb:.1f}MB)")
    
    with open(temp_path, "wb") as buffer:
        buffer.write(content)
    
    logger.info(f"Successfully saved file: {temp_path} ({len(content)} bytes)")
    return temp_path

def make_temp_dir() -> str:
    """Create a unique temporary directory to prevent collisions."""
    # Ensure temp directory exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="preprocess_", dir=settings.TEMP_DIR)
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_file(path: str) -> None:
    """Clean up temporary files and directories."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"Cleaned up temporary file: {path}")
            
            # Also try to remove the parent directory if it's empty
            parent_dir = os.path.dirname(path)
            try:
                os.rmdir(parent_dir)
                logger.info(f"Cleaned up temporary directory: {parent_dir}")
            except OSError:
                # Directory not empty or other error, ignore
                pass
                
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
            logger.info(f"Cleaned up temporary directory: {path}")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary path {path}: {str(e)}")

def _validate_file_size(file_path: str, max_size_mb: int = None) -> None:
    """Validate file size before processing."""
    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")

def _validate_dataframe_limits(df: pd.DataFrame) -> None:
    """Validate DataFrame row and column limits."""
    if len(df) > settings.MAX_ROWS:
        raise ValueError(f"DataFrame has {len(df)} rows, exceeding limit ({settings.MAX_ROWS})")
    
    if len(df.columns) > settings.MAX_COLUMNS:
        raise ValueError(f"DataFrame has {len(df.columns)} columns, exceeding limit ({settings.MAX_COLUMNS})")

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file with logging, validation, and stream safety for large files."""
    logger.info(f"Loading CSV file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Use chunked reading for large files to prevent OOM
        if file_size_mb > 50:
            logger.info(f"Large CSV file detected ({file_size_mb:.1f}MB), using chunked reading")
            chunks = []
            for chunk in pd.read_csv(path, chunksize=10_000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {path}: {str(e)}")
        raise

def load_excel(path: str, engine: Optional[str] = None) -> pd.DataFrame:
    """Load Excel file with fallback engine support for malformed files."""
    logger.info(f"Loading Excel file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        # Try default engine first
        df = pd.read_excel(path, engine=engine)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded Excel: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        if engine is None:
            # Try with openpyxl engine as fallback for malformed files
            logger.warning(f"Default engine failed for {path}: {str(e)}. Trying openpyxl engine...")
            try:
                df = pd.read_excel(path, engine="openpyxl")
                
                # Validate DataFrame limits
                _validate_dataframe_limits(df)
                
                logger.info(f"Successfully loaded Excel with openpyxl: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
            except Exception as fallback_e:
                logger.error(f"Both default and openpyxl engines failed for {path}: {str(fallback_e)}")
                raise fallback_e
        else:
            logger.error(f"Failed to load Excel file {path} with engine {engine}: {str(e)}")
            raise

def load_json(path: str, orient: str = "records") -> pd.DataFrame:
    """Load JSON file with logging and validation."""
    logger.info(f"Loading JSON file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        df = pd.read_json(path, orient=orient)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded JSON: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load JSON file {path}: {str(e)}")
        raise

def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet file with logging and validation."""
    logger.info(f"Loading Parquet file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        df = pd.read_parquet(path)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded Parquet: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load Parquet file {path}: {str(e)}")
        raise

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with logging."""
    logger.info(f"Saving CSV file: {path}")
    try:
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved CSV: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file {path}: {str(e)}")
        raise

def save_excel(df: pd.DataFrame, path: str, engine: str = "openpyxl", sheet_name: str = "Sheet1") -> None:
    """Save DataFrame to Excel with logging."""
    logger.info(f"Saving Excel file: {path}")
    try:
        df.to_excel(path, index=False, engine=engine, sheet_name=sheet_name)
        logger.info(f"Successfully saved Excel: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except Exception as e:
        logger.error(f"Failed to save Excel file {path}: {str(e)}")
        raise

def save_json(df: pd.DataFrame, path: str, orient: str = "records", indent: int = 2) -> None:
    """Save DataFrame to JSON with logging."""
    logger.info(f"Saving JSON file: {path}")
    try:
        df.to_json(path, orient=orient, indent=indent)
        logger.info(f"Successfully saved JSON: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {path}: {str(e)}")
        raise

def save_parquet(df: pd.DataFrame, path: str, compression: str = "snappy") -> None:
    """Save DataFrame to Parquet with logging."""
    logger.info(f"Saving Parquet file: {path}")
    try:
        df.to_parquet(path, compression=compression, index=False)
        logger.info(f"Successfully saved Parquet: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except Exception as e:
        logger.error(f"Failed to save Parquet file {path}: {str(e)}")
        raise