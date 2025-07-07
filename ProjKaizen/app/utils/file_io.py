# File I/O utility functions 
import pandas as pd
import os
import tempfile
import logging
from pathlib import Path
from fastapi import UploadFile
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file to a temporary directory with collision prevention."""
    temp_dir = make_temp_dir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    logger.info(f"Saving uploaded file to: {temp_path}")
    
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    logger.info(f"Successfully saved file: {temp_path} ({len(content)} bytes)")
    return temp_path

def make_temp_dir() -> str:
    """Create a unique temporary directory to prevent collisions."""
    temp_dir = tempfile.mkdtemp(prefix="preprocess_")
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

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file with logging."""
    logger.info(f"Loading CSV file: {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {path}: {str(e)}")
        raise

def load_excel(path: str, engine: Optional[str] = None) -> pd.DataFrame:
    """Load Excel file with fallback engine support for malformed files."""
    logger.info(f"Loading Excel file: {path}")
    
    try:
        # Try default engine first
        df = pd.read_excel(path, engine=engine)
        logger.info(f"Successfully loaded Excel: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        if engine is None:
            # Try with openpyxl engine as fallback for malformed files
            logger.warning(f"Default engine failed for {path}: {str(e)}. Trying openpyxl engine...")
            try:
                df = pd.read_excel(path, engine="openpyxl")
                logger.info(f"Successfully loaded Excel with openpyxl: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
            except Exception as fallback_e:
                logger.error(f"Both default and openpyxl engines failed for {path}: {str(fallback_e)}")
                raise fallback_e
        else:
            logger.error(f"Failed to load Excel file {path} with engine {engine}: {str(e)}")
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