# File I/O utility functions 
import pandas as pd
import os
import tempfile
import logging
import json
import shutil
import re
from uuid import uuid4
from pathlib import Path
from fastapi import UploadFile, HTTPException
from typing import Optional
from app.core.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

# Optional: Add content type detection for security
try:
    import filetype
    HAS_FILETYPE = True
except ImportError:
    try:
        import magic
        HAS_MAGIC = True
        HAS_FILETYPE = False
    except ImportError:
        HAS_MAGIC = HAS_FILETYPE = False
        logger.warning("Neither filetype nor python-magic is available. Content type validation disabled.")


def secure_filename(filename: str) -> str:
    """
    Sanitize filename by removing path separators and unsafe characters.
    Returns a safe filename suitable for use in filesystem operations.
    """
    if not filename:
        return "upload"
    
    # Remove path separators and potentially dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename.replace('/', '').replace('\\', ''))
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure we have something left
    if not filename:
        return "upload"
    
    # Limit length to prevent filesystem issues
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def _validate_content_type(file_path: Path, declared_extension: str) -> bool:
    """
    Validate that file content matches declared extension using MIME detection.
    Returns True if validation passes or is unavailable.
    """
    if not (HAS_FILETYPE or HAS_MAGIC):
        logger.debug("Content type validation skipped - no detection library available")
        return True
    
    try:
        if HAS_FILETYPE:
            kind = filetype.guess(str(file_path))
            if kind is None and declared_extension.lower() not in ['.csv', '.json', '.txt']:
                logger.warning(f"Could not detect file type for {file_path}")
                return False
            
            if kind and f".{kind.extension}" != declared_extension.lower():
                logger.warning(f"Extension mismatch: expected {declared_extension}, detected .{kind.extension}")
                return False
                
        elif HAS_MAGIC:
            detected_type = magic.from_file(str(file_path), mime=True)
            # Basic MIME type mappings for common formats
            extension_mime_map = {
                '.csv': ['text/csv', 'text/plain'],
                '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
                '.xls': ['application/vnd.ms-excel'],
                '.json': ['application/json', 'text/plain'],
                '.parquet': ['application/octet-stream']
            }
            expected_mimes = extension_mime_map.get(declared_extension.lower(), [])
            if expected_mimes and detected_type not in expected_mimes:
                logger.warning(f"Content type mismatch: expected one of {expected_mimes}, got {detected_type}")
                return False
                
        return True
        
    except Exception as e:
        logger.warning(f"Content type validation failed: {e}")
        return True  # Fail open for availability


async def save_temp_file(file: UploadFile) -> Path:
    """
    Save uploaded file to a temporary directory with streaming, collision prevention, and validation.
    Returns Path to saved file.
    
    Raises:
        HTTPException(413): File size exceeds limit
        HTTPException(415): Unsupported file type or MIME mismatch
        HTTPException(507): Insufficient disk space
        HTTPException(500): Unexpected I/O errors
    """
    if not file.filename:
        raise HTTPException(status_code=415, detail="No filename provided")
    
    # Validate file extension early
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: {file_extension}. "
                  f"Supported formats: {list(settings.SUPPORTED_FILE_EXTENSIONS)}"
        )
    
    # Create isolated temp directory
    base = Path(settings.TEMP_DIR)
    base.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=base))
    tmp_dir.chmod(0o700)
    
    # Build sanitized filename
    safe_filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
    temp_path = tmp_dir / safe_filename
    
    logger.info(f"Saving uploaded file to: {temp_path}")
    
    # Stream file to disk to avoid memory issues
    total_size = 0
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    chunk_size = 4 * 1024 * 1024  # 4MB chunks
    
    try:
        with open(temp_path, "wb") as buffer:
            # Set secure file permissions (owner read/write only)
            temp_path.chmod(0o600)
            
            # Stream in chunks to control memory usage
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                    
                total_size += len(chunk)
                
                # Check size limit during streaming
                if total_size > max_size_bytes:
                    # Clean up partial file
                    temp_path.unlink(missing_ok=True)
                    tmp_dir.rmdir()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds limit ({settings.MAX_FILE_SIZE_MB}MB)"
                    )
                
                buffer.write(chunk)
    
    except OSError as e:
        # Handle disk full or permission errors
        logger.error(f"OS error writing file {temp_path}: {e}")
        cleanup_temp_file(temp_path)
        
        if e.errno == 28:  # ENOSPC - No space left on device
            raise HTTPException(status_code=507, detail="Insufficient disk space to save file")
        elif e.errno == 13:  # EACCES - Permission denied
            raise HTTPException(status_code=500, detail="Permission denied writing to temporary directory")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    except Exception as e:
        cleanup_temp_file(temp_path)
        logger.error(f"Unexpected error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Validate content type matches extension
    if not _validate_content_type(temp_path, file_extension):
        cleanup_temp_file(temp_path)
        raise HTTPException(
            status_code=415,
            detail=f"File content does not match declared format {file_extension}"
        )
    
    file_size_mb = total_size / (1024 * 1024)
    logger.info(f"Successfully saved file: {temp_path} ({file_size_mb:.1f}MB)")
    return temp_path


def cleanup_temp_file(path: Path) -> None:
    """Clean up temporary files and directories with robust error handling."""
    if not path:
        return
        
    try:
        if path.is_file():
            path.unlink()
            logger.info(f"Cleaned up temporary file: {path}")
            
            # Try to remove the parent directory if it's empty and looks like our temp dir
            parent_dir = path.parent
            if parent_dir and parent_dir.name.startswith('tmp'):
                try:
                    parent_dir.rmdir()
                    logger.info(f"Cleaned up temporary directory: {parent_dir}")
                except OSError as e:
                    # Directory not empty, in use, or permission issue - log but don't fail
                    logger.debug(f"Could not remove temp directory {parent_dir}: {e}")
                    
        elif path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Cleaned up temporary directory: {path}")
            
    except FileNotFoundError:
        # Already deleted, no problem
        logger.debug(f"Temp path {path} already cleaned up")
    except PermissionError as e:
        logger.warning(f"Permission denied cleaning up {path}: {e}")
    except OSError as e:
        logger.warning(f"OS error cleaning up {path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error cleaning up {path}: {e}")


def _validate_file_size(file_path: Path, max_size_mb: int = None) -> None:
    """Validate file size before processing."""
    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB
    
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)"
            )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not check file size: {e}")


def _validate_dataframe_limits(df: pd.DataFrame) -> None:
    """Validate DataFrame row and column limits."""
    if len(df) > settings.MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"DataFrame has {len(df)} rows, exceeding limit ({settings.MAX_ROWS})"
        )
    
    if len(df.columns) > settings.MAX_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"DataFrame has {len(df.columns)} columns, exceeding limit ({settings.MAX_COLUMNS})"
        )


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV file with logging, validation, and stream safety for large files.
    
    Raises:
        HTTPException(413): File size exceeds limits
        HTTPException(422): DataFrame exceeds row/column limits
        HTTPException(500): File parsing or I/O errors
    """
    logger.info(f"Loading CSV file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        # Use chunked reading for large files to prevent OOM
        if file_size_mb > 50:
            logger.info(f"Large CSV file detected ({file_size_mb:.1f}MB), using chunked reading")
            chunks = []
            chunk_count = 0
            for chunk in pd.read_csv(path, chunksize=100_000):
                chunks.append(chunk)
                chunk_count += 1
                # Safety valve for extremely large files
                if chunk_count > 1000:  # More than 100M rows
                    logger.warning("Chunk count limit reached during CSV loading")
                    break
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=422, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV file: {e}")
    except MemoryError:
        raise HTTPException(status_code=413, detail="File too large to process - insufficient memory")
    except Exception as e:
        logger.error(f"Failed to load CSV file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load CSV file: {e}")


def load_excel(path: Path, engine: Optional[str] = None) -> pd.DataFrame:
    """
    Load Excel file with fallback engine support for malformed files.
    
    Raises:
        HTTPException(413): File size exceeds limits
        HTTPException(422): DataFrame exceeds row/column limits
        HTTPException(500): File parsing or I/O errors
    """
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
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load Excel file with both engines: {fallback_e}"
                )
        else:
            logger.error(f"Failed to load Excel file {path} with engine {engine}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load Excel file: {e}")


def load_json(path: Path, orient: str = "records") -> pd.DataFrame:
    """
    Load JSON file with logging and validation.
    
    Raises:
        HTTPException(413): File size exceeds limits
        HTTPException(422): DataFrame exceeds row/column limits or invalid JSON format
        HTTPException(500): File parsing or I/O errors
    """
    logger.info(f"Loading JSON file: {path}")
    
    # Validate file size before reading
    _validate_file_size(path)
    
    try:
        df = pd.read_json(path, orient=orient)
        
        # Validate DataFrame limits
        _validate_dataframe_limits(df)
        
        logger.info(f"Successfully loaded JSON: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Failed to load JSON file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load JSON file: {e}")


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load Parquet file with logging and validation.
    
    Raises:
        HTTPException(413): File size exceeds limits
        HTTPException(422): DataFrame exceeds row/column limits
        HTTPException(500): File parsing or I/O errors
    """
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
        raise HTTPException(status_code=500, detail=f"Failed to load Parquet file: {e}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV with logging and error handling."""
    logger.info(f"Saving CSV file: {path}")
    try:
        # Set secure permissions on output file
        df.to_csv(path, index=False)
        path.chmod(0o644)  # Read-write for owner, read-only for group/others
        logger.info(f"Successfully saved CSV: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except OSError as e:
        logger.error(f"OS error saving CSV file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save CSV file: {e}")
    except Exception as e:
        logger.error(f"Failed to save CSV file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save CSV file: {e}")


def save_excel(df: pd.DataFrame, path: Path, engine: str = "openpyxl", sheet_name: str = "Sheet1") -> None:
    """Save DataFrame to Excel with logging and error handling."""
    logger.info(f"Saving Excel file: {path}")
    try:
        df.to_excel(path, index=False, engine=engine, sheet_name=sheet_name)
        path.chmod(0o644)
        logger.info(f"Successfully saved Excel: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except OSError as e:
        logger.error(f"OS error saving Excel file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save Excel file: {e}")
    except Exception as e:
        logger.error(f"Failed to save Excel file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save Excel file: {e}")


def save_json(df: pd.DataFrame, path: Path, orient: str = "records", indent: int = 2) -> None:
    """Save DataFrame to JSON with logging and error handling."""
    logger.info(f"Saving JSON file: {path}")
    try:
        df.to_json(path, orient=orient, indent=indent)
        path.chmod(0o644)
        logger.info(f"Successfully saved JSON: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except OSError as e:
        logger.error(f"OS error saving JSON file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save JSON file: {e}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save JSON file: {e}")


def save_parquet(df: pd.DataFrame, path: Path, compression: str = "snappy") -> None:
    """Save DataFrame to Parquet with logging and error handling."""
    logger.info(f"Saving Parquet file: {path}")
    try:
        df.to_parquet(path, compression=compression, index=False)
        path.chmod(0o644)
        logger.info(f"Successfully saved Parquet: {df.shape[0]} rows, {df.shape[1]} columns to {path}")
    except OSError as e:
        logger.error(f"OS error saving Parquet file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save Parquet file: {e}")
    except Exception as e:
        logger.error(f"Failed to save Parquet file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save Parquet file: {e}")


# Legacy compatibility - keeping these functions but marking as deprecated
def make_temp_dir() -> str:
    """Create a unique temporary directory to prevent collisions."""
    logger.warning("make_temp_dir() is deprecated. Use save_temp_file() instead.")
    try:
        base = Path(settings.TEMP_DIR)
        base.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=base)
        
        # Set secure permissions on temp directory
        os.chmod(temp_dir, 0o700)
        
        logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir
        
    except OSError as e:
        logger.error(f"Failed to create temporary directory: {e}")
        raise HTTPException(status_code=500, detail=f"Could not create temporary directory: {e}")