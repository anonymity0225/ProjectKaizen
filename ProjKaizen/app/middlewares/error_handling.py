"""
Error handling middleware for centralized exception-to-HTTPException mapping.
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Type, Callable
import logging

from app.services.preprocessing import DataPreprocessingError
from app.utils.file_io import cleanup_temp_file

logger = logging.getLogger(__name__)

# Error mapping configuration
ERROR_MAPPING: Dict[Type[Exception], Callable[[Exception], HTTPException]] = {
    DataPreprocessingError: lambda e: HTTPException(status_code=422, detail=str(e)),
    ValueError: lambda e: HTTPException(status_code=400, detail=str(e)),
    FileNotFoundError: lambda e: HTTPException(status_code=404, detail="File not found"),
    PermissionError: lambda e: HTTPException(status_code=403, detail="Permission denied"),
    MemoryError: lambda e: HTTPException(status_code=413, detail="File too large to process"),
}

def get_http_exception(exception: Exception) -> HTTPException:
    """
    Convert application exceptions to appropriate HTTPExceptions.
    
    Args:
        exception: The original exception
        
    Returns:
        HTTPException with appropriate status code and detail
    """
    exception_type = type(exception)
    
    # Check if we have a specific mapping for this exception type
    if exception_type in ERROR_MAPPING:
        return ERROR_MAPPING[exception_type](exception)
    
    # Default mapping for unexpected exceptions
    logger.error(f"Unhandled exception: {str(exception)}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error")

async def error_handling_middleware(request: Request, call_next):
    """
    Middleware to handle exceptions and convert them to appropriate HTTP responses.
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        http_exception = get_http_exception(e)
        return JSONResponse(
            status_code=http_exception.status_code,
            content={"detail": http_exception.detail}
        )

def cleanup_temp_file_safe(temp_path: str) -> None:
    """
    Safely cleanup temporary file with error handling.
    
    Args:
        temp_path: Path to the temporary file to cleanup
    """
    if temp_path:
        try:
            cleanup_temp_file(temp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {temp_path}: {str(e)}") 