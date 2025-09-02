"""
Preprocessing router with proper error handling and HTTPException conversion.
"""
from fastapi import APIRouter, UploadFile, HTTPException, Depends, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import logging
import json
import os
from pathlib import Path
from pydantic import ValidationError

from app.services.preprocessing import (
    handle_file_upload, 
    preprocess_pipeline,
    DataPreprocessingService
)
from app.schemas.preprocess import (
    PreprocessingRequest, 
    PreprocessingResponse,
    ValidationReport,
    CleaningConfig,
    EncodingConfig,
    FileUploadResponse,
    CleaningResponse,
    EncodingResponse,
    CleanlinessReport
)
from app.utils.file_io import cleanup_temp_file
from app.core.exceptions import DataPreprocessingError

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])
logger = logging.getLogger(__name__)


# Security placeholder - TODO: implement real authentication
async def require_api_key() -> str:
    """
    Placeholder for API key authentication.
    TODO: Implement real API key validation and rate limiting.
    """
    return "placeholder_key"


@router.exception_handler(DataPreprocessingError)
async def dp_error_handler(request: Request, exc: DataPreprocessingError):
    """Handle DataPreprocessingError exceptions at router level."""
    logger.error(f"DataPreprocessingError: {str(exc)}")
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def parse_preprocessing_config(
    preprocessing_config: Optional[str] = Form(None),
    request_body: Optional[PreprocessingRequest] = None
) -> PreprocessingRequest:
    """
    Parse preprocessing config from either form field or request body.
    Handles both multipart form data and JSON body scenarios.
    """
    if preprocessing_config:
        # Parse JSON string from form field
        try:
            config_dict = json.loads(preprocessing_config)
            return PreprocessingRequest(**config_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid JSON in preprocessing_config: {str(e)}"
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid preprocessing configuration: {e.errors()}"
            )
    elif request_body:
        return request_body
    else:
        # Return default config if none provided
        return PreprocessingRequest()


@router.post(
    "/upload", 
    response_model=FileUploadResponse,
    summary="Upload and validate file for preprocessing",
    description="Upload a file and get initial validation results including file metadata and cleanliness report.",
    responses={
        200: {
            "description": "File uploaded and validated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "data.csv",
                        "file_size": 1024000,
                        "row_count": 5000,
                        "column_count": 10,
                        "preview_rows": [{"col1": "value1", "col2": "value2"}],
                        "initial_cleanliness_report": {
                            "missing_values_count": 150,
                            "duplicate_rows_count": 23,
                            "data_quality_score": 0.85
                        }
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation error"}
    }
)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Upload and validate a file for preprocessing.
    
    Returns:
        FileUploadResponse with file information and validation results
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Get actual file size in bytes
        file_size_bytes = os.path.getsize(str(temp_path_obj)) if temp_path_obj else 0
        
        # Validate the data and get cleanliness report
        service = DataPreprocessingService()
        validation_report = service.validate_data(df)
        
        # Generate initial cleanliness report
        cleanliness_report = service.get_cleanliness_report(df)
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        return FileUploadResponse(
            filename=file.filename,
            file_size=file_size_bytes,
            row_count=df.shape[0],
            column_count=df.shape[1],
            preview_rows=df.head().to_dict('records'),
            initial_cleanliness_report=cleanliness_report
        )
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post(
    "/run", 
    response_model=PreprocessingResponse,
    summary="Run complete preprocessing pipeline",
    description="Run the complete preprocessing pipeline with either multipart form data or JSON configuration.",
    responses={
        200: {
            "description": "Preprocessing completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "num_rows": 4850,
                        "num_columns": 12,
                        "audit_log": ["Removed 150 rows with missing values", "Applied one-hot encoding to 3 columns"],
                        "preview": [{"col1": "cleaned_value", "col2": 1.0}]
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation or configuration error"}
    }
)
async def run_preprocessing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    preprocessing_config: Optional[str] = Form(None),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Run the complete preprocessing pipeline with form upload and config.
    Accepts preprocessing_config as JSON string in form field.
    
    Returns:
        PreprocessingResponse with results and audit log
    """
    temp_path = None
    try:
        # Parse configuration from form field
        config = await parse_preprocessing_config(preprocessing_config=preprocessing_config)
        
        # Handle file upload
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Run preprocessing pipeline
        result = preprocess_pipeline(
            df=df,
            cleaning_config=config.cleaning_config,
            encoding_config=config.encoding_config,
            validate_first=config.validate_first
        )
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        # Create response
        return PreprocessingResponse(
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            audit_log=result["audit_log"],
            preview=result["preview"]
        )
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post(
    "/json", 
    response_model=PreprocessingResponse,
    summary="Run preprocessing with JSON configuration",
    description="Run the complete preprocessing pipeline with explicit JSON configuration and multipart file upload.",
    responses={
        200: {
            "description": "Preprocessing completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "num_rows": 4850,
                        "num_columns": 12,
                        "audit_log": ["Removed 150 rows with missing values", "Applied one-hot encoding to 3 columns"],
                        "preview": [{"col1": "cleaned_value", "col2": 1.0}]
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation or configuration error"}
    }
)
async def run_preprocessing_json(
    background_tasks: BackgroundTasks,
    config: PreprocessingRequest, 
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Run the complete preprocessing pipeline with JSON config and file upload.
    
    Returns:
        PreprocessingResponse with results and audit log
    """
    temp_path = None
    try:
        # Handle file upload
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Run preprocessing pipeline
        result = preprocess_pipeline(
            df=df,
            cleaning_config=config.cleaning_config,
            encoding_config=config.encoding_config,
            validate_first=config.validate_first
        )
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        # Create response
        return PreprocessingResponse(
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            audit_log=result["audit_log"],
            preview=result["preview"]
        )
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post(
    "/validate", 
    response_model=ValidationReport,
    summary="Validate uploaded data",
    description="Validate uploaded data without processing and return detailed validation results.",
    responses={
        200: {
            "description": "Data validated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "is_valid": True,
                        "errors": [],
                        "warnings": ["Column 'age' has 5% missing values"],
                        "summary": {"total_rows": 5000, "total_columns": 10}
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation error"}
    }
)
async def validate_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Validate uploaded data without processing.
    
    Returns:
        ValidationReport with detailed validation results
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Validate the data
        service = DataPreprocessingService()
        validation_report = service.validate_data(df)
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        return validation_report
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post(
    "/clean", 
    response_model=CleaningResponse,
    summary="Clean data with configuration",
    description="Clean data using specified cleaning configuration without full preprocessing.",
    responses={
        200: {
            "description": "Data cleaned successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "original_shape": [5000, 10],
                        "final_shape": [4850, 10],
                        "rows_removed": 150,
                        "columns_removed": 0,
                        "cleaning_actions": ["Removed rows with missing values"],
                        "preview": [{"col1": "cleaned_value", "col2": "cleaned_value2"}]
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation or configuration error"}
    }
)
async def clean_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    cleaning_config: Optional[str] = Form(None),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Clean data with specified configuration.
    
    Returns:
        CleaningResponse with cleaning results and audit log
    """
    temp_path = None
    try:
        # Parse cleaning config from form field or use defaults
        if cleaning_config:
            try:
                config_dict = json.loads(cleaning_config)
                config = CleaningConfig(**config_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Invalid cleaning configuration: {str(e)}"
                )
        else:
            config = CleaningConfig()
            
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Clean the data
        service = DataPreprocessingService()
        cleaning_result = service.clean_data(df, config)
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        return CleaningResponse(
            success=True,
            original_shape=cleaning_result.original_shape,
            final_shape=cleaning_result.final_shape,
            rows_removed=cleaning_result.num_rows_removed,
            columns_removed=cleaning_result.num_columns_removed,
            cleaning_actions=cleaning_result.cleaning_actions,
            preview=cleaning_result.data.head().to_dict('records')
        )
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post(
    "/encode", 
    response_model=EncodingResponse,
    summary="Encode data with configuration",
    description="Encode categorical data using specified encoding configuration without full preprocessing.",
    responses={
        200: {
            "description": "Data encoded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "original_shape": [4850, 10],
                        "final_shape": [4850, 15],
                        "encoders_used": ["one_hot", "label"],
                        "encoding_actions": ["Applied one-hot encoding to 'category' column"],
                        "preview": [{"col1": "value", "category_A": 1, "category_B": 0}]
                    }
                }
            }
        },
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Validation or configuration error"}
    }
)
async def encode_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    encoding_config: Optional[str] = Form(None),
    api_key: str = Depends(require_api_key)  # TODO: Enable real auth
):
    """
    Encode data with specified configuration.
    
    Returns:
        EncodingResponse with encoding results and audit log
    """
    temp_path = None
    try:
        # Parse encoding config from form field or use defaults
        if encoding_config:
            try:
                config_dict = json.loads(encoding_config)
                config = EncodingConfig(**config_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Invalid encoding configuration: {str(e)}"
                )
        else:
            config = EncodingConfig()
            
        df, temp_path = await handle_file_upload(file)
        temp_path_obj = Path(temp_path) if temp_path else None
        
        # Encode the data
        service = DataPreprocessingService()
        encoding_result = service.encode_data(df, config)
        
        # Schedule cleanup
        if temp_path_obj:
            background_tasks.add_task(cleanup_temp_file, str(temp_path_obj))
        
        return EncodingResponse(
            success=True,
            original_shape=encoding_result.original_shape,
            final_shape=encoding_result.final_shape,
            encoders_used=encoding_result.encoders_used,
            encoding_actions=encoding_result.encoding_actions,
            preview=encoding_result.data.head().to_dict('records')
        )
        
    except (HTTPException,) as e:
        # Preserve original HTTPExceptions (415, 413, 504, etc.) from file handling
        raise e
    except DataPreprocessingError as e:
        # Let the exception handler deal with this
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed and no background task was scheduled
        if temp_path and 'background_tasks' not in locals():
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")