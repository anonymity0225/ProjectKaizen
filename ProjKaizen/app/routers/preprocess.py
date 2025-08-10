"""
Preprocessing router with proper error handling and HTTPException conversion.
"""
from fastapi import APIRouter, UploadFile, HTTPException, Depends, File, Form
from typing import Dict, Any, Optional
import logging

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
    FileUploadResponse
)
from app.utils.file_io import cleanup_temp_file
from app.core.exceptions import DataPreprocessingError

router = APIRouter(prefix="/preprocess", tags=["preprocessing"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and validate a file for preprocessing.
    
    Returns:
        FileUploadResponse with file information and validation results
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        
        # Validate the data
        service = DataPreprocessingService()
        validation_report = service.validate_data(df)
        
        return FileUploadResponse(
            filename=file.filename,
            file_size=len(df),
            num_rows=df.shape[0],
            num_columns=df.shape[1],
            column_names=list(df.columns),
            data_types=df.dtypes.astype(str).to_dict(),
            preview=df.head().to_dict('records')
        )
        
    except DataPreprocessingError as e:
        logger.error(f"Preprocessing error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if validation failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post("/run", response_model=PreprocessingResponse)
async def run_preprocessing(
    file: UploadFile = File(...), 
    config: PreprocessingRequest = Depends()
):
    """
    Run the complete preprocessing pipeline with form upload and JSON config.
    
    Returns:
        PreprocessingResponse with results and audit log
    """
    temp_path = None
    try:
        # Handle file upload
        df, temp_path = await handle_file_upload(file)
        
        # Run preprocessing pipeline with temp_path for cleanup
        result = preprocess_pipeline(
            df=df,
            cleaning_config=config.cleaning_config,
            encoding_config=config.encoding_config,
            validate_first=config.validate_first,
            temp_path=temp_path  # Pass temp_path for cleanup
        )
        
        # Create response
        return PreprocessingResponse(
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            audit_log=result["audit_log"],
            preview=result["preview"]
        )
        
    except DataPreprocessingError as e:
        logger.error(f"Preprocessing error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post("/json", response_model=PreprocessingResponse)
async def run_preprocessing_json(
    config: PreprocessingRequest,
    file: UploadFile = File(...)
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
        
        # Run preprocessing pipeline with temp_path for cleanup
        result = preprocess_pipeline(
            df=df,
            cleaning_config=config.cleaning_config,
            encoding_config=config.encoding_config,
            validate_first=config.validate_first,
            temp_path=temp_path  # Pass temp_path for cleanup
        )
        
        # Create response
        return PreprocessingResponse(
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            audit_log=result["audit_log"],
            preview=result["preview"]
        )
        
    except DataPreprocessingError as e:
        logger.error(f"Preprocessing error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post("/validate", response_model=ValidationReport)
async def validate_data(file: UploadFile = File(...)):
    """
    Validate uploaded data without processing.
    
    Returns:
        ValidationReport with detailed validation results
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        
        # Validate the data
        service = DataPreprocessingService()
        validation_report = service.validate_data(df)
        
        return validation_report
        
    except DataPreprocessingError as e:
        logger.error(f"Validation error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if validation failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post("/clean", response_model=Dict[str, Any])
async def clean_data(
    file: UploadFile = File(...),
    cleaning_config: CleaningConfig = Depends()
):
    """
    Clean data with specified configuration.
    
    Returns:
        Dict with cleaning results and audit log
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        
        # Clean the data
        service = DataPreprocessingService()
        cleaning_result = service.clean_data(df, cleaning_config)
        
        return {
            "success": True,
            "original_shape": cleaning_result.original_shape,
            "final_shape": cleaning_result.final_shape,
            "rows_removed": cleaning_result.num_rows_removed,
            "columns_removed": cleaning_result.num_columns_removed,
            "cleaning_actions": cleaning_result.cleaning_actions,
            "preview": cleaning_result.data.head().to_dict('records')
        }
        
    except DataPreprocessingError as e:
        logger.error(f"Cleaning error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


@router.post("/encode", response_model=Dict[str, Any])
async def encode_data(
    file: UploadFile = File(...),
    encoding_config: EncodingConfig = Depends()
):
    """
    Encode data with specified configuration.
    
    Returns:
        Dict with encoding results and audit log
    """
    temp_path = None
    try:
        df, temp_path = await handle_file_upload(file)
        
        # Encode the data
        service = DataPreprocessingService()
        encoding_result = service.encode_data(df, encoding_config)
        
        return {
            "success": True,
            "original_shape": encoding_result.original_shape,
            "final_shape": encoding_result.final_shape,
            "encoders_used": encoding_result.encoders_used,
            "encoding_actions": encoding_result.encoding_actions,
            "preview": encoding_result.data.head().to_dict('records')
        }
        
    except DataPreprocessingError as e:
        logger.error(f"Encoding error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": "DataPreprocessingError"
        })
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", extra={
            "filename": file.filename,
            "error_type": type(e).__name__
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temp file if processing failed
        if temp_path:
            try:
                cleanup_temp_file(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")