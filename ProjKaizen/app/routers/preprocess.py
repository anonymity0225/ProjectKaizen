# Preprocessing routes 
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
import json
import logging
from ..services.preprocessing import handle_file_upload, preprocess_pipeline
from ..schemas.preprocess import PreprocessingRequest, PreprocessingResponse
from ..utils.config_helper import convert_preprocessing_config
from ..utils.file_io import cleanup_temp_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])

@router.post(
    "/run",
    response_model=PreprocessingResponse,
    summary="Run data preprocessing pipeline",
    description="Upload a CSV or Excel file and a preprocessing configuration. Returns cleaned data summary, audit log, and preview."
)
async def run_preprocessing(
    file: UploadFile = File(..., description="CSV or Excel file to preprocess"),
    preprocessing_config: str = Form(
        ..., 
        description="JSON string matching PreprocessingRequest schema",
        example='{"drop_duplicates": true, "handle_missing": "drop", "encoding_method": "label", "encoding_columns": ["category"], "validate_first": true}'
    )
):
    """
    Run the preprocessing pipeline on an uploaded file and configuration.
    
    The preprocessing_config should be a JSON string with the following structure:
    {
        "drop_duplicates": boolean,
        "handle_missing": "drop" | "fill" | null,
        "numeric_columns": ["col1", "col2"] | null,
        "categorical_columns": ["col1", "col2"] | null,
        "encoding_method": "label" | "onehot" | null,
        "encoding_columns": ["col1", "col2"] | null,
        "validate_first": boolean
    }
    """
    temp_file_path = None
    
    try:
        # Parse and validate configuration
        try:
            config_dict = json.loads(preprocessing_config)
            config = PreprocessingRequest(**config_dict)
            logger.info(f"Parsed preprocessing config: {config_dict}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in preprocessing_config: {str(e)}")
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid JSON format in preprocessing_config: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Invalid preprocessing_config schema: {str(e)}")
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid preprocessing_config schema: {str(e)}"
            )
        
        # Handle file upload
        try:
            df = await handle_file_upload(file)
            # Store the temp file path for cleanup (assuming handle_file_upload returns it somehow)
            # Note: You might need to modify handle_file_upload to return the temp path
            logger.info(f"Successfully uploaded and processed file: {file.filename}")
        except HTTPException as e:
            logger.error(f"File upload failed: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {str(e)}")
            raise HTTPException(
                status_code=422, 
                detail=f"File upload failed: {str(e)}"
            )
        
        # Convert configuration using centralized helper
        cleaning_config, encoding_config = convert_preprocessing_config(config)
        
        logger.info(f"Converted configs - Cleaning: {cleaning_config is not None}, Encoding: {encoding_config is not None}")
        
        # Run preprocessing pipeline
        response = preprocess_pipeline(
            df,
            cleaning_config=cleaning_config,
            encoding_config=encoding_config,
            validate_first=config.validate_first
        )
        
        logger.info(f"Preprocessing completed successfully. Result shape: {response.num_rows}x{response.num_columns}")
        
        return JSONResponse(content=jsonable_encoder(response))
        
    except HTTPException as e:
        logger.error(f"HTTP error during preprocessing: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Preprocessing failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

# Alternative endpoint using JSON body (bonus implementation)
@router.post(
    "/run-json",
    response_model=PreprocessingResponse,
    summary="Run preprocessing with JSON config (alternative endpoint)",
    description="Alternative endpoint that accepts JSON configuration directly in request body along with file."
)
async def run_preprocessing_json(
    file: UploadFile = File(..., description="CSV or Excel file to preprocess"),
    config: PreprocessingRequest = Body(..., description="Preprocessing configuration")
):
    """
    Alternative endpoint that accepts JSON configuration directly in the request body.
    This provides better type safety and documentation but requires clients to structure
    their requests differently (not standard multipart form with file + JSON string).
    """
    temp_file_path = None
    
    try:
        logger.info(f"Processing file with JSON config: {config.dict()}")
        
        # Handle file upload
        df = await handle_file_upload(file)
        logger.info(f"Successfully uploaded and processed file: {file.filename}")
        
        # Convert configuration
        cleaning_config, encoding_config = convert_preprocessing_config(config)
        
        # Run preprocessing pipeline  
        response = preprocess_pipeline(
            df,
            cleaning_config=cleaning_config,
            encoding_config=encoding_config,
            validate_first=config.validate_first
        )
        
        logger.info(f"Preprocessing completed successfully. Result shape: {response.num_rows}x{response.num_columns}")
        
        return response
        
    except HTTPException as e:
        logger.error(f"HTTP error during JSON preprocessing: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during JSON preprocessing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {str(e)}"
        )
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)