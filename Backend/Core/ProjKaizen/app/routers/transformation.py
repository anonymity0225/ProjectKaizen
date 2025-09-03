# app/routers/transformation.py
"""
FastAPI router for data transformation operations.
Provides comprehensive endpoints for various data transformation capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Callable, Optional, Union
import pandas as pd
import json
import logging
import time
import ast

from app.services.transformation import (
    extract_date_components,
    encode_categorical,
    scale_numeric,
    tokenize_text,
    apply_tfidf_vectorization,
    apply_pca,
    apply_custom_user_function,
    get_transformation_summary,
    clear_transformation_history,
    apply_multiple_transformations,
    TransformationError
)
from app.schemas.transformation import (
    DateExtractionRequest, DateExtractionResponse,
    ScalingRequest, ScalingResponse,
    CategoricalEncodingRequest, CategoricalEncodingResponse,
    TextPreprocessingRequest, TextPreprocessingResponse,
    TFIDFRequest, TFIDFResponse,
    PCATransformRequest, PCATransformResponse,
    CustomTransformationRequest, CustomTransformationResponse,
    BatchTransformationRequest, BatchTransformationResponse,
    BaseTransformationResponse
)
from app.config import settings
from app.api import verify_token  # JWT dependency

# Configure logging
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(
    prefix="/transform",
    tags=["transformation"],
    dependencies=[Depends(verify_token)]
)

# Unsafe AST nodes that should be rejected in custom functions
UNSAFE_AST_NODES = {
    ast.Import,
    ast.ImportFrom,
    ast.Exec,
    ast.Eval,
    ast.Call,  # We'll filter specific calls
}

UNSAFE_ATTRIBUTES = {
    'os', 'sys', 'subprocess', 'eval', 'exec', 'open', 'file',
    '__import__', '__builtins__', 'globals', 'locals', 'vars', 'dir'
}


def _convert_dict_to_dataframe(data_dict: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Convert dictionary representation to pandas DataFrame.
    
    Args:
        data_dict: Dictionary or list containing DataFrame data
        
    Returns:
        pandas DataFrame
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        if isinstance(data_dict, dict):
            return pd.DataFrame(data_dict)
        elif isinstance(data_dict, list):
            return pd.DataFrame(data_dict)
        else:
            raise ValueError("Invalid dataframe format")
    except Exception as e:
        raise ValueError(f"Failed to convert data to DataFrame: {str(e)}")


def _convert_dataframe_to_dict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert pandas DataFrame to list of dictionary records.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of dictionary records
    """
    return df.to_dict(orient="records")


def _normalize_columns(columns: Union[str, List[str]]) -> List[str]:
    """
    Normalize columns input to always be a list.
    
    Args:
        columns: Single column name or list of column names
        
    Returns:
        List of column names
    """
    if isinstance(columns, str):
        return [columns]
    elif isinstance(columns, list):
        return columns
    else:
        raise ValueError("Columns must be string or list of strings")


def _validate_custom_function_ast(code: str) -> None:
    """
    Validate that custom function code doesn't contain unsafe operations.
    
    Args:
        code: Function code to validate
        
    Raises:
        ValueError: If code contains unsafe operations
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {str(e)}")
    
    for node in ast.walk(tree):
        # Check for unsafe imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in custom functions")
        
        # Check for unsafe attribute access
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in UNSAFE_ATTRIBUTES:
                raise ValueError(f"Access to '{node.value.id}' is not allowed")
        
        # Check for unsafe function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {'eval', 'exec', 'compile', '__import__', 'open'}:
                    raise ValueError(f"Function '{node.func.id}' is not allowed")
            elif isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id in UNSAFE_ATTRIBUTES):
                    raise ValueError(f"Method calls on '{node.func.value.id}' are not allowed")


def _create_standard_response(
    data: List[Dict[str, Any]], 
    metadata: Dict[str, Any],
    transformed_df: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create standardized response format.
    
    Args:
        data: Transformed data as list of records
        metadata: Transformation metadata
        transformed_df: Deprecated field for backward compatibility
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "data": data,
        "metadata": metadata
    }
    
    # Add transformed_df for backward compatibility (deprecated)
    if transformed_df is not None:
        response["transformed_df"] = transformed_df
    
    return response


@router.post("/date", response_model=DateExtractionResponse, status_code=200)
async def extract_date_features(request: DateExtractionRequest):
    """
    Extract date components from a datetime column.
    
    Args:
        request: DateExtractionRequest containing dataframe, column, and components
        
    Returns:
        DateExtractionResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    logger.info(f"Date extraction request for column: {request.column}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation
        result_df, metadata = extract_date_components(
            df=df,
            column=request.column,
            components=request.components
        )
        
        # Add timing information
        duration_ms = int((time.time() - start_time) * 1000)
        metadata["duration_ms"] = duration_ms
        
        # Ensure required metadata fields
        metadata.setdefault("status", "success")
        metadata.setdefault("method", "date_extraction")
        metadata.setdefault("message", f"Successfully extracted date components from column '{request.column}'")
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Date extraction failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "date_extraction",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in date extraction: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": "date_extraction",
                "message": "Internal server error during date extraction",
                "duration_ms": duration_ms
            }
        )


@router.post("/scale", response_model=ScalingResponse, status_code=200)
async def scale_numerical_features(request: ScalingRequest):
    """
    Scale numerical features using various methods.
    
    Args:
        request: ScalingRequest containing dataframe, columns, method, and custom_range
        
    Returns:
        ScalingResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    columns = _normalize_columns(request.columns)
    logger.info(f"Scaling request for columns: {columns}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation for each column
        result_df = df.copy()
        all_scaled_columns = []
        
        for column in columns:
            temp_df, metadata = scale_numeric(
                df=result_df,
                column=column,
                method=request.method,
                custom_range=request.custom_range
            )
            
            result_df = temp_df
            all_scaled_columns.extend(metadata.get("scaled_columns", []))
        
        # Add timing and consolidate metadata
        duration_ms = int((time.time() - start_time) * 1000)
        final_metadata = {
            "status": "success",
            "method": request.method,
            "message": f"Successfully scaled {len(all_scaled_columns)} columns",
            "scaled_columns": all_scaled_columns,
            "duration_ms": duration_ms
        }
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=final_metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Scaling failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": request.method,
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in scaling: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": request.method,
                "message": "Internal server error during scaling",
                "duration_ms": duration_ms
            }
        )


@router.post("/encode", response_model=CategoricalEncodingResponse, status_code=200)
async def encode_categorical_features(request: CategoricalEncodingRequest):
    """
    Encode categorical features using various methods.
    
    Args:
        request: CategoricalEncodingRequest containing dataframe, columns, method, and custom_mapping
        
    Returns:
        CategoricalEncodingResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    columns = _normalize_columns(request.columns)
    logger.info(f"Categorical encoding request for columns: {columns}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation for each column
        result_df = df.copy()
        all_encoded_columns = []
        
        for column in columns:
            temp_df, metadata = encode_categorical(
                df=result_df,
                column=column,
                method=request.method,
                custom_mapping=request.custom_mapping
            )
            
            result_df = temp_df
            all_encoded_columns.extend(metadata.get("encoded_columns", []))
        
        # Add timing and consolidate metadata
        duration_ms = int((time.time() - start_time) * 1000)
        final_metadata = {
            "status": "success",
            "method": request.method,
            "message": f"Successfully encoded {len(all_encoded_columns)} columns",
            "encoded_columns": all_encoded_columns,
            "duration_ms": duration_ms
        }
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=final_metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Categorical encoding failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": request.method,
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in categorical encoding: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": request.method,
                "message": "Internal server error during categorical encoding",
                "duration_ms": duration_ms
            }
        )


@router.post("/text", response_model=TextPreprocessingResponse, status_code=200)
async def preprocess_text_features(request: TextPreprocessingRequest):
    """
    Preprocess text features via tokenization and optional stemming/lemmatization.
    
    Args:
        request: TextPreprocessingRequest containing dataframe, column, and method
        
    Returns:
        TextPreprocessingResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    logger.info(f"Text preprocessing request for column: {request.column}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation
        result_df, metadata = tokenize_text(
            df=df,
            column=request.column,
            method=request.method
        )
        
        # Add timing information
        duration_ms = int((time.time() - start_time) * 1000)
        metadata["duration_ms"] = duration_ms
        
        # Ensure required metadata fields
        metadata.setdefault("status", "success")
        metadata.setdefault("method", request.method)
        metadata.setdefault("message", f"Successfully preprocessed text in column '{request.column}'")
        metadata["processed_column"] = request.column
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Text preprocessing failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": request.method,
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in text preprocessing: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": request.method,
                "message": "Internal server error during text preprocessing",
                "duration_ms": duration_ms
            }
        )


@router.post("/tfidf", response_model=TFIDFResponse, status_code=200)
async def apply_tfidf_transformation(request: TFIDFRequest):
    """
    Apply TF-IDF vectorization to text features.
    
    Args:
        request: TFIDFRequest containing dataframe, column, max_features, lowercase, and remove_stopwords
        
    Returns:
        TFIDFResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    logger.info(f"TF-IDF request for column: {request.column}, max_features: {request.max_features}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation
        result_df, metadata = apply_tfidf_vectorization(
            df=df,
            column=request.column,
            max_features=request.max_features
        )
        
        # Add timing information
        duration_ms = int((time.time() - start_time) * 1000)
        metadata["duration_ms"] = duration_ms
        
        # Ensure required metadata fields
        metadata.setdefault("status", "success")
        metadata.setdefault("method", "tfidf")
        metadata.setdefault("message", f"Successfully applied TF-IDF to column '{request.column}'")
        metadata["tfidf_columns"] = metadata.get("added_columns", [])
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"TF-IDF transformation failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "tfidf",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in TF-IDF transformation: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": "tfidf",
                "message": "Internal server error during TF-IDF transformation",
                "duration_ms": duration_ms
            }
        )


@router.post("/pca", response_model=PCATransformResponse, status_code=200)
async def apply_pca_transformation(request: PCATransformRequest):
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Args:
        request: PCATransformRequest containing dataframe, columns, n_components, and whiten
        
    Returns:
        PCATransformResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    columns = _normalize_columns(request.columns) if request.columns else None
    logger.info(f"PCA request for columns: {columns}, n_components: {request.n_components}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply transformation
        result_df, metadata = apply_pca(
            df=df,
            n_components=request.n_components,
            columns=columns
        )
        
        # Add timing information
        duration_ms = int((time.time() - start_time) * 1000)
        metadata["duration_ms"] = duration_ms
        
        # Ensure required metadata fields
        metadata.setdefault("status", "success")
        metadata.setdefault("method", "pca")
        metadata.setdefault("message", f"Successfully applied PCA with {request.n_components} components")
        metadata["principal_components"] = metadata.get("added_columns", [])
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"PCA transformation failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "pca",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in PCA transformation: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": "pca",
                "message": "Internal server error during PCA transformation",
                "duration_ms": duration_ms
            }
        )


@router.post("/custom", response_model=CustomTransformationResponse, status_code=200)
async def apply_custom_transformation(request: CustomTransformationRequest):
    """
    Apply a custom user-defined transformation function.
    
    Args:
        request: CustomTransformationRequest containing dataframe, function_name, and function_code
        
    Returns:
        CustomTransformationResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    logger.info(f"Custom transformation request with function: {request.function_name}")
    
    try:
        # Validate the function code for safety
        _validate_custom_function_ast(request.function_code)
        
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Create safe namespace for function execution
        safe_namespace = {
            'pd': pd,
            'np': __import__('numpy'),
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'max': max,
                'min': min,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed
            }
        }
        
        # Execute function code in safe namespace
        try:
            exec(request.function_code, safe_namespace)
            
            # Extract the function from namespace
            func = None
            for name, obj in safe_namespace.items():
                if callable(obj) and name not in ['pd', 'np'] and not name.startswith('__'):
                    func = obj
                    break
            
            if func is None:
                raise ValueError("No callable function found in provided code")
                
        except Exception as e:
            raise ValueError(f"Failed to execute function code: {str(e)}")
        
        # Apply transformation
        result_df, metadata = apply_custom_user_function(
            df=df,
            func=func,
            function_name=request.function_name
        )
        
        # Add timing information
        duration_ms = int((time.time() - start_time) * 1000)
        metadata["duration_ms"] = duration_ms
        
        # Ensure required metadata fields
        metadata.setdefault("status", "success")
        metadata.setdefault("method", "custom")
        metadata.setdefault("message", f"Successfully applied custom function '{request.function_name}'")
        
        # Convert result to standard format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return _create_standard_response(
            data=result_data,
            metadata=metadata,
            transformed_df=result_data  # Deprecated backward compatibility
        )
        
    except ValueError as e:
        # Validation errors
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Custom transformation validation failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "custom",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Custom transformation failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "custom",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in custom transformation: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": "custom",
                "message": "Internal server error during custom transformation",
                "duration_ms": duration_ms
            }
        )


@router.post("/batch", response_model=BatchTransformationResponse, status_code=200)
async def apply_batch_transformations(request: BatchTransformationRequest):
    """
    Apply multiple transformations in sequence.
    
    Args:
        request: BatchTransformationRequest containing dataframe and transformations list
        
    Returns:
        BatchTransformationResponse with final data and metadata list
        
    Raises:
        HTTPException: If transformation fails
    """
    start_time = time.time()
    logger.info(f"Batch transformation request with {len(request.transformations)} transformations")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Apply batch transformations
        result_df, metadata_list = apply_multiple_transformations(
            df=df,
            transformations=request.transformations
        )
        
        # Add timing information to final metadata
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Create final metadata summary
        final_metadata = {
            "status": "success",
            "method": "batch",
            "message": f"Successfully applied {len(request.transformations)} transformations",
            "transformations_applied": len(request.transformations),
            "duration_ms": duration_ms
        }
        
        # Convert result to standard format
        final_data = _convert_dataframe_to_dict(result_df)
        
        return {
            "data": final_data,
            "metadata": final_metadata,
            "transformation_metadata": metadata_list
        }
        
    except TransformationError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Batch transformation failed: {str(e)}")
        
        error_metadata = {
            "status": "error",
            "method": "batch",
            "message": str(e),
            "duration_ms": duration_ms
        }
        
        raise HTTPException(
            status_code=422,
            detail=error_metadata
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in batch transformation: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "method": "batch",
                "message": "Internal server error during batch transformation",
                "duration_ms": duration_ms
            }
        )


@router.get("/history", status_code=200)
async def get_transformation_history():
    """
    Get a comprehensive summary of all transformations applied.
    
    Returns:
        Dictionary containing transformation summary
    """
    try:
        summary = get_transformation_summary()
        return JSONResponse(
            status_code=200,
            content=summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get transformation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve transformation history"
            }
        )


@router.delete("/history", status_code=200)
async def clear_transformation_record():
    """
    Clear the transformation history.
    
    Returns:
        Dictionary with status confirmation
    """
    try:
        clear_transformation_history()
        return JSONResponse(
            status_code=200,
            content={
                "status": "cleared",
                "message": "Transformation history cleared successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to clear transformation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to clear transformation history"
            }
        )


# Health check endpoint for the transformation service
@router.get("/health", status_code=200)
async def transformation_health_check():
    """
    Health check endpoint for transformation service.
    
    Returns:
        Dictionary with service status
    """
    return {
        "status": "healthy",
        "service": "transformation",
        "version": "1.0.0"
    }