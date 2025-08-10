# app/routers/transformation.py
"""
FastAPI router for data transformation operations.
Provides comprehensive endpoints for various data transformation capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import json
import logging

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
    apply_multiple_transformations
)
from app.schemas.transformation import (
    DateExtractionRequest, DateExtractionResponse,
    ScalingRequest, ScalingResponse,
    CategoricalEncodingRequest, CategoricalEncodingResponse,
    TextPreprocessingRequest, TextPreprocessingResponse,
    TFIDFRequest, TFIDFResponse,
    PCATransformRequest, PCATransformResponse,
    CustomTransformationResponse,
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


def _convert_dict_to_dataframe(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert dictionary representation to pandas DataFrame.
    
    Args:
        data_dict: Dictionary containing DataFrame data
        
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


def _convert_dataframe_to_dict(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """
    Convert pandas DataFrame to dictionary representation.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary representation of DataFrame
    """
    return df.to_dict(orient="records")


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
    logger.debug(f"Date extraction request for column: {request.column}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation
        result_df, metadata = extract_date_components(
            df=df,
            column=request.column,
            components=request.components
        )
        
        # Check for errors
        if metadata.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail=metadata["error"]
            )
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return DateExtractionResponse(
            transformed_df=result_data,
            added_columns=metadata.get("added_columns", []),
            status="success",
            message=metadata.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Date extraction failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
    logger.debug(f"Scaling request for columns: {request.columns}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation for each column
        result_df = df.copy()
        scaled_columns = []
        
        for column in request.columns:
            temp_df, metadata = scale_numeric(
                df=result_df,
                column=column,
                method=request.method,
                custom_range=request.custom_range
            )
            
            # Check for errors
            if metadata.get("status") == "error":
                raise HTTPException(
                    status_code=422,
                    detail=metadata["error"]
                )
            
            result_df = temp_df
            scaled_columns.extend(metadata.get("scaled_columns", []))
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return ScalingResponse(
            transformed_df=result_data,
            scaled_columns=scaled_columns,
            status="success",
            message=f"Successfully scaled {len(scaled_columns)} columns"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scaling failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
    logger.debug(f"Categorical encoding request for columns: {request.columns}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation for each column
        result_df = df.copy()
        encoded_columns = []
        
        for column in request.columns:
            temp_df, metadata = encode_categorical(
                df=result_df,
                column=column,
                method=request.method,
                custom_mapping=request.custom_mapping
            )
            
            # Check for errors
            if metadata.get("status") == "error":
                raise HTTPException(
                    status_code=422,
                    detail=metadata["error"]
                )
            
            result_df = temp_df
            encoded_columns.extend(metadata.get("encoded_columns", []))
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return CategoricalEncodingResponse(
            transformed_df=result_data,
            encoded_columns=encoded_columns,
            status="success",
            message=f"Successfully encoded {len(encoded_columns)} columns"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Categorical encoding failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
    logger.debug(f"Text preprocessing request for column: {request.column}, method: {request.method}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation
        result_df, metadata = tokenize_text(
            df=df,
            column=request.column,
            method=request.method
        )
        
        # Check for errors
        if metadata.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail=metadata["error"]
            )
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return TextPreprocessingResponse(
            transformed_df=result_data,
            processed_column=request.column,
            status="success",
            message=metadata.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text preprocessing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
    logger.debug(f"TF-IDF request for column: {request.column}, max_features: {request.max_features}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation
        result_df, metadata = apply_tfidf_vectorization(
            df=df,
            column=request.column,
            max_features=request.max_features
        )
        
        # Check for errors
        if metadata.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail=metadata["error"]
            )
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return TFIDFResponse(
            transformed_df=result_data,
            tfidf_columns=metadata.get("added_columns", []),
            status=metadata["status"],
            message=metadata.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TF-IDF transformation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
    logger.debug(f"PCA request for columns: {request.columns}, n_components: {request.n_components}")
    
    try:
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(request.dataframe)
        
        # Apply transformation
        result_df, metadata = apply_pca(
            df=df,
            n_components=request.n_components,
            columns=request.columns
        )
        
        # Check for errors
        if metadata.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail=metadata["error"]
            )
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return PCATransformResponse(
            transformed_df=result_data,
            principal_components=metadata.get("added_columns", []),
            status=metadata["status"],
            message=metadata.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PCA transformation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.post("/custom", response_model=CustomTransformationResponse, status_code=200)
async def apply_custom_transformation(request: Dict[str, Any]):
    """
    Apply a custom user-defined transformation function.
    
    Args:
        request: Dictionary containing dataframe, function_name, and function_code
        
    Returns:
        CustomTransformationResponse with transformed data and metadata
        
    Raises:
        HTTPException: If transformation fails
    """
    # TODO: Create CustomTransformationRequest and CustomTransformationResponse schemas
    logger.debug(f"Custom transformation request with function: {request.get('function_name', 'unknown')}")
    
    try:
        # Extract request components
        dataframe_data = request.get("dataframe")
        function_name = request.get("function_name", "custom_function")
        function_code = request.get("function_code")
        
        if not dataframe_data:
            raise ValueError("Missing 'dataframe' in request")
        if not function_code:
            raise ValueError("Missing 'function_code' in request")
        
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(dataframe_data)
        
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
                'round': round
            }
        }
        
        # Execute function code in safe namespace
        try:
            exec(function_code, safe_namespace)
            
            # Extract the function from namespace
            # Assume the function is named in the code or get the last defined function
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
            function_name=function_name
        )
        
        # Check for errors
        if metadata.get("status") == "error":
            raise HTTPException(
                status_code=422,
                detail=metadata["error"]
            )
        
        # Convert result back to dictionary format
        result_data = _convert_dataframe_to_dict(result_df)
        
        return CustomTransformationResponse(
            transformed_df=result_data,
            status=metadata["status"],
            message=metadata.get("message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom transformation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.post("/batch", status_code=200)
async def apply_batch_transformations(request: Dict[str, Any]):
    """
    Apply multiple transformations in sequence.
    
    Args:
        request: Dictionary containing dataframe and transformations list
        
    Returns:
        Dictionary with final_data and metadata list
        
    Raises:
        HTTPException: If transformation fails
    """
    # TODO: Create BatchTransformationRequest and BatchTransformationResponse schemas
    logger.debug(f"Batch transformation request with {len(request.get('transformations', []))} transformations")
    
    try:
        # Extract request components
        dataframe_data = request.get("dataframe")
        transformations = request.get("transformations", [])
        
        if not dataframe_data:
            raise ValueError("Missing 'dataframe' in request")
        if not transformations:
            raise ValueError("Missing 'transformations' in request")
        
        # Convert request data to DataFrame
        df = _convert_dict_to_dataframe(dataframe_data)
        
        # Apply batch transformations
        result_df, metadata_list = apply_multiple_transformations(
            df=df,
            transformations=transformations
        )
        
        # Convert result back to dictionary format
        final_data = _convert_dataframe_to_dict(result_df)
        
        return {
            "transformed_df": final_data,
            "metadata": metadata_list,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Batch transformation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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
            status_code=400,
            detail=str(e)
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
                "status": "cleared"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to clear transformation history: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
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