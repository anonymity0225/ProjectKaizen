from fastapi import APIRouter, Depends, HTTPException, Path, Body, status
from typing import List, Optional
from app.schemas.modeling import (
    ModelConfig,
    TrainingResponse,
    PredictionResponse,
    ModelInfo,
    ModelListResponse,
    ModelConfigExport,
    CleanupResult,
    ModelHealthStatus
)
from app.services import modeling as modeling_service
from app.core.auth import verify_token
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/modeling", tags=["Modeling"])


@router.post(
    "/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train a new machine learning model",
    description="Train a new machine learning model with the provided configuration including algorithm, hyperparameters, and training data"
)
async def train_model(config: ModelConfig):
    """
    Train a new model using the provided configuration.
    
    Args:
        config: Model configuration including algorithm, hyperparameters, and training data
        
    Returns:
        TrainingResponse: Training results including model ID and performance metrics
        
    Raises:
        HTTPException: 400 if training fails due to invalid configuration or data issues
        HTTPException: 422 if validation errors occur
    """
    try:
        logger.info(f"Starting model training with config: {config.dict()}")
        result = await modeling_service.train_model(config)
        logger.info(f"Model training completed successfully. Model ID: {result.model_id}")
        return result
    except ValueError as e:
        logger.error(f"Invalid configuration for model training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid configuration: {str(e)}"
        )
    except Exception as e:
        logger.exception("Unexpected error during model training")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Training failed: {str(e)}"
        )


@router.post(
    "/predict/{model_id}",
    response_model=PredictionResponse,
    summary="Make predictions using a trained model",
    description="Generate predictions using a previously trained model with input data",
    dependencies=[Depends(verify_token)]
)
async def predict(
    model_id: str = Path(..., description="ID of the trained model to use for prediction"),
    data: dict = Body(..., description="Input data for prediction")
):
    """
    Make predictions using a trained model.
    
    Args:
        model_id: Unique identifier of the trained model
        data: Input data for making predictions
        
    Returns:
        PredictionResponse: Prediction results and confidence scores
        
    Raises:
        HTTPException: 404 if model not found
        HTTPException: 400 if prediction fails due to invalid input data
    """
    try:
        logger.info(f"Making predictions with model {model_id}")
        result = await modeling_service.predict(model_id, data)
        logger.info(f"Predictions completed successfully for model {model_id}")
        return result
    except FileNotFoundError:
        logger.error(f"Model {model_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id} not found"
        )
    except ValueError as e:
        logger.error(f"Invalid input data for prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Error during prediction with model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/list",
    response_model=ModelListResponse,
    summary="List all trained models",
    description="Retrieve a list of all available trained models with basic information",
    dependencies=[Depends(verify_token)]
)
async def list_models():
    """
    List all trained models available in the system.
    
    Returns:
        ModelListResponse: List of models with basic information including IDs, names, and creation dates
        
    Raises:
        HTTPException: 500 if unable to retrieve model list
    """
    try:
        logger.info("Retrieving list of all trained models")
        result = await modeling_service.list_models()
        logger.info(f"Successfully retrieved {len(result.models)} models")
        return result
    except Exception as e:
        logger.exception("Error retrieving model list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to retrieve models: {str(e)}"
        )


@router.get(
    "/info/{model_id}",
    response_model=ModelInfo,
    summary="Get information about a specific model",
    description="Retrieve detailed information about a trained model including metadata and performance metrics",
    dependencies=[Depends(verify_token)]
)
async def get_model_info(
    model_id: str = Path(..., description="ID of the model to get information about")
):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Unique identifier of the model
        
    Returns:
        ModelInfo: Detailed model information including metrics, metadata, and configuration
        
    Raises:
        HTTPException: 404 if model not found
        HTTPException: 500 if unable to retrieve model information
    """
    try:
        logger.info(f"Retrieving information for model {model_id}")
        result = await modeling_service.get_model_info(model_id)
        logger.info(f"Successfully retrieved info for model {model_id}")
        return result
    except FileNotFoundError:
        logger.error(f"Model {model_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id} not found"
        )
    except Exception as e:
        logger.exception(f"Error retrieving info for model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@router.delete(
    "/delete/{model_id}",
    response_model=dict,
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a trained model",
    description="Permanently delete a trained model and its associated files from the system",
    dependencies=[Depends(verify_token)]
)
async def delete_model(
    model_id: str = Path(..., description="ID of the model to delete")
):
    """
    Delete a trained model permanently.
    
    Args:
        model_id: Unique identifier of the model to delete
        
    Returns:
        dict: Confirmation message with deleted model ID
        
    Raises:
        HTTPException: 404 if model not found
        HTTPException: 500 if deletion fails
    """
    try:
        logger.info(f"Deleting model {model_id}")
        result = await modeling_service.delete_model(model_id)
        logger.info(f"Model {model_id} deleted successfully")
        return result
    except FileNotFoundError:
        logger.error(f"Cannot delete model {model_id}: not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id} not found"
        )
    except Exception as e:
        logger.exception(f"Error deleting model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get(
    "/compare",
    response_model=dict,
    summary="Compare multiple trained models",
    description="Compare performance metrics across multiple trained models to help select the best model",
    dependencies=[Depends(verify_token)]
)
async def compare_models(
    model_ids: List[str] = Body(..., description="List of model IDs to compare")
):
    """
    Compare multiple trained models based on their performance metrics.
    
    Args:
        model_ids: List of unique identifiers of models to compare
        
    Returns:
        dict: Comparison results with metrics and rankings for each model
        
    Raises:
        HTTPException: 404 if one or more models not found
        HTTPException: 400 if comparison fails or invalid request
    """
    try:
        logger.info(f"Comparing models: {model_ids}")
        result = await modeling_service.get_model_performance_comparison(model_ids)
        logger.info(f"Model comparison completed for {len(model_ids)} models")
        return result
    except FileNotFoundError as e:
        logger.error(f"One or more models not found during comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model not found: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Invalid comparison request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid comparison request: {str(e)}"
        )
    except Exception as e:
        logger.exception("Error during model comparison")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Comparison failed: {str(e)}"
        )


@router.get(
    "/export/{model_id}",
    response_model=ModelConfigExport,
    summary="Export a trained model configuration",
    description="Export a trained model's configuration in a portable format for deployment or backup",
    dependencies=[Depends(verify_token)]
)
async def export_model(
    model_id: str = Path(..., description="ID of the model to export")
):
    """
    Export a trained model's configuration to a portable format.
    
    Args:
        model_id: Unique identifier of the model to export
        
    Returns:
        ModelConfigExport: Export details including configuration data and metadata
        
    Raises:
        HTTPException: 404 if model not found
        HTTPException: 500 if export fails
    """
    try:
        logger.info(f"Exporting model configuration for {model_id}")
        result = await modeling_service.export_model_config(model_id)
        logger.info(f"Model {model_id} configuration exported successfully")
        return result
    except FileNotFoundError:
        logger.error(f"Cannot export model {model_id}: not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id} not found"
        )
    except Exception as e:
        logger.exception(f"Error exporting model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to export model: {str(e)}"
        )


@router.post(
    "/cleanup",
    response_model=CleanupResult,
    summary="Remove broken or corrupted models",
    description="Clean up broken or corrupted models to free up storage space and maintain system health",
    dependencies=[Depends(verify_token)]
)
async def cleanup_models():
    """
    Clean up broken or corrupted models from the system.
    
    Returns:
        CleanupResult: Details of cleanup operation including removed models and space freed
        
    Raises:
        HTTPException: 500 if cleanup fails
    """
    try:
        logger.info("Starting model cleanup operation")
        result = await modeling_service.cleanup_broken_models()
        logger.info(f"Cleanup completed. Removed {result.deleted_count} models, freed {result.space_freed_mb} MB")
        return result
    except Exception as e:
        logger.exception("Error during model cleanup")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get(
    "/health/{model_id}",
    response_model=ModelHealthStatus,
    summary="Check model health status",
    description="Validate the health and integrity of a specific model"
)
async def validate_model_health(
    model_id: str = Path(..., description="ID of the model to validate")
):
    """
    Check the health status and integrity of a specific model.
    
    Args:
        model_id: Unique identifier of the model to validate
        
    Returns:
        ModelHealthStatus: Health status information including validation results and diagnostics
        
    Raises:
        HTTPException: 404 if model not found
        HTTPException: 500 if health check fails
    """
    try:
        logger.info(f"Performing health check for model {model_id}")
        result = await modeling_service.validate_model_health(model_id)
        logger.info(f"Health check completed for model {model_id}. Status: {result.status}")
        return result
    except FileNotFoundError:
        logger.error(f"Model {model_id} not found for health check")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_id} not found"
        )
    except Exception as e:
        logger.exception(f"Error during health check for model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Health check failed: {str(e)}"
        )