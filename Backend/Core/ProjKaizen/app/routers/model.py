from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from typing import List, Optional
from app.schemas.modeling import (
    ModelConfig,
    TrainingResponse,
    PredictionResponse,
    EvaluationMetrics,
    ModelInfo,
    ModelListResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ExportResponse,
    CleanupResponse,
    HealthResponse,
    FieldValidationError
)
from app.services import modeling as modeling_service
from app.core.auth import verify_token
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/modeling", tags=["Modeling"])


@router.post(
    "/train",
    response_model=TrainingResponse,
    summary="Train a new model",
    description="Train a new machine learning model with the provided configuration",
    dependencies=[Depends(verify_token)]
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
    """
    try:
        logger.info(f"Starting model training with config: {config.dict()}")
        result = modeling_service.train_model(config)
        logger.info(f"Model training completed successfully. Model ID: {result.model_id}")
        return result
    except ValueError as e:
        logger.error(f"Invalid configuration for model training: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during model training")
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")


@router.post(
    "/predict/{model_id}",
    response_model=PredictionResponse,
    summary="Make predictions using a trained model",
    description="Generate predictions using a previously trained model",
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
        HTTPException: 404 if model not found, 400 if prediction fails
    """
    try:
        logger.info(f"Making predictions with model {model_id}")
        result = modeling_service.make_predictions(model_id, data)
        logger.info(f"Predictions completed successfully for model {model_id}")
        return result
    except FileNotFoundError:
        logger.error(f"Model {model_id} not found")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except ValueError as e:
        logger.error(f"Invalid input data for prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        logger.exception(f"Error during prediction with model {model_id}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@router.get(
    "/list",
    response_model=ModelListResponse,
    summary="List all trained models",
    description="Retrieve a list of all available trained models",
    dependencies=[Depends(verify_token)]
)
async def list_models():
    """
    List all trained models available in the system.
    
    Returns:
        ModelListResponse: List of models with basic information
        
    Raises:
        HTTPException: 500 if unable to retrieve model list
    """
    try:
        logger.info("Retrieving list of all trained models")
        result = modeling_service.list_models()
        logger.info(f"Successfully retrieved {len(result.models)} models")
        return result
    except Exception as e:
        logger.exception("Error retrieving model list")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get(
    "/info/{model_id}",
    response_model=ModelInfo,
    summary="Get information about a specific model",
    description="Retrieve detailed information about a trained model",
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
        ModelInfo: Detailed model information including metrics and metadata
        
    Raises:
        HTTPException: 404 if model not found, 500 if unable to retrieve info
    """
    try:
        logger.info(f"Retrieving information for model {model_id}")
        result = modeling_service.get_model_info(model_id)
        logger.info(f"Successfully retrieved info for model {model_id}")
        return result
    except FileNotFoundError:
        logger.error(f"Model {model_id} not found")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.exception(f"Error retrieving info for model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")


@router.delete(
    "/delete/{model_id}",
    summary="Delete a trained model",
    description="Permanently delete a trained model and its associated files",
    dependencies=[Depends(verify_token)]
)
async def delete_model(
    model_id: str = Path(..., description="ID of the model to delete")
):
    """
    Delete a trained model.
    
    Args:
        model_id: Unique identifier of the model to delete
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: 404 if model not found, 500 if deletion fails
    """
    try:
        logger.info(f"Deleting model {model_id}")
        modeling_service.delete_model(model_id)
        logger.info(f"Model {model_id} deleted successfully")
        return {"message": f"Model {model_id} deleted successfully"}
    except FileNotFoundError:
        logger.error(f"Cannot delete model {model_id}: not found")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.exception(f"Error deleting model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare multiple trained models",
    description="Compare performance metrics across multiple trained models",
    dependencies=[Depends(verify_token)]
)
async def compare_models(request: ModelComparisonRequest):
    """
    Compare multiple trained models.
    
    Args:
        request: Comparison request containing model IDs and comparison criteria
        
    Returns:
        ModelComparisonResponse: Comparison results with metrics and rankings
        
    Raises:
        HTTPException: 400 if comparison fails or invalid request
    """
    try:
        logger.info(f"Comparing models: {request.model_ids}")
        result = modeling_service.compare_models(request)
        logger.info(f"Model comparison completed for {len(request.model_ids)} models")
        return result
    except FileNotFoundError as e:
        logger.error(f"One or more models not found during comparison: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid comparison request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid comparison request: {str(e)}")
    except Exception as e:
        logger.exception("Error during model comparison")
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")


@router.post(
    "/export/{model_id}",
    response_model=ExportResponse,
    summary="Export a trained model to file",
    description="Export a trained model in a portable format for deployment",
    dependencies=[Depends(verify_token)]
)
async def export_model(
    model_id: str = Path(..., description="ID of the model to export")
):
    """
    Export a trained model to a file.
    
    Args:
        model_id: Unique identifier of the model to export
        
    Returns:
        ExportResponse: Export details including file path and format
        
    Raises:
        HTTPException: 404 if model not found, 500 if export fails
    """
    try:
        logger.info(f"Exporting model {model_id}")
        result = modeling_service.export_model(model_id)
        logger.info(f"Model {model_id} exported successfully to {result.file_path}")
        return result
    except FileNotFoundError:
        logger.error(f"Cannot export model {model_id}: not found")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.exception(f"Error exporting model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")


@router.post(
    "/cleanup",
    response_model=CleanupResponse,
    summary="Remove unused or old models",
    description="Clean up unused or old models to free up storage space",
    dependencies=[Depends(verify_token)]
)
async def cleanup_models():
    """
    Clean up unused or old models.
    
    Returns:
        CleanupResponse: Details of cleanup operation including removed models
        
    Raises:
        HTTPException: 500 if cleanup fails
    """
    try:
        logger.info("Starting model cleanup operation")
        result = modeling_service.cleanup_models()
        logger.info(f"Cleanup completed. Removed {result.deleted_count} models, freed {result.space_freed_mb} MB")
        return result
    except Exception as e:
        logger.exception("Error during model cleanup")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check modeling service health",
    description="Check the health status of the modeling service"
)
async def health_check():
    """
    Check the health of the modeling service.
    
    Returns:
        HealthResponse: Service health status and diagnostic information
        
    Raises:
        HTTPException: 500 if health check fails
    """
    try:
        logger.debug("Performing modeling service health check")
        result = modeling_service.health_check()
        logger.debug(f"Health check completed. Status: {result.status}")
        return result
    except Exception as e:
        logger.exception("Error during health check")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")