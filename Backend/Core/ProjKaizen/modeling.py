import pandas as pd
import numpy as np
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
import joblib

from fastapi import HTTPException
from ..schemas.modeling import (
    ModelConfig, TrainingResponse, EvaluationMetrics, 
    PredictionResponse
)
from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

class ModelingService:
    """Production-grade machine learning service for training, evaluation, and prediction."""
    
    def __init__(self):
        self.model_registry_path = Path(settings.ml.model_path) / "model_registry.json"
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        Path(settings.ml.model_path).mkdir(parents=True, exist_ok=True)
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the model registry from disk."""
        if self.model_registry_path.exists():
            try:
                with open(self.model_registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load model registry: {e}")
        return {}
    
    def _save_model_registry(self, registry: Dict[str, Any]) -> None:
        """Save the model registry to disk."""
        try:
            with open(self.model_registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save model registry: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save model registry: {str(e)}"
            )
    
    def _validate_dataframe(self, df: pd.DataFrame, config: ModelConfig) -> None:
        """Validate input dataframe and configuration."""
        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="DataFrame is empty"
            )
        
        if len(df) < 5:
            raise HTTPException(
                status_code=422,
                detail="Dataset too small for training (minimum 5 samples required)"
            )
        
        # Check if target column exists
        if config.target_column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Target column '{config.target_column}' not found in DataFrame. "
                       f"Available columns: {list(df.columns)}"
            )
        
        # Check if feature columns exist
        missing_features = [col for col in config.feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Feature columns not found: {missing_features}. "
                       f"Available columns: {list(df.columns)}"
            )
        
        # Check for sufficient data after removing NaN
        clean_df = df[config.feature_columns + [config.target_column]].dropna()
        if len(clean_df) < 5:
            raise HTTPException(
                status_code=422,
                detail="Insufficient clean data after removing NaN values (minimum 5 samples required)"
            )
    
    def _validate_model_type(self, model_type: str) -> None:
        """Validate model type."""
        if model_type not in ["classification", "regression"]:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported model type: {model_type}. "
                       f"Supported types: ['classification', 'regression']"
            )
    
    def _prepare_data(self, df: pd.DataFrame, config: ModelConfig) -> tuple:
        """Prepare data for training."""
        try:
            # Select features and target
            X = df[config.feature_columns].copy()
            y = df[config.target_column].copy()
            
            # Remove rows with NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            # Split data
            test_size = getattr(config, 'test_size', 0.2)
            random_state = getattr(config, 'random_state', 42)
            
            if len(X) < 5:
                raise ValueError("Insufficient clean data for train/test split")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y if config.model_type == "classification" and len(np.unique(y)) > 1 else None
            )
            
            return X_train, X_test, y_train, y_test, X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Data preparation failed: {str(e)}"
            )
    
    def _initialize_model(self, config: ModelConfig):
        """Initialize the appropriate model based on configuration."""
        try:
            hyperparameters = config.hyperparameters or {}
            
            if config.model_type == "classification":
                # Set default random_state if not provided
                if 'random_state' not in hyperparameters:
                    hyperparameters['random_state'] = getattr(config, 'random_state', 42)
                return RandomForestClassifier(**hyperparameters)
            
            elif config.model_type == "regression":
                # Set default random_state if not provided
                if 'random_state' not in hyperparameters:
                    hyperparameters['random_state'] = getattr(config, 'random_state', 42)
                return RandomForestRegressor(**hyperparameters)
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Model initialization failed. Check hyperparameters: {str(e)}"
            )
    
    def train_model(self, df: pd.DataFrame, config: ModelConfig) -> TrainingResponse:
        """Train a machine learning model based on configuration."""
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        logger.info(f"Starting model training - ID: {model_id}, Type: {config.model_type}")
        logger.info(f"Config: Features={config.feature_columns}, Target={config.target_column}")
        
        try:
            # Validation
            self._validate_model_type(config.model_type)
            self._validate_dataframe(df, config)
            
            # Prepare data
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(df, config)
            
            logger.info(f"Data prepared - Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Initialize and train model
            model = self._initialize_model(config)
            
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            metrics = self.evaluate_model(y_test, y_pred, config.model_type)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save model
            model_filename = f"model_{model_id}.joblib"
            model_path = Path(settings.ml.model_path) / model_filename
            
            try:
                joblib.dump(model, model_path)
                logger.info(f"Model saved to: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save model to {model_path}: {str(e)}"
                )
            
            # Update model registry
            registry = self._load_model_registry()
            registry[model_id] = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": config.model_type,
                "feature_columns": config.feature_columns,
                "target_column": config.target_column,
                "model_path": str(model_path),
                "hyperparameters": config.hyperparameters,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "training_time": training_time,
                "metrics": metrics.dict()
            }
            self._save_model_registry(registry)
            
            # Create response
            response = TrainingResponse(
                model_id=model_id,
                model_type=config.model_type,
                feature_columns=config.feature_columns,
                model_path=str(model_path),
                training_time=training_time,
                train_size=len(X_train),
                test_size=len(X_test)
            )
            
            # Set metrics based on model type
            if config.model_type == "classification":
                response.accuracy = metrics.accuracy
                response.precision = metrics.precision
                response.recall = metrics.recall
                response.f1_score = metrics.f1_score
            else:
                response.mse = metrics.mse
                response.r2_score = metrics.r2_score
            
            logger.info(f"Model training completed successfully - ID: {model_id}")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during training: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during model training: {str(e)}"
            )
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_type: str) -> EvaluationMetrics:
        """Evaluate model performance and return metrics."""
        try:
            metrics = EvaluationMetrics()
            
            if model_type == "classification":
                metrics.accuracy = float(accuracy_score(y_true, y_pred))
                
                # Handle multi-class and binary classification
                average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
                
                metrics.precision = float(precision_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                metrics.recall = float(recall_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                metrics.f1_score = float(f1_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                
                # Add confusion matrix if requested
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    metrics.confusion_matrix = cm.tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute confusion matrix: {e}")
                    
            elif model_type == "regression":
                metrics.mse = float(mean_squared_error(y_true, y_pred))
                metrics.r2_score = float(r2_score(y_true, y_pred))
                metrics.rmse = float(np.sqrt(metrics.mse))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model evaluation failed: {str(e)}"
            )
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        try:
            path = Path(model_path)
            if not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {model_path}"
                )
            
            model = joblib.load(path)
            logger.info(f"Model loaded successfully from: {model_path}")
            return model
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def predict(self, model_path: str, input_data: pd.DataFrame) -> PredictionResponse:
        """Make predictions using a trained model."""
        try:
            # Load model
            model = self.load_model(model_path)
            
            # Get model info from registry
            model_id = Path(model_path).stem.replace('model_', '')
            registry = self._load_model_registry()
            model_info = registry.get(model_id, {})
            
            model_type = model_info.get('model_type', 'unknown')
            feature_columns = model_info.get('feature_columns', [])
            
            # Validate input data
            if input_data.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Input data is empty"
                )
            
            # Check if required features are present
            if feature_columns:
                missing_features = [col for col in feature_columns if col not in input_data.columns]
                if missing_features:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required features: {missing_features}. "
                               f"Expected features: {feature_columns}"
                    )
                
                # Select only the required features in the correct order
                X = input_data[feature_columns]
            else:
                X = input_data
            
            # Handle missing values
            if X.isnull().any().any():
                logger.warning("Input data contains NaN values, predictions may be unreliable")
            
            # Make predictions
            predictions = model.predict(X)
            
            # Get confidence scores if available (for classification)
            confidence_scores = None
            if hasattr(model, 'predict_proba') and model_type == 'classification':
                try:
                    proba = model.predict_proba(X)
                    confidence_scores = np.max(proba, axis=1).tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute confidence scores: {e}")
            
            response = PredictionResponse(
                predictions=predictions.tolist(),
                model_id=model_id,
                model_type=model_type,
                confidence_scores=confidence_scores,
                num_predictions=len(predictions)
            )
            
            logger.info(f"Predictions generated successfully - {len(predictions)} samples")
            return response
            
        except HTTPException:
            raise
        except NotFittedError:
            raise HTTPException(
                status_code=400,
                detail="Model is not fitted. Please train the model first."
            )
        except ValueError as e:
            if "shape" in str(e).lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Input data shape mismatch: {str(e)}"
                )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input data: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model from the registry."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found in registry"
            )
        
        return registry[model_id]
    
    def list_models(self) -> Dict[str, Any]:
        """List all models in the registry."""
        return self._load_model_registry()
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and remove it from the registry."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        model_info = registry[model_id]
        model_path = Path(model_info['model_path'])
        
        try:
            # Remove model file
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Model file deleted: {model_path}")
            
            # Remove from registry
            del registry[model_id]
            self._save_model_registry(registry)
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model: {str(e)}"
            )


# Create service instance
modeling_service = ModelingService()

# Convenience functions for backward compatibility
def train_model(df: pd.DataFrame, config: ModelConfig) -> TrainingResponse:
    """Train a machine learning model based on configuration."""
    return modeling_service.train_model(df, config)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_type: str) -> EvaluationMetrics:
    """Evaluate model performance and return metrics."""
    return modeling_service.evaluate_model(y_true, y_pred, model_type)

def load_model(model_path: str):
    """Load a trained model from disk."""
    return modeling_service.load_model(model_path)

def predict(model_path: str, input_data: pd.DataFrame) -> PredictionResponse:
    """Make predictions using a trained model."""
    return modeling_service.predict(model_path, input_data)