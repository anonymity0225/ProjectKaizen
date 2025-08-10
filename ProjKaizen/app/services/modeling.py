import pandas as pd
import numpy as np
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import time
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

class ModelAlgorithm(str, Enum):
    """Supported machine learning algorithms."""
    # Classification algorithms
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    SVC = "svc"
    KNN_CLASSIFIER = "knn_classifier"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    
    # Regression algorithms
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    SVR = "svr"
    KNN_REGRESSOR = "knn_regressor"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"

class ProblemType(str, Enum):
    """Machine learning problem types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class ModelingService:
    """Production-grade machine learning service for training, evaluation, and prediction."""
    
    def __init__(self):
        self.model_registry_path = Path(settings.ml.model_path) / "model_registry.json"
        self._ensure_directories()
        
        # Define available algorithms
        self.classification_algorithms = {
            ModelAlgorithm.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
            ModelAlgorithm.LOGISTIC_REGRESSION: LogisticRegression,
            ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier,
            ModelAlgorithm.SVC: SVC,
            ModelAlgorithm.KNN_CLASSIFIER: KNeighborsClassifier,
            ModelAlgorithm.DECISION_TREE_CLASSIFIER: DecisionTreeClassifier,
        }
        
        self.regression_algorithms = {
            ModelAlgorithm.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
            ModelAlgorithm.LINEAR_REGRESSION: LinearRegression,
            ModelAlgorithm.RIDGE_REGRESSION: Ridge,
            ModelAlgorithm.LASSO_REGRESSION: Lasso,
            ModelAlgorithm.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor,
            ModelAlgorithm.SVR: SVR,
            ModelAlgorithm.KNN_REGRESSOR: KNeighborsRegressor,
            ModelAlgorithm.DECISION_TREE_REGRESSOR: DecisionTreeRegressor,
        }
        
        # Default hyperparameters for each algorithm
        self.default_hyperparameters = {
            # Classification
            ModelAlgorithm.RANDOM_FOREST_CLASSIFIER: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.LOGISTIC_REGRESSION: {"random_state": 42, "max_iter": 1000},
            ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER: {"random_state": 42},
            ModelAlgorithm.SVC: {"random_state": 42, "probability": True},  # Enable probability for confidence scores
            ModelAlgorithm.KNN_CLASSIFIER: {"n_neighbors": 5},
            ModelAlgorithm.DECISION_TREE_CLASSIFIER: {"random_state": 42},
            
            # Regression
            ModelAlgorithm.RANDOM_FOREST_REGRESSOR: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.LINEAR_REGRESSION: {},
            ModelAlgorithm.RIDGE_REGRESSION: {"random_state": 42},
            ModelAlgorithm.LASSO_REGRESSION: {"random_state": 42},
            ModelAlgorithm.GRADIENT_BOOSTING_REGRESSOR: {"random_state": 42},
            ModelAlgorithm.SVR: {},
            ModelAlgorithm.KNN_REGRESSOR: {"n_neighbors": 5},
            ModelAlgorithm.DECISION_TREE_REGRESSOR: {"random_state": 42},
        }
    
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
        if df is None or df.empty:
            raise HTTPException(
                status_code=422,
                detail="DataFrame is empty or None"
            )
        
        if len(df) < 10:  # Increased minimum for better model training
            raise HTTPException(
                status_code=422,
                detail=f"Dataset too small for training (minimum 10 samples required, got {len(df)})"
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
        if len(clean_df) < 10:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient clean data after removing NaN values "
                       f"(minimum 10 samples required, got {len(clean_df)} clean samples)"
            )
        
        # Check for feature-target correlation issues
        if len(clean_df[config.target_column].unique()) == 1:
            raise HTTPException(
                status_code=422,
                detail="Target column has only one unique value. Cannot train a meaningful model."
            )
    
    def _detect_problem_type(self, df: pd.DataFrame, target_column: str) -> ProblemType:
        """Automatically detect whether this is a classification or regression problem."""
        target_series = df[target_column].dropna()
        
        # If target is clearly non-numeric, it's classification
        if target_series.dtype.kind not in 'biufc':
            return ProblemType.CLASSIFICATION
        
        # For numeric targets, use heuristics
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # If very few unique values relative to dataset size, likely classification
        if unique_values <= min(10, max(2, total_values * 0.05)):
            return ProblemType.CLASSIFICATION
        
        # If all values are integers and reasonable number of unique values, could be classification
        if target_series.dtype.kind in 'iu' and unique_values <= 20:
            return ProblemType.CLASSIFICATION
        
        # Otherwise, assume regression
        return ProblemType.REGRESSION
    
    def _validate_model_type(self, model_type: str) -> None:
        """Validate model type."""
        if model_type not in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported model type: {model_type}. "
                       f"Supported types: {[e.value for e in ProblemType]}"
            )
    
    def _prepare_data(self, df: pd.DataFrame, config: ModelConfig) -> tuple:
        """Prepare data for training with improved error handling."""
        try:
            # Select features and target
            X = df[config.feature_columns].copy()
            y = df[config.target_column].copy()
            
            # Handle missing values with explicit strategy
            initial_rows = len(X)
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            removed_rows = initial_rows - len(X)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing values ({removed_rows/initial_rows*100:.1f}%)")
            
            if len(X) < 10:
                raise HTTPException(
                    status_code=422,
                    detail=f"Insufficient clean data for train/test split (got {len(X)} samples after cleaning)"
                )
            
            # Check for non-numeric features and raise informative error
            non_numeric_features = []
            for col in X.columns:
                if X[col].dtype.kind not in 'biufc':
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                raise HTTPException(
                    status_code=422,
                    detail=f"Non-numeric features detected: {non_numeric_features}. "
                           f"Please encode categorical features before training."
                )
            
            # Split data with improved error handling
            test_size = getattr(config, 'test_size', 0.2)
            random_state = getattr(config, 'random_state', 42)
            
            # For small datasets, ensure at least 2 samples in test set
            if len(X) * test_size < 2:
                test_size = max(2 / len(X), 0.1)
                logger.warning(f"Adjusted test_size to {test_size:.3f} to ensure minimum test samples")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=y if config.model_type == ProblemType.CLASSIFICATION and len(np.unique(y)) > 1 else None
                )
            except ValueError as e:
                if "stratify" in str(e):
                    # Retry without stratification
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    logger.warning("Disabled stratification due to insufficient samples per class")
                else:
                    raise
            
            return X_train, X_test, y_train, y_test, X, y
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Data preparation failed: {str(e)}"
            )
    
    def _get_algorithms_for_problem_type(self, problem_type: ProblemType, 
                                       specific_algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get available algorithms for the given problem type."""
        if problem_type == ProblemType.CLASSIFICATION:
            algorithms = self.classification_algorithms
        else:
            algorithms = self.regression_algorithms
        
        if specific_algorithm:
            if specific_algorithm not in algorithms:
                available = list(algorithms.keys())
                raise HTTPException(
                    status_code=422,
                    detail=f"Algorithm '{specific_algorithm}' not available for {problem_type}. "
                           f"Available algorithms: {available}"
                )
            return {specific_algorithm: algorithms[specific_algorithm]}
        
        return algorithms
    
    def _initialize_model(self, algorithm: str, hyperparameters: Optional[Dict] = None):
        """Initialize a model with given algorithm and hyperparameters."""
        try:
            # Get the model class
            all_algorithms = {**self.classification_algorithms, **self.regression_algorithms}
            model_class = all_algorithms.get(algorithm)
            
            if not model_class:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Merge default hyperparameters with provided ones
            default_params = self.default_hyperparameters.get(algorithm, {})
            final_params = {**default_params}
            if hyperparameters:
                final_params.update(hyperparameters)
            
            # Initialize model
            model = model_class(**final_params)
            return model
            
        except Exception as e:
            logger.error(f"Model initialization failed for {algorithm}: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Model initialization failed for {algorithm}: {str(e)}"
            )
    
    def train_model(self, df: pd.DataFrame, config: ModelConfig) -> TrainingResponse:
        """Train machine learning models with auto-selection of best performing algorithm."""
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        logger.info(f"Starting model training - ID: {model_id}")
        logger.info(f"Config: Features={config.feature_columns}, Target={config.target_column}")
        
        try:
            # Auto-detect problem type if not specified
            if not config.model_type:
                config.model_type = self._detect_problem_type(df, config.target_column)
                logger.info(f"Auto-detected problem type: {config.model_type}")
            
            # Validation
            self._validate_model_type(config.model_type)
            self._validate_dataframe(df, config)
            
            # Prepare data
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(df, config)
            
            logger.info(f"Data prepared - Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Get algorithms to try
            specific_algorithm = getattr(config, 'algorithm', None)
            algorithms = self._get_algorithms_for_problem_type(config.model_type, specific_algorithm)
            
            best_model = None
            best_algorithm = None
            best_score = -np.inf
            best_metrics = None
            all_results = {}
            
            # Train and evaluate each algorithm
            for algorithm_name, algorithm_class in algorithms.items():
                try:
                    logger.info(f"Training {algorithm_name}...")
                    model_start_time = time.time()
                    
                    # Initialize model
                    model = self._initialize_model(algorithm_name, config.hyperparameters)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluate model
                    metrics = self.evaluate_model(y_test, y_pred, config.model_type)
                    
                    # Calculate score for model selection
                    if config.model_type == ProblemType.CLASSIFICATION:
                        score = metrics.accuracy
                    else:
                        score = metrics.r2_score
                    
                    model_time = time.time() - model_start_time
                    
                    all_results[algorithm_name] = {
                        "score": score,
                        "metrics": metrics,
                        "training_time": model_time
                    }
                    
                    # Update best model if this one is better
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_algorithm = algorithm_name
                        best_metrics = metrics
                    
                    logger.info(f"Completed {algorithm_name} - Score: {score:.4f}, Time: {model_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {algorithm_name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise HTTPException(
                    status_code=500,
                    detail="No model could be trained successfully. Check your data and configuration."
                )
            
            total_training_time = time.time() - start_time
            logger.info(f"Best algorithm: {best_algorithm} with score: {best_score:.4f}")
            
            # Save best model
            model_filename = f"model_{model_id}.joblib"
            model_path = Path(settings.ml.model_path) / model_filename
            
            try:
                joblib.dump(best_model, model_path)
                logger.info(f"Model saved to: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save model to {model_path}: {str(e)}"
                )
            
            # Update model registry with comprehensive information
            registry = self._load_model_registry()
            registry[model_id] = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": config.model_type,
                "algorithm": best_algorithm,
                "feature_columns": config.feature_columns,
                "target_column": config.target_column,
                "model_path": str(model_path),
                "hyperparameters": config.hyperparameters,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "total_training_time": total_training_time,
                "best_score": best_score,
                "metrics": best_metrics.dict(),
                "all_algorithm_results": {k: {"score": v["score"], "training_time": v["training_time"]} 
                                        for k, v in all_results.items()},
                "data_info": {
                    "total_samples": len(df),
                    "clean_samples": len(X_full),
                    "features_count": len(config.feature_columns),
                    "target_unique_values": int(df[config.target_column].nunique())
                }
            }
            self._save_model_registry(registry)
            
            # Create response
            response = TrainingResponse(
                model_id=model_id,
                model_type=config.model_type,
                feature_columns=config.feature_columns,
                model_path=str(model_path),
                training_time=total_training_time,
                train_size=len(X_train),
                test_size=len(X_test)
            )
            
            # Set metrics based on model type
            if config.model_type == ProblemType.CLASSIFICATION:
                response.accuracy = best_metrics.accuracy
                response.precision = best_metrics.precision
                response.recall = best_metrics.recall
                response.f1_score = best_metrics.f1_score
            else:
                response.mse = best_metrics.mse
                response.r2_score = best_metrics.r2_score
            
            # Add best algorithm info to response
            response.algorithm = best_algorithm
            response.best_score = best_score
            
            logger.info(f"Model training completed successfully - ID: {model_id}, Algorithm: {best_algorithm}")
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
        """Evaluate model performance and return comprehensive metrics."""
        try:
            metrics = EvaluationMetrics()
            
            if model_type == ProblemType.CLASSIFICATION:
                metrics.accuracy = float(accuracy_score(y_true, y_pred))
                
                # Handle multi-class and binary classification
                unique_labels = len(np.unique(y_true))
                average_method = 'weighted' if unique_labels > 2 else 'binary'
                
                metrics.precision = float(precision_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                metrics.recall = float(recall_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                metrics.f1_score = float(f1_score(
                    y_true, y_pred, average=average_method, zero_division=0
                ))
                
                # Add confusion matrix
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    metrics.confusion_matrix = cm.tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute confusion matrix: {e}")
                    
            elif model_type == ProblemType.REGRESSION:
                metrics.mse = float(mean_squared_error(y_true, y_pred))
                metrics.r2_score = float(r2_score(y_true, y_pred))
                metrics.rmse = float(np.sqrt(metrics.mse))
                
                # Add mean absolute error
                try:
                    from sklearn.metrics import mean_absolute_error
                    metrics.mae = float(mean_absolute_error(y_true, y_pred))
                except ImportError:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model evaluation failed: {str(e)}"
            )
    
    def load_model(self, model_path: str):
        """Load a trained model from disk with enhanced error handling."""
        try:
            path = Path(model_path)
            if not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {model_path}"
                )
            
            if not path.suffix == '.joblib':
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model file format. Expected .joblib, got {path.suffix}"
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
    
    def predict(self, model_path: str, input_data: pd.DataFrame, 
               validate_input: bool = True) -> PredictionResponse:
        """Make predictions using a trained model with comprehensive validation."""
        try:
            # Load model
            model = self.load_model(model_path)
            
            # Get model info from registry
            model_id = Path(model_path).stem.replace('model_', '')
            registry = self._load_model_registry()
            model_info = registry.get(model_id, {})
            
            model_type = model_info.get('model_type', 'unknown')
            feature_columns = model_info.get('feature_columns', [])
            algorithm = model_info.get('algorithm', 'unknown')
            
            # Validate input data
            if input_data is None or input_data.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Input data is empty or None"
                )
            
            if validate_input and feature_columns:
                # Check if required features are present
                missing_features = [col for col in feature_columns if col not in input_data.columns]
                if missing_features:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required features: {missing_features}. "
                               f"Expected features: {feature_columns}"
                    )
                
                # Check for extra features
                extra_features = [col for col in input_data.columns if col not in feature_columns]
                if extra_features:
                    logger.warning(f"Input contains extra features that will be ignored: {extra_features}")
                
                # Select only the required features in the correct order
                X = input_data[feature_columns].copy()
            else:
                X = input_data.copy()
            
            # Handle missing values with strict validation
            if X.isnull().any().any():
                null_columns = X.columns[X.isnull().any()].tolist()
                null_count = X.isnull().sum().sum()
                
                if validate_input:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input data contains {null_count} missing values in columns: {null_columns}. "
                               f"Please handle missing values before prediction."
                    )
                else:
                    logger.warning(f"Input data contains missing values in columns: {null_columns}")
            
            # Check for non-numeric data
            non_numeric_cols = []
            for col in X.columns:
                if X[col].dtype.kind not in 'biufc':
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Non-numeric features detected: {non_numeric_cols}. "
                           f"Please encode categorical features before prediction."
                )
            
            # Make predictions
            try:
                predictions = model.predict(X)
            except ValueError as e:
                if "feature" in str(e).lower():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Feature mismatch error: {str(e)}. "
                               f"Expected features: {feature_columns if feature_columns else 'unknown'}"
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prediction error: {str(e)}"
                    )
            
            # Get confidence scores if available (for classification)
            confidence_scores = None
            if hasattr(model, 'predict_proba') and model_type == ProblemType.CLASSIFICATION:
                try:
                    proba = model.predict_proba(X)
                    confidence_scores = np.max(proba, axis=1).tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute confidence scores: {e}")
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_') and feature_columns:
                try:
                    importance_dict = dict(zip(feature_columns, model.feature_importances_))
                    # Sort by importance
                    feature_importance = dict(sorted(importance_dict.items(), 
                                                   key=lambda x: x[1], reverse=True))
                except Exception as e:
                    logger.warning(f"Failed to compute feature importance: {e}")
            
            response = PredictionResponse(
                predictions=predictions.tolist(),
                model_id=model_id,
                model_type=model_type,
                confidence_scores=confidence_scores,
                num_predictions=len(predictions)
            )
            
            # Add additional info
            response.algorithm = algorithm
            response.feature_importance = feature_importance
            
            logger.info(f"Predictions generated successfully - {len(predictions)} samples using {algorithm}")
            return response
            
        except HTTPException:
            raise
        except NotFittedError:
            raise HTTPException(
                status_code=400,
                detail="Model is not fitted. Please train the model first."
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific model from the registry."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found in registry"
            )
        
        model_info = registry[model_id].copy()
        
        # Add model file existence check
        model_path = Path(model_info.get('model_path', ''))
        model_info['model_file_exists'] = model_path.exists()
        model_info['model_file_size'] = model_path.stat().st_size if model_path.exists() else 0
        
        return model_info
    
    def list_models(self, include_stats: bool = True) -> Dict[str, Any]:
        """List all models in the registry with optional statistics."""
        registry = self._load_model_registry()
        
        if not include_stats:
            return registry
        
        # Add summary statistics
        total_models = len(registry)
        classification_models = sum(1 for model in registry.values() 
                                  if model.get('model_type') == ProblemType.CLASSIFICATION)
        regression_models = sum(1 for model in registry.values() 
                              if model.get('model_type') == ProblemType.REGRESSION)
        
        # Get algorithm distribution
        algorithm_counts = {}
        for model in registry.values():
            algorithm = model.get('algorithm', 'unknown')
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        # Check for broken models (missing files)
        broken_models = []
        for model_id, model_info in registry.items():
            model_path = Path(model_info.get('model_path', ''))
            if not model_path.exists():
                broken_models.append(model_id)
        
        summary = {
            "total_models": total_models,
            "classification_models": classification_models,
            "regression_models": regression_models,
            "algorithm_distribution": algorithm_counts,
            "broken_models": broken_models,
            "models": registry
        }
        
        return summary
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model and remove it from the registry with safety checks."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        model_info = registry[model_id]
        model_path = Path(model_info['model_path'])
        
        try:
            # Safety check - prevent deletion of recently used models unless forced
            if not force:
                timestamp_str = model_info.get('timestamp', '')
                if timestamp_str:
                    try:
                        model_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        time_diff = datetime.now() - model_timestamp.replace(tzinfo=None)
                        if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                            logger.warning(f"Attempting to delete recent model {model_id} (created {time_diff})")
                    except ValueError:
                        pass  # Invalid timestamp format, proceed with deletion
            
            # Remove model file
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Model file deleted: {model_path}")
            else:
                logger.warning(f"Model file not found during deletion: {model_path}")
            
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
    
    def cleanup_broken_models(self) -> Dict[str, Any]:
        """Clean up models that have missing files or corrupted registry entries."""
        registry = self._load_model_registry()
        cleaned_models = []
        failed_cleanups = []
        
        for model_id, model_info in list(registry.items()):
            model_path = Path(model_info.get('model_path', ''))
            
            # Check if model file exists
            if not model_path.exists():
                try:
                    del registry[model_id]
                    cleaned_models.append(model_id)
                    logger.info(f"Removed registry entry for missing model: {model_id}")
                except Exception as e:
                    failed_cleanups.append({"model_id": model_id, "error": str(e)})
                    logger.error(f"Failed to remove registry entry for {model_id}: {e}")
        
        # Save updated registry
        if cleaned_models:
            try:
                self._save_model_registry(registry)
            except Exception as e:
                logger.error(f"Failed to save registry after cleanup: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save registry after cleanup: {str(e)}"
                )
        
        return {
            "cleaned_models": cleaned_models,
            "failed_cleanups": failed_cleanups,
            "total_cleaned": len(cleaned_models)
        }
    
    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison of all models in the registry."""
        registry = self._load_model_registry()
        
        if not registry:
            return {"message": "No models found in registry"}
        
        classification_models = []
        regression_models = []
        
        for model_id, model_info in registry.items():
            model_type = model_info.get('model_type')
            metrics = model_info.get('metrics', {})
            
            model_summary = {
                "model_id": model_id,
                "algorithm": model_info.get('algorithm', 'unknown'),
                "timestamp": model_info.get('timestamp'),
                "train_size": model_info.get('train_size'),
                "test_size": model_info.get('test_size'),
                "training_time": model_info.get('total_training_time'),
                "best_score": model_info.get('best_score')
            }
            
            if model_type == ProblemType.CLASSIFICATION:
                model_summary.update({
                    "accuracy": metrics.get('accuracy'),
                    "precision": metrics.get('precision'),
                    "recall": metrics.get('recall'),
                    "f1_score": metrics.get('f1_score')
                })
                classification_models.append(model_summary)
            elif model_type == ProblemType.REGRESSION:
                model_summary.update({
                    "mse": metrics.get('mse'),
                    "rmse": metrics.get('rmse'),
                    "r2_score": metrics.get('r2_score'),
                    "mae": metrics.get('mae')
                })
                regression_models.append(model_summary)
        
        # Sort models by performance
        if classification_models:
            classification_models.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        if regression_models:
            regression_models.sort(key=lambda x: x.get('r2_score', -float('inf')), reverse=True)
        
        return {
            "classification_models": classification_models,
            "regression_models": regression_models,
            "summary": {
                "total_classification": len(classification_models),
                "total_regression": len(regression_models),
                "best_classification": classification_models[0] if classification_models else None,
                "best_regression": regression_models[0] if regression_models else None
            }
        }
    
    def export_model_config(self, model_id: str) -> Dict[str, Any]:
        """Export model configuration for reproducibility."""
        model_info = self.get_model_info(model_id)
        
        config = {
            "model_id": model_id,
            "model_type": model_info.get('model_type'),
            "algorithm": model_info.get('algorithm'),
            "feature_columns": model_info.get('feature_columns'),
            "target_column": model_info.get('target_column'),
            "hyperparameters": model_info.get('hyperparameters'),
            "training_config": {
                "train_size": model_info.get('train_size'),
                "test_size": model_info.get('test_size'),
                "training_time": model_info.get('total_training_time')
            },
            "performance": model_info.get('metrics'),
            "data_info": model_info.get('data_info'),
            "timestamp": model_info.get('timestamp')
        }
        
        return config
    
    def validate_model_health(self, model_id: str) -> Dict[str, Any]:
        """Perform health check on a specific model."""
        try:
            model_info = self.get_model_info(model_id)
            model_path = Path(model_info['model_path'])
            
            health_status = {
                "model_id": model_id,
                "healthy": True,
                "issues": [],
                "warnings": []
            }
            
            # Check if model file exists
            if not model_path.exists():
                health_status["healthy"] = False
                health_status["issues"].append("Model file not found")
                return health_status
            
            # Try to load the model
            try:
                model = self.load_model(str(model_path))
            except Exception as e:
                health_status["healthy"] = False
                health_status["issues"].append(f"Failed to load model: {str(e)}")
                return health_status
            
            # Check model attributes
            required_attrs = ['predict']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    health_status["healthy"] = False
                    health_status["issues"].append(f"Model missing required method: {attr}")
            
            # Check if model is fitted
            try:
                # Try to get feature importance or other fitted attributes
                if hasattr(model, 'feature_importances_'):
                    if model.feature_importances_ is None:
                        health_status["warnings"].append("Model may not be properly fitted")
            except NotFittedError:
                health_status["healthy"] = False
                health_status["issues"].append("Model is not fitted")
            
            # Check registry consistency
            expected_features = model_info.get('feature_columns', [])
            if hasattr(model, 'n_features_in_'):
                actual_features = model.n_features_in_
                if len(expected_features) != actual_features:
                    health_status["warnings"].append(
                        f"Feature count mismatch: registry={len(expected_features)}, model={actual_features}"
                    )
            
            # Check file size
            file_size = model_path.stat().st_size
            if file_size < 1000:  # Less than 1KB seems suspicious
                health_status["warnings"].append(f"Model file unusually small: {file_size} bytes")
            elif file_size > 100_000_000:  # Larger than 100MB
                health_status["warnings"].append(f"Model file very large: {file_size/1_000_000:.1f} MB")
            
            health_status["file_size"] = file_size
            health_status["last_modified"] = model_path.stat().st_mtime
            
            return health_status
            
        except Exception as e:
            return {
                "model_id": model_id,
                "healthy": False,
                "issues": [f"Health check failed: {str(e)}"],
                "warnings": []
            }


# Create service instance
modeling_service = ModelingService()

# Convenience functions for backward compatibility and ease of use
def train_model(df: pd.DataFrame, config: ModelConfig) -> TrainingResponse:
    """Train a machine learning model based on configuration."""
    return modeling_service.train_model(df, config)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_type: str) -> EvaluationMetrics:
    """Evaluate model performance and return metrics."""
    return modeling_service.evaluate_model(y_true, y_pred, model_type)

def load_model(model_path: str):
    """Load a trained model from disk."""
    return modeling_service.load_model(model_path)

def predict(model_path: str, input_data: pd.DataFrame, validate_input: bool = True) -> PredictionResponse:
    """Make predictions using a trained model."""
    return modeling_service.predict(model_path, input_data, validate_input)

def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    return modeling_service.get_model_info(model_id)

def list_models(include_stats: bool = True) -> Dict[str, Any]:
    """List all models in the registry."""
    return modeling_service.list_models(include_stats)

def delete_model(model_id: str, force: bool = False) -> bool:
    """Delete a model from the registry."""
    return modeling_service.delete_model(model_id, force)

def cleanup_broken_models() -> Dict[str, Any]:
    """Clean up models with missing files."""
    return modeling_service.cleanup_broken_models()

def get_model_performance_comparison() -> Dict[str, Any]:
    """Get performance comparison of all models."""
    return modeling_service.get_model_performance_comparison()

def export_model_config(model_id: str) -> Dict[str, Any]:
    """Export model configuration for reproducibility."""
    return modeling_service.export_model_config(model_id)

def validate_model_health(model_id: str) -> Dict[str, Any]:
    """Perform health check on a specific model."""
    return modeling_service.validate_model_health(model_id)