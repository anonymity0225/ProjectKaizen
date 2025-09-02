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
    mean_squared_error, r2_score, confusion_matrix, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
import joblib

from fastapi import HTTPException

from app.schemas.modeling import (
    ModelConfig, TrainingResponse, EvaluationMetrics, 
    PredictionResponse, ModelInfo, ModelListResponse,
    ModelConfigExport, CleanupResult, ModelHealthStatus,
    ModelAlgorithm, ProblemType, ModelSummary
)
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class ModelingService:
    """Production-grade machine learning service for training, evaluation, and prediction."""
    
    def __init__(self):
        self.model_registry_path = Path(settings.ml.model_path) / "model_registry.json"
        self._ensure_directories()
        
        # Define available algorithms with full names
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
        
        # Algorithm mapping for aliases
        self.algorithm_mapping = {
            # Classification aliases
            "random_forest": "random_forest_classifier",
            "logistic_regression": "logistic_regression",
            "gradient_boosting": "gradient_boosting_classifier",
            "svc": "svc",
            "knn": "knn_classifier",
            "decision_tree": "decision_tree_classifier",
            
            # Regression aliases
            "linear_regression": "linear_regression",
            "ridge": "ridge_regression",
            "lasso": "lasso_regression",
        }
        
        # Default hyperparameters for each algorithm
        self.default_hyperparameters = {
            # Classification
            ModelAlgorithm.RANDOM_FOREST_CLASSIFIER: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.LOGISTIC_REGRESSION: {"random_state": 42, "max_iter": 1000},
            ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER: {"random_state": 42},
            ModelAlgorithm.SVC: {"random_state": 42, "probability": True},
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
                    content = f.read().strip()
                    if not content:
                        logger.debug("Model registry file is empty, returning empty dict")
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load model registry: {e}, returning empty dict")
                return {}
        else:
            logger.debug("Model registry file does not exist, returning empty dict")
            return {}
    
    def _save_model_registry(self, registry: Dict[str, Any]) -> None:
        """Save the model registry to disk."""
        try:
            # Ensure parent directory exists
            self.model_registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
            logger.debug(f"Model registry saved to {self.model_registry_path}")
        except IOError as e:
            logger.error(f"Failed to save model registry: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save model registry: {str(e)}"
            )
    
    def _resolve_algorithm(self, algorithm: str, problem_type: ProblemType) -> ModelAlgorithm:
        """Resolve algorithm alias to full algorithm name based on problem type."""
        # First check if it's already a full algorithm name
        try:
            return ModelAlgorithm(algorithm)
        except ValueError:
            pass
        
        # Check if it's an alias
        if algorithm in self.algorithm_mapping:
            base_name = self.algorithm_mapping[algorithm]
            
            # Map to specific problem type
            if problem_type == ProblemType.CLASSIFICATION:
                if base_name == "random_forest_classifier":
                    return ModelAlgorithm.RANDOM_FOREST_CLASSIFIER
                elif base_name == "logistic_regression":
                    return ModelAlgorithm.LOGISTIC_REGRESSION
                elif base_name == "gradient_boosting_classifier":
                    return ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER
                elif base_name == "svc":
                    return ModelAlgorithm.SVC
                elif base_name == "knn_classifier":
                    return ModelAlgorithm.KNN_CLASSIFIER
                elif base_name == "decision_tree_classifier":
                    return ModelAlgorithm.DECISION_TREE_CLASSIFIER
            elif problem_type == ProblemType.REGRESSION:
                if algorithm == "random_forest":
                    return ModelAlgorithm.RANDOM_FOREST_REGRESSOR
                elif algorithm == "linear_regression":
                    return ModelAlgorithm.LINEAR_REGRESSION
                elif algorithm == "ridge":
                    return ModelAlgorithm.RIDGE_REGRESSION
                elif algorithm == "lasso":
                    return ModelAlgorithm.LASSO_REGRESSION
                elif algorithm == "gradient_boosting":
                    return ModelAlgorithm.GRADIENT_BOOSTING_REGRESSOR
                elif algorithm == "knn":
                    return ModelAlgorithm.KNN_REGRESSOR
                elif algorithm == "decision_tree":
                    return ModelAlgorithm.DECISION_TREE_REGRESSOR
        
        raise HTTPException(
            status_code=422,
            detail=f"Unknown algorithm: {algorithm} for problem type: {problem_type}"
        )
    
    def _validate_dataframe(self, df: pd.DataFrame, config: ModelConfig) -> None:
        """Validate input dataframe and configuration."""
        if df is None or df.empty:
            raise HTTPException(
                status_code=422,
                detail="DataFrame is empty or None"
            )
        
        if len(df) < 10:
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
                    stratify=y if config.problem_type == ProblemType.CLASSIFICATION and len(np.unique(y)) > 1 else None
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
                                       specific_algorithm: Optional[ModelAlgorithm] = None) -> Dict[ModelAlgorithm, Any]:
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
                           f"Available algorithms: {[a.value for a in available]}"
                )
            return {specific_algorithm: algorithms[specific_algorithm]}
        
        return algorithms
    
    def _initialize_model(self, algorithm: ModelAlgorithm, hyperparameters: Optional[Dict] = None):
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
    
    def train_model(self, config: ModelConfig) -> TrainingResponse:
        """Train machine learning models with auto-selection of best performing algorithm."""
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        logger.info(f"Starting model training - ID: {model_id}")
        logger.info(f"Config: Features={config.feature_columns}, Target={config.target_column}")
        
        try:
            # Load data from file
            if not Path(config.data_file_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Data file not found: {config.data_file_path}"
                )
            
            try:
                df = pd.read_csv(config.data_file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to load data file: {str(e)}"
                )
            
            # Auto-detect problem type if not specified
            if not config.problem_type:
                config.problem_type = self._detect_problem_type(df, config.target_column)
                logger.info(f"Auto-detected problem type: {config.problem_type}")
            
            # Validation
            self._validate_dataframe(df, config)
            
            # Prepare data
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(df, config)
            
            logger.info(f"Data prepared - Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Resolve algorithm if specified
            specific_algorithm = None
            if config.algorithm:
                specific_algorithm = self._resolve_algorithm(config.algorithm, config.problem_type)
            
            # Get algorithms to try
            algorithms = self._get_algorithms_for_problem_type(config.problem_type, specific_algorithm)
            
            best_model = None
            best_algorithm = None
            best_score = -np.inf
            best_metrics = None
            all_results = {}
            
            # Train and evaluate each algorithm
            for algorithm_enum, algorithm_class in algorithms.items():
                try:
                    logger.info(f"Training {algorithm_enum.value}...")
                    model_start_time = time.time()
                    
                    # Initialize model
                    model = self._initialize_model(algorithm_enum, config.hyperparameters)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluate model
                    metrics = self._evaluate_model(y_test, y_pred, config.problem_type)
                    
                    # Calculate score for model selection
                    if config.problem_type == ProblemType.CLASSIFICATION:
                        score = metrics.accuracy
                    else:
                        score = metrics.r2_score if metrics.r2_score is not None else 0
                    
                    model_time = time.time() - model_start_time
                    
                    all_results[algorithm_enum.value] = {
                        "score": score,
                        "metrics": metrics,
                        "training_time": model_time
                    }
                    
                    # Update best model if this one is better
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_algorithm = algorithm_enum
                        best_metrics = metrics
                    
                    logger.info(f"Completed {algorithm_enum.value} - Score: {score:.4f}, Time: {model_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {algorithm_enum.value}: {str(e)}")
                    continue
            
            if best_model is None:
                raise HTTPException(
                    status_code=500,
                    detail="No model could be trained successfully. Check your data and configuration."
                )
            
            total_training_time = time.time() - start_time
            logger.info(f"Best algorithm: {best_algorithm.value} with score: {best_score:.4f}")
            
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
            
            # Update model registry
            registry = self._load_model_registry()
            registry[model_id] = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "problem_type": config.problem_type.value,
                "algorithm": best_algorithm.value,
                "feature_columns": config.feature_columns,
                "target_column": config.target_column,
                "model_path": str(model_path),
                "hyperparameters": config.hyperparameters or {},
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
                problem_type=config.problem_type,
                algorithm=best_algorithm.value,
                feature_columns=config.feature_columns,
                model_path=str(model_path),
                training_time=total_training_time,
                train_size=len(X_train),
                test_size=len(X_test)
            )
            
            # Set metrics based on problem type
            if config.problem_type == ProblemType.CLASSIFICATION:
                response.accuracy = best_metrics.accuracy
                response.precision = best_metrics.precision
                response.recall = best_metrics.recall
                response.f1_score = best_metrics.f1_score
            else:
                response.mse = best_metrics.mse
                response.r2_score = best_metrics.r2_score
            
            logger.info(f"Model training completed successfully - ID: {model_id}, Algorithm: {best_algorithm.value}")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during training: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during model training: {str(e)}"
            )
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, problem_type: ProblemType) -> EvaluationMetrics:
        """Evaluate model performance and return comprehensive metrics."""
        try:
            metrics = EvaluationMetrics()
            
            if problem_type == ProblemType.CLASSIFICATION:
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
                    
            elif problem_type == ProblemType.REGRESSION:
                metrics.mse = float(mean_squared_error(y_true, y_pred))
                metrics.r2_score = float(r2_score(y_true, y_pred))
                metrics.rmse = float(np.sqrt(metrics.mse))
                
                # Add mean absolute error
                try:
                    metrics.mae = float(mean_absolute_error(y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Failed to compute MAE: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model evaluation failed: {str(e)}"
            )
    
    def predict(self, model_id: str, data: pd.DataFrame) -> PredictionResponse:
        """Make predictions using a trained model."""
        try:
            # Get model info from registry
            registry = self._load_model_registry()
            if model_id not in registry:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model with ID {model_id} not found in registry"
                )
            
            model_info = registry[model_id]
            model_path = model_info['model_path']
            
            # Load model
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {model_path}"
                )
            
            try:
                model = joblib.load(model_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model: {str(e)}"
                )
            
            # Validate input data
            if data is None or data.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Input data is empty or None"
                )
            
            feature_columns = model_info.get('feature_columns', [])
            problem_type = model_info.get('problem_type', 'unknown')
            algorithm = model_info.get('algorithm', 'unknown')
            
            # Check if required features are present
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_features}. "
                           f"Expected features: {feature_columns}"
                )
            
            # Select only the required features in the correct order
            X = data[feature_columns].copy()
            
            # Handle missing values
            if X.isnull().any().any():
                null_columns = X.columns[X.isnull().any()].tolist()
                null_count = X.isnull().sum().sum()
                raise HTTPException(
                    status_code=400,
                    detail=f"Input data contains {null_count} missing values in columns: {null_columns}. "
                           f"Please handle missing values before prediction."
                )
            
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
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction error: {str(e)}"
                )
            
            # Get confidence scores if available (for classification)
            confidence_scores = None
            if hasattr(model, 'predict_proba') and problem_type == ProblemType.CLASSIFICATION.value:
                try:
                    proba = model.predict_proba(X)
                    confidence_scores = np.max(proba, axis=1).tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute confidence scores: {e}")
            
            response = PredictionResponse(
                model_id=model_id,
                predictions=predictions.tolist(),
                confidence_scores=confidence_scores,
                num_predictions=len(predictions)
            )
            
            logger.info(f"Predictions generated successfully - {len(predictions)} samples using {algorithm}")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model from the registry."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found in registry"
            )
        
        model_data = registry[model_id]
        
        # Convert to ModelInfo with safe defaults
        try:
            model_info = ModelInfo(
                model_id=model_data.get('model_id', model_id),
                timestamp=model_data.get('timestamp', datetime.now().isoformat()),
                problem_type=model_data.get('problem_type', 'unknown'),
                algorithm=model_data.get('algorithm', 'unknown'),
                feature_columns=model_data.get('feature_columns', []),
                target_column=model_data.get('target_column', ''),
                model_path=model_data.get('model_path', ''),
                hyperparameters=model_data.get('hyperparameters', {}),
                train_size=model_data.get('train_size', 0),
                test_size=model_data.get('test_size', 0),
                training_time=model_data.get('total_training_time', 0.0),
                metrics=model_data.get('metrics', {}),
                best_score=model_data.get('best_score', 0.0)
            )
            return model_info
        except Exception as e:
            logger.error(f"Failed to convert model data to ModelInfo: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve model info: {str(e)}"
            )
    
    def list_models(self, include_stats: bool = True) -> ModelListResponse:
        """List all models in the registry."""
        registry = self._load_model_registry()
        
        # Convert registry entries to ModelInfo objects
        models = []
        for model_id, model_data in registry.items():
            try:
                model_info = ModelInfo(
                    model_id=model_data.get('model_id', model_id),
                    timestamp=model_data.get('timestamp', datetime.now().isoformat()),
                    problem_type=model_data.get('problem_type', 'unknown'),
                    algorithm=model_data.get('algorithm', 'unknown'),
                    feature_columns=model_data.get('feature_columns', []),
                    target_column=model_data.get('target_column', ''),
                    model_path=model_data.get('model_path', ''),
                    hyperparameters=model_data.get('hyperparameters', {}),
                    train_size=model_data.get('train_size', 0),
                    test_size=model_data.get('test_size', 0),
                    training_time=model_data.get('total_training_time', 0.0),
                    metrics=model_data.get('metrics', {}),
                    best_score=model_data.get('best_score', 0.0)
                )
                models.append(model_info)
            except Exception as e:
                logger.warning(f"Failed to parse model {model_id}: {e}")
                continue
        
        # Create summary if requested
        summary = None
        if include_stats:
            total_models = len(models)
            classification_models = sum(1 for model in models 
                                      if model.problem_type == ProblemType.CLASSIFICATION.value)
            regression_models = sum(1 for model in models 
                                  if model.problem_type == ProblemType.REGRESSION.value)
            
            # Get algorithm distribution
            algorithm_counts = {}
            for model in models:
                algorithm = model.algorithm
                algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
            
            # Check for broken models (missing files)
            broken_models = []
            for model in models:
                model_path = Path(model.model_path)
                if not model_path.exists():
                    broken_models.append(model.model_id)
            
            summary = ModelSummary(
                total_models=total_models,
                classification_models=classification_models,
                regression_models=regression_models,
                algorithm_distribution=algorithm_counts,
                broken_models=broken_models
            )
        
        return ModelListResponse(
            models=models,
            summary=summary
        )
    
    def delete_model(self, model_id: str) -> dict:
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
            # Remove model file if it exists
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Model file deleted: {model_path}")
            else:
                logger.warning(f"Model file not found during deletion: {model_path}")
            
            # Remove from registry
            del registry[model_id]
            self._save_model_registry(registry)
            
            logger.info(f"Model {model_id} deleted successfully")
            return {"message": f"Model {model_id} deleted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model: {str(e)}"
            )
    
    def get_model_performance_comparison(self) -> dict:
        """Get performance comparison of all models in the registry."""
        registry = self._load_model_registry()
        
        if not registry:
            return {"message": "No models found in registry"}
        
        classification_models = []
        regression_models = []
        
        for model_id, model_info in registry.items():
            problem_type = model_info.get('problem_type')
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
            
            if problem_type == ProblemType.CLASSIFICATION.value:
                model_summary.update({
                    "accuracy": metrics.get('accuracy'),
                    "precision": metrics.get('precision'),
                    "recall": metrics.get('recall'),
                    "f1_score": metrics.get('f1_score')
                })
                classification_models.append(model_summary)
            elif problem_type == ProblemType.REGRESSION.value:
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
    
    def export_model_config(self, model_id: str) -> ModelConfigExport:
        """Export model configuration for reproducibility."""
        model_info = self.get_model_info(model_id)
        
        config = ModelConfigExport(
            model_id=model_id,
            problem_type=model_info.problem_type,
            algorithm=model_info.algorithm,
            feature_columns=model_info.feature_columns,
            target_column=model_info.target_column,
            hyperparameters=model_info.hyperparameters,
            training_config={
                "train_size": model_info.train_size,
                "test_size": model_info.test_size,
                "training_time": model_info.training_time
            },
            performance=model_info.metrics,
            timestamp=model_info.timestamp
        )
        
        return config
    
    def cleanup_broken_models(self) -> CleanupResult:
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
        
        return CleanupResult(
            cleaned_models=cleaned_models,
            failed_cleanups=failed_cleanups,
            total_cleaned=len(cleaned_models)
        )
    
    def validate_model_health(self, model_id: str) -> ModelHealthStatus:
        """Perform health check on a specific model."""
        try:
            model_info = self.get_model_info(model_id)
            model_path = Path(model_info.model_path)
            
            issues = []
            warnings = []
            
            # Check if model file exists
            if not model_path.exists():
                issues.append("Model file not found")
                return ModelHealthStatus(
                    model_id=model_id,
                    healthy=False,
                    issues=issues,
                    warnings=warnings
                )
            
            # Try to load the model
            try:
                model = joblib.load(model_path)
            except Exception as e:
                issues.append(f"Failed to load model: {str(e)}")
                return ModelHealthStatus(
                    model_id=model_id,
                    healthy=False,
                    issues=issues,
                    warnings=warnings
                )
            
            # Check model attributes
            required_attrs = ['predict']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    issues.append(f"Model missing required method: {attr}")
            
            # Check if model is fitted
            try:
                # Try to get feature importance or other fitted attributes
                if hasattr(model, 'feature_importances_'):
                    if model.feature_importances_ is None:
                        warnings.append("Model may not be properly fitted")
            except NotFittedError:
                issues.append("Model is not fitted")
            
            # Check registry consistency
            expected_features = model_info.feature_columns
            if hasattr(model, 'n_features_in_'):
                actual_features = model.n_features_in_
                if len(expected_features) != actual_features:
                    warnings.append(
                        f"Feature count mismatch: registry={len(expected_features)}, model={actual_features}"
                    )
            
            # Check file size
            file_size = model_path.stat().st_size
            if file_size < 1000:  # Less than 1KB seems suspicious
                warnings.append(f"Model file unusually small: {file_size} bytes")
            elif file_size > 100_000_000:  # Larger than 100MB
                warnings.append(f"Model file very large: {file_size/1_000_000:.1f} MB")
            
            healthy = len(issues) == 0
            
            return ModelHealthStatus(
                model_id=model_id,
                healthy=healthy,
                issues=issues,
                warnings=warnings,
                file_size=file_size,
                last_modified=model_path.stat().st_mtime
            )
            
        except Exception as e:
            return ModelHealthStatus(
                model_id=model_id,
                healthy=False,
                issues=[f"Health check failed: {str(e)}"],
                warnings=[]
            )


# Create service instance
modeling_service = ModelingService()

# Public interface functions
def train_model(config: ModelConfig) -> TrainingResponse:
    """Train a machine learning model based on configuration."""
    return modeling_service.train_model(config)

def predict(model_id: str, data: pd.DataFrame) -> PredictionResponse:
    """Make predictions using a trained model."""
    return modeling_service.predict(model_id, data)

def list_models(include_stats: bool = True) -> ModelListResponse:
    """List all models in the registry."""
    return modeling_service.list_models(include_stats)

def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    return modeling_service.get_model_info(model_id)

def delete_model(model_id: str) -> dict:
    """Delete a model from the registry."""
    return modeling_service.delete_model(model_id)

def get_model_performance_comparison() -> dict:
    """Get performance comparison of all models."""
    return modeling_service.get_model_performance_comparison()

def export_model_config(model_id: str) -> ModelConfigExport:
    """Export model configuration for reproducibility."""
    return modeling_service.export_model_config(model_id)

def cleanup_broken_models() -> CleanupResult:
    """Clean up models with missing files."""
    return modeling_service.cleanup_broken_models()

def validate_model_health(model_id: str) -> ModelHealthStatus:
    """Perform health check on a specific model."""
    return modeling_service.validate_model_health(model_id)