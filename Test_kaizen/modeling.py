import pandas as pd
import numpy as np
import json
import uuid
import logging
import hashlib
import sys
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Generator, Callable
import time
import os
import warnings
import gc
import psutil
import signal
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from contextlib import contextmanager
from functools import wraps
from threading import RLock, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, mean_absolute_error,
    roc_auc_score, precision_recall_curve, auc, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Model explanability features will be limited.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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
from app.observability.metrics import (
    modeling_training_seconds,
    service_errors_total
)
from app.observability.tracing import traced_span

# ==========================================
# ENTERPRISE INFRASTRUCTURE
# ==========================================

# Configuration from environment
CONFIG = {
    'DATA_LOADING_MAX_MEMORY_GB': int(os.getenv('ML_DATA_LOADING_MAX_MEMORY_GB', '2')),
    'FEATURE_ENG_MAX_MEMORY_GB': int(os.getenv('ML_FEATURE_ENG_MAX_MEMORY_GB', '1')),
    'MODEL_TRAINING_MAX_MEMORY_GB': int(os.getenv('ML_MODEL_TRAINING_MAX_MEMORY_GB', '4')),
    'MEMORY_WARNING_THRESHOLD': float(os.getenv('ML_MEMORY_WARNING_THRESHOLD', '0.8')),
    'DEFAULT_CHUNK_SIZE': int(os.getenv('ML_DEFAULT_CHUNK_SIZE', '10000')),
    'MAX_TUNING_TIME_MINUTES': int(os.getenv('ML_MAX_TUNING_TIME_MINUTES', '30')),
    'MAX_TRIALS_PER_ALGORITHM': int(os.getenv('ML_MAX_TRIALS_PER_ALGORITHM', '50')),
    'TRIAL_TIMEOUT_MINUTES': int(os.getenv('ML_TRIAL_TIMEOUT_MINUTES', '5')),
    'EARLY_STOPPING_PATIENCE': int(os.getenv('ML_EARLY_STOPPING_PATIENCE', '10')),
}

class TrainingPhase(Enum):
    """Training pipeline phases for monitoring."""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_SAVING = "model_saving"
    COMPLETED = "completed"
    FAILED = "failed"

class SecurityLevel(Enum):
    """Security levels for model artifacts."""
    BASIC = "basic"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    ENCRYPTED_SIGNED = "encrypted_signed"

class DataBackend(Enum):
    """Data processing backends."""
    PANDAS = "pandas"
    DASK = "dask"
    PYARROW = "pyarrow"

class MemoryState(Enum):
    """Memory usage states."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

# ==========================================
# ENTERPRISE EXCEPTIONS
# ==========================================

class MLServiceError(Exception):
    """Base exception for ML service errors."""
    pass

class DataValidationError(MLServiceError):
    """Error during data validation."""
    pass

class TrainingError(MLServiceError):
    """Error during model training."""
    pass

class HyperparameterError(MLServiceError):
    """Error during hyperparameter optimization."""
    pass

class ResourceExhaustionError(MLServiceError):
    """Error due to resource exhaustion."""
    pass

class SecurityError(MLServiceError):
    """Security-related error."""
    pass

class CheckpointError(MLServiceError):
    """Error with training checkpoints."""
    pass

class DistributedTrainingError(MLServiceError):
    """Error in distributed training."""
    pass

# ==========================================
# ENTERPRISE DATA CLASSES
# ==========================================

@dataclass
class MemoryUsage:
    """Memory usage information."""
    process_memory_mb: float
    system_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    phase: TrainingPhase
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class TrainingCheckpoint:
    """Training checkpoint data."""
    checkpoint_id: str
    phase: TrainingPhase
    algorithm: str
    hyperparameters: Dict[str, Any]
    best_score: Optional[float]
    model_state: Optional[bytes]
    optimizer_state: Optional[bytes]
    training_history: List[Dict[str, Any]]
    created_at: datetime

@dataclass
class ModelQualityGate:
    """Model quality gate configuration."""
    name: str
    metric: str
    threshold: float
    comparison: str  # "greater_than", "less_than", "equal", "greater_equal", "less_equal"
    required: bool = True

class ScoringMetric(str, Enum):
    """Available scoring metrics for model selection."""
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision_weighted"
    RECALL = "recall_weighted" 
    F1 = "f1_weighted"
    ROC_AUC = "roc_auc_ovr_weighted"
    
    # Regression
    R2 = "r2"
    NEG_MSE = "neg_mean_squared_error"
    NEG_MAE = "neg_mean_absolute_error"
    NEG_RMSE = "neg_root_mean_squared_error"

class SamplingStrategy(str, Enum):
    """Data sampling strategies for imbalanced datasets."""
    NONE = "none"
    SMOTE = "smote"
    UNDERSAMPLE = "undersample"
    CLASS_WEIGHT = "class_weight"

@dataclass
class ExperimentMetadata:
    """Metadata for experiment reproducibility."""
    python_version: str
    sklearn_version: str
    pandas_version: str
    numpy_version: str
    platform: str
    timestamp: str
    git_commit: Optional[str] = None
    
    @classmethod
    def create_current(cls) -> 'ExperimentMetadata':
        """Create metadata for current environment."""
        import sklearn
        
        return cls(
            python_version=sys.version,
            sklearn_version=sklearn.__version__,
            pandas_version=pd.__version__,
            numpy_version=np.__version__,
            platform=platform.platform(),
            timestamp=datetime.now().isoformat(),
            git_commit=cls._get_git_commit()
        )
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

@dataclass
class CrossValidationResults:
    """Results from cross-validation."""
    cv_scores: List[float]
    mean_score: float
    std_score: float
    best_params: Optional[Dict] = None

@dataclass 
class ExplanationResults:
    """Model explanation results."""
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[List[str]] = None

class EnhancedModelingService:
    """Enterprise-grade machine learning service with advanced features."""
    
    def __init__(self):
        self.model_registry_path = Path(settings.ml.model_path) / "model_registry.json"
        self.explanations_path = Path(settings.ml.model_path) / "explanations"
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
        
        # Enhanced hyperparameters with tuning ranges
        self.default_hyperparameters = {
            # Classification
            ModelAlgorithm.RANDOM_FOREST_CLASSIFIER: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.LOGISTIC_REGRESSION: {"random_state": 42, "max_iter": 1000},
            ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.SVC: {"random_state": 42, "probability": True},
            ModelAlgorithm.KNN_CLASSIFIER: {"n_neighbors": 5},
            ModelAlgorithm.DECISION_TREE_CLASSIFIER: {"random_state": 42},
            
            # Regression
            ModelAlgorithm.RANDOM_FOREST_REGRESSOR: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.LINEAR_REGRESSION: {},
            ModelAlgorithm.RIDGE_REGRESSION: {"random_state": 42},
            ModelAlgorithm.LASSO_REGRESSION: {"random_state": 42},
            ModelAlgorithm.GRADIENT_BOOSTING_REGRESSOR: {"random_state": 42, "n_estimators": 100},
            ModelAlgorithm.SVR: {},
            ModelAlgorithm.KNN_REGRESSOR: {"n_neighbors": 5},
            ModelAlgorithm.DECISION_TREE_REGRESSOR: {"random_state": 42},
        }
        
        # Hyperparameter tuning grids
        self.tuning_grids = {
            ModelAlgorithm.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelAlgorithm.GRADIENT_BOOSTING_CLASSIFIER: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelAlgorithm.LOGISTIC_REGRESSION: {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            ModelAlgorithm.SVC: {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            # Add regression tuning grids
            ModelAlgorithm.RANDOM_FOREST_REGRESSOR: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            ModelAlgorithm.GRADIENT_BOOSTING_REGRESSOR: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        # Default scoring metrics
        self.default_scoring = {
            ProblemType.CLASSIFICATION: ScoringMetric.F1,
            ProblemType.REGRESSION: ScoringMetric.R2
        }
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist with proper permissions."""
        directories = [
            Path(settings.ml.model_path),
            self.explanations_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions (owner read/write/execute only)
            try:
                os.chmod(directory, 0o700)
            except (OSError, AttributeError):
                # Skip chmod on Windows or if not supported
                pass
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataset for reproducibility tracking."""
        # Create a deterministic hash of the dataframe
        df_string = df.to_string(index=False).encode('utf-8')
        return hashlib.sha256(df_string).hexdigest()[:16]
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the model registry from disk with enhanced error handling."""
        if self.model_registry_path.exists():
            try:
                with open(self.model_registry_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.debug("Model registry file is empty, returning empty dict")
                        return {}
                    registry = json.loads(content)
                    
                    # Validate registry structure
                    if not isinstance(registry, dict):
                        logger.warning("Invalid registry format, creating new registry")
                        return {}
                    
                    return registry
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load model registry: {e}, returning empty dict")
                return {}
        else:
            logger.debug("Model registry file does not exist, returning empty dict")
            return {}
    
    def _save_model_registry(self, registry: Dict[str, Any]) -> None:
        """Save the model registry to disk with atomic write and backup."""
        try:
            # Ensure parent directory exists
            self.model_registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if registry exists
            if self.model_registry_path.exists():
                backup_path = self.model_registry_path.with_suffix('.json.backup')
                self.model_registry_path.rename(backup_path)
            
            # Write to temporary file first (atomic write)
            temp_path = self.model_registry_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
            
            # Move temp file to final location
            temp_path.rename(self.model_registry_path)
            
            # Set restrictive permissions
            try:
                os.chmod(self.model_registry_path, 0o600)
            except (OSError, AttributeError):
                pass
                
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
        """Enhanced dataframe validation with schema checking."""
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
        
        # Enhanced target validation
        target_series = clean_df[config.target_column]
        unique_targets = len(target_series.unique())
        
        if unique_targets == 1:
            raise HTTPException(
                status_code=422,
                detail="Target column has only one unique value. Cannot train a meaningful model."
            )
        
        # Check for class imbalance in classification
        if config.problem_type == ProblemType.CLASSIFICATION:
            class_counts = target_series.value_counts()
            min_class_size = class_counts.min()
            max_class_size = class_counts.max()
            imbalance_ratio = max_class_size / min_class_size
            
            if imbalance_ratio > 10:
                logger.warning(f"Highly imbalanced dataset detected. Ratio: {imbalance_ratio:.1f}:1")
            
            if min_class_size < 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"Some classes have fewer than 2 samples. Cannot perform train/test split."
                )
    
    def _detect_problem_type(self, df: pd.DataFrame, target_column: str) -> ProblemType:
        """Enhanced problem type detection with better heuristics."""
        target_series = df[target_column].dropna()
        
        # If target is clearly non-numeric, it's classification
        if target_series.dtype.kind not in 'biufc':
            return ProblemType.CLASSIFICATION
        
        # For numeric targets, use enhanced heuristics
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # Check if values are discrete integers
        is_integer = target_series.dtype.kind in 'iu' or (target_series % 1 == 0).all()
        
        # Classification heuristics
        classification_indicators = [
            unique_values <= 10,  # Few unique values
            unique_values <= max(2, total_values * 0.05),  # <5% of total samples
            is_integer and unique_values <= 20,  # Integer with reasonable range
            target_series.min() >= 0 and target_series.max() <= 1 and is_integer  # Binary-like
        ]
        
        # If multiple indicators suggest classification
        if sum(classification_indicators) >= 2:
            return ProblemType.CLASSIFICATION
        
        # Otherwise, assume regression
        return ProblemType.REGRESSION
    
    def _apply_sampling_strategy(self, X: pd.DataFrame, y: pd.Series, 
                                strategy: SamplingStrategy, problem_type: ProblemType,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply data sampling strategy for imbalanced datasets."""
        if strategy == SamplingStrategy.NONE or problem_type != ProblemType.CLASSIFICATION:
            return X, y
        
        try:
            if strategy == SamplingStrategy.SMOTE:
                # Check if we have enough samples for SMOTE
                min_samples = min(pd.Series(y).value_counts())
                if min_samples < 2:
                    logger.warning("Not enough samples for SMOTE, skipping sampling")
                    return X, y
                
                smote = SMOTE(random_state=random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            
            elif strategy == SamplingStrategy.UNDERSAMPLE:
                undersampler = RandomUnderSampler(random_state=random_state)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        except Exception as e:
            logger.warning(f"Sampling strategy {strategy} failed: {e}. Using original data.")
        
        return X, y
    
    def _prepare_data(self, df: pd.DataFrame, config: ModelConfig) -> tuple:
        """Enhanced data preparation with sampling and validation."""
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
            
            # Enhanced feature validation
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
            
            # Check for features with zero variance
            zero_var_features = []
            for col in X.columns:
                if X[col].var() == 0:
                    zero_var_features.append(col)
            
            if zero_var_features:
                logger.warning(f"Zero variance features detected: {zero_var_features}")
            
            # Apply sampling strategy before train/test split
            sampling_strategy = getattr(config, 'sampling_strategy', SamplingStrategy.NONE)
            if sampling_strategy != SamplingStrategy.NONE:
                X, y = self._apply_sampling_strategy(X, y, sampling_strategy, config.problem_type)
                logger.info(f"Applied sampling strategy: {sampling_strategy}")
            
            # Enhanced train/test split
            test_size = getattr(config, 'test_size', 0.2)
            random_state = getattr(config, 'random_state', 42)
            
            # For small datasets, ensure at least 2 samples in test set
            if len(X) * test_size < 2:
                test_size = max(2 / len(X), 0.1)
                logger.warning(f"Adjusted test_size to {test_size:.3f} to ensure minimum test samples")
            
            # Stratified split for classification
            try:
                if config.problem_type == ProblemType.CLASSIFICATION and len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
            except ValueError as e:
                if "stratify" in str(e):
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
    
    def _perform_cross_validation(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                problem_type: ProblemType, scoring: str = None,
                                cv_folds: int = 5) -> CrossValidationResults:
        """Perform cross-validation with appropriate scoring."""
        try:
            # Select appropriate cross-validation strategy
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                if scoring is None:
                    scoring = 'f1_weighted'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                if scoring is None:
                    scoring = 'r2'
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            
            return CrossValidationResults(
                cv_scores=cv_scores.tolist(),
                mean_score=float(cv_scores.mean()),
                std_score=float(cv_scores.std())
            )
        
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return CrossValidationResults(
                cv_scores=[],
                mean_score=0.0,
                std_score=0.0
            )
    
    def _tune_hyperparameters(self, model_class, X_train: pd.DataFrame, y_train: pd.Series,
                             algorithm: ModelAlgorithm, problem_type: ProblemType,
                             scoring: str = None, n_trials: int = 50) -> Dict:
        """Enhanced hyperparameter tuning with Optuna or GridSearch."""
        if algorithm not in self.tuning_grids:
            return {}
        
        param_grid = self.tuning_grids[algorithm]
        
        try:
            # Use Optuna if available for more efficient search
            if OPTUNA_AVAILABLE and len(param_grid) > 0:
                return self._optuna_hyperparameter_tuning(
                    model_class, X_train, y_train, param_grid, problem_type, scoring, n_trials
                )
            else:
                # Fallback to GridSearchCV
                return self._grid_search_tuning(
                    model_class, X_train, y_train, param_grid, problem_type, scoring
                )
        
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}")
            return {}
    
    def _optuna_hyperparameter_tuning(self, model_class, X_train: pd.DataFrame, y_train: pd.Series,
                                     param_grid: Dict, problem_type: ProblemType,
                                     scoring: str, n_trials: int) -> Dict:
        """Hyperparameter tuning using Optuna."""
        def objective(trial):
            params = {}
            base_params = self.default_hyperparameters.get(
                ModelAlgorithm(model_class.__name__.lower()), {}
            )
            
            for param, values in param_grid.items():
                if isinstance(values, list):
                    if all(isinstance(v, (int, float)) for v in values):
                        if all(isinstance(v, int) for v in values):
                            params[param] = trial.suggest_int(param, min(values), max(values))
                        else:
                            params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)
            
            # Merge with base parameters
            final_params = {**base_params, **params}
            
            try:
                model = model_class(**final_params)
                
                # Cross-validation
                cv_results = self._perform_cross_validation(
                    model, X_train, y_train, problem_type, scoring
                )
                
                return cv_results.mean_score
            except Exception:
                return -float('inf')
        
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
        
        return study.best_params
    
    def _grid_search_tuning(self, model_class, X_train: pd.DataFrame, y_train: pd.Series,
                           param_grid: Dict, problem_type: ProblemType, scoring: str) -> Dict:
        """Hyperparameter tuning using GridSearchCV."""
        base_params = self.default_hyperparameters.get(
            ModelAlgorithm(model_class.__name__.lower()), {}
        )
        
        model = model_class(**base_params)
        
        # Reduce grid size for faster search
        reduced_grid = {}
        for param, values in param_grid.items():
            if isinstance(values, list) and len(values) > 3:
                # Take first, middle, and last values for large grids
                reduced_grid[param] = [values[0], values[len(values)//2], values[-1]]
            else:
                reduced_grid[param] = values
        
        cv = 3 if problem_type == ProblemType.CLASSIFICATION else 3
        
        grid_search = GridSearchCV(
            model, reduced_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    def _generate_model_explanations(self, model, X_train: pd.DataFrame, 
                                   feature_names: List[str]) -> ExplanationResults:
        """Generate model explanations using feature importance and SHAP."""
        explanations = ExplanationResults()
        
        try:
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                # Sort by importance
                explanations.feature_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
            elif hasattr(model, 'coef_'):
                # For linear models, use coefficient magnitudes
                if len(model.coef_.shape) == 1:
                    coef = model.coef_
                else:
                    coef = np.mean(np.abs(model.coef_), axis=0)
                
                importance_dict = dict(zip(feature_names, np.abs(coef)))
                explanations.feature_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
            
            # SHAP values (if SHAP is available)
            if SHAP_AVAILABLE:
                try:
                    # Sample data for SHAP to avoid memory issues
                    sample_size = min(100, len(X_train))
                    X_sample = X_train.sample(n=sample_size, random_state=42)
                    
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.Explainer(model.predict_proba, X_sample)
                    else:
                        explainer = shap.Explainer(model.predict, X_sample)
                    
                    shap_values = explainer(X_sample)
                    
                    if hasattr(shap_values, 'values'):
                        explanations.shap_values = shap_values.values
                        explanations.shap_feature_names = feature_names
                    
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
        
        except Exception as e:
            logger.warning(f"Model explanation generation failed: {e}")
        
        return explanations
    
    def _save_model_explanations(self, model_id: str, explanations: ExplanationResults) -> None:
        """Save model explanations to disk."""
        try:
            explanation_file = self.explanations_path / f"{model_id}_explanations.joblib"
            joblib.dump(explanations, explanation_file)
            
            # Set restrictive permissions
            try:
                os.chmod(explanation_file, 0o600)
            except (OSError, AttributeError):
                pass
                
        except Exception as e:
            logger.warning(f"Failed to save explanations for model {model_id}: {e}")
    
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
    
    def _initialize_model(self, algorithm: ModelAlgorithm, hyperparameters: Optional[Dict] = None,
                         sampling_strategy: SamplingStrategy = SamplingStrategy.NONE,
                         problem_type: ProblemType = None):
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
            
            # Handle class weight for imbalanced datasets
            if (sampling_strategy == SamplingStrategy.CLASS_WEIGHT and 
                problem_type == ProblemType.CLASSIFICATION and
                hasattr(model_class(), 'class_weight')):
                final_params['class_weight'] = 'balanced'
            
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
        """Enhanced model training with cross-validation, hyperparameter tuning, and explanations."""
        start_time = time.time()
        span_name = "modeling.train_model"
        try:
            _traced_ctx = traced_span(span_name)
            _traced_ctx.__enter__()
        except Exception:
            _traced_ctx = None
        model_id = str(uuid.uuid4())
        
        logger.info(f"Starting enhanced model training - ID: {model_id}")
        logger.info(f"Config: Features={config.feature_columns}, Target={config.target_column}")
        
        try:
            # Load data from file with path validation
            data_path = Path(config.data_file_path)
            if not data_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Data file not found: {config.data_file_path}"
                )
            
            # Validate path to prevent traversal attacks
            if not str(data_path.resolve()).startswith(str(Path.cwd())):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file path"
                )
            
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to load data file: {str(e)}"
                )
            
            # Create experiment metadata
            experiment_metadata = ExperimentMetadata.create_current()
            data_hash = self._compute_data_hash(df)
            
            # Auto-detect problem type if not specified
            if not config.problem_type:
                config.problem_type = self._detect_problem_type(df, config.target_column)
                logger.info(f"Auto-detected problem type: {config.problem_type}")
            
            # Enhanced validation
            self._validate_dataframe(df, config)
            
            # Prepare data with sampling
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(df, config)
            
            logger.info(f"Data prepared - Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Resolve algorithm if specified
            specific_algorithm = None
            if config.algorithm:
                specific_algorithm = self._resolve_algorithm(config.algorithm, config.problem_type)
            
            # Get algorithms to try
            algorithms = self._get_algorithms_for_problem_type(config.problem_type, specific_algorithm)
            
            # Training configuration
            use_cv = getattr(config, 'use_cross_validation', True)
            tune_hyperparams = getattr(config, 'tune_hyperparameters', False)
            scoring_metric = getattr(config, 'scoring_metric', 
                                   self.default_scoring[config.problem_type].value)
            
            best_model = None
            best_algorithm = None
            best_score = -np.inf
            best_metrics = None
            best_cv_results = None
            all_results = {}
            
            # Train and evaluate each algorithm
            for algorithm_enum, algorithm_class in algorithms.items():
                try:
                    logger.info(f"Training {algorithm_enum.value}...")
                    model_start_time = time.time()
                    
                    # Hyperparameter tuning
                    best_params = {}
                    if tune_hyperparams:
                        logger.info("Performing hyperparameter tuning...")
                        best_params = self._tune_hyperparameters(
                            algorithm_class, X_train, y_train, algorithm_enum,
                            config.problem_type, scoring_metric
                        )
                        logger.info(f"Best hyperparameters: {best_params}")
                    
                    # Merge tuned parameters with config hyperparameters
                    final_hyperparams = {**best_params}
                    if config.hyperparameters:
                        final_hyperparams.update(config.hyperparameters)
                    
                    # Initialize model
                    sampling_strategy = getattr(config, 'sampling_strategy', SamplingStrategy.NONE)
                    model = self._initialize_model(
                        algorithm_enum, final_hyperparams, sampling_strategy, config.problem_type
                    )
                    
                    # Cross-validation
                    cv_results = None
                    if use_cv:
                        cv_results = self._perform_cross_validation(
                            model, X_train, y_train, config.problem_type, scoring_metric
                        )
                        logger.info(f"CV Score: {cv_results.mean_score:.4f} (+/- {cv_results.std_score:.4f})")
                    
                    # Train final model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluate model
                    metrics = self._evaluate_model(y_test, y_pred, config.problem_type, model, X_test)
                    
                    # Calculate score for model selection (use CV score if available)
                    if cv_results and cv_results.mean_score > 0:
                        score = cv_results.mean_score
                    elif config.problem_type == ProblemType.CLASSIFICATION:
                        score = metrics.accuracy
                    else:
                        score = metrics.r2_score if metrics.r2_score is not None else 0
                    
                    model_time = time.time() - model_start_time
                    
                    all_results[algorithm_enum.value] = {
                        "score": score,
                        "metrics": metrics,
                        "training_time": model_time,
                        "cv_results": cv_results,
                        "hyperparameters": final_hyperparams
                    }
                    
                    # Update best model if this one is better
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_algorithm = algorithm_enum
                        best_metrics = metrics
                        best_cv_results = cv_results
                    
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
            
            # Generate model explanations
            explanations = self._generate_model_explanations(
                best_model, X_train, config.feature_columns
            )
            
            # Save best model with secure permissions
            model_filename = f"model_{model_id}.joblib"
            model_path = Path(settings.ml.model_path) / model_filename
            
            try:
                joblib.dump(best_model, model_path)
                # Set restrictive permissions
                try:
                    os.chmod(model_path, 0o600)
                except (OSError, AttributeError):
                    pass
                logger.info(f"Model saved to: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save model to {model_path}: {str(e)}"
                )
            
            # Save explanations
            self._save_model_explanations(model_id, explanations)
            
            # Enhanced registry entry
            registry = self._load_model_registry()
            registry[model_id] = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "problem_type": config.problem_type.value,
                "algorithm": best_algorithm.value,
                "feature_columns": config.feature_columns,
                "target_column": config.target_column,
                "model_path": str(model_path),
                "hyperparameters": all_results[best_algorithm.value]["hyperparameters"],
                "train_size": len(X_train),
                "test_size": len(X_test),
                "total_training_time": total_training_time,
                "best_score": best_score,
                "metrics": best_metrics.dict(),
                "cross_validation": best_cv_results.dict() if best_cv_results else None,
                "all_algorithm_results": {
                    k: {
                        "score": v["score"], 
                        "training_time": v["training_time"],
                        "cv_score": v["cv_results"].mean_score if v["cv_results"] else None
                    } for k, v in all_results.items()
                },
                "data_info": {
                    "total_samples": len(df),
                    "clean_samples": len(X_full),
                    "features_count": len(config.feature_columns),
                    "target_unique_values": int(df[config.target_column].nunique()),
                    "data_hash": data_hash
                },
                "experiment_metadata": experiment_metadata.__dict__,
                "model_explanations": {
                    "has_feature_importance": explanations.feature_importance is not None,
                    "has_shap_values": explanations.shap_values is not None,
                    "top_features": list(explanations.feature_importance.keys())[:5] 
                                  if explanations.feature_importance else None
                },
                "training_config": {
                    "use_cross_validation": use_cv,
                    "tune_hyperparameters": tune_hyperparams,
                    "scoring_metric": scoring_metric,
                    "sampling_strategy": getattr(config, 'sampling_strategy', SamplingStrategy.NONE).value
                }
            }
            self._save_model_registry(registry)
            
            # Create enhanced response
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
            
            logger.info(f"Enhanced model training completed - ID: {model_id}, Algorithm: {best_algorithm.value}")
            # Observe total training duration with model label
            try:
                modeling_training_seconds.labels(model=best_algorithm.value).observe(time.time() - start_time)
            except Exception:
                pass
            return response
            
        except HTTPException:
            # Count error
            try:
                service_errors_total.labels(service="modeling", error_type="HTTPException").inc()
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"Unexpected error during training: {str(e)}")
            try:
                service_errors_total.labels(service="modeling", error_type=type(e).__name__).inc()
            except Exception:
                pass
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during model training: {str(e)}"
            )
        finally:
            if _traced_ctx is not None:
                try:
                    _traced_ctx.__exit__(None, None, None)
                except Exception:
                    pass
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       problem_type: ProblemType, model=None, X_test=None) -> EvaluationMetrics:
        """Enhanced model evaluation with comprehensive metrics."""
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
                
                # Add ROC-AUC if model has probability prediction
                try:
                    if model and hasattr(model, 'predict_proba') and X_test is not None:
                        if unique_labels == 2:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            metrics.roc_auc = float(roc_auc_score(y_true, y_proba))
                        elif unique_labels > 2:
                            y_proba = model.predict_proba(X_test)
                            metrics.roc_auc = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
                except Exception as e:
                    logger.warning(f"Failed to compute ROC-AUC: {e}")
                    
            elif problem_type == ProblemType.REGRESSION:
                metrics.mse = float(mean_squared_error(y_true, y_pred))
                metrics.r2_score = float(r2_score(y_true, y_pred))
                metrics.rmse = float(np.sqrt(metrics.mse))
                
                # Add mean absolute error
                try:
                    metrics.mae = float(mean_absolute_error(y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Failed to compute MAE: {e}")
                
                # Add median absolute error and explained variance
                try:
                    from sklearn.metrics import median_absolute_error, explained_variance_score
                    metrics.median_ae = float(median_absolute_error(y_true, y_pred))
                    metrics.explained_variance = float(explained_variance_score(y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Failed to compute additional regression metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model evaluation failed: {str(e)}"
            )
    
    def predict(self, model_id: str, data: pd.DataFrame, 
                return_probabilities: bool = False) -> PredictionResponse:
        """Enhanced prediction with input validation and probability scores."""
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
            
            # Validate model path
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {model_path}"
                )
            
            # Validate path security
            model_path_resolved = Path(model_path).resolve()
            if not str(model_path_resolved).startswith(str(Path(settings.ml.model_path).resolve())):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid model path"
                )
            
            try:
                model = joblib.load(model_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model: {str(e)}"
                )
            
            # Enhanced input validation
            if data is None or data.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Input data is empty or None"
                )
            
            feature_columns = model_info.get('feature_columns', [])
            problem_type = model_info.get('problem_type', 'unknown')
            algorithm = model_info.get('algorithm', 'unknown')
            
            # Validate input schema
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_features}. "
                           f"Expected features: {feature_columns}"
                )
            
            # Select only the required features in the correct order
            X = data[feature_columns].copy()
            
            # Enhanced data validation
            if X.isnull().any().any():
                null_columns = X.columns[X.isnull().any()].tolist()
                null_count = X.isnull().sum().sum()
                raise HTTPException(
                    status_code=400,
                    detail=f"Input data contains {null_count} missing values in columns: {null_columns}. "
                           f"Please handle missing values before prediction."
                )
            
            # Check data types
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
            
            # Check for data drift (basic statistics comparison)
            try:
                training_stats = model_info.get('feature_statistics', {})
                if training_stats:
                    drift_warnings = []
                    for col in feature_columns:
                        if col in training_stats:
                            train_mean = training_stats[col].get('mean', 0)
                            train_std = training_stats[col].get('std', 1)
                            current_mean = X[col].mean()
                            
                            # Simple drift detection
                            if abs(current_mean - train_mean) > 2 * train_std:
                                drift_warnings.append(f"Potential drift in feature '{col}'")
                    
                    if drift_warnings:
                        logger.warning(f"Data drift detected: {drift_warnings}")
            except Exception as e:
                logger.warning(f"Data drift check failed: {e}")
            
            # Make predictions with timeout
            try:
                predictions = model.predict(X)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction error: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction failed: {str(e)}"
                )
            
            # Get confidence scores and probabilities
            confidence_scores = None
            probabilities = None
            
            if problem_type == ProblemType.CLASSIFICATION.value:
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        confidence_scores = np.max(proba, axis=1).tolist()
                        
                        if return_probabilities:
                            probabilities = proba.tolist()
                            
                    except Exception as e:
                        logger.warning(f"Failed to compute confidence scores: {e}")
                
                elif hasattr(model, 'decision_function'):
                    try:
                        decision_scores = model.decision_function(X)
                        if len(decision_scores.shape) == 1:
                            # Binary classification
                            confidence_scores = np.abs(decision_scores).tolist()
                        else:
                            # Multi-class
                            confidence_scores = np.max(decision_scores, axis=1).tolist()
                    except Exception as e:
                        logger.warning(f"Failed to compute decision scores: {e}")
            
            response = PredictionResponse(
                model_id=model_id,
                predictions=predictions.tolist(),
                confidence_scores=confidence_scores,
                probabilities=probabilities,
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
    
    def get_model_explanations(self, model_id: str) -> Dict:
        """Retrieve model explanations."""
        try:
            explanation_file = self.explanations_path / f"{model_id}_explanations.joblib"
            
            if not explanation_file.exists():
                return {"message": "No explanations available for this model"}
            
            explanations = joblib.load(explanation_file)
            
            result = {}
            
            if explanations.feature_importance:
                result["feature_importance"] = explanations.feature_importance
            
            if explanations.shap_values is not None:
                # Return summary statistics of SHAP values instead of full arrays
                result["shap_summary"] = {
                    "mean_absolute_shap": np.mean(np.abs(explanations.shap_values), axis=0).tolist(),
                    "feature_names": explanations.shap_feature_names
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve explanations for model {model_id}: {e}")
            return {"error": f"Failed to retrieve explanations: {str(e)}"}
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Enhanced model info retrieval with additional metadata."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found in registry"
            )
        
        model_data = registry[model_id]
        
        # Convert to ModelInfo with enhanced data
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
                best_score=model_data.get('best_score', 0.0),
                cross_validation=model_data.get('cross_validation', {}),
                experiment_metadata=model_data.get('experiment_metadata', {}),
                data_hash=model_data.get('data_info', {}).get('data_hash', ''),
                explanations_available=model_data.get('model_explanations', {}).get('has_feature_importance', False)
            )
            return model_info
        except Exception as e:
            logger.error(f"Failed to convert model data to ModelInfo: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve model info: {str(e)}"
            )
    
    def list_models(self, include_stats: bool = True) -> ModelListResponse:
        """Enhanced model listing with better statistics."""
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
        
        # Enhanced summary statistics
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
            healthy_models = 0
            for model in models:
                model_path = Path(model.model_path)
                if not model_path.exists():
                    broken_models.append(model.model_id)
                else:
                    healthy_models += 1
            
            # Performance statistics
            if models:
                avg_training_time = sum(model.training_time for model in models) / len(models)
                total_training_time = sum(model.training_time for model in models)
            else:
                avg_training_time = 0
                total_training_time = 0
            
            summary = ModelSummary(
                total_models=total_models,
                classification_models=classification_models,
                regression_models=regression_models,
                algorithm_distribution=algorithm_counts,
                broken_models=broken_models,
                healthy_models=healthy_models,
                average_training_time=avg_training_time,
                total_training_time=total_training_time
            )
        
        return ModelListResponse(
            models=models,
            summary=summary
        )
    
    def delete_model(self, model_id: str) -> dict:
        """Enhanced model deletion with cleanup of explanations."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        model_info = registry[model_id]
        model_path = Path(model_info['model_path'])
        explanation_path = self.explanations_path / f"{model_id}_explanations.joblib"
        
        deleted_files = []
        failed_deletions = []
        
        try:
            # Remove model file if it exists
            if model_path.exists():
                model_path.unlink()
                deleted_files.append(str(model_path))
                logger.info(f"Model file deleted: {model_path}")
            else:
                logger.warning(f"Model file not found during deletion: {model_path}")
            
            # Remove explanation file if it exists
            if explanation_path.exists():
                explanation_path.unlink()
                deleted_files.append(str(explanation_path))
                logger.info(f"Explanation file deleted: {explanation_path}")
            
            # Remove from registry
            del registry[model_id]
            self._save_model_registry(registry)
            
            logger.info(f"Model {model_id} deleted successfully")
            return {
                "message": f"Model {model_id} deleted successfully",
                "deleted_files": deleted_files,
                "failed_deletions": failed_deletions
            }
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model: {str(e)}"
            )
    
    def get_model_performance_comparison(self, problem_type: Optional[str] = None) -> dict:
        """Enhanced performance comparison with cross-validation results."""
        registry = self._load_model_registry()
        
        if not registry:
            return {"message": "No models found in registry"}
        
        classification_models = []
        regression_models = []
        
        for model_id, model_info in registry.items():
            model_problem_type = model_info.get('problem_type')
            
            # Filter by problem type if specified
            if problem_type and model_problem_type != problem_type:
                continue
            
            metrics = model_info.get('metrics', {})
            cv_results = model_info.get('cross_validation', {})
            
            model_summary = {
                "model_id": model_id,
                "algorithm": model_info.get('algorithm', 'unknown'),
                "timestamp": model_info.get('timestamp'),
                "train_size": model_info.get('train_size'),
                "test_size": model_info.get('test_size'),
                "training_time": model_info.get('total_training_time'),
                "best_score": model_info.get('best_score'),
                "cv_score": cv_results.get('mean_score') if cv_results else None,
                "cv_std": cv_results.get('std_score') if cv_results else None,
                "hyperparameters": model_info.get('hyperparameters', {}),
                "data_hash": model_info.get('data_info', {}).get('data_hash', '')
            }
            
            if model_problem_type == ProblemType.CLASSIFICATION.value:
                model_summary.update({
                    "accuracy": metrics.get('accuracy'),
                    "precision": metrics.get('precision'),
                    "recall": metrics.get('recall'),
                    "f1_score": metrics.get('f1_score'),
                    "roc_auc": metrics.get('roc_auc')
                })
                classification_models.append(model_summary)
                
            elif model_problem_type == ProblemType.REGRESSION.value:
                model_summary.update({
                    "mse": metrics.get('mse'),
                    "rmse": metrics.get('rmse'),
                    "r2_score": metrics.get('r2_score'),
                    "mae": metrics.get('mae'),
                    "explained_variance": metrics.get('explained_variance')
                })
                regression_models.append(model_summary)
        
        # Sort models by performance (with CV score priority)
        if classification_models:
            classification_models.sort(
                key=lambda x: x.get('cv_score') if x.get('cv_score') is not None else x.get('f1_score', 0), 
                reverse=True
            )
        if regression_models:
            regression_models.sort(
                key=lambda x: x.get('cv_score') if x.get('cv_score') is not None else x.get('r2_score', -float('inf')), 
                reverse=True
            )
        
        return {
            "classification_models": classification_models,
            "regression_models": regression_models,
            "summary": {
                "total_classification": len(classification_models),
                "total_regression": len(regression_models),
                "best_classification": classification_models[0] if classification_models else None,
                "best_regression": regression_models[0] if regression_models else None,
                "filter_applied": problem_type is not None
            }
        }
    
    def export_model_config(self, model_id: str) -> ModelConfigExport:
        """Enhanced model configuration export with full reproducibility data."""
        model_info = self.get_model_info(model_id)
        registry = self._load_model_registry()
        model_data = registry.get(model_id, {})
        
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
                "training_time": model_info.training_time,
                "cross_validation": model_data.get('cross_validation', {}),
                "training_config": model_data.get('training_config', {})
            },
            performance=model_info.metrics,
            timestamp=model_info.timestamp,
            experiment_metadata=model_data.get('experiment_metadata', {}),
            data_info=model_data.get('data_info', {}),
            reproducibility_hash=model_data.get('data_info', {}).get('data_hash', '')
        )
        
        return config
    
    def cleanup_broken_models(self, dry_run: bool = False) -> CleanupResult:
        """Enhanced cleanup with dry-run option and detailed reporting."""
        registry = self._load_model_registry()
        cleaned_models = []
        failed_cleanups = []
        orphaned_files = []
        
        # Check registry entries
        for model_id, model_info in list(registry.items()):
            model_path = Path(model_info.get('model_path', ''))
            
            # Check if model file exists
            if not model_path.exists():
                if not dry_run:
                    try:
                        del registry[model_id]
                        cleaned_models.append(model_id)
                        logger.info(f"Removed registry entry for missing model: {model_id}")
                    except Exception as e:
                        failed_cleanups.append({"model_id": model_id, "error": str(e)})
                        logger.error(f"Failed to remove registry entry for {model_id}: {e}")
                else:
                    cleaned_models.append(model_id)
        
        # Check for orphaned model files
        model_dir = Path(settings.ml.model_path)
        if model_dir.exists():
            for model_file in model_dir.glob("model_*.joblib"):
                model_id = model_file.stem.replace("model_", "")
                if model_id not in registry:
                    orphaned_files.append(str(model_file))
                    if not dry_run:
                        try:
                            model_file.unlink()
                            logger.info(f"Removed orphaned model file: {model_file}")
                        except Exception as e:
                            failed_cleanups.append({
                                "file": str(model_file), 
                                "error": str(e)
                            })
        
        # Check for orphaned explanation files
        if self.explanations_path.exists():
            for exp_file in self.explanations_path.glob("*_explanations.joblib"):
                model_id = exp_file.stem.replace("_explanations", "")
                if model_id not in registry:
                    orphaned_files.append(str(exp_file))
                    if not dry_run:
                        try:
                            exp_file.unlink()
                            logger.info(f"Removed orphaned explanation file: {exp_file}")
                        except Exception as e:
                            failed_cleanups.append({
                                "file": str(exp_file), 
                                "error": str(e)
                            })
        
        # Save updated registry if not dry run
        if cleaned_models and not dry_run:
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
            total_cleaned=len(cleaned_models),
            orphaned_files=orphaned_files,
            dry_run=dry_run
        )
    
    def validate_model_health(self, model_id: str, 
                             check_predictions: bool = False) -> ModelHealthStatus:
        """Enhanced model health validation with prediction testing."""
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
                if hasattr(model, 'feature_importances_'):
                    if model.feature_importances_ is None:
                        warnings.append("Model may not be properly fitted")
                elif hasattr(model, 'coef_'):
                    if model.coef_ is None:
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
            
            # Test predictions if requested
            if check_predictions and len(issues) == 0:
                try:
                    # Create dummy data for prediction test
                    dummy_data = pd.DataFrame({
                        col: [0.0] for col in expected_features
                    })
                    
                    predictions = model.predict(dummy_data)
                    if len(predictions) != 1:
                        warnings.append("Unexpected prediction output format")
                        
                except Exception as e:
                    warnings.append(f"Prediction test failed: {str(e)}")
            
            # Check file properties
            file_size = model_path.stat().st_size
            last_modified = model_path.stat().st_mtime
            
            if file_size < 1000:  # Less than 1KB
                warnings.append(f"Model file unusually small: {file_size} bytes")
            elif file_size > 100_000_000:  # Larger than 100MB
                warnings.append(f"Model file very large: {file_size/1_000_000:.1f} MB")
            
            # Check explanation files
            explanation_path = self.explanations_path / f"{model_id}_explanations.joblib"
            has_explanations = explanation_path.exists()
            
            healthy = len(issues) == 0
            
            return ModelHealthStatus(
                model_id=model_id,
                healthy=healthy,
                issues=issues,
                warnings=warnings,
                file_size=file_size,
                last_modified=last_modified,
                has_explanations=has_explanations,
                prediction_test_passed=check_predictions and len(issues) == 0
            )
            
        except Exception as e:
            return ModelHealthStatus(
                model_id=model_id,
                healthy=False,
                issues=[f"Health check failed: {str(e)}"],
                warnings=[]
            )
    
    def batch_validate_models(self) -> Dict[str, ModelHealthStatus]:
        """Validate health of all models in the registry."""
        registry = self._load_model_registry()
        results = {}
        
        for model_id in registry.keys():
            try:
                health_status = self.validate_model_health(model_id, check_predictions=True)
                results[model_id] = health_status
            except Exception as e:
                results[model_id] = ModelHealthStatus(
                    model_id=model_id,
                    healthy=False,
                    issues=[f"Validation failed: {str(e)}"],
                    warnings=[]
                )
        
        return results
    
    def get_feature_statistics(self, model_id: str) -> Dict:
        """Get feature statistics from the training data (if available)."""
        registry = self._load_model_registry()
        
        if model_id not in registry:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID {model_id} not found"
            )
        
        model_info = registry[model_id]
        feature_stats = model_info.get('feature_statistics', {})
        
        if not feature_stats:
            return {
                "message": "Feature statistics not available for this model",
                "suggestion": "Re-train the model with enhanced version to get feature statistics"
            }
        
        return {
            "model_id": model_id,
            "feature_statistics": feature_stats,
            "feature_count": len(feature_stats)
        }

# ==========================================
# ADVANCED ENTERPRISE COMPONENTS
# ==========================================

class HyperparameterTuningEngine:
    """Advanced hyperparameter tuning with resource constraints and early stopping."""
    
    def __init__(self):
        self.active_studies: Dict[str, Any] = {}
        self.trial_history: Dict[str, List] = defaultdict(list)
        self._lock = RLock()
    
    def optimize_hyperparameters(self, model_class: type, X_train: pd.DataFrame, 
                                y_train: pd.Series, problem_type: str,
                                algorithm_name: str, scoring_metric: str = None,
                                max_trials: int = None, max_time_minutes: int = None) -> Dict[str, Any]:
        """Resource-constrained hyperparameter optimization."""
        
        max_trials = max_trials or CONFIG['MAX_TRIALS_PER_ALGORITHM']
        max_time_minutes = max_time_minutes or CONFIG['MAX_TUNING_TIME_MINUTES']
        
        try:
            if OPTUNA_AVAILABLE:
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                
                def objective(trial):
                    try:
                        params = self._generate_trial_params(trial, algorithm_name)
                        model = model_class(**params)
                        cv_scores = cross_val_score(
                            model, X_train, y_train, 
                            cv=3, scoring=scoring_metric or 'accuracy',
                            n_jobs=1
                        )
                        return cv_scores.mean()
                    except Exception:
                        return -float('inf')
                
                study.optimize(objective, n_trials=max_trials, timeout=max_time_minutes * 60)
                
                return {
                    'best_params': study.best_params,
                    'best_score': study.best_value,
                    'n_trials': len(study.trials)
                }
            else:
                return {'best_params': {}, 'best_score': 0.0, 'n_trials': 0}
                
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise HyperparameterError(f"Optimization failed: {e}")
    
    def _generate_trial_params(self, trial, algorithm_name: str) -> Dict[str, Any]:
        """Generate hyperparameters for a trial."""
        params = {'random_state': 42}
        
        if algorithm_name in ["RandomForestClassifier", "RandomForestRegressor"]:
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20)
            })
        elif algorithm_name == "LogisticRegression":
            params.update({
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            })
        
        if algorithm_name == "SVC":
            params['probability'] = True
            
        return params

class ModelQualityGateSystem:
    """System for enforcing model quality gates."""
    
    def __init__(self):
        self.quality_gates: List[ModelQualityGate] = []
        self.validation_history: List[Dict] = []
        self._lock = RLock()
        self._setup_default_gates()
    
    def _setup_default_gates(self):
        """Setup default quality gates."""
        default_gates = [
            ModelQualityGate(
                name="minimum_accuracy",
                metric="accuracy", 
                threshold=0.6,
                comparison="greater_equal",
                required=True
            )
        ]
        
        with self._lock:
            self.quality_gates.extend(default_gates)
    
    def validate_model(self, model_metrics: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Validate model against quality gates."""
        validation_result = {
            'model_id': model_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'passed': True,
            'failed_gates': [],
            'passed_gates': []
        }
        
        for gate in self.quality_gates:
            if gate.metric in model_metrics:
                actual_value = model_metrics[gate.metric]
                gate_passed = actual_value >= gate.threshold if gate.comparison == "greater_equal" else True
                
                gate_result = {
                    'name': gate.name,
                    'metric': gate.metric,
                    'expected': f"{gate.comparison} {gate.threshold}",
                    'actual': actual_value,
                    'passed': gate_passed
                }
                
                if gate_passed:
                    validation_result['passed_gates'].append(gate_result)
                else:
                    validation_result['failed_gates'].append(gate_result)
                    if gate.required:
                        validation_result['passed'] = False
        
        with self._lock:
            self.validation_history.append(validation_result)
        
        return validation_result

# Initialize enterprise components
hyperparameter_engine = HyperparameterTuningEngine()
quality_gate_system = ModelQualityGateSystem()

# Create enhanced service instance
modeling_service = EnhancedModelingService()

# Public interface functions with backward compatibility
def train_model(config: ModelConfig) -> TrainingResponse:
    """Train a machine learning model based on configuration."""
    return modeling_service.train_model(config)

def predict(model_id: str, data: pd.DataFrame, 
           return_probabilities: bool = False) -> PredictionResponse:
    """Make predictions using a trained model."""
    return modeling_service.predict(model_id, data, return_probabilities)

def list_models(include_stats: bool = True) -> ModelListResponse:
    """List all models in the registry."""
    return modeling_service.list_models(include_stats)

def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    return modeling_service.get_model_info(model_id)

def delete_model(model_id: str) -> dict:
    """Delete a model from the registry."""
    return modeling_service.delete_model(model_id)

def get_model_performance_comparison(problem_type: Optional[str] = None) -> dict:
    """Get performance comparison of all models."""
    return modeling_service.get_model_performance_comparison(problem_type)

def export_model_config(model_id: str) -> ModelConfigExport:
    """Export model configuration for reproducibility."""
    return modeling_service.export_model_config(model_id)

def cleanup_broken_models(dry_run: bool = False) -> CleanupResult:
    """Clean up models with missing files."""
    return modeling_service.cleanup_broken_models(dry_run)

def validate_model_health(model_id: str, check_predictions: bool = False) -> ModelHealthStatus:
    """Perform health check on a specific model."""
    return modeling_service.validate_model_health(model_id, check_predictions)

# Enhanced public interface functions
def get_model_explanations(model_id: str) -> Dict:
    """Retrieve model explanations."""
    return modeling_service.get_model_explanations(model_id)

def batch_validate_models() -> Dict[str, ModelHealthStatus]:
    """Validate health of all models in the registry."""
    return modeling_service.batch_validate_models()

def get_feature_statistics(model_id: str) -> Dict:
    """Get feature statistics from training data."""
    return modeling_service.get_feature_statistics(model_id)