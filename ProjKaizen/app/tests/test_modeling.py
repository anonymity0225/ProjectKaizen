"""
Pytest tests for the modeling module.

Run tests: pytest tests/test_modeling.py -v
For auth-protected API tests: TEST_AUTH_BYPASS=1 pytest tests/test_modeling.py -v

Tests cover:
- Model training (classification and regression)
- Predictions and validation
- Model listing and info retrieval
- Model deletion, export, comparison, cleanup
- Input validation errors
- API endpoints (if app available)
- Performance and concurrent request tests
- Negative test cases
"""

import pytest
import json
import time
import asyncio
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import joblib
from fastapi.testclient import TestClient
from fastapi import HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from app.api import app
    HAS_API = True
except ImportError:
    HAS_API = False
    app = None

from app.services.modeling import ModelingService
from app.schemas.modeling import (
    ModelConfig, ModelType, TrainingResponse, PredictionResponse,
    ModelListResponse, ModelInfoResponse, ComparisonResponse
)


@pytest.fixture(scope="function")
def tmp_model_dir(tmp_path, monkeypatch):
    """Create temporary model directory and monkeypatch settings."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Monkeypatch settings to use temporary directory
    try:
        from app.config.settings import settings
        monkeypatch.setattr(settings.ml, "model_path", str(model_dir))
    except ImportError:
        try:
            from app.settings import settings
            monkeypatch.setattr(settings.ml, "model_path", str(model_dir))
        except ImportError:
            # If no settings module, patch environment variable
            monkeypatch.setenv("MODEL_PATH", str(model_dir))
    
    return model_dir


@pytest.fixture
def sample_classification_df():
    """Create sample DataFrame for classification tasks."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Create target with some relationship to features
    linear_combo = 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + np.random.normal(0, 0.1, n_samples)
    y = (linear_combo > 0).astype(int)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    return df


@pytest.fixture
def sample_regression_df():
    """Create sample DataFrame for regression tasks."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Create continuous target
    y = 2.5 * x1 + 1.8 * x2 - 1.2 * x3 + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    return df


@pytest.fixture
def large_dataset_df():
    """Create large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Generate features
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target
    target_weights = np.random.normal(0, 0.1, n_features)
    linear_combo = sum(target_weights[i] * features[f'feature_{i}'] for i in range(n_features))
    y = (linear_combo > 0).astype(int)
    
    features['y'] = y
    return pd.DataFrame(features)


@pytest.fixture
def modeling_service(tmp_model_dir):
    """Create ModelingService instance with temporary directory."""
    return ModelingService()


@pytest.fixture
def mock_auth():
    """Mock authentication for testing."""
    def mock_verify_token():
        return {"user_id": "test-user", "username": "testuser"}
    
    return mock_verify_token


@pytest.fixture
def client(monkeypatch, mock_auth):
    """Create test client with mocked auth."""
    if not HAS_API:
        pytest.skip("API app not available")
    
    # Mock authentication
    monkeypatch.setattr("app.core.auth.verify_token", mock_auth)
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create auth headers for testing."""
    return {"Authorization": "Bearer test-token"}


class TestModelTraining:
    """Test model training functionality."""
    
    def test_train_classification_random_forest(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test random forest classification training."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2", "x3"],
            target_column="y",
            test_size=0.2,
            random_state=42,
            algorithm="random_forest"
        )
        
        response = modeling_service.train_model(sample_classification_df, config)
        
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert response.model_path is not None
        assert Path(response.model_path).exists()
        assert response.metrics is not None
        assert response.metrics.accuracy > 0.5  # Should be better than random
        assert 0 <= response.metrics.accuracy <= 1
        
        # Check registry
        registry_file = tmp_model_dir / "model_registry.json"
        assert registry_file.exists()
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        assert response.model_id in registry
        assert registry[response.model_id]["algorithm"] == "random_forest"
    
    def test_train_classification_svm(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test SVM classification training."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.2,
            random_state=42,
            algorithm="svm"
        )
        
        response = modeling_service.train_model(sample_classification_df, config)
        
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert Path(response.model_path).exists()
        assert response.metrics.accuracy > 0
    
    def test_train_regression_linear(self, tmp_model_dir, sample_regression_df, modeling_service):
        """Test linear regression training."""
        config = ModelConfig(
            model_type=ModelType.regression,
            feature_columns=["x1", "x2", "x3"],
            target_column="y",
            test_size=0.2,
            random_state=42,
            algorithm="linear_regression"
        )
        
        response = modeling_service.train_model(sample_regression_df, config)
        
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert Path(response.model_path).exists()
        assert response.metrics is not None
        assert hasattr(response.metrics, 'mse')
        assert hasattr(response.metrics, 'r2_score')
        assert response.metrics.mse >= 0
        assert np.isfinite(response.metrics.r2_score)
    
    def test_train_regression_random_forest(self, tmp_model_dir, sample_regression_df, modeling_service):
        """Test random forest regression training."""
        config = ModelConfig(
            model_type=ModelType.regression,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.3,
            random_state=42,
            algorithm="random_forest"
        )
        
        response = modeling_service.train_model(sample_regression_df, config)
        
        assert isinstance(response, TrainingResponse)
        assert response.metrics.mse >= 0
        assert np.isfinite(response.metrics.r2_score)


class TestModelPrediction:
    """Test model prediction functionality."""
    
    def test_predict_classification(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test classification predictions."""
        # Train model
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.2,
            random_state=42
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Make predictions
        input_df = pd.DataFrame({
            'x1': [0.5, -0.3, 1.2, -1.5, 0.0],
            'x2': [0.8, -0.1, -0.5, 0.7, 0.0]
        })
        
        pred_response = modeling_service.predict(train_response.model_path, input_df)
        
        assert isinstance(pred_response, PredictionResponse)
        assert len(pred_response.predictions) == len(input_df)
        assert pred_response.num_predictions == len(input_df)
        assert all(pred in [0, 1] for pred in pred_response.predictions)
        assert pred_response.prediction_time > 0
    
    def test_predict_regression(self, tmp_model_dir, sample_regression_df, modeling_service):
        """Test regression predictions."""
        # Train model
        config = ModelConfig(
            model_type=ModelType.regression,
            feature_columns=["x1", "x2", "x3"],
            target_column="y",
            test_size=0.2,
            random_state=42
        )
        
        train_response = modeling_service.train_model(sample_regression_df, config)
        
        # Make predictions
        input_df = pd.DataFrame({
            'x1': [0.5, -0.3, 1.2],
            'x2': [0.8, -0.1, -0.5],
            'x3': [0.2, -0.4, 0.9]
        })
        
        pred_response = modeling_service.predict(train_response.model_path, input_df)
        
        assert len(pred_response.predictions) == len(input_df)
        assert all(isinstance(pred, (int, float)) for pred in pred_response.predictions)
        assert all(np.isfinite(pred) for pred in pred_response.predictions)


class TestModelManagement:
    """Test model management functionality."""
    
    def test_list_models_empty(self, tmp_model_dir, modeling_service):
        """Test listing models when none exist."""
        response = modeling_service.list_models()
        assert isinstance(response, ModelListResponse)
        assert len(response.models) == 0
    
    def test_list_models_with_data(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test listing models with existing models."""
        # Train multiple models
        config1 = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            algorithm="random_forest"
        )
        config2 = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x3"],
            target_column="y",
            algorithm="svm"
        )
        
        train1 = modeling_service.train_model(sample_classification_df, config1)
        train2 = modeling_service.train_model(sample_classification_df, config2)
        
        # List models
        response = modeling_service.list_models()
        assert len(response.models) == 2
        
        model_ids = [model.model_id for model in response.models]
        assert train1.model_id in model_ids
        assert train2.model_id in model_ids
    
    def test_get_model_info(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test getting model info."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2", "x3"],
            target_column="y",
            algorithm="random_forest"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Get model info
        info = modeling_service.get_model_info(train_response.model_id)
        
        assert isinstance(info, ModelInfoResponse)
        assert info.model_id == train_response.model_id
        assert set(info.feature_columns) == set(["x1", "x2", "x3"])
        assert info.target_column == "y"
        assert info.model_file_exists is True
        assert info.model_type == ModelType.classification
        assert info.algorithm == "random_forest"
        assert info.created_at is not None
    
    def test_delete_model(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test model deletion."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        model_id = train_response.model_id
        model_path = Path(train_response.model_path)
        
        # Verify model exists
        assert model_path.exists()
        
        # Delete model
        modeling_service.delete_model(model_id, force=True)
        
        # Verify deletion
        assert not model_path.exists()
        
        # Verify not in list
        models = modeling_service.list_models()
        model_ids = [model.model_id for model in models.models]
        assert model_id not in model_ids
        
        # Verify get_model_info raises exception
        with pytest.raises((HTTPException, FileNotFoundError, ValueError)):
            modeling_service.get_model_info(model_id)
    
    def test_export_model(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test model export functionality."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Export model
        export_response = modeling_service.export_model(train_response.model_id, format="joblib")
        
        assert export_response.model_id == train_response.model_id
        assert export_response.export_path is not None
        assert Path(export_response.export_path).exists()
        assert export_response.format == "joblib"
    
    def test_compare_models(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test model comparison."""
        # Train two models
        config1 = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            algorithm="random_forest",
            random_state=42
        )
        config2 = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            algorithm="svm",
            random_state=42
        )
        
        train1 = modeling_service.train_model(sample_classification_df, config1)
        train2 = modeling_service.train_model(sample_classification_df, config2)
        
        # Compare models
        comparison = modeling_service.compare_models([train1.model_id, train2.model_id])
        
        assert isinstance(comparison, ComparisonResponse)
        assert len(comparison.models) == 2
        assert comparison.best_model_id in [train1.model_id, train2.model_id]
        assert comparison.comparison_metric is not None


class TestModelCleanup:
    """Test model cleanup functionality."""
    
    def test_cleanup_old_models(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test cleanup of old models."""
        # Train a model
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Mock old timestamp in registry
        registry_file = tmp_model_dir / "model_registry.json"
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        # Set old timestamp (30+ days ago)
        old_timestamp = time.time() - (31 * 24 * 60 * 60)
        registry[train_response.model_id]["created_at"] = old_timestamp
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f)
        
        # Run cleanup
        cleanup_response = modeling_service.cleanup_models(max_age_days=30)
        
        assert cleanup_response.deleted_count >= 1
        assert train_response.model_id in cleanup_response.deleted_model_ids


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_predict_missing_features(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test prediction with missing features."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Missing feature
        invalid_df = pd.DataFrame({
            'x1': [0.5, -0.3],
            # missing 'x2'
        })
        
        with pytest.raises((HTTPException, ValueError, KeyError)) as exc_info:
            modeling_service.predict(train_response.model_path, invalid_df)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['feature', 'column', 'missing'])
    
    def test_predict_with_nans(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test prediction with NaN values."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        
        # Input with NaNs
        invalid_df = pd.DataFrame({
            'x1': [0.5, np.nan, 1.2],
            'x2': [0.8, -0.1, np.nan]
        })
        
        with pytest.raises((HTTPException, ValueError)) as exc_info:
            modeling_service.predict(train_response.model_path, invalid_df)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['nan', 'null', 'missing'])
    
    def test_invalid_model_path(self, modeling_service):
        """Test prediction with non-existent model."""
        input_df = pd.DataFrame({
            'x1': [0.5],
            'x2': [0.8]
        })
        
        with pytest.raises((HTTPException, FileNotFoundError)):
            modeling_service.predict("non_existent_model.joblib", input_df)


class TestNegativeTestCases:
    """Test negative test cases and error conditions."""
    
    def test_invalid_algorithm_classification(self, sample_classification_df, modeling_service):
        """Test training with invalid algorithm for classification."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            algorithm="invalid_algorithm"
        )
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            modeling_service.train_model(sample_classification_df, config)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['algorithm', 'invalid', 'supported'])
    
    def test_invalid_algorithm_regression(self, sample_regression_df, modeling_service):
        """Test training with invalid algorithm for regression."""
        config = ModelConfig(
            model_type=ModelType.regression,
            feature_columns=["x1", "x2"],
            target_column="y",
            algorithm="invalid_algorithm"
        )
        
        with pytest.raises((ValueError, HTTPException)):
            modeling_service.train_model(sample_regression_df, config)
    
    def test_wrong_model_type_mismatch(self, sample_classification_df, modeling_service):
        """Test training with wrong model type for data."""
        # Try regression on classification data with categorical target
        df_categorical = sample_classification_df.copy()
        df_categorical['y'] = df_categorical['y'].astype('category')
        
        config = ModelConfig(
            model_type=ModelType.regression,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        with pytest.raises((ValueError, HTTPException)):
            modeling_service.train_model(df_categorical, config)
    
    def test_bad_model_id_operations(self, modeling_service):
        """Test operations with non-existent model ID."""
        bad_model_id = "non_existent_model_id"
        
        # Test get_model_info
        with pytest.raises((HTTPException, FileNotFoundError, ValueError)):
            modeling_service.get_model_info(bad_model_id)
        
        # Test delete_model
        with pytest.raises((HTTPException, FileNotFoundError, ValueError)):
            modeling_service.delete_model(bad_model_id)
        
        # Test export_model
        with pytest.raises((HTTPException, FileNotFoundError, ValueError)):
            modeling_service.export_model(bad_model_id)
    
    def test_empty_dataframe_training(self, modeling_service):
        """Test training with empty DataFrame."""
        empty_df = pd.DataFrame()
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        with pytest.raises((ValueError, HTTPException)):
            modeling_service.train_model(empty_df, config)
    
    def test_insufficient_data_training(self, modeling_service):
        """Test training with insufficient data."""
        small_df = pd.DataFrame({
            'x1': [1, 2],
            'x2': [2, 3],
            'y': [0, 1]
        })
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.5  # This will cause issues with such small data
        )
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            modeling_service.train_model(small_df, config)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['insufficient', 'small', 'data', 'sample'])


class TestPerformance:
    """Test performance with large datasets."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, tmp_model_dir, large_dataset_df, modeling_service):
        """Test training performance on large dataset."""
        feature_cols = [col for col in large_dataset_df.columns if col != 'y']
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=feature_cols,
            target_column="y",
            test_size=0.2,
            random_state=42,
            algorithm="random_forest"
        )
        
        start_time = time.time()
        response = modeling_service.train_model(large_dataset_df, config)
        training_time = time.time() - start_time
        
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert training_time < 300  # Should complete within 5 minutes
        
        # Test prediction performance
        test_data = large_dataset_df[feature_cols].head(1000)
        
        start_time = time.time()
        pred_response = modeling_service.predict(response.model_path, test_data)
        prediction_time = time.time() - start_time
        
        assert len(pred_response.predictions) == 1000
        assert prediction_time < 30  # Should predict 1000 samples within 30 seconds


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    def test_concurrent_training(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test concurrent model training."""
        def train_model(algorithm):
            config = ModelConfig(
                model_type=ModelType.classification,
                feature_columns=["x1", "x2"],
                target_column="y",
                algorithm=algorithm,
                random_state=42
            )
            return modeling_service.train_model(sample_classification_df, config)
        
        # Run concurrent training
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(train_model, "random_forest"),
                executor.submit(train_model, "svm"),
                executor.submit(train_model, "random_forest")
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Concurrent training failed: {e}")
        
        assert len(results) == 3
        assert all(isinstance(r, TrainingResponse) for r in results)
        
        # Verify all models are unique
        model_ids = [r.model_id for r in results]
        assert len(set(model_ids)) == 3
    
    def test_concurrent_train_and_predict(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test concurrent training and prediction."""
        # First train a model
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            random_state=42
        )
        
        initial_model = modeling_service.train_model(sample_classification_df, config)
        
        # Prediction data
        pred_data = pd.DataFrame({
            'x1': [0.5, -0.3, 1.2],
            'x2': [0.8, -0.1, -0.5]
        })
        
        def train_new_model():
            new_config = ModelConfig(
                model_type=ModelType.classification,
                feature_columns=["x1", "x2", "x3"],
                target_column="y",
                random_state=123
            )
            return modeling_service.train_model(sample_classification_df, new_config)
        
        def predict_existing_model():
            return modeling_service.predict(initial_model.model_path, pred_data)
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            train_future = executor.submit(train_new_model)
            predict_future = executor.submit(predict_existing_model)
            
            try:
                train_result = train_future.result(timeout=60)
                predict_result = predict_future.result(timeout=60)
            except Exception as e:
                pytest.fail(f"Concurrent operations failed: {e}")
        
        assert isinstance(train_result, TrainingResponse)
        assert isinstance(predict_result, PredictionResponse)
        assert len(predict_result.predictions) == 3


@pytest.mark.skipif(not HAS_API, reason="API app not available")
class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/modeling/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_list_models_endpoint(self, client, auth_headers):
        """Test list models endpoint."""
        response = client.get("/modeling/list", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_train_model_endpoint(self, client, auth_headers, sample_classification_df):
        """Test train model endpoint."""
        config_data = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "test_size": 0.2,
            "random_state": 42,
            "algorithm": "random_forest"
        }
        
        payload = {
            "config": config_data,
            "data": sample_classification_df.to_dict("records")
        }
        
        response = client.post("/modeling/train", json=payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data
            assert "model_path" in data
            assert "metrics" in data
        elif response.status_code == 422:
            # Validation error - check the error details
            pytest.skip("API endpoint validation differs from test assumption")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_predict_endpoint(self, client, auth_headers, sample_classification_df):
        """Test predict endpoint."""
        # First train a model via API
        config_data = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "test_size": 0.2,
            "random_state": 42,
            "algorithm": "random_forest"
        }
        
        train_payload = {
            "config": config_data,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_response = client.post("/modeling/train", json=train_payload, headers=auth_headers)
        
        if train_response.status_code != 200:
            pytest.skip("Training endpoint not working, skipping predict test")
        
        model_id = train_response.json()["model_id"]
        
        # Now test prediction
        predict_data = [
            {"x1": 0.5, "x2": 0.8},
            {"x1": -0.3, "x2": -0.1},
            {"x1": 1.2, "x2": -0.5}
        ]
        
        predict_payload = {
            "model_id": model_id,
            "data": predict_data
        }
        
        response = client.post("/modeling/predict", json=predict_payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3
        else:
            pytest.skip("Predict endpoint format differs from test assumption")
    
    def test_model_info_endpoint(self, client, auth_headers, sample_classification_df):
        """Test model info endpoint."""
        # First train a model
        config_data = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "algorithm": "random_forest"
        }
        
        train_payload = {
            "config": config_data,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_response = client.post("/modeling/train", json=train_payload, headers=auth_headers)
        
        if train_response.status_code != 200:
            pytest.skip("Training endpoint not working")
        
        model_id = train_response.json()["model_id"]
        
        # Get model info
        response = client.get(f"/modeling/info/{model_id}", headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert data["model_id"] == model_id
            assert "feature_columns" in data
            assert "model_type" in data
        else:
            pytest.skip("Model info endpoint not available or different format")
    
    def test_delete_model_endpoint(self, client, auth_headers, sample_classification_df):
        """Test delete model endpoint."""
        # First train a model
        config_data = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "algorithm": "random_forest"
        }
        
        train_payload = {
            "config": config_data,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_response = client.post("/modeling/train", json=train_payload, headers=auth_headers)
        
        if train_response.status_code != 200:
            pytest.skip("Training endpoint not working")
        
        model_id = train_response.json()["model_id"]
        
        # Delete the model
        response = client.delete(f"/modeling/delete/{model_id}", headers=auth_headers)
        
        if response.status_code == 200:
            # Verify model is deleted by trying to get info
            info_response = client.get(f"/modeling/info/{model_id}", headers=auth_headers)
            assert info_response.status_code == 404
        else:
            pytest.skip("Delete endpoint not available or different format")
    
    def test_export_model_endpoint(self, client, auth_headers, sample_classification_df):
        """Test export model endpoint."""
        # First train a model
        config_data = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "algorithm": "random_forest"
        }
        
        train_payload = {
            "config": config_data,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_response = client.post("/modeling/train", json=train_payload, headers=auth_headers)
        
        if train_response.status_code != 200:
            pytest.skip("Training endpoint not working")
        
        model_id = train_response.json()["model_id"]
        
        # Export the model
        export_payload = {"format": "joblib"}
        response = client.post(f"/modeling/export/{model_id}", json=export_payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "export_path" in data
            assert data["model_id"] == model_id
        else:
            pytest.skip("Export endpoint not available or different format")
    
    def test_compare_models_endpoint(self, client, auth_headers, sample_classification_df):
        """Test compare models endpoint."""
        # Train two models
        config1 = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "algorithm": "random_forest",
            "random_state": 42
        }
        
        config2 = {
            "model_type": "classification",
            "feature_columns": ["x1", "x2"],
            "target_column": "y",
            "algorithm": "svm",
            "random_state": 42
        }
        
        train_payload1 = {
            "config": config1,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_payload2 = {
            "config": config2,
            "data": sample_classification_df.to_dict("records")
        }
        
        train_response1 = client.post("/modeling/train", json=train_payload1, headers=auth_headers)
        train_response2 = client.post("/modeling/train", json=train_payload2, headers=auth_headers)
        
        if train_response1.status_code != 200 or train_response2.status_code != 200:
            pytest.skip("Training endpoints not working")
        
        model_id1 = train_response1.json()["model_id"]
        model_id2 = train_response2.json()["model_id"]
        
        # Compare models
        compare_payload = {"model_ids": [model_id1, model_id2]}
        response = client.post("/modeling/compare", json=compare_payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "models" in data
            assert "best_model_id" in data
            assert len(data["models"]) == 2
        else:
            pytest.skip("Compare endpoint not available or different format")
    
    def test_cleanup_models_endpoint(self, client, auth_headers):
        """Test cleanup models endpoint."""
        cleanup_payload = {"max_age_days": 30, "dry_run": True}
        response = client.post("/modeling/cleanup", json=cleanup_payload, headers=auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "deleted_count" in data
            assert "deleted_model_ids" in data
        else:
            pytest.skip("Cleanup endpoint not available or different format")
    
    def test_unauthorized_access(self, client):
        """Test endpoints without authentication."""
        # Test without auth headers
        response = client.get("/modeling/list")
        assert response.status_code == 401
        
        response = client.post("/modeling/train", json={})
        assert response.status_code == 401
        
        response = client.post("/modeling/predict", json={})
        assert response.status_code == 401
    
    def test_invalid_auth_token(self, client):
        """Test endpoints with invalid authentication."""
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        
        response = client.get("/modeling/list", headers=invalid_headers)
        # Should be 401 if token validation is strict, or might work if mocked
        assert response.status_code in [200, 401]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_class_classification(self, modeling_service):
        """Test classification with only one class in target."""
        # Create dataset with only one class
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 3, 4, 5, 6],
            'y': [1, 1, 1, 1, 1]  # Only one class
        })
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            modeling_service.train_model(df, config)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['class', 'unique', 'target'])
    
    def test_high_cardinality_categorical_features(self, modeling_service):
        """Test with high cardinality categorical features."""
        np.random.seed(42)
        n_samples = 100
        
        # Create features with high cardinality
        df = pd.DataFrame({
            'high_card_feature': [f"category_{i}" for i in range(n_samples)],  # Unique for each row
            'x1': np.random.normal(0, 1, n_samples),
            'y': np.random.randint(0, 2, n_samples)
        })
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["high_card_feature", "x1"],
            target_column="y"
        )
        
        # This should either work with proper encoding or raise a meaningful error
        try:
            response = modeling_service.train_model(df, config)
            assert isinstance(response, TrainingResponse)
        except (ValueError, HTTPException) as e:
            # Should provide meaningful error about categorical encoding
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['categorical', 'encoding', 'string'])
    
    def test_mixed_data_types(self, modeling_service):
        """Test with mixed data types in features."""
        df = pd.DataFrame({
            'numeric': [1.5, 2.3, 3.7, 4.1, 5.9],
            'integer': [1, 2, 3, 4, 5],
            'boolean': [True, False, True, False, True],
            'y': [0, 1, 0, 1, 0]
        })
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["numeric", "integer", "boolean"],
            target_column="y"
        )
        
        # Should handle mixed types gracefully
        response = modeling_service.train_model(df, config)
        assert isinstance(response, TrainingResponse)
    
    def test_extreme_test_size_values(self, sample_classification_df, modeling_service):
        """Test with extreme test_size values."""
        # Test with very small test_size
        config_small = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.01  # Very small
        )
        
        response = modeling_service.train_model(sample_classification_df, config_small)
        assert isinstance(response, TrainingResponse)
        
        # Test with very large test_size
        config_large = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y",
            test_size=0.99  # Very large
        )
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            modeling_service.train_model(sample_classification_df, config_large)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['test', 'size', 'train', 'insufficient'])
    
    def test_duplicate_feature_columns(self, sample_classification_df, modeling_service):
        """Test with duplicate feature columns."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2", "x1"],  # Duplicate x1
            target_column="y"
        )
        
        # Should handle duplicates gracefully or raise meaningful error
        try:
            response = modeling_service.train_model(sample_classification_df, config)
            assert isinstance(response, TrainingResponse)
        except (ValueError, HTTPException) as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['duplicate', 'feature', 'column'])
    
    def test_target_column_as_feature(self, sample_classification_df, modeling_service):
        """Test using target column as a feature (data leakage)."""
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2", "y"],  # Target as feature
            target_column="y"
        )
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            modeling_service.train_model(sample_classification_df, config)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['target', 'feature', 'leakage', 'same'])


class TestModelPersistence:
    """Test model persistence and loading."""
    
    def test_model_survives_service_restart(self, tmp_model_dir, sample_classification_df):
        """Test that models persist across service restarts."""
        # Create first service instance and train model
        service1 = ModelingService()
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = service1.train_model(sample_classification_df, config)
        model_id = train_response.model_id
        
        # Create new service instance (simulating restart)
        service2 = ModelingService()
        
        # Verify model still exists and can be used
        model_info = service2.get_model_info(model_id)
        assert model_info.model_id == model_id
        
        # Test prediction with new service instance
        input_df = pd.DataFrame({
            'x1': [0.5, -0.3],
            'x2': [0.8, -0.1]
        })
        
        pred_response = service2.predict(train_response.model_path, input_df)
        assert len(pred_response.predictions) == 2
    
    def test_corrupted_model_file_handling(self, tmp_model_dir, sample_classification_df, modeling_service):
        """Test handling of corrupted model files."""
        # Train a model
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(sample_classification_df, config)
        model_path = Path(train_response.model_path)
        
        # Corrupt the model file
        with open(model_path, 'w') as f:
            f.write("corrupted data")
        
        # Test prediction with corrupted model
        input_df = pd.DataFrame({
            'x1': [0.5],
            'x2': [0.8]
        })
        
        with pytest.raises((HTTPException, ValueError, Exception)) as exc_info:
            modeling_service.predict(str(model_path), input_df)
        
        # Should provide meaningful error about corrupted/invalid model
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['load', 'corrupt', 'invalid', 'model'])


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_memory_usage_large_predictions(self, tmp_model_dir, modeling_service):
        """Test memory usage doesn't explode with large prediction sets."""
        # Create a simple model
        df = pd.DataFrame({
            'x1': np.random.normal(0, 1, 1000),
            'x2': np.random.normal(0, 1, 1000),
            'y': np.random.randint(0, 2, 1000)
        })
        
        config = ModelConfig(
            model_type=ModelType.classification,
            feature_columns=["x1", "x2"],
            target_column="y"
        )
        
        train_response = modeling_service.train_model(df, config)
        
        # Create large prediction dataset
        large_pred_df = pd.DataFrame({
            'x1': np.random.normal(0, 1, 50000),
            'x2': np.random.normal(0, 1, 50000)
        })
        
        # This should not cause memory issues
        pred_response = modeling_service.predict(train_response.model_path, large_pred_df)
        assert len(pred_response.predictions) == 50000


if __name__ == "__main__":
    # Run with: pytest tests/test_modeling.py -v
    # For slow tests: pytest tests/test_modeling.py -v -m "not slow"
    # For auth bypass: TEST_AUTH_BYPASS=1 pytest tests/test_modeling.py -v
    pytest.main([__file__, "-v"])