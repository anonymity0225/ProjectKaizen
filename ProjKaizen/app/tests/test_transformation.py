import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import time

from app.services.transformation import (
    extract_date_components,
    scale_numeric,
    encode_categorical,
    tokenize_text,
    apply_tfidf_vectorization,
    apply_pca,
    apply_custom_user_function,
    apply_multiple_transformations,
    get_transformation_summary,
    clear_transformation_history,
    TransformationError
)
from app.schemas.transformation import (
    DateExtractionRequest, DateExtractionResponse,
    ScalingRequest, ScalingResponse,
    CategoricalEncodingRequest, CategoricalEncodingResponse,
    TextPreprocessingRequest, TextPreprocessingResponse,
    TFIDFRequest, TFIDFResponse,
    PCATransformRequest, PCATransformResponse,
    BatchTransformRequest, BatchTransformResponse,
    HistorySummaryResponse,
    BaseTransformationResponse
)
from app.api import app


# Mock JWT verification for tests
def mock_verify_token():
    """Mock JWT verification that always passes."""
    return {"user_id": "test_user", "email": "test@example.com"}


@pytest.fixture(autouse=True)
def mock_auth():
    """Automatically mock JWT verification for all tests."""
    with patch('app.dependencies.verify_token', return_value=mock_verify_token()):
        yield


# Fixtures
@pytest.fixture
def sample_df():
    """Sample DataFrame for testing transformations."""
    return pd.DataFrame({
        "date_col": ["2020-01-01", "2021-06-15", None],
        "num_col": [1.0, 2.5, None],
        "cat_col": ["A", "B", "A"],
        "text_col": ["Hello world", "Test case", ""]
    })


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Valid JWT headers for protected routes."""
    return {"Authorization": "Bearer valid_jwt_token"}


@pytest.fixture
def numeric_df():
    """DataFrame with numeric columns for scaling tests."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feature3": [0.1, 0.2, 0.3, 0.4, 0.5]
    })


@pytest.fixture
def categorical_df():
    """DataFrame with categorical columns for encoding tests."""
    return pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue"],
        "size": ["small", "medium", "large", "small", "medium"],
        "category": ["A", "B", "A", "C", "B"]
    })


@pytest.fixture
def text_df():
    """DataFrame with text columns for text processing tests."""
    return pd.DataFrame({
        "text": [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is fascinating",
            "Natural language processing is amazing"
        ]
    })


# Service-Level Tests
class TestDateExtraction:
    """Test cases for date component extraction."""
    
    def test_extract_date_components_success(self, sample_df):
        """Test successful date component extraction."""
        df, metadata = extract_date_components(sample_df, "date_col", ["year", "month", "day", "weekday"])
        
        assert "date_col_year" in df.columns
        assert "date_col_month" in df.columns
        assert "date_col_day" in df.columns
        assert "date_col_weekday" in df.columns
        assert metadata["status"] == "success"
        assert metadata["method"] == "date_component_extraction"
        assert metadata["added_columns"] == ["date_col_year", "date_col_month", "date_col_day", "date_col_weekday"]
        assert metadata["transformed_columns"] == ["date_col"]
        
        # Check values for non-null dates
        assert df.loc[0, "date_col_year"] == 2020
        assert df.loc[0, "date_col_month"] == 1
        assert df.loc[0, "date_col_day"] == 1
        assert df.loc[1, "date_col_year"] == 2021
        assert df.loc[1, "date_col_month"] == 6
        assert df.loc[1, "date_col_day"] == 15
        
        # Check null handling
        assert pd.isna(df.loc[2, "date_col_year"])
    
    def test_extract_date_components_invalid_column(self, sample_df):
        """Test date extraction with invalid column."""
        with pytest.raises(TransformationError) as exc_info:
            extract_date_components(sample_df, "invalid_column", ["year"])
        assert "not found in DataFrame" in str(exc_info.value)
    
    def test_extract_date_components_invalid_component(self, sample_df):
        """Test date extraction with invalid component."""
        with pytest.raises(TransformationError) as exc_info:
            extract_date_components(sample_df, "date_col", ["invalid_component"])
        assert "Invalid components" in str(exc_info.value)
    
    def test_extract_date_components_empty_dataframe(self):
        """Test date extraction with empty DataFrame."""
        empty_df = pd.DataFrame({"date_col": []})
        with pytest.raises(TransformationError) as exc_info:
            extract_date_components(empty_df, "date_col", ["year"])
        assert "cannot be empty" in str(exc_info.value)


class TestNumericScaling:
    """Test cases for numeric feature scaling."""
    
    @pytest.mark.parametrize("method", ["minmax", "zscore", "log", "custom"])
    def test_scale_numeric_methods(self, numeric_df, method):
        """Test different scaling methods."""
        kwargs = {}
        if method == "custom":
            kwargs["custom_range"] = (0, 10)
        elif method == "log":
            # Ensure positive values for log scaling
            numeric_df["feature1"] = numeric_df["feature1"] + 1
        
        df, metadata = scale_numeric(numeric_df, "feature1", method, **kwargs)
        
        assert metadata["status"] == "success"
        assert metadata["method"] == f"numeric_scaling_{method}"
        assert metadata["transformed_columns"] == ["feature1"]
        
        scaled_values = df["feature1"].values
        if method == "minmax":
            assert scaled_values.min() == pytest.approx(0.0, rel=1e-10)
            assert scaled_values.max() == pytest.approx(1.0, rel=1e-10)
        elif method == "zscore":
            # Z-score should have mean close to 0 and std close to 1
            assert abs(scaled_values.mean()) < 1e-10
            assert abs(scaled_values.std() - 1.0) < 1e-10
        elif method == "custom":
            assert scaled_values.min() == pytest.approx(0.0, rel=1e-10)
            assert scaled_values.max() == pytest.approx(10.0, rel=1e-10)
        elif method == "log":
            # All values should be positive after log transformation
            assert all(val > 0 for val in scaled_values if not pd.isna(val))
    
    def test_scale_numeric_multiple_columns(self, numeric_df):
        """Test scaling multiple columns."""
        df, metadata = scale_numeric(numeric_df, ["feature1", "feature2"], "minmax")
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "numeric_scaling_minmax"
        assert set(metadata["transformed_columns"]) == {"feature1", "feature2"}
        
        for col in ["feature1", "feature2"]:
            scaled_values = df[col].values
            assert scaled_values.min() == pytest.approx(0.0, rel=1e-10)
            assert scaled_values.max() == pytest.approx(1.0, rel=1e-10)
    
    def test_scale_numeric_invalid_column(self, numeric_df):
        """Test scaling with invalid column."""
        with pytest.raises(TransformationError) as exc_info:
            scale_numeric(numeric_df, "invalid_column", "minmax")
        assert "not found in DataFrame" in str(exc_info.value)
    
    def test_scale_numeric_non_numeric_column(self, sample_df):
        """Test scaling non-numeric column."""
        with pytest.raises(TransformationError) as exc_info:
            scale_numeric(sample_df, "cat_col", "minmax")
        assert "not numeric" in str(exc_info.value)
    
    def test_scale_numeric_log_negative_values(self, numeric_df):
        """Test log scaling with negative values."""
        numeric_df["feature1"] = [-1, 0, 1, 2, 3]
        with pytest.raises(TransformationError) as exc_info:
            scale_numeric(numeric_df, "feature1", "log")
        assert "requires all values to be positive" in str(exc_info.value)


class TestCategoricalEncoding:
    """Test cases for categorical feature encoding."""
    
    @pytest.mark.parametrize("method", ["onehot", "label", "frequency"])
    def test_encode_categorical_methods(self, categorical_df, method):
        """Test different encoding methods."""
        df, metadata = encode_categorical(categorical_df, "color", method)
        
        assert metadata["status"] == "success"
        assert metadata["method"] == f"categorical_encoding_{method}"
        assert metadata["transformed_columns"] == ["color"]
        
        if method == "onehot":
            # Check that one-hot columns were created
            onehot_cols = [col for col in df.columns if col.startswith("color_")]
            assert len(onehot_cols) > 0
            assert len(metadata["added_columns"]) > 0
            # Check that original column was dropped
            assert "color" not in df.columns
        elif method == "label":
            # Check that values are numeric
            assert df["color"].dtype in [np.int64, np.int32]
            # Check that unique values are consecutive starting from 0
            unique_values = sorted(df["color"].unique())
            assert unique_values == list(range(len(unique_values)))
        elif method == "frequency":
            # Check that values are numeric (frequencies)
            assert df["color"].dtype in [np.float64, np.float32]
            # Check that frequencies are between 0 and 1
            assert all(0 <= val <= 1 for val in df["color"])
    
    def test_encode_categorical_multiple_columns(self, categorical_df):
        """Test encoding multiple columns."""
        df, metadata = encode_categorical(categorical_df, ["color", "size"], "label")
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "categorical_encoding_label"
        assert set(metadata["transformed_columns"]) == {"color", "size"}
        
        for col in ["color", "size"]:
            assert df[col].dtype in [np.int64, np.int32]
    
    def test_encode_categorical_custom(self, categorical_df):
        """Test custom encoding."""
        custom_mapping = {"red": 1, "blue": 2, "green": 3}
        df, metadata = encode_categorical(categorical_df, "color", "custom", custom_mapping)
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "categorical_encoding_custom"
        assert metadata["transformed_columns"] == ["color"]
        # Check that custom mapping was applied
        assert set(df["color"].unique()) == set(custom_mapping.values())
    
    def test_encode_categorical_invalid_column(self, categorical_df):
        """Test encoding with invalid column."""
        with pytest.raises(TransformationError) as exc_info:
            encode_categorical(categorical_df, "invalid_column", "onehot")
        assert "not found in DataFrame" in str(exc_info.value)
    
    def test_encode_categorical_custom_without_mapping(self, categorical_df):
        """Test custom encoding without mapping."""
        with pytest.raises(TransformationError) as exc_info:
            encode_categorical(categorical_df, "color", "custom")
        assert "Custom mapping required" in str(exc_info.value)


class TestTextPreprocessing:
    """Test cases for text preprocessing."""
    
    @pytest.mark.parametrize("method", ["stemming", "lemmatization", "none"])
    def test_tokenize_text_methods(self, text_df, method):
        """Test different text processing methods."""
        df, metadata = tokenize_text(text_df, "text", method)
        
        assert metadata["status"] == "success"
        assert metadata["method"] == f"text_preprocessing_{method}"
        assert metadata["transformed_columns"] == ["text"]
        
        # Check that text was processed
        assert all(isinstance(text, str) for text in df["text"])
        
        if method in ["stemming", "lemmatization"]:
            # Check that text processing occurred (should be shorter or different)
            original_lengths = [len(text.split()) for text in text_df["text"]]
            processed_lengths = [len(text.split()) for text in df["text"]]
            # At least some processing should have occurred
            assert any(proc <= orig for proc, orig in zip(processed_lengths, original_lengths))
        elif method == "none":
            # Text should remain unchanged
            assert all(df["text"] == text_df["text"])
    
    def test_tokenize_text_invalid_column(self, text_df):
        """Test text processing with invalid column."""
        with pytest.raises(TransformationError) as exc_info:
            tokenize_text(text_df, "invalid_column", "stemming")
        assert "not found in DataFrame" in str(exc_info.value)


class TestTFIDFVectorization:
    """Test cases for TF-IDF vectorization."""
    
    def test_apply_tfidf_vectorization(self, text_df):
        """Test TF-IDF vectorization."""
        df, metadata = apply_tfidf_vectorization(text_df, "text", max_features=10)
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "tfidf_vectorization"
        assert metadata["transformed_columns"] == ["text"]
        # Check that TF-IDF columns were created
        tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
        assert len(tfidf_cols) <= 10
        assert len(tfidf_cols) > 0
        assert len(metadata["added_columns"]) > 0
        # Check that original text column was dropped
        assert "text" not in df.columns
    
    def test_apply_tfidf_vectorization_empty_strings(self, sample_df):
        """Test TF-IDF with empty strings."""
        df, metadata = apply_tfidf_vectorization(sample_df, "text_col", max_features=5)
        
        assert metadata["status"] == "success"
        # Should handle empty strings gracefully
        tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
        assert len(tfidf_cols) > 0
    
    def test_apply_tfidf_vectorization_high_max_features(self, text_df):
        """Test TF-IDF with too high max_features."""
        # This should work but with fewer features than requested
        df, metadata = apply_tfidf_vectorization(text_df, "text", max_features=1000)
        
        assert metadata["status"] == "success"
        tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
        # Should create features but not exceed vocabulary size
        assert len(tfidf_cols) > 0
        assert len(tfidf_cols) <= 1000
    
    def test_apply_tfidf_vectorization_invalid_column(self, text_df):
        """Test TF-IDF with invalid column."""
        with pytest.raises(TransformationError) as exc_info:
            apply_tfidf_vectorization(text_df, "invalid_column", max_features=10)
        assert "not found in DataFrame" in str(exc_info.value)


class TestPCATransformation:
    """Test cases for PCA transformation."""
    
    def test_apply_pca_single_component(self, numeric_df):
        """Test PCA with single component."""
        df, metadata = apply_pca(numeric_df, n_components=1, columns=["feature1", "feature2"])
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "pca_transformation"
        assert metadata["n_components"] == 1
        assert set(metadata["transformed_columns"]) == {"feature1", "feature2"}
        # Check that PCA column was created
        pca_cols = [col for col in df.columns if col.startswith("PC")]
        assert len(pca_cols) == 1
        assert "PC1" in df.columns
        assert len(metadata["added_columns"]) == 1
    
    def test_apply_pca_multiple_components(self, numeric_df):
        """Test PCA with multiple components."""
        df, metadata = apply_pca(numeric_df, n_components=2, columns=["feature1", "feature2", "feature3"])
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "pca_transformation"
        assert metadata["n_components"] == 2
        # Check that PCA columns were created
        pca_cols = [col for col in df.columns if col.startswith("PC")]
        assert len(pca_cols) == 2
        assert "PC1" in df.columns
        assert "PC2" in df.columns
        assert len(metadata["added_columns"]) == 2
    
    def test_apply_pca_too_many_components(self, numeric_df):
        """Test PCA with more components than features."""
        with pytest.raises(TransformationError) as exc_info:
            apply_pca(numeric_df, n_components=10, columns=["feature1", "feature2"])
        assert "n_components cannot exceed" in str(exc_info.value)
    
    def test_apply_pca_invalid_columns(self, numeric_df):
        """Test PCA with invalid columns."""
        with pytest.raises(TransformationError) as exc_info:
            apply_pca(numeric_df, n_components=1, columns=["invalid_column"])
        assert "not found in DataFrame" in str(exc_info.value)


class TestCustomUserFunction:
    """Test cases for custom user function application."""
    
    def test_apply_custom_user_function_success(self, sample_df):
        """Test custom user function application."""
        def sample_func(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["constant_col"] = 42
            return df
        
        df, metadata = apply_custom_user_function(sample_df, sample_func, "add_constant")
        
        assert metadata["status"] == "success"
        assert metadata["method"] == "custom_function"
        assert metadata["function_name"] == "add_constant"
        assert "constant_col" in df.columns
        assert all(df["constant_col"] == 42)
    
    def test_apply_custom_user_function_with_error(self, sample_df):
        """Test custom function that raises an error."""
        def error_func(df: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("Test error")
        
        with pytest.raises(TransformationError) as exc_info:
            apply_custom_user_function(sample_df, error_func, "error_func")
        assert "Test error" in str(exc_info.value)
    
    def test_apply_custom_user_function_unsafe_code(self, sample_df):
        """Test handling of potentially unsafe custom function."""
        def potentially_unsafe_func(df: pd.DataFrame) -> pd.DataFrame:
            # This function is safe but could be used to test safety measures
            df = df.copy()
            df["processed"] = "safe"
            return df
        
        df, metadata = apply_custom_user_function(sample_df, potentially_unsafe_func, "safe_func")
        assert metadata["status"] == "success"
        assert "processed" in df.columns
        assert all(df["processed"] == "safe")


class TestBatchTransformations:
    """Test cases for batch transformations."""
    
    def test_apply_multiple_transformations_success(self, sample_df):
        """Test applying multiple transformations."""
        transformations = [
            {"action": "extract_date_components", "params": {"column": "date_col", "components": ["year", "month"]}},
            {"action": "encode_categorical", "params": {"column": "cat_col", "method": "label"}},
            {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}}
        ]
        
        df, metas = apply_multiple_transformations(sample_df, transformations)
        
        # Check that all transformations were applied
        assert len(metas) == 3
        assert all(meta["status"] == "success" for meta in metas)
        
        # Check that transformations were applied correctly
        assert "date_col_year" in df.columns
        assert "date_col_month" in df.columns
        assert df["cat_col"].dtype in [np.int64, np.int32]  # Should be encoded
        # Check that scaling was applied (values should be between 0 and 1)
        non_null_values = df["num_col"].dropna()
        if len(non_null_values) > 0:
            assert non_null_values.min() >= 0
            assert non_null_values.max() <= 1
    
    def test_apply_multiple_transformations_function_format(self, sample_df):
        """Test batch transformations with function/args format."""
        transformations = [
            {"function": "scale_numeric", "args": {"column": "num_col", "method": "minmax"}},
            {"function": "encode_categorical", "args": {"column": "cat_col", "method": "label"}}
        ]
        
        df, metas = apply_multiple_transformations(sample_df, transformations)
        
        assert len(metas) == 2
        assert all(meta["status"] == "success" for meta in metas)
        assert df["cat_col"].dtype in [np.int64, np.int32]
    
    def test_apply_multiple_transformations_with_error(self, sample_df):
        """Test batch transformations with one failing."""
        transformations = [
            {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}},
            {"action": "scale_numeric", "params": {"column": "invalid_column", "method": "minmax"}}
        ]
        
        df, metas = apply_multiple_transformations(sample_df, transformations)
        
        # Check that first transformation succeeded and second failed
        assert len(metas) == 2
        assert metas[0]["status"] == "success"
        assert metas[1]["status"] == "error"
    
    def test_apply_multiple_transformations_invalid_action(self, sample_df):
        """Test batch transformations with invalid action."""
        transformations = [
            {"action": "invalid_action", "params": {"column": "num_col"}}
        ]
        
        df, metas = apply_multiple_transformations(sample_df, transformations)
        
        assert len(metas) == 1
        assert metas[0]["status"] == "error"
        assert "Unknown transformation action" in metas[0]["error"]


class TestHistoryAndClear:
    """Test cases for transformation history and clearing."""
    
    def test_history_tracking(self, sample_df):
        """Test transformation history tracking."""
        # Clear history first
        clear_transformation_history()
        assert get_transformation_summary()["total_operations"] == 0
        
        # Run one transformation
        extract_date_components(sample_df, "date_col", ["year"])
        summary = get_transformation_summary()
        assert summary["total_operations"] == 1
        assert len(summary["operations"]) == 1
        assert "date_component_extraction" in summary["operations"][0]["method"]
        
        # Run another transformation
        scale_numeric(sample_df, "num_col", "minmax")
        summary = get_transformation_summary()
        assert summary["total_operations"] == 2
        assert len(summary["operations"]) == 2
    
    def test_clear_transformation_history(self, sample_df):
        """Test clearing transformation history."""
        # Run some transformations
        extract_date_components(sample_df, "date_col", ["year"])
        scale_numeric(sample_df, "num_col", "minmax")
        
        # Verify history exists
        summary = get_transformation_summary()
        assert summary["total_operations"] > 0
        
        # Clear history
        clear_transformation_history()
        summary = get_transformation_summary()
        assert summary["total_operations"] == 0
        assert len(summary["operations"]) == 0


# API-Level Tests
class TestTransformationAPI:
    """Test cases for transformation API endpoints."""
    
    def test_date_extraction_endpoint_success(self, client, auth_headers):
        """Test date extraction endpoint with valid payload."""
        payload = {
            "data": [
                {"date_col": "2020-01-01", "num_col": 1.0},
                {"date_col": "2021-06-15", "num_col": 2.5}
            ],
            "column": "date_col",
            "components": ["year", "month", "day"]
        }
        
        response = client.post("/transform/date", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "metadata" in data
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "date_component_extraction"
        assert "date_col_year" in data["data"][0]
        assert "date_col_month" in data["data"][0]
        assert "date_col_day" in data["data"][0]
    
    def test_date_extraction_endpoint_invalid_payload(self, client, auth_headers):
        """Test date extraction endpoint with invalid payload."""
        payload = {
            "data": [{"date_col": "2020-01-01"}],
            # Missing required fields
        }
        
        response = client.post("/transform/date", json=payload, headers=auth_headers)
        assert response.status_code == 422
    
    def test_date_extraction_endpoint_unauthorized(self, client):
        """Test date extraction endpoint without authorization."""
        payload = {
            "data": [{"date_col": "2020-01-01"}],
            "column": "date_col",
            "components": ["year"]
        }
        
        with patch('app.dependencies.verify_token', side_effect=Exception("Unauthorized")):
            response = client.post("/transform/date", json=payload)
            assert response.status_code == 401
    
    @pytest.mark.parametrize("method", ["minmax", "zscore", "log"])
    def test_scale_endpoint_methods(self, client, auth_headers, method):
        """Test scaling endpoint with different methods."""
        payload = {
            "data": [
                {"num_col": 1.0, "other": "a"},
                {"num_col": 2.0, "other": "b"},
                {"num_col": 3.0, "other": "c"}
            ],
            "column": "num_col",
            "method": method
        }
        
        if method == "custom":
            payload["custom_range"] = [0, 10]
        
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == f"numeric_scaling_{method}"
    
    def test_scale_endpoint_multiple_columns(self, client, auth_headers):
        """Test scaling endpoint with multiple columns."""
        payload = {
            "data": [
                {"num_col1": 1.0, "num_col2": 10.0},
                {"num_col1": 2.0, "num_col2": 20.0},
                {"num_col1": 3.0, "num_col2": 30.0}
            ],
            "column": ["num_col1", "num_col2"],
            "method": "minmax"
        }
        
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert set(data["metadata"]["transformed_columns"]) == {"num_col1", "num_col2"}
    
    def test_encode_endpoint_success(self, client, auth_headers):
        """Test categorical encoding endpoint."""
        payload = {
            "data": [
                {"cat_col": "A", "num_col": 1},
                {"cat_col": "B", "num_col": 2},
                {"cat_col": "A", "num_col": 3}
            ],
            "column": "cat_col",
            "method": "label"
        }
        
        response = client.post("/transform/encode", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "categorical_encoding_label"
    
    def test_encode_endpoint_multiple_columns(self, client, auth_headers):
        """Test encoding endpoint with multiple columns."""
        payload = {
            "data": [
                {"cat_col1": "A", "cat_col2": "X"},
                {"cat_col1": "B", "cat_col2": "Y"},
                {"cat_col1": "A", "cat_col2": "X"}
            ],
            "column": ["cat_col1", "cat_col2"],
            "method": "label"
        }
        
        response = client.post("/transform/encode", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert set(data["metadata"]["transformed_columns"]) == {"cat_col1", "cat_col2"}
    
    def test_text_preprocessing_endpoint_success(self, client, auth_headers):
        """Test text preprocessing endpoint."""
        payload = {
            "data": [
                {"text_col": "Hello world", "id": 1},
                {"text_col": "Test case", "id": 2}
            ],
            "column": "text_col",
            "method": "stemming"
        }
        
        response = client.post("/transform/text", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "text_preprocessing_stemming"
    
    def test_tfidf_endpoint_success(self, client, auth_headers):
        """Test TF-IDF vectorization endpoint."""
        payload = {
            "data": [
                {"text_col": "Hello world test", "id": 1},
                {"text_col": "Test case example", "id": 2},
                {"text_col": "Another text sample", "id": 3}
            ],
            "column": "text_col",
            "max_features": 5
        }
        
        response = client.post("/transform/tfidf", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "tfidf_vectorization"
        # Check that TF-IDF columns were created
        tfidf_cols = [col for col in data["data"][0].keys() if col.startswith("tfidf_")]
        assert len(tfidf_cols) > 0
        assert len(tfidf_cols) <= 5
    
    def test_pca_endpoint_success(self, client, auth_headers):
        """Test PCA transformation endpoint."""
        payload = {
            "data": [
                {"feature1": 1.0, "feature2": 10.0, "id": 1},
                {"feature1": 2.0, "feature2": 20.0, "id": 2},
                {"feature1": 3.0, "feature2": 30.0, "id": 3}
            ],
            "n_components": 1,
            "columns": ["feature1", "feature2"]
        }
        
        response = client.post("/transform/pca", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "pca_transformation"
        assert data["metadata"]["n_components"] == 1
        assert "PC1" in data["data"][0]
    
    def test_pca_endpoint_invalid_components(self, client, auth_headers):
        """Test PCA endpoint with too many components."""
        payload = {
            "data": [
                {"feature1": 1.0, "feature2": 10.0},
                {"feature1": 2.0, "feature2": 20.0}
            ],
            "n_components": 5,  # More than available features
            "columns": ["feature1", "feature2"]
        }
        
        response = client.post("/transform/pca", json=payload, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "n_components cannot exceed" in data["detail"]
    
    def test_custom_function_endpoint_success(self, client, auth_headers):
        """Test custom function endpoint."""
        payload = {
            "data": [
                {"num_col": 1.0, "id": 1},
                {"num_col": 2.0, "id": 2}
            ],
            "function_code": "def transform(df): df = df.copy(); df['doubled'] = df['num_col'] * 2; return df",
            "function_name": "double_values"
        }
        
        response = client.post("/transform/custom", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["method"] == "custom_function"
        assert data["metadata"]["function_name"] == "double_values"
        assert "doubled" in data["data"][0]
        assert data["data"][0]["doubled"] == 2.0
    
    def test_custom_function_endpoint_unsafe_code(self, client, auth_headers):
        """Test custom function endpoint with potentially unsafe code."""
        payload = {
            "data": [
                {"num_col": 1.0, "id": 1}
            ],
            "function_code": "import os; def transform(df): os.system('echo test'); return df",
            "function_name": "unsafe_function"
        }
        
        response = client.post("/transform/custom", json=payload, headers=auth_headers)
        
        # Should either reject unsafe code or handle it safely
        assert response.status_code in [400, 500]
    
    def test_batch_transformation_endpoint_success(self, client, auth_headers):
        """Test batch transformation endpoint."""
        payload = {
            "data": [
                {"date_col": "2020-01-01", "num_col": 1.0, "cat_col": "A"},
                {"date_col": "2021-06-15", "num_col": 2.5, "cat_col": "B"}
            ],
            "transformations": [
                {"action": "extract_date_components", "params": {"column": "date_col", "components": ["year"]}},
                {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}}
            ]
        }
        
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "metadata" in data
        assert len(data["metadata"]) == 2  # Two transformations
        assert all(meta["status"] == "success" for meta in data["metadata"])
        assert "date_col_year" in data["data"][0]
    
    def test_batch_transformation_function_format(self, client, auth_headers):
        """Test batch transformation with function/args format."""
        payload = {
            "data": [
                {"num_col": 1.0, "cat_col": "A"},
                {"num_col": 2.0, "cat_col": "B"}
            ],
            "transformations": [
                {"function": "scale_numeric", "args": {"column": "num_col", "method": "minmax"}},
                {"function": "encode_categorical", "args": {"column": "cat_col", "method": "label"}}
            ]
        }
        
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["metadata"]) == 2
        assert all(meta["status"] == "success" for meta in data["metadata"])
    
    def test_batch_transformation_unknown_action(self, client, auth_headers):
        """Test batch transformation with unknown action."""
        payload = {
            "data": [
                {"num_col": 1.0},
                {"num_col": 2.0}
            ],
            "transformations": [
                {"action": "unknown_transformation", "params": {"column": "num_col"}}
            ]
        }
        
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["metadata"]) == 1
        assert data["metadata"][0]["status"] == "error"
        assert "Unknown transformation action" in data["metadata"][0]["error"]
    
    def test_get_history_endpoint_success(self, client, auth_headers):
        """Test get transformation history endpoint."""
        # First, perform some transformations to populate history
        payload = {
            "data": [{"num_col": 1.0}, {"num_col": 2.0}],
            "column": "num_col",
            "method": "minmax"
        }
        client.post("/transform/scale", json=payload, headers=auth_headers)
        
        response = client.get("/transform/history", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_operations" in data
        assert "operations" in data
        assert data["total_operations"] > 0
        assert len(data["operations"]) > 0
    
    def test_clear_history_endpoint_success(self, client, auth_headers):
        """Test clear transformation history endpoint."""
        # First, perform a transformation to populate history
        payload = {
            "data": [{"num_col": 1.0}, {"num_col": 2.0}],
            "column": "num_col",
            "method": "minmax"
        }
        client.post("/transform/scale", json=payload, headers=auth_headers)
        
        # Verify history exists
        history_response = client.get("/transform/history", headers=auth_headers)
        assert history_response.json()["total_operations"] > 0
        
        # Clear history
        response = client.delete("/transform/history", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        
        # Verify history is empty
        history_response = client.get("/transform/history", headers=auth_headers)
        assert history_response.json()["total_operations"] == 0
    
    def test_endpoints_with_invalid_auth(self, client):
        """Test endpoints with invalid authorization."""
        payload = {"data": [{"col": "value"}]}
        
        endpoints = [
            ("/transform/date", "POST"),
            ("/transform/scale", "POST"),
            ("/transform/encode", "POST"),
            ("/transform/text", "POST"),
            ("/transform/tfidf", "POST"),
            ("/transform/pca", "POST"),
            ("/transform/custom", "POST"),
            ("/transform/batch", "POST"),
            ("/transform/history", "GET"),
            ("/transform/history", "DELETE")
        ]
        
        for endpoint, method in endpoints:
            with patch('app.dependencies.verify_token', side_effect=Exception("Unauthorized")):
                if method == "POST":
                    response = client.post(endpoint, json=payload)
                elif method == "GET":
                    response = client.get(endpoint)
                elif method == "DELETE":
                    response = client.delete(endpoint)
                
                assert response.status_code == 401


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions."""
    
    def test_single_row_dataframe_transformations(self, client, auth_headers):
        """Test transformations with single row DataFrame."""
        # Date extraction
        payload = {
            "data": [{"date_col": "2020-01-01", "num_col": 5.0, "cat_col": "A"}],
            "column": "date_col",
            "components": ["year", "month"]
        }
        response = client.post("/transform/date", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        
        # Scaling with single value
        payload = {
            "data": [{"num_col": 5.0}],
            "column": "num_col",
            "method": "minmax"
        }
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        assert response.status_code == 200
        # Single value should be scaled to middle of range (0.5)
        data = response.json()
        assert data["data"][0]["num_col"] == pytest.approx(0.5, rel=1e-10)
    
    def test_empty_dataframe_handling(self, client, auth_headers):
        """Test handling of empty DataFrame."""
        payload = {
            "data": [],
            "column": "date_col",
            "components": ["year"]
        }
        
        response = client.post("/transform/date", json=payload, headers=auth_headers)
        assert response.status_code in [400, 422]
    
    def test_missing_values_handling(self, client, auth_headers):
        """Test handling of missing values in transformations."""
        # Date extraction with null values
        payload = {
            "data": [
                {"date_col": "2020-01-01", "num_col": 1.0},
                {"date_col": None, "num_col": 2.0},
                {"date_col": "2021-12-31", "num_col": None}
            ],
            "column": "date_col",
            "components": ["year", "month"]
        }
        
        response = client.post("/transform/date", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        # Check that null handling worked properly
        assert data["data"][1]["date_col_year"] is None
        assert data["data"][1]["date_col_month"] is None
    
    def test_large_dataset_handling(self, client, auth_headers):
        """Test handling of larger datasets."""
        # Create a dataset with 100 rows
        large_data = []
        for i in range(100):
            large_data.append({
                "date_col": f"2020-{(i % 12) + 1:02d}-01",
                "num_col": float(i),
                "cat_col": ["A", "B", "C"][i % 3],
                "text_col": f"Sample text {i}"
            })
        
        payload = {
            "data": large_data,
            "column": "num_col",
            "method": "minmax"
        }
        
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert len(data["data"]) == 100
    
    def test_special_characters_in_text(self, client, auth_headers):
        """Test text processing with special characters."""
        payload = {
            "data": [
                {"text_col": "Hello! How are you? I'm fine.", "id": 1},
                {"text_col": "Special chars: @#$%^&*()", "id": 2},
                {"text_col": "Unicode: café, naïve, résumé", "id": 3},
                {"text_col": "", "id": 4}  # Empty string
            ],
            "column": "text_col",
            "method": "stemming"
        }
        
        response = client.post("/transform/text", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        # Should handle special characters gracefully
        assert len(data["data"]) == 4
    
    def test_numeric_edge_cases(self, client, auth_headers):
        """Test numeric transformations with edge cases."""
        payload = {
            "data": [
                {"num_col": 0.0, "id": 1},
                {"num_col": -1.0, "id": 2},
                {"num_col": 1000000.0, "id": 3},
                {"num_col": 0.000001, "id": 4}
            ],
            "column": "num_col",
            "method": "minmax"
        }
        
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        
        # Check that scaling worked correctly with extreme values
        scaled_values = [row["num_col"] for row in data["data"]]
        assert min(scaled_values) == pytest.approx(0.0, rel=1e-10)
        assert max(scaled_values) == pytest.approx(1.0, rel=1e-10)
    
    def test_categorical_edge_cases(self, client, auth_headers):
        """Test categorical encoding with edge cases."""
        payload = {
            "data": [
                {"cat_col": "A", "id": 1},
                {"cat_col": "A", "id": 2},  # Duplicate
                {"cat_col": "B", "id": 3},
                {"cat_col": "C_with_underscore", "id": 4},  # Special characters
                {"cat_col": "123", "id": 5},  # Numeric string
                {"cat_col": "", "id": 6}  # Empty string
            ],
            "column": "cat_col",
            "method": "label"
        }
        
        response = client.post("/transform/encode", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        
        # Check that all categories were encoded properly
        encoded_values = [row["cat_col"] for row in data["data"]]
        assert all(isinstance(val, int) for val in encoded_values)


class TestPerformance:
    """Test cases for performance with large datasets."""
    
    def test_large_dataset_performance(self, client, auth_headers):
        """Test transformation performance with large dataset (~10k rows)."""
        # Create a large dataset
        large_data = []
        for i in range(10000):
            large_data.append({
                "num_col": float(i % 1000),
                "cat_col": ["A", "B", "C", "D", "E"][i % 5],
                "text_col": f"Sample text number {i}",
                "id": i
            })
        
        # Test numeric scaling performance
        payload = {
            "data": large_data,
            "column": "num_col",
            "method": "minmax"
        }
        
        start_time = time.time()
        response = client.post("/transform/scale", json=payload, headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "success"
        assert len(data["data"]) == 10000
        
        # Should complete within reasonable time (2 seconds)
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Transformation took {execution_time:.2f} seconds, expected < 2.0"
    
    def test_batch_transformation_performance(self, client, auth_headers):
        """Test batch transformation performance."""
        # Create moderate-sized dataset for batch operations
        data_size = 5000
        large_data = []
        for i in range(data_size):
            large_data.append({
                "date_col": f"202{i % 3}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "num_col": float(i % 100),
                "cat_col": ["Type_A", "Type_B", "Type_C"][i % 3],
                "id": i
            })
        
        payload = {
            "data": large_data,
            "transformations": [
                {"action": "extract_date_components", "params": {"column": "date_col", "components": ["year", "month"]}},
                {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}},
                {"action": "encode_categorical", "params": {"column": "cat_col", "method": "label"}}
            ]
        }
        
        start_time = time.time()
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["metadata"]) == 3
        assert all(meta["status"] == "success" for meta in data["metadata"])
        assert len(data["data"]) == data_size
        
        # Should complete within reasonable time (3 seconds for batch)
        execution_time = end_time - start_time
        assert execution_time < 3.0, f"Batch transformation took {execution_time:.2f} seconds, expected < 3.0"


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_complete_data_pipeline(self, client, auth_headers):
        """Test a complete data transformation pipeline."""
        # Step 1: Extract date components
        initial_data = [
            {"date_col": "2020-01-15", "num_col": 100.0, "cat_col": "Type_A", "text_col": "Product review positive"},
            {"date_col": "2020-02-20", "num_col": 250.0, "cat_col": "Type_B", "text_col": "Service quality excellent"},
            {"date_col": "2020-03-10", "num_col": 150.0, "cat_col": "Type_A", "text_col": "Customer satisfaction high"}
        ]
        
        # Use batch transformation for the complete pipeline
        payload = {
            "data": initial_data,
            "transformations": [
                {"action": "extract_date_components", "params": {"column": "date_col", "components": ["year", "month", "day"]}},
                {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}},
                {"action": "encode_categorical", "params": {"column": "cat_col", "method": "label"}},
                {"action": "tokenize_text", "params": {"column": "text_col", "method": "stemming"}}
            ]
        }
        
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        # Verify all transformations were applied successfully
        assert len(data["metadata"]) == 4
        assert all(meta["status"] == "success" for meta in data["metadata"])
        
        # Verify the final data structure
        final_data = data["data"][0]
        assert "date_col_year" in final_data
        assert "date_col_month" in final_data
        assert "date_col_day" in final_data
        assert isinstance(final_data["cat_col"], int)
        assert 0 <= final_data["num_col"] <= 1  # Should be scaled
        assert isinstance(final_data["text_col"], str)
    
    def test_error_recovery_in_pipeline(self, client, auth_headers):
        """Test error handling and recovery in transformation pipeline."""
        payload = {
            "data": [
                {"date_col": "2020-01-01", "num_col": 1.0, "cat_col": "A"},
                {"date_col": "2020-02-01", "num_col": 2.0, "cat_col": "B"}
            ],
            "transformations": [
                {"action": "extract_date_components", "params": {"column": "date_col", "components": ["year"]}},
                {"action": "scale_numeric", "params": {"column": "nonexistent_col", "method": "minmax"}},  # This will fail
                {"action": "encode_categorical", "params": {"column": "cat_col", "method": "label"}}
            ]
        }
        
        response = client.post("/transform/batch", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        # Check that first and third transformations succeeded, second failed
        assert len(data["metadata"]) == 3
        assert data["metadata"][0]["status"] == "success"
        assert data["metadata"][1]["status"] == "error"
        assert data["metadata"][2]["status"] == "success"
        
        # The final data should still have successful transformations applied
        final_data = data["data"][0]
        assert "date_col_year" in final_data
        assert isinstance(final_data["cat_col"], int)
    
    def test_performance_with_repeated_operations(self, client, auth_headers):
        """Test performance and consistency with repeated operations."""
        base_data = [
            {"num_col": 1.0, "id": 1},
            {"num_col": 2.0, "id": 2},
            {"num_col": 3.0, "id": 3}
        ]
        
        # Perform the same transformation multiple times
        results = []
        for _ in range(5):
            payload = {
                "data": base_data.copy(),
                "column": "num_col",
                "method": "minmax"
            }
            response = client.post("/transform/scale", json=payload, headers=auth_headers)
            assert response.status_code == 200
            results.append(response.json())
        
        # Results should be consistent across runs
        first_result = results[0]["data"]
        for result in results[1:]:
            for i, row in enumerate(result["data"]):
                assert row["num_col"] == pytest.approx(first_result[i]["num_col"], rel=1e-10)
        
        # Check history accumulation
        history_response = client.get("/transform/history", headers=auth_headers)
        history_data = history_response.json()
        assert history_data["total_operations"] >= 5


class TestSchemaValidation:
    """Test cases for request/response schema validation."""
    
    def test_date_extraction_schema_validation(self, client, auth_headers):
        """Test date extraction request schema validation."""
        # Valid request
        valid_payload = {
            "data": [{"date_col": "2020-01-01"}],
            "column": "date_col",
            "components": ["year", "month"]
        }
        response = client.post("/transform/date", json=valid_payload, headers=auth_headers)
        assert response.status_code == 200
        
        # Invalid request - missing required field
        invalid_payload = {
            "data": [{"date_col": "2020-01-01"}],
            "column": "date_col"
            # Missing components
        }
        response = client.post("/transform/date", json=invalid_payload, headers=auth_headers)
        assert response.status_code == 422
    
    def test_scaling_schema_validation(self, client, auth_headers):
        """Test scaling request schema validation."""
        # Valid request
        valid_payload = {
            "data": [{"num_col": 1.0}],
            "column": "num_col",
            "method": "minmax"
        }
        response = client.post("/transform/scale", json=valid_payload, headers=auth_headers)
        assert response.status_code == 200
        
        # Invalid method
        invalid_payload = {
            "data": [{"num_col": 1.0}],
            "column": "num_col",
            "method": "invalid_method"
        }
        response = client.post("/transform/scale", json=invalid_payload, headers=auth_headers)
        assert response.status_code == 422
    
    def test_batch_transformation_schema_validation(self, client, auth_headers):
        """Test batch transformation request schema validation."""
        # Valid request with action/params format
        valid_payload = {
            "data": [{"num_col": 1.0}],
            "transformations": [
                {"action": "scale_numeric", "params": {"column": "num_col", "method": "minmax"}}
            ]
        }
        response = client.post("/transform/batch", json=valid_payload, headers=auth_headers)
        assert response.status_code == 200
        
        # Valid request with function/args format
        valid_payload_2 = {
            "data": [{"num_col": 1.0}],
            "transformations": [
                {"function": "scale_numeric", "args": {"column": "num_col", "method": "minmax"}}
            ]
        }
        response = client.post("/transform/batch", json=valid_payload_2, headers=auth_headers)
        assert response.status_code == 200
        
        # Invalid request - malformed transformation
        invalid_payload = {
            "data": [{"num_col": 1.0}],
            "transformations": [
                {"invalid_key": "scale_numeric"}
            ]
        }
        response = client.post("/transform/batch", json=invalid_payload, headers=auth_headers)
        assert response.status_code == 422


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up transformation history after each test."""
    yield
    # Clean up after test
    try:
        clear_transformation_history()
    except:
        pass  # Ignore cleanup errors