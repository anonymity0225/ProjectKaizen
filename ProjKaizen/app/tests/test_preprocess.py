# Tests for preprocessing logic 

import os
import json
import pytest
import tempfile
import pandas as pd
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.services.preprocessing import DataPreprocessingService, DataPreprocessingError
from app.schemas.preprocess import CleaningConfig, EncodingConfig
from app.core.settings import settings

client = TestClient(app)

TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.csv")
TEST_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.xlsx")
TEST_LARGE_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "large_sample.csv")

# Fixture to create test data if it doesn't exist
@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data files if they don't exist."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create sample CSV if it doesn't exist
    if not os.path.exists(TEST_CSV_PATH):
        sample_data = pd.DataFrame({
            'id': [1, 2, 2, 3, 4, 5],  # Contains duplicate
            'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],  # Contains missing
            'category': ['A', 'B', 'B', 'A', 'C', 'A'],
            'value': [10.5, 20.0, 20.0, None, 40.2, 50.1],  # Contains missing
            'active': [True, False, False, True, True, False]
        })
        sample_data.to_csv(TEST_CSV_PATH, index=False)
    
    # Create sample Excel if it doesn't exist
    if not os.path.exists(TEST_EXCEL_PATH):
        sample_data = pd.DataFrame({
            'product': ['Widget A', 'Widget B', 'Widget C'],
            'price': [19.99, 29.99, 39.99],
            'category': ['electronics', 'home', 'electronics']
        })
        sample_data.to_excel(TEST_EXCEL_PATH, index=False)

@pytest.fixture
def tmp_path(tmp_path):
    """Isolate settings.TEMP_DIR and clean up after tests."""
    # Create a temporary directory for this test session
    original_temp_dir = settings.TEMP_DIR
    test_temp_dir = tmp_path / "preprocess_temp"
    test_temp_dir.mkdir(exist_ok=True)
    
    # Temporarily override settings
    settings.TEMP_DIR = str(test_temp_dir)
    
    yield tmp_path
    
    # Cleanup and restore original settings
    try:
        if test_temp_dir.exists():
            shutil.rmtree(test_temp_dir)
    except Exception:
        pass
    settings.TEMP_DIR = original_temp_dir

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def temp_files():
    """Track temporary files and directories for cleanup."""
    files = []
    dirs = []
    yield files, dirs
    # Cleanup after test
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
    for dir_path in dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception:
            pass

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'category': ['A', 'B', 'B', 'A', 'C', 'A'],
        'value': [10.5, 20.0, 20.0, None, 40.2, 50.1],
        'active': [True, False, False, True, True, False]
    })

@pytest.fixture
def preprocessing_service():
    """Create a preprocessing service instance for testing."""
    return DataPreprocessingService()

# Unit tests for service functions in isolation
class TestDataPreprocessingService:
    """Unit tests for DataPreprocessingService methods."""
    
    def test_validate_data_valid(self, preprocessing_service, sample_df):
        """Test data validation with valid data."""
        result = preprocessing_service.validate_data(sample_df)
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.issues, dict)
        assert isinstance(result.warnings, list)
        assert result.row_count == 6
        assert result.column_count == 5
        assert result.memory_usage_mb > 0
        assert result.validation_duration > 0
    
    def test_validate_data_with_duplicates(self, preprocessing_service):
        """Test data validation with duplicate rows."""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 2],
            'name': ['Alice', 'Alice', 'Bob', 'Bob', 'Bob']
        })
        
        result = preprocessing_service.validate_data(df)
        
        # Should have duplicate issues
        assert 'duplicate_rows' in result.issues
        assert result.issues['duplicate_rows'] > 0
    
    def test_validate_data_with_missing_values(self, preprocessing_service):
        """Test data validation with missing values."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', None, 'Charlie'],
            'value': [10.5, None, 30.0]
        })
        
        result = preprocessing_service.validate_data(df)
        
        # Should have missing value issues
        assert len(result.issues) > 0
    
    def test_validate_data_exceeds_missing_threshold(self, preprocessing_service):
        """Test data validation with missing values exceeding threshold."""
        # Create DataFrame with > 50% missing values in a column
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', None, None, None, None],  # 80% missing
            'value': [10.5, 20.0, 30.0, 40.0, 50.0]
        })
        
        with pytest.raises(DataPreprocessingError, match="missing value threshold"):
            preprocessing_service.validate_data(df)
    
    def test_clean_data_remove_duplicates(self, preprocessing_service, sample_df):
        """Test data cleaning with duplicate removal."""
        config = CleaningConfig(
            remove_duplicates=True,
            handle_missing_values=False,
            normalize_column_names=False
        )
        
        result = preprocessing_service.clean_data(sample_df, config)
        
        assert len(result.data) < len(sample_df)  # Should have fewer rows
        assert result.original_shape == sample_df.shape
        assert result.final_shape[0] < result.original_shape[0]  # Rows removed
        assert len(result.cleaning_actions) > 0
    
    def test_clean_data_handle_missing_values(self, preprocessing_service, sample_df):
        """Test data cleaning with missing value handling."""
        config = CleaningConfig(
            remove_duplicates=False,
            handle_missing_values=True,
            missing_value_strategy="fill_mean",
            normalize_column_names=False
        )
        
        result = preprocessing_service.clean_data(sample_df, config)
        
        # Should have handled missing values
        assert len(result.cleaning_actions) > 0
        assert result.data['value'].isna().sum() == 0  # No missing values in numeric column
    
    def test_clean_data_normalize_column_names(self, preprocessing_service, sample_df):
        """Test data cleaning with column name normalization."""
        # Create DataFrame with mixed case column names
        df = sample_df.copy()
        df.columns = ['ID', 'Name', 'Category', 'Value', 'Active']
        
        config = CleaningConfig(
            remove_duplicates=False,
            handle_missing_values=False,
            normalize_column_names=True
        )
        
        result = preprocessing_service.clean_data(df, config)
        
        # Column names should be normalized
        expected_columns = ['id', 'name', 'category', 'value', 'active']
        assert list(result.data.columns) == expected_columns
    
    def test_encode_data_label_encoding(self, preprocessing_service, sample_df):
        """Test data encoding with label encoding."""
        config = EncodingConfig(
            categorical_encoding_method="label",
            categorical_columns=["category"],
            scale_numerical=False
        )
        
        result = preprocessing_service.encode_data(sample_df, config)
        
        assert len(result.encoding_actions) > 0
        assert "category" in result.encoders_used
        assert result.data["category"].dtype in ['int32', 'int64']  # Should be numeric
    
    def test_encode_data_onehot_encoding(self, preprocessing_service, sample_df):
        """Test data encoding with one-hot encoding."""
        config = EncodingConfig(
            categorical_encoding_method="onehot",
            categorical_columns=["category"],
            scale_numerical=False,
            drop_first=True
        )
        
        result = preprocessing_service.encode_data(sample_df, config)
        
        assert len(result.encoding_actions) > 0
        assert "category" in result.encoders_used
        # Should have more columns after one-hot encoding
        assert result.final_shape[1] > result.original_shape[1]
    
    def test_encode_data_high_cardinality_categorical(self, preprocessing_service):
        """Test encoding with categorical column exceeding MAX_CATEGORIES_FOR_ONEHOT."""
        # Create DataFrame with high cardinality categorical column
        categories = [f"category_{i}" for i in range(settings.MAX_CATEGORIES_FOR_ONEHOT + 10)]
        df = pd.DataFrame({
            'id': range(len(categories)),
            'high_cardinality_col': categories
        })
        
        config = EncodingConfig(
            categorical_encoding_method="onehot",
            categorical_columns=["high_cardinality_col"],
            scale_numerical=False
        )
        
        with pytest.raises(DataPreprocessingError, match="high cardinality"):
            preprocessing_service.encode_data(df, config)
    
    def test_encode_data_scaling(self, preprocessing_service, sample_df):
        """Test data encoding with numerical scaling."""
        config = EncodingConfig(
            categorical_encoding_method=None,
            scale_numerical=True,
            scaling_method="standard",
            numerical_columns=["value"]
        )
        
        result = preprocessing_service.encode_data(sample_df, config)
        
        assert len(result.encoding_actions) > 0
        assert result.scaler_used is not None
        # Scaled values should have mean close to 0 and std close to 1
        scaled_values = result.data["value"].dropna()
        assert abs(scaled_values.mean()) < 0.1
        assert abs(scaled_values.std() - 1.0) < 0.1
    
    def test_encode_data_invalid_column(self, preprocessing_service, sample_df):
        """Test data encoding with invalid column name."""
        config = EncodingConfig(
            categorical_encoding_method="label",
            categorical_columns=["nonexistent_column"],
            scale_numerical=False
        )
        
        with pytest.raises(DataPreprocessingError):
            preprocessing_service.encode_data(sample_df, config)
    
    def test_clean_data_invalid_config(self, preprocessing_service, sample_df):
        """Test data cleaning with invalid configuration."""
        config = CleaningConfig(
            remove_duplicates=True,
            handle_missing_values=True,
            missing_value_strategy="invalid_strategy"  # Invalid strategy
        )
        
        with pytest.raises(DataPreprocessingError):
            preprocessing_service.clean_data(sample_df, config)

# Edge-case tests for API endpoints
class TestEdgeCases:
    """Edge-case tests for preprocessing endpoints."""
    
    def test_upload_empty_csv(self, client, tmp_path):
        """Test uploading empty CSV file."""
        # Create empty CSV file
        empty_csv_path = tmp_path / "empty.csv"
        empty_csv_path.write_text("")
        
        with open(empty_csv_path, "rb") as f:
            file_data = f.read()
        
        response = client.post(
            "/preprocess/upload",
            files={"file": ("empty.csv", file_data, "text/csv")}
        )
        
        assert response.status_code in [400, 422]
        assert "empty" in response.json()["detail"].lower()
    
    def test_upload_large_dataset(self, client, tmp_path):
        """Test uploading dataset exceeding MAX_ROWS."""
        # Create CSV with more rows than MAX_ROWS
        large_csv_path = tmp_path / "large.csv"
        
        # Generate synthetic data exceeding MAX_ROWS
        rows = settings.MAX_ROWS + 100
        data = {
            'id': range(rows),
            'name': [f'User_{i}' for i in range(rows)],
            'value': [i * 1.5 for i in range(rows)]
        }
        df = pd.DataFrame(data)
        df.to_csv(large_csv_path, index=False)
        
        with open(large_csv_path, "rb") as f:
            file_data = f.read()
        
        response = client.post(
            "/preprocess/upload",
            files={"file": ("large.csv", file_data, "text/csv")}
        )
        
        assert response.status_code == 422
        assert "exceeds maximum" in response.json()["detail"]
    
    def test_upload_unsupported_extension(self, client, tmp_path):
        """Test uploading file with unsupported extension."""
        # Create text file
        text_file_path = tmp_path / "test.txt"
        text_file_path.write_text("This is not a CSV file")
        
        with open(text_file_path, "rb") as f:
            file_data = f.read()
        
        response = client.post(
            "/preprocess/upload",
            files={"file": ("test.txt", file_data, "text/plain")}
        )
        
        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"].lower()
    
    def test_upload_high_missing_threshold(self, client, tmp_path):
        """Test uploading dataset with missing values exceeding threshold."""
        # Create CSV with > 50% missing values
        high_missing_csv_path = tmp_path / "high_missing.csv"
        
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', None, None, None, None],  # 80% missing
            'value': [10.5, 20.0, 30.0, 40.0, 50.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(high_missing_csv_path, index=False)
        
        with open(high_missing_csv_path, "rb") as f:
            file_data = f.read()
        
        response = client.post(
            "/preprocess/upload",
            files={"file": ("high_missing.csv", file_data, "text/csv")}
        )
        
        assert response.status_code == 422
        assert "missing value threshold" in response.json()["detail"]

# Integration tests for API endpoints
@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_run_basic(client, endpoint):
    """Test basic preprocessing with drop duplicates and encoding."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    config = {
        "drop_duplicates": True,
        "handle_missing": None,
        "numeric_columns": None,
        "categorical_columns": None,
        "encoding_method": "label",
        "encoding_columns": ["category"],
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            data={"preprocessing_config": json.dumps(config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            json=config
        )
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Validate response structure
    assert "audit_log" in data
    assert "num_rows" in data
    assert "num_columns" in data
    assert "preview" in data
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0
    assert isinstance(data["preview"], list)
    
    # Validate audit log contents
    audit_log = data["audit_log"]
    assert isinstance(audit_log, list)
    assert len(audit_log) > 0
    
    # Check for expected operations in audit log
    log_messages = [entry.get("message", "") for entry in audit_log]
    has_duplicate_removal = any("duplicate" in msg.lower() for msg in log_messages)
    has_encoding = any("encoding" in msg.lower() or "label" in msg.lower() for msg in log_messages)
    
    assert has_duplicate_removal, f"Expected duplicate removal in audit log: {log_messages}"
    assert has_encoding, f"Expected encoding operation in audit log: {log_messages}"

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_run_full_pipeline(client, endpoint):
    """Test preprocessing with all features: duplicates, missing values, and encoding."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    config = {
        "drop_duplicates": True,
        "handle_missing": "drop",
        "numeric_columns": ["value"],
        "categorical_columns": ["category", "name"],
        "encoding_method": "onehot",
        "encoding_columns": ["category"],
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            data={"preprocessing_config": json.dumps(config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            json=config
        )
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Validate response
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0
    
    # Validate comprehensive audit log
    audit_log = data["audit_log"]
    log_messages = [entry.get("message", "") for entry in audit_log]
    
    # Check for all expected operations
    has_duplicate_removal = any("duplicate" in msg.lower() for msg in log_messages)
    has_missing_handling = any("missing" in msg.lower() or "drop" in msg.lower() for msg in log_messages)
    has_encoding = any("encoding" in msg.lower() or "onehot" in msg.lower() for msg in log_messages)
    
    assert has_duplicate_removal, f"Expected duplicate removal: {log_messages}"
    assert has_missing_handling, f"Expected missing value handling: {log_messages}"
    assert has_encoding, f"Expected encoding: {log_messages}"

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_missing_config(client, endpoint):
    """Test preprocessing with missing configuration."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            # Missing preprocessing_config
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            # Missing JSON body
        )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "preprocessing_config" in str(error_detail).lower() or "validation error" in str(error_detail).lower()

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_invalid_json_config(client, endpoint):
    """Test preprocessing with invalid JSON configuration."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    invalid_config = '{"drop_duplicates": true, "invalid_json":'  # Malformed JSON
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            data={"preprocessing_config": invalid_config}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            # Invalid JSON body
        )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "json" in str(error_detail).lower() or "validation error" in str(error_detail).lower()

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_invalid_config_schema(client, endpoint):
    """Test preprocessing with invalid configuration schema."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    invalid_config = {
        "drop_duplicates": "not_a_boolean",  # Should be boolean
        "encoding_method": "invalid_method",  # Invalid method
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            data={"preprocessing_config": json.dumps(invalid_config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            json=invalid_config
        )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "schema" in str(error_detail).lower() or "validation error" in str(error_detail).lower()

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_bad_file_format(client, endpoint):
    """Test preprocessing with unsupported file format."""
    # Create a fake file with wrong content
    fake_file_content = b"This is not a CSV or Excel file content"
    
    config = {
        "drop_duplicates": True,
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("fake.txt", fake_file_content, "text/plain")},
            data={"preprocessing_config": json.dumps(config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("fake.txt", fake_file_content, "text/plain")},
            json=config
        )
    
    # Should fail due to unsupported file format or parsing error
    assert response.status_code in [422, 415, 500]

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_excel_file(client, endpoint):
    """Test preprocessing with Excel file."""
    # Only run if Excel test file exists
    if not os.path.exists(TEST_EXCEL_PATH):
        pytest.skip("Excel test file not available")
    
    with open(TEST_EXCEL_PATH, "rb") as f:
        file_data = f.read()
    
    config = {
        "drop_duplicates": False,
        "handle_missing": None,
        "encoding_method": "label",
        "encoding_columns": ["category"],
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.xlsx", file_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"preprocessing_config": json.dumps(config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.xlsx", file_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            json=config
        )
    
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0

@pytest.mark.parametrize("endpoint", ["/preprocess/run", "/preprocess/json"])
def test_preprocess_minimal_config(client, endpoint):
    """Test preprocessing with minimal configuration."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    # Minimal config - just validation
    config = {
        "drop_duplicates": False,
        "handle_missing": None,
        "numeric_columns": None,
        "categorical_columns": None,
        "encoding_method": None,
        "encoding_columns": None,
        "validate_first": True
    }
    
    if endpoint == "/preprocess/run":
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            data={"preprocessing_config": json.dumps(config)}
        )
    else:
        response = client.post(
            endpoint,
            files={"file": ("sample.csv", file_data, "text/csv")},
            json=config
        )
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Should still return valid response even with minimal processing
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0
    assert "audit_log" in data
    
    # Audit log should be minimal but present
    audit_log = data["audit_log"]
    assert isinstance(audit_log, list)