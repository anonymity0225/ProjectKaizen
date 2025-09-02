#tests/test_visualization.py
# Tests for preprocessing logic 

import os
import json
import pytest
import tempfile
import pandas as pd
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.preprocessing import DataPreprocessingService, DataPreprocessingError
from app.schemas.preprocess import CleaningConfig, EncodingConfig, FileUploadResponse, PreprocessingResponse
from app.core.settings import settings

# Test data paths
TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.csv")
TEST_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.xlsx")
TEST_LARGE_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "large_sample.csv")

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Automatically set testing environment and temporary directories."""
    # Set testing environment
    monkeypatch.setenv("ENVIRONMENT", "testing")
    
    # Create temporary preprocessing directory
    test_preprocess_dir = tmp_path / "preprocessing_temp"
    test_preprocess_dir.mkdir(exist_ok=True)
    
    # Override settings
    monkeypatch.setattr(settings, "TEMP_DIR", str(test_preprocess_dir))
    monkeypatch.setattr(settings, "PREPROCESSING_TEMP_DIR", str(test_preprocess_dir))
    
    # Create models directories
    encoders_dir = test_preprocess_dir / "models" / "encoders"
    scalers_dir = test_preprocess_dir / "models" / "scalers"
    encoders_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)
    
    yield test_preprocess_dir
    
    # Cleanup after test
    try:
        if test_preprocess_dir.exists():
            shutil.rmtree(test_preprocess_dir)
    except Exception:
        pass

@pytest.fixture
def client():
    """Test client fixture using FastAPI TestClient."""
    return TestClient(app)

@pytest.fixture
def csv_generator():
    """Utility to generate CSVs of arbitrary size and missingness."""
    def _generate_csv(rows: int, cols: int = 5, missing_rate: float = 0.0, filepath: str = None):
        """
        Generate CSV with specified dimensions and missing data rate.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            missing_rate: Fraction of values to make missing (0.0 to 1.0)
            filepath: Optional file path to save CSV
            
        Returns:
            pandas.DataFrame or filepath if provided
        """
        import numpy as np
        import random
        
        # Generate base data
        data = {}
        for i in range(cols):
            if i == 0:
                data['id'] = list(range(1, rows + 1))
            elif i == 1:
                data['name'] = [f'User_{j}' for j in range(rows)]
            elif i == 2:
                data['category'] = [random.choice(['A', 'B', 'C', 'D']) for _ in range(rows)]
            else:
                data[f'value_{i}'] = [random.uniform(0, 100) for _ in range(rows)]
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        if missing_rate > 0:
            total_cells = rows * cols
            missing_count = int(total_cells * missing_rate)
            
            for _ in range(missing_count):
                row_idx = random.randint(0, rows - 1)
                col_idx = random.randint(0, cols - 1)
                col_name = df.columns[col_idx]
                df.loc[row_idx, col_name] = None
        
        if filepath:
            df.to_csv(filepath, index=False)
            return filepath
        
        return df
    
    return _generate_csv

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

# Must-Have Tests
class TestUploadStreaming:
    """Test upload streaming for memory-safe handling of large files."""
    
    def test_upload_large_csv_streaming(self, client, csv_generator, tmp_path):
        """Upload a ~60MB CSV using TestClient streaming; assert 200."""
        # Generate large CSV (~60MB)
        large_csv_path = tmp_path / "large_streaming_test.csv"
        # Approximately 60MB with 500k rows, 10 columns
        csv_generator(rows=500000, cols=10, filepath=str(large_csv_path))
        
        # Verify file size is approximately 60MB
        file_size = os.path.getsize(large_csv_path)
        assert file_size > 50 * 1024 * 1024  # At least 50MB
        
        with open(large_csv_path, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("large_streaming_test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Confirm file_size in response equals OS file size
        assert data["file_size"] == file_size
        
        # Validate response matches FileUploadResponse schema
        assert "file_id" in data
        assert "filename" in data
        assert "file_size" in data
        assert "upload_timestamp" in data
        assert "validation_report" in data

class TestSizeAndTypeEnforcement:
    """Test size and type enforcement."""
    
    def test_oversized_file_returns_413(self, client, csv_generator, tmp_path, monkeypatch):
        """Oversized file → expect 413."""
        # Set a low max file size for testing
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)  # 1MB limit
        
        # Generate file larger than limit
        large_csv_path = tmp_path / "oversized.csv"
        csv_generator(rows=100000, cols=5, filepath=str(large_csv_path))
        
        with open(large_csv_path, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("oversized.csv", f, "text/csv")}
            )
        
        assert response.status_code == 413
        assert "file size" in response.json()["detail"].lower()
    
    def test_unsupported_extension_returns_415(self, client, tmp_path):
        """Unsupported extension/MIME mismatch → expect 415."""
        # Create text file with unsupported extension
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is not a CSV or Excel file")
        
        with open(txt_file, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"].lower()
    
    def test_mime_type_mismatch_returns_415(self, client, csv_generator, tmp_path):
        """MIME type mismatch → expect 415."""
        csv_path = tmp_path / "test.csv"
        csv_generator(rows=100, cols=3, filepath=str(csv_path))
        
        with open(csv_path, "rb") as f:
            # Send CSV with wrong MIME type
            response = client.post(
                "/preprocess/upload",
                files={"file": ("test.csv", f, "application/pdf")}
            )
        
        assert response.status_code == 415

class TestRowColumnLimitsValidation:
    """Test row/column limits and validation."""
    
    def test_dataframe_exceeding_max_rows_returns_422(self, client, csv_generator, tmp_path, monkeypatch):
        """DataFrame exceeding MAX_ROWS triggers 422 from validation."""
        # Set low MAX_ROWS for testing
        monkeypatch.setattr(settings, "MAX_ROWS", 1000)
        
        # Generate CSV with more rows than limit
        large_csv_path = tmp_path / "too_many_rows.csv"
        csv_generator(rows=1500, cols=3, filepath=str(large_csv_path))
        
        with open(large_csv_path, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("too_many_rows.csv", f, "text/csv")}
            )
        
        assert response.status_code == 422
        assert "exceeds maximum" in response.json()["detail"]
    
    def test_dataframe_exceeding_max_columns_returns_422(self, client, csv_generator, tmp_path, monkeypatch):
        """DataFrame exceeding MAX_COLUMNS triggers 422 from validation."""
        # Set low MAX_COLUMNS for testing
        monkeypatch.setattr(settings, "MAX_COLUMNS", 5)
        
        # Generate CSV with more columns than limit
        wide_csv_path = tmp_path / "too_many_columns.csv"
        csv_generator(rows=100, cols=10, filepath=str(wide_csv_path))
        
        with open(wide_csv_path, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("too_many_columns.csv", f, "text/csv")}
            )
        
        assert response.status_code == 422
        assert "exceeds maximum" in response.json()["detail"]
    
    def test_excessive_missing_values_returns_422(self, client, csv_generator, tmp_path):
        """Excessive missing values returns 422 with errors in ValidationReport."""
        # Generate CSV with high missing rate
        high_missing_csv_path = tmp_path / "high_missing.csv"
        csv_generator(rows=1000, cols=5, missing_rate=0.7, filepath=str(high_missing_csv_path))
        
        with open(high_missing_csv_path, "rb") as f:
            response = client.post(
                "/preprocess/upload",
                files={"file": ("high_missing.csv", f, "text/csv")}
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "missing value threshold" in data["detail"]

class TestTimeout:
    """Test timeout handling."""
    
    def test_processing_timeout_returns_504(self, client, monkeypatch):
        """Processing timeout returns 504 with clear message."""
        # Set very short timeout
        monkeypatch.setattr(settings, "PROCESSING_TIMEOUT_SECONDS", 1)
        
        # Mock service to sleep longer than timeout
        original_service = DataPreprocessingService
        
        class SlowPreprocessingService(DataPreprocessingService):
            def validate_data(self, df):
                time.sleep(2)  # Sleep longer than timeout
                return super().validate_data(df)
        
        with patch('app.services.preprocessing.DataPreprocessingService', SlowPreprocessingService):
            with open(TEST_CSV_PATH, "rb") as f:
                response = client.post(
                    "/preprocess/upload",
                    files={"file": ("sample.csv", f, "text/csv")}
                )
        
        assert response.status_code == 504
        assert "timeout" in response.json()["detail"].lower()

class TestEncoderPersistence:
    """Test encoder persistence and reloading."""
    
    def test_encoder_artifacts_created_and_persisted(self, client, setup_test_environment):
        """First /run with encoding → artifacts created under models/encoders/ and /scalers/."""
        config = {
            "drop_duplicates": False,
            "handle_missing": None,
            "encoding_method": "label",
            "encoding_columns": ["category"],
            "scale_numerical": True,
            "numerical_columns": ["value"],
            "validate_first": True
        }
        
        with open(TEST_CSV_PATH, "rb") as f:
            response = client.post(
                "/preprocess/run",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert response.status_code == 200
        
        # Check that encoder and scaler artifacts were created
        encoders_dir = setup_test_environment / "models" / "encoders"
        scalers_dir = setup_test_environment / "models" / "scalers"
        
        # Should have at least one encoder file
        encoder_files = list(encoders_dir.glob("*.joblib"))
        assert len(encoder_files) > 0
        
        # Should have at least one scaler file
        scaler_files = list(scalers_dir.glob("*.joblib"))
        assert len(scaler_files) > 0
    
    def test_persisted_artifacts_loaded_on_subsequent_runs(self, client, setup_test_environment):
        """Second /run with same schema/config loads persisted artifacts."""
        config = {
            "drop_duplicates": False,
            "handle_missing": None,
            "encoding_method": "label",
            "encoding_columns": ["category"],
            "validate_first": True
        }
        
        # First run - creates artifacts
        with open(TEST_CSV_PATH, "rb") as f:
            response1 = client.post(
                "/preprocess/run",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert response1.status_code == 200
        
        # Get creation time of encoder files
        encoders_dir = setup_test_environment / "models" / "encoders"
        encoder_files = list(encoders_dir.glob("*.joblib"))
        creation_times = {f: f.stat().st_mtime for f in encoder_files}
        
        # Small delay to ensure timestamp difference
        time.sleep(0.1)
        
        # Second run - should load existing artifacts
        with open(TEST_CSV_PATH, "rb") as f:
            response2 = client.post(
                "/preprocess/run",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert response2.status_code == 200
        
        # Verify timestamps didn't change (artifacts were loaded, not recreated)
        current_times = {f: f.stat().st_mtime for f in encoder_files}
        for file_path in creation_times:
            assert creation_times[file_path] == current_times[file_path]
    
    def test_unseen_category_handled_gracefully(self, client, csv_generator, tmp_path):
        """Unseen category handled per config without error; assert audit log entry."""
        # Create training data
        train_csv = tmp_path / "train.csv"
        train_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40]
        })
        train_df.to_csv(train_csv, index=False)
        
        config = {
            "drop_duplicates": False,
            "handle_missing": None,
            "encoding_method": "label",
            "encoding_columns": ["category"],
            "validate_first": True
        }
        
        # First run with training data
        with open(train_csv, "rb") as f:
            response1 = client.post(
                "/preprocess/run",
                files={"file": ("train.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert response1.status_code == 200
        
        # Create test data with unseen category
        test_csv = tmp_path / "test.csv"
        test_df = pd.DataFrame({
            'id': [5, 6, 7],
            'category': ['A', 'C', 'B'],  # 'C' is unseen
            'value': [50, 60, 70]
        })
        test_df.to_csv(test_csv, index=False)
        
        # Second run with unseen category
        with open(test_csv, "rb") as f:
            response2 = client.post(
                "/preprocess/run",
                files={"file": ("test.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert response2.status_code == 200
        
        # Check audit log for unseen category handling
        audit_log = response2.json()["audit_log"]
        log_messages = [entry.get("message", "").lower() for entry in audit_log]
        has_unseen_category_message = any("unseen" in msg or "unknown" in msg for msg in log_messages)
        assert has_unseen_category_message

class TestCleanup:
    """Test cleanup of temporary files."""
    
    def test_temp_files_cleaned_after_upload_and_run(self, client, setup_test_environment):
        """After /upload and /run, verify temp files are deleted."""
        initial_files = set(setup_test_environment.rglob("*"))
        
        # Upload file
        with open(TEST_CSV_PATH, "rb") as f:
            upload_response = client.post(
                "/preprocess/upload",
                files={"file": ("sample.csv", f, "text/csv")}
            )
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Run preprocessing
        config = {
            "drop_duplicates": True,
            "handle_missing": None,
            "encoding_method": "label",
            "encoding_columns": ["category"],
            "validate_first": True
        }
        
        with open(TEST_CSV_PATH, "rb") as f:
            run_response = client.post(
                "/preprocess/run",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"preprocessing_config": json.dumps(config)}
            )
        
        assert run_response.status_code == 200
        
        # Check that temp files are cleaned (excluding persistent model artifacts)
        final_files = set(setup_test_environment.rglob("*"))
        
        # Should only have model artifacts remaining, no temporary processing files
        temp_files = [f for f in final_files - initial_files 
                     if not str(f).endswith('.joblib') and f.is_file()]
        
        assert len(temp_files) == 0, f"Temporary files not cleaned: {temp_files}"

class TestConcurrency:
    """Test concurrent operations."""
    
    def test_parallel_uploads_and_runs_succeed(self, client):
        """Fire N=5 parallel uploads/runs; all succeed; no cross-pollution."""
        
        config = {
            "drop_duplicates": True,
            "handle_missing": None,
            "encoding_method": "label",
            "encoding_columns": ["category"],
            "validate_first": True
        }
        
        def upload_and_run(thread_id):
            """Single upload and run operation."""
            try:
                # Run preprocessing
                with open(TEST_CSV_PATH, "rb") as f:
                    run_response = client.post(
                        "/preprocess/run",
                        files={"file": (f"sample_{thread_id}.csv", f, "text/csv")},
                        data={"preprocessing_config": json.dumps(config)}
                    )
                
                if run_response.status_code != 200:
                    return False, f"Run failed for thread {thread_id}: {run_response.status_code}"
                
                # Verify