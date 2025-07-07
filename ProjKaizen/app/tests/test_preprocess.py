# Tests for preprocessing logic 

import os
import json
import pytest
import tempfile
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.csv")
TEST_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.xlsx")

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
    """Track temporary files for cleanup."""
    files = []
    yield files
    # Cleanup after test
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

def test_preprocess_run_basic():
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
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        data={"preprocessing_config": json.dumps(config)}
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

def test_preprocess_run_full_pipeline():
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
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        data={"preprocessing_config": json.dumps(config)}
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

def test_preprocess_missing_config():
    """Test preprocessing with missing configuration."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        # Missing preprocessing_config
    )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "preprocessing_config" in str(error_detail).lower()

def test_preprocess_invalid_json_config():
    """Test preprocessing with invalid JSON configuration."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    invalid_config = '{"drop_duplicates": true, "invalid_json":'  # Malformed JSON
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        data={"preprocessing_config": invalid_config}
    )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "json" in str(error_detail).lower()

def test_preprocess_invalid_config_schema():
    """Test preprocessing with invalid configuration schema."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    invalid_config = {
        "drop_duplicates": "not_a_boolean",  # Should be boolean
        "encoding_method": "invalid_method",  # Invalid method
        "validate_first": True
    }
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        data={"preprocessing_config": json.dumps(invalid_config)}
    )
    
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert "schema" in str(error_detail).lower()

def test_preprocess_bad_file_format():
    """Test preprocessing with unsupported file format."""
    # Create a fake file with wrong content
    fake_file_content = b"This is not a CSV or Excel file content"
    
    config = {
        "drop_duplicates": True,
        "validate_first": True
    }
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("fake.txt", fake_file_content, "text/plain")},
        data={"preprocessing_config": json.dumps(config)}
    )
    
    # Should fail due to unsupported file format or parsing error
    assert response.status_code in [422, 500]

def test_preprocess_excel_file():
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
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.xlsx", file_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"preprocessing_config": json.dumps(config)}
    )
    
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0

def test_preprocess_minimal_config():
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
    
    response = client.post(
        "/preprocess/run",
        files={"file": ("sample.csv", file_data, "text/csv")},
        data={"preprocessing_config": json.dumps(config)}
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

def test_preprocess_json_endpoint():
    """Test alternative JSON endpoint if implemented."""
    with open(TEST_CSV_PATH, "rb") as f:
        file_data = f.read()
    
    config = {
        "drop_duplicates": True,
        "handle_missing": "drop",
        "encoding_method": "label",
        "encoding_columns": ["category"],
        "validate_first": True
    }
    
    # Test the JSON endpoint
    response = client.post(
        "/preprocess/run-json",
        files={"file": ("sample.csv", file_data, "text/csv")},
        json=config  # JSON in body instead of form data
    )
    
    # This endpoint might not exist yet, so handle both cases
    if response.status_code == 404:
        pytest.skip("JSON endpoint not implemented yet")
    
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["num_rows"] > 0
    assert data["num_columns"] > 0