# Pytest fixtures and configuration 
import pytest
import pandas as pd
import numpy as np

# Simple fixtures for transformation tests
@pytest.fixture
def sample_df():
    """Sample DataFrame for testing transformations."""
    return pd.DataFrame({
        "date": ["2020-01-01", "2021-12-31", "2022-06-15"],
        "num": [1, 9, 5],
        "cat": ["a", "b", "a"],
        "text": ["Running runs", "Cats cat", "Dogs dog"],
    }) 