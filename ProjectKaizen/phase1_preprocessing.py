import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- Utility Functions ------------------- #
def generate_cleanliness_report(df):
    """
    Generates a cleanliness report for a dataset.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        dict: A report summarizing missing values, duplicates, and data types.
    """
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().mean() * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    return report

def standardize_column_names(df):
    """Standardize column names by stripping, lowering case, and replacing spaces with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=True)
    return df

def handle_missing_values(df, strategies=None, fill_values=None):
    """Handle missing values based on user-specified strategies."""
    strategies = strategies or {}
    fill_values = fill_values or {}

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            strategy = strategies.get(col, "none")
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
                df[col] = df[col].round()  # Round the column values
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
                df[col] = df[col].round()  # Round the column values
            elif strategy == "mode":
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
            elif strategy == "constant" and col in fill_values:
                df[col].fillna(fill_values[col], inplace=True)
            elif strategy == "drop_rows":
                df.dropna(subset=[col], inplace=True)
    return df

def remove_duplicates(df, keep="first"):
    """Remove duplicate rows from the dataset."""
    df.drop_duplicates(keep=keep, inplace=True)
    return df

def detect_outliers(df, methods=None, thresholds=None):
    """Detect outliers based on user-specified methods and thresholds."""
    outlier_log = {}
    methods = methods or {}
    thresholds = thresholds or {}

    for col, method in methods.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        threshold = thresholds.get(col, 3.0)
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outlier_mask = (z_scores < -threshold) | (z_scores > threshold)
        else:
            continue

        outlier_log[col] = {
            "outlier_count": outlier_mask.sum(),
            "outlier_mask": outlier_mask
        }
    return outlier_log

def handle_outliers(df, outlier_log, strategy="replace"):
    """Handle outliers based on user-specified strategy."""
    for col, outlier_info in outlier_log.items():
        outlier_mask = outlier_info["outlier_mask"]
        if strategy == "replace":
            df.loc[outlier_mask, col] = np.nan
        elif strategy == "clip":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        # Ensure handled values are rounded for numerical consistency
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round()
    return df

def preprocess_data(df, actions):
    """Preprocess data based on user-selected actions."""
    if actions.get("standardize_column_names"):
        df = standardize_column_names(df)

    if actions.get("handle_missing_values"):
        df = handle_missing_values(
            df,
            strategies=actions.get("missing_value_strategies"),
            fill_values=actions.get("fill_values")
        )

    if actions.get("remove_duplicates"):
        df = remove_duplicates(df, keep=actions.get("duplicates_keep", "first"))

    return df

def delete_rows_or_columns(df, delete_rows=None, delete_columns=None):
    """
    Delete specified rows or columns from the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        delete_rows (list): List of row indices to delete.
        delete_columns (list): List of column names to delete.

    Returns:
        pd.DataFrame: Modified dataframe after deletion.
    """
    if delete_rows:
        df = df.drop(delete_rows, axis=0)
    if delete_columns:
        df = df.drop(delete_columns, axis=1)
    return df

def execute_custom_code(df, custom_code):
    """
    Executes user-provided custom Python code on the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        custom_code (str): User-provided Python code as a string.

    Returns:
        pd.DataFrame: Modified dataframe after executing custom code.
    """
    local_vars = {"df": df}
    try:
        exec(custom_code, {}, local_vars)
        df = local_vars.get("df", df)  # Retrieve updated dataframe
    except Exception as e:
        logging.error(f"Error executing custom code: {e}")
    return df
