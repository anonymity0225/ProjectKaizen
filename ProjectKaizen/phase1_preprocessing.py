import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to check data cleanliness
def check_cleanliness(df):
    """
    Checks the cleanliness of the dataset.

    Parameters:
    - df (pd.DataFrame): Input dataframe

    Returns:
    - dict: Summary of data cleanliness
    """
    logging.info("Checking data cleanliness...")
    cleanliness_report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "outliers": {
            col: ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
                  (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum()
            for col in df.select_dtypes(include=[np.number]).columns
        },
        "string_inconsistencies": [
            col for col in df.select_dtypes(include=["object", "category"]).columns
            if df[col].str.strip().ne(df[col]).any()
        ],
    }
    logging.info("Cleanliness report generated.")
    return cleanliness_report

# Function to check if data is already clean
def is_data_clean(df):
    """
    Checks if the dataset is already clean.

    Parameters:
    - df (pd.DataFrame): Input dataframe

    Returns:
    - bool: True if data is clean, False otherwise
    """
    cleanliness = check_cleanliness(df)
    if all(v == 0 for v in cleanliness["missing_values"].values()) and cleanliness["duplicate_rows"] == 0 and \
       all(count == 0 for count in cleanliness["outliers"].values()) and not cleanliness["string_inconsistencies"]:
        return True
    return False

# Function to handle missing values
def handle_missing_values(df, strategies=None, fill_values=None):
    """
    Handles missing values in the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - strategies (dict): Column-wise strategy for numeric and categorical columns
      Example: {"numeric_col": "mean", "categorical_col": "mode"}
    - fill_values (dict): Column-wise constants for the 'constant' strategy

    Returns:
    - pd.DataFrame: Dataframe with missing values handled
    """
    logging.info("Handling missing values...")
    strategies = strategies or {}
    fill_values = fill_values or {}

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in strategies:
                strategy = strategies[col]
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "constant" and col in fill_values:
                    df[col] = df[col].fillna(fill_values[col])
                elif strategy == "drop_rows":
                    df = df.dropna(subset=[col])
    logging.info("Missing values handled.")
    return df

# Function to standardize column names
def standardize_column_names(df):
    """
    Standardizes column names by converting to lowercase and replacing spaces with underscores.
    """
    logging.info("Standardizing column names...")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    logging.info("Column names standardized.")
    return df

# Function to handle outliers
def handle_outliers(df, methods=None, thresholds=None, log_changes=False):
    """
    Handles outliers in numeric columns.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - methods (dict): Column-wise outlier detection method ('iqr', 'zscore')
    - thresholds (dict): Column-wise thresholds for outlier detection
    - log_changes (bool): Whether to log outlier adjustments

    Returns:
    - pd.DataFrame: Dataframe with outliers handled
    - dict: Outlier log (if log_changes=True)
    """
    logging.info("Handling outliers...")
    methods = methods or {}
    thresholds = thresholds or {}
    outlier_log = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        method = methods.get(col, "iqr")
        threshold = thresholds.get(col, 3.0)

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if log_changes:
                outlier_log[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outliers_count": ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                }
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            if log_changes:
                outlier_log[col] = {
                    "threshold": threshold,
                    "outliers_count": ((z_scores < -threshold) | (z_scores > threshold)).sum()
                }
            df[col] = np.where(z_scores < -threshold, mean - threshold * std, df[col])
            df[col] = np.where(z_scores > threshold, mean + threshold * std, df[col])

    logging.info("Outliers handled.")
    if log_changes:
        return df, outlier_log
    return df

# Function to normalize numeric columns
def normalize_numeric_columns(df, include_columns=None):
    """
    Normalizes numeric columns using StandardScaler.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - include_columns (list): List of numeric columns to normalize (default: all numeric columns)

    Returns:
    - pd.DataFrame: Dataframe with numeric columns normalized
    """
    logging.info("Normalizing numeric columns...")
    numeric_columns = include_columns or df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) == 0:
        logging.warning("No numeric columns found for normalization.")
        return df

    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    logging.info("Numeric columns normalized.")
    return df

# Custom User Code Execution
def execute_custom_code(df, custom_code):
    """
    Allows users to execute custom Python code on the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - custom_code (str): User-provided Python code to execute

    Returns:
    - pd.DataFrame: Modified dataframe
    """
    logging.info("Executing custom user code...")
    try:
        exec(custom_code, {'df': df, 'pd': pd, 'np': np})
        logging.info("Custom code executed successfully.")
    except Exception as e:
        logging.error(f"Error executing custom code: {e}")
        raise e
    return df

# Main function for Phase 1 preprocessing
def execute_phase_1_cleaning(
    df,
    missing_value_strategies=None,
    fill_values=None,
    outlier_methods=None,
    outlier_thresholds=None,
    outlier_log=False,
    include_scaling_columns=None,
    custom_code=None
):
    """
    Executes Phase 1 cleaning pipeline.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - missing_value_strategies (dict): Column-wise strategies for missing values
    - fill_values (dict): Constants for 'constant' strategy
    - outlier_methods (dict): Column-wise outlier detection methods
    - outlier_thresholds (dict): Column-wise outlier thresholds
    - outlier_log (bool): Whether to log outlier adjustments
    - include_scaling_columns (list): Columns to include for scaling
    - custom_code (str): User-provided code for custom data manipulations

    Returns:
    - pd.DataFrame: Cleaned dataframe
    """
    try:
        logging.info("Starting Phase 1 Cleaning...")

        if is_data_clean(df):
            logging.info("Data is already clean. Skipping preprocessing steps.")
            return df

        # Step 1: Check cleanliness
        cleanliness_report = check_cleanliness(df)
        logging.info("Cleanliness Report: %s", cleanliness_report)

        # Step 2: Handle missing values
        df_cleaned = handle_missing_values(df, strategies=missing_value_strategies, fill_values=fill_values)
        logging.info("Data after handling missing values: \n%s", df_cleaned.head())

        # Step 3: Standardize column names
        df_cleaned = standardize_column_names(df_cleaned)
        logging.info("Data after standardizing column names: \n%s", df_cleaned.head())

        # Step 4: Handle outliers
        if outlier_log:
            df_cleaned, log = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds, log_changes=True)
            logging.info("Outlier log: %s", log)
        else:
            df_cleaned = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds)
        logging.info("Data after handling outliers: \n%s", df_cleaned.head())

        # Step 5: Normalize numeric columns
        df_cleaned = normalize_numeric_columns(df_cleaned, include_columns=include_scaling_columns)
        logging.info("Data after normalization: \n%s", df_cleaned.head())

        # Step 6: Execute custom user code (if provided)
        if custom_code:
            df_cleaned = execute_custom_code(df_cleaned, custom_code)

        logging.info("Phase 1 Cleaning completed successfully.")
        return df_cleaned

    except Exception as e:
        logging.error("An error occurred during preprocessing: %s", e)
        raise e
