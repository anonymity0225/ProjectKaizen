import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Main function for Phase 1 preprocessing
def execute_phase_1_cleaning(
    df,
    missing_value_strategies=None,
    fill_values=None,
    outlier_methods=None,
    outlier_thresholds=None,
    outlier_log=False,
    include_scaling_columns=None,
    custom_code=None,
    handle_outliers_flag=True  # Added parameter to control outlier handling
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
    - handle_outliers_flag (bool): Flag to enable or disable outlier handling

    Returns:
    - pd.DataFrame: Cleaned dataframe
    """
    try:
        logging.info("Starting Phase 1 Cleaning...")

        # Step 1: Handle missing values
        df_cleaned = handle_missing_values(df, strategies=missing_value_strategies, fill_values=fill_values)
        logging.info("Data after handling missing values: \n%s", df_cleaned.head())

        # Step 2: Standardize column names
        df_cleaned = standardize_column_names(df_cleaned)
        logging.info("Data after standardizing column names: \n%s", df_cleaned.head())

        # Step 3: Handle outliers (only if handle_outliers_flag is True)
        if handle_outliers_flag:
            if outlier_log:
                df_cleaned, log = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds, log_changes=True)
                logging.info("Outlier log: %s", log)
            else:
                df_cleaned = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds)
        else:
            logging.info("Skipping outlier handling as per user request.")

        # Step 4: Normalize numeric columns
        df_cleaned = normalize_numeric_columns(df_cleaned, include_columns=include_scaling_columns)
        logging.info("Data after normalization: \n%s", df_cleaned.head())

        # Step 5: Execute custom user code (if provided)
        if custom_code:
            df_cleaned = execute_custom_code(df_cleaned, custom_code)

        logging.info("Phase 1 Cleaning completed successfully.")
        return df_cleaned

    except Exception as e:
        logging.error("An error occurred during preprocessing: %s", e)
        raise e
