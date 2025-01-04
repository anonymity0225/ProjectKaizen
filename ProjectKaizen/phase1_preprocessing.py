import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def handle_missing_values(df, strategies=None, fill_values=None):
    """
    Handles missing values in the dataframe.
    """
    logging.info("Handling missing values...")
    strategies = strategies or {}
    fill_values = fill_values or {}
    
    for col, strategy in strategies.items():
        if strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == "constant":
            df[col].fillna(fill_values.get(col, ""), inplace=True)
        elif strategy == "drop_rows":
            df.dropna(subset=[col], inplace=True)

    return df


def standardize_column_names(df):
    """
    Standardizes column names to lowercase and replaces spaces with underscores.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def normalize_numeric_columns(df, include_columns=None):
    """
    Normalizes numeric columns using StandardScaler.
    """
    logging.info("Normalizing numeric columns...")
    if include_columns:
        scaler = StandardScaler()
        df[include_columns] = scaler.fit_transform(df[include_columns])
    return df


def handle_outliers(df, methods=None, thresholds=None, log_changes=False):
    """
    Handles outliers in numeric columns.
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


def execute_custom_code(df, custom_code):
    """
    Executes custom user-provided code on the dataframe.
    """
    logging.info("Executing custom user code...")
    try:
        local_vars = {"df": df}
        exec(custom_code, {}, local_vars)
        return local_vars["df"]
    except Exception as e:
        logging.error("Error executing custom code: %s", e)
        raise e


def check_cleanliness(df):
    """
    Checks the cleanliness of the dataframe and returns a report.
    """
    logging.info("Checking dataset cleanliness...")
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "outliers_detected": {
            col: {"iqr": _detect_outliers_iqr(df[col]), "zscore": _detect_outliers_zscore(df[col])}
            for col in df.select_dtypes(include=[np.number]).columns
        },
    }
    return report


def _detect_outliers_iqr(series):
    """
    Detects outliers in a series using IQR.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()


def _detect_outliers_zscore(series, threshold=3.0):
    """
    Detects outliers in a series using z-score.
    """
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return ((z_scores < -threshold) | (z_scores > threshold)).sum()


def is_data_clean(df):
    """
    Determines if the dataset is clean.
    """
    return df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0


def execute_phase_1_cleaning(
    df,
    missing_value_strategies=None,
    fill_values=None,
    outlier_methods=None,
    outlier_thresholds=None,
    outlier_log=False,
    include_scaling_columns=None,
    custom_code=None,
    handle_outliers_flag=True
):
    """
    Executes Phase 1 cleaning pipeline.
    """
    try:
        logging.info("Starting Phase 1 Cleaning...")

        # Step 1: Handle missing values
        df_cleaned = handle_missing_values(df, strategies=missing_value_strategies, fill_values=fill_values)
        logging.info("Step 1: Missing values handled.")

        # Step 2: Standardize column names
        df_cleaned = standardize_column_names(df_cleaned)
        logging.info("Step 2: Column names standardized.")

        # Step 3: Handle outliers
        if handle_outliers_flag:
            if outlier_log:
                df_cleaned, log = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds, log_changes=True)
                logging.info("Step 3: Outliers handled. Outlier log: %s", log)
            else:
                df_cleaned = handle_outliers(df_cleaned, methods=outlier_methods, thresholds=outlier_thresholds)
                logging.info("Step 3: Outliers handled without logging.")

        # Step 4: Normalize numeric columns
        df_cleaned = normalize_numeric_columns(df_cleaned, include_columns=include_scaling_columns)
        logging.info("Step 4: Numeric columns normalized.")

        # Step 5: Execute custom code
        if custom_code:
            df_cleaned = execute_custom_code(df_cleaned, custom_code)
            logging.info("Step 5: Custom code executed.")

        logging.info("Phase 1 Cleaning completed successfully.")
        return df_cleaned

    except Exception as e:
        logging.error("An error occurred during preprocessing: %s", e)
        raise e
