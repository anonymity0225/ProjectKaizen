# phase1_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to handle missing values
def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handles missing values in the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - strategy (str): Strategy for numeric columns ('mean', 'median', 'constant')
    - fill_value: Constant value for 'constant' strategy (used for numeric and categorical)

    Returns:
    - pd.DataFrame: Dataframe with missing values handled
    """
    print("Handling missing values...")

    # Handle numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'constant' and fill_value is not None:
                df[col] = df[col].fillna(fill_value)

    # Handle categorical columns
    for col in df.select_dtypes(include=[object]).columns:
        if df[col].isnull().sum() > 0:
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print("Missing values handled.")
    return df

# Function to standardize column names
def standardize_column_names(df):
    print("Standardizing column names...")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print("Column names standardized.")
    return df

# Function to handle outliers
def handle_outliers(df, log_changes=False):
    print("Handling outliers...")
    outlier_log = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if log_changes:
            outlier_log[col] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "original_outliers": df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()
            }

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    print("Outliers handled.")
    if log_changes:
        return df, outlier_log
    return df

# Function to normalize numeric columns
def normalize_numeric_columns(df):
    print("Normalizing numeric columns...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        print("No numeric columns found for normalization.")
        return df
    
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    print("Numeric columns normalized.")
    return df

# Main function for Phase 1 preprocessing
def execute_phase_1_cleaning(df, missing_value_strategy='mean', outlier_log=False):
    try:
        print("Starting Phase 1 Cleaning...")

        # Step 1: Handle missing values
        df_cleaned = handle_missing_values(df, strategy=missing_value_strategy)

        # Step 2: Standardize column names
        df_cleaned = standardize_column_names(df_cleaned)

        # Step 3: Handle outliers
        if outlier_log:
            df_cleaned, log = handle_outliers(df_cleaned, log_changes=True)
            print("Outlier log:", log)
        else:
            df_cleaned = handle_outliers(df_cleaned)

        # Step 4: Normalize numeric columns
        df_cleaned = normalize_numeric_columns(df_cleaned)
        
        print("Phase 1 Cleaning completed successfully.")
        return df_cleaned
    
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise e
