# phase2_transformation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler

### Task 1: Data Normalization/Scaling
def scale_data(df, scaler_type='StandardScaler', columns=None):
    """
    Scales specified numerical columns in the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - scaler_type (str): Type of scaler to use ('StandardScaler', 'MinMaxScaler', or 'RobustScaler')
    - columns (list): List of columns to scale; scales all numerical columns if None

    Returns:
    - pd.DataFrame: Scaled dataframe
    """
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    scaler = scalers.get(scaler_type)
    if not scaler:
        raise ValueError("Invalid scaler type. Choose 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'.")

    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled

### Task 2: Encoding Categorical Variables
def encode_categorical(df, encoding_type='onehot', columns=None):
    """
    Encodes specified categorical columns in the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - encoding_type (str): Type of encoding ('onehot' or 'label')
    - columns (list): List of columns to encode; encodes all categorical columns if None

    Returns:
    - pd.DataFrame: Encoded dataframe
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    df_encoded = df.copy()
    if encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
    elif encoding_type == 'label':
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    else:
        raise ValueError("Invalid encoding type. Choose 'onehot' or 'label'.")
    
    return df_encoded

### Main Function
def execute_phase_2_transformation(df, scaler_type='StandardScaler', encoding_type='onehot', columns_to_scale=None, columns_to_encode=None):
    """
    Executes the transformations for phase 2.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - scaler_type (str): Scaler type for numeric scaling
    - encoding_type (str): Encoding type for categorical variables
    - columns_to_scale (list): Columns to scale; scales all numeric columns if None
    - columns_to_encode (list): Columns to encode; encodes all categorical columns if None

    Returns:
    - pd.DataFrame: Transformed dataframe
    """
    print("Starting Phase 2 Transformation...")

    # Scale data
    df_transformed = scale_data(df, scaler_type=scaler_type, columns=columns_to_scale)
    
    # Encode categorical variables
    df_transformed = encode_categorical(df_transformed, encoding_type=encoding_type, columns=columns_to_encode)
    
    print("Phase 2 Transformation completed.")
    return df_transformed
