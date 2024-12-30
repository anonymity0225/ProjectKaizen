# phase2_transformation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
import streamlit as st

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
    try:
        df_scaled[columns] = scaler.fit_transform(df[columns])
    except Exception as e:
        st.error(f"Error scaling data: {e}")
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
    try:
        if encoding_type == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        elif encoding_type == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
        else:
            raise ValueError("Invalid encoding type. Choose 'onehot' or 'label'.")
    except Exception as e:
        st.error(f"Error encoding categorical data: {e}")
    return df_encoded


### Task 3: Custom Code Execution
def execute_custom_code(df, user_code=None, user_prompt=None):
    """
    Allows the user to execute custom code or use a prompt-based helper on the dataset.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - user_code (str): Custom Python code to execute
    - user_prompt (str): Prompt for automatic dataset transformation

    Returns:
    - pd.DataFrame: Transformed dataframe
    """
    df_transformed = df.copy()
    try:
        if user_code:
            exec(user_code)
        elif user_prompt:
            st.info(f"Executing transformation based on user prompt: {user_prompt}")
            # Example: Simple prompt-to-code mapping (extend this with AI-based models if needed)
            if "add column" in user_prompt.lower():
                df_transformed["new_column"] = 0  # Placeholder logic
            else:
                raise ValueError("Prompt not recognized. Please specify an appropriate transformation.")
    except Exception as e:
        st.error(f"Error executing custom code/prompt: {e}")
    return df_transformed


### Task 4: Transformation Check
def is_already_transformed(df, transformed_flags=None):
    """
    Checks if the dataset has already been transformed.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - transformed_flags (dict): Flags indicating transformation states (optional)

    Returns:
    - bool: True if dataset is already transformed, False otherwise
    """
    if transformed_flags is None:
        # Example heuristic: Check for standard transformation artifacts
        return "new_column" in df.columns or any("_encoded" in col for col in df.columns)
    return any(transformed_flags.values())


### Main Function
def execute_phase_2_transformation(
    df,
    scaler_type='StandardScaler',
    encoding_type='onehot',
    columns_to_scale=None,
    columns_to_encode=None,
    custom_code=None,
    custom_prompt=None
):
    """
    Executes the transformations for phase 2.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - scaler_type (str): Scaler type for numeric scaling
    - encoding_type (str): Encoding type for categorical variables
    - columns_to_scale (list): Columns to scale; scales all numeric columns if None
    - columns_to_encode (list): Columns to encode; encodes all categorical columns if None
    - custom_code (str): Custom Python code for transformation
    - custom_prompt (str): Prompt-based helper for transformation

    Returns:
    - pd.DataFrame: Transformed dataframe
    """
    st.write("Checking if the dataset is already transformed...")
    if is_already_transformed(df):
        st.warning("The dataset appears to be already transformed. Proceeding with optional transformations only.")

    try:
        # Scale data
        if st.checkbox("Apply Scaling?"):
            st.write("Scaling selected columns...")
            df = scale_data(df, scaler_type=scaler_type, columns=columns_to_scale)

        # Encode categorical variables
        if st.checkbox("Apply Encoding?"):
            st.write("Encoding selected columns...")
            df = encode_categorical(df, encoding_type=encoding_type, columns=columns_to_encode)

        # Custom code or prompt execution
        if custom_code or custom_prompt:
            st.write("Executing custom code or prompt-based transformation...")
            df = execute_custom_code(df, user_code=custom_code, user_prompt=custom_prompt)

        st.success("Phase 2 Transformation completed.")
    except Exception as e:
        st.error(f"Error during transformation: {e}")

    return df
