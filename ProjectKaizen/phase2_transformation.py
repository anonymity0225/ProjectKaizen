# data_transform_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Utility Functions for Data Transformation

def extract_date_components(df, column, components):
    """
    Extract specified components from a datetime column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the datetime column.
        components (list): List of components to extract (e.g., ["year", "month", "day"]).
        
    Returns:
        pd.DataFrame: Updated DataFrame with new columns for the extracted components.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not np.issubdtype(df[column].dtype, np.datetime64):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    
    for component in components:
        if component == "year":
            df[f"{column}_year"] = df[column].dt.year
        elif component == "month":
            df[f"{column}_month"] = df[column].dt.month
        elif component == "day":
            df[f"{column}_day"] = df[column].dt.day
        elif component == "hour":
            df[f"{column}_hour"] = df[column].dt.hour
    return df

def encode_categorical(df, column, method="onehot", custom_mapping=None):
    """
    Encode categorical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to encode.
        method (str): Encoding method ("onehot", "label", "frequency", "custom").
        custom_mapping (dict): Mapping for custom encoding (if applicable).
        
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if method == "onehot":
        ohe = OneHotEncoder(sparse=False)
        encoded = ohe.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([column]))
        df = pd.concat([df, encoded_df], axis=1).drop(column, axis=1)
    elif method == "label":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    elif method == "frequency":
        freq = df[column].value_counts(normalize=True)
        df[column] = df[column].map(freq)
    elif method == "custom" and custom_mapping:
        df[column] = df[column].map(custom_mapping)
    else:
        raise ValueError("Invalid encoding method or missing custom mapping.")
    
    return df

def scale_numeric(df, column, method="minmax", custom_range=(0, 1)):
    """
    Scale numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to scale.
        method (str): Scaling method ("minmax", "standard", "log", "custom").
        custom_range (tuple): Custom min/max range for scaling (if applicable).
        
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=custom_range)
        df[column] = scaler.fit_transform(df[[column]])
    elif method == "standard":
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
    elif method == "log":
        df[column] = np.log1p(df[column])
    elif method == "custom":
        min_val, max_val = custom_range
        df[column] = (df[column] - min_val) / (max_val - min_val)
    else:
        raise ValueError("Invalid scaling method.")
    
    return df

def tokenize_text(df, column, method="stemming"):
    """
    Process text features via tokenization and optional stemming or lemmatization.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Text column to process.
        method (str): Text processing method ("stemming" or "lemmatization").
        
    Returns:
        pd.DataFrame: Updated DataFrame with processed text.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    tokens_list = []
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    for text in df[column].astype(str):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        if method == "stemming":
            tokens = [stemmer.stem(word) for word in tokens]
        elif method == "lemmatization":
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens_list.append(" ".join(tokens))
    
    df[column] = tokens_list
    return df

def apply_tfidf_vectorization(df, column, max_features=1000):
    """
    Apply TF-IDF vectorization to a text column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Text column to vectorize.
        max_features (int): Maximum number of features for the TF-IDF matrix.
        
    Returns:
        pd.DataFrame: Updated DataFrame with vectorized features.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df[column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df = pd.concat([df.drop(column, axis=1), tfidf_df], axis=1)
    return df

# Custom User Code Area
def custom_user_prompt(df, action):
    """
    Allow users to define custom actions on the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        action (callable): User-defined function or lambda to apply on the DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame after applying the custom action.
    """
    if not callable(action):
        raise ValueError("Provided action must be a callable function or lambda.")
    return action(df)
