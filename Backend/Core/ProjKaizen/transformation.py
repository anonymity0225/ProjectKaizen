# services/transformation.py
"""
Production-ready data transformation module for FastAPI backend.
Provides comprehensive data transformation capabilities with full audit logging.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for better readability
TransformationResult = Tuple[pd.DataFrame, Dict[str, Any]]
TransformationMetadata = Dict[str, Any]

class TransformationError(Exception):
    """Custom exception for transformation errors."""
    pass

class TransformationRegistry:
    """Registry to track all transformations applied to a dataset."""
    
    def __init__(self):
        self.operations: List[TransformationMetadata] = []
    
    def add_operation(self, metadata: TransformationMetadata) -> None:
        """Add a transformation operation to the registry."""
        self.operations.append(metadata)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all applied transformations."""
        return {
            "total_operations": len(self.operations),
            "operations": self.operations,
            "columns_modified": list(set().union(*[op.get("transformed_columns", []) for op in self.operations])),
            "columns_added": list(set().union(*[op.get("added_columns", []) for op in self.operations])),
            "columns_dropped": list(set().union(*[op.get("dropped_columns", []) for op in self.operations]))
        }
    
    def clear(self) -> None:
        """Clear all recorded operations."""
        self.operations.clear()

# Global registry instance
transformation_registry = TransformationRegistry()

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TransformationError("Input must be a pandas DataFrame")
    if df.empty:
        raise TransformationError("DataFrame cannot be empty")

def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Validate that column exists in DataFrame."""
    if column not in df.columns:
        raise TransformationError(f"Column '{column}' not found in DataFrame")

def _create_success_metadata(
    method: str,
    transformed_columns: List[str] = None,
    added_columns: List[str] = None,
    dropped_columns: List[str] = None,
    message: str = "",
    additional_info: Dict[str, Any] = None
) -> TransformationMetadata:
    """Create standardized success metadata."""
    metadata = {
        "status": "success",
        "method": method,
        "transformed_columns": transformed_columns or [],
        "added_columns": added_columns or [],
        "dropped_columns": dropped_columns or [],
        "message": message
    }
    if additional_info:
        metadata.update(additional_info)
    return metadata

def _create_error_metadata(method: str, error: str) -> TransformationMetadata:
    """Create standardized error metadata."""
    return {
        "status": "error",
        "method": method,
        "error": error,
        "transformed_columns": [],
        "added_columns": [],
        "dropped_columns": []
    }

def validate_datetime_column(df: pd.DataFrame, column: str) -> bool:
    """
    Validate if a column can be converted to datetime.
    
    Args:
        df: Input DataFrame
        column: Column name to validate
        
    Returns:
        bool: True if column is valid datetime or can be converted
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return True
        
        # Try to convert a sample
        test_series = pd.to_datetime(df[column].head(min(10, len(df))), errors='coerce')
        return not test_series.isna().all()
    except Exception:
        return False

def validate_text_column(df: pd.DataFrame, column: str) -> bool:
    """
    Validate if a column contains text data suitable for processing.
    
    Args:
        df: Input DataFrame
        column: Column name to validate
        
    Returns:
        bool: True if column contains valid text data
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        # Check if column is string-like
        if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
            return False
        
        # Check if column has meaningful text content
        sample_text = df[column].dropna().astype(str).head(min(10, len(df)))
        if sample_text.empty:
            return False
        
        # Check average length to ensure it's not just single characters or numbers
        avg_length = sample_text.str.len().mean()
        return avg_length > 2
    except Exception:
        return False

def extract_date_components(
    df: pd.DataFrame,
    column: str,
    components: List[str]
) -> TransformationResult:
    """
    Extract specified components from a datetime column.
    
    Args:
        df: Input DataFrame
        column: Name of the datetime column
        components: List of components to extract (year, month, day, hour, minute, second, weekday)
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        # Validate components
        valid_components = {"year", "month", "day", "hour", "minute", "second", "weekday"}
        invalid_components = set(components) - valid_components
        if invalid_components:
            raise TransformationError(f"Invalid components: {invalid_components}. Valid options: {valid_components}")
        
        df_copy = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            if df_copy[column].isna().all():
                raise TransformationError(f"Column '{column}' cannot be converted to datetime")
        
        added_columns = []
        
        for component in components:
            new_col_name = f"{column}_{component}"
            if component == "year":
                df_copy[new_col_name] = df_copy[column].dt.year
            elif component == "month":
                df_copy[new_col_name] = df_copy[column].dt.month
            elif component == "day":
                df_copy[new_col_name] = df_copy[column].dt.day
            elif component == "hour":
                df_copy[new_col_name] = df_copy[column].dt.hour
            elif component == "minute":
                df_copy[new_col_name] = df_copy[column].dt.minute
            elif component == "second":
                df_copy[new_col_name] = df_copy[column].dt.second
            elif component == "weekday":
                df_copy[new_col_name] = df_copy[column].dt.dayofweek
            
            added_columns.append(new_col_name)
        
        metadata = _create_success_metadata(
            method="date_component_extraction",
            transformed_columns=[column],
            added_columns=added_columns,
            message=f"Extracted {len(components)} date components from '{column}': {', '.join(components)}"
        )
        
        transformation_registry.add_operation(metadata)
        return df_copy, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata("date_component_extraction", str(e))
        logger.error(f"Date component extraction failed: {e}")
        return df, error_metadata

def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_mapping: Optional[Dict[str, Any]] = None
) -> TransformationResult:
    """
    Encode categorical features using various methods.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        method: Encoding method (onehot, label, frequency, custom)
        custom_mapping: Custom mapping dictionary for custom encoding
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        valid_methods = {"onehot", "label", "frequency", "custom"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid encoding method '{method}'. Valid options: {valid_methods}")
        
        if method == "custom" and not custom_mapping:
            raise TransformationError("Custom mapping required for custom encoding method")
        
        df_copy = df.copy()
        added_columns = []
        dropped_columns = []
        
        if method == "onehot":
            # Handle missing values
            if df_copy[column].isna().any():
                df_copy[column] = df_copy[column].fillna('missing')
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df_copy[[column]])
            
            # Create feature names
            feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_copy.index)
            
            df_copy = pd.concat([df_copy.drop(column, axis=1), encoded_df], axis=1)
            added_columns = feature_names
            dropped_columns = [column]
            
        elif method == "label":
            encoder = LabelEncoder()
            # Handle missing values
            non_null_mask = df_copy[column].notna()
            if non_null_mask.any():
                df_copy.loc[non_null_mask, column] = encoder.fit_transform(df_copy.loc[non_null_mask, column])
            
        elif method == "frequency":
            freq_map = df_copy[column].value_counts(normalize=True).to_dict()
            df_copy[column] = df_copy[column].map(freq_map)
            
        elif method == "custom":
            df_copy[column] = df_copy[column].map(custom_mapping)
            # Check for unmapped values
            unmapped_count = df_copy[column].isna().sum() - df[column].isna().sum()
            if unmapped_count > 0:
                logger.warning(f"Custom encoding left {unmapped_count} values unmapped")
        
        metadata = _create_success_metadata(
            method=f"{method}_encoding",
            transformed_columns=[column] if method in ["label", "frequency", "custom"] else [],
            added_columns=added_columns,
            dropped_columns=dropped_columns,
            message=f"Applied {method} encoding to '{column}'" + (f", created {len(added_columns)} new columns" if added_columns else "")
        )
        
        transformation_registry.add_operation(metadata)
        return df_copy, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata(f"{method}_encoding", str(e))
        logger.error(f"Categorical encoding failed: {e}")
        return df, error_metadata

def scale_numeric(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_range: Tuple[float, float] = (0, 1)
) -> TransformationResult:
    """
    Scale numerical features using various methods.
    
    Args:
        df: Input DataFrame
        column: Column to scale
        method: Scaling method (minmax, standard, log, custom)
        custom_range: Custom range for minmax scaling
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        valid_methods = {"minmax", "standard", "log", "custom"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid scaling method '{method}'. Valid options: {valid_methods}")
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TransformationError(f"Column '{column}' is not numeric")
        
        df_copy = df.copy()
        
        if method == "minmax":
            scaler = MinMaxScaler(feature_range=custom_range)
            df_copy[column] = scaler.fit_transform(df_copy[[column]]).flatten()
            
        elif method == "standard":
            scaler = StandardScaler()
            df_copy[column] = scaler.fit_transform(df_copy[[column]]).flatten()
            
        elif method == "log":
            if (df_copy[column] <= 0).any():
                raise TransformationError("Log scaling requires all values to be positive")
            df_copy[column] = np.log1p(df_copy[column])
            
        elif method == "custom":
            min_val, max_val = custom_range
            col_min, col_max = df_copy[column].min(), df_copy[column].max()
            if col_max == col_min:
                raise TransformationError("Cannot scale column with constant values")
            df_copy[column] = min_val + (df_copy[column] - col_min) * (max_val - min_val) / (col_max - col_min)
        
        metadata = _create_success_metadata(
            method=f"{method}_scaling",
            transformed_columns=[column],
            message=f"Applied {method} scaling to '{column}'" + (f" with range {custom_range}" if method in ["minmax", "custom"] else "")
        )
        
        transformation_registry.add_operation(metadata)
        return df_copy, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata(f"{method}_scaling", str(e))
        logger.error(f"Numeric scaling failed: {e}")
        return df, error_metadata

def tokenize_text(
    df: pd.DataFrame,
    column: str,
    method: str = "stemming"
) -> TransformationResult:
    """
    Process text features via tokenization and optional stemming or lemmatization.
    
    Args:
        df: Input DataFrame
        column: Text column to process
        method: Text processing method (stemming, lemmatization, none)
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        valid_methods = {"stemming", "lemmatization", "none"}
        if method not in valid_methods:
            raise TransformationError(f"Invalid text processing method '{method}'. Valid options: {valid_methods}")
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        if method == "lemmatization":
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
        
        df_copy = df.copy()
        processed_texts = []
        
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer() if method == "stemming" else None
        lemmatizer = WordNetLemmatizer() if method == "lemmatization" else None
        
        for text in df_copy[column].astype(str):
            if pd.isna(text) or text.lower() in ['nan', 'none', '']:
                processed_texts.append("")
                continue
            
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]
            
            # Apply stemming or lemmatization
            if method == "stemming" and stemmer:
                tokens = [stemmer.stem(word) for word in tokens]
            elif method == "lemmatization" and lemmatizer:
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            processed_texts.append(" ".join(tokens))
        
        df_copy[column] = processed_texts
        
        metadata = _create_success_metadata(
            method=f"text_tokenization_{method}",
            transformed_columns=[column],
            message=f"Applied {method} tokenization to '{column}'"
        )
        
        transformation_registry.add_operation(metadata)
        return df_copy, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata(f"text_tokenization_{method}", str(e))
        logger.error(f"Text tokenization failed: {e}")
        return df, error_metadata

def apply_tfidf_vectorization(
    df: pd.DataFrame,
    column: str,
    max_features: int = 1000
) -> TransformationResult:
    """
    Apply TF-IDF vectorization to a text column.
    
    Args:
        df: Input DataFrame
        column: Text column to vectorize
        max_features: Maximum number of features for the TF-IDF matrix
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        _validate_column_exists(df, column)
        
        if max_features <= 0:
            raise TransformationError("max_features must be positive")
        
        df_copy = df.copy()
        
        # Prepare text data
        text_data = df_copy[column].astype(str).fillna("")
        
        if text_data.str.strip().eq("").all():
            raise TransformationError(f"Column '{column}' contains no meaningful text data")
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
        )
        
        tfidf_matrix = vectorizer.fit_transform(text_data)
        
        # Create feature names
        feature_names = [f"tfidf_{feature}" for feature in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=df_copy.index
        )
        
        # Combine with original DataFrame (drop original text column)
        df_result = pd.concat([df_copy.drop(column, axis=1), tfidf_df], axis=1)
        
        metadata = _create_success_metadata(
            method="tfidf_vectorization",
            added_columns=feature_names,
            dropped_columns=[column],
            message=f"Applied TF-IDF vectorization to '{column}', created {len(feature_names)} features",
            additional_info={
                "vocabulary_size": len(vectorizer.vocabulary_),
                "max_features": max_features
            }
        )
        
        transformation_registry.add_operation(metadata)
        return df_result, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata("tfidf_vectorization", str(e))
        logger.error(f"TF-IDF vectorization failed: {e}")
        return df, error_metadata

def apply_pca(
    df: pd.DataFrame,
    n_components: int,
    columns: Optional[List[str]] = None
) -> TransformationResult:
    """
    Apply Principal Component Analysis to reduce dimensionality.
    
    Args:
        df: Input DataFrame
        n_components: Number of principal components to keep
        columns: Specific columns to apply PCA to (if None, uses all numeric columns)
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        
        if n_components <= 0:
            raise TransformationError("n_components must be positive")
        
        df_copy = df.copy()
        
        # Select columns for PCA
        if columns is None:
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise TransformationError("No numeric columns found for PCA")
            columns = numeric_columns
        else:
            # Validate specified columns exist and are numeric
            for col in columns:
                _validate_column_exists(df_copy, col)
                if not pd.api.types.is_numeric_dtype(df_copy[col]):
                    raise TransformationError(f"Column '{col}' is not numeric")
        
        if n_components > len(columns):
            raise TransformationError(f"n_components ({n_components}) cannot be greater than number of features ({len(columns)})")
        
        # Extract data for PCA
        pca_data = df_copy[columns].fillna(0)  # Handle NaN values
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(pca_data)
        
        # Create new column names
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df_copy.index)
        
        # Combine with original DataFrame (drop original columns used for PCA)
        df_result = pd.concat([df_copy.drop(columns, axis=1), pca_df], axis=1)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        metadata = _create_success_metadata(
            method="pca",
            added_columns=pca_columns,
            dropped_columns=columns,
            message=f"Applied PCA to {len(columns)} columns, reduced to {n_components} components",
            additional_info={
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_variance_explained": cumulative_variance.tolist(),
                "total_variance_explained": float(cumulative_variance[-1])
            }
        )
        
        transformation_registry.add_operation(metadata)
        return df_result, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata("pca", str(e))
        logger.error(f"PCA failed: {e}")
        return df, error_metadata

def apply_custom_user_function(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    function_name: str = "custom_function"
) -> TransformationResult:
    """
    Apply a custom user-defined function to the DataFrame with safety checks.
    
    Args:
        df: Input DataFrame
        func: User-defined function that takes and returns a DataFrame
        function_name: Name of the function for logging purposes
        
    Returns:
        Tuple of (updated_df, transformation_metadata)
    """
    try:
        _validate_dataframe(df)
        
        if not callable(func):
            raise TransformationError("Provided function must be callable")
        
        # Safety check: try function on a small sample first
        sample_df = df.head(min(5, len(df))).copy()
        try:
            result_sample = func(sample_df)
            if not isinstance(result_sample, pd.DataFrame):
                raise TransformationError("Custom function must return a pandas DataFrame")
        except Exception as e:
            raise TransformationError(f"Custom function failed on sample data: {str(e)}")
        
        # Apply function to full dataset
        df_result = func(df.copy())
        
        # Analyze changes
        original_columns = set(df.columns)
        result_columns = set(df_result.columns)
        
        added_columns = list(result_columns - original_columns)
        dropped_columns = list(original_columns - result_columns)
        transformed_columns = list(original_columns & result_columns)
        
        metadata = _create_success_metadata(
            method=f"custom_function_{function_name}",
            transformed_columns=transformed_columns,
            added_columns=added_columns,
            dropped_columns=dropped_columns,
            message=f"Applied custom function '{function_name}'"
        )
        
        transformation_registry.add_operation(metadata)
        return df_result, metadata
        
    except Exception as e:
        error_metadata = _create_error_metadata(f"custom_function_{function_name}", str(e))
        logger.error(f"Custom function application failed: {e}")
        return df, error_metadata

def get_transformation_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of all transformations applied.
    
    Returns:
        Dictionary containing transformation summary
    """
    return transformation_registry.get_summary()

def clear_transformation_history() -> None:
    """Clear the transformation history."""
    transformation_registry.clear()

# Utility function for batch operations
def apply_multiple_transformations(
    df: pd.DataFrame,
    transformations: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, List[TransformationMetadata]]:
    """
    Apply multiple transformations in sequence.
    
    Args:
        df: Input DataFrame
        transformations: List of transformation configurations
        
    Returns:
        Tuple of (final_df, list_of_metadata)
    """
    current_df = df.copy()
    all_metadata = []
    
    for transform_config in transformations:
        func_name = transform_config.get("function")
        if not func_name:
            continue
        
        # Map function names to actual functions
        func_mapping = {
            "extract_date_components": extract_date_components,
            "encode_categorical": encode_categorical,
            "scale_numeric": scale_numeric,
            "tokenize_text": tokenize_text,
            "apply_tfidf_vectorization": apply_tfidf_vectorization,
            "apply_pca": apply_pca
        }
        
        if func_name in func_mapping:
            func = func_mapping[func_name]
            args = transform_config.get("args", {})
            
            try:
                current_df, metadata = func(current_df, **args)
                all_metadata.append(metadata)
            except Exception as e:
                error_metadata = _create_error_metadata(func_name, str(e))
                all_metadata.append(error_metadata)
                logger.error(f"Batch transformation failed at {func_name}: {e}")
                break
    
    return current_df, all_metadata