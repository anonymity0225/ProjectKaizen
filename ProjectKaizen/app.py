import streamlit as st
import pandas as pd
import numpy as np
import logging
from io import StringIO
from phase1_preprocessing import (
    generate_cleanliness_report,
    standardize_column_names,
    handle_missing_values,
    delete_rows_or_columns,
    remove_duplicates,
    detect_outliers,
    handle_outliers,
    execute_custom_code,
    preprocess_data
)

# Set page configuration
#st.set_page_config(page_title="Data Preprocessing Tool", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'preprocessing_history' not in st.session_state:
    st.session_state.preprocessing_history = []
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

def reset_state():
    """Reset the data to original state"""
    if st.session_state.original_data is not None:
        st.session_state.data = st.session_state.original_data.copy()
        st.session_state.preprocessing_history = []

def display_data_info():
    """Display current data information"""
    if st.session_state.data is not None:
        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.data.head())
        st.write(f"Current Shape: {st.session_state.data.shape}")
        
        # Display cleanliness report
        report = generate_cleanliness_report(st.session_state.data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing Values:")
            missing_df = pd.DataFrame({
                'Count': report['missing_values'],
                'Percentage': report['missing_percentage']
            }).round(2)
            st.dataframe(missing_df)
        
        with col2:
            st.write("Data Types:")
            st.dataframe(pd.DataFrame.from_dict(report['data_types'], 
                                              orient='index', 
                                              columns=['Type']))
            st.write(f"Number of Duplicates: {report['duplicates']}")

def main():
    st.title("Data Preprocessing Tool")
    
    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or XLSX)", 
        type=['csv', 'xlsx'],
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    # Reset Button
    if st.session_state.original_data is not None:
        if st.button("Reset to Original Data"):
            reset_state()
    
    if uploaded_file is not None:
        try:
            # Load data only if it's a new file
            if st.session_state.original_data is None:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.original_data = df.copy()
                st.session_state.data = df.copy()
                st.session_state.preprocessing_history = []
            
            # Display current data info
            display_data_info()
            
            # Preprocessing Options
            st.header("Preprocessing Options")
            
            # 1. Standardize Column Names
            with st.expander("Standardize Column Names"):
                if st.button("Standardize Names"):
                    st.session_state.data = standardize_column_names(st.session_state.data)
                    st.session_state.preprocessing_history.append("Standardized column names")
                    st.success("Column names standardized!")
                    display_data_info()
            
            # 2. Handle Missing Values
            with st.expander("Handle Missing Values"):
                cols_with_missing = st.session_state.data.columns[
                    st.session_state.data.isnull().any()
                ].tolist()
                
                if cols_with_missing:
                    strategies = {}
                    fill_values = {}
                    
                    for col in cols_with_missing:
                        col_type = st.session_state.data[col].dtype
                        strategy_options = ["none", "drop_rows"]
                        
                        if pd.api.types.is_numeric_dtype(col_type):
                            strategy_options.extend(["mean", "median", "constant"])
                        else:
                            strategy_options.extend(["mode", "constant"])
                        
                        strategy = st.selectbox(
                            f"Strategy for {col}",
                            strategy_options,
                            key=f"strategy_{col}"
                        )
                        
                        if strategy != "none":
                            strategies[col] = strategy
                            if strategy == "constant":
                                fill_values[col] = st.text_input(
                                    f"Fill value for {col}",
                                    key=f"fill_{col}"
                                )
                    
                    if st.button("Apply Missing Value Handling"):
                        st.session_state.data = handle_missing_values(
                            st.session_state.data,
                            strategies,
                            fill_values
                        )
                        st.session_state.preprocessing_history.append(
                            f"Handled missing values: {strategies}"
                        )
                        st.success("Missing values handled!")
                        display_data_info()
                else:
                    st.info("No missing values found in the dataset.")
            
            # 3. Handle Outliers
            with st.expander("Handle Outliers"):
                numeric_cols = st.session_state.data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                
                if numeric_cols:
                    methods = {}
                    thresholds = {}
                    
                    cols_to_process = st.multiselect(
                        "Select columns to check for outliers",
                        numeric_cols
                    )
                    
                    if cols_to_process:
                        method = st.selectbox(
                            "Select outlier detection method",
                            ["iqr", "zscore"]
                        )
                        threshold = st.slider(
                            "Select threshold",
                            1.0, 5.0, 3.0
                        )
                        
                        for col in cols_to_process:
                            methods[col] = method
                            thresholds[col] = threshold
                        
                        strategy = st.selectbox(
                            "Select handling strategy",
                            ["replace", "clip"]
                        )
                        
                        if st.button("Handle Outliers"):
                            outlier_log = detect_outliers(
                                st.session_state.data,
                                methods,
                                thresholds
                            )
                            st.session_state.data = handle_outliers(
                                st.session_state.data,
                                outlier_log,
                                strategy
                            )
                            st.session_state.preprocessing_history.append(
                                f"Handled outliers: {methods} with {strategy} strategy"
                            )
                            st.success("Outliers handled!")
                            display_data_info()
                else:
                    st.info("No numeric columns found for outlier detection.")
            
            # 4. Remove Duplicates
            with st.expander("Remove Duplicates"):
                keep_option = st.selectbox(
                    "Which duplicates to keep?",
                    ["first", "last", "False"]
                )
                if st.button("Remove Duplicates"):
                    st.session_state.data = remove_duplicates(
                        st.session_state.data,
                        keep=keep_option
                    )
                    st.session_state.preprocessing_history.append(
                        f"Removed duplicates (keep={keep_option})"
                    )
                    st.success("Duplicates removed!")
                    display_data_info()
            
            # 5. Delete Rows/Columns
            with st.expander("Delete Rows or Columns"):
                col1, col2 = st.columns(2)
                
                with col1:
                    rows_to_delete = st.text_input(
                        "Enter row indices to delete (comma-separated)"
                    )
                
                with col2:
                    columns_to_delete = st.multiselect(
                        "Select columns to delete",
                        st.session_state.data.columns.tolist()
                    )
                
                if st.button("Apply Deletion"):
                    if rows_to_delete:
                        rows_list = [int(x.strip()) for x in rows_to_delete.split(",")]
                    else:
                        rows_list = None
                    
                    st.session_state.data = delete_rows_or_columns(
                        st.session_state.data,
                        rows_list,
                        columns_to_delete
                    )
                    st.session_state.preprocessing_history.append(
                        f"Deleted rows/columns: Rows={rows_list}, Columns={columns_to_delete}"
                    )
                    st.success("Deletion completed!")
                    display_data_info()
            
            # 6. Custom Code
            with st.expander("Custom Code"):
                custom_code = st.text_area(
                    "Enter your custom Python code",
                    "# Example:\n# df['new_column'] = df['existing_column'] * 2"
                )
                if st.button("Execute Custom Code"):
                    st.session_state.data = execute_custom_code(
                        st.session_state.data,
                        custom_code
                    )
                    st.session_state.preprocessing_history.append("Executed custom code")
                    st.success("Custom code executed!")
                    display_data_info()
            
            # Display Preprocessing History
            if st.session_state.preprocessing_history:
                st.header("Preprocessing History")
                for i, step in enumerate(st.session_state.preprocessing_history, 1):
                    st.write(f"{i}. {step}")
            
            # Download processed data
            if st.session_state.data is not None:
                st.download_button(
                    label="Download Processed Data",
                    data=st.session_state.data.to_csv(index=False),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in preprocessing: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from phase2_transformation import (
    extract_date_components,
    encode_categorical,
    scale_numeric,
    tokenize_text,
    apply_tfidf_vectorization,
    custom_user_prompt
)
def transformation_section():
    st.header("Data Transformation Options")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload a dataset first in the preprocessing section.")
        return
    
    # Add a preview of current data state
    st.subheader("Current Data Preview")
    st.dataframe(st.session_state.data.head())
    
    # Create tabs for different types of transformations
    date_tab, categorical_tab, numeric_tab, text_tab, custom_tab = st.tabs([
        "Date Transformation", 
        "Categorical Encoding", 
        "Numeric Scaling", 
        "Text Processing",
        "Custom Transformation"
    ])
    
    with date_tab:
        st.subheader("Date Component Extraction")
        date_columns = st.session_state.data.select_dtypes(include=['datetime64', 'object']).columns
        
        if len(date_columns) > 0:
            selected_date_cols = st.multiselect("Select date columns", date_columns)
            date_components = st.multiselect(
                "Select components to extract",
                ["year", "month", "day", "hour"],
                default=["year", "month"]
            )
            
            if st.button("Extract Date Components") and selected_date_cols:
                try:
                    for col in selected_date_cols:
                        st.session_state.data = extract_date_components(
                            st.session_state.data,
                            col,
                            date_components
                        )
                    st.success("Date components extracted successfully!")
                    st.write("Preview after transformation:")
                    st.dataframe(st.session_state.data.head())
                except Exception as e:
                    st.error(f"Error in date extraction: {str(e)}")
        else:
            st.info("No date columns detected in the dataset.")
    
    with categorical_tab:
        st.subheader("Categorical Encoding")
        categorical_columns = st.session_state.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) > 0:
            # Create a container for transformations
            cat_transforms = st.container()
            
            with cat_transforms:
                selected_cat_cols = st.multiselect("Select categorical columns", categorical_columns)
                encoding_method = st.selectbox(
                    "Select encoding method",
                    ["onehot", "label", "frequency", "custom"]
                )
                
                custom_mapping = None
                if encoding_method == "custom":
                    st.write("Enter custom mapping (one mapping per line, format: original_value:encoded_value)")
                    mapping_text = st.text_area("Custom mapping")
                    if mapping_text:
                        try:
                            custom_mapping = dict(
                                line.split(":") for line in mapping_text.strip().split("\n")
                            )
                        except:
                            st.error("Invalid mapping format")
                
                if st.button("Apply Encoding") and selected_cat_cols:
                    try:
                        for col in selected_cat_cols:
                            st.session_state.data = encode_categorical(
                                st.session_state.data,
                                col,
                                method=encoding_method,
                                custom_mapping=custom_mapping
                            )
                        st.success("Encoding applied successfully!")
                        st.write("Preview after transformation:")
                        st.dataframe(st.session_state.data.head())
                    except Exception as e:
                        st.error(f"Error in encoding: {str(e)}")
        else:
            st.info("No categorical columns detected in the dataset.")
    
    with numeric_tab:
        st.subheader("Numeric Scaling")
        numeric_columns = st.session_state.data.select_dtypes(include=np.number).columns
        
        if len(numeric_columns) > 0:
            # Create a container for transformations
            num_transforms = st.container()
            
            with num_transforms:
                selected_num_cols = st.multiselect("Select numeric columns", numeric_columns)
                scaling_method = st.selectbox(
                    "Select scaling method",
                    ["minmax", "standard", "log", "custom"]
                )
                
                custom_range = (0, 1)
                if scaling_method == "custom":
                    min_val = st.number_input("Minimum value", value=0.0)
                    max_val = st.number_input("Maximum value", value=1.0)
                    custom_range = (min_val, max_val)
                
                if st.button("Apply Scaling") and selected_num_cols:
                    try:
                        for col in selected_num_cols:
                            st.session_state.data = scale_numeric(
                                st.session_state.data,
                                col,
                                method=scaling_method,
                                custom_range=custom_range
                            )
                        st.success("Scaling applied successfully!")
                        st.write("Preview after transformation:")
                        st.dataframe(st.session_state.data.head())
                    except Exception as e:
                        st.error(f"Error in scaling: {str(e)}")
        else:
            st.info("No numeric columns detected in the dataset.")
    
    with text_tab:
        st.subheader("Text Processing")
        text_columns = st.session_state.data.select_dtypes(include=['object']).columns
        
        if len(text_columns) > 0:
            selected_text_cols = st.multiselect("Select text columns", text_columns)
            text_process = st.radio(
                "Select text processing method",
                ["stemming", "lemmatization"]
            )
            
            apply_tfidf = st.checkbox("Apply TF-IDF Vectorization after processing")
            max_features = None
            if apply_tfidf:
                max_features = st.number_input(
                    "Maximum number of features for TF-IDF",
                    min_value=10,
                    value=1000
                )
            
            if st.button("Process Text") and selected_text_cols:
                try:
                    # Download NLTK resources if needed
                    nltk.download('punkt')
                    nltk.download('stopwords')
                    nltk.download('wordnet')
                    
                    # Apply text processing to each selected column
                    for col in selected_text_cols:
                        # Apply text processing
                        st.session_state.data = tokenize_text(
                            st.session_state.data,
                            col,
                            method=text_process
                        )
                        
                        # Apply TF-IDF if selected
                        if apply_tfidf:
                            st.session_state.data = apply_tfidf_vectorization(
                                st.session_state.data,
                                col,
                                max_features=max_features
                            )
                    
                    st.success("Text processing completed successfully!")
                    st.write("Preview after transformation:")
                    st.dataframe(st.session_state.data.head())
                except Exception as e:
                    st.error(f"Error in text processing: {str(e)}")
        else:
            st.info("No text columns detected in the dataset.")
    
    with custom_tab:
        st.subheader("Custom Transformation")
        st.write("Enter custom Python code to transform the data (use 'df' as the variable name)")
        custom_code = st.text_area("Custom transformation code", height=150)
        
        if st.button("Apply Custom Transformation"):
            try:
                # Create a lambda function from the custom code
                custom_action = eval(f"lambda df: {custom_code}")
                st.session_state.data = custom_user_prompt(
                    st.session_state.data,
                    custom_action
                )
                st.success("Custom transformation applied successfully!")
                st.write("Preview after transformation:")
                st.dataframe(st.session_state.data.head())
            except Exception as e:
                st.error(f"Error in custom transformation: {str(e)}")
    
    # Download transformed data
    if st.button("Download Transformed Data"):
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="transformed_data.csv",
            mime="text/csv"
        )

# Add this to the main() function in the previous code
def extend_main():
    """
    Add this section to the main() function after the preprocessing section
    """
    st.markdown("---")
    transformation_section()

# The main() function from the previous code remains the same
# Just add the following line at the end of main():
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle
from phase3_modelling import (
    detect_problem_type,
    split_dataset,
    train_and_evaluate_models,
    save_model
)

def ml_modeling_section():
    st.header("Machine Learning Modeling")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload and process your data first.")
        return
    
    df = st.session_state.data
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    # Target Selection
    target_column = st.selectbox(
        "Select target column (leave empty for clustering)",
        options=[None] + list(df.columns),
        index=0
    )
    
    if target_column:
        st.write("Preview of target variable distribution:")
        fig, ax = plt.subplots(figsize=(8, 4))
        if df[target_column].dtype in ['object', 'category']:
            df[target_column].value_counts().plot(kind='bar', ax=ax)
        else:
            df[target_column].hist(ax=ax)
        st.pyplot(fig)
    
    # Detect problem type
    try:
        problem_type = detect_problem_type(df, target_column)
        st.success(f"Detected problem type: {problem_type.upper()}")
    except ValueError as e:
        st.error(str(e))
        return
    
    # Model Training Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        val_size = st.slider("Validation set size", 0.0, 0.3, 0.0, 0.05)
    with col3:
        random_state = st.number_input("Random seed", value=42)
    
    # Feature Selection
    st.subheader("Feature Selection")
    if target_column:
        feature_columns = st.multiselect(
            "Select features for modeling",
            options=[col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )
    else:
        feature_columns = st.multiselect(
            "Select features for clustering",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
    
    if not feature_columns:
        st.warning("Please select at least one feature.")
        return
    
    # Model Training
    if st.button("Train Models"):
        st.subheader("Model Training Results")
        
        # Prepare data
        model_df = df[feature_columns + ([target_column] if target_column else [])]
        
        with st.spinner("Training models..."):
            try:
                metrics, best_model = train_and_evaluate_models(
                    model_df,
                    target_column,
                    problem_type,
                    random_state=random_state
                )
                
                # Display metrics
                st.write("Model Performance Metrics:")
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df)
                
                # Visualize metrics
                st.subheader("Performance Visualization")
                if problem_type == 'regression':
                    metric_to_plot = 'R2'
                elif problem_type == 'classification':
                    metric_to_plot = 'Accuracy'
                else:
                    metric_to_plot = 'Score'
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df[metric_to_plot].plot(kind='bar', ax=ax)
                plt.title(f"Model Comparison - {metric_to_plot}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Save best model
                st.subheader("Best Model")
                best_model_name = max(metrics, key=lambda m: 
                    metrics[m].get('R2', 0) if problem_type == 'regression' 
                    else metrics[m].get('Accuracy', 0) if problem_type == 'classification'
                    else metrics[m].get('Score', 0)
                )
                st.success(f"Best performing model: {best_model_name}")
                
                # Save model to session state
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_model_name
                
                # Model Download
                if st.button("Download Best Model"):
                    model_bytes = BytesIO()
                    pickle.dump(best_model, model_bytes)
                    st.download_button(
                        label="Download Model",
                        data=model_bytes.getvalue(),
                        file_name=f"best_{problem_type}_model.pkl",
                        mime="application/octet-stream"
                    )
                
                # Model Prediction Interface
                if target_column and st.checkbox("Show Prediction Interface"):
                    st.subheader("Make Predictions")
                    
                    # Create input fields for features
                    input_data = {}
                    for feature in feature_columns:
                        if df[feature].dtype in ['int64', 'float64']:
                            input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
                        else:
                            input_data[feature] = st.selectbox(f"Select {feature}", df[feature].unique())
                    
                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        prediction = best_model.predict(input_df)
                        st.success(f"Prediction: {prediction[0]}")
                
                # Clustering Visualization
                if problem_type == 'clustering' and hasattr(best_model, 'labels_'):
                    st.subheader("Clustering Visualization")
                    if len(feature_columns) >= 2:
                        x_col = st.selectbox("Select X axis", feature_columns)
                        y_col = st.selectbox("Select Y axis", feature_columns)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = plt.scatter(
                            df[x_col],
                            df[y_col],
                            c=best_model.labels_,
                            cmap='viridis'
                        )
                        plt.colorbar(scatter)
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title("Clustering Results")
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# Add this to the main() function after the transformation section
def extend_main1():
    st.markdown("---")  # Add a visual separator
    ml_modeling_section()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import io

def visualization_section():
    st.header("Data Visualization Dashboard")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload and process your data first.")
        return
    
    df = st.session_state.data
    
    # Sidebar for visualization options
    st.sidebar.subheader("Visualization Options")
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Dataset Overview", "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", "Custom Visualization"]
    )
    
    # Dataset Overview
    if viz_type == "Dataset Overview":
        st.subheader("Dataset Overview")
        
        # Summary statistics
        st.write("### Summary Statistics")
        st.dataframe(df.describe())
        
        # Data types info
        st.write("### Data Types Information")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtypes_df)
        
        # Missing values heatmap
        st.write("### Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Pattern")
        st.pyplot(fig)
        
        # Automatic plotting based on data types
        st.write("### Distribution Plots")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.write("#### Numeric Variables")
            selected_numeric = st.multiselect(
                "Select numeric columns to plot",
                numeric_cols,
                default=list(numeric_cols)[:3]
            )
            
            for col in selected_numeric:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig)
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("#### Categorical Variables")
            selected_categorical = st.multiselect(
                "Select categorical columns to plot",
                categorical_cols,
                default=list(categorical_cols)[:3]
            )
            
            for col in selected_categorical:
                fig = px.bar(df[col].value_counts().reset_index(), 
                           x='index', y=col, 
                           title=f"Distribution of {col}")
                st.plotly_chart(fig)
    
    # Univariate Analysis
    elif viz_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        
        column = st.selectbox("Select column for analysis", df.columns)
        plot_type = st.selectbox(
            "Select plot type",
            ["Histogram", "Box Plot", "Violin Plot", "Bar Plot", "Pie Chart"]
        )
        
        if pd.api.types.is_numeric_dtype(df[column]):
            if plot_type == "Histogram":
                fig = px.histogram(df, x=column, nbins=30)
            elif plot_type == "Box Plot":
                fig = px.box(df, y=column)
            elif plot_type == "Violin Plot":
                fig = px.violin(df, y=column)
            else:
                st.warning(f"{plot_type} is not suitable for numeric data")
                return
        else:
            if plot_type in ["Bar Plot", "Pie Chart"]:
                value_counts = df[column].value_counts()
                if plot_type == "Bar Plot":
                    fig = px.bar(x=value_counts.index, y=value_counts.values)
                else:
                    fig = px.pie(values=value_counts.values, names=value_counts.index)
            else:
                st.warning(f"{plot_type} is not suitable for categorical data")
                return
        
        st.plotly_chart(fig)
        
        # Summary statistics
        st.write("### Summary Statistics")
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(df[column].describe())
        else:
            st.write(df[column].value_counts())
    
    # Bivariate Analysis
    elif viz_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis variable", df.columns)
        with col2:
            y_col = st.selectbox("Select Y-axis variable", df.columns)
        
        plot_type = st.selectbox(
            "Select plot type",
            ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Violin Plot"]
        )
        
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            if plot_type == "Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col)
            elif plot_type == "Line Plot":
                fig = px.line(df, x=x_col, y=y_col)
            else:
                st.warning(f"{plot_type} might not be suitable for two numeric variables")
                return
        elif pd.api.types.is_numeric_dtype(df[y_col]):
            if plot_type in ["Bar Plot", "Box Plot", "Violin Plot"]:
                if plot_type == "Bar Plot":
                    fig = px.bar(df, x=x_col, y=y_col)
                elif plot_type == "Box Plot":
                    fig = px.box(df, x=x_col, y=y_col)
                else:
                    fig = px.violin(df, x=x_col, y=y_col)
            else:
                st.warning(f"{plot_type} is not suitable for this combination of variables")
                return
        else:
            st.warning("Selected combination of variables is not suitable for bivariate analysis")
            return
        
        st.plotly_chart(fig)
    
    # Correlation Analysis
    elif viz_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for correlation analysis")
            return
        
        correlation_method = st.selectbox(
            "Select correlation method",
            ["pearson", "spearman", "kendall"]
        )
        
        corr_matrix = df[numeric_cols].corr(method=correlation_method)
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig)
        
        # Detailed correlation values
        st.write("### Correlation Matrix")
        st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
    
    # Custom Visualization
    else:
        st.subheader("Custom Visualization")
        
        st.write("Write custom Python code to create visualizations")
        st.write("Available variables: `df` (your dataset), `px` (plotly express), `go` (plotly graph objects)")
        
        custom_code = st.text_area(
            "Enter your visualization code here",
            height=200,
            help="Your code should create a Plotly figure object named 'fig'"
        )
        
        if st.button("Generate Visualization"):
            try:
                # Create a local namespace
                local_dict = {'df': df, 'px': px, 'go': go, 'np': np, 'pd': pd}
                
                # Execute the code
                exec(custom_code, {}, local_dict)
                
                # Get the figure from the local namespace
                if 'fig' in local_dict:
                    st.plotly_chart(local_dict['fig'])
                else:
                    st.error("Please create a Plotly figure named 'fig' in your code")
            except Exception as e:
                st.error(f"Error executing code: {str(e)}")

    # Download options
    st.sidebar.subheader("Export Options")
    if st.sidebar.button("Export Current Plot"):
        # Save current plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        st.sidebar.download_button(
            label="Download Plot",
            data=buf,
            file_name="visualization.png",
            mime="image/png"
        )

# Add this to the main() function after the ML modeling section
def extend_main2():
    st.markdown("---")  # Add a visual separator
    visualization_section()

# The main() function from the previous code remains the same
# Just add the following line at the end of main():
# extend_main()
if __name__ == "__main__":
    main()
extend_main()
extend_main1()
extend_main2()





















