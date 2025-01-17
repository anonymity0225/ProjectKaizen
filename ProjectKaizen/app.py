import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from phase1_preprocessing import (
    generate_cleanliness_report,
    preprocess_data,
    detect_outliers,
    handle_outliers,
    execute_custom_code as preprocess_custom_code,
    delete_rows_or_columns
)
from phase2_transformation import (
    extract_date_components,
    encode_categorical,
    scale_numeric,
    tokenize_text,
    apply_tfidf_vectorization,
    custom_user_prompt as transform_custom_code
)
from phase3_modelling import (
    detect_problem_type,
    train_and_evaluate_models,
    save_model,
    custom_interface as model_custom_code
)
from phase4_visualization import (
    custom_code_interface as visualize_custom_code
)

# Utility Functions
def load_dataset(uploaded_file):
    """Load a dataset from a file uploader."""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None

def save_plot(fig, filename, directory="visualizations", file_format="png"):
    """Save a plot to the specified directory."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.{file_format}")
    fig.savefig(path, format=file_format, bbox_inches='tight')
    return path

def preview_dataset(df, title="Dataset Preview"):
    """Display a preview of the dataset."""
    st.subheader(title)
    st.dataframe(df)

def run_custom_code(df, phase_name, custom_code_function):
    """Allow users to execute custom Python code for a specific phase."""
    st.subheader(f"Custom Code Execution: {phase_name}")
    custom_code = st.text_area(f"Enter your Python code for {phase_name} (use 'df' as the dataset variable)")
    if st.button(f"Run Custom Code ({phase_name})"):
        try:
            df = custom_code_function(df, custom_code)
            st.success(f"Custom code executed successfully for {phase_name}.")
        except Exception as e:
            st.error(f"Error executing custom code for {phase_name}: {e}")
    return df

def main():
    st.set_page_config(page_title="Data Processing and Visualization Tool", layout="wide")
    st.title("Data Processing and Visualization Tool")

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Please upload a dataset to begin.")
        return

    df = load_dataset(uploaded_file)
    if df is None:
        return

    preview_dataset(df)

    # Phase 1: Preprocessing
    st.sidebar.title("Phase 1: Preprocessing")
    if st.sidebar.checkbox("Enable Preprocessing"):
        st.subheader("Preprocessing Options")
        actions = {}

        if st.checkbox("Standardize Column Names"):
            actions["standardize_column_names"] = True

        if st.checkbox("Handle Missing Values"):
            missing_value_strategies = {}
            for col in df.columns:
                strategy = st.selectbox(
                    f"Strategy for '{col}'", ["none", "mean", "median", "mode", "constant", "drop_rows"], key=f"missing_{col}"
                )
                missing_value_strategies[col] = strategy
            actions["missing_value_strategies"] = missing_value_strategies

        if st.checkbox("Remove Duplicates"):
            actions["remove_duplicates"] = True
            actions["duplicates_keep"] = st.radio("Keep which duplicate occurrence?", ["first", "last", "none"], index=0)

        if st.button("Run Preprocessing"):
            st.write("Running preprocessing...")
            df = preprocess_data(df, actions)

        df = run_custom_code(df, "Preprocessing", preprocess_custom_code)
        preview_dataset(df, title="Preprocessed Dataset")

    # Phase 2: Transformation
    st.sidebar.title("Phase 2: Transformation")
    if st.sidebar.checkbox("Enable Transformation"):
        st.subheader("Transformation Options")

        if st.checkbox("Extract Date Components"):
            date_column = st.selectbox("Select Date Column", df.columns)
            components = st.multiselect("Components to Extract", ["year", "month", "day", "hour"])
            if st.button("Apply Date Extraction"):
                df = extract_date_components(df, date_column, components)

        if st.checkbox("Encode Categorical Variables"):
            cat_column = st.selectbox("Select Categorical Column", df.select_dtypes(include="object").columns)
            encoding_method = st.selectbox("Encoding Method", ["onehot", "label", "frequency", "custom"])
            mapping = None
            if encoding_method == "custom":
                mapping = st.text_area("Enter Custom Mapping (e.g., {'Low': 1, 'Medium': 2, 'High': 3})")
                mapping = eval(mapping) if mapping else None
            if st.button("Apply Encoding"):
                df = encode_categorical(df, cat_column, method=encoding_method, custom_mapping=mapping)

        df = run_custom_code(df, "Transformation", transform_custom_code)
        preview_dataset(df, title="Transformed Dataset")

    # Phase 3: Modelling
    st.sidebar.title("Phase 3: Modelling")
    if st.sidebar.checkbox("Enable Modelling"):
        st.subheader("Modelling Options")

        target_column = st.selectbox("Select Target Column", [None] + list(df.columns))
        if st.button("Train and Evaluate Models"):
            problem_type = detect_problem_type(df, target_column) if target_column else "clustering"
            st.write(f"Detected Problem Type: {problem_type}")

            with st.spinner("Training models..."):
                metrics, best_model = train_and_evaluate_models(df, target_column, problem_type)
                st.success("Model training completed.")

            st.write("### Model Performance Metrics")
            st.json(metrics)

            save_filename = st.text_input("Enter filename to save the best model", "best_model.pkl")
            if st.button("Save Best Model"):
                save_model(best_model, save_filename)
                st.success(f"Model saved as {save_filename}.")

        df = run_custom_code(df, "Modelling", model_custom_code)

    # Phase 4: Visualization
    st.sidebar.title("Phase 4: Visualization")
    if st.sidebar.checkbox("Enable Visualization"):
        st.subheader("Visualization Options")
        plot_type = st.selectbox("Select Plot Type", ["Univariate", "Bivariate", "Correlation Heatmap"])

        if plot_type == "Univariate":
            column = st.selectbox("Select Column", df.columns)
            if pd.api.types.is_numeric_dtype(df[column]):
                st.write(f"### Histogram of {column}")
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

        elif plot_type == "Bivariate":
            x = st.selectbox("Select X-Axis", df.columns)
            y = st.selectbox("Select Y-Axis", df.columns)
            st.write(f"### Scatter Plot: {x} vs {y}")
            fig, ax = plt.subplots()
            sns.scatterplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=np.number).columns
            corr_matrix = df[numeric_cols].corr()
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        df = run_custom_code(df, "Visualization", visualize_custom_code)

    # Save or Download Processed Dataset
    st.write("### Final Dataset")
    st.dataframe(df.head())
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Processed Dataset", data=csv, file_name="processed_dataset.csv", mime="text/csv")

if __name__ == "__main__":
    main()
