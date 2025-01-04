import streamlit as st
import pandas as pd
from phase1_preprocessing import execute_phase_1_cleaning, check_cleanliness, is_data_clean
from phase2_transformation import execute_phase_2_transformation
from phase3_modelling import run_model_building
from phase4_visualization import (
    visualize_model_performance,
    show_visualization_options,
    visualize_classification,
    visualize_regression
)

# Initialize Streamlit app
st.set_page_config(page_title="ProjectKaizen", layout="wide")
st.title("Project Kaizen: Preprocessing Workflow")

def main():
    st.header("Data Preprocessing")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to begin.")
        return

    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.dataframe(df)

    if is_data_clean(df):
        st.success("The dataset is already clean. No preprocessing is required.")
        return

    if st.button("Generate Cleanliness Report"):
        cleanliness_report = check_cleanliness(df)
        st.write("### Cleanliness Report")
        st.json(cleanliness_report)

    st.subheader("Preprocessing Options")
    
    missing_values = {col: st.selectbox(f"Missing value strategy for {col}", ["mean", "median", "mode", "constant", "drop_rows"]) 
                      for col in df.columns[df.isnull().any()]}
    scaling_columns = st.multiselect("Columns to normalize", df.select_dtypes(include=["float", "int"]).columns.tolist())
    custom_code = st.text_area("Custom Code")

    # New flags for selective function execution
    standardize_columns_flag = st.checkbox("Standardize Column Names", value=True)
    handle_outliers_flag = st.checkbox("Handle Outliers", value=False)
    normalize_columns_flag = st.checkbox("Normalize Numeric Columns", value=True)

    if st.button("Run Preprocessing"):
        cleaned_df = execute_phase_1_cleaning(
            df,
            missing_value_strategies=missing_values,
            include_scaling_columns=scaling_columns,
            custom_code=custom_code
        )

        # Apply additional steps based on user flags
        if standardize_columns_flag:
            cleaned_df = standardize_column_names(cleaned_df)

        if handle_outliers_flag:
            cleaned_df = handle_outliers(cleaned_df)

        if normalize_columns_flag:
            cleaned_df = normalize_numeric_columns(cleaned_df, include_columns=scaling_columns)
        
        st.write("### Cleaned Dataset")
        st.dataframe(cleaned_df)
        st.download_button("Download Dataset", cleaned_df.to_csv(index=False), "cleaned_dataset.csv")

# Phase 2: Transformation
def transformation_ui():
    st.header("Phase 2: Data Transformation")
    uploaded_file = st.file_uploader("Upload Preprocessed Dataset (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset")
        st.dataframe(df)

        # Transformation options
        scale_data = st.checkbox("Apply Scaling?")
        scaler_type = st.selectbox("Scaler Type", options=["StandardScaler", "MinMaxScaler", "RobustScaler"]) if scale_data else None
        encoding_data = st.checkbox("Apply Encoding?")
        encoding_type = st.selectbox("Encoding Type", options=["onehot", "label"]) if encoding_data else None

        # Run Transformation
        if st.button("Run Transformation"):
            try:
                transformed_df = execute_phase_2_transformation(
                    df,
                    scaler_type=scaler_type,
                    encoding_type=encoding_type
                )
                st.write("### Transformed Dataset")
                st.dataframe(transformed_df)
                st.download_button("Download Transformed Dataset", transformed_df.to_csv(index=False), "transformed.csv")
            except Exception as e:
                st.error(f"Error during transformation: {e}")

# Phase 3: Modeling
def modeling_ui():
    st.header("Phase 3: Modeling")
    uploaded_file = st.file_uploader("Upload Transformed Dataset (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset")
        st.dataframe(df)

        target_column = st.selectbox("Select Target Column", df.columns)
        task_type = st.radio("Task Type", ["classification", "regression"])
        selected_models = st.multiselect("Models to Train", options=select_model_options(task_type))
        if st.button("Run Modeling"):
            try:
                results = run_model_building(df, target_column, task_type, selected_models)
                st.write("### Modeling Results")
                st.dataframe(results)
            except Exception as e:
                st.error(f"Error during modeling: {e}")

def select_model_options(task_type):
    models = {
        "classification": ["Random Forest", "Logistic Regression", "SVC", "KNN", "Decision Tree", "XGBoost", "AdaBoost"],
        "regression": ["Random Forest", "Linear Regression", "SVR", "KNN", "Decision Tree", "XGBoost", "AdaBoost"]
    }
    return models[task_type]

# Phase 4: Visualization
def visualization_ui():
    st.header("Phase 4: Visualization")
    test_data_file = st.file_uploader("Upload Test Data (CSV)", type="csv")
    predictions_file = st.file_uploader("Upload Predictions (CSV)", type="csv")
    if test_data_file and predictions_file:
        test_data = pd.read_csv(test_data_file)
        predictions = pd.read_csv(predictions_file)
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        y_preds = predictions.to_dict()
        visualize_model_performance_ui(X_test, y_test, y_preds)

def visualize_model_performance_ui(X_test, y_test, y_preds):
    task_type = st.radio("Task Type", ["classification", "regression"])
    default_vis, additional_vis = show_visualization_options()
    if default_vis:
        visualize_model_performance(task_type, None, X_test, y_test, y_preds)
    for vis in additional_vis:
        st.write(f"Additional Visualization: {vis} is under development.")

if __name__ == "__main__":
    main()
