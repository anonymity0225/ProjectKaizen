import streamlit as st
import pandas as pd
from phase1_preprocessing import execute_phase_1_cleaning
from phase2_transformation import execute_phase_2_transformation
from phase3_modelling import run_model_building
from phase4_visualization import visualize_model_performance

# Initialize Streamlit app
st.set_page_config(page_title="Data Pipeline App", layout="wide")
st.title("Data Processing, Modeling, and Visualization App")

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Phase", ["Upload & Preprocessing", "Transformation", "Modeling", "Visualization"])

    if app_mode == "Upload & Preprocessing":
        upload_and_preprocess()
    elif app_mode == "Transformation":
        transformation()
    elif app_mode == "Modeling":
        modeling()
    elif app_mode == "Visualization":
        visualization()

# Phase 1: Upload & Preprocessing
def upload_and_preprocess():
    st.header("Phase 1: Upload & Preprocessing")
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Dataset")
            st.dataframe(df)

            if st.button("Run Preprocessing"):
                # Execute Phase 1: Preprocessing
                st.write("Running Preprocessing...")
                df_cleaned = execute_phase_1_cleaning(df)
                st.write("### Preprocessed Dataset")
                st.dataframe(df_cleaned)

                # Save option for preprocessed data
                st.download_button(
                    label="Download Preprocessed Dataset",
                    data=df_cleaned.to_csv(index=False).encode('utf-8'),
                    file_name="preprocessed_dataset.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error loading or preprocessing the dataset: {e}")

# Phase 2: Transformation
def transformation():
    st.header("Phase 2: Transformation")
    uploaded_file = st.file_uploader("Upload Preprocessed Dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preprocessed Dataset")
            st.dataframe(df)

            if st.button("Run Transformation"):
                # Execute Phase 2: Transformation
                st.write("Running Transformation...")
                df_transformed = execute_phase_2_transformation(df)
                st.write("### Transformed Dataset")
                st.dataframe(df_transformed)

                # Save option for transformed data
                st.download_button(
                    label="Download Transformed Dataset",
                    data=df_transformed.to_csv(index=False).encode('utf-8'),
                    file_name="transformed_dataset.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error during transformation: {e}")

# Phase 3: Modeling
def modeling():
    st.header("Phase 3: Modeling")
    uploaded_file = st.file_uploader("Upload Transformed Dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Transformed Dataset")
            st.dataframe(df)

            # Task and Target Selection
            task_type = st.selectbox("Select Task Type", ["classification", "regression"])
            target_column = st.selectbox("Select Target Column", df.columns)

            if st.button("Run Model Selection"):
                # Execute Phase 3: Modeling
                st.write(f"Running {task_type.capitalize()} Model Selection...")
                results = run_model_building(df, target_column, task_type)

                if task_type == "classification":
                    st.write("### Classification Results")
                    for result in results:
                        st.subheader(result['Model'])
                        st.write(f"**Best Parameters**: {result['Best Parameters']}")
                        st.write(f"**Accuracy**: {result['Accuracy']:.4f}")
                        st.write(f"**F1 Score**: {result['F1 Score']:.4f}")
                        st.json(result['Classification Report'])

                elif task_type == "regression":
                    st.write("### Regression Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    st.write("### Metrics Summary")
                    st.bar_chart(results_df[['MSE', 'MAE', 'R2 Score']].set_index(results_df['Model']))

        except Exception as e:
            st.error(f"Error during modeling: {e}")

# Phase 4: Visualization
def visualization():
    st.header("Phase 4: Visualization")
    uploaded_test_data = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
    uploaded_predictions = st.file_uploader("Upload Model Predictions (CSV)", type=["csv"])

    if uploaded_test_data and uploaded_predictions:
        try:
            # Load Data
            test_data = pd.read_csv(uploaded_test_data)
            predictions = pd.read_csv(uploaded_predictions)

            X_test = test_data.iloc[:, :-1]  # Features
            y_test = test_data.iloc[:, -1]  # Target
            y_preds = predictions.to_dict()  # Model predictions

            st.write("Visualization in Progress...")
            visualize_model_performance(task_type=None, models=None, X_test=X_test, y_test=y_test, y_preds=y_preds)
        except Exception as e:
            st.error(f"Error processing the files: {e}")
    else:
        st.info("Upload both test data and predictions to start visualization.")

# Run the main function
if __name__ == "__main__":
    main()
