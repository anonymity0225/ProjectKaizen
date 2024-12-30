import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import streamlit as st

# Enhanced visualization functions

def visualize_model_performance(task_type, models, X_test, y_test, y_preds):
    """
    Visualizes model performance for classification or regression tasks.

    Parameters:
    - task_type (str): Type of task ('classification' or 'regression')
    - models (dict): Models to visualize (optional)
    - X_test (pd.DataFrame): Test features
    - y_test (pd.Series): Test labels
    - y_preds (dict): Predictions from models
    """
    try:
        if task_type == 'classification':
            visualize_classification(models, X_test, y_test, y_preds)
        elif task_type == 'regression':
            visualize_regression(models, X_test, y_test, y_preds)
        else:
            raise ValueError("Invalid task type for visualization. Choose 'classification' or 'regression'.")
    except Exception as e:
        st.error(f"Error during visualization: {e}")

# Classification Visualizations
def visualize_classification(models, X_test, y_test, y_preds):
    for model_name, y_pred in y_preds.items():
        st.subheader(f"Visualizing performance for {model_name}")

        # Confusion Matrix
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'Confusion Matrix for {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        # ROC and Precision-Recall Curves
        classes = np.unique(y_test)
        y_test_binarized = label_binarize(y_test, classes=classes)

        # Macro and Micro ROC for Multi-class
        if len(classes) > 2:  # Multi-class
            # Macro-average ROC
            st.write("**Macro-Average ROC Curve**")
            all_fpr = np.unique(np.concatenate([roc_curve(y_test_binarized[:, i], y_pred == i)[0] for i in range(len(classes))]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred == i)
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= len(classes)
            roc_auc_macro = auc(all_fpr, mean_tpr)

            fig, ax = plt.subplots()
            ax.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {roc_auc_macro:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title(f'Macro-Average ROC Curve for {model_name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)
        
        # Binary or One-vs-Rest ROC for Multi-class
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred == class_label)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title(f'ROC Curve for {model_name} (Class {class_label})')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)

        # Precision-Recall Curve
        st.write("**Precision-Recall Curve**")
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred == class_label)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=f'Class {class_label}')
            ax.set_title(f'Precision-Recall Curve for {model_name} (Class {class_label})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend()
            st.pyplot(fig)

# Regression Visualizations
def visualize_regression(models, X_test, y_test, y_preds):
    for model_name, y_pred in y_preds.items():
        st.subheader(f"Visualizing performance for {model_name}")

        # Residuals Plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=residuals, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'Residual Plot for {model_name}')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Residuals')
        st.pyplot(fig)

        # Actual vs Predicted Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        ax.set_title(f'Actual vs Predicted for {model_name}')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        st.pyplot(fig)

        # Error Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_title(f'Error Distribution for {model_name}')
        ax.set_xlabel('Residuals')
        st.pyplot(fig)

# Custom Code Execution
@st.cache_data
def execute_custom_code(df, code):
    """
    Executes custom Python code on the provided DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - code (str): Python code as a string

    Returns:
    - pd.DataFrame: Modified DataFrame
    """
    try:
        local_vars = {'df': df}
        exec(code, {}, local_vars)
        return local_vars['df']
    except Exception as e:
        st.error(f"Error executing custom code: {e}")
        return df

# Natural Language to Python Code
@st.cache_data
def generate_code_from_prompt(prompt):
    """
    Generates Python code based on a natural language prompt.

    Parameters:
    - prompt (str): User-provided description of the operation

    Returns:
    - str: Generated Python code
    """
    # Basic implementation: Can be replaced with an AI model integration
    if "top 10 records" in prompt.lower():
        return "df = df.head(10)"
    elif "remove nulls" in prompt.lower():
        return "df = df.dropna()"
    elif "add column" in prompt.lower():
        return "df['new_column'] = 0"
    else:
        return "# Custom operation: modify df as needed"

# Dynamic Visualization Options

def show_visualization_options():
    st.sidebar.header("Visualization Options")
    default = st.sidebar.selectbox(
        "Recommended Visualizations", ["Confusion Matrix", "ROC Curve", "Residuals Plot"]
    )
    other_options = st.sidebar.multiselect(
        "Additional Visualizations", ["Precision-Recall Curve", "Error Distribution"]
    )
    return default, other_options
