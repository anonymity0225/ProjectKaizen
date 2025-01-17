import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from mpl_toolkits.mplot3d import Axes3D

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Helper function to save the plot
def save_plot(fig, filename, directory="visualizations", file_format="png"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.{file_format}")
    fig.savefig(path, format=file_format, bbox_inches='tight')
    print(f"Plot saved to: {path}")

# Dataset Overview Visualizations
def plot_overview(df, save=False):
    """
    Generate dataset overview plots including histograms, count plots, and heatmap of missing values.
    """
    # Histograms for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        if save:
            save_plot(plt.gcf(), f"histogram_{col}")
        plt.show()

    # Count plots for categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        if save:
            save_plot(plt.gcf(), f"countplot_{col}")
        plt.show()

    # Heatmap of missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Heatmap of Missing Values")
    if save:
        save_plot(plt.gcf(), "missing_values_heatmap")
    plt.show()

# Univariate Analysis
def plot_univariate(df, column, save=False):
    """
    Create univariate analysis plots for the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    if pd.api.types.is_numeric_dtype(df[column]):
        # Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde=True, bins=30, color='green')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        if save:
            save_plot(plt.gcf(), f"histogram_{column}")
        plt.show()

        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[column], color='cyan')
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        if save:
            save_plot(plt.gcf(), f"boxplot_{column}")
        plt.show()

    elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
        # Pie chart
        pie_data = df[column].value_counts()
        plt.figure(figsize=(8, 5))
        pie_data.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        plt.title(f"Pie Chart of {column}")
        if save:
            save_plot(plt.gcf(), f"piechart_{column}")
        plt.show()
    else:
        print(f"Unsupported data type for column '{column}'.")

# Bivariate Analysis
def plot_bivariate(df, x, y, save=False):
    """
    Generate bivariate analysis plots for specified x and y columns.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' or '{y}' not found in dataset.")

    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        # Scatter plot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=x, y=y, data=df, color='blue', alpha=0.7)
        plt.title(f"Scatter Plot: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        if save:
            save_plot(plt.gcf(), f"scatter_{x}_vs_{y}")
        plt.show()

    elif pd.api.types.is_categorical_dtype(df[x]):
        # Bar plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=x, y=y, data=df, palette='coolwarm')
        plt.title(f"Bar Plot: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        if save:
            save_plot(plt.gcf(), f"barplot_{x}_vs_{y}")
        plt.show()

    else:
        print(f"Unsupported data types for columns '{x}' and '{y}'.")

# Correlation and Multivariate Analysis
def plot_correlation(df, save=False):
    """
    Generate correlation heatmap for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    if save:
        save_plot(plt.gcf(), "correlation_heatmap")
    plt.show()

# Custom Code Interface
def custom_code_interface(df):
    """
    Provide an interactive interface for users to run custom code on the dataset.
    """
    print("\nEnter your Python code to interact with the dataset.")
    print("The dataset is available as the variable 'df'.")
    print("Type 'exit' to leave the interface.")

    while True:
        user_input = input(">>> ")
        if user_input.strip().lower() == 'exit':
            print("Exiting custom code interface.")
            break
        try:
            exec(user_input, {'df': df, 'pd': pd, 'np': np, 'sns': sns, 'plt': plt})
        except Exception as e:
            print(f"Error executing your code: {e}")

'''# Example Usage
if __name__ == "__main__":
    # Load example dataset
    try:
        file_path = input("Enter the path to your dataset (CSV or Excel): ")
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        print("Dataset loaded successfully!")

        # Example visualizations
        plot_overview(data, save=True)
        plot_univariate(data, column=data.columns[0], save=True)
        plot_bivariate(data, x=data.columns[0], y=data.columns[1], save=True)
        plot_correlation(data, save=True)

        # Launch custom code interface
        custom_code_interface(data)

    except Exception as e:
        print(f"Error: {e}")'''
