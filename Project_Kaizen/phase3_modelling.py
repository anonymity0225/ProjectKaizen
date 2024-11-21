from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import warnings
import pandas as pd

# Suppress warnings for better output clarity
warnings.filterwarnings("ignore")

# Helper to detect if classes are imbalanced
def is_imbalanced(y):
    class_counts = pd.Series(y).value_counts(normalize=True)
    return any(class_counts < 0.1)  # Example threshold for imbalance

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Classification model selection and training
def classification_model_selection(X_train, X_test, y_train, y_test):
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
    }
    model_results = []
    for model_name, model in classifiers.items():
        param_grid = {
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
            'Logistic Regression': {'C': [0.1, 1, 10], 'max_iter': [100, 200]},
            'SVC': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'Decision Tree': {'max_depth': [5, 10]},
        }
        # Handle class imbalance if needed
        if is_imbalanced(y_train) and model_name in ['Random Forest', 'Logistic Regression']:
            model.set_params(class_weight='balanced')

        try:
            grid = GridSearchCV(model, param_grid.get(model_name, {}), cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            f1_avg = 'binary' if len(set(y_train)) == 2 else 'weighted'
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average=f1_avg)

            model_results.append({
                'Model': model_name,
                'Best Parameters': grid.best_params_,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Classification Report': classification_report(y_test, y_pred, output_dict=True),
            })
        except Exception as e:
            print(f"An error occurred with model {model_name}: {e}")
    return model_results

# Regression model selection and training
def regression_model_selection(X_train, X_test, y_train, y_test):
    regressors = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
    }
    model_results = []
    for model_name, model in regressors.items():
        param_grid = {
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
            'SVR': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'Decision Tree': {'max_depth': [5, 10]},
        }
        if model_name == 'SVR':
            X_train, X_test = scale_data(X_train, X_test)

        try:
            grid = GridSearchCV(model, param_grid.get(model_name, {}), cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model_results.append({
                'Model': model_name,
                'Best Parameters': grid.best_params_,
                'MSE': mse,
                'MAE': mae,
                'R2 Score': r2,
            })
        except Exception as e:
            print(f"An error occurred with model {model_name}: {e}")
    return model_results

# Main function for task selection and execution
def run_model_building(df, target_column, task_type=None):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Automatically determine task type if not provided
    if task_type is None:
        task_type = 'classification' if y.nunique() <= 10 else 'regression'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        return classification_model_selection(X_train, X_test, y_train, y_test)
    elif task_type == 'regression':
        return regression_model_selection(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")
