from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress warnings for better output clarity
warnings.filterwarnings("ignore")

def is_imbalanced(y):
    """
    Checks if the target variable is imbalanced.
    
    Parameters:
    - y (pd.Series or np.ndarray): Target variable
    
    Returns:
    - bool: True if any class occurs less than 10% of the time
    """
    class_counts = pd.Series(y).value_counts(normalize=True)
    return any(class_counts < 0.1)

def scale_data(X_train, X_test):
    """
    Scales training and test datasets using StandardScaler.
    
    Parameters:
    - X_train (pd.DataFrame): Training features
    - X_test (pd.DataFrame): Test features

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Scaled training and test datasets
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def select_model_classification(models, X_train, X_test, y_train, y_test, use_random_search=False):
    """
    Performs classification model selection and training.

    Parameters:
    - models (dict): Dictionary of model names and instantiated models.
    - X_train, X_test, y_train, y_test: Train and test data splits.
    - use_random_search (bool): Use RandomizedSearchCV instead of GridSearchCV.

    Returns:
    - list[dict]: Model performance results.
    """
    model_results = []
    for model_name, model in models.items():
        param_grid = {
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
            'Logistic Regression': {'C': [0.1, 1, 10], 'max_iter': [100, 200]},
            'SVC': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'Decision Tree': {'max_depth': [5, 10]},
            'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        }

        if is_imbalanced(y_train) and model_name in ['Random Forest', 'Logistic Regression', 'XGBoost']:
            model.set_params(class_weight='balanced')

        search = RandomizedSearchCV if use_random_search else GridSearchCV

        try:
            logging.info(f"Training {model_name}...")
            grid = search(model, param_grid.get(model_name, {}), cv=3, n_jobs=-1, verbose=1)
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
            logging.error(f"Error training {model_name}: {e}")
            model_results.append({
                'Model': model_name,
                'Error': str(e),
            })

    return model_results

def select_model_regression(models, X_train, X_test, y_train, y_test, use_random_search=False):
    """
    Performs regression model selection and training.

    Parameters:
    - models (dict): Dictionary of model names and instantiated models.
    - X_train, X_test, y_train, y_test: Train and test data splits.
    - use_random_search (bool): Use RandomizedSearchCV instead of GridSearchCV.

    Returns:
    - list[dict]: Model performance results.
    """
    model_results = []
    for model_name, model in models.items():
        param_grid = {
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
            'SVR': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'Decision Tree': {'max_depth': [5, 10]},
            'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        }

        if model_name == 'SVR':
            X_train, X_test = scale_data(X_train, X_test)

        search = RandomizedSearchCV if use_random_search else GridSearchCV

        try:
            logging.info(f"Training {model_name}...")
            grid = search(model, param_grid.get(model_name, {}), cv=3, n_jobs=-1, verbose=1)
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
            logging.error(f"Error training {model_name}: {e}")
            model_results.append({
                'Model': model_name,
                'Error': str(e),
            })

    return model_results

def interactive_code_editor(data):
    """
    Allows the user to interact with the data via code snippets or natural language prompts.
    
    Parameters:
    - data (pd.DataFrame): Input dataset

    Returns:
    - pd.DataFrame: Modified dataset
    """
    print("Interactive Data Editor")
    print("Choose an option:")
    print("1. Write Python code")
    print("2. Describe what you want to do (e.g., 'filter rows where column A > 5')")
    choice = input("Your choice: ")

    if choice == "1":
        print("Write your Python code to manipulate the dataset. Use 'data' as the DataFrame variable.")
        code = input("Enter your code: \n")
        try:
            exec(code, {'data': data})
        except Exception as e:
            print(f"Error in executing code: {e}")
    elif choice == "2":
        prompt = input("Describe your operation: ")
        print(f"Interpreting prompt: {prompt}")
        print("Feature to convert prompt to code is under development.")

    return data

def run_model_building(df, target_column, task_type=None, selected_models=None, use_random_search=False):
    """
    Runs model selection and training.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - target_column (str): Target variable column name
    - task_type (str): Type of task ('classification' or 'regression')
    - selected_models (list): User-specified models to train on
    - use_random_search (bool): Use RandomizedSearchCV for optimization

    Returns:
    - list[dict]: Results of the model selection
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Detect task type if not specified
    if task_type is None:
        task_type = 'classification' if y.nunique() <= 10 else 'regression'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Filter models based on user input
    all_models = {
        'classification': {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'SVC': SVC(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'XGBoost': XGBClassifier(),
            'AdaBoost': AdaBoostClassifier(),
        },
        'regression': {
            'Random Forest': RandomForestRegressor(),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'XGBoost': XGBRegressor(),
            'AdaBoost': AdaBoostRegressor(),
        }
    }

    models = all_models[task_type]
    if selected_models:
        models = {name: model for name, model in models.items() if name in selected_models}

    if task_type == 'classification':
        return select_model_classification(models, X_train, X_test, y_train, y_test, use_random_search)
    elif task_type == 'regression':
        return select_model_regression(models, X_train, X_test, y_train, y_test, use_random_search)
    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")
