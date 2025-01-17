import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

def detect_problem_type(df, target_column=None):
    if target_column is None:
        return 'clustering'
    target = df[target_column]
    if target.nunique() < 10 and target.dtype in ['object', 'category', 'bool'] or np.issubdtype(target.dtype, np.integer):
        return 'classification'
    if np.issubdtype(target.dtype, np.floating):
        return 'regression'
    raise ValueError("Unsupported target column data type.")

def split_dataset(df, target_column=None, test_size=0.2, val_size=0, random_state=42):
    if target_column is None:
        return train_test_split(df, test_size=test_size, random_state=random_state)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)
    if val_size > 0:
        val_split = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_temp, y_train, y_temp

def train_and_evaluate_models(df, target_column, problem_type, scoring=None, random_state=42):
    if problem_type == 'clustering':
        models = {
            'KMeans': KMeans(n_clusters=3, random_state=random_state),
            'DBSCAN': DBSCAN(),
            'Agglomerative Clustering': AgglomerativeClustering(n_clusters=3)
        }
        metrics = {}
        best_score = -np.inf
        best_model = None
        for name, model in models.items():
            try:
                model.fit(df)
                if name == 'KMeans':
                    score = silhouette_score(df, model.labels_)
                else:
                    score = davies_bouldin_score(df, model.labels_)
                metrics[name] = {'Score': score}
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                print(f"Error training model {name}: {e}")
        return metrics, best_model

    X_train, X_test, y_train, y_test = split_dataset(df, target_column, test_size=0.2)
    if problem_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=random_state),
            'Lasso': Lasso(random_state=random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'AdaBoost': AdaBoostRegressor(random_state=random_state),
            'KNN': KNeighborsRegressor()
        }
        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }

    elif problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'AdaBoost': AdaBoostClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=random_state)
        }
        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            metrics[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1-Score': f1_score(y_test, y_pred, average='weighted'),
                'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
            }

    best_model = max(metrics, key=lambda m: metrics[m].get('R2', 0) if problem_type == 'regression' else metrics[m]['Accuracy'])
    return metrics, models[best_model]

def save_model(model, filename='best_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def custom_interface(df):
    print("Custom Code Interface")
    print("Provide your custom analysis or transformations as Python code.")
    while True:
        code = input(">>> ")
        if code.lower() in ['exit', 'quit']:
            break
        try:
            exec(code)
        except Exception as e:
            print(f"Error in execution: {e}")
