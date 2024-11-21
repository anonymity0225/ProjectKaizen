import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np


def visualize_model_performance(task_type, models, X_test, y_test, y_preds):
    try:
        if task_type == 'classification':
            visualize_classification(models, X_test, y_test, y_preds)
        elif task_type == 'regression':
            visualize_regression(models, X_test, y_test, y_preds)
        else:
            raise ValueError("Invalid task type for visualization. Choose 'classification' or 'regression'.")
    except Exception as e:
        print(f"Error during visualization: {e}")


### Classification Visualizations
def visualize_classification(models, X_test, y_test, y_preds):
    for model_name, y_pred in y_preds.items():
        print(f"\nVisualizing performance for {model_name}...")

        # Confusion Matrix
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        # ROC and Precision-Recall Curves (One-vs-Rest for Multi-Class)
        classes = np.unique(y_test)
        y_test_binarized = label_binarize(y_test, classes=classes)
        if len(classes) > 2:  # Multi-class setting
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred == class_label)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f'Class {class_label} (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.title(f'Multi-Class ROC Curve for {model_name} (Class {class_label})')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.show()
        else:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.title(f'ROC Curve for {model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

        # Precision-Recall Curve
        if len(classes) == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            plt.figure()
            plt.plot(recall, precision, marker='.', color='purple')
            plt.title(f'Precision-Recall Curve for {model_name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.tight_layout()
            plt.show()


### Regression Visualizations
def visualize_regression(models, X_test, y_test, y_preds):
    for model_name, y_pred in y_preds.items():
        print(f"\nVisualizing performance for {model_name}...")

        # Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residual Plot for {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.show()

        # Error Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True, color='blue')
        plt.title(f'Error Distribution for {model_name}')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()

        # Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Actual vs Predicted for {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.tight_layout()
        plt.show()
