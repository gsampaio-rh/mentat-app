import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_model_performance(y_train, y_pred_train, y_test, y_pred_test, model_name):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.3)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Training Set")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Testing Set")

    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, features, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {model_name}")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals Plot - {model_name}")
    plt.tight_layout()
    plt.show()


def compare_model_performance(y_true, y_pred_before, y_pred_after, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred_before, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Before Hyperparameter Tuning")

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred_after, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("After Hyperparameter Tuning")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
