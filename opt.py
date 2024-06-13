import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Select only numeric columns for imputation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle missing values for numeric columns
    imputer = SimpleImputer(strategy='median')
    numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    
    # Power transform to handle skewness
    pt = PowerTransformer()
    numeric_df[numeric_df.columns] = pt.fit_transform(numeric_df[numeric_df.columns])
    
    # Replace the original numeric columns with the processed ones
    df[numeric_df.columns] = numeric_df
    
    return df

def feature_engineering(df):
    # Create CPU utilization ratio
    df["CPU Utilization Ratio"] = df["CPU Usage (seconds)"] / df["CPU Utilization (%)"]

    # Create memory utilization ratio
    df["Memory Utilization Ratio"] = df["Memory Usage (bytes)"] / df["Memory Utilization (%)"]

    # Create network I/O ratio
    df["Network I/O Ratio"] = (df["Network Receive (bytes)"] + df["Network Transmit (bytes)"]) / df["Network I/O Throughput (Mbps)"]

    # Create storage I/O ratio
    df["Storage I/O Ratio"] = (df["Storage Read (bytes)"] + df["Storage Write (bytes)"]) / df["Disk I/O Throughput (MB/s)"]

    # Create system load per core (assuming 4 cores for this example)
    df["System Load Per Core"] = df["System Load Average"] / 4

    # Indicator for peak usage times (example: usage above 90%)
    df["Peak CPU Usage"] = (df["CPU Utilization (%)"] > 90).astype(int)
    df["Peak Memory Usage"] = (df["Memory Utilization (%)"] > 90).astype(int)
    df["Peak Network Usage"] = (df["Network I/O Throughput (Mbps)"] > 90).astype(int)
    df["Peak Storage Usage"] = (df["Disk I/O Throughput (MB/s)"] > 90).astype(int)

    return df


def train_bottleneck_model(df):
    # Define target variable (bottleneck indicator: 1 if any peak usage, else 0)
    df["Bottleneck"] = df[
        [
            "Peak CPU Usage",
            "Peak Memory Usage",
            "Peak Network Usage",
            "Peak Storage Usage",
        ]
    ].max(axis=1)

    # Ensure feature columns are consistent
    feature_cols = [
        "CPU Utilization Ratio",
        "Memory Utilization Ratio",
        "Network I/O Ratio",
        "Storage I/O Ratio",
        "System Load Per Core",
    ]
    X = df[feature_cols]
    y = df["Bottleneck"]

    # Split the data into training and testing sets, keep the indices
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test_indices = X_test.index

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Random Forest model with GridSearchCV for hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # Predict bottlenecks on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluate the model, handling the case where only one class is present
    if len(np.unique(y_test)) == 1:
        print(
            "Only one class present in y_test. Skipping ROC AUC, Precision, Recall, and F1 scores."
        )
        roc_auc = precision = recall = f1 = "N/A"
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

    return best_rf, feature_cols, X_test, y_test, y_pred, X_test_indices


def plot_ellipses(gmm, ax):
    for n, color in enumerate(["r", "g", "b"]):
        if n >= len(gmm.means_):
            continue
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = plt.matplotlib.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=angle, color=color, alpha=0.3
        )
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)


def plot_classification_results_with_clusters(
    X_test, y_test, y_pred, df, X_test_indices
):
    if "Number of Servers" not in df.columns:
        print(
            "Warning: 'Number of Servers' column not found. Adding placeholder values."
        )
        df["Number of Servers"] = np.random.randint(1, 100, size=len(df))
    else:
        if (
            df["Number of Servers"].isnull().any()
            or (df["Number of Servers"] <= 0).any()
        ):
            print(
                "Warning: 'Number of Servers' column has invalid values. Replacing with placeholder values."
            )
            df["Number of Servers"] = np.random.randint(1, 100, size=len(df))

    n_components = min(len(X_test), 3)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_test[:, :2])

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sizes = df.loc[X_test_indices, "Number of Servers"] * 10
    scatter = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pred,
        s=sizes,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="w",
        label=y_pred,
    )
    plot_ellipses(gmm, ax)
    plt.xlabel("CPU Utilization Ratio")
    plt.ylabel("Memory Utilization Ratio")
    plt.title("Random Forest Classification Results with Server Counts and Clusters")

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend1 = ax.legend(handles, labels, loc="upper right", title="Number of Servers")
    ax.add_artist(legend1)

    handles, labels = scatter.legend_elements()
    legend2 = ax.legend(
        handles, ["Non-Bottleneck", "Bottleneck"], loc="upper left", title="Prediction"
    )
    ax.add_artist(legend2)

    plt.show()


def visualize_bottlenecks(df, model):
    feature_cols = ["CPU Utilization Ratio", "Memory Utilization Ratio", "Network I/O Ratio", "Storage I/O Ratio", "System Load Per Core"]
    df["Predicted Bottleneck"] = model.predict(df[feature_cols])

    bottleneck_counts = df["Predicted Bottleneck"].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=bottleneck_counts.index, y=bottleneck_counts.values, palette="viridis"
    )

    plt.title("Predicted Resource Bottlenecks", fontsize=16)
    plt.xlabel("Bottleneck", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    for index, value in enumerate(bottleneck_counts.values):
        plt.text(index, value + 10, str(value), ha="center", fontsize=12)

    plt.xticks([0, 1], ["No Bottleneck", "Bottleneck"], fontsize=12)

    explanation_text = (
        "Bottleneck (1): A predicted resource constraint\n"
        "that could impact system performance. If a bottleneck occurs,\n"
        "users may experience slower response times or interruptions.\n"
        "No Bottleneck (0): System is operating normally without\n"
        "resource constraints, ensuring smooth and efficient performance."
    )
    plt.gcf().text(
        0.75,
        0.9,
        explanation_text,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
        ha="left",
    )

    plt.show()

def plot_feature_importance(model, feature_cols):
    feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
    feature_importance = feature_importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title("Feature Importance in Bottleneck Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def main():
    # Load the dataset
    df = load_data("data/real_faang.csv")

    # Preprocess the data
    df = preprocess_data(df)

    # Feature engineering
    df = feature_engineering(df)

    # Train the bottleneck model
    model, feature_cols, X_test, y_test, y_pred, X_test_indices = (
        train_bottleneck_model(df)
    )

    # Visualize the results
    plot_classification_results_with_clusters(
        X_test, y_test, y_pred, df, X_test_indices
    )
    visualize_bottlenecks(df, model)
    plot_feature_importance(model, feature_cols)

if __name__ == "__main__":
    main()
