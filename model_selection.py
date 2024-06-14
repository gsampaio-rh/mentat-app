from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def linear_regression(X_train, y_train, features, target):
    X = X_train[features]
    y = y_train[target]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mse, r2


def random_forest(X_train, y_train, features, target):
    X = X_train[features]
    y = y_train[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mse, r2


def random_forest_with_hyperparameter_tuning(X_train, y_train, features, target):
    X = X_train[features]
    y = y_train[target]

    # Expanded parameter grid for more thorough search
    rf_param_grid = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 11),
        "bootstrap": [True, False],
        "max_features": ["auto", "sqrt", "log2"],
    }

    rf_random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=100,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    rf_random_search.fit(X, y)
    best_rf_model = rf_random_search.best_estimator_

    y_pred = best_rf_model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return best_rf_model, mse, r2


def xgboost_regressor(X_train, y_train, features, target):
    X = X_train[features]
    y = y_train[target]

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=100, random_state=42
    )
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mse, r2


def xgboost_with_hyperparameter_tuning(X_train, y_train, features, target):
    X = X_train[features]
    y = y_train[target]

    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=xgb_param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=2,  # Set verbosity level to 2
    )
    xgb_grid_search.fit(X, y)
    best_xgb_model = xgb_grid_search.best_estimator_

    y_pred = best_xgb_model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return best_xgb_model, mse, r2


def k_means_clustering(df, features, n_clusters):
    X = df[features]

    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)

    df["Cluster"] = clusters

    # plt.scatter(df[features[0]], df[features[1]], c=clusters, cmap="viridis")
    # plt.xlabel(features[0])
    # plt.ylabel(features[1])
    # plt.title("K-Means Clustering")
    # plt.show()

    return model, df
