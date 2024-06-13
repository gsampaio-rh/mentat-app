from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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


def k_means_clustering(df, features, n_clusters):
    X = df[features]

    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)

    df["Cluster"] = clusters

    plt.scatter(df[features[0]], df[features[1]], c=clusters, cmap="viridis")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("K-Means Clustering")
    plt.show()

    return model, df
