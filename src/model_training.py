from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def train_models(X_train, y_train):
    """
    Trains models on the training dataset.
    """
    models = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("SVR", SVR()),
        ("Linear Regression", LinearRegression()),
        ("KNN", KNeighborsRegressor()),
        ("Decision Tree", DecisionTreeRegressor())
    ]
    trained_models = []
    for name, model in models:
        model.fit(X_train, y_train)
        trained_models.append((name, model))
    return trained_models
