from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluates trained models and returns results.
    """
    accuracies = []
    mses = []
    for name, model in models:
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        accuracies.append(accuracy)
        mses.append(mse)
        print(f"{name} Accuracy:", accuracy)
        print(f"{name} Mean Squared Error:", mse)
    return accuracies, mses
