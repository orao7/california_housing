from src.data_processing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models
from src.model_visualization import visualize_results

def main():
    X_train, X_test, y_train, y_test = preprocess_data()

    models = train_models(X_train, y_train)

    accuracies, mses = evaluate_models(models, X_test, y_test)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_mse = sum(mses) / len(mses)

    print("Average Accuracy:", avg_accuracy)
    print("Average Mean Squared Error:", avg_mse)

    visualize_results(models, accuracies, mses, avg_accuracy, avg_mse)

if __name__ == "__main__":
    main()