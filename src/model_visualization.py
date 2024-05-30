import matplotlib.pyplot as plt
import numpy as np

def visualize_results(models, accuracies, mses, avg_accuracy, avg_mse):
    """
    Visualizes results.
    """
    model_names = [name for name, _ in models]
    x = np.arange(len(model_names))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(model_names, accuracies, color='blue')
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', label='Avg Accuracy')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(model_names, mses, color='green')
    plt.axhline(y=avg_mse, color='orange', linestyle='--', label='Avg MSE')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.title('Model MSE')
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.plot(model_names, accuracies, marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(model_names, mses, marker='x', linestyle='--', color='green', label='MSE')
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', label='Avg Accuracy')
    plt.axhline(y=avg_mse, color='orange', linestyle='--', label='Avg MSE')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Results')
    plt.legend()
    plt.grid(True)

    plt.show()

