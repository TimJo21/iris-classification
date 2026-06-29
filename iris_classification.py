from src.utils import set_seed
from src.data import load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import (
    plot_feature_importance,
    plot_confusion_matrix
)

def print_report(metrics):

    print("\n==============================")
    print("IRIS CLASSIFICATION REPORT")
    print("==============================")

    print("\n[MODEL PERFORMANCE]")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\n[CLASS PERFORMANCE]")

    report = metrics["classification_report"]

    for cls, vals in report.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(f"Class {cls}: F1-score = {vals['f1-score']:.2f}")

    print("\n[INSIGHT]")
    print("- Petal features are expected to dominate classification.")
    print("- Class separability is high for Setosa.")
    print("- Versicolor and Virginica show overlap.")

    print("\n[SAVED OUTPUTS]")
    print("- models/iris_model.pkl")
    print("- reports/metrics.json")
    print("- reports/figures/feature_importance.png")
    print("- reports/figures/confusion_matrix.png")


def main():

    set_seed(42)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    model = train_model(X_train, y_train)

    preds, metrics = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(y_test, preds)

    print_report(metrics)


if __name__ == "__main__":
    main()