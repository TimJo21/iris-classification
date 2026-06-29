from src.utils import set_seed
from src.data import load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_feature_importance

def main():

    set_seed(42)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    model = train_model(X_train, y_train)

    preds, metrics = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, feature_names)

    print("=== MODEL TRAINING COMPLETE ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()