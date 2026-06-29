import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

    import os
    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return preds, metrics