from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    # 저장
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ===== PRINT IMPROVED REPORT =====
    print("\n================ MODEL REPORT ================")
    print(f"\n[OVERALL PERFORMANCE]")
    print(f"Accuracy : {acc:.2f}")
    print(f"Macro F1 : {report['macro avg']['f1-score']:.2f}")

    print(f"\n[CLASS PERFORMANCE]")
    for cls, vals in report.items():
        if cls in ["0", "1", "2"]:
            print(f"Class {cls}: Precision {vals['precision']:.2f} / Recall {vals['recall']:.2f} / F1 {vals['f1-score']:.2f}")

    print(f"\n[CONFUSION MATRIX]")
    print(cm)

    print("\n[SAVED ARTIFACTS]")
    print("- reports/metrics.json")

    return preds, metrics