import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    # =========================
    # 1. 기본 성능 지표
    # =========================
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    # =========================
    # 2. 결과 저장 구조
    # =========================
    metrics = {
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # =========================
    # 3. 해석 (핵심 업그레이드)
    # =========================
    print("\n================ MODEL REPORT ================")

    print("\n[OVERALL PERFORMANCE]")
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro F1 : {report['macro avg']['f1-score']:.3f}")

    print("\n[CLASS PERFORMANCE]")
    for cls, vals in report.items():
        if cls in ["0", "1", "2"]:
            print(
                f"Class {cls} → "
                f"P:{vals['precision']:.2f} "
                f"R:{vals['recall']:.2f} "
                f"F1:{vals['f1-score']:.2f}"
            )

    print("\n[ERROR ANALYSIS]")
    print("Confusion Matrix:")
    print(cm)

    # =========================
    # 4. 핵심 인사이트 생성
    # =========================
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)

    max_error = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    print("\n[KEY INSIGHT]")
    print(f"Most confusion between class {max_error[0]} ↔ {max_error[1]}")

    print("\n[SAVED ARTIFACTS]")
    print("- reports/metrics.json")

    print("\n==============================================")

    return preds, metrics