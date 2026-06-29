import matplotlib.pyplot as plt
import numpy as np
import os

def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_
    idx = np.argsort(importances)

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure()
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.title("Feature Importance")

    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.close()