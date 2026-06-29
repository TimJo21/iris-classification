import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    import numpy as np

    importance = model.feature_importances_
    idx = np.argsort(importance)

    plt.figure()
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.title("Feature Importance")
    plt.show()