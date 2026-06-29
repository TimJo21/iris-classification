from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, seed=42):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, iris.feature_names