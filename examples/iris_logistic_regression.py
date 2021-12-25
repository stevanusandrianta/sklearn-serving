from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)

joblib.dump(clf, "iris_logistic_regression.joblib")
