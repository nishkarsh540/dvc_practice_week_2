import unittest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestModel(unittest.TestCase):
    def test_accuracy(self):
        df = pd.read_csv("data/iris.csv")
        X = df.drop("species", axis=1)
        y = df["species"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

        model = joblib.load("model.joblib")
        y_pred = model.predict(X_test)

        self.assertGreater(accuracy_score(y_test, y_pred), 0.7)


if __name__ == "__main__":
    unittest.main()
