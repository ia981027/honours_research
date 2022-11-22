from math import inf

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

import numpy as np

from src.ML_training.base_model import BaseModel


class DecisionTreeRegression(BaseModel):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.model = None

    def score(self, arg):
        arg = np.array(arg, dtype=bool)
        x_train = self.X_train[:, arg]
        if x_train.shape[1] == 0:
            return -inf
        x_test = self.X_test[:, arg]
        self.model = DecisionTreeRegressor()
        self.model.fit(x_train, self.Y_train)
        score = mean_squared_error(self.Y_test, self.model.predict(x_test))
        return -score
        return score

    @classmethod
    def get_model(cls, x_train, y_train, indices):
        assert x_train.shape[1] != 0, "No features selected"

        model = DecisionTreeClassifier().fit(x_train, y_train)
        return model
