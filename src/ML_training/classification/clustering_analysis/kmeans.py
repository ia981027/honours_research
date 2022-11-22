import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from src.ML_training.base_model import BaseModel


class KMeans_(BaseModel):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.model = None

    def score(self, arg):
        arg = np.array(arg, dtype=bool)
        x_train = self.X_train[:, arg]
        if x_train.shape[1] == 0:
            return 0
        x_test = self.X_test[:, arg]
        self.model = KMeans(n_clusters=10, random_state=0).fit(x_train)
        score = accuracy_score(self.Y_test, self.model.predict(x_test))
        return score
