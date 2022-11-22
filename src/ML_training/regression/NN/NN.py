import math

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

from src.ML_training.base_model import BaseModel


class NN(BaseModel):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)

    def score(self, arg):
        arg = np.array(arg, dtype=bool)
        x_train = self.X_train[:, arg]
        if x_train.shape[1] == -math.inf:
            return 0
        x_test = self.X_test[:, arg]
        model = Sequential()
        model.add(Dense(x_train.shape[1], activation='selu'))
        model.add(Dense(2 * x_train.shape[1], activation='elu'))
        model.add(Dense(1, activation='relu'))  # Can't go below zero percent
        model.compile(loss='log_cosh', optimizer='sgd', metrics=['accuracy', 'mse', 'mae'])
        model.fit(x_train, self.Y_train, epochs=200, batch_size=250, verbose=0)
        score = model.evaluate(x_test, self.Y_test)
        return -score[2]
