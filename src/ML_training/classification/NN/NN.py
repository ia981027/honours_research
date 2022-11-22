import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

from src.ML_training.base_model import BaseModel


class NN(BaseModel):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)

    def score(self, indices: np.array):
        indices = np.array(indices, dtype=bool)
        x_train = self.X_train[:, indices]
        if x_train.shape[1] == 0:
            return 0
        x_test = self.X_test[:, indices]
        model = Sequential()
        model.add(Dense(x_train.shape[1], activation='softsign'))
        model.add(Dense(2 * x_train.shape[1], activation='elu'))
        model.add(Dense(2 * x_train.shape[1], activation='selu'))
        model.add(Dense(20, activation='elu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.fit(x_train, self.Y_train, epochs=200, batch_size=250, verbose=0)
        score = model.evaluate(x_test, self.Y_test)
        return score[1]
