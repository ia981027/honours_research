from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_test = y_test

    @abstractmethod
    def score(self,arg):
        pass

