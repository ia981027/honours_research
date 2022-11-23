import numpy as np
from geneal.genetic_algorithms import BinaryGenAlgSolver
from sys import path
path.append("/home/vhutali/Desktop/projects/python/ML_finance_project_fast_api/")
print(path)

from src.ML_training.base_model import BaseModel

from src.ML_training.regression.regression.huber_regression import MyHuberRegressor
from src.ML_training.regression.regression.linear_regression import MyLinearRegressor
from src.ML_training.regression.regression.sgd_classifier import MySGD
from src.ML_training.regression.trees_and_ensembles.ada_boost import ADABoostRegression
from src.ML_training.regression.trees_and_ensembles.bagging import BaggingRegression
from src.ML_training.regression.trees_and_ensembles.decision_tree import DecisionTreeRegression
from src.ML_training.regression.trees_and_ensembles.extra_tree import ExtraTreeRegression
from src.ML_training.regression.trees_and_ensembles.gradient_boosting import GradientBoostingRegression
from src.ML_training.regression.trees_and_ensembles.random_forest import RandomForestRegression
from src.ML_training.regression.trees_and_ensembles.xgb import XGBoostRegression
from src.ML_training.regression.trees_and_ensembles.xgbrf import XGBRFRegression

from src.utils.data_processor import get_pct_change_train_test_split



x_train, x_test, y_train, y_test, x_test, y_test, columns = get_pct_change_train_test_split()
models: {str, BaseModel} = {
    "LinearRegression": MyLinearRegressor(x_train, x_test, y_train, y_test),
    "HuberRegression": MyHuberRegressor(x_train, x_test, y_train, y_test),
    "SGD": MySGD(x_train, x_test, y_train, y_test),
    "XGBoostRegression": XGBoostRegression(x_train, x_test, y_train, y_test),
    "XGBRFRegression": XGBRFRegression(x_train, x_test, y_train, y_test),
    "DecisionTreeRegression": DecisionTreeRegression(x_train, x_test, y_train, y_test),
    "BaggingRegression": BaggingRegression(x_train, x_test, y_train, y_test),
    "ADABoostRegression": ADABoostRegression(x_train, x_test, y_train, y_test),
    "GradientBoostingRegression": GradientBoostingRegression(x_train, x_test, y_train, y_test),
    "ExtraTreeRegression": ExtraTreeRegression(x_train, x_test, y_train, y_test),
    "RandomForestRegression": RandomForestRegression(x_train, x_test, y_train, y_test),
}

def ga():

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_test.shape, y_test.shape)
    for model in models:
        print("-" * 50)
        print(f"Testing {model}")
        print("-" * 50)
        try:
            ga_solver = BinaryGenAlgSolver(
                n_genes=x_train.shape[1],
                fitness_function=models[model].score,
                n_bits=1,
                pop_size=40,
                max_gen=100,
                mutation_rate=0.05,
                selection_strategy="roulette_wheel",
                verbose=True,
                selection_rate=0.6,
            )

            ga_solver.solve()

        except KeyError:
            print(f"skipping {model}")
        print("End of testing")
        print("-" * 50)
        print("\n\n")


def get_model(algorithm: str) -> BaseModel:
    try:
        return models[algorithm]
    except KeyError:
        raise Exception("No model found")

if __name__ == '__main__':
    print("Regression GA")
    ga()
