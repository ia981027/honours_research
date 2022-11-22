from pprint import pprint

import numpy as np
from geneal.genetic_algorithms import BinaryGenAlgSolver
from sklearn.metrics import accuracy_score

from src.ML_training.classification.NN.NN import NN
from src.ML_training.classification.clustering_analysis.dbscan import DBScan
from src.ML_training.classification.clustering_analysis.hierarchical_cluster import HierarchicalCluster
from src.ML_training.classification.clustering_analysis.kmeans import KMeans_
from src.ML_training.classification.regression.huber_regression import MyHuberRegressor
from src.ML_training.classification.regression.logistic_regression import MyLogisticRegressor
from src.ML_training.classification.regression.linear_regression import LinearRegression, MyLinearRegressor
from src.ML_training.classification.regression.ridge_regression import MyRidgeRegressor
from src.ML_training.classification.regression.sgd_classifier import MySGD
from src.ML_training.classification.trees_and_ensembles.ada_boost import ADABoostClassifier_
from src.ML_training.base_model import BaseModel
from src.ML_training.classification.trees_and_ensembles.decision_tree import DecisionTreeClassifier_
from src.ML_training.classification.trees_and_ensembles.bagging_classifier import BaggingClassifier_
from src.ML_training.classification.trees_and_ensembles.extra_tree_classifier import ExtraTreeClassifier_
from src.ML_training.classification.trees_and_ensembles.gradient_boosting import GradientBoostingClassifier_
from src.ML_training.classification.trees_and_ensembles.random_forest import RandomForestClassifier_
from src.ML_training.classification.trees_and_ensembles.xgb import XGBoost
from src.ML_training.classification.trees_and_ensembles.xgbrf import XGBRFClassifier_

from src.utils.data_processor import get_credit_ratings_train_test_split


def ga():
    x_train, x_test, y_train, y_test, x_validation, y_validation, columns = get_credit_ratings_train_test_split()

    models: {str, BaseModel} = {
        # "NN": NN(x_train, x_test, y_train, y_test),
        "LinearRegression": MyLinearRegressor(x_train, x_test, y_train, y_test),
        "HuberRegression": MyHuberRegressor(x_train, x_test, y_train, y_test),
        "RidgeRegressor": MyRidgeRegressor(x_train, x_test, y_train, y_test),
        "LogisticRegression": MyLogisticRegressor(x_train, x_test, y_train, y_test),
        "SGD": MySGD(x_train, x_test, y_train, y_test),
        # "XGBoost": XGBoost(x_train, x_test, y_train, y_test),
        # "XGBRFClassifier": XGBRFClassifier_(x_train, x_test, y_train, y_test),
        # "DecisionTreeClassifier": DecisionTreeClassifier_(x_train, x_test, y_train, y_test),
        # "BaggingClassifier": BaggingClassifier_(x_train, x_test, y_train, y_test),
        # "RandomForestClassifier": RandomForestClassifier_(x_train, x_test, y_train, y_test),
        # "ADABoost": ADABoostClassifier_(x_train, x_test, y_train, y_test),
        # "GradientBoosting": GradientBoostingClassifier_(x_train, x_test, y_train, y_test),
        # "ExtraTreeClassifier": ExtraTreeClassifier_(x_train, x_test, y_train, y_test),
        # "HierarchicalCluster": HierarchicalCluster(x_train, x_test, y_train, y_test),
        # "DBScan": DBScan(x_train, x_test, y_train, y_test),
        # "KMeans": KMeans_(x_train, x_test, y_train, y_test)
    }

    #
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_validation.shape, y_validation.shape)

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

        except ValueError:
            print(f"skipping {model}")
        print("End of testing")
        print("-" * 50)
        print("\n\n")


def get_model() -> BaseModel:
    X_train, X_test, Y_train, Y_test, columns = get_credit_ratings_train_test_split()
    indices = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    X_train = X_train[:, indices]
    X_test = X_test[:, indices]
    pprint(columns)
    print("\n\n")
    pprint(columns[indices])
    model = DecisionTreeClassifier_.get_model(X_train, Y_train, indices)
    print(accuracy_score(Y_test, model.predict(X_test)))


if __name__ == '__main__':
    print("classification")
    ga()
