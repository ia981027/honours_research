import numpy as np
import typing
import matplotlib.pyplot as plt

from src.utils.data_processor import get_pct_change_train_test_split, get_credit_ratings_train_test_split

columns: typing.Optional[np.array] = None
used_columns = {}


def get_columns_used(indices: np.ndarray) -> None:
    assert len(indices) == len(columns), "indices not the same length as columns"
    print(columns[indices])
    for column in columns[indices]:
        try:
            used_columns[column] += 1
        except KeyError:
            used_columns[column] = 1
    print("\n\n")


def set_classification() -> None:
    global columns
    _, _, _, _, columns = get_credit_ratings_train_test_split()


def classification():
    set_classification()
    print(columns)
    indices = [
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, ], # ExtraTreeClassifier
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,], # GradientBoostingClassifier
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, ], # AdaBoostClassifier
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, ], # RandomForestClassifier
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, ], # BaggingClassifier
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, ], # DecisionTreeClassifier
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, ], # XGBRFClassifier
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, ], # XGBoostClassifier
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ], # LinearRegression
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, ], # KMeans
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], # DBScan
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, ], # HierarchicalCluster

    ]
    plot_indices(indices)


def plot_indices(indices):
    print(len(indices))
    indices = np.array(indices, dtype=bool)
    for index in indices:
        get_columns_used(index)
    plt.bar(range(len(used_columns)), list(used_columns.values()), align='center', tick_label=list(used_columns.keys()))
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    # plt.legend()
    plt.show()
    plt.pie(list(used_columns.values()), labels=list(used_columns.keys()))
    plt.show()
    print(used_columns)


def set_pct_columns() -> None:
    global columns
    _, _, _, _, columns = get_pct_change_train_test_split()
    columns = columns.drop(["change"])


def pct_change() -> None:
    set_pct_columns()
    indices = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # LinearRegression
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, ],
        # XGBoostRegression
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, ],
        # XGBRFRegression
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, ],
        # DecisionTreeRegression
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
        # BaggingRegression
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, ],
        # RandomForestRegression
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, ],
        # AdaBoostRegression
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, ],
        # GradientBoosting
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # ExtraTreeRegression

    ]

    plot_indices(indices)


if __name__ == '__main__':
    # while True:
    #     set_pct_columns()
    #     # print("\n\n")
    #     get_columns_used(np.array(np.array(input().replace("[", "").replace("]", "").split(",")[:-1], dtype=int), dtype=bool))
    #     # print(np.array(input().split(",")))
    #     print(used_columns.keys())
    #     print("\n\n")
    # pct_change()

    classification()
