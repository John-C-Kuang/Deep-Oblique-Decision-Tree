import pandas as pd
import numpy as np
import data_partition as dp
import time
from sklearn.model_selection import train_test_split
from multi_process import multi_process_tuning as tn
from preprocess import preprocess_wine_quality
from itertools import product
from sklearn.metrics import accuracy_score
from src.utils import ml_utils
from tree import DODTree

# local
import warnings


class TuneDODT:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, label_col: str):
        """
        Hyperparameter tuning class for the KNN classification algorithm. Uses multithreading to increase performance.
        @param train: the training set
        @param test: the testing set
        @param label_col: the column that represents the label
        """
        self.train = train
        self.test = test
        self.label_col = label_col
        ml_utils.numpy()

    def tune(self, ff_dim_range: range, momentum_range: list, max_depth_range: range,
             target_impurity_range: list, num_epochs_range: range, learning_rate_range: list,
             reg_range: list, processes: int, random_state: int = 42) -> dict:
        combinations = list(product(
            ff_dim_range, momentum_range, max_depth_range, target_impurity_range,
            num_epochs_range, learning_rate_range, reg_range))
        validation_result = tn.tune(task_function=self._batch_train_n_predict,
                                    tasks_param=combinations, max_active_processes=processes)
        hyper_params = max(validation_result, key=lambda data: data["accuracy"])

        test = self.test.to_numpy()
        x_test = test[:, :-1]
        y_test = test[:, -1]
        meta_tree = hyper_params["tree"]
        test_pred = [meta_tree.predict(features=x_test[i]) for i in range(len(x_test))]
        accuracy = accuracy_score(y_test, test_pred)
        return {
            "accuracy": hyper_params["accuracy"], "ff_dim": hyper_params["ff_dim"],
            "momentum": hyper_params["momentum"], "max_depth": hyper_params["max_depth"],
            "target_impurity": hyper_params["target_impurity"], "num_epochs": hyper_params["num_epochs"],
            "learning_rate": hyper_params["learning_rate"], "reg": hyper_params["reg"]
        }

    def _batch_train_n_predict(
            self, ff_dim: int, momentum: float, max_depth: int, target_impurity: float, num_epochs: int = 1000,
            learning_rate: float = 0.001, reg: float = 0.0, random_state: int = 42) -> dict:
        x_train, x_valid, y_train, y_valid = dp.partition_data(
            df=self.train, label_col=self.label_col, test_set_prop=0.2, random_state=random_state
        )
        # convert to ndarray
        x_train = x_train.to_numpy()
        x_valid = x_valid.to_numpy()
        y_train = y_train.to_numpy()
        y_valid = y_valid.to_numpy()

        meta_tree = DODTree()
        meta_tree.train(
            ff_dim=ff_dim, momentum=momentum, max_depth=max_depth, target_impurity=target_impurity,
            num_epochs=num_epochs, learning_rate=learning_rate, reg=reg,
            train=np.concatenate((x_train, y_train[:, np.newaxis]), axis=-1)
        )

        valid_pred = [meta_tree.predict(features=x_valid[i])
                      for i in range(len(x_valid))]
        accuracy = accuracy_score(y_valid, valid_pred)
        return {
            "accuracy": accuracy, "ff_dim": ff_dim, "momentum": momentum, "max_depth": max_depth,
            "target_impurity": target_impurity, "num_epochs": num_epochs, "learning_rate": learning_rate,
            "reg": reg, "tree": meta_tree
        }


def main():
    df = pd.read_csv('./../../dataset/Wine_Quality_Data.csv')
    df = preprocess_wine_quality(df)
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    tuner = TuneDODT(train=train, test=test, label_col="quality")
    start = time.time()
    result = tuner.tune(
        ff_dim_range=range(12, 30, 4),
        momentum_range=[0.7, 0.8, 0.9],
        max_depth_range=range(15, 25, 3),
        target_impurity_range=[0, 0.1, 0.2],
        num_epochs_range=range(200, 600, 200),
        learning_rate_range=[1e-5, 1e-6],
        reg_range=[0, 0.1, 0.2],
        processes=5
    )
    end = time.time()
    print(result)
    print("Time(sec): ")
    print(end - start)


# Must include for multiprocess to work
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
