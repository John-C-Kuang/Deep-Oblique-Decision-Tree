import pandas as pd
import data_partition as dp
from thread import multi_thread_tuning as tn
from sklearn.metrics import accuracy_score
from src.utils import ml_utils
from typing import Callable


class Tune_Knn:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, label_col: str):
        self.train = train
        self.test = test
        self.label_col = label_col

    def tune(self, k_val: int, distance_funcs: list[Callable], random_state: int = 42) -> dict:

        function_args = []
        for k in range(k_val):
            function_args.append([k, random_state, distance_funcs])

        validation_result = tn.tune(task_function=self.__batch_train_n_predict,
                                    tasks_param=[function_args])
        best_valid_hyper_params = max(validation_result, key=lambda data: data[0])
        # split test data into x_test and y_test
        x_train, x_test, y_train, y_test = dp.partition_data(
            df=self.test, label_col=self.label_col, test_set_prop=0.2, random_state=random_state
        )

        best_valid_knn = ml_utils.experimental.KNN()
        best_valid_knn.train(feature_set=x_train, labels=y_train)
        test_pred = [best_valid_knn.predict(feature=x_test.loc[i], k=best_valid_hyper_params[1],
                                            dist_func=best_valid_hyper_params[2]) for i in range(x_test.shape[0])]
        test_accuracy = accuracy_score(y_test, test_pred)
        return {
            "validation accuracy": best_valid_hyper_params[0],
            "test accuracy": test_accuracy,
            "k": best_valid_hyper_params[1],
            "distance function": best_valid_hyper_params[2]
        }

    def __batch_train_n_predict(self, k: int, random_state: int,
                                distance_funcs: list[Callable]) -> (float, int, str):
        for dist_func in distance_funcs:
            dist = ml_utils.metric.entropy if dist_func == "entropy" else ml_utils.metric.gini
            knn = ml_utils.experimental.KNN()
            x_train, x_valid, y_train, y_valid = dp.partition_data(
                df=self.train, label_col=self.label_col, test_set_prop=0.1, random_state=random_state
            )
            knn.train(feature_set=x_train, labels=y_train)
            valid_pred = [knn.predict(feature=x_valid.loc[i], k=k, dist_func=dist)
                          for i in range(x_valid.shape[0])]
            accuracy = accuracy_score(y_valid, valid_pred)
            return accuracy, k, dist_func



# df = pd.read_csv("../../dataset/Wine_Quality_Data.csv")

