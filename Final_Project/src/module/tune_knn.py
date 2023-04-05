import pandas as pd
import data_partition as dp
import src.utils.ml_utils.metric
from handle_process import multi_process_tuning as tn
from sklearn.metrics import accuracy_score
from src.utils import ml_utils
from typing import Callable


class Tune_Knn:

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

    def tune(self, k_val: int, distance_funcs: list[str], random_state: int = 42, processes: int = 4) -> dict:
        """
        Perform hyperparameter tuning for KNN
        @param k_val: the highest k-value of KNN to try
        @param distance_funcs: the distance function to use in KNN
        @param random_state: random seed
        @param processes: the number of threads allowed to tune
        @return: a report of the tuning
        """

        function_args = []
        for k in range(1, k_val + 1):
            function_args.append([k, random_state, distance_funcs])

        validation_result = tn.tune(task_function=self._batch_train_n_predict,
                                    tasks_param=function_args, max_active_processes=processes)
        best_valid_hyper_params = max(validation_result, key=lambda data: data[0])
        # split test data into x_test and y_test
        x_train, x_test, y_train, y_test = dp.partition_data(
            df=self.test, label_col=self.label_col, test_set_prop=0.2, random_state=random_state
        )

        best_valid_knn = ml_utils.experimental.KNN()
        best_valid_knn.train(feature_set=x_train.to_numpy(), labels=y_train.to_numpy())
        # could make dist_func more flexible if want to
        test_pred = [best_valid_knn.predict(feature=x_test.iloc[i].to_numpy(), k=best_valid_hyper_params[1],
                                            dist_func=src.utils.ml_utils.metric.CosineSimilarity) for i in
                     range(x_test.shape[0])]
        test_accuracy = accuracy_score(y_test, test_pred)
        return {
            "validation accuracy": best_valid_hyper_params[0],
            "test accuracy": test_accuracy,
            "k": best_valid_hyper_params[1],
            "distance function": best_valid_hyper_params[2]
        }

    def _batch_train_n_predict(self, k: int, random_state: int,
                                distance_funcs: list[Callable]) -> (float, int, str):
        ml_utils.numpy()
        for dist_func in distance_funcs:
            # could support other types if we really want to
            dist = src.utils.ml_utils.metric.CosineSimilarity
            knn = ml_utils.experimental.KNN()
            x_train, x_valid, y_train, y_valid = dp.partition_data(
                df=self.train, label_col=self.label_col, test_set_prop=0.1, random_state=random_state
            )
            knn.train(feature_set=x_train.to_numpy(), labels=y_train.to_numpy())
            valid_pred = [knn.predict(feature=x_valid.iloc[i].to_numpy(), k=k, dist_func=dist)
                          for i in range(x_valid.shape[0])]
            accuracy = accuracy_score(y_valid, valid_pred)
            return accuracy, k, dist_func


from sklearn.model_selection import train_test_split
from preprocess import preprocess_wine_quality
import time



def main():
    df = preprocess_wine_quality(pd.read_csv("../../dataset/Wine_Quality_Data.csv"))

    train, test = train_test_split(df, test_size=0.5, random_state=42)
    tunner = Tune_Knn(train=train, test=test, label_col="quality")
    start = time.time()
    result = tunner.tune(10, ["cos_sim"], 42, 5)
    end = time.time()
    print(result)
    print(end - start)



# Must include for multiprocess to work
if __name__ == "__main__":
    main()
