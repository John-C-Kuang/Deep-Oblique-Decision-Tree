import time

from sklearn.metrics import accuracy_score

import data_partition as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.module.preprocess import preprocess_wine_quality
from src.utils import ml_utils
from itertools import product
from handle_process import multi_process_tuning as tn


def calc_accuracy(y_pred, y_actual):
    correct_cnt = 0
    total = len(y_actual)
    # assume that y_pred and y_actual have same length

    for i in range(len(y_actual)):
        pre = y_pred[i]
        act = y_actual[i]
        if pre == act:
            correct_cnt += 1

    return correct_cnt / total


def n_folds(folds, train):
    for f in range(folds):
        train_fold = train[train.index % folds != f]
        valid_fold = train[train.index % folds == f]
    return train_fold, valid_fold


class Tune_DTree:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, label_col: str):
        self.train = train
        self.test = test
        self.label_col = label_col
        ml_utils.numpy()

    # Given the prediction and actual label, calculate the accuracy of our modal

    # function given by lab, used to perform 10-fold cross validation

    def predict_data(self, data, tree):
        # given a subset dataframe as test data, predict its output alongside with its original answer
        y_actual = data[self.label_col].to_numpy()
        # apply decision tree prediction to each of the row
        y_pred = data.apply(lambda row: tree.predict(row), axis=1).to_numpy()

        return y_actual, y_pred

    def tune(self, max_depth, processes: int = 4) -> dict:
        impurity_funcs = ['entropy', 'gini']
        function_args = []
        for min_instances, target_impurity, impurity_func in product(range(2, 12, 2),
                                                                     np.arange(0, 0.2, 0.05),
                                                                     impurity_funcs):
            function_args.append([impurity_func, max_depth, min_instances, target_impurity, self.test])

        print("There are ", len(function_args), " instances.")

        records = tn.tune(task_function=self._batch_train_n_predict,
                          tasks_param=function_args, max_active_processes=processes)
        hyper_params = max(records, key=lambda data: data["validation_accuracy"])

        return hyper_params

    def _batch_train_n_predict(self, impurity_func, max_depth, min_instances,
                               target_impurity, test, random_state: int = 42) -> dict:
        x_train, x_valid, y_train, y_valid = dp.partition_data(
            df=self.train, label_col=self.label_col, test_set_prop=0.2, random_state=random_state
        )

        train_fold = pd.concat([x_train, y_train.T], axis=1)

        impurity_func = ml_utils.metric.entropy if impurity_func == 'entropy' else ml_utils.metric.gini
        tree = ml_utils.experimental.DecisionTree(discrete_threshold=10,
                                                  max_depth=max_depth,
                                                  min_instances=min_instances,
                                                  target_impurity=target_impurity,
                                                  impurity_func=impurity_func)
        tree.train(train_fold, self.label_col)
        valid_fold = pd.concat([x_valid, y_valid.T], axis=1)
        validation_accuracy = calc_accuracy(*self.predict_data(valid_fold, tree))
        test_accuracy = calc_accuracy(*self.predict_data(test, tree))

        print("Process Finished.")
        row = {'impurity_func': impurity_func, 'max_depth': max_depth, 'min_instances': min_instances,
               'target_impurity': target_impurity, 'validation_accuracy': validation_accuracy,
               'test accuracy': test_accuracy}
        return row


def main():
    df = pd.read_csv('./../../dataset/Wine_Quality_Data.csv')
    df = preprocess_wine_quality(df)
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    tunner = Tune_DTree(train=train, test=test, label_col="quality")
    start = time.time()
    result = tunner.tune(5)
    end = time.time()
    print(result)
    print("Time(sec): ")
    print(end - start)


if __name__ == "__main__":
    main()
