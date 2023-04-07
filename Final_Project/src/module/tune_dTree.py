import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.module.preprocess import preprocess_wine_quality
from src.utils import ml_utils
from tqdm.contrib.itertools import product


# Given the prediction and actual label, calculate the accuracy of our modal
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


# function given by lab, used to perform 10 fold cross validation
def n_folds(folds, train):
    for f in range(folds):
        train_fold = train[train.index % folds != f]
        valid_fold = train[train.index % folds == f]
    return train_fold, valid_fold


def predict_data(data, tree):
    # given a subset dataframe as test data, predict its output alongside with its original answer
    y_actual = data['quality'].to_numpy()
    # apply decision tree prediction to each of the row
    y_pred = data.apply(lambda row: tree.predict(row), axis=1).to_numpy()

    return y_actual, y_pred


# function for apllying modals with different hyperparameters.
def apply_DTree(train: pd.DataFrame,
                validation: pd.DataFrame,
                test: pd.DataFrame,
                impurity_func: str,
                discrete_threshold: int = 10,
                max_depth: int = 15,
                min_instances: int = 2,
                target_impurity: float = 0.0,
                ):
    impurity_func = ml_utils.metric.entropy if impurity_func == 'entropy' else ml_utils.metric.gini
    tree = ml_utils.experimental.DecisionTree(discrete_threshold=discrete_threshold,
                                              max_depth=max_depth,
                                              min_instances=min_instances,
                                              target_impurity=target_impurity,
                                              impurity_func=impurity_func)
    tree.train(train, 'quality')
    validation_accuracy = 'N/A'
    test_accuracy = 'N/A'
    if validation is not None:
        validation_accuracy = calc_accuracy(*predict_data(validation, tree))
    if test is not None:
        test_accuracy = calc_accuracy(*predict_data(test, tree))

    return validation_accuracy, test_accuracy


class Tune_DTree:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, label_col: str):
        self.train = train
        self.test = test
        self.label_col = label_col
        ml_utils.numpy()

    def para_tuning(self):
        train_fold, valid_fold = n_folds(10, self.train)

        # finding the best hyperparameter combo and record them in a dataframe
        impurity_funcs = ['entorpy', 'gini']
        records = pd.DataFrame(
            columns=['impurity_func', 'max_depth', 'min_instances', 'target_impurity', 'validation_accuracy',
                     'test_accuracy'])
        for min_instances, target_impurity, impurity_func in product(range(2, 12, 2),
                                                                     np.arange(0, 0.2, 0.05),
                                                                     impurity_funcs):
            try:
                validation_accuracy, test_accuracy = apply_DTree(train=train_fold,
                                                                 validation=valid_fold,
                                                                 test=None,
                                                                 impurity_func=impurity_func,
                                                                 discrete_threshold=10,
                                                                 max_depth=5,
                                                                 min_instances=min_instances,
                                                                 target_impurity=target_impurity)
                row = {'impurity_func': impurity_func, 'max_depth': max_depth, 'min_instances': min_instances,
                       'target_impurity': target_impurity, 'validation_accuracy': validation_accuracy,
                       'test_accuracy': test_accuracy}
                records = pd.concat([records, pd.DataFrame(row, index=[len(records)])])
            except Exception as e:
                print('Failed with: impurity_func: {}, max_depth: {}, min_instances: {}, target_impurity: {}'.format(
                    impurity_func, max_depth, min_instances, target_impurity))
                print(e)
        return records


def main():
    df = pd.read_csv('./../../dataset/Wine_Quality_Data.csv')
    df = preprocess_wine_quality(df)
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    tunner = Tune_DTree(train=train, test=test, label_col="quality")
    records = tunner.para_tuning()
    best = records.loc[records['validation_accuracy'].idxmax()]
    print(best)


if __name__ == "__main__":
    main()
