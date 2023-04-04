import pandas as pd

from src.utils import ml_utils

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
                max_depth: int = None,
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