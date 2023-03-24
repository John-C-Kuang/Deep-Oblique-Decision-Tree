# global
import numpy as np
import pandas as pd

# local
from typing import Union, Sequence
from collections import Counter
from utils import ml_utils


def mse(actual: Union[pd.Series, np.ndarray], pred: Union[pd.Series, np.ndarray]):
    """
    Calculates the Mean Squared Error for actual and prediction values.

    @param actual: sequence of actual values of the dataset.
    @param pred: sequence of predicted values of the dataset.
    @return: calculated Mean Squared Error of the predictions.
    """
    if ml_utils.framework == 'numpy':
        actual: np.ndarray
        pred: np.ndarray
        actual = np.squeeze(actual)
        pred = np.squeeze(pred)

        if np.shape(actual) != np.shape(pred):
            raise ValueError('Arrays of actual and predicted values must have same shape')

        return np.mean(np.square(actual - pred), axis=-1)

    if ml_utils.framework == 'pandas':
        actual: pd.Series
        pred: pd.Series
        return ((actual - pred) ** 2).mean()

    raise RuntimeError('Framework unspecified')


def mae(actual: Union[pd.Series, np.ndarray], pred: Union[pd.Series, np.ndarray]):
    """
    Calculates the Mean Absolute Error for actual and prediction values.

    @param actual: sequence of actual values of the dataset.
    @param pred: sequence of predicted values of the dataset.
    @return: calculated Mean Absolute Error of the predictions.
    """
    if len(actual) != len(pred):
        raise ValueError('Sequences of actual and predicted values must have same length')

    if ml_utils.framework == 'numpy':
        actual: np.ndarray
        pred: np.ndarray
        actual = np.squeeze(actual)
        pred = np.squeeze(pred)

        if np.shape(actual) != np.shape(pred):
            raise ValueError('Arrays of actual and predicted values must have same shape')

        return np.mean(np.abs(actual - pred), axis=-1)

    if ml_utils.framework == 'pandas':
        actual: pd.Series
        pred: pd.Series
        return abs(actual - pred).mean()

    raise RuntimeError('Framework unspecified')


def confusion_matrix(y, y_pred) -> np.ndarray:
    """
    Generates the Confusion Matrix for actual and prediction labels.

    @param y: sequence of actual class labels.
    @param y_pred: sequence of predicted class labels.
    @return: generated Confusion Matrix of the predictions.
    """
    unique_classes = set(y) | set(y_pred)
    n_classes = len(unique_classes)

    ret = np.zeros(shape=(n_classes, n_classes), dtype=int)

    actual_prediction = list(zip(y, y_pred))
    for i, j in actual_prediction:
        ret[i, j] += 1

    return ret


def metrics(y: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray],
            *,
            safe_factor: float = 1e-7) -> dict[str, float]:
    """
    Calculates the Metrics evaluating accuracy, sensitivity, specificity, precision and F-1 score for actual and
    prediction labels.

    @param y: sequence of actual class labels.
    @param y_pred: sequence of predicted class labels.
    @param safe_factor: minimal value to prevent divided by 0 warning.
    @return: dictionary of metric names as key with corresponding metric values.
    """
    if len(y) != len(y_pred):
        raise ValueError('Sequences of actual and predicted values must have same length')

    if ml_utils.framework == 'pandas':
        y = y.to_numpy()
        y_pred = y_pred.to_numpy()

    y: np.ndarray
    y_pred: np.ndarray

    if ml_utils.framework == 'numpy':
        y = np.squeeze(y)
        y_pred = np.squeeze(y_pred)

        if np.shape(y) != np.shape(y_pred):
            raise ValueError('Arrays of actual and predicted values must have same shape')
    elif ml_utils.framework is None:
        raise RuntimeError('Framework unspecified')

    ret = None
    # prevent divide by 0 results in NaN
    cm = confusion_matrix(y, y_pred) + safe_factor

    accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)

    if np.shape(cm) == (2, 2):
        tn, fp, fn, tp = np.ravel(cm)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

        ret = {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity,
               'precision': precision, 'f1_score': f1_score}
    else:
        pass

    return ret


def mpe(actual: Union[pd.Series, np.ndarray], score: Union[pd.Series, np.ndarray],
        pred: Union[pd.Series, np.ndarray]):
    """
    Calculates the Mean Perceptron Error for actual and prediction labels.

    @param actual: sequence of actual class labels.
    @param score: sequence of scores calculated during prediction.
    @param pred: sequence of predicted class labels.
    @return: calculated Mean Perceptron Error of the predictions.
    """
    if len(actual) != len(pred):
        raise ValueError('Sequences of actual and predicted values must have same length')

    if ml_utils.framework == 'numpy':
        actual: np.ndarray
        pred: np.ndarray
        actual = np.squeeze(actual)
        pred = np.squeeze(pred)

        if np.shape(actual) != np.shape(pred):
            raise ValueError('Arrays of actual and predicted values must have same shape')

        return np.mean(np.abs(actual - pred) * np.abs(score))

    if ml_utils.framework == 'pandas':
        actual: pd.Series
        pred: pd.Series
        return (abs(actual - pred) * abs(score)).mean()

    raise RuntimeError('Framework unspecified')


def phi(positive: Counter, summary: Counter, total_pos: int, total_sum: int, key) -> float:
    """
    Calculates the Phi Coefficient for given key correlates with the positive class.

    @param positive: frequency counter for all features associate with positive label.
    @param summary: summary frequency counter for all features.
    @param total_pos: number of occurrences of positive labels.
    @param total_sum: number of all occurrences.
    @param key: target feature for calculating correlation.
    @return: calculated Phi Coefficient of target feature associated with positive label.
    """

    pos_1 = positive[key]
    pos_0 = total_pos - pos_1
    neg_1 = summary[key] - pos_1
    total_neg = total_sum - total_pos
    neg_0 = total_neg - neg_1

    return (pos_1 * neg_0 - pos_0 * neg_1) / np.sqrt(total_pos * total_neg * summary[key] * (pos_0 + neg_0))


def cos_sim(x: Union[Sequence, np.ndarray], y: Union[Sequence, np.ndarray],
            *,
            safety_factor: float = 1e-7):
    """
    Calculates Cosine Similarities of two 1-D feature vectors with same length.

    @param x: feature vector sequence with shape (N, ).
    @param y: feature vector sequence with shape (N, ).
    @param safety_factor: minimal value to prevent divided by 0 warning.
    @return: calculated Cosine Similarity of two feature vectors.
    """
    if isinstance(x, Sequence):
        x = np.array(x)
    if isinstance(y, Sequence):
        y = np.array(y)

    if np.ndim(x) != 1 or np.ndim(y) != 1:
        raise ValueError('Vector dimensions need to be 1')
    elif len(x) != len(y):
        raise ValueError('Vector lengths need to be the same')

    abs_x = np.sqrt(x @ x)
    abs_y = np.sqrt(y @ y)
    return np.dot(x, y) / (abs_x * abs_y + safety_factor)


def gini(cls: Counter):
    """
    Calculates Gini Index of the class labels.

    @param cls: frequency counter of the class labels.
    @return: calculated Gini Index of the current split.
    """
    return 1 - sum([(_ / sum(cls.values())) ** 2 for _ in cls.values()])


def entropy(cls: Counter):
    """
    Calculates Entropy of the class labels.

    @param cls: frequency counter of the class labels
    @return: calculated Entropy of the current split.
    """
    tot = sum(cls.values())
    return -sum([(_ / tot) * np.log(_ / tot) for _ in cls.values()])


# aliases
MeanSquaredError = mse
MeanAbsoluteError = mae
MeanPerceptronError = mpe
PhiCoefficient = phi
CosineSimilarity = cos_sim
GiniIndex = gini
Entropy = entropy
