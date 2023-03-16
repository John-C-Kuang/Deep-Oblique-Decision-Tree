# global
import pandas as pd
import numpy as np

# local
from utils import ml_utils
from typing import Callable


class KNN:
    def __init__(self, label_dict: dict = None):
        """
        Constructor for a stateful KNN instance.

        :param label_dict: optional dictionary for pre-trained feature-label pairs.
        """
        self.labels = None
        self.num_cls = None
        if label_dict is not None:
            self.labels = label_dict
            self.num_cls = set(label_dict)

    def reset_labels(self):
        """
        Resets the trained features and class labels to be ``None``.
        """
        self.labels = None
        self.num_cls = None

    def train(self, feature_set, labels):
        """
        Trains the stateful KNN instance with given feature-label pair.

        :param feature_set: sequence of training features.
        :param labels: sequence of training labels associate with features
        :return: None
        """
        self.num_cls = tuple(set(labels))
        if ml_utils.framework == 'pandas':
            feature_set: pd.Series
            labels: pd.Series
            if len(feature_set) != len(labels):
                raise ValueError('Number of training features must be the same as number labels')

            self.labels = dict(zip(tuple(feature_set), labels.apply(lambda l: self.num_cls.index(l))))
            return
        elif ml_utils.framework == 'numpy':
            feature_set: np.ndarray
            labels: np.ndarray
            if np.shape(feature_set)[0] != len(labels):
                raise ValueError('Number of training features must be the same as number labels')

            features = [tuple(_) for _ in feature_set]
            self.labels = dict(zip(features, labels))
            return

        raise RuntimeError('Framework unspecified')

    def predict(self, feature, k: int, dist_func: Callable,
                *,
                ascending: bool = True) -> int:
        """
        Predicts the class of the given feature using k-nearest neighbor and given distance metric.

        :param feature: feature vector for class prediction.
        :param k: hyperparameter k for k-nearest neighbor algorithm.
        :param dist_func: function for distance metric calculation.
        :param ascending: boolean flag for sorting distance metrics from most to least similar.
        :return: predicted int class label.
        """
        if self.labels is None:
            raise AttributeError('Current KNN instance has not been trained')

        dist_array = np.array([[dist_func(feature, _), self.labels[_]] for _ in self.labels])
        sorted_indices = np.argsort(dist_array[:, 0])
        if ascending:
            sorted_labels = np.take(dist_array, sorted_indices[:k], axis=0)[:, 1]
        else:
            sorted_labels = np.take(dist_array, sorted_indices[-1:-1-k:-1], axis=0)[:, 1]

        label_dict = {}
        for cls in self.num_cls:
            label_dict[cls] = np.sum(sorted_labels == cls)

        return max(label_dict, key=label_dict.get)
