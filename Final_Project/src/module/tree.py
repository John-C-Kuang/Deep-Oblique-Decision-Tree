import numpy as np
import pandas as pd

import src.utils.ml_utils.metric
from nn import FeedForward
from typing import Union, Any, Callable
from collections import Counter


class _DODTree:

    def __init__(self, left: Union['_DODTree', None], right: Union['_DODTree', None],
                 feedforward: FeedForward = None, cls: Any = None):
        """
        Deep Oblique Decision Tree. A variant of Oblique Decision Tree that utilizes
        concepts of deep learning.
        @param left: the left branch of _DODTree
        @param right: the right branch of _DODTree
        @param feedforward: internal feed forward network
        @param cls: the class of the leaf
        @return an instance of _DODTree
        """
        self.left = left
        self.right = right
        self.feedforward = feedforward
        self.cls = cls

    def predict(self, features: np.ndarray) -> Any:
        """
        Classify the given features
        @param features: the features to be classified
        @return: the predicted class of the features given
        """
        feature_vector = self.feedforward.forward(xs=features)
        sigmoid = feature_vector[-1]
        feature_vector = feature_vector[:-1]
        if sigmoid >= 0.5:
            return self.right.predict(feature_vector)
        return self.left.predict(feature_vector)

    @classmethod
    def build_tree(cls,
                   train: Union[np.ndarray, None],
                   ff_dim: int,
                   num_epochs: int,
                   learning_rate: float,
                   momentum: float = 0.9,
                   reg: float = 0.0,
                   impurity_func: Callable = src.utils.ml_utils.metric.Entropy,
                   target_impurity: float = 0.0,
                   ) -> '_DODTree':
        """
        Build an instance of the _DODTree.
        @param target_impurity: impurity metric to stop recursion
        @param train: training data
        @param ff_dim: the dimension of the FeedForward output
        @param num_epochs: number since epochs for the perceptron in FeedForward
        @param learning_rate: learning rate of the perceptron
        @param momentum: scalar giving momentum strength for gradient descent
        @param reg: strength of the L2-Regularization.
        @param impurity_func: inpurity measure functions
        @return: an instance of the _DODTree.
        """
        class_order = cls.__get_class_order(train, -1)
        if impurity_func(train) <= target_impurity:
            return _DODTree(left=None, right=None, feedforward=None, cls=class_order[0])

        input_dim = train.shape[0] - 1

        feedforward = FeedForward(input_dim=input_dim, ff_dim=ff_dim, target_cls=class_order[0], reg=reg)
        scores = feedforward.train(data=train, num_epochs=num_epochs, learning_rate=learning_rate, momentum=momentum)
        sigmoid = scores[:, -1]
        data = scores[:, :-1]
        mask = sigmoid >= 0.5

        left_branch = cls.build_tree(
            train=data[~mask],
            ff_dim=ff_dim,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            reg=reg,
            impurity_func=impurity_func,
            target_impurity=target_impurity
        )

        right_branch = cls.build_tree(
            train=data[mask],
            ff_dim=ff_dim,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            reg=reg,
            impurity_func=impurity_func,
            target_impurity=target_impurity
        )

        return _DODTree(
            left=left_branch,
            right=right_branch,
            feedforward=feedforward,
            cls=None
        )

    @classmethod
    def __get_class_order(cls, data: np.ndarray, label_index: int) -> tuple:
        return Counter(data[:, label_index]).most_common(1)[0]


class DODTree:

    def __init__(self):
        """
        Metaclass for _DODTree. Designed for hyperparameter tuning.
        """
        self.train_result = {}
        self.root = None

    def train(self,
              train: Union[pd.DataFrame, np.ndarray],
              ff_dim: int,
              momentum: float,
              num_epochs: int = 1000,
              learning_rate: float = 0.001,
              reg: float = 0.0):
        """
        Builds a complete DODTree with given hyperparameters.

        @param momentum: scalar giving momentum strength for gradient descent
        @param train: training dataset to be split on.
        @param ff_dim: integer dimension of the hidden layer.
        @param num_epochs: number of iteration to run until epoch for perceptron
        @param learning_rate: learning rate for perceptron
        @param reg: strength of the L2-Regularization in perceptron.
        @return: None
        """
        if isinstance(train, pd.DataFrame):
            train = train.to_numpy()

        self.root = _DODTree.build_tree(
            train=train, ff_dim=ff_dim, num_epochs=num_epochs, learning_rate=learning_rate,
            momentum=momentum, reg=reg, impurity_func=src.utils.ml_utils.Entropy,
            target_impurity=0.0
        )

    def predict(self, features: np.ndarray) -> Any:
        """
        Classify the given features
        @param features: features to be classified
        @return: the predicted class of the given features
        """
        if self.root is None:
            raise "Decision Tree instance has not been trained"

        return self.root.predict(features)
