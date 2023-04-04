import numpy as np
import pandas as pd

from nn import FeedForward
from typing import Union, Any


class _DODTree:

    def __init__(self, next: Union['_DODTree', None],
                 perceptron: FeedForward = None):
        """
        Deep Oblique Decision Tree. A variant of Oblique Decision Tree that utilizes
        concepts of deep learning.
        @param next: the next layer of _DODTree
        """
        self.next = next
        self.perceptron = perceptron

    def predict(self, features: np.ndarray) -> Any:
        """
        Classify the given features
        @param features: the features to be classified
        @return: the predicted class of the features given
        """
        prediction, new_features_or_result_class = self.perceptron.forward(features)
        if prediction:
            return new_features_or_result_class
        return self.next.predict(new_features_or_result_class)

    @classmethod
    def build_tree(cls,
                   train: Union[np.ndarray, None],
                   class_order: list[int],
                   ff_dim: int,
                   num_epochs: int,
                   learning_rate: float,
                   weight_scale: float = 1e-3,
                   reg: float = 0.0
                   ) -> '_DODTree':
        """
        Build an instance of the _DODTree.
        @param train: training data
        @param ff_dim: the dimension of the FeedForward output
        @param num_epochs: number since epochs for the perceptron in FeedForward
        @param learning_rate: learning rate of the perceptron
        @param weight_scale: scale of the normal distribution for random initialization in FeedForward
        @param reg: strength of the L2-Regularization.
        @return: an instance of the _DODTree.
        """

        # only one class left, stop splitting.
        if len(class_order) == 1:
            return _DODTree(next=None)

        class_to_determine = class_order.pop(0)
        rest_cls = class_order[0] if len(class_order) == 1 else None

        perceptron = FeedForward(input_dim=train.size - 1, ff_dim=ff_dim,
                                 weight_scale=weight_scale, reg=reg,
                                 target_cls=class_to_determine, rest_cls=rest_cls)
        new_train = perceptron.train(data=train, num_epochs=num_epochs, learning_rate=learning_rate)

        # Left is classified
        return _DODTree(
            next=cls.build_tree(
                train=new_train,
                class_order=class_order,
                ff_dim=ff_dim,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_scale=weight_scale,
                reg=reg
            )
        )


class DODTree:

    def __init__(self):
        """
        Metaclass for _DODTree. Designed for hyperparameter tuning.
        """
        self.train_result = {}
        self.root = None

    @staticmethod
    def __determine_class_order(train: pd.DataFrame, label_col: str) -> list[int]:
        """
        Determine the order of classes for the nodes using the frequency of that class in the training data
        @param train: training dataset to be split on.
        @param label_col: name of the column containing the labels.
        @return: a list of class ordered in decreasing frequency
        """
        cls_freq_df = train.groupby(label_col).size()
        cls_freq_vals = cls_freq_df.values.tolist()
        cls_freq_indices = cls_freq_df.index.tolist()
        cls_freq_joint_list = [(cls_freq_indices[i], cls_freq_vals[i]) for i in range(len(cls_freq_indices))]
        cls_freq_joint_list = sorted(cls_freq_joint_list, key=lambda pair: pair[1], reverse=True)
        return [pair[0] for pair in cls_freq_joint_list]

    def train(self,
              train: Union[pd.DataFrame, np.ndarray],
              label_col: str,
              ff_dim: int,
              num_epochs: int = 1000,
              learning_rate: float = 0.001,
              weight_scale: float = 1e-3,
              reg: float = 0.0):
        """
        Builds a complete DODTree with given hyperparameters.

        @param train: training dataset to be split on.
        @param label_col: name of the column containing the labels.
        @param ff_dim: integer dimension of the hidden layer.
        @param num_epochs: number of iteration to run until epoch for perceptron
        @param learning_rate: learning rate for perceptron
        @param weight_scale: scale of the normal distribution for perceptron random initialization.
        @param reg: strength of the L2-Regularization in perceptron.
        @return: None
        """
        class_order = DODTree.__determine_class_order(train, label_col)

        if isinstance(train, pd.DataFrame):
            train = train.to_numpy()

        self.root = _DODTree.build_tree(
            train, class_order, ff_dim,
            num_epochs, learning_rate, weight_scale, reg
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
