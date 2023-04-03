import pandas as pd

from nn import FeedForward
from typing import Union, Any


class _DODTree:

    def __init__(self, label_col: str, left: Union['_DODTree', None],
                 right: Union['_DODTree', None], cls: Any = None,
                 perceptron: FeedForward = None):
        """
        @param label_col: name of the column containing the labels.
        @param left: decision tree connected to the left branch
        @param right: decision tree connected to the right branch
        @param cls: the class of the leaf. Set to None if current _DODTree is a node.
        """
        self.label_col = label_col
        self.right = right
        self.left = left
        self.cls = cls
        self.perceptron = perceptron

    def predict(self, feature: Union[pd.Series, dict]):
        pass

    @classmethod
    def build_tree(cls,
                   train: pd.DataFrame,
                   label_col: str,
                   class_order: list[str],
                   ff_dim: int,
                   num_epochs: int,
                   learning_rate: float,
                   weight_scale: float = 1e-3,
                   reg: float = 0.0
                   ) -> '_DODTree':

        # only one class left, stop splitting.
        if len(class_order) == 1:
            return _DODTree(left=None, right=None, cls=class_order[0], label_col=label_col)

        perceptron = FeedForward(input_dim=len(train.columns) - 1, ff_dim=ff_dim, weight_scale=weight_scale, reg=reg)
        # perceptron.train() -- expecting updates in Perceptron





class DODTree:

    def __init__(self):
        """
        Metaclass for _DODTree
        """
        self.train_result = {}
        self.root = None

    @staticmethod
    def __determine_class_order(train: pd.DataFrame, label_col: str) -> list[str]:
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
              train: pd.DataFrame,
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
        self.root = _DODTree.build_tree(
            train, label_col, class_order, ff_dim,
            num_epochs, learning_rate, weight_scale, reg
        )

    def predict(self, feature: Union[pd.Series, dict]) -> Any:
        if self.root is None:
            raise "Decision Tree instance has not been trained"

        if isinstance(feature, pd.Series):
            return self.root.predict(feature.to_dict())
        return self.root.predict(feature)
