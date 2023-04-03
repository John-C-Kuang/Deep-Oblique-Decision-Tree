import pandas as pd

from typing import Union, Any


class _DODTree:

    def __init__(self, label_col: str, left: Union['_DODTree', None],
                 right: Union['_DODTree', None], cls: Any = None):
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

    def predict(self):
        pass

    @classmethod
    def build_tree(cls,
                   train: pd.DataFrame,
                   label_col: str,
                   class_order: list[str],
                   num_epochs: int,
                   learning_rate: float
                   ) -> '_DODTree':

        # only one class left, stop splitting.
        if len(class_order) == 1:
            return _DODTree(left=None, right=None, cls=class_order[0], label_col=label_col)






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
              num_epochs: int = 1000,
              learning_rate: float = 0.001):
        """
        Builds a complete DODTree with given hyperparameters.

        @param train: training dataset to be split on.
        @param label_col: name of the column containing the labels.
        @param num_epochs: number of iteration to run until epoch for perceptron
        @param learning_rate: learning rate for perceptron
        @return: None
        """
        class_order = DODTree.__determine_class_order(train, label_col)
        self.root = _DODTree.build_tree(train, label_col, class_order, num_epochs, learning_rate)

    def predict(self):
        pass
