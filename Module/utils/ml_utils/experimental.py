# global
import pandas as pd
import numpy as np

# local
from utils import ml_utils
from collections import Counter
from typing import Callable, Union, Any


class KNN:
    def __init__(self, label_dict: dict = None):
        """
        Constructor for a stateful KNN instance.

        @param label_dict: optional dictionary for pre-trained feature-label pairs.
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

        @param feature_set: sequence of training features.
        @param labels: sequence of training labels associate with features
        @return: None
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

        @param feature: feature vector for class prediction.
        @param k: hyperparameter k for k-nearest neighbor algorithm.
        @param dist_func: function for distance metric calculation.
        @param ascending: boolean flag for sorting distance metrics from most to least similar.
        @return: predicted int class label.
        """
        if self.labels is None:
            raise AttributeError('Current KNN instance has not been trained')

        dist_array = np.array([[dist_func(feature, _), self.labels[_]] for _ in self.labels])
        sorted_indices = np.argsort(dist_array[:, 0])
        if ascending:
            sorted_labels = np.take(dist_array, sorted_indices[:k], axis=0)[:, 1]
        else:
            sorted_labels = np.take(dist_array, sorted_indices[-1:-1 - k:-1], axis=0)[:, 1]

        label_dict = {}
        for cls in self.num_cls:
            label_dict[cls] = np.sum(sorted_labels == cls)

        return max(label_dict, key=label_dict.get)


class _DTreeNode:
    def __init__(self, header: str, discrete_flag: bool, val: Any,
                 *,
                 cls: Any = None,
                 left: Union['_DTreeNode', None], right: Union['_DTreeNode', None]):
        """
        Constructor of a binary decision tree instance.

        @param header: header of the split column.
        @param discrete_flag: flag indicates if the column is discrete or continuous.
        @param val: value for splitting the column
        @param left: decision tree connected to the left branch
        @param right: decision tree connected to the right branch
        """
        self.header = header
        self.discrete_flag = discrete_flag
        self.val = val
        self._left = left
        self._right = right
        self.cls = cls

    @classmethod
    def build_tree(cls, train: pd.DataFrame, label_col: str,
                   *,
                   discrete_threshold: int,
                   max_depth: int = None,
                   min_instances: int,
                   impurity_func: Callable = None,
                   target_impurity: float,
                   build_cls: bool = False) -> '_DTreeNode':
        """
        Builds the binary decision tree with given hyperparameters.

        @param train: target dataset to be split on.
        @param label_col: column containing the class labels.
        @param discrete_threshold: number of unique values to determine if the column values are discrete.
        @param max_depth: maximum depth of the decision tree.
        @param min_instances: minimum number of instances within the dataset to terminate splitting.
        @param impurity_func: function of the impurity measure.
        @param target_impurity: target impurity to terminate splitting.
        @param build_cls: flag indicates if the tree has reach terminate condition.
        @return: built binary decision tree.
        """
        if len(train) <= min_instances or (max_depth is not None and max_depth <= 0) \
                or train[label_col].nunique() == 1 or build_cls:
            target_cls = Counter(train[label_col]).most_common(1)[0][0]
            return _DTreeNode(label_col, True, None, cls=target_cls, left=None, right=None)

        best_col, discrete, best_val, best_imp = cls._best_split(train, label_col, discrete_threshold, impurity_func)
        if discrete:
            split0 = train[train[best_col] == best_val]
            split1 = train[train[best_col] != best_val]
        else:
            split0 = train[train[best_col] <= best_val]
            split1 = train[train[best_col] > best_val]

        if len(split0) == 0 or len(split1) == 0:
            raise RuntimeError('Found duplicate feature vectors within the dataset but with different labels')

        if best_imp <= target_impurity:
            return _DTreeNode(best_col, discrete, best_val,
                              left=cls.build_tree(split0, label_col,
                                                  build_cls=True,
                                                  discrete_threshold=discrete_threshold,
                                                  min_instances=min_instances,
                                                  target_impurity=target_impurity),
                              right=cls.build_tree(split1, label_col,
                                                   build_cls=True,
                                                   discrete_threshold=discrete_threshold,
                                                   min_instances=min_instances,
                                                   target_impurity=target_impurity)
                              )

        max_depth = max_depth - 1 if max_depth is not None else max_depth

        return _DTreeNode(best_col, discrete, best_val,
                          left=cls.build_tree(split0, label_col,
                                              discrete_threshold=discrete_threshold,
                                              max_depth=max_depth,
                                              min_instances=min_instances,
                                              impurity_func=impurity_func,
                                              target_impurity=target_impurity),
                          right=cls.build_tree(split1, label_col,
                                               discrete_threshold=discrete_threshold,
                                               max_depth=max_depth,
                                               min_instances=min_instances,
                                               impurity_func=impurity_func,
                                               target_impurity=target_impurity)
                          )

    def predict(self, feature: dict) -> Any:
        if self._left is None and self._right is None:
            return self.cls

        if self.discrete_flag:
            if feature[self.header] == self.val:
                return self._left.predict(feature)
            else:
                return self._right.predict(feature)
        else:
            if feature[self.header] <= self.val:
                return self._left.predict(feature)
            else:
                return self._right.predict(feature)

    @classmethod
    def _best_split(cls, train: pd.DataFrame, label_col: str, discrete_threshold: int, impurity_func: Callable) \
            -> (str, bool, Any, float):
        """
        Finds the best split within the current dataset.

        @param train: target dataset to be split on.
        @param label_col: column containing the class labels.
        @param discrete_threshold: number of unique values to determine if the column values are discrete.
        @param impurity_func: function of the impurity measure.
        @return: column name of the best split, flag indicates if the column is discrete and the value to be split on.
        """
        best_col = None
        best_val = None
        best_imp = float('inf')
        discrete = None
        for split_col in train.columns:
            if split_col != label_col:
                flag, v, imp = cls._best_split_for_col(train, label_col, discrete_threshold, split_col, impurity_func)
                if imp < best_imp:
                    best_col = split_col
                    best_val = v
                    best_imp = imp
                    discrete = flag

        return best_col, discrete, best_val, best_imp

    @classmethod
    def _best_split_for_col(cls, train: pd.DataFrame, label_col: str, discrete_threshold: int, split_col: str,
                            impurity_func: Callable) -> (bool, Any, float):
        """
        Finds the best split within the given column.

        @param train: target dataset to be split on.
        @param label_col: column containing the class labels.
        @param discrete_threshold: number of unique values to determine if the column values are discrete.
        @param split_col: header of the column to be split on.
        @param impurity_func: function of the impurity measure.
        @return: flag indicates if the column is discrete and the value to be split on.
        """
        unique_val = set(train[split_col])
        discrete = train[split_col].dtype == 'object' or len(unique_val) < discrete_threshold
        best_val = None
        best_imp = float('inf')

        for v in unique_val:
            imp = cls._evaluate_split(train, label_col, split_col, discrete, v, impurity_func)
            if imp < best_imp:
                best_val = v
                best_imp = imp

        return discrete, best_val, best_imp

    @classmethod
    def _evaluate_split(cls, train: pd.DataFrame, label_col: str, split_col: str, discrete_flag: bool,
                        split_val: Any, impurity_func: Callable) -> float:
        """
        Evaluates the impurity measure of the current split.

        @param train: target dataset to be split on.
        @param label_col: column containing the class labels.
        @param split_col: header of the column to be split on.
        @param discrete_flag: flag indicates if the column is discrete.
        @param split_val: value to be split on.
        @param impurity_func: function of the impurity measure.
        @return: calculated impurity measure of the split.
        """
        if discrete_flag:
            split0 = train[train[split_col] == split_val]
            split1 = train[train[split_col] != split_val]
        else:
            split0 = train[train[split_col] <= split_val]
            split1 = train[train[split_col] > split_val]

        cnt0 = Counter(split0[label_col])
        cnt1 = Counter(split1[label_col])
        return cls._wavg(cnt0, cnt1, impurity_func)

    @classmethod
    def _wavg(cls, cnt0: Counter, cnt1: Counter, impurity_func: Callable) -> float:
        """
        Calculates the weighted average of the given split.

        @param cnt0: first Counter object of the split.
        @param cnt1: second Counter object of the split.
        @param impurity_func: function of the impurity measure.
        @return: calculated weighted average measure of the split.
        """
        tot0 = sum(cnt0.values())
        tot1 = sum(cnt1.values())
        tot = tot0 + tot1
        return (impurity_func(cnt0) * tot0 + impurity_func(cnt1) * tot1) / tot


class DecisionTree:
    def __init__(self,
                 *,
                 discrete_threshold: int = 10,
                 max_depth: int = None,
                 min_instances: int = 2,
                 impurity_func: Callable,
                 target_impurity: float = 0.0):
        """
        Constructor for a stateful decision tree builder instance.

        @param discrete_threshold: number of unique values to determine if the column values are discrete.
        @param max_depth: maximum depth of the decision tree.
        @param min_instances: minimum number of instances within the dataset to terminate splitting.
        @param impurity_func: function for measuring the dataset impurity.
        @param target_impurity: target impurity to terminate splitting.
        """
        self.discrete_threshold = discrete_threshold
        if max_depth is not None and max_depth < 0:
            raise ValueError('Maximum depth must not be negative')
        self.max_depth = max_depth
        if min_instances <= 0:
            raise ValueError('Number of minimum instances must be positive')
        self.min_instances = min_instances
        self.impurity_func = impurity_func
        self.target_impurity = target_impurity

        self.root = None

    def train(self, train: pd.DataFrame, label_col: str):
        """
        Builds a complete decision tree with given hyperparameters.

        @param train: training dataset to be split on.
        @param label_col: name of the column containing the labels.
        @return: None
        """
        if label_col not in train:
            raise ValueError('Header of label column not presented in dataset')

        self.root = _DTreeNode.build_tree(train, label_col,
                                          discrete_threshold=self.discrete_threshold,
                                          max_depth=self.max_depth,
                                          min_instances=self.min_instances,
                                          target_impurity=self.target_impurity,
                                          impurity_func=self.impurity_func)

    def predict(self, feature: Union[pd.Series, dict]) -> Any:
        """
        Predicts the class label of the given feature vector.

        @param feature: feature vector with its keys and values.
        @return: predicted class label of the feature vector.
        """
        if self.root is None:
            raise RuntimeError("Decision Tree instance has not been trained")

        return self.root.predict(feature)
