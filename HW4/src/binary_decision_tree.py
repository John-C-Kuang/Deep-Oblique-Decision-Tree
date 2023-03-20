from typing import Union, Iterable
from collections import Counter

import pandas as pd


class DTree:

    # Do not use the constructor outside the scope of DTree. Use build_tree instead.

    def __init__(self, val: str, left: Union[Iterable["DTree"], pd.DataFrame],
                 right: Union[Iterable["DTree"], pd.DataFrame]):
        """
        A Binary Tree
        :param val: the value of the node
        :param left: the data on the left branch
        :param right: the data on the right branch
        """
        self.val = val
        self.__left = left
        self.__right = right

    @classmethod
    def build_tree(cls, train: pd.DataFrame, class_col: int, criterion,
                   max_depth=None, min_instances=2, target_impurity=0.0):
        """
        Build a decision tree
        :param target_impurity: the target impurity to stop building tree node
        :param min_instances: the minimum instances to keep spliting
        :param max_depth: the maximum depth of the tree
        :param train: the data to partition
        :param class_col: the column containing the classes in df
        :param criterion: the measure function of impurity
        :return: a decision tree
        """

        best_col, best_v, best_meas = cls.best_split(train, class_col, criterion)

        if len(train) <= min_instances or max_depth == 0 or best_meas <= target_impurity:
            return train

        df1, df2 = train[train[best_col] == best_v], train[train[best_col] != best_v]

        max_depth = max_depth - 1 if max_depth is not None else max_depth

        root = DTree(
            val=best_v,
            left=cls.build_tree(
                df1, class_col, criterion, max_depth - 1,
                min_instances, target_impurity
            ),
            right=cls.build_tree(
                df2, class_col, criterion, max_depth - 1,
                min_instances, target_impurity
            )
        )

        return root

    ########### Private methods ###########

    @classmethod
    def total(cls, cnt: dict):
        return sum(cnt.values())

    @classmethod
    def wavg(cls, cnt1: dict, cnt2: dict, measure):
        tot1 = cls.total(cnt1)
        tot2 = cls.total(cnt2)
        tot = tot1 + tot2
        return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot

    @classmethod
    def evaluate_split(cls, df: pd.DataFrame, class_col: int, split_col: int, feature_val: any, measure):
        """ Evaluate a partition / split based on a specific column and a feature value """
        df1, df2 = df[df[split_col] == feature_val], df[df[split_col] != feature_val]
        cnt1, cnt2 = Counter(df1[class_col]), Counter(df2[class_col])
        return cls.wavg(cnt1, cnt2, measure)

    @classmethod
    def best_split_for_column(cls, df: pd.DataFrame, class_col: int, split_col: int, measure):
        """ Find the best split for a given column """
        best_v = ""
        best_meas = float("inf")
        for v in set(df[split_col]):
            meas = cls.evaluate_split(df, class_col, split_col, v, measure)
            if meas < best_meas:
                best_v = v
                best_meas = meas
        return best_v, best_meas

    @classmethod
    def best_split(cls, df: pd.DataFrame, class_col: int, measure):
        """ Find the best split in a df """
        best_col = 0
        best_v = ""
        best_meas = float("inf")
        for split_col in df.columns:
            if split_col != class_col:
                v, meas = cls.best_split_for_column(df, class_col, split_col, measure)
                if meas < best_meas:
                    best_v = v
                    best_meas = meas
                    best_col = split_col
        return best_col, best_v, best_meas
