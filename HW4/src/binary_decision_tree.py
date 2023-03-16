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

    def build_tree(self, df: pd.DataFrame, class_col: int, measure):
        """
        Build a decision tree
        :param df: the data to partition
        :param class_col: the column containing the classes in df
        :param measure: the measure function of impurity
        :return: a decision tree
        """
        best_col, best_v, best_meas = self.__best_split(df, class_col, measure)
        df1, df2 = df[df[best_col] == best_v], df[df[best_col] != best_v]

        # need to verify if impurity will become 0
        # maybe also consider the depth of the decision tree
        if best_meas == 0:
            self.__left = df1
            self.__right = df2
        else:
            self.__left = self.build_tree(df1, class_col, measure)
            self.__right = self.build_tree(df2, class_col, measure)
        return self

    ########### Private methods ###########

    def __total(self, cnt: dict):
        return sum(cnt.values())

    def __wavg(self, cnt1: dict, cnt2: dict, measure):
        tot1 = self.__total(cnt1)
        tot2 = self.__total(cnt2)
        tot = tot1 + tot2
        return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot

    def __evaluate_split(self, df: pd.DataFrame, class_col: int, split_col: int, feature_val: any, measure):
        """ Evaluate a partition / split based on a specific column and a feature value """
        df1, df2 = df[df[split_col] == feature_val], df[df[split_col] != feature_val]
        cnt1, cnt2 = Counter(df1[class_col]), Counter(df2[class_col])
        return wavg(cnt1, cnt2, measure)

    def __best_split_for_column(self, df: pd.DataFrame, class_col: int, split_col: int, measure):
        """ Find the best split for a given column """
        best_v = ""
        best_meas = float("inf")
        for v in set(df[split_col]):
            meas = self.__evaluate_split(df, class_col, split_col, v, measure)
            if meas < best_meas:
                best_v = v
                best_meas = meas
        return best_v, best_meas

    def __best_split(self, df: pd.DataFrame, class_col: int, measure):
        """ Find the best split in a df """
        best_col = 0
        best_v = ""
        best_meas = float("inf")
        for split_col in df.columns:
            if split_col != class_col:
                v, meas = self.__best_split_for_column(df, class_col, split_col, measure)
                if meas < best_meas:
                    best_v = v
                    best_meas = meas
                    best_col = split_col
        return best_col, best_v, best_meas
