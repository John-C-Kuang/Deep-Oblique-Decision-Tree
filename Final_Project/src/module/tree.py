import pandas as pd


class _DODTree:

    def __init__(self):
        pass

    @classmethod
    def build_tree(cls, train: pd.DataFrame, label_col: str) -> '_DODTree':
        pass


class DODTree:

    def __init__(self):
        """
        Metaclass for _DODTree
        """
        # Each split class is determined by the frequency of that class in the training data
        self.train_result = {}
        self.root = None

    def __determine_class_order(self, train: pd.DataFrame) -> list[str]:
        """
        Determine the order of classes for the nodes
        """
        pass

    def train(self, train: pd.DataFrame, label_col: str):
        class_order = self.__determine_class_order(train)
        self.root = _DODTree.build_tree(train, label_col)

    def predict(self):
        pass
