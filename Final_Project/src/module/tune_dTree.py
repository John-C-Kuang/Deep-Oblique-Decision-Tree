import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.module.preprocess import preprocess_wine_quality
from src.utils import ml_utils
import dTreeHelpers
from tqdm.contrib.itertools import product

class Tune_DTree:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, label_col: str):
        self.train = train
        self.test = test
        self.label_col = label_col
        ml_utils.numpy()

    def para_tuning(self):
        train_fold, valid_fold = dTreeHelpers.n_folds(10, self.train)

        # finding the best hyperparameter combo and record them in a dataframe
        impurity_funcs = ['entorpy', 'gini']
        records = pd.DataFrame(
            columns=['impurity_func', 'max_depth', 'min_instances', 'target_impurity', 'validation_accuracy',
                     'test_accuracy'])
        for max_depth, min_instances, target_impurity, impurity_func in product(range(5, 30, 2), range(2, 12, 2),
                                                                                np.arange(0, 0.2, 0.05),
                                                                                impurity_funcs):
            try:
                validation_accuracy, test_accuracy = dTreeHelpers.apply_DTree(train=train_fold,
                                                                              validation=valid_fold,
                                                                              test=None,
                                                                              impurity_func=impurity_func,
                                                                              discrete_threshold=10,
                                                                              max_depth=max_depth,
                                                                              min_instances=min_instances,
                                                                              target_impurity=target_impurity)
                row = {'impurity_func': impurity_func, 'max_depth': max_depth, 'min_instances': min_instances,
                       'target_impurity': target_impurity, 'validation_accuracy': validation_accuracy,
                       'test_accuracy': test_accuracy}
                records = pd.concat([records, pd.DataFrame(row, index=[len(records)])])
            except:
                print('Failed with: impurity_func: {}, max_depth: {}, min_instances: {}, target_impurity: {}'.format(
                    impurity_func, max_depth, min_instances, target_impurity))
        return records
def main():
    df = preprocess_wine_quality()
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    tunner = Tune_DTree(train=train, test=test, label_col="quality")
    records = tunner.para_tuning()
    best = records.loc[records['validation_accuracy'].idxmax()]
    print(best)

if __name__ == "__main__":
    main()
