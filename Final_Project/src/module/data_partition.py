from sklearn.model_selection import train_test_split
import pandas as pd


def partition_data(df: pd.DataFrame, label_col: str, test_set_prop: float = 0.2, random_state=42):
    """
    Partition the given data into training set and test set
    @param df: data to be partitioned
    @param label_col: the y-label of class
    @param test_set_prop: the proportion of the test set in relation to the not partitioned data
    @param random_state: the seed of randomness
    """
    x_vals = df.drop(label_col, axis=1)
    y_vals = df[label_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x_vals, y_vals, test_size=test_set_prop, random_state=random_state)
    return x_train, x_test, y_train, y_test
