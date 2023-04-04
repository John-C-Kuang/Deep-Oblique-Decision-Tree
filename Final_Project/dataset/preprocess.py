import pandas as pd
import numpy as np

DATASET_PATH = "Wine_Quality_Data.csv"
CLASS_LABEL = "quality"
DISCRETE_LABEL = "color"


def __handle_discrete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all discrete columns using one hot encoding.
    """
    # The only discrete column is "color." Hard coding one hot for now.
    df[DISCRETE_LABEL] = np.where(df[DISCRETE_LABEL] == "red", 1, 10)
    return df


def __handle_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score normalization to all continuous data
    """
    for column in df.columns:
        if column != CLASS_LABEL and column != DISCRETE_LABEL:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


def preprocess_wine_quality() -> pd.DataFrame:
    """
    Preprocess Wine_Quality_data.csv. Encode all categorical/discrete data using one hot encoding.
    Normalize all continuous data.
    @return: Data ready to train the ML models
    """
    df = pd.read_csv(DATASET_PATH)
    df = __handle_discrete(df)
    df = __handle_continuous(df)
    # Make the label column the last column in df
    label_col = df.pop(CLASS_LABEL)
    df[CLASS_LABEL] = label_col
    return df
