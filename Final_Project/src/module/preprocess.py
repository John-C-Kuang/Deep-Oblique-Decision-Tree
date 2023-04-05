import pandas as pd
import numpy as np

DATASET_PATH = "../../dataset/Wine_Quality_Data.csv"
CLASS_LABEL = "quality"
DISCRETE_LABEL = "color"


def __handle_discrete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all discrete columns using one hot encoding.
    """
    # The only discrete column is "color." Hard coding one hot for now.
    df[DISCRETE_LABEL] = np.where(df[DISCRETE_LABEL] == "red", 1, 0)
    return df


def preprocess_wine_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Wine_Quality_data.csv. Encode all categorical/discrete data using one hot encoding.
    Normalize all continuous data.
    @return: Data ready to train the ML models
    """
    df = __handle_discrete(df)
    # Make the label column the last column in df
    label_col = df.pop(CLASS_LABEL)
    df[CLASS_LABEL] = label_col
    return df
