# global
import numpy as np

# local
from typing import Union


def array_mesh(array_0: Union[np.ndarray, list], array_1: Union[np.ndarray, list]) -> np.ndarray:
    """
    Creates a 2-D array of all coordinate pairs for two 1-D arrays.

    @param array_0: 1-D array or list.
    @param array_1: 1-D array or list.
    @return: 2-D array of exhaustive coordinate pairs.
    """
    return np.array(np.meshgrid(array_0, array_1)).T
