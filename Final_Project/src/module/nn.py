# global
import numpy as np
import pandas as pd


# local


class FeedForward:
    def __init__(self, input_dim: int, ff_dim: int, *, weight_scale: float = 1e-3, reg: float = 0.0):
        """
        Initialize the parameters for a feed forward layer.

        @param input_dim: integer dimension of the input data.
        @param ff_dim: integer dimension of the hidden layer.
        @param weight_scale: scale of the normal distribution for random initialization.
        @param reg: strength of the L2-Regularization.
        """
        self.w = np.random.normal(0., weight_scale, (input_dim, ff_dim))
        self.b = np.zeros(ff_dim)
        self.reg = reg

    def __call__(self, *args, **kwargs) -> dict[int, tuple(np.ndarray, np.ndarray)]:
        """
        Default behavior for split dataset and feature processing

        @param args: dataframe as input feature vectors
        @param kwargs: None
        @return: dictionary for features classified as 1 and processed features for 0
        """
        if len(args) != 1:
            raise TypeError('FeedForward object expects 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0:
            raise TypeError('FeedForward object expects no keywords argument, found {}'.format(len(kwargs)))
        return self.forward(*args)

    def forward(self, xs: np.ndarray) -> dict[int, tuple(np.ndarray, np.ndarray)]:
        """
        Forward pass of the feed forward layer.

        @param xs: input feature vectors with shape (batch, input_dim)
        @return: processed feature vectors with shape (batch, ff_dim)
        """
        return None

    def train(self,
              data: np.ndarray,
              *,
              target_cls: int,
              num_epochs: int,
              learning_rate: float) -> np.ndarray:
        xs = data[:, :-1]
        ys = data[:, -1]
        processed = xs @ self.w + self.b
        processed = np.concatenate((processed, np.expand_dims(ys, axis=-1)), axis=-1)

        return None
