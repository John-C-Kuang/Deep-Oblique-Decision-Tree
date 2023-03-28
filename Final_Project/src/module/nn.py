# global
import numpy as np

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

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the feed forward layer.

        @param xs: input feature vector with shape (input_dim, )
        @return: processed feature vector with shape (ff_dim, )
        """
        return xs @ self.w + self.b

    def loss(self):
        return

    def auto_grad(self):
        return

    def train(self):
        return
