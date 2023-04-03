# global
import numpy as np
import pandas as pd

# local
from typing import Any


class Linear:
    def __init__(self, input_dim: int, ff_dim: int, *,
                 weight_scale: float = 1e-3,
                 reg: float = 0.0):
        """
        Initialize the parameters for a linear layer.

        @param input_dim: integer dimension of the input data.
        @param ff_dim: integer dimension of the hidden layer.
        @param weight_scale: scale of the normal distribution for random initialization.
        @param reg: strength of the L2-Regularization.
        """
        self.cache_x = None
        self.w = np.random.normal(0., weight_scale, (input_dim, ff_dim))
        self.b = np.zeros(ff_dim)
        self.reg = reg

    def __call__(self, *args, **kwargs) -> np.ndarray:
        if len(args) != 1:
            raise TypeError('Linear layer forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0 and 'auto_grad' not in kwargs:
            raise TypeError("Linear layer forward pass expects only 1 keyword argument 'auto_grad'")

        return self.forward(*args)

    def forward(self, xs: np.ndarray, *, auto_grad: bool = True) -> np.ndarray:
        """
        Forward pass of the linear layer.

        @param xs: input feature vectors with shape (batch, input_dim).
        @param auto_grad: boolean flag indicates if layer parameters are trainable.
        @return: processed feature vectors with shape (batch, ff_dim).
        """
        if auto_grad:
            self.cache_x = xs
        out = xs @ self.w + self.b
        return out

    def auto_grad(self, dout: np.ndarray, config: dict[str, float]) -> dict[str, float]:
        """
        Gradient descent optimizer of the linear layer.

        @param dout: upstream derivative with shape (batch, ff_dim).
        @param config: keyword configurations for gradient descent with momentum.
        @return: configurations for the next optimizer iteration.
        """
        dw = self.cache_x.T @ dout + self.reg * self.w
        db = np.sum(dout, axis=0)
        self.b -= config['learning_rate'] * db

        v = config['momentum'] * config['velocity'] - config['learning_rate'] * dw
        config['velocity'] = v
        self.w += v

        return config


class ReLU:
    def __init__(self):
        self.cache_x = None

    def __call__(self, *args, **kwargs) -> np.ndarray:
        if len(args) != 1:
            raise TypeError('ReLU layer forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0 and 'auto_grad' not in kwargs:
            raise TypeError("ReLU layer forward pass expects only 1 keyword argument 'auto_grad'")

        return self.forward(*args)

    def forward(self, xs: np.ndarray, auto_grad: bool = True) -> np.ndarray:
        """
        Forward pass of the ReLU activation layer

        @param xs: input feature vectors with shape (batch, input_dim).
        @param auto_grad: boolean flag indicates if layer parameters are trainable.
        @return: processed feature vectors with shape (batch, ff_dim).
        """
        if auto_grad:
            self.cache_x = xs
        return np.maximum(xs, 0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass of the ReLU activation layer.

        @param dout: upstream derivative with shape (batch, ff_dim).
        @return: downstream derivative with respect to x.
        """
        return dout * (np.sum(self.cache_x, axis=-1) >= 0).astype(int)


class InverseNormalize:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> np.ndarray:
        if len(args) != 1:
            raise TypeError(
                'Normalize layer forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0:
            raise TypeError("Normalize layer forward pass expects no keyword argument, found {}".format(len(kwargs)))

        return self.forward(*args)

    @classmethod
    def forward(cls, xs: np.ndarray) -> np.ndarray:
        """
        Inverse z-score normalize the given dataset.

        @param xs: the input batched feature vectors as 2d array.
        @return:
        """
        mean = np.mean(xs, axis=0)
        std = np.std(xs, axis=0)
        out = (mean - xs) / std
        return out


class FeedForward:
    def __init__(self, input_dim: int, ff_dim: int, *,
                 target_cls: int,
                 rest_cls: int = None,
                 weight_scale: float = 1e-3,
                 reg: float = 0.0):
        self.input_dim = input_dim
        self.ff_dim = ff_dim
        self.linear = Linear(input_dim, ff_dim, weight_scale=weight_scale, reg=reg)
        self.relu = ReLU()
        self.norm = InverseNormalize()

        self.target_cls = target_cls
        self.rest_cls = rest_cls
        self.history = {'loss': [], 'accuracy': []}

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the feed forward network.

        @param xs: input feature vector as 1d or batched array.
        @return: processed feature vectors
        """
        fc = self.linear(xs, auto_grad=False)
        out = self.relu(fc, auto_grad=False)

        return out

    def train(self,
              data: np.ndarray,
              *,
              num_epochs: int,
              learning_rate: float,
              momentum: float = 0.9) -> (np.ndarray, np.ndarray):
        """
        Train the feed forward layer on given dataset with hyperparameters.

        @param data: the input dataset as 2d array with last column as labels.
        @param num_epochs: number of epochs to be trained.
        @param learning_rate: scalar learning rate of optimization.
        @param momentum: scalar giving momentum strength for gradient descent.
        @return: the processed feature vectors to be passed into next layer
        """
        xs = data[:, :-1]
        ys = data[:, -1]

        config = {'momentum': momentum, 'velocity': np.zeros((self.input_dim, self.ff_dim)),
                  'learning_rate': learning_rate}
        for _ in range(num_epochs):
            fc = self.linear(xs)
            relu = self.relu(fc)
            scores = np.sum(relu, axis=-1)

            binary_label = (ys == self.target_cls).astype(int)
            predict_label = (scores >= 0).astype(int)
            self.history['accuracy'].append(np.mean(np.equal(binary_label, predict_label)))
            self.history['loss'].append(np.mean(np.abs(binary_label - predict_label) * np.abs(scores)))

            dout = 2 * (predict_label - binary_label) * np.abs(predict_label - binary_label) * np.sign(scores)
            dout = self.relu.backward(dout)
            config = self.linear.auto_grad(dout, config)

        return np.concatenate((self.forward(xs), np.expand_dims(ys, axis=-1)), axis=-1)
