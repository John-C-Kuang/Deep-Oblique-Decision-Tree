# global
import numpy as np
import pandas as pd

# local
from tqdm import tqdm


class Linear:
    def __init__(self, input_dim: int, ff_dim: int, *,
                 weight_scale: float = 0.1,
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

        return self.forward(*args, **kwargs)

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

    def backward(self, dout: np.ndarray, config: dict[str, float]) -> (np.ndarray, dict[str, float]):
        """
        Gradient descent optimizer of the linear layer.

        @param dout: upstream derivative with shape (batch, ff_dim).
        @param config: keyword configurations for gradient descent with momentum.
        @return: configurations for the next optimizer iteration.
        """
        dx = dout @ self.w.T
        dw = self.cache_x.T @ dout + self.reg * self.w
        db = np.sum(dout, axis=0)
        self.b -= config['learning_rate'] * db
        v = config['momentum'] * config['velocity'] - config['learning_rate'] * dw
        config['velocity'] = v
        self.w += v

        return dx, config


class ReLU:
    def __init__(self):
        self.cache_x = None

    def __call__(self, *args, **kwargs) -> np.ndarray:
        if len(args) != 1:
            raise TypeError('ReLU layer forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0 and 'auto_grad' not in kwargs:
            raise TypeError("ReLU layer forward pass expects only 1 keyword argument 'auto_grad'")

        return self.forward(*args, **kwargs)

    def forward(self, xs: np.ndarray, *, auto_grad: bool = True) -> np.ndarray:
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
        return dout * (self.cache_x >= 0).astype(int)


class Sigmoid:
    def __init__(self):
        self.cache = None

    def __call__(self, *args, **kwargs) -> np.ndarray:
        if len(args) != 1:
            raise TypeError('Sigmoid layer forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0 and 'auto_grad' not in kwargs:
            raise TypeError("Sigmoid layer forward pass expects only 1 keyword argument 'auto_grad'")

        return self.forward(*args, **kwargs)

    def forward(self, xs: np.ndarray, *, auto_grad: bool = True) -> np.ndarray:
        """
        Forward pass of the sigmoid activation layer.

        @param xs: input feature vectors with shape (batch,).
        @param auto_grad: boolean flag indicates if layer parameters are trainable.
        @return: processed probability vectors with shape (batch,).
        """
        sigmoid = 1 / (1 + np.exp(-xs))
        if auto_grad:
            self.cache = sigmoid

        return sigmoid

    def backward(self, dout: np.array):
        """
        Backward pass of the Sigmoid activation layer.

        @param dout: upstream derivative with shape (batch,).
        @return: downstream derivative with respect to x.
        """
        return dout * self.cache * (1 - self.cache)


class Normalize:
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
        Z-score normalize the given dataset.

        @param xs: the input batched feature vectors as 2d array.
        @return:
        """
        mean = np.mean(xs, axis=0)
        std = np.std(xs, axis=0)
        out = (xs - mean) / std
        return out


class FeedForward:
    def __init__(self, input_dim: int, ff_dim: int, *,
                 target_cls: int,
                 reg: float = 0.0,
                 sigmoid_threshold: float = 0.5):
        """
        Initialize the parameters of a feed forward network.

        @param input_dim: integer dimension of the input data.
        @param ff_dim: integer dimension of the hidden layer.
        @param target_cls: integer label for the target class to be classified as 1.
        @param reg: strength of the L2-Regularization.
        @param sigmoid_threshold: threshold for label prediction using sigmoid score.
        """
        self.input_dim = input_dim
        self.ff_dim = ff_dim
        self.linear = Linear(input_dim, ff_dim, weight_scale=np.sqrt(1 / input_dim), reg=reg)
        self.relu = ReLU()
        self.perceptron = Linear(ff_dim, 1, weight_scale=np.sqrt(1 / ff_dim), reg=reg)
        self.sigmoid = Sigmoid()
        self.sigmoid_threshold = sigmoid_threshold

        self.target_cls = target_cls
        self.history = {'loss': [], 'accuracy': []}

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise TypeError('Network forward pass expects only 1 positional argument, found {}'.format(len(args)))
        if len(kwargs) > 0 and 'train' not in kwargs:
            raise TypeError("Network pass expects only 1 keyword argument 'train'")

        return self.forward(*args, **kwargs)

    def forward(self, xs: np.ndarray, *, train: bool = False) -> np.ndarray:
        """
        Forward pass of the feed forward network.

        @param xs: input feature vector as 1d or batched array.
        @param train: boolean flag indicates if the networking is training.
        @return: processed feature vectors
        """
        fc = self.linear(xs, auto_grad=False)
        if train:
            return fc

        relu = self.relu(fc, auto_grad=False)
        score = self.perceptron(relu, auto_grad=False)
        out = self.sigmoid(score, auto_grad=False)

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

        config_linear = {'momentum': momentum, 'velocity': np.zeros((self.input_dim, self.ff_dim)),
                         'learning_rate': learning_rate}
        config_perceptron = {'momentum': momentum, 'velocity': np.zeros((self.ff_dim, 1)),
                             'learning_rate': learning_rate}
        gt_label = (ys == self.target_cls).astype(int)
        for _ in tqdm(range(num_epochs), 'Training for {} epochs'.format(num_epochs)):
            fc = self.linear(xs)
            relu = self.relu(fc)
            scores = self.perceptron(relu)
            prob = self.sigmoid(scores)

            predict_label = np.squeeze((prob >= self.sigmoid_threshold).astype(int))
            # predict_label = (scores >= 0).astype(int)
            self.history['accuracy'].append(1 - np.mean(np.abs(gt_label - predict_label)))
            self.history['loss'].append(
                -np.mean(gt_label * np.log(np.squeeze(prob)) + (1 - gt_label) * np.log(1 - np.squeeze(prob))))

            dout = (prob - gt_label[:, np.newaxis]) / (gt_label.shape[0] * prob * (1 - prob))
            dout = self.sigmoid.backward(dout)
            dout, config_perceptron = self.perceptron.backward(dout, config_perceptron)
            dout = self.relu.backward(dout)
            _, config_linear = self.linear.backward(dout, config_linear)

        return np.concatenate((self.forward(xs, train=True), ys[:, np.newaxis]), axis=-1)


if __name__ == '__main__':
    """
    from collections import Counter
    from dataset.preprocess import preprocess_wine_quality
    from sklearn.metrics import accuracy_score

    df = pd.read_csv('./../../dataset/Wine_Quality_Data.csv')
    df = preprocess_wine_quality(df)
    ff = FeedForward(12, 30, target_cls=6, reg=0.1)
    print(Counter(df['quality']))
    data = df.to_numpy()
    ff.train(data, num_epochs=800, learning_rate=1e-6)
    print(ff.history['loss'])
    print(ff.history['accuracy'])
    x = data[:, :-1]
    y = data[:, -1]
    score = (np.squeeze(ff(x)) >= 0.5).astype(int)
    print(np.sum(score))
    perceptron = Linear(12, 1, reg=0.3)
    x = data[:, :-1]
    y = data[:, -1]
    gt = (y == 5).astype(int)
    config_perceptron = {'momentum': 0.9, 'velocity': np.zeros((12, 1)),
                         'learning_rate': 1e-6}
    for _ in range(200):
        y_pred = (np.sum(perceptron(x), axis=-1) >= 0).astype(int)
        dout = (y_pred - gt)[:, np.newaxis]
        _, config_perceptron = perceptron.backward(dout, config_perceptron)

    pred = (np.sum(perceptron(x), axis=-1) >= 0).astype(int)
    print(sum(pred))
    print(accuracy_score(gt, pred))
    """
