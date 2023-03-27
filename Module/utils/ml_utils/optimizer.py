# global
import numpy as np
import pandas as pd

# local
from utils import ml_utils
from typing import Union, Sequence, Callable, Any


def random_step(arr_size: Union[int, Sequence], num_epochs: int,
                /,
                init_min: float, init_max: float, step_min: float, step_max: float,
                *,
                x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray],
                pred_func: Callable, loss_func: Callable) -> dict[str, Any]:
    """
    Trains the model with given size of parameters using Random Step algorithm.

    @param arr_size: shape of the training parameters to be initialized.
    @param num_epochs: number of training epochs.
    @param init_min: minimum value for parameter initialization.
    @param init_max: maximum value for parameter initialization.
    @param step_min: minimum value of random step taken.
    @param step_max: maximum value of random step taken.
    @param x: sequence of training data.
    @param y: sequence of training labels.
    @param pred_func: prediction function takes in `x` and array of parameters.
    @param loss_func: loss function takes in actual and predicted values.
    @return: dictionary of training results and training history.
    """
    if ml_utils.framework == 'pandas':
        x = x.to_numpy()
        y = y.to_numpy()
    elif ml_utils.framework is None:
        raise RuntimeError('Framework unspecified')

    # using numpy for backend execution
    ml_utils.numpy()
    x: np.ndarray
    y: np.ndarray

    if isinstance(arr_size, int):
        beta = np.random.rand(arr_size) * (init_max - init_min) + init_min
    else:
        beta = np.random.rand(*arr_size) * (init_max - init_min) + init_min
    loss = loss_func(pred_func(x, beta), y)
    history = [beta]

    for _ in range(num_epochs):
        step = np.random.uniform(step_min, step_max, arr_size)
        new_beta = beta + step
        new_loss = loss_func(pred_func(x, new_beta), y)
        # loss check
        if new_loss < loss:
            beta = new_beta
            loss = new_loss
        history.append(beta)

    while True:
        step = np.random.uniform(step_min, step_max, arr_size)
        new_beta = beta + step
        new_loss = loss_func(pred_func(x, new_beta), y)
        # loss check
        if new_loss < loss:
            beta = new_beta
            loss = new_loss
            history.append(beta)
        else:
            break

    return {'result': beta, 'history': history}


def gradient_descent(arr_size: Union[int, Sequence], num_epochs: int,
                     /,
                     init_min: float, init_max: float, learning_rate: float,
                     *,
                     x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray],
                     pred_func: Callable, grad_func: Callable) -> dict[str, Any]:
    """
    Train the model with given size of parameters using Gradient Descent algorithm.

    @param arr_size: shape of the training parameters to be initialized.
    @param num_epochs: number of training epochs.
    @param init_min: minimum value for parameter initialization.
    @param init_max: maximum value for parameter initialization.
    @param learning_rate: learning rate at each epoch.
    @param x: sequence of training data.
    @param y: sequence of training labels.
    @param pred_func: prediction function takes in `x` and array of parameters.
    @param grad_func: gradient function takes in `x`, `y` and prediction values.
    @return: dictionary of training results and training history.
    """
    if ml_utils.framework == 'pandas':
        x = x.to_numpy()
        y = y.to_numpy()
    elif ml_utils.framework is None:
        raise RuntimeError('Framework unspecified')

    # using numpy for backend execution
    ml_utils.numpy()
    x: np.ndarray
    y: np.ndarray

    if isinstance(arr_size, int):
        beta = np.random.rand(arr_size) * (init_max - init_min) + init_min
    else:
        beta = np.random.rand(*arr_size) * (init_max - init_min) + init_min
    history = [beta]

    for _ in range(num_epochs):
        grad = grad_func(x, y, pred_func(x, beta))
        step = -grad * learning_rate
        beta = beta + step
        history.append(beta)

    return {'result': beta, 'history': history}


def perceptron(data: Union[pd.Series, np.ndarray],
               /,
               learning_rate: float, num_epochs: int,
               *,
               pred_func: Callable, grad_func: Callable, seed: int = None) -> dict[str, Any]:
    """
    Train the model with basic Perceptron algorithm.

    @param data: sequence of all data with the last column as class labels.
    @param learning_rate: learning rate at each epoch.
    @param num_epochs: number of training epochs.
    @param pred_func: prediction function takes in `x` and array of parameters, returns a dictionary with keys for
    computed `score` and predicted labels `pred`.
    @param grad_func: gradient function takes in `x`, `y` and prediction labels.
    @param seed: random seed for random parameter initialization.
    @return: dictionary of training results and training history.
    """
    if ml_utils.framework == 'pandas':
        data: pd.Series
        x = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
    elif ml_utils.framework == 'numpy':
        data: np.ndarray
        x = data[:, :-1]
        y = data[:, -1]
    else:
        raise RuntimeError('Framework unspecified')

    # using numpy for backend execution
    ml_utils.numpy()
    x: np.ndarray
    y: np.ndarray

    if seed is not None:
        np.random.seed(seed)
    w = np.insert(np.random.normal(0, size=np.shape(x)[-1]), 0, 0)
    y_pred = pred_func(x, w)
    history = {'mpe': [ml_utils.mpe(y, **y_pred)], 'accuracy': [ml_utils.metrics(y, y_pred['pred'])['accuracy']]}

    for _ in range(num_epochs):
        grad = grad_func(x, y, pred_func(x, w)['pred'])
        step = grad * learning_rate
        w = w + step

        y_pred = pred_func(x, w)
        history['mpe'].append(ml_utils.mpe(y, **y_pred))
        history['accuracy'].append(ml_utils.metrics(y, y_pred['pred'])['accuracy'])

    return {'result': w, 'history': history}
