# -*- coding: utf-8 -*-
# Standard Library Imports
import cmath
import random

# Third-Party Imports
import numpy as np
from numpy.polynomial.polynomial import polydiv
from scipy.optimize import minimize
from scipy.stats import mode
from sklearn.utils.extmath import randomized_svd, weighted_mode

# Set random seed for reproducibility
random.seed(4217)

"""
RDML Implementation
Reference:
    Jin, R., Wang, S., & Zhou, Y. (2009). Regularized distance metric learning: Theory and algorithm.
    In Advances in neural information processing systems (pp. 862-870).

Overview:
    The algorithm examines the generalization error of regularized distance metric learning. It shows that with appropriate constraints, the generalization error could be independent from dimensionality, making it suitable for high-dimensional data. An efficient online learning algorithm is also presented, which has been found to be effective and robust in empirical studies.
"""


from typing import Tuple, List

def RDML(X: np.ndarray, Y: List[int], learning_rate: float = 0.1, max_iter: int = 1000) -> np.ndarray:
    """
    Implements the Regularized Discriminative Metric Learning (RDML) algorithm.
    Params:
    - X (np.ndarray): n x d input matrix with n patterns and d features.
    - Y (List[int]): n-dimensional array of labels.
    - learning_rate (float): The learning rate for the algorithm.
    - max_iter (int): The maximum number of iterations.
    
    Returns:
    - np.ndarray: The optimized matrix A.
    """
    X = ensure_matrix(X)
    n, d, A = initialize_params(X)

    for i in range(max_iter):
        pair, labels = select_pair(Y, n)
        A = update_matrix(A, X, pair, labels, learning_rate)
        
    return A

def ensure_matrix(X: np.ndarray) -> np.ndarray:
    """Ensures that X is a NumPy matrix."""
    return np.matrix(X) if not isinstance(X, np.matrix) else X

def initialize_params(X: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Initializes dimensions and zero matrix A."""
    n, d = X.shape
    A = np.matrix(np.zeros((d, d)))
    return n, d, A

def select_pair(Y: List[int], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Selects a random pair of samples."""
    pair = np.array(random.sample(range(n), 2))
    labels = Y[pair]
    return pair, labels

def update_matrix(A: np.ndarray, X: np.ndarray, pair: np.ndarray, labels: List[int], learning_rate: float) -> np.ndarray:
    """
    Updates the matrix A based on the selected pair and labels.
    """
    yt = 1.0 if labels[0] == labels[1] else -1.0
    xs = X[pair, :]
    xd = xs[0, :] - xs[1, :]

    if yt * (xd @ A @ xd.T) <= 0.0:
        if yt == -1:
            A -= learning_rate * yt * xd.T @ xd
        else:
            lambda_t = lambda_CG(A, xd, learning_rate)
            A -= lambda_t * yt * xd.T @ xd
            
    return A


def lambda_CG(A: np.ndarray, xd: np.ndarray, learning_rate: float) -> float:
    """
    Uses Newton CG to find an approximate solution for lambda.
    
    Parameters:
        A (np.ndarray): The current matrix A.
        xd (np.ndarray): The difference between selected data points.
        learning_rate (float): The learning rate.
    
    Returns:
        float: The optimized lambda value.
    """
    result = minimize(
        f_loss,
        x0=np.zeros(xd.shape[1]),
        hess=f_hess,
        options={'disp': False},
        method='Newton-CG',
        jac=f_grad,
        args=(A, xd.T)
    )
    if result.fun == 0.0:
        return 0.0
    else:
        return max(0, min(learning_rate, (-result.fun) ** -1))

def f_loss(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> Any:
    """
    Computes the loss function for the optimization.
    """
    u = np.matrix(u)
    return np.squeeze(np.asarray(-2.0 * u @ xdT + u @ A @ u.T))

def f_grad(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> np.ndarray:
    """
    Computes the gradient for the optimization.
    """
    u = np.matrix(u)
    return np.squeeze(np.asarray(-2.0 * xdT + (A + A.T) @ u.T))

def f_hess(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> np.ndarray:
    """
    Computes the Hessian for the optimization.
    """
    return np.asarray(A + A.T)

