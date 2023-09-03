# -*- coding: utf-8 -*-
# Standard Library Imports
from typing import List, Tuple, Any
import scipy
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


def select_pair(Y: List[int], n: int) -> Tuple:
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

#####################  LowRankBiLinear #####################
# Method due to Liu et al. (2015)
#           Low-Rank Similarity Metric Learning in High Dimensions


def Tfunc_fast(M: np.ndarray, theta: float) -> np.ndarray:
    """
    Fast proximal mapping function for the matrix M.

    Parameters:
        M (np.ndarray): Input matrix.
        theta (float): Threshold for the mapping.

    Returns:
        np.ndarray: Matrix after applying fast proximal mapping.
    """
    A = M.copy()
    A[M > 0] = 0
    return np.maximum(M - theta, 0) + A

def nearPSD_simple(A: np.ndarray) -> np.ndarray:
    """
    Projects a given matrix onto the positive semi-definite (PSD) cone.

    Parameters:
        A (np.ndarray): Input matrix to be projected.

    Returns:
        np.ndarray: Matrix projected onto the PSD cone.
    """
    eig_val, eig_vec = np.linalg.eig(A)
    eig_ind = eig_val > 0
    B = np.zeros(A.shape)
    for val, vec in zip(eig_val[eig_ind], eig_vec[eig_ind]):
        B += val * vec.T @ vec
    return B

def perform_ADMM_step(Xtil: np.ndarray, XtilT: np.ndarray, I: np.ndarray, W: np.ndarray, E2: np.ndarray,
                      Ym: np.ndarray, Ytil: np.ndarray, L: np.ndarray, rho: float, alpha: float, tau: float) -> Tuple:
    """
    Performs one step of the Alternating Direction Method of Multipliers (ADMM) algorithm.

    Parameters:
        Xtil, XtilT, I, W, E2, Ym, Ytil, L (np.ndarray): Matrices and vectors involved in the algorithm.
        rho, alpha, tau (float): Scalar parameters for the ADMM algorithm.

    Returns:
        Tuple: Updated matrices and vectors (Z, W, S, Delta, L) after one ADMM step.
    """
    Z = Tfunc_fast(Ytil - np.multiply(Ym, XtilT @ Xtil) - L, 1.0 / rho)
    G = I + Xtil @ np.multiply(Ym, Z - Ytil + L) @ XtilT + E2 @ W @ E2
    W = nearPSD_simple(W - tau * G)
    S = XtilT @ W @ Xtil
    Delta = Z - Ytil + np.multiply(Ym, S)
    L += Delta
    return Z, W, S, Delta, L

def convergence_check(Delta: np.ndarray, n: int, tol: float) -> bool:
    """
    Checks if the algorithm has reached convergence based on the given tolerance level.

    Parameters:
        Delta (np.ndarray): The change in the matrix after an iteration.
        n (int): Number of observations.
        tol (float): Convergence tolerance.

    Returns:
        bool: True if the algorithm has converged, False otherwise.
    """
    return np.sum(np.multiply(Delta, Delta)) / n ** 2 <= tol


def LowRankBiLinear(m: int, X: np.ndarray, Y: List[int], alpha: float, eps: float, rho: float, tau: float,
                    T: int, tol: float = 1e-6, epsilon: float = 1e-8) -> np.ndarray:
    """
    Implements the Low-Rank Bi-Linear similarity metric learning algorithm based on the method by Liu et al. (2015).

    Parameters:
        m (int): Dimension of similarity function.
        X (np.ndarray): An n x d data matrix with n observations and d dimensions.
        Y (List[int]): An n x 1 label vector.
        alpha (float): Regularization strength.
        eps (float): Margin parameter for the metric. Should be > 0, a small positive value.
        rho (float): Penalty parameter for the augmentation term of the Lagrangian in ADMM. Should be > 0.
        tau (float): Step size for the W updates in ADMM.
        T (int): Maximum number of iterations for the ADMM algorithm.
        tol (float, optional): Convergence tolerance. Default is 1e-6.
        epsilon (float, optional): Parameter for the projection onto the PSD cone. Default is 1e-8.

    Returns:
        np.ndarray: The optimal low-rank basis.

    Notes:
        The function employs the Alternating Direction Method of Multipliers (ADMM) for optimization.
        It also uses projections onto the positive semi-definite (PSD) cone.
    """

    if not isinstance(X, np.matrix):
        X = np.matrix(X)

    n, d = X.shape
    Ym = -np.matrix(np.ones((n, n)))
    Ytil = eps * np.matrix(np.ones((n, n)))

    for y in np.unique(Y):
        y_vec = Y == y
        Y_set = np.outer(y_vec, y_vec)
        Ym[Y_set] = 1
        Ytil[Y_set] = 1

    X = X.T

    if scipy.sparse.issparse(X):
        U, E, _ = scipy.sparse.linalg.svds(X, k=m)
    else:
        U, E, _ = np.linalg.svd(X, full_matrices=False)
        U = U[:, :m]
        E = E[:m]

    U = np.asmatrix(U)
    E2 = np.matrix(np.diag(E ** 2))
    Xtil = U.T @ X
    XtilT = Xtil.T
    I = np.matrix(np.identity(m))
    W = I.copy()
    I = alpha / rho * I
    L = np.matrix(np.zeros((n, n)))

    for k in range(T):
        Z, W, S, Delta, L = perform_ADMM_step(
            Xtil, XtilT, I, W, E2, Ym, Ytil, L, rho, alpha, tau)

        if convergence_check(Delta, n, tol):
            break

    E, H = np.linalg.eig(W)
    E = np.maximum(E, epsilon)
    out = U @ H @ np.diag(np.sqrt(E))

    return out
