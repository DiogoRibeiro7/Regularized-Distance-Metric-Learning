import numpy as np
from scipy.optimize import minimize
from scipy.stats import mode
from sklearn.utils.extmath import randomized_svd, weighted_mode

# Set random seed for reproducibility
np.random.seed(4217)

#####################   RDML  #####################   
# Jin, R., Wang, S., & Zhou, Y. (2009). 
# Regularized distance metric learning: Theory and algorithm. 
# In Advances in neural information processing systems (pp. 862-870).

def RDML(X: np.ndarray, Y: np.ndarray, lmbda: float = 0.1, T: int = 1000) -> np.ndarray:
    """
    Regularized Distance Metric Learning (RDML).

    Parameters:
    X : np.ndarray
        Input matrix with shape (n_samples, n_features).
    Y : np.ndarray
        Labels for the input data with shape (n_samples,).
    lmbda : float
        Learning rate.
    T : int
        Maximum number of iterations.

    Returns:
    A : np.ndarray
        Learned distance metric.
    """
    X = np.array(X)
    n, d = X.shape
    A = np.zeros((d, d))

    for _ in range(T):
        # Choose a pair of indices without replacement
        pair = np.random.choice(n, size=2, replace=False)
        ys = Y[pair]
        yt = 1.0 if ys[0] == ys[1] else -1.0
        xs = X[pair, :]
        xd = xs[0, :] - xs[1, :]

        if yt * (xd @ A @ xd.T) > 0.0:
            # Correctly classified, no adaptation needed
            continue

        # Update A using approximate solution derived in RDML paper
        if yt == -1:
            A -= lmbda * yt * np.outer(xd, xd)
        else:
            lmbda_t = lambda_CG(A, xd, lmbda)
            A -= lmbda_t * yt * np.outer(xd, xd)

    return A


def lambda_CG(A: np.ndarray, xd: np.ndarray, lmbda: float) -> float:
    """
    Compute lambda using Newton Conjugate Gradient method.

    Parameters:
    A : np.ndarray
        Current distance metric matrix.
    xd : np.ndarray
        Difference between selected samples.
    lmbda : float
        Learning rate.

    Returns:
    float
        Computed lambda value.
    """
    result = minimize(f_loss, x0=np.zeros(xd.shape[0]), hess=f_hess, jac=f_grad, method='Newton-CG', args=(A, xd.T))
    if result.fun == 0.0:
        return 0.0
    else:
        return max(0, min(lmbda, (-result.fun) ** -1))


def f_loss(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> float:
    u = np.array(u)
    return -2.0 * u @ xdT + u @ A @ u.T


def f_grad(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> np.ndarray:
    u = np.array(u)
    return -2.0 * xdT + (A + A.T) @ u


def f_hess(u: np.ndarray, A: np.ndarray, xdT: np.ndarray) -> np.ndarray:
    return A + A.T


#####################  Utility Functions #####################

def near_psd(A: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Project a matrix onto the nearest positive semi-definite matrix.

    Parameters:
    A : np.ndarray
        Input matrix.
    epsilon : float
        Small positive value to ensure positive definiteness.

    Returns:
    np.ndarray
        Nearest positive semi-definite matrix.
    """
    eigval, eigvec = np.linalg.eigh(A)
    eigval = np.maximum(eigval, epsilon)
    return eigvec @ np.diag(eigval) @ eigvec.T


def Tfunc_fast(M: np.ndarray, theta: float) -> np.ndarray:
    """
    Faster proximal mapping.

    Parameters:
    M : np.ndarray
        Input matrix.
    theta : float
        Threshold parameter.

    Returns:
    np.ndarray
        Thresholded matrix.
    """
    return np.maximum(M - theta, 0) + np.minimum(M, 0)


##################### Tests #####################

def mahalanobis(x: np.ndarray, y: np.ndarray, A: np.ndarray) -> float:
    """
    Compute Mahalanobis distance between two points.

    Parameters:
    x, y : np.ndarray
        Input points.
    A : np.ndarray
        Metric matrix.

    Returns:
    float
        Mahalanobis distance.
    """
    xd = x - y
    return float(xd @ A @ xd.T)


def sim_knn_predict(X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, L: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Similarity-based k-NN prediction.

    Parameters:
    X : np.ndarray
        Training data.
    Y : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    L : np.ndarray
        Transformation matrix.
    k : int
        Number of neighbors.

    Returns:
    np.ndarray
        Predicted labels.
    """
    X = X @ L.T
    X_test = X_test @ L.T
    n = X_test.shape[0]
    y_pred = np.empty(n, dtype=Y.dtype)

    for i in range(n):
        distances = np.dot(X, X_test[i, :])
        neighbors = np.argpartition(-distances, k)[:k]
        label, _ = weighted_mode(Y[neighbors], distances[neighbors])
        y_pred[i] = label

    return y_pred


def test():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    digits = load_digits()
    X = digits.data
    Y = digits.target

    skf = StratifiedKFold(n_splits=10, random_state=17, shuffle=True)
    res = []
    print('Testing Liu et al.')
    for train_idx, test_idx in skf.split(X, Y):
        L = RDML(X[train_idx], Y[train_idx], lmbda=0.1, T=500)
        preds = sim_knn_predict(X[train_idx], Y[train_idx], X[test_idx], L, k=3)
        acc = np.mean(preds == Y[test_idx])
        res.append(acc)
        print(f'Accuracy: {acc:.4f}')

    res = np.array(res)
    print(f'Average accuracy: {res.mean():.4f}')
    print(f'Standard deviation: {res.std():.4f}')


if __name__ == "__main__":
    test()

