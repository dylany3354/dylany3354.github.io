"""
Linear basis function regression: basis functions, least-squares and ridge solvers,
RMS error and k-fold cross-validation.

The model is
    y_hat = f(x) = sum_k w_k phi_k(x) = phi(x)^T w.
Phi is the N x K design matrix with one row phi(x_n)^T per sample, so y_hat = Phi w.

    lambda = 0:  w* = (Phi^T Phi)^-1 Phi^T y = pinv(Phi) y    (pseudo-inverse)
    lambda > 0:  w* = (Phi^T Phi + lambda I)^-1 Phi^T y       (ridge regression)
"""
import numpy as np


# ---------------------------------------------------------------- basis functions

def polynomial_design(X, degree):
    """
    Power basis phi(x) = x^p applied to each feature separately (no cross-terms).

    Columns: [1, x_1..x_d, x_1^2..x_d^2, ..., x_1^M..x_d^M] for M = degree, so there are
    1 + d*M columns and column 0 is the bias. X is N x d (or a 1-D array for one feature).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    cols = [np.ones((X.shape[0], 1))]
    for p in range(1, degree + 1):
        cols.append(X ** p)
    return np.hstack(cols)


def gaussian_design(X, centers, s):
    """
    Gaussian basis phi_k(x) = exp(a ||x - c_k||^2) with a = -1/(2 s^2), plus a bias column.

    X is N x d, centers is K x d, s is the width (the same for every basis function).
    Returns an N x (K + 1) matrix; column 0 is the bias.
    """
    X = np.atleast_2d(X)
    sq_dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    a = -1.0 / (2.0 * s ** 2)
    return np.hstack([np.ones((X.shape[0], 1)), np.exp(a * sq_dist)])


def choose_centers(X_train, K, rng):
    """Random permutation of the training points; the first K become the centers c_k."""
    perm = rng.permutation(X_train.shape[0])
    return X_train[perm[:K]]


# ---------------------------------------------------------------- solvers

def fit(Phi, y, lam=0.0):
    """
    Minimize J(w) = 1/2 ||y - Phi w||^2 + lambda/2 ||w||^2.

    lambda = 0: least squares through the SVD-based pseudo-inverse (np.linalg.lstsq). If some
    columns are linearly dependent (CHAS is constant in the training rows, and high powers of
    features with few distinct values repeat each other), it returns the minimum-norm solution.
    lambda > 0: ridge regression, w* = (Phi^T Phi + lambda I)^-1 Phi^T y. Every weight is
    penalized, including the bias w_0, so a large lambda shrinks the predictions toward 0.
    """
    if lam == 0:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return w
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y)


def predict(Phi, w):
    """y_hat = Phi w, one prediction per row of Phi."""
    return Phi @ w


def rms_error(y_hat, y):
    """Root-mean-square error, sqrt(mean((y_hat - y)^2)), in the units of y ($1000s)."""
    return float(np.sqrt(np.mean((np.asarray(y_hat) - np.asarray(y)) ** 2)))


# ---------------------------------------------------------------- cross-validation

def kfold_indices(n, k, rng):
    """Shuffle 0..n-1 once and cut it into k folds of (almost) equal size."""
    return np.array_split(rng.permutation(n), k)


def cross_validate(X, y, make_design, lam, k=10, seed=0, normalizer=None):
    """
    k-fold cross-validation of one value of lambda.

    The rows are shuffled once with `seed`, so the same seed gives the same folds for every
    lambda. For each fold, the other k-1 folds are the training part. If `normalizer` is
    given (e.g. data_utils.fit_normalizer), the features are normalized with the training
    part's mean and standard deviation. make_design(X_train_part, X_val_part) returns the two
    design matrices. Returns the validation RMS error of each fold; the CV error is their mean.
    """
    rng = np.random.default_rng(seed)
    folds = kfold_indices(len(y), k, rng)
    errors = []
    for i in range(k):
        val = folds[i]
        trn = np.hstack([folds[j] for j in range(k) if j != i])
        X_trn, X_val = X[trn], X[val]
        if normalizer is not None:
            mean, std = normalizer(X_trn)
            X_trn, X_val = (X_trn - mean) / std, (X_val - mean) / std
        Phi_trn, Phi_val = make_design(X_trn, X_val)
        w = fit(Phi_trn, y[trn], lam)
        errors.append(rms_error(predict(Phi_val, w), y[val]))
    return np.array(errors)
