"""
SYDE 572 Assignment 1, Part 2: least-squares fitting by minimising the MSE.

Both models are linear in their parameters, y_hat = A @ theta, where the
design matrix A has one row per data point and one column per basis function
(notation from the lecture slides "Fitting a straight line" and
"Example: best fit parabola"):

    line      y = c0 + c1 x          A = [1  x]
    parabola  y = B x^2 + C x + D    A = [x^2  x  1]

For N points,

    MSE(theta)       = (1/N) ||A theta - y||^2
    gradient g       = (2/N) A^T (A theta - y)
    Hessian  H       = (2/N) A^T A          (constant, since MSE is quadratic)

Solvers:

    analytic_fit       set g = 0, giving the normal equations
                       A^T A theta = A^T y, and solve them by Gaussian elimination
    coordinate_newton  the method the assignment asks for: Newton-Raphson on one
                       parameter at a time, holding the others fixed
    full_newton        one multivariate Newton step theta - H^{-1} g, which lands
                       exactly on the minimum because the MSE is quadratic
"""
import numpy as np

# "order" is the order in which coordinate Newton updates the parameters.
# The line updates the slope first, as the assignment lists slope then intercept.
MODELS = {
    "line": {
        "label": "y = c₀ + c₁x",
        "params": ["c₀", "c₁"],
        "basis": lambda x: [np.ones_like(x), x],
        "order": [1, 0],
    },
    "parabola": {
        "label": "y = Bx² + Cx + D",
        "params": ["B", "C", "D"],
        "basis": lambda x: [x ** 2, x, np.ones_like(x)],
        "order": [0, 1, 2],
    },
}


def design_matrix(model, x):
    """A[i, j] = j-th basis function evaluated at x_i."""
    return np.column_stack(MODELS[model]["basis"](np.asarray(x, float)))


def predict(model, theta, x):
    return design_matrix(model, x) @ np.asarray(theta, float)


def mse(model, theta, x, y):
    residual = predict(model, theta, x) - y
    return float(np.mean(residual ** 2))


def gradient(model, theta, x, y):
    A = design_matrix(model, x)
    return (2 / len(x)) * A.T @ (A @ theta - y)


def hessian(model, x):
    A = design_matrix(model, x)
    return (2 / len(x)) * A.T @ A


def gauss_solve(M, b):
    """Solve M z = b by Gaussian elimination with partial pivoting."""
    M = np.array(M, float)
    b = np.array(b, float)
    n = len(b)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(M[col:, col])))
        M[[col, pivot]], b[[col, pivot]] = M[[pivot, col]], b[[pivot, col]]
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]
            b[row] -= factor * b[col]
    z = np.zeros(n)
    for row in range(n - 1, -1, -1):
        z[row] = (b[row] - M[row, row + 1:] @ z[row + 1:]) / M[row, row]
    return z


def normal_equations(model, x, y):
    """Return (A^T A, A^T y), the two sides of the normal equations."""
    A = design_matrix(model, x)
    return A.T @ A, A.T @ y


def analytic_fit(model, x, y):
    """Exact least-squares parameters from the normal equations."""
    return gauss_solve(*normal_equations(model, x, y))


def coordinate_newton(model, x, y, tolerance=1e-7, max_iterations=10000):
    """
    Newton-Raphson on one parameter at a time, starting from theta = 0.

    Each iteration n updates every parameter j once, in MODELS[model]["order"],
    using theta_j <- theta_j - g_j / H_jj with the other parameters held at
    their latest values. Stops when no parameter changes by more than
    `tolerance` in an iteration.

    Returns (theta, history). history[0] is the start; every later entry is
    one single-parameter update with its iteration n.
    """
    theta = np.zeros(len(MODELS[model]["params"]))
    H = hessian(model, x)
    history = [{"iteration": 0, "theta": theta.copy(), "mse": mse(model, theta, x, y)}]
    for iteration in range(1, max_iterations + 1):
        previous = theta.copy()
        for j in MODELS[model]["order"]:
            g = gradient(model, theta, x, y)[j]
            theta[j] -= g / H[j, j]
            history.append({"iteration": iteration, "theta": theta.copy(), "mse": mse(model, theta, x, y)})
        if np.max(np.abs(theta - previous)) < tolerance:
            break
    return theta, history


def full_newton(model, x, y):
    """One multivariate Newton step from theta = 0: theta = -H^{-1} g(0)."""
    theta0 = np.zeros(len(MODELS[model]["params"]))
    return theta0 - gauss_solve(hessian(model, x), gradient(model, theta0, x, y))
