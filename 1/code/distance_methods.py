"""
SYDE 572 - Assignment 1, Part 1
Shortest distance from a point (x0, y0) to a curve y = f(x).

Solvers:
  * analytic_parabola  - exact solution for any parabola y = a x^2 + b x + c
                         (D'(x) = 0 reduces to a cubic; its real roots come from np.roots)
  * find_distance_newton  - Algorithm A from the assignment handout:
                            Newton-Raphson on D'(x) = 0 (needs f, f', f'')
  * golden_section_search - Algorithm B from the assignment handout:
                            golden section search on D(x) over [a, b]
                            (needs only f)
Algorithms A and B are copied from the handout; the only change is an optional `history`
list so that every iteration can be tabulated and plotted.
"""
import math
import numpy as np


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def dist_sq(x, x0, y0, f):
    """Squared distance D(x) = (x - x0)^2 + (f(x) - y0)^2."""
    return (x - x0) ** 2 + (f(x) - y0) ** 2


# ---------------------------------------------------------------------------
# Algorithm A: Newton-Raphson
# Copied from the assignment handout. The only change is the optional
# `history` argument, which records each iteration for the tables and plots
# (lines marked "# added").
# ---------------------------------------------------------------------------
def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0,
                         tolerance=1e-7, max_iter=100, history=None):   # added: history
    x = initial_guess
    for _ in range(max_iter):
        # D'(x)
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        # D''(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        # Newton-Raphson update step
        next_x = x - D_prime / D_double_prime
        if history is not None:                                          # added
            history.append({"k": len(history), "x": x, "Dp": D_prime,    # added
                            "x_next": next_x})                           # added
        if abs(next_x - x) < tolerance:
            break
        x = next_x
    shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5
    return shortest_distance, x


# ---------------------------------------------------------------------------
# Algorithm B: Golden Section Search
# Copied from the assignment handout. The only change is the optional
# `history` argument (lines marked "# added").
# ---------------------------------------------------------------------------
def golden_section_search(x0, y0, f, a, b, tolerance=1e-7, history=None):  # added: history
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    while abs(b - a) > tolerance:
        if history is not None:                                          # added
            history.append({"k": len(history), "a": a, "b": b, "x1": x1, "x2": x2,   # added
                            "keep": "left" if f_x1 < f_x2 else "right"})  # added
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
    if history is not None:                                              # added
        history.append({"k": len(history), "a": a, "b": b})                # added
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x


# Convenience wrappers used by part1_distance.py: call the handout functions
# and also return the recorded iterations.
def newton_distance(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    hist = []
    d, x = find_distance_newton(x0, y0, f, df, ddf, initial_guess, tolerance, max_iter, history=hist)
    return d, x, hist


def golden_distance(x0, y0, f, a, b, tolerance=1e-7):
    hist = []
    d, x = golden_section_search(x0, y0, f, a, b, tolerance, history=hist)
    return d, x, hist


# ---------------------------------------------------------------------------
# Analytic solution for any parabola
# ---------------------------------------------------------------------------
def parabola(a, b, c):
    """Return f, f', f'' for y = a x^2 + b x + c."""
    f = lambda x: a * x ** 2 + b * x + c
    df = lambda x: 2 * a * x + b
    ddf = lambda x: 2 * a + 0 * x
    return f, df, ddf


def analytic_parabola(a, b, c, x0, y0):
    """
    Exact minimiser for y = a x^2 + b x + c.

    D'(x)/2 = (x - x0) + (a x^2 + b x + c - y0)(2 a x + b) = 0 expands to
        2a^2 x^3 + 3ab x^2 + (b^2 + 2a(c - y0) + 1) x + b(c - y0) - x0 = 0.
    Every real root is a critical point; the one with the smallest D is the
    global minimum. Returns (distance, x_star).
    """
    f, _, _ = parabola(a, b, c)
    coeffs = [2 * a * a, 3 * a * b, b * b + 2 * a * (c - y0) + 1, b * (c - y0) - x0]
    roots = np.roots(coeffs)
    real = [r.real for r in roots if abs(r.imag) < 1e-9]
    best = min(real, key=lambda r: dist_sq(r, x0, y0, f))
    return math.sqrt(dist_sq(best, x0, y0, f)), best


# ---------------------------------------------------------------------------
# Any curve: robust starting point
# ---------------------------------------------------------------------------
def shortest_distance(x0, y0, f, df, ddf, lo, hi, n_grid=400):
    """
    Distance from (x0, y0) to y = f(x) on the domain [lo, hi].

    A coarse grid scan finds the best starting region. Newton then starts from
    the best grid point, and golden section searches the grid cell around it.
    This avoids converging to a local maximum or a far-away local minimum
    when D(x) has more than one critical point.
    """
    grid = np.linspace(lo, hi, n_grid)
    D = np.array([dist_sq(x, x0, y0, f) for x in grid])
    i = int(np.argmin(D))
    step = grid[1] - grid[0]
    a, b = max(lo, grid[i] - step), min(hi, grid[i] + step)
    newton = newton_distance(x0, y0, f, df, ddf, initial_guess=grid[i])
    golden = golden_distance(x0, y0, f, a, b)
    return {"newton": newton, "golden": golden, "start": grid[i], "bracket": (a, b)}
