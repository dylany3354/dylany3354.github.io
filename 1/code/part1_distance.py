"""
SYDE 572 Assignment 1, Part 1.

Run from this directory with::

    python part1_distance.py

The script follows the assignment's two numerical methods:

* Algorithm A: Newton-Raphson on D'(x) = 0, starting at x^(0) = 0.
* Algorithm B: golden-section search on D(x), using [-2, 2].

It prints a results table and writes six figures to ../figures/part1:

* p1_five_points.png: both methods for all five assigned points.
* p1_newton_geometry.png: Newton iterates on the curve for (-4, 0).
* p1_newton_convergence.png: |D'| and step size per Newton iteration.
* p1_golden_brackets.png: golden-section brackets, probes and final x*.
* p1_convergence_comparison.png: Newton error vs golden bracket width.
* p1_non_polynomial.png: both methods applied to four other curves.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from distance_methods import (
    analytic_parabola,
    golden_distance,
    newton_distance,
    parabola,
    shortest_distance,
)
from plot_style import (CURVE, GRID, GUIDE, INK, INK2, NEWTON, NEWTON_LIGHT, NEWTON_MARKER, OTHER,
                        OTHER_LIGHT, OTHER_MARKER, POINT_SIZE, SINGLE_FIGSIZE, SOLUTION, apply_style,
                        figure_title, note_box, plot_points, save)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "part1")
os.makedirs(FIG, exist_ok=True)
apply_style()

TOLERANCE = 1e-7
GOLDEN_BRACKET = (-2.0, 2.0)
ASSIGNED_POINTS = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]


def point_label(point):
    return f"({point[0]:g}, {point[1]:g})"


def run_assigned_problem():
    """Run both required methods for the five points in the handout."""
    f, df, ddf = parabola(1.0, 0.0, 5.0)
    results = []
    for point in ASSIGNED_POINTS:
        x0, y0 = point
        analytic_d, analytic_x = analytic_parabola(1.0, 0.0, 5.0, x0, y0)
        newton_d, newton_x, newton_history = newton_distance(
            x0, y0, f, df, ddf, initial_guess=0.0, tolerance=TOLERANCE
        )
        golden_d, golden_x, golden_history = golden_distance(
            x0, y0, f, *GOLDEN_BRACKET, tolerance=TOLERANCE
        )
        results.append({
            "point": point,
            "analytic": {"x": analytic_x, "d": analytic_d},
            "newton": {"x": newton_x, "d": newton_d,
                       "iterations": len(newton_history), "history": newton_history},
            "golden": {"x": golden_x, "d": golden_d,
                       "iterations": len(golden_history) - 1, "history": golden_history},
        })
    return f, results


def run_non_polynomial_examples():
    """Run both methods on four representative non-polynomial curves."""
    examples = [
        ("Exponential", "y = eˣ", lambda x: np.exp(x), lambda x: np.exp(x),
         lambda x: np.exp(x), (1, 0), (-2, 2), (-2.5, 2.5)),
        ("Logarithmic", "y = ln x", lambda x: np.log(x), lambda x: 1 / x,
         lambda x: -1 / x ** 2, (0, 0), (0.05, 3), (0.05, 3)),
        ("Reciprocal", "y = 1/x", lambda x: 1 / x, lambda x: -1 / x ** 2,
         lambda x: 2 / x ** 3, (0, 0), (0.1, 4), (0.1, 4)),
        ("Radical", "y = √x", lambda x: np.sqrt(x), lambda x: 0.5 / np.sqrt(x),
         lambda x: -0.25 * x ** -1.5, (4, 0), (0.01, 6), (0.01, 6)),
    ]
    results = []
    for name, equation, f, df, ddf, point, domain, plot_window in examples:
        numerical = shortest_distance(point[0], point[1], f, df, ddf, *domain)
        newton_d, newton_x, newton_history = numerical["newton"]
        golden_d, golden_x, golden_history = numerical["golden"]
        results.append({
            "name": name,
            "equation": equation,
            "f": f,
            "point": point,
            "plot_window": plot_window,
            "newton": {"x": newton_x, "d": newton_d, "history": newton_history},
            "golden": {"x": golden_x, "d": golden_d, "history": golden_history},
        })
    return results


def draw_curve(ax, f, window, label):
    xs = np.linspace(*window, 700)
    with np.errstate(all="ignore"):
        ys = f(xs)
    ax.plot(xs, ys, color=CURVE, linewidth=2, label=label, zorder=2)


def draw_closest_points(ax, f, result, segments=True):
    """Newton and golden-section closest points, optionally with their distance segments."""
    x0, y0 = result["point"]
    nx, gx = result["newton"]["x"], result["golden"]["x"]
    if segments:
        ax.plot([x0, nx], [y0, f(nx)], color=NEWTON, linestyle="--", linewidth=1.3)
        ax.plot([x0, gx], [y0, f(gx)], color=OTHER, linestyle=":", linewidth=1.8)
    plot_points(ax, nx, f(nx), "newton")
    plot_points(ax, gx, f(gx), "other", zorder=6)


def closest_point_legend(ax, segments=True):
    plot_points(ax, [], [], "data", label="given point")
    plot_points(ax, [], [], "newton", label="Newton closest point")
    plot_points(ax, [], [], "other", label="golden closest point")
    if segments:
        ax.plot([], [], color=NEWTON, linestyle="--", label="Newton segment")
        ax.plot([], [], color=OTHER, linestyle=":", label="golden segment")


def plot_five_points(f, results):
    fig, (overview_ax, detail_ax) = plt.subplots(1, 2, figsize=(12.5, 7.2),
                                                  gridspec_kw={"width_ratios": [1.25, 1]})
    draw_curve(overview_ax, f, (-9, 7), r"$y=x^2+5$")
    draw_curve(detail_ax, f, (-0.82, 0.82), r"$y=x^2+5$")
    for result in results:
        x0, y0 = result["point"]
        plot_points(overview_ax, x0, y0, "data")
        draw_closest_points(overview_ax, f, result)
        overview_ax.annotate(point_label(result["point"]), (x0, y0), xytext=(0, -18),
                             textcoords="offset points", ha="center", fontsize=8)
        draw_closest_points(detail_ax, f, result, segments=False)
        nx = result["newton"]["x"]
        detail_ax.annotate(point_label(result["point"]), (nx, f(nx)), xytext=(6, 7),
                           textcoords="offset points", fontsize=8)
    closest_point_legend(overview_ax)
    closest_point_legend(detail_ax, segments=False)
    overview_ax.set_title("All five points: full geometry")
    detail_ax.set_title("Zoom: closest points on the curve")
    for ax in (overview_ax, detail_ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    overview_ax.set_xlim(-9, 7)
    overview_ax.set_ylim(-1.2, 13)
    detail_ax.set_xlim(-0.82, 0.82)
    detail_ax.set_ylim(4.8, 6.2)
    detail_ax.legend(loc="upper center")
    overview_ax.legend(loc="upper left")
    figure_title(fig, "Shortest distance for the five assigned points")
    save(fig, os.path.join(FIG, "p1_five_points.png"))


PROGRESS_POINT = (-4, 0)


def plot_progress(f, results):
    """Write four separate progress figures for the representative point (-4, 0)."""
    result = next(item for item in results if item["point"] == PROGRESS_POINT)
    plot_newton_geometry(f, result)
    plot_newton_convergence(result)
    plot_golden_brackets(result)
    plot_convergence_comparison(result)


def newton_iterates(result):
    history = result["newton"]["history"]
    return [history[0]["x"]] + [item["x_next"] for item in history]


def plot_newton_geometry(f, result):
    """Only n=0 and n=1 are visually distinct; later iterates coincide with x*."""
    point = result["point"]
    newton_x = newton_iterates(result)
    x_star = result["newton"]["x"]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    draw_curve(ax, f, (-4.6, 1.5), r"$y=x^2+5$")
    plot_points(ax, *point, "data", label=f"given point {point_label(point)}")
    ax.plot([point[0], newton_x[0]], [point[1], f(newton_x[0])], color=NEWTON,
            linestyle=":", linewidth=1.3)
    ax.plot([point[0], x_star], [point[1], f(x_star)], color=SOLUTION, linestyle="--",
            linewidth=1.6, label=f"shortest segment, d = {result['newton']['d']:.4f}")
    plot_points(ax, newton_x[0], f(newton_x[0]), "newton", label=r"Newton start $x^{(0)}$")
    ax.annotate("", xy=(newton_x[1], f(newton_x[1]) + 0.35),
                xytext=(newton_x[0], f(newton_x[0]) + 0.35),
                arrowprops={"arrowstyle": "->", "color": NEWTON, "linewidth": 1.4})
    ax.annotate(r"$x^{(0)}=0$", (newton_x[0], f(newton_x[0])), xytext=(8, -16),
                textcoords="offset points", fontsize=9, color=NEWTON)
    plot_points(ax, x_star, f(x_star), "solution", zorder=7,
                label=r"$x^{(1)}\ldots x^{(4)} \approx x^*$")
    ax.annotate(f"$x^* = {x_star:.5f}$", (x_star, f(x_star)), xytext=(-20, 34),
                textcoords="offset points", ha="center", fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": GUIDE, "linewidth": 0.8})
    note_box(ax, "$x^{(n+1)}=x^{(n)}-D'(x^{(n)})/D''(x^{(n)})$", 0.03, 0.97)
    ax.set_xlim(-4.6, 1.5)
    ax.set_ylim(-1.2, 13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Newton-Raphson geometry for point {point_label(point)}")
    ax.legend(loc="upper right")
    save(fig, os.path.join(FIG, "p1_newton_geometry.png"))


def tolerance_line(ax, x_text):
    ax.axhline(TOLERANCE, color=GUIDE, linewidth=1, linestyle=":")
    ax.text(x_text, TOLERANCE * 1.8, f"tolerance {TOLERANCE:g}", fontsize=8, color=INK2)


def plot_newton_convergence(result):
    """The iterates differ by orders of magnitude, so show them on a log scale."""
    history = result["newton"]["history"]
    iterations = [step["k"] for step in history]
    gradient = [abs(step["Dp"]) for step in history]
    step_size = [abs(step["x_next"] - step["x"]) for step in history]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.semilogy(iterations, gradient, color=NEWTON, marker=NEWTON_MARKER, label=r"$|D'(x^{(n)})|$")
    ax.semilogy(iterations, step_size, color=NEWTON_LIGHT, marker=NEWTON_MARKER, linestyle="--",
                label=r"step $|x^{(n+1)}-x^{(n)}|$")
    tolerance_line(ax, iterations[0])
    for k, value in zip(iterations, gradient):
        ax.annotate(f"{value:.1e}", (k, value), xytext=(7, 4),
                    textcoords="offset points", fontsize=8, color=NEWTON)
    for step in history:
        ax.annotate(f"x={step['x']:.7f}", (step["k"], 1e-13), ha="center", fontsize=8, color=INK2)
    ax.set_ylim(1e-14, 1e3)
    ax.set_xticks(iterations)
    ax.set_xticklabels([f"n={k}" for k in iterations])
    ax.set_xlim(iterations[0] - 0.4, iterations[-1] + 0.6)
    ax.set_xlabel("iteration n")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title(f"Newton-Raphson quadratic convergence for point {point_label(result['point'])}")
    ax.legend(loc="upper right")
    save(fig, os.path.join(FIG, "p1_newton_convergence.png"))


def plot_golden_brackets(result):
    """Each row is one bracket [a, b]; the kept sub-interval is highlighted."""
    history = result["golden"]["history"]
    golden_x = result["golden"]["x"]
    shown_history = history[:6]
    top = len(shown_history) - 1
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for iteration, step in enumerate(shown_history):
        level = top - iteration
        keep_left = step["keep"] == "left"
        kept = (step["a"], step["x2"]) if keep_left else (step["x1"], step["b"])
        better, worse = (step["x1"], step["x2"]) if keep_left else (step["x2"], step["x1"])
        ax.hlines(level, step["a"], step["b"], color=GRID, linewidth=7, zorder=2,
                  label="bracket [a, b]" if iteration == 0 else None)
        ax.hlines(level, *kept, color=OTHER_LIGHT, linewidth=7, zorder=3,
                  label="kept for next iteration" if iteration == 0 else None)
        plot_points(ax, better, level, "other", label="probe with smaller D" if iteration == 0 else None)
        ax.scatter(worse, level, color="white", marker=OTHER_MARKER, s=POINT_SIZE, zorder=5,
                   edgecolor=OTHER, linewidth=1.4, label="probe with larger D" if iteration == 0 else None)
    final_level = -1
    ax.axvline(golden_x, color=SOLUTION, linewidth=1, linestyle="--", zorder=1)
    plot_points(ax, golden_x, final_level, "solution", label=f"final x* = {golden_x:.5f}")
    note_box(ax, "$x_1=a+r(b-a),\\ x_2=b-r(b-a),\\ r=2-\\varphi$\nkeep the side of the smaller-D probe",
             0.98, 0.03, ha="right", va="bottom")
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(final_level - 0.7, top + 0.6)
    ax.set_yticks(range(final_level, top + 1))
    ax.set_yticklabels([f"n={history[-1]['k']} (final)"]
                       + [f"n={top - level}" for level in range(top + 1)])
    ax.set_xlabel("x")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Golden-section brackets for point {point_label(result['point'])}")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8))
    save(fig, os.path.join(FIG, "p1_golden_brackets.png"))


def plot_convergence_comparison(result):
    """Distance from the analytical minimiser per iteration, for both methods."""
    analytic_x = result["analytic"]["x"]
    newton_error = [(n, abs(x - analytic_x)) for n, x in enumerate(newton_iterates(result))]
    golden_width = [(step["k"], step["b"] - step["a"]) for step in result["golden"]["history"]]
    floor = 1e-16  # errors below machine precision are drawn at the floor
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.semilogy([n for n, _ in newton_error], [max(e, floor) for _, e in newton_error],
                color=NEWTON, marker=NEWTON_MARKER, label=r"Newton $|x^{(n)}-x^*|$")
    ax.semilogy([n for n, _ in golden_width], [w for _, w in golden_width],
                color=OTHER, marker=OTHER_MARKER, markersize=4, label="golden bracket width $b-a$")
    tolerance_line(ax, 6)
    ax.annotate(f"Newton: {result['newton']['iterations']} iterations",
                (newton_error[-1][0], floor), xytext=(12, 30), textcoords="offset points",
                fontsize=9, color=NEWTON,
                arrowprops={"arrowstyle": "-", "color": NEWTON, "linewidth": 0.8})
    ax.annotate(f"golden: {result['golden']['iterations']} iterations\n(width × 0.618 per iteration)",
                golden_width[-1], xytext=(-10, -60), textcoords="offset points",
                ha="right", fontsize=9, color=OTHER,
                arrowprops={"arrowstyle": "-", "color": OTHER, "linewidth": 0.8})
    ax.set_ylim(floor / 10, 20)
    ax.set_xlabel("iteration n")
    ax.set_ylabel("error / bracket width (log scale)")
    ax.set_title(f"Newton vs golden-section convergence for point {point_label(result['point'])}")
    ax.legend(loc="upper right")
    save(fig, os.path.join(FIG, "p1_convergence_comparison.png"))


def plot_non_polynomial(examples):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, example in zip(axes.flat, examples):
        draw_curve(ax, example["f"], example["plot_window"], example["equation"])
        plot_points(ax, *example["point"], "data")
        draw_closest_points(ax, example["f"], example)
        closest_point_legend(ax)
        ax.set_title(f"{example['name']}: {example['equation']}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_box_aspect(1)
        ax.legend(loc="best")
    figure_title(fig, "Two distance methods on four non-polynomial curves")
    save(fig, os.path.join(FIG, "p1_non_polynomial.png"))


def print_results(assigned, non_polynomial):
    """Print the closest point x* and distance d found by each method."""
    print("Assigned problem: y = x^2 + 5")
    print(f"{'point':>9} {'analytic x*':>12} {'Newton x*':>12} {'Newton d':>11} {'iter':>4}"
          f" {'golden x*':>12} {'golden d':>11} {'iter':>4}")
    for result in assigned:
        newton, golden = result["newton"], result["golden"]
        print(f"{point_label(result['point']):>9} {result['analytic']['x']:12.7f} {newton['x']:12.7f}"
              f" {newton['d']:11.7f} {newton['iterations']:4d} {golden['x']:12.7f} {golden['d']:11.7f}"
              f" {golden['iterations']:4d}")
    print("\nNon-polynomial curves")
    print(f"{'curve':>11} {'point':>9} {'Newton x*':>12} {'Newton d':>11} {'golden x*':>12} {'golden d':>11}")
    for example in non_polynomial:
        newton, golden = example["newton"], example["golden"]
        print(f"{example['name']:>11} {point_label(example['point']):>9} {newton['x']:12.7f}"
              f" {newton['d']:11.7f} {golden['x']:12.7f} {golden['d']:11.7f}")


def main():
    f, assigned = run_assigned_problem()
    non_polynomial = run_non_polynomial_examples()
    plot_five_points(f, assigned)
    plot_progress(f, assigned)
    plot_non_polynomial(non_polynomial)
    print_results(assigned, non_polynomial)
    print("\nWrote figures/part1/*.png")


if __name__ == "__main__":
    main()
