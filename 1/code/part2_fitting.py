"""
SYDE 572 Assignment 1, Part 2.

Run from this directory with::

    python part2_fitting.py

Fits a line y = c0 + c1 x and a parabola y = B x^2 + C x + D to the four
assignment points by minimising the MSE, using

* the analytical solution (normal equations), and
* Newton-Raphson one parameter at a time (coordinate Newton), as the
  assignment asks, with one full multivariate Newton step for comparison.

It prints the fitted parameters and MSE and writes these figures to ../figures/part2:

* p2_line_iterations.png: the line after each coordinate Newton iteration.
* p2_line_contour.png: the path of both Newton variants on the MSE(c0, c1) surface.
* p2_parabola_iterations.png: the parabola after each coordinate Newton iteration.
* p2_convergence.png: MSE - MSE* per iteration for both models.
* p2_final_fits.png: the best line and parabola with their residuals.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from fitting_methods import (MODELS, analytic_fit, coordinate_newton, full_newton, mse,
                             normal_equations, predict)
from plot_style import (GUIDE, INK2, ITERATION_RAMP, NEWTON, NEWTON_MARKER, OTHER, SINGLE_FIGSIZE,
                        SOLUTION, apply_style, figure_title, note_box, plot_points, save)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures", "part2")
os.makedirs(FIG, exist_ok=True)
apply_style()

X = np.array([0, 2, 1, 3], float)
Y = np.array([0.5, 3.5, 1.5, 7.5])
TOLERANCE = 1e-7
SHOWN_ITERATIONS = [0, 1, 2, 3, 5, 10]
X_PLOT = np.linspace(-0.3, 3.3, 300)
ROUND_OFF = 1e-14  # MSE gaps below this are floating-point noise


def run_fits():
    """Fit every model with all three solvers."""
    fits = {}
    for model in MODELS:
        analytic = analytic_fit(model, X, Y)
        newton, history = coordinate_newton(model, X, Y, tolerance=TOLERANCE)
        full = full_newton(model, X, Y)
        fits[model] = {
            "analytic": analytic,
            "analytic_mse": mse(model, analytic, X, Y),
            "newton": newton,
            "newton_mse": mse(model, newton, X, Y),
            "iterations": history[-1]["iteration"],
            "history": history,
            "full_newton": full,
            "full_newton_mse": mse(model, full, X, Y),
        }
    return fits


def end_of_iteration(history, iteration):
    """History entry after the last parameter update of `iteration`."""
    return [entry for entry in history if entry["iteration"] <= iteration][-1]


def update_rules(model):
    """
    Coordinate Newton update for each parameter, as in the hand solution.
    The gradient is g = H theta - b with H = (2/N) A^T A and b = (2/N) A^T y,
    so the step theta_j - g_j / H_jj simplifies to
    theta_j = (b_j - sum_{k != j} H_jk theta_k) / H_jj.
    """
    M, r = normal_equations(model, X, Y)
    H, b = (2 / len(X)) * M, (2 / len(X)) * r
    names = MODELS[model]["params"]
    rules = []
    for j in MODELS[model]["order"]:
        others = "".join(f" − {H[j, k]:g}{names[k]}" for k in range(len(names)) if k != j)
        rules.append(f"{names[j]} ← ({b[j]:g}{others}) / {H[j, j]:g}")
    return rules


def params_text(model, theta):
    return ", ".join(f"{name} = {value:.4f}" for name, value in zip(MODELS[model]["params"], theta))


def plot_iterations(model, fit):
    """Fitted curve after selected coordinate Newton iterations (light to dark blue), then the solution."""
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for iteration, colour in zip(SHOWN_ITERATIONS, ITERATION_RAMP):
        entry = end_of_iteration(fit["history"], iteration)
        ax.plot(X_PLOT, predict(model, entry["theta"], X_PLOT), color=colour, linewidth=1.6,
                label=f"n = {iteration}: MSE = {entry['mse']:.4f}")
    ax.plot(X_PLOT, predict(model, fit["newton"], X_PLOT), color=SOLUTION, linewidth=2.2,
            label=f"n = {fit['iterations']} (converged): MSE = {fit['newton_mse']:.6f}")
    plot_points(ax, X, Y, "data", label="data")
    note_box(ax, "update each iteration, in order:\n" + "\n".join(update_rules(model)),
             0.97, 0.04, ha="right", va="bottom")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-1, 9)
    ax.set_title(f"{MODELS[model]['label']} after each coordinate Newton iteration")
    ax.legend(loc="upper left")
    save(fig, os.path.join(FIG, f"p2_{model}_iterations.png"))


def mse_grid(c0_range, c1_range, points=300):
    """MSE of the line over a grid of (c0, c1) values."""
    C0, C1 = np.meshgrid(np.linspace(*c0_range, points), np.linspace(*c1_range, points))
    residuals = C0[..., None] + C1[..., None] * X - Y
    return C0, C1, np.mean(residuals ** 2, axis=-1)


def plot_line_contour(fit):
    """Coordinate Newton's staircase and the single full Newton step on MSE(c0, c1)."""
    best = fit["analytic"]
    best_mse = fit["analytic_mse"]
    path = np.array([entry["theta"] for entry in fit["history"]])
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    C0, C1, Z = mse_grid((-1.2, 1.2), (-0.3, 3.0))
    levels = best_mse + np.geomspace(1e-3, Z.max() - best_mse, 18)
    ax.contour(C0, C1, Z, levels=levels, colors=GUIDE, linewidths=0.8)
    ax.plot(path[:, 0], path[:, 1], color=NEWTON, linewidth=1.6, marker=NEWTON_MARKER, markersize=3.5,
            label="coordinate Newton (one parameter at a time)")
    ax.plot([0, fit["full_newton"][0]], [0, fit["full_newton"][1]], color=OTHER, linewidth=1.8,
            linestyle="--", label="full multivariate Newton (one step)")
    plot_points(ax, *best, "solution", zorder=6,
                label=f"minimum ({best[0]:.1f}, {best[1]:.1f}), MSE = {best_mse:.3f}")
    ax.annotate("start (0, 0)", (0, 0), textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 3.0)
    ax.set_xlabel("intercept c₀")
    ax.set_ylabel("slope c₁")
    ax.set_title("Newton paths on the MSE(c₀, c₁) surface")
    # Framed legend so it stays readable over the contour lines.
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GUIDE)

    # Zoom on the staircase near the minimum, with its own finer grid.
    zoom = ax.inset_axes([0.07, 0.06, 0.38, 0.34])
    zoom_c0, zoom_c1 = (-0.22, 0.01), (2.19, 2.32)
    C0, C1, Z = mse_grid(zoom_c0, zoom_c1)
    zoom.contour(C0, C1, Z, levels=best_mse + np.geomspace(1e-5, 2e-2, 12), colors=GUIDE, linewidths=0.6)
    zoom.plot(path[:, 0], path[:, 1], color=NEWTON, linewidth=1.4, marker=NEWTON_MARKER, markersize=3)
    plot_points(zoom, *best, "solution", zorder=6)
    zoom.set_xlim(*zoom_c0)
    zoom.set_ylim(*zoom_c1)
    zoom.set_title("zoom: staircase toward the minimum", fontsize=9, loc="left")
    zoom.tick_params(labelsize=8)
    zoom.set_facecolor("white")
    ax.indicate_inset_zoom(zoom, edgecolor=INK2)
    save(fig, os.path.join(FIG, "p2_line_contour.png"))


def plot_convergence(fits):
    """
    MSE - MSE* after each iteration, for both models. Both curves are coordinate
    Newton, so both are blue; the line style tells the models apart. No tolerance
    line is drawn: the 1e-7 tolerance applies to the parameter change, and near
    the minimum the MSE gap is quadratic in the parameter error (about 1e-13).
    """
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for model, linestyle in zip(MODELS, ["-", "--"]):
        fit = fits[model]
        iterations = np.arange(fit["iterations"] + 1)
        gaps = np.array([end_of_iteration(fit["history"], n)["mse"] for n in iterations]) - fit["analytic_mse"]
        shown = gaps > ROUND_OFF
        ax.semilogy(iterations[shown], gaps[shown], color=NEWTON, linestyle=linestyle,
                    label=f"{MODELS[model]['label']}: {fit['iterations']} iterations")
    ax.set_xlabel("iteration n (every parameter updated once)")
    ax.set_ylabel("MSE − MSE* (log scale)")
    ax.set_title("Coordinate Newton convergence")
    ax.legend(loc="upper right")
    save(fig, os.path.join(FIG, "p2_convergence.png"))


def plot_final_fits(fits):
    """Best line and parabola with their residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        theta = fits[model]["analytic"]
        ax.plot(X_PLOT, predict(model, theta, X_PLOT), color=SOLUTION, linewidth=2.2, label="best fit")
        for xi, yi, fitted in zip(X, Y, predict(model, theta, X)):
            ax.plot([xi, xi], [yi, fitted], color=GUIDE, linewidth=1.4, linestyle=":")
        ax.plot([], [], color=GUIDE, linewidth=1.4, linestyle=":", label="residual")
        plot_points(ax, X, Y, "data", label="data")
        note_box(ax, params_text(model, theta).replace(", ", "\n")
                 + f"\nMSE = {fits[model]['analytic_mse']:.4f}", 0.04, 0.96)
        ax.set_title(MODELS[model]["label"])
        ax.set_xlabel("x")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("y")
    figure_title(fig, "Best-fit line and parabola")
    save(fig, os.path.join(FIG, "p2_final_fits.png"))


def params_ascii(model, theta):
    """Parameter values with ASCII names, so the Windows console can print them."""
    names = [name.replace("₀", "0").replace("₁", "1") for name in MODELS[model]["params"]]
    return ", ".join(f"{name} = {value:.7f}" for name, value in zip(names, theta))


def print_results(fits):
    """Print the parameters and MSE found by each solver."""
    for model, fit in fits.items():
        print(f"{model}:")
        print(f"  analytic           {params_ascii(model, fit['analytic'])}  MSE = {fit['analytic_mse']:.8f}")
        print(f"  coordinate Newton  {params_ascii(model, fit['newton'])}  MSE = {fit['newton_mse']:.8f}"
              f"  ({fit['iterations']} iterations)")
        print(f"  full Newton        {params_ascii(model, fit['full_newton'])}  MSE = {fit['full_newton_mse']:.8f}"
              f"  (1 step)")


def main():
    fits = run_fits()
    for model in MODELS:
        plot_iterations(model, fits[model])
    plot_line_contour(fits["line"])
    plot_convergence(fits)
    plot_final_fits(fits)
    print_results(fits)
    print("\nWrote figures/part2/*.png")


if __name__ == "__main__":
    main()
