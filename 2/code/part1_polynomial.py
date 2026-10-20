"""
Part 1: linear basis function regression with polynomial basis functions.

    python part1_polynomial.py

(a) polynomial_regression   train/test RMS error for degree 1 to 7, all 13 features
(b) degree_one_weights      the 14 weights of the degree-1 model (6th feature RM, 8th feature DIS)
(c) polynomial_regression_1d_vis   single feature Xn(:,2) = ZN, fitted curves
(d) polynomial_regression_reg      degree 8 on ZN, L2 regularization, 10-fold CV

Figures go to ../figures/part1 and the results are printed to the terminal.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from data_utils import (FEATURE_1D, FEATURE_DESCRIPTIONS, FEATURE_NAMES, N_TRAIN, fit_normalizer, load_housing,
                        load_normalized, normalize)
from regression_core import cross_validate, fit, polynomial_design, predict, rms_error

FIG_DIR = r"C:\Users\dylan\OneDrive\Documents\4A\SYDE 572\Assignments\Assignment 2\figures\part1"

LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
CV_FOLDS = 10
CV_SEED = 0            # seed for the random assignment of towns to folds


# ---------------------------------------------------------------- (a)

def polynomial_regression(degrees=range(1, 8)):
    """
    Fit a degree-M polynomial basis regression (single-feature monomials, no cross-terms,
    no regularization) on the first 100 towns, for each M in `degrees`.
    Returns one row per degree with the training and test RMS error.
    """
    Z_tr, y_tr, Z_te, y_te, _, _ = load_normalized()
    rows = []
    for M in degrees:
        Phi_tr = polynomial_design(Z_tr, M)
        Phi_te = polynomial_design(Z_te, M)
        w = fit(Phi_tr, y_tr)                       # lambda = 0
        rows.append({
            "degree": M,
            "n_weights": Phi_tr.shape[1],
            "rank": int(np.linalg.matrix_rank(Phi_tr)),
            "train_rms": rms_error(predict(Phi_tr, w), y_tr),
            "test_rms": rms_error(predict(Phi_te, w), y_te),
        })
    return rows


def _fmt(v):
    return f"{v:.2f}" if v < 1000 else f"{v:.2e}"


def plot_a(rows, path):
    M = [r["degree"] for r in rows]
    tr = [r["train_rms"] for r in rows]
    te = [r["test_rms"] for r in rows]
    fig, ax = plt.subplots(figsize=ps.SINGLE_FIGSIZE)
    ax.plot(M, tr, "-o", color=ps.TRAIN, label="Training error (first 100 towns)")
    ax.plot(M, te, "-s", color=ps.TEST, label="Test error (remaining 406 towns)")
    ax.set_yscale("log")    # the test error spans many orders of magnitude
    for m, t in zip(M, te):
        ax.annotate(_fmt(t), (m, t), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=ps.TEST)
    for m, t in zip(M, tr):
        ax.annotate(_fmt(t), (m, t), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=ps.TRAIN)
    ax.set_xlabel("Polynomial degree M")
    ax.set_ylabel("RMS error ($1000s, log scale)")
    ax.set_title("(a) Polynomial basis, all 13 features: RMS error vs degree")
    ax.set_xticks(M)
    ax.legend(loc="upper left")
    ps.save(fig, path)


# ---------------------------------------------------------------- (b)

def degree_one_weights(n_train=100):
    """
    Weights of the degree-1 model y_hat = w_0 + sum_j w_j z_j, where z_j = (x_j - mean_j)/SD_j.

    Undoing the normalization, y_hat = w_0 + sum_j w_j (x_j - mean_j)/SD_j, so the weight per
    raw unit of feature j is w_j/SD_j and the raw intercept is w_0 - sum_j w_j mean_j/SD_j.
    Also returns each feature's correlation with MEDV in the training towns (for the comments).
    With n_train=506 the same model is fitted to all towns, for comparison only.
    """
    X, y = load_housing()
    X_tr, y_tr = X[:n_train], y[:n_train]
    mean, std = fit_normalizer(X_tr)
    w = fit(polynomial_design(normalize(X_tr, mean, std), 1), y_tr)
    rows = [{"index": None, "name": "bias", "mean": None, "std": None, "w": w[0],
             "raw": w[0] - np.sum(w[1:] * mean / std), "corr": None}]
    for j, name in enumerate(FEATURE_NAMES):
        corr = np.corrcoef(X_tr[:, j], y_tr)[0, 1] if X_tr[:, j].std() > 0 else None
        rows.append({"index": j, "name": name, "mean": mean[j], "std": std[j], "w": w[j + 1],
                     "raw": w[j + 1] / std[j], "corr": corr})
    return rows


# ---------------------------------------------------------------- (c)

def polynomial_regression_1d_vis(degrees=range(1, 11), feature=FEATURE_1D):
    """
    Polynomial regression on one feature only (default Xn(:,2) = the 2nd feature, ZN),
    normalized with the training mean and SD, no regularization. Returns the data and, for
    each degree, the training/test RMS error and the learned curve on a grid of raw x values.
    """
    X, y = load_housing()
    x_tr, y_tr = X[:N_TRAIN, feature], y[:N_TRAIN]
    x_te, y_te = X[N_TRAIN:, feature], y[N_TRAIN:]
    mean, std = fit_normalizer(x_tr[:, None])
    mean, std = float(mean[0]), float(std[0])
    z = lambda x: (x - mean) / std
    x_grid = np.linspace(min(x_tr.min(), x_te.min()), max(x_tr.max(), x_te.max()), 600)
    fits = []
    for M in degrees:
        w = fit(polynomial_design(z(x_tr), M), y_tr)
        fits.append({"degree": M,
                     "train_rms": rms_error(predict(polynomial_design(z(x_tr), M), w), y_tr),
                     "test_rms": rms_error(predict(polynomial_design(z(x_te), M), w), y_te),
                     "curve": predict(polynomial_design(z(x_grid), M), w)})
    return {"feature": feature, "mean": mean, "std": std, "x_tr": x_tr, "y_tr": y_tr,
            "x_te": x_te, "y_te": y_te, "x_grid": x_grid, "fits": fits}


def plot_c(vis, M, path, note):
    row = next(f for f in vis["fits"] if f["degree"] == M)
    name = FEATURE_NAMES[vis["feature"]]
    fig, ax = plt.subplots(figsize=ps.SINGLE_FIGSIZE)
    ax.scatter(vis["x_te"], vis["y_te"], s=26, color=ps.TEST, alpha=0.55, edgecolor="none",
               label="Test data", zorder=2)
    ax.scatter(vis["x_tr"], vis["y_tr"], s=26, color=ps.TRAIN, edgecolor="white", linewidth=0.6,
               label="Training data", zorder=3)
    ax.plot(vis["x_grid"], row["curve"], color=ps.FIT, lw=2.2,
            label=f"Learned polynomial (M = {M})", zorder=4)
    # MEDV is between 5 and 50; the range grows when the curve leaves it, so the whole curve
    # shows, and there is room at the top for the note box
    lo, hi = min(0, row["curve"].min()), max(50, row["curve"].max())
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.35 * (hi - lo))
    ax.set_xlim(vis["x_grid"].min() - 0.5, vis["x_grid"].max() + 0.5)
    ax.set_xlabel(f"{name}: {FEATURE_DESCRIPTIONS[vis['feature']]} (%)")
    ax.set_ylabel("MEDV, median home value ($1000s)")
    ax.set_title(f"(c) Degree {M} polynomial in {name} only")
    ps.note_box(ax, f"Train RMS = {row['train_rms']:.2f}\nTest RMS = {_fmt(row['test_rms'])}\n{note}\n"
                    f"Fitted on z = (x - {vis['mean']:.2f}) / {vis['std']:.2f}",
                0.99, 0.98, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    ps.save(fig, path)


# ---------------------------------------------------------------- (d)

def polynomial_regression_reg(degree=8, lambdas=LAMBDAS, feature=FEATURE_1D, k=CV_FOLDS,
                              seed=CV_SEED):
    """
    L2-regularized degree-8 polynomial regression on one feature (ZN), first 100 towns.
    For each lambda, k-fold cross-validation gives the average validation RMS error. The
    folds are the same for every lambda, and the feature is renormalized inside each fold using
    that fold's training part. Every weight, including the bias, is penalized.
    """
    X, y = load_housing()
    x_tr, y_tr = X[:N_TRAIN, [feature]], y[:N_TRAIN]

    def make_design(a, b):
        return polynomial_design(a, degree), polynomial_design(b, degree)

    rows = []
    for lam in lambdas:
        errs = cross_validate(x_tr, y_tr, make_design, lam, k=k, seed=seed,
                              normalizer=fit_normalizer)
        rows.append({"lambda": lam, "cv_rms": errs.mean(), "cv_std": errs.std()})
    best = min(rows, key=lambda r: r["cv_rms"])
    return {"rows": rows, "best_lambda": best["lambda"], "k": k}


def plot_cv(result, path, title):
    """Average validation RMS error vs lambda (semilogx), with lambda = 0 as a dashed baseline."""
    rows = result["rows"]
    pos = [r for r in rows if r["lambda"] > 0]
    base = next(r for r in rows if r["lambda"] == 0)
    best = min(rows, key=lambda r: r["cv_rms"])
    lam = [r["lambda"] for r in pos]
    cv = [r["cv_rms"] for r in pos]
    fig, ax = plt.subplots(figsize=ps.SINGLE_FIGSIZE)
    ax.semilogx(lam, cv, "-o", color=ps.VALID, label=f"{result['k']}-fold CV error, λ > 0")
    if base["cv_rms"] <= 2 * max(cv):
        ax.axhline(base["cv_rms"], color=ps.BASELINE, ls="--", lw=1.6,
                   label=f"Unregularized baseline, λ = 0  ({base['cv_rms']:.2f})")
    else:
        # The lambda = 0 error is far above the rest, so drawing it would flatten the curve.
        # The handout allows reporting it separately instead.
        ps.note_box(ax, f"Unregularized (λ = 0) CV error = {base['cv_rms']:.2f}\n"
                        "(off the scale of this plot)", 0.99, 0.98, ha="right")
        ax.set_ylim(top=max(cv) * 1.3)   # headroom so the note does not cover the curve
    for l, c in zip(lam, cv):
        ax.annotate(f"{c:.2f}", (l, c), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=ps.INK2)
    if best["lambda"] > 0:
        ax.scatter([best["lambda"]], [best["cv_rms"]], marker="*", s=ps.STAR_SIZE * 2,
                   color=ps.CHOSEN, edgecolor="white", zorder=6,
                   label=f"Chosen λ = {best['lambda']:g}  ({best['cv_rms']:.2f})")
    ax.set_xlabel("Regularizing constant λ (log scale)")
    ax.set_ylabel("Average validation RMS error ($1000s)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ps.save(fig, path)


# ---------------------------------------------------------------- main

def main():
    ps.apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    lines = []

    a = polynomial_regression()
    plot_a(a, os.path.join(FIG_DIR, "p1a_rms_vs_degree.png"))
    lines.append("(a) Polynomial regression, all 13 features, no regularization")
    lines.append("    degree  weights  rank   train RMS    test RMS")
    for r in a:
        lines.append(f"    {r['degree']:6d} {r['n_weights']:8d} {r['rank']:5d} "
                     f"{r['train_rms']:11.4f} {r['test_rms']:11.4g}")

    b = degree_one_weights()
    b_all = degree_one_weights(n_train=506)
    lines.append("")
    lines.append("(b) Degree-1 weights, trained on the first 100 towns")
    lines.append("    feature  weight     train mean   train SD   w (normalized)   w/SD (raw unit)"
                 "   corr with MEDV   | w, all 506 towns")
    for r, ra in zip(b, b_all):
        idx = "-" if r["index"] is None else str(r["index"] + 1)   # feature number, from 1
        mean = "-" if r["mean"] is None else f"{r['mean']:.3f}"
        std = "-" if r["std"] is None else f"{r['std']:.3f}"
        corr = "-" if r["corr"] is None else f"{r['corr']:+.3f}"
        lines.append(f"    {idx:>7s}  {r['name']:8s} {mean:>11s} {std:>10s} {r['w']:+16.4f} "
                     f"{r['raw']:+17.5f} {corr:>16s}   | {ra['w']:+.4f}")
    lines.append("    (bias row: w/SD column is the raw-unit intercept)")

    c = polynomial_regression_1d_vis()
    notes = {1: "Low order: an almost flat line;\nZN alone explains little of MEDV",
             4: "Moderate order: still smooth,\nonly slightly lower training error",
             10: "High order: swings wildly in the gaps\nbetween the few distinct ZN values"}
    for M, note in notes.items():
        plot_c(c, M, os.path.join(FIG_DIR, f"p1c_degree_{M}.png"), note)
    lines.append("")
    lines.append(f"(c) {FEATURE_NAMES[c['feature']]} only (mean {c['mean']:.3f}, SD {c['std']:.3f}), "
                 "no regularization")
    lines.append("    degree   train RMS    test RMS")
    for f in c["fits"]:
        lines.append(f"    {f['degree']:6d} {f['train_rms']:11.4f} {f['test_rms']:11.4g}")

    d = polynomial_regression_reg()
    plot_cv(d, os.path.join(FIG_DIR, "p1d_cv_vs_lambda.png"),
            f"(d) Degree 8 polynomial in {FEATURE_NAMES[FEATURE_1D]}: 10-fold CV error vs λ")
    lines.append("")
    lines.append(f"(d) Degree 8 polynomial in {FEATURE_NAMES[FEATURE_1D]}, L2 regularization (all weights penalized), "
                 "10-fold CV on the first 100 towns")
    lines.append("     lambda    CV RMS   fold SD")
    for r in d["rows"]:
        lines.append(f"    {r['lambda']:7g} {r['cv_rms']:9.4f} {r['cv_std']:9.4f}")
    lines.append(f"    chosen lambda = {d['best_lambda']:g}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
