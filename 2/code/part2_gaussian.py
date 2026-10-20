"""
Part 2: linear basis function regression with Gaussian basis functions.

    python part2_gaussian.py

The Gaussian width S and the random SEED are set at the top of this file.

(e) gaussian_regression       train/test RMS error for K = 5, 15, ..., 95 basis functions
(f) gaussian_regression_reg   K = 90, L2 regularization, 10-fold CV over lambda

Basis: phi_k(x) = exp(a ||x - c_k||^2), a = -1/(2 s^2), on all 13 normalized features, plus a
bias. The centers c_k are randomly chosen training points.
Figures go to ../figures/part2 and the results are printed to the terminal.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from data_utils import N_TRAIN, fit_normalizer, load_housing, load_normalized
from part1_polynomial import CV_FOLDS, LAMBDAS, _fmt, plot_cv
from regression_core import choose_centers, cross_validate, fit, gaussian_design, predict, rms_error

FIG_DIR = r"C:\Users\dylan\OneDrive\Documents\4A\SYDE 572\Assignments\Assignment 2\figures\part2"

S = 2.0                # Gaussian width s
SEED = 0               # seed for the random permutation that picks the centers
K_VALUES = list(range(5, 96, 10))
K_REG = 90             # number of basis functions in (f)


# ---------------------------------------------------------------- (e)

def gaussian_regression(s=S, seed=SEED, K_values=K_VALUES):
    """
    Unregularized Gaussian basis regression on the first 100 towns, for each K.
    One random permutation of the 100 training points is drawn, and its first K points are
    the centers, so the center sets are nested (the K = 15 centers include the K = 5 ones).
    Returns one row per K with the training and test RMS error.
    """
    Z_tr, y_tr, Z_te, y_te, _, _ = load_normalized()
    perm = np.random.default_rng(seed).permutation(len(y_tr))
    rows = []
    for K in K_values:
        centers = Z_tr[perm[:K]]
        Phi_tr = gaussian_design(Z_tr, centers, s)
        Phi_te = gaussian_design(Z_te, centers, s)
        w = fit(Phi_tr, y_tr)                       # lambda = 0
        rows.append({"K": K, "n_weights": Phi_tr.shape[1],
                     "train_rms": rms_error(predict(Phi_tr, w), y_tr),
                     "test_rms": rms_error(predict(Phi_te, w), y_te)})
    # Reference: test error of always predicting the training mean (a bias-only model)
    mean_rms = rms_error(np.full_like(y_te, y_tr.mean()), y_te)
    return {"rows": rows, "s": s, "mean_rms": mean_rms}


def plot_e(result, path):
    rows = result["rows"]
    K = [r["K"] for r in rows]
    tr = [r["train_rms"] for r in rows]
    te = [r["test_rms"] for r in rows]
    fig, ax = plt.subplots(figsize=ps.SINGLE_FIGSIZE)
    ax.plot(K, tr, "-o", color=ps.TRAIN, label="Training error (first 100 towns)")
    ax.plot(K, te, "-s", color=ps.TEST, label="Test error (remaining 406 towns)")
    ax.axhline(result["mean_rms"], color=ps.BASELINE, ls="--", lw=1.4,
               label=f"Test error of always predicting the training mean ({result['mean_rms']:.2f})")
    log = max(te) / min(te) > 30     # switch to a log axis only if the test error blows up
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.7, max(te + [result["mean_rms"]]) * 1.15)
    for k, t in zip(K, te):
        ax.annotate(_fmt(t), (k, t), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=ps.TEST)
    for k, t in zip(K, tr):
        ax.annotate(_fmt(t), (k, t), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=ps.TRAIN)
    ax.set_xlabel("Number of Gaussian basis functions K (plus a bias)")
    ax.set_ylabel("RMS error ($1000s" + (", log scale)" if log else ")"))
    ax.set_title(f"(e) Gaussian basis, all 13 features, s = {result['s']:g}: RMS error vs K")
    ax.set_xticks(K)
    ax.legend(loc="center right")
    ps.save(fig, path)


# ---------------------------------------------------------------- (f)

def gaussian_regression_reg(s=S, seed=SEED, K=K_REG, lambdas=LAMBDAS, k=CV_FOLDS):
    """
    L2-regularized Gaussian basis regression with K = 90 basis functions, using all 13
    features of the first 100 towns. For each lambda, k-fold cross-validation gives the
    average validation RMS error. The folds are the same for every lambda. In each fold the
    features are normalized with that fold's training part, and the K centers are drawn at
    random from it (the same centers for every lambda). Every weight, including the bias, is
    penalized.
    """
    X, y = load_housing()
    X_tr, y_tr = X[:N_TRAIN], y[:N_TRAIN]
    rows = []
    for lam in lambdas:
        rng = np.random.default_rng(seed)       # restart so every lambda gets the same centers

        def make_design(a, b):
            centers = choose_centers(a, K, rng)
            return gaussian_design(a, centers, s), gaussian_design(b, centers, s)

        errs = cross_validate(X_tr, y_tr, make_design, lam, k=k, seed=seed,
                              normalizer=fit_normalizer)
        rows.append({"lambda": lam, "cv_rms": errs.mean(), "cv_std": errs.std()})
    best = min(rows, key=lambda r: r["cv_rms"])
    return {"rows": rows, "best_lambda": best["lambda"], "k": k}


# ---------------------------------------------------------------- main

def main():
    ps.apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    lines = []

    e = gaussian_regression()
    plot_e(e, os.path.join(FIG_DIR, "p2e_rms_vs_K.png"))
    lines.append(f"(e) Gaussian basis, all 13 features, s = {S:g}, seed = {SEED}, "
                 "no regularization")
    lines.append("      K  weights   train RMS    test RMS")
    for r in e["rows"]:
        lines.append(f"    {r['K']:3d} {r['n_weights']:8d} {r['train_rms']:11.4f} "
                     f"{r['test_rms']:11.4g}")
    lines.append(f"    test RMS of always predicting the training mean = {e['mean_rms']:.4f}")

    f = gaussian_regression_reg()
    plot_cv(f, os.path.join(FIG_DIR, "p2f_cv_vs_lambda.png"),
            f"(f) Gaussian basis, K = {K_REG}, s = {S:g}: 10-fold CV error vs λ")
    lines.append("")
    lines.append(f"(f) Gaussian basis, K = {K_REG}, s = {S:g}, all 13 features, L2 regularization "
                 "(all weights penalized), 10-fold CV on the first 100 towns")
    lines.append("     lambda    CV RMS   fold SD")
    for r in f["rows"]:
        lines.append(f"    {r['lambda']:7g} {r['cv_rms']:9.4f} {r['cv_std']:9.4f}")
    lines.append(f"    chosen lambda = {f['best_lambda']:g}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
