"""Figures for the README: held-out parity plots (GP) and the active-learning vs random curve.
    python3 plots.py   ->  results/parity.png, results/active_boundary_hard.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogate import OUT, TARGETS, fit_gp, load, split

LABELS = {"T_surface_max_C": "Peak surface T (°C)", "T_interface_max_C": "Peak interface T (°C)",
          "vm_W_p99_MPa": "W von Mises, p99 (MPa)", "vm_Cu_p99_MPa": "Cu von Mises, p99 (MPa)"}


def parity():
    X, Y = load()
    Xtr, Ytr, Xte, Yte = split(X, Y)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for j, (name, ax) in enumerate(zip(TARGETS, axes)):
        mean, std = fit_gp(Xtr, Ytr[:, j])(Xte, return_std=True)
        ax.errorbar(Yte[:, j], mean, yerr=1.96 * std, fmt="o", ms=3, lw=0.6, capsize=0, color="#2b6cb0", alpha=0.8)
        lo, hi = Yte[:, j].min(), Yte[:, j].max()
        ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8)
        ax.set_title(LABELS[name], fontsize=10)
        ax.set_xlabel("finite element")
        ax.set_ylabel("GP surrogate (± 95 %)") if j == 0 else None
    fig.suptitle(f"Held-out designs (n = {len(Xte)}), GP trained on {len(Xtr)} FE solves", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "parity.png", dpi=150)


def active_curve():
    d = json.loads((OUT / "active_learning.json").read_text())
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for mode, colour in (("random", "#555"), ("active", "#c05621")):
        ax.plot(d[mode]["n_train"], d[mode]["mean_test_MAE_MPa"], "o-", ms=3, color=colour,
                label="highest GP uncertainty" if mode == "active" else "random")
    ax.set_yscale("log")
    ax.set_xlabel("FE solves in the training set")
    ax.set_ylabel("held-out MAE, Cu p99 stress (MPa)")
    ax.legend(frameon=False)
    ax.set_title("Picking the most uncertain design did not beat random", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "active_learning.png", dpi=150)


def boundary_curve():
    d = json.loads((OUT / "active_boundary_hard.json").read_text())
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    style = {"random": ("#555", "random"), "uncertainty": ("#2b6cb0", "most uncertain point"),
             "boundary": ("#c05621", "nearest the 450 °C limit (straddle)")}
    for rule, (colour, label) in style.items():
        n = np.array(d[rule]["n_train"]); m = np.array(d[rule]["misclassified_mean"]); s = np.array(d[rule]["misclassified_sd"])
        ax.plot(n, m, "o-", ms=3, color=colour, label=label)
        ax.fill_between(n, m - s, m + s, color=colour, alpha=0.12)
    ax.set_xlabel("FE solves used to train the GP")
    ax.set_ylabel(f"wrong allowed/not-allowed calls (of {d['n_test']})")
    ax.set_title("Hard test set: 50 designs near the copper limit, 10 seeds", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "active_boundary_hard.png", dpi=150)


if __name__ == "__main__":
    parity()
    boundary_curve()
