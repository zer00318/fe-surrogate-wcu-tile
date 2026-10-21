"""Surrogates for the W/Cu tile model: data generation, held-out scoring, uncertainty calibration, active learning.

    python3 surrogate.py generate   # 300 Latin-hypercube FE solves -> data/fe_samples.csv
    python3 surrogate.py evaluate   # GP and MLP vs a 1-D physics baseline on a held-out set -> results/
    python3 surrogate.py active     # GP active learning vs random sampling -> results/

Targets are converged quantities only: peak surface and interface temperature, and area-weighted p99 von Mises
stress in W and in Cu (the raw maximum is singular at the free-edge corners and grows with mesh refinement).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fe_model import Design, analytical_1d_surface_T, solve_design

HERE = Path(__file__).parent
DATA = HERE / "data" / "fe_samples.csv"
OUT = HERE / "results"
INPUTS = ["q_MW", "t_w_mm", "t_cu_mm", "h_kW", "r_mm"]
TARGETS = ["T_surface_max_C", "T_interface_max_C", "vm_W_p99_MPa", "vm_Cu_p99_MPa"]
N = 300
N_TEST = 60


def designs(n: int, seed: int = 0) -> list[Design]:
    u = qmc.LatinHypercube(d=5, seed=seed).random(n)
    out = []
    for q, tw, tcu, h, r in u:
        t_cu = 8 + 6 * tcu
        r_max = min(4.0, t_cu / 2 - 1.5)          # keep at least 1.5 mm of copper above and below the channel
        out.append(Design(round(5 + 15 * q, 3), round(2 + 6 * tw, 3), round(t_cu, 3), round(20 + 130 * h, 3),
                          round(2 + (r_max - 2) * r, 3)))
    return out


def generate() -> None:
    DATA.parent.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for i, d in enumerate(designs(N)):
        rows.append(solve_design(d, h_size=0.4e-3))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{N} solves, {time.time() - t0:.0f} s", flush=True)
    with DATA.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load():
    with DATA.open() as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in INPUTS] for r in rows])
    Y = np.array([[float(r[k]) for k in TARGETS] for r in rows])
    return X, Y


def split(X, Y, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    te, tr = idx[:N_TEST], idx[N_TEST:]
    return X[tr], Y[tr], X[te], Y[te]


def gp_model():
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


def fit_gp(X, y):
    sx = StandardScaler().fit(X)
    gp = gp_model().fit(sx.transform(X), y)
    return lambda Z, return_std=False: gp.predict(sx.transform(Z), return_std=return_std)


def fit_mlp(X, y):
    m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, alpha=1e-4,
                                                    learning_rate_init=3e-3, random_state=0))
    sy = StandardScaler().fit(y[:, None])
    m.fit(X, sy.transform(y[:, None]).ravel())
    return lambda Z: sy.inverse_transform(m.predict(Z)[:, None]).ravel()


def scores(y, p):
    err = p - y
    return {"R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "MAE": float(np.mean(np.abs(err))), "max_abs_err": float(np.max(np.abs(err))),
            "MAPE_pct": float(100 * np.mean(np.abs(err / y)))}


def evaluate() -> dict:
    X, Y = load()
    Xtr, Ytr, Xte, Yte = split(X, Y)
    res = {"n_train": len(Xtr), "n_test": len(Xte), "targets": {}}
    for j, name in enumerate(TARGETS):
        gp = fit_gp(Xtr, Ytr[:, j])
        mean, std = gp(Xte, return_std=True)
        inside = np.abs(Yte[:, j] - mean) <= 1.96 * std
        entry = {"GP": scores(Yte[:, j], mean), "GP_95pct_interval_coverage": float(inside.mean()),
                 "MLP": scores(Yte[:, j], fit_mlp(Xtr, Ytr[:, j])(Xte))}
        if name == "T_surface_max_C":
            base = np.array([analytical_1d_surface_T(Design(*x)) for x in Xte])
            entry["1-D analytical baseline"] = scores(Yte[:, j], base)
        res["targets"][name] = entry
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(res, indent=1))
    return res


def active(budget: int = 120, start: int = 15, seeds=(0, 1, 2)) -> dict:
    """Grow the training set five FE results at a time from the pool: highest GP std (active) vs random."""
    X, Y = load()
    Xpool, Ypool, Xte, Yte = split(X, Y)
    j = TARGETS.index("vm_Cu_p99_MPa")
    curves = {"active": [], "random": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        first = list(rng.choice(len(Xpool), start, replace=False))
        for mode in curves:
            chosen, curve = list(first), []
            while len(chosen) <= budget:
                gp = fit_gp(Xpool[chosen], Ypool[chosen, j])
                curve.append((len(chosen), scores(Yte[:, j], gp(Xte))["MAE"]))
                rest = np.array([i for i in range(len(Xpool)) if i not in chosen])
                if mode == "active":   # the 5 pool points the current model is least sure about
                    _, std = gp(Xpool[rest], return_std=True)
                    chosen += list(rest[np.argsort(std)[-5:]])
                else:
                    chosen += list(rng.choice(rest, 5, replace=False))
            curves[mode].append(curve)
    summary = {}
    for mode, cs in curves.items():
        n = [p[0] for p in cs[0]]
        mae = np.mean([[p[1] for p in c] for c in cs], axis=0)
        summary[mode] = {"n_train": n, "mean_test_MAE_MPa": [round(float(m), 2) for m in mae]}
    OUT.mkdir(exist_ok=True)
    (OUT / "active_learning.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    print(json.dumps({"generate": lambda: generate() or "done", "evaluate": evaluate, "active": active}[cmd](), indent=1))
