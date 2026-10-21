"""Which designs are allowed? Surrogates for the thermal limits of the W/Cu tile, judged as an engineer would use them.

    python3 feasibility.py mlp      # cross-validated MLP vs GP on the held-out set -> results/mlp_tuned.json
    python3 feasibility.py classify # held-out feasibility classification from the GP -> results/feasibility.json
    python3 feasibility.py active   # boundary-targeted vs uncertainty vs random sampling -> results/active_boundary.json
    python3 feasibility.py map      # feasibility map over heat flux and coolant HTC -> results/feasibility_map.png

Screening limits, not design-code limits: copper interface <= 450 °C (CuCrZr loses its precipitation hardening above
roughly 400-500 °C) and tungsten surface <= 1300 °C (onset of recrystallisation). Both can be changed below.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surrogate import INPUTS, OUT, TARGETS, fit_gp, load, scores, split

T_INT_MAX = 450.0
T_SURF_MAX = 1300.0
J_SURF, J_INT = TARGETS.index("T_surface_max_C"), TARGETS.index("T_interface_max_C")


def feasible(t_surf, t_int):
    return (t_surf <= T_SURF_MAX) & (t_int <= T_INT_MAX)


def mlp():
    """The first comparison used an untuned MLP. Tune it by 5-fold CV on the training set only, then score once."""
    X, Y = load()
    Xtr, Ytr, Xte, Yte = split(X, Y)
    grid = {"mlpregressor__hidden_layer_sizes": [(64, 64), (128, 128), (256, 256), (128, 128, 128)],
            "mlpregressor__alpha": [1e-3, 1e-1, 1.0, 10.0],
            "mlpregressor__learning_rate_init": [1e-3, 3e-3]}
    res = {}
    for j, name in enumerate(TARGETS):
        sy = StandardScaler().fit(Ytr[:, [j]])
        pipe = make_pipeline(StandardScaler(), MLPRegressor(max_iter=8000, random_state=0, early_stopping=False))
        cv = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        cv.fit(Xtr, sy.transform(Ytr[:, [j]]).ravel())
        pred = sy.inverse_transform(cv.predict(Xte)[:, None]).ravel()
        res[name] = {"MLP_tuned": scores(Yte[:, j], pred), "best_params": cv.best_params_,
                     "GP": scores(Yte[:, j], fit_gp(Xtr, Ytr[:, j])(Xte))}
    (OUT / "mlp_tuned.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def classify():
    X, Y = load()
    Xtr, Ytr, Xte, Yte = split(X, Y)
    truth = feasible(Yte[:, J_SURF], Yte[:, J_INT])
    ms, ss = fit_gp(Xtr, Ytr[:, J_SURF])(Xte, return_std=True)
    mi, si = fit_gp(Xtr, Ytr[:, J_INT])(Xte, return_std=True)
    pred = feasible(ms, mi)
    # A design is "sure" if both limits are cleared, or one is violated, by more than 2 GP standard deviations.
    margin = np.minimum((T_SURF_MAX - ms) / ss, (T_INT_MAX - mi) / si)
    sure = np.abs(margin) > 2
    res = {"limits": {"T_interface_max_C": T_INT_MAX, "T_surface_max_C": T_SURF_MAX},
           "n_test": int(len(truth)), "feasible_in_test": int(truth.sum()),
           "correct": int((pred == truth).sum()),
           "false_feasible": int((pred & ~truth).sum()), "false_infeasible": int((~pred & truth).sum()),
           "flagged_uncertain": int((~sure).sum()), "wrong_among_sure": int(((pred != truth) & sure).sum()),
           "feasible_fraction_all_300": float(feasible(Y[:, J_SURF], Y[:, J_INT]).mean())}
    (OUT / "feasibility.json").write_text(json.dumps(res, indent=1))
    return res


def _batch(gp_fit, Xc, Yc, pool_idx, Xpool, k, score_fn):
    """Pick k points one at a time, each time adding the previous pick at its predicted value ("kriging believer"),
    so a batch spreads out instead of piling into one uncertain corner."""
    chosen, Xb, yb = [], list(Xc), list(Yc)
    for _ in range(k):
        f = gp_fit(np.array(Xb), np.array(yb))
        rest = [i for i in pool_idx if i not in chosen]
        m, s = f(Xpool[rest], return_std=True)
        i = rest[int(np.argmax(score_fn(m, s)))]
        chosen.append(i)
        Xb.append(Xpool[i]); yb.append(float(f(Xpool[[i]])[0]))
    return chosen


HARD = OUT.parent / "data" / "boundary_test.csv"


def hard_test(n=50, band=30.0, seed=1):
    """The random 60-design test set has only 3 designs within 25 °C of the 450 °C limit, so 'no misclassifications'
    mostly tests easy cases. Build a hard test set: new designs whose interface temperature (predicted by a GP on all
    300 solves) lies within +-band of the limit, then solve each with the FE model for the true label."""
    import csv
    from fe_model import Design, solve_design
    from surrogate import designs
    X, Y = load()
    cand = designs(20000, seed=seed)
    C = np.array([[d.q_MW, d.t_w_mm, d.t_cu_mm, d.h_kW, d.r_mm] for d in cand])
    near = np.where(np.abs(fit_gp(X, Y[:, J_INT])(C) - T_INT_MAX) <= band)[0]
    pick = np.random.default_rng(seed).choice(near, n, replace=False)
    rows = [solve_design(cand[i]) for i in pick]
    with HARD.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ti = np.array([r["T_interface_max_C"] for r in rows])
    return {"candidates_in_band": int(len(near)), "solved": n, "feasible": int((ti <= T_INT_MAX).sum()),
            "within_10C_true": int((np.abs(ti - T_INT_MAX) <= 10).sum()), "within_25C_true": int((np.abs(ti - T_INT_MAX) <= 25).sum())}


def _load_hard():
    import csv
    rows = list(csv.DictReader(HARD.open()))
    return (np.array([[float(r[k]) for k in INPUTS] for r in rows]),
            np.array([[float(r[k]) for k in TARGETS] for r in rows]))


def active(budget=90, start=15, step=5, seeds=range(10), test="random"):
    """Target: classify held-out designs as feasible or not on the copper-interface limit, the binding one here.
    'boundary' samples where the GP is most likely to be on the wrong side of 450 °C (straddle rule, Bryan et al., NIPS 2005);
    'uncertainty' samples the largest GP std; 'random' samples uniformly. Same start set per seed."""
    X, Y = load()
    Xp, Yp, Xte, Yte = split(X, Y)
    if test == "hard":
        Xte, Yte = _load_hard()
    truth = Yte[:, J_INT] <= T_INT_MAX
    rules = {"random": None,
             "uncertainty": lambda m, s: s,
             "boundary": lambda m, s: 1.96 * s - np.abs(m - T_INT_MAX)}
    curves = {r: [] for r in rules}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        first = list(rng.choice(len(Xp), start, replace=False))
        for rule, fn in rules.items():
            chosen, curve = list(first), []
            while len(chosen) <= budget:
                f = fit_gp(Xp[chosen], Yp[chosen, J_INT])
                m = f(Xte)
                curve.append((len(chosen), int(((m <= T_INT_MAX) != truth).sum()), scores(Yte[:, J_INT], m)["MAE"]))
                if len(chosen) == budget:
                    break
                pool = [i for i in range(len(Xp)) if i not in chosen]
                if fn is None:
                    chosen += list(rng.choice(pool, step, replace=False))
                else:
                    chosen += _batch(fit_gp, Xp[chosen], Yp[chosen, J_INT], pool, Xp, step, fn)
            curves[rule].append(curve)
        print("seed", seed, {r: c[-1][-1][1] for r, c in curves.items()}, flush=True)
    out = {}
    for rule, cs in curves.items():
        a = np.array(cs, dtype=float)          # seeds x steps x 3
        out[rule] = {"n_train": a[0, :, 0].astype(int).tolist(),
                     "misclassified_mean": a[:, :, 1].mean(0).round(2).tolist(),
                     "misclassified_sd": a[:, :, 1].std(0).round(2).tolist(),
                     "T_interface_MAE_mean": a[:, :, 2].mean(0).round(2).tolist()}
    out["n_test"] = int(len(truth)); out["feasible_in_test"] = int(truth.sum()); out["seeds"] = len(list(seeds)); out["test_set"] = test
    (OUT / f"active_boundary_{test}.json").write_text(json.dumps(out, indent=1))
    return out


def fmap(t_w=5.0, t_cu=10.0, r=3.0):
    """Feasibility over heat flux x coolant HTC at one geometry, from the GP trained on all 300 FE solves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, Y = load()
    q, h = np.meshgrid(np.linspace(5, 20, 121), np.linspace(20, 150, 121))
    G = np.column_stack([q.ravel(), np.full(q.size, t_w), np.full(q.size, t_cu), h.ravel(), np.full(q.size, r)])
    Ts = fit_gp(X, Y[:, J_SURF])(G).reshape(q.shape)
    Ti = fit_gp(X, Y[:, J_INT])(G).reshape(q.shape)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.contourf(q, h, feasible(Ts, Ti).astype(float), levels=[-0.5, 0.5, 1.5], colors=["#f2d7d5", "#d6eadf"])
    c1 = ax.contour(q, h, Ti, levels=[T_INT_MAX], colors="#9b2c2c", linewidths=1.4)
    c2 = ax.contour(q, h, Ts, levels=[T_SURF_MAX], colors="#2b6cb0", linewidths=1.4, linestyles="--")
    ax.clabel(c1, fmt={T_INT_MAX: "Cu interface 450 °C"}, fontsize=8)
    ax.clabel(c2, fmt={T_SURF_MAX: "W surface 1300 °C"}, fontsize=8)
    ax.set_xlabel("surface heat flux (MW/m²)")
    ax.set_ylabel("coolant heat-transfer coefficient (kW/m²K)")
    ax.set_title(f"Allowed designs (green): {t_w:g} mm W, {t_cu:g} mm Cu, {r:g} mm channel", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "feasibility_map.png", dpi=150)
    return {"feasible_fraction_on_map": float(feasible(Ts, Ti).mean())}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "classify"
    print(json.dumps({"mlp": mlp, "classify": classify, "active": active, "map": fmap, "hard": hard_test,
                      "active_hard": lambda: active(test="hard")}[cmd](), indent=1, default=str))
