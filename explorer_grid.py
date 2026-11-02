"""Rebuilds docs/grid.json for the interactive explorer (docs/index.html): GP surrogates of the peak
W surface and Cu interface temperatures, trained on all 350 FE solves, evaluated on a regular grid.
Usage: .venv/bin/python explorer_grid.py"""
import csv, json
import numpy as np
from surrogate import INPUTS, fit_gp

rows = list(csv.DictReader(open("data/fe_samples.csv"))) + list(csv.DictReader(open("data/boundary_test.csv")))
X = np.array([[float(r[k]) for k in INPUTS] for r in rows])
q, h = np.arange(5, 20.01, 0.5), np.arange(20, 150.01, 5)
tw, tcu, rr = [2, 3, 4, 5, 6, 7, 8], [8, 10, 12, 14], [2, 3, 4]
G = np.array([[a, b, c, d, e] for b in tw for c in tcu for e in rr for d in h for a in q])  # q, t_w, t_cu, h, r
out = {"q": q.tolist(), "h": h.tolist(), "tw": tw, "tcu": tcu, "r": rr, "n_fe": len(rows)}
for target, key in [("T_surface_max_C", "Ts"), ("T_interface_max_C", "Ti")]:
    mean, sd = fit_gp(X, np.array([float(r[target]) for r in rows]))(G, return_std=True)
    out[key] = np.round(mean, 1).tolist()
    if key == "Ti":
        out["Ti_sd"] = np.round(sd, 1).tolist()
json.dump(out, open("docs/grid.json", "w"), separators=(",", ":"))
