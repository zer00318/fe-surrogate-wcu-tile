"""2-D thermo-mechanical finite-element model of a tungsten-on-copper tile with a cooling channel.

Steady heat conduction, then plane-strain linear thermo-elasticity, with scikit-fem on a gmsh mesh.
Everything here is generic and public: textbook material constants, a parametric rectangle-with-hole
geometry, and no data from any institute.

    T solve:  -div(k grad T) = 0,  q on the top surface, h (T - T_cool) on the channel wall, insulated elsewhere
    u solve:  div sigma = 0,  sigma = lambda tr(eps) I + 2 mu eps - (3 lambda + 2 mu) alpha (T - T_ref) I
              bottom edge fixed in y, bottom-left corner fixed in x

`validate_1d()` replaces the channel by a convective bottom edge, where the exact answer is known.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import gmsh
import numpy as np
from skfem import (Basis, ElementTriP2, ElementVector, FacetBasis, MeshTri, BilinearForm, LinearForm,
                   asm, condense, solve)
from skfem.helpers import dot, grad, sym_grad, trace, div, eye, ddot

# Textbook room-temperature constants, held constant (no temperature dependence, no plasticity).
W = dict(k=130.0, E=400e9, nu=0.28, alpha=4.5e-6)
CU = dict(k=380.0, E=120e9, nu=0.33, alpha=16.7e-6)
T_COOL = 100.0     # °C, coolant bulk temperature
T_REF = 20.0       # °C, stress-free reference temperature
WIDTH = 24e-3      # m, tile width


@dataclass(frozen=True)
class Design:
    q_MW: float       # surface heat flux, MW/m^2
    t_w_mm: float     # tungsten armour thickness
    t_cu_mm: float    # copper heat-sink thickness (channel sits at its mid-height)
    h_kW: float       # coolant heat-transfer coefficient, kW/(m^2 K)
    r_mm: float       # channel radius


def _mesh(d: Design | None, h_size: float, flat: bool = False, t_w: float = 5e-3, t_cu: float = 10e-3) -> MeshTri:
    """Triangular mesh with boundary tags: 'top', 'bottom', 'channel'. Nodes lie on the W/Cu interface."""
    if d is not None:
        t_w, t_cu = d.t_w_mm * 1e-3, d.t_cu_mm * 1e-3
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tile")
    occ = gmsh.model.occ
    cu = occ.addRectangle(0, 0, 0, WIDTH, t_cu)
    w = occ.addRectangle(0, t_cu, 0, WIDTH, t_w)
    if not flat:
        hole = occ.addDisk(WIDTH / 2, t_cu / 2, 0, d.r_mm * 1e-3, d.r_mm * 1e-3)
        cu_cut, _ = occ.cut([(2, cu)], [(2, hole)])
        cu = cu_cut[0][1]
    occ.fragment([(2, cu)], [(2, w)])
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_size / 4)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xy = np.zeros((2, int(tags.max()) + 1))
    xy[:, tags.astype(int)] = coords.reshape(-1, 3)[:, :2].T
    tris = []
    for et, _, nodes in zip(*gmsh.model.mesh.getElements(2)):
        if et == 2:
            tris.append(nodes.reshape(-1, 3).astype(int))
    gmsh.finalize()
    t = np.vstack(tris).T
    used = np.unique(t)
    remap = -np.ones(xy.shape[1], dtype=int)
    remap[used] = np.arange(used.size)
    m = MeshTri(xy[:, used], remap[t])
    top_y = t_cu + t_w
    tol = 1e-9
    fx, fy = m.p[:, m.facets].mean(axis=1)
    boundary = m.boundary_facets()
    on = lambda mask: boundary[mask[boundary]]
    cx, cy = WIDTH / 2, t_cu / 2
    radius = np.hypot(fx - cx, fy - cy)
    m = m.with_boundaries({
        "top": lambda x: np.abs(x[1] - top_y) < tol,
        "bottom": lambda x: np.abs(x[1]) < tol,
    })
    if not flat:
        chan = on((radius < d.r_mm * 1e-3 * 1.05) & (np.abs(fy - top_y) > tol))
        m = m.with_boundaries({"channel": lambda x: np.hypot(x[0] - cx, x[1] - cy) < d.r_mm * 1e-3 * 1.05}, boundaries_only=True)
    return m, t_w, t_cu


def _field(x_y, t_cu, key):
    return np.where(x_y > t_cu, W[key], CU[key])


def solve_design(d: Design, h_size: float = 0.4e-3, flat: bool = False) -> dict:
    m, t_w, t_cu = _mesh(d, h_size, flat=flat)
    q = d.q_MW * 1e6
    h = d.h_kW * 1e3
    e = ElementTriP2()
    basis = Basis(m, e)
    top = FacetBasis(m, e, facets=m.boundaries["top"])
    cool = FacetBasis(m, e, facets=m.boundaries["bottom" if flat else "channel"])

    @BilinearForm
    def conduction(u, v, w):
        return _field(w.x[1], t_cu, "k") * dot(grad(u), grad(v))

    @BilinearForm
    def robin(u, v, w):
        return h * u * v

    @LinearForm
    def flux(v, w):
        return q * v

    @LinearForm
    def robin_rhs(v, w):
        return h * T_COOL * v

    K = asm(conduction, basis) + asm(robin, cool)
    f = asm(flux, top) + asm(robin_rhs, cool)
    T = solve(K, f)

    # Plane-strain thermo-elasticity with the temperature field interpolated onto the vector basis.
    ev = ElementVector(ElementTriP2())
    vb = basis.with_element(ev)
    Tq = basis.interpolate(T)

    def lame(y):
        E, nu = _field(y, t_cu, "E"), _field(y, t_cu, "nu")
        return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

    @BilinearForm
    def stiffness(u, v, w):
        lam, mu = lame(w.x[1])
        eps = sym_grad(u)
        return ddot(eye(lam * trace(eps), 2) + 2 * mu * eps, sym_grad(v))

    @LinearForm
    def thermal(v, w):
        lam, mu = lame(w.x[1])
        a = _field(w.x[1], t_cu, "alpha")
        return (3 * lam + 2 * mu) * a * (w.T - T_REF) * div(v)

    A = asm(stiffness, vb)
    b = asm(thermal, vb, T=Tq)
    bottom_dofs = vb.get_dofs("bottom").nodal["u^2"]
    # get_dofs(lambda) selects facets by midpoint and so never matches a single vertex: pin the x DOF of the
    # bottom-left vertex directly, or the x rigid-body translation stays free and the system is singular.
    corner_node = int(np.argmin(np.hypot(m.p[0], m.p[1])))
    corner = np.atleast_1d(vb.nodal_dofs[0, corner_node])
    D = np.concatenate([bottom_dofs, corner])
    u = solve(*condense(A, b, D=D))

    # Stresses at quadrature points (plane strain: sigma_zz = nu (sxx + syy) - E alpha dT).
    du = vb.interpolate(u).grad
    eps = 0.5 * (du + np.transpose(du, (1, 0, 2, 3)))
    y = basis.global_coordinates().value[1]
    lam, mu = lame(y)
    E, nu, a = _field(y, t_cu, "E"), _field(y, t_cu, "nu"), _field(y, t_cu, "alpha")
    dT = Tq.value - T_REF
    tr = eps[0, 0] + eps[1, 1]
    sxx = lam * tr + 2 * mu * eps[0, 0] - (3 * lam + 2 * mu) * a * dT
    syy = lam * tr + 2 * mu * eps[1, 1] - (3 * lam + 2 * mu) * a * dT
    sxy = 2 * mu * eps[0, 1]
    szz = nu * (sxx + syy) - E * a * dT
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
    in_w = y > t_cu
    wts = basis.dx  # quadrature weights (area per point), same shape as y

    def p99(mask):
        """Area-weighted 99th percentile: robust to the free-edge stress singularity at the bimaterial corners."""
        v, a = vm[mask], wts[mask]
        o = np.argsort(v)
        c = np.cumsum(a[o]) / a.sum()
        return float(v[o][np.searchsorted(c, 0.99)])

    top_T = T[basis.get_dofs("top").all()]
    near_iface = np.abs(y - t_cu) < 0.5e-3
    return {
        **asdict(d),
        "T_surface_max_C": float(top_T.max()),
        "T_interface_max_C": float(Tq.value[near_iface].max()),
        "vm_W_max_MPa": float(vm[in_w].max() / 1e6),
        "vm_Cu_max_MPa": float(vm[~in_w].max() / 1e6),
        "vm_W_p99_MPa": p99(in_w) / 1e6,
        "vm_Cu_p99_MPa": p99(~in_w) / 1e6,
        "n_nodes": int(m.p.shape[1]),
    }


def analytical_1d_surface_T(d: Design) -> float:
    """Exact surface temperature for the flat (channel-free) stack cooled on its bottom edge."""
    q, h = d.q_MW * 1e6, d.h_kW * 1e3
    return T_COOL + q / h + q * d.t_cu_mm * 1e-3 / CU["k"] + q * d.t_w_mm * 1e-3 / W["k"]


def validate_1d() -> list[dict]:
    rows = []
    for d in [Design(10, 5, 10, 50, 3), Design(20, 3, 8, 100, 3), Design(5, 8, 12, 30, 3)]:
        fe = solve_design(d, flat=True)["T_surface_max_C"]
        ex = analytical_1d_surface_T(d)
        rows.append({"design": asdict(d), "fe_C": round(fe, 3), "exact_C": round(ex, 3), "rel_err": abs(fe - ex) / ex})
    return rows


if __name__ == "__main__":
    import json
    import time
    for r in validate_1d():
        print("1-D check", json.dumps(r))
    d = Design(10, 5, 10, 50, 3)
    for hs in (0.8e-3, 0.4e-3, 0.2e-3, 0.1e-3):
        t0 = time.time()
        out = solve_design(d, h_size=hs)
        print(f"mesh {hs*1e3:.1f} mm, {out['n_nodes']} nodes, {time.time()-t0:.1f} s:",
              {k: round(v, 2) for k, v in out.items() if k.endswith(("_C", "_MPa"))})
