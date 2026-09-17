"""
tests.py -- validation suite for the network security game solver.

Run:  python tests.py
Every check is an independent verification of a mathematical claim made in
REPORT.md.  Nothing here is self-referential: closed forms, brute force and
random sampling are all computed independently of the solver.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import closed_form as cf
import graphs
from core import (PathOracle, brute_force_grid, cut_from, dual_bound_path,
                  evader_alloc, game_value, heuristic_allocation, psi,
                  solve_defender)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- inner problem
def test_inner_vs_random_sampling():
    print("\n1. Inner problem: KKT solution beats random feasible splits")
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(6):
        n = rng.integers(2, 6)
        x = rng.uniform(0.2, 8.0, n)
        XB = float(rng.uniform(1.0, 20.0))
        sol = evader_alloc(x, XB)
        W = rng.dirichlet(np.ones(n), size=40000) * XB
        vals = np.log(W / (x + W)).sum(axis=1)
        worst = min(worst, sol["logval"] - vals.max())
    check("KKT value >= best of 240k random feasible splits", worst >= -1e-9,
          f"min margin {worst:.2e}")


def test_inner_kkt_and_budget():
    print("\n2. Inner problem: stationarity and budget feasibility")
    rng = np.random.default_rng(1)
    dlam, dbud = 0.0, 0.0
    for _ in range(300):
        n = int(rng.integers(1, 8))
        x = rng.uniform(1e-3, 50.0, n)
        XB = float(rng.uniform(0.01, 100.0))
        s = evader_alloc(x, XB)
        lam = x / (s["y"] * (x + s["y"]))
        dlam = max(dlam, (lam.max() - lam.min()) / lam.mean())
        dbud = max(dbud, abs(s["y"].sum() - XB) / XB)
    check("multipliers x_i/(y_i(x_i+y_i)) identical across nodes", dlam < 1e-8,
          f"max rel spread {dlam:.2e}")
    check("evader budget exactly exhausted", dbud < 1e-12, f"max rel err {dbud:.2e}")


def test_inner_symmetry():
    print("\n3. Inner problem: symmetric defence -> symmetric attack")
    s = evader_alloc(np.full(5, 2.0), 7.0)
    check("equal x_i gives equal y_i = XB/n", np.allclose(s["y"], 7.0 / 5, rtol=1e-12))
    check("p_i = XB/n / (x + XB/n)",
          abs(math.exp(s["logval"]) - (1.4 / (2.0 + 1.4)) ** 5) < 1e-12)


def test_dual_bound():
    print("\n4. Lagrangian relaxation: valid upper bound, tight at lambda*")
    rng = np.random.default_rng(2)
    viol, tight = 0.0, 0.0
    for _ in range(200):
        n = int(rng.integers(1, 7))
        x = rng.uniform(1e-2, 20.0, n)
        XB = float(rng.uniform(0.1, 50.0))
        s = evader_alloc(x, XB)
        for lam in 10.0 ** rng.uniform(-4, 2, 6):
            viol = min(viol, dual_bound_path(x, XB, lam) - s["logval"])
        tight = max(tight, abs(dual_bound_path(x, XB, s["lam"]) - s["logval"]))
    check("g_P(x) <= lam*XB + sum psi  for every lambda", viol >= -1e-9,
          f"min slack {viol:.2e}")
    check("bound is tight at the optimal multiplier", tight < 1e-9,
          f"max error {tight:.2e}")


def test_danskin():
    print("\n5. Danskin / envelope theorem: analytic vs numerical gradient")
    rng = np.random.default_rng(4)
    err = 0.0
    for _ in range(40):
        n = int(rng.integers(2, 6))
        x = rng.uniform(0.5, 10.0, n)
        XB = float(rng.uniform(1.0, 20.0))
        s = evader_alloc(x, XB)
        ana = cut_from(x, s["y"])
        for i in range(n):
            h = 1e-6 * max(x[i], 1.0)
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            num = (evader_alloc(xp, XB)["logval"] - evader_alloc(xm, XB)["logval"]) / (2 * h)
            err = max(err, abs(num - ana[i]) / max(abs(ana[i]), 1e-9))
    check("dg/dx_i = -1/(x_i+y_i*) matches central differences", err < 1e-5,
          f"max rel err {err:.2e}")


# ---------------------------------------------------------------- convexity
def test_convexity():
    print("\n6. Convexity of the defender objective G(x) = max_P g_P(x)")
    rng = np.random.default_rng(5)
    worst = 0.0
    for name in ["grid-4x4", "random-12", "layered-3x3", "unequal-1-2-4"]:
        G, S, D = graphs.CATALOG[name]()
        orc = PathOracle(G, S, D)
        n = len(orc.nodes)
        for _ in range(40):
            a = rng.dirichlet(np.ones(n)) * 10.0
            b = rng.dirichlet(np.ones(n)) * 10.0
            t = float(rng.uniform(0.05, 0.95))
            lhs = orc.best_response(t * a + (1 - t) * b, 10.0)["logval"]
            rhs = (t * orc.best_response(a, 10.0)["logval"]
                   + (1 - t) * orc.best_response(b, 10.0)["logval"])
            worst = max(worst, lhs - rhs)
    check("G(ta+(1-t)b) <= tG(a)+(1-t)G(b) on 160 random chords (m=1)",
          worst <= 1e-9, f"max violation {worst:.2e}")


# ---------------------------------------------------------------- closed forms
def test_closed_forms():
    print("\n7. Solver vs analytical benchmarks")
    err = []
    for XA, XB in [(10.0, 10.0), (25.0, 4.0), (3.0, 17.0)]:
        for n in [1, 2, 3, 5]:
            r = solve_defender(*graphs.chain(n), XA, XB, tol=1e-10)
            err.append(abs(r["value"] - cf.chain_value(n, XA, XB)))
    check("single chain,  V* = (XB/(XA+XB))^n", max(err) < 1e-9,
          f"max abs err {max(err):.2e}")

    err = []
    for k, n in [(2, 1), (3, 2), (4, 3), (2, 4)]:
        for XA, XB in [(10.0, 10.0), (20.0, 5.0)]:
            r = solve_defender(*graphs.parallel_chains(k, n), XA, XB, tol=1e-10)
            err.append(abs(r["value"] - cf.parallel_value(k, n, XA, XB)))
    check("k parallel chains,  V* = (XB/(XA/k+XB))^n", max(err) < 1e-7,
          f"max abs err {max(err):.2e}")

    err = []
    for L, w in [(2, 2), (3, 3), (2, 4), (4, 2)]:
        r = solve_defender(*graphs.layered(L, w), 10.0, 10.0, tol=1e-10)
        err.append(abs(r["value"] - cf.layered_value(L, w, 10.0, 10.0)))
    check("complete layered DAG,  V* = (XB/(XA/w+XB))^L", max(err) < 1e-7,
          f"max abs err {max(err):.2e}")

    err, eqz = [], []
    for Ls in [[1, 2], [1, 2, 4], [2, 3, 3], [1, 1, 5]]:
        r = solve_defender(*graphs.unequal_branches(Ls), 10.0, 10.0,
                           tol=1e-10, max_iter=400)
        V, a = cf.unequal_branches_value(Ls, 10.0, 10.0)
        err.append(abs(r["value"] - V))
        got = [sum(v for k, v in r["x"].items() if k.startswith(f"b{j}_"))
               for j in range(len(Ls))]
        eqz.append(max(abs(g - t) for g, t in zip(got, a)))
    check("unequal branches, value matches equalisation formula",
          max(err) < 1e-6, f"max abs err {max(err):.2e}")
    check("unequal branches, per-branch budgets match a_j = XB(V^(-1/L_j)-1)",
          max(eqz) < 1e-3, f"max abs err {max(eqz):.2e}")

    r = solve_defender(*graphs.chain(1), 7.0, 3.0, tol=1e-12)
    check("single contested node,  V* = XB/(XA+XB)",
          abs(r["value"] - cf.single_node(7.0, 3.0)) < 1e-12)


# ---------------------------------------------------------------- oracles
def test_oracle_equivalence():
    print("\n8. Certified path oracle == exhaustive enumeration")
    rng = np.random.default_rng(6)
    bad, uncert, touched, total = 0, 0, [], []
    for seed in range(10):
        G, S, D = graphs.random_dag(12, 0.25, seed=seed)
        oe = PathOracle(G, S, D, mode="enum")
        ok = PathOracle(G, S, D, mode="ksp")
        for _ in range(8):
            x = rng.dirichlet(np.ones(len(oe.nodes))) * 10.0
            a, b = oe.best_response(x, 10.0), ok.best_response(x, 10.0)
            bad += abs(a["logval"] - b["logval"]) > 1e-9
            uncert += not b["certified"]
            touched.append(b["n_evaluated"])
            total.append(oe.n_paths_known)
    check("k-shortest-path oracle returns the true best response", bad == 0,
          f"{bad} mismatches in 80 trials")
    check("stopping rule certifies optimality every time", uncert == 0,
          f"mean paths touched {np.mean(touched):.1f} of {np.mean(total):.1f}")


def test_brute_force():
    print("\n9. Cutting plane vs brute-force grid search (tiny instances)")
    errs = []
    for gname, XA, XB, steps in [("diamond", 10.0, 10.0, 200),
                                 ("lecture", 10.0, 10.0, 60),
                                 ("unequal-1-1-2", 6.0, 9.0, 30)]:
        G, S, D = (graphs.unequal_branches([1, 1, 2]) if gname == "unequal-1-1-2"
                   else graphs.CATALOG[gname]())
        bf = brute_force_grid(G, S, D, XA, XB, steps=steps)
        r = solve_defender(G, S, D, XA, XB, tol=1e-10, max_iter=300)
        errs.append((gname, r["value"], bf["value"]))
        check(f"grid search never beats the solver ({gname})",
              bf["value"] >= r["value"] - 1e-9,
              f"solver {r['value']:.8f} vs grid {bf['value']:.8f}")
    # a finite grid can only approach the optimum from above: check that the
    # residual shrinks as the grid is refined (i.e. it is discretisation error)
    G, S, D = graphs.CATALOG["lecture"]()
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-10, max_iter=300)
    res = [brute_force_grid(G, S, D, 10.0, 10.0, steps=k)["value"] - r["value"]
           for k in (20, 60, 180, 540)]
    # nested grids => the residual can only go down, and must tend to 0
    check("grid-search residual is non-negative, non-increasing, and -> 0",
          all(-1e-12 <= b <= a + 1e-12 for a, b in zip(res, res[1:]))
          and res[-1] < 0.2 * res[0],
          "  ".join(f"{e:.2e}" for e in res))


def test_heuristics_never_better():
    print("\n10. Baselines are never better than the certified optimum")
    worst = 0.0
    rows = []
    for seed in range(6):
        G, S, D = graphs.random_dag(11, 0.28, seed=seed)
        r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9, max_iter=300)
        for kind in ["uniform", "pathcount", "mincut", "betweenness"]:
            xh = heuristic_allocation(G, S, D, 10.0, kind, XB=10.0,
                                      oracle=r["oracle"])
            vh = game_value(G, S, D, xh, 10.0, oracle=r["oracle"])["value"]
            worst = min(worst, vh - r["value"])
            rows.append((seed, kind, vh / r["value"] - 1.0))
    check("no heuristic allocation beats the optimum", worst >= -1e-7,
          f"min margin {worst:.2e}")
    for kind in ["uniform", "pathcount", "mincut", "betweenness"]:
        g = [e for _, k, e in rows if k == kind]
        print(f"        {kind:<13s} mean excess value {100*np.mean(g):6.2f}%"
              f"   max {100*max(g):6.2f}%")


def test_comparative_statics():
    print("\n11. Comparative statics and feasibility")
    G, S, D = graphs.grid_dag(3, 3)
    vals = [solve_defender(G, S, D, XA, 10.0, tol=1e-8)["value"]
            for XA in [1.0, 5.0, 10.0, 40.0, 200.0]]
    check("V* strictly decreasing in the defender budget",
          all(a > b for a, b in zip(vals, vals[1:])),
          " > ".join(f"{v:.4f}" for v in vals))
    vals = [solve_defender(G, S, D, 10.0, XB, tol=1e-8)["value"]
            for XB in [1.0, 5.0, 10.0, 40.0, 200.0]]
    check("V* strictly increasing in the evader budget",
          all(a < b for a, b in zip(vals, vals[1:])),
          " < ".join(f"{v:.4f}" for v in vals))
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9)
    check("defender budget exactly exhausted",
          abs(sum(r["x"].values()) - 10.0) < 1e-9)
    check("evader budget exactly exhausted",
          abs(sum(r["y"].values()) - 10.0) < 1e-9)
    check("certified bracket  LB <= V* <= UB",
          r["log_lb"] <= r["log_ub"] + 1e-12 and r["value"] <= 1.0)


def test_uncovered_path():
    print("\n12. Structural necessary condition (vertex-cut / covering)")
    G, S, D = graphs.parallel_chains(3, 2)
    orc = PathOracle(G, S, D)
    x = {v: 0.0 for v in orc.nodes}
    for v in orc.nodes:
        if v.startswith("b0") or v.startswith("b1"):
            x[v] = 5.0
    v = game_value(G, S, D, x, 10.0, oracle=orc)["value"]
    check("leaving one branch undefended gives V = 1", abs(v - 1.0) < 1e-12)
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9)
    check("optimal support hits every S-D path",
          all(any(r["x"][v] > 1e-9 for v in orc.interiors[i])
              for i in range(len(orc.paths))))


def test_general_m():
    print("\n13. General contest intensity m (ratio-form CSF)")
    rng = np.random.default_rng(8)
    worst = 0.0
    for m in [0.4, 0.7, 1.0, 1.6]:
        for _ in range(3):
            n = int(rng.integers(2, 5))
            x = rng.uniform(0.5, 6.0, n)
            XB = float(rng.uniform(2.0, 15.0))
            s = evader_alloc(x, XB, m=m)
            W = rng.dirichlet(np.ones(n), size=30000) * XB
            v = np.log(W ** m / (x ** m + W ** m)).sum(axis=1).max()
            worst = min(worst, s["logval"] - v)
    check("inner solver optimal for m in {0.4,0.7,1.0,1.6}", worst >= -1e-8,
          f"min margin {worst:.2e}")

    rng = np.random.default_rng(9)
    G, S, D = graphs.grid_dag(3, 3)
    out = {}
    for m in [0.5, 1.0]:
        orc = PathOracle(G, S, D, m=m)
        n = len(orc.nodes)
        worst = 0.0
        for _ in range(40):
            a = rng.dirichlet(np.ones(n)) * 10.0
            b = rng.dirichlet(np.ones(n)) * 10.0
            t = float(rng.uniform(0.05, 0.95))
            worst = max(worst,
                        orc.best_response(t * a + (1 - t) * b, 10.0)["logval"]
                        - t * orc.best_response(a, 10.0)["logval"]
                        - (1 - t) * orc.best_response(b, 10.0)["logval"])
        out[m] = worst
    check("G is convex for m <= 1", out[0.5] <= 1e-9 and out[1.0] <= 1e-9,
          f"max violation m=0.5: {out[0.5]:.2e},  m=1: {out[1.0]:.2e}")

    # explicit counterexample for m > 1.  -log(x^m + c) has second derivative
    # -m x^(m-2)[(m-1)c - x^m]/(x^m+c)^2, which is NEGATIVE when x^m < (m-1)c.
    orc = PathOracle(*graphs.diamond(), m=2.0)
    f = lambda x1: orc.best_response(np.array([x1, 10.0 - x1]), 10.0)["logval"]
    viol = max(0.5 * f(a) + 0.5 * f(b) - f(0.5 * (a + b))
               for a, b in [(2.0, 4.0), (1.0, 3.0), (2.0, 3.0), (3.0, 5.0)])
    check("m > 1 genuinely breaks convexity (constructed counterexample)",
          viol < -1e-3,
          f"max chord deficit {viol:.3e} on the diamond, m=2 "
          "-> cutting-plane optimality guarantee does NOT extend to m > 1")


def test_contested_endpoints():
    print("\n15. Contested endpoints (contest_endpoints=True)")
    # The Lagrangian cost of a node is carried on its INCOMING edges, so the
    # source contributes nothing unless it is added explicitly.  With
    # uncontested endpoints S has no cost and this is moot; with contested
    # endpoints, omitting it leaves the bound valid but loose (S is on every
    # S-D path, so the omission is a constant shift), which makes the
    # stopping rule fire later than it should.
    rng = np.random.default_rng(12)
    bad, uncert, touched = 0, 0, []
    for seed in range(6):
        G, S, D = graphs.random_dag(8, 0.3, seed=seed)
        oe = PathOracle(G, S, D, mode="enum", contest_endpoints=True)
        ok = PathOracle(G, S, D, mode="ksp", contest_endpoints=True)
        for _ in range(6):
            x = rng.dirichlet(np.ones(len(oe.nodes))) * 10.0
            a = oe.best_response(x, 10.0)
            b = ok.best_response(x, 10.0)
            bad += abs(a["logval"] - b["logval"]) > 1e-9
            bad += b["ub"] < a["logval"] - 1e-9          # bound must be valid
            uncert += not b["certified"]
            touched.append(b["n_evaluated"])
    check("ksp oracle matches enumeration with endpoints contested", bad == 0,
          f"{bad} mismatches in 36 trials")
    check("dual bound stays valid and tight with endpoints contested",
          uncert == 0, f"mean paths touched {np.mean(touched):.2f}")

    # the source's cost must actually be included: zeroing it must loosen the
    # bound (more paths touched), which is what the bug did.
    G, S, D = graphs.random_dag(8, 0.3, seed=1)
    ok = PathOracle(G, S, D, mode="ksp", contest_endpoints=True)
    rng = np.random.default_rng(3)
    xs = [rng.dirichlet(np.ones(len(ok.nodes))) * 10.0 for _ in range(6)]
    fixed = [ok.best_response(x, 10.0)["n_evaluated"] for x in xs]
    orig = ok._set_weights

    def patched(xv, lam):
        c = orig(xv, lam)
        ok._src_cost = 0.0                      # reintroduce the bug
        return c

    ok._set_weights = patched
    buggy = [ok.best_response(x, 10.0)["n_evaluated"] for x in xs]
    ok._set_weights = orig
    check("including the source cost tightens the stopping rule",
          sum(fixed) <= sum(buggy),
          f"paths touched {sum(fixed)} (fixed) vs {sum(buggy)} (source cost omitted)")


def test_uniqueness():
    print("\n14. Uniqueness of the optimal allocation x* (EVIDENCE, not proof)")
    # NOTE ON STATUS.  G is convex for m <= 1 (proved, [T-6]).  Strict
    # convexity in the FULL vector x is NOT established: g_P depends only on
    # the coordinates of P, so two allocations differing only off the
    # maximising path give equality, not strict inequality.  Uniqueness of
    # x* is therefore an open question, supported by -- but not proved by --
    # the check below, which runs the solver from very different starting
    # points and confirms it lands on the same x*.  See FORMULATION.md S3.
    rng = np.random.default_rng(11)
    worst = 0.0
    for gname in ["lecture", "diamond", "unequal-1-2-4", "grid-4x4"]:
        G, S, D = graphs.CATALOG[gname]()
        oracle = PathOracle(G, S, D)
        n = len(oracle.nodes)
        starts = [None,                                   # default: uniform
                  {v: 10.0 * w for v, w in
                   zip(oracle.nodes, rng.dirichlet(np.ones(n)))},
                  {v: (10.0 if i == 0 else 1e-6) for i, v in enumerate(oracle.nodes)}]
        xs = []
        for x0 in starts:
            r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-10, max_iter=400,
                               x0=x0, oracle=oracle)
            xs.append(np.array([r["x"][v] for v in oracle.nodes]))
        base = xs[0]
        for xv in xs[1:]:
            worst = max(worst, float(np.max(np.abs(xv - base))))
    check("solver converges to the same x* from different starting points "
          "(evidence for uniqueness, not a proof)",
          worst < 5e-3, f"max coordinate spread across starts {worst:.2e}")


def main():
    print("=" * 74)
    print("  NETWORK SECURITY GAME -- VALIDATION SUITE")
    print("=" * 74)
    for t in [test_inner_vs_random_sampling, test_inner_kkt_and_budget,
              test_inner_symmetry, test_dual_bound, test_danskin,
              test_convexity, test_closed_forms, test_oracle_equivalence,
              test_brute_force, test_heuristics_never_better,
              test_comparative_statics, test_uncovered_path, test_general_m,
              test_contested_endpoints, test_uniqueness]:
        try:
            t()
        except Exception as exc:                        # pragma: no cover
            RESULTS.append((t.__name__, False))
            print(f"  [FAIL] {t.__name__} raised {type(exc).__name__}: {exc}")
    n_ok = sum(ok for _, ok in RESULTS)
    print("\n" + "=" * 74)
    print(f"  {n_ok}/{len(RESULTS)} checks passed")
    for name, ok in RESULTS:
        if not ok:
            print(f"    FAILED: {name}")
    print("=" * 74)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
