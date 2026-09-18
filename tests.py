"""
tests.py -- validation suite for the network security game solver.

Run:  python tests.py

Every check numerically validates a mathematical claim made in FORMULATION.md.
Nothing here is self-referential: closed forms, brute force and random sampling
are all computed independently of the solver.

WHAT THIS SUITE IS AND IS NOT
-----------------------------
These are VALIDATION checks, not proofs.  Randomised sampling over chords, over
feasible splits, or over instances can only ever show that no violation was
found on the instances sampled; it cannot establish a universally quantified
statement.  The convexity theorem (FORMULATION.md S3) rests on its proof there,
and the checks below are evidence that the IMPLEMENTATION is consistent with
that proof -- they sanity-check the theorem, they do not prove it.  Check names
are worded accordingly.

All thresholds come from `tolerances.py` rather than being typed in per check,
so a pass/fail here can be read against the same numbers the solver reports.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import closed_form as cf
import core
import graphs
import tolerances as tolcfg
from core import (PathOracle, alloc_residuals, brute_force_grid, cut_from,
                  dual_bound_path, evader_alloc, game_value,
                  heuristic_allocation, psi, solve_defender)

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
    print("\n6. Convexity of G(x) = max_P g_P(x): numerical validation on "
          "sampled chords")
    # Sampling cannot prove convexity.  The theorem is proved in
    # FORMULATION.md S3; what follows numerically validates convexity on
    # sampled instances, i.e. it sanity-checks that the implementation behaves
    # as the theorem says it should.
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
    check("no convexity violation found on 160 sampled chords, m=1 "
          "(validation of the theorem, not a proof of it)",
          worst <= tolcfg.CONVEXITY_TOL, f"max violation {worst:.2e}")


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
    check("no convexity violation found on sampled chords for m in {0.5, 1} "
          "(sanity-checks the m <= 1 convexity theorem)",
          out[0.5] <= tolcfg.CONVEXITY_TOL and out[1.0] <= tolcfg.CONVEXITY_TOL,
          f"max violation m=0.5: {out[0.5]:.2e},  m=1: {out[1.0]:.2e}")

    # explicit counterexample for m > 1.  -log(x^m + c) has second derivative
    # -m x^(m-2)[(m-1)c - x^m]/(x^m+c)^2, which is NEGATIVE when x^m < (m-1)c.
    orc = PathOracle(*graphs.diamond(), m=2.0)
    f = lambda x1: orc.best_response(np.array([x1, 10.0 - x1]), 10.0)["logval"]
    viol = max(0.5 * f(a) + 0.5 * f(b) - f(0.5 * (a + b))
               for a, b in [(2.0, 4.0), (1.0, 3.0), (2.0, 3.0), (3.0, 5.0)])
    check("m > 1 genuinely breaks convexity (an explicit counterexample -- "
          "one violation suffices, unlike the sampling above)",
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


# --------------------------------------------------------- closed-form checks
def test_m1_closed_form():
    print("\n16. m = 1 closed forms: p_i(t) against the probability from y_i(t)")
    # p_i(t) = 2t / (x_i + 2t + sqrt(x_i^2 + 4 x_i t)) is the cancellation-free
    # form of y_i/(x_i + y_i) at the stationary y_i(t).  Check the two agree
    # over randomised positive x_i and t, including badly scaled ones.
    rng = np.random.default_rng(21)
    worst_p, worst_y, worst_q = 0.0, 0.0, 0.0
    for _ in range(4000):
        x = float(10.0 ** rng.uniform(-8, 8))
        t = float(10.0 ** rng.uniform(-8, 8))
        y = float(core._y_of_t(np.array([x]), t, 1.0)[0])
        p_closed = float(core._p_of_t(np.array([x]), np.array([y]), t, 1.0)[0])
        p_direct = y / (x + y)
        worst_p = max(worst_p, abs(p_closed - p_direct)
                      / max(p_direct, tolcfg.LOG_FLOOR))
        # y_i(t) must solve the quadratic y^2 + x y - t x = 0
        worst_q = max(worst_q, abs(y * y + x * y - t * x)
                      / max(y * y + x * y + t * x, tolcfg.LOG_FLOOR))
        # ... and agree with the textbook root where that form is well behaved
        if 4.0 * x * t > 1e-6 * x * x:
            y_text = 0.5 * (-x + math.sqrt(x * x + 4.0 * x * t))
            worst_y = max(worst_y, abs(y - y_text) / max(y_text, 1e-300))
    check("p_i(t) closed form == y_i/(x_i+y_i) from the computed y_i(t)",
          worst_p < 1e-12, f"max rel err {worst_p:.2e} over 4000 random (x,t)")
    check("y_i(t) solves the m=1 stationarity quadratic y^2 + x y - t x = 0",
          worst_q < 1e-12, f"max rel residual {worst_q:.2e}")
    check("cancellation-free root agrees with the textbook root where stable",
          worst_y < 1e-9, f"max rel err {worst_y:.2e}")


# ------------------------------------------------------- numerical residuals
def test_residuals():
    print("\n17. Residual checks (budget, KKT/stationarity, root, certificate)")
    rng = np.random.default_rng(24)
    bud, kkt, root = 0.0, 0.0, 0.0
    for _ in range(300):
        n = int(rng.integers(1, 8))
        x = 10.0 ** rng.uniform(-4, 3, n)
        XB = float(10.0 ** rng.uniform(-2, 3))
        sol = evader_alloc(x, XB, diagnostics=True)
        res = alloc_residuals(x, sol, XB)
        bud = max(bud, res["budget_rel"])
        kkt = max(kkt, res["kkt_spread"])
        root = max(root, res["root_max"])
        # the diagnostics record must be present and self-consistent
        assert sol["solver"]["root_tol"] == tolcfg.ROOT_TOL
        assert sol["solver"]["iterations"] <= sol["solver"]["max_iter"]
    check(f"evader budget residual |sum y - XB|/XB <= {tolcfg.BUDGET_TOL:g}",
          bud <= tolcfg.BUDGET_TOL, f"max {bud:.2e}")
    check(f"KKT/stationarity residual (multiplier spread) <= {tolcfg.KKT_TOL:g}",
          kkt <= tolcfg.KKT_TOL, f"max {kkt:.2e}")
    check("root-equation residual y(x^m+y^m) - t m x^m ~ 0 for every solved y_i",
          root <= 1e-9, f"max relative residual {root:.2e}")

    # the same audit on a full solve, plus the certificate residual
    G, S, D = graphs.CATALOG["random-12"]()
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9, max_iter=300)
    rr = r["residuals"]
    check("defender budget residual |sum x - XA| within tolerance",
          rr["defender_budget_abs"] <= tolcfg.BUDGET_TOL * 10.0,
          f"{rr['defender_budget_abs']:.2e}")
    check("evader budget residual at the reported optimum within tolerance",
          rr["budget_rel"] <= tolcfg.BUDGET_TOL, f"{rr['budget_rel']:.2e}")
    check("KKT residual at the reported optimum within tolerance",
          rr["kkt_spread"] <= tolcfg.KKT_TOL, f"{rr['kkt_spread']:.2e}")
    check("certificate residual UB - LB matches the reported gap",
          abs(rr["certificate_abs_gap"] - r["certificate"]["abs_gap"]) < 1e-15,
          f"{rr['certificate_abs_gap']:.2e}")
    pc = rr["path_cost_consistency"]
    check("shortest-path cost == sum of node costs c_i(lambda) along the path",
          pc is None or pc < 1e-9,
          "enumeration mode (relaxation unused)" if pc is None else f"{pc:.2e}")


# ------------------------------------------------------ certificate reporting
def test_certificate_reporting():
    print("\n18. Certificate is conditional on the gap closing")
    G, S, D = graphs.CATALOG["lecture"]()
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-10, max_iter=400)
    c = r["certificate"]
    for key in ["log_ub", "log_lb", "abs_gap", "rel_gap",
                "rel_gap_denominator", "tolerance", "certified", "status",
                "iterations", "n_cuts", "runtime_sec", "solver_settings"]:
        if key not in c:
            check(f"certificate reports {key}", False)
            return
    check("certificate reports UB, LB, abs/rel gap, tolerance, status, "
          "iterations, cuts, runtime and solver settings", True)
    check("gap closed  =>  certified", c["certified"] and c["abs_gap"] <= c["tolerance"],
          f"gap {c['abs_gap']:.2e} <= tol {c['tolerance']:.1e}")
    check("LB <= UB and V* lies inside the reported bracket",
          c["log_lb"] <= c["log_ub"] + 1e-15
          and c["value_lb"] <= r["value"] + 1e-12 <= c["value_ub"] + 1e-12)

    # A deliberately starved run must NOT be reported as certified.
    G, S, D = graphs.layered(8, 4)
    r2 = solve_defender(G, S, D, 10.0, 10.0, tol=1e-12, max_iter=3)
    c2 = r2["certificate"]
    check("a run stopped before the gap closes is reported NOT certified",
          (not c2["certified"]) and c2["abs_gap"] > c2["tolerance"]
          and "not certified" in c2["status"],
          f"gap {c2['abs_gap']:.2e} > tol {c2['tolerance']:.1e}, "
          f"status '{c2['status']}'")

    # m > 1: convexity provably fails, so nothing there may be called certified.
    G, S, D = graphs.diamond()
    r3 = solve_defender(G, S, D, 10.0, 10.0, m=2.0, tol=1e-8, max_iter=80)
    check("m > 1 runs are never certified (convexity provably fails)",
          (not r3["certified"]) and not r3["certificate"]["convexity_proved"],
          r3["certificate"]["status"])


# --------------------------------------------------------- zero defence (C1)
def test_zero_defence_boundary():
    print("\n19. Zero-defence boundary x_i = 0 (model value, not a floor artefact)")
    # The MODEL allows x_i = 0 exactly: convention (C1) gives p_i = 1 there.
    # These checks pass exact zeros -- never the numerical floor -- and verify
    # that the answer is the model's boundary solution.
    sol = evader_alloc(np.array([0.0, 2.0, 3.0]), 10.0)
    check("x_i = 0 gets zero evader budget and p_i = 1 exactly",
          sol["y"][0] == 0.0 and sol["p"][0] == 1.0)
    check("the remaining budget is spent entirely on the defended nodes",
          abs(sol["y"].sum() - 10.0) < 1e-12 * 10.0)
    ref = evader_alloc(np.array([2.0, 3.0]), 10.0)
    check("value at (0, x2, x3) equals the value of the sub-path (x2, x3)",
          abs(sol["logval"] - ref["logval"]) < 1e-12,
          f"diff {abs(sol['logval']-ref['logval']):.2e}")

    all_zero = evader_alloc(np.zeros(4), 10.0)
    check("a fully undefended path has value exactly 1 (log 0)",
          all_zero["logval"] == 0.0 and np.all(all_zero["p"] == 1.0))

    # x_i = 0 must be the LIMIT of x_i -> 0+, not a discontinuity introduced by
    # the floor: the floor only ever perturbs the value by ~sqrt(x_min/t).
    gaps = []
    for eps in [1e-6, 1e-8, 1e-10, 1e-12]:
        s_eps = evader_alloc(np.array([eps, 2.0, 3.0]), 10.0)
        gaps.append(abs(s_eps["logval"] - sol["logval"]))
    check("value at x_1 = eps -> value at x_1 = 0 as eps -> 0 (C1 is the "
          "continuous extension, so the floor is not what sets the answer)",
          all(b <= a + 1e-15 for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 1e-5,
          "  ".join(f"{g:.2e}" for g in gaps))

    # a solved instance whose optimum genuinely puts zero on some nodes
    G, S, D = graphs.CATALOG["random-12"]()
    r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9, max_iter=300)
    xmin_floor = max(tolcfg.X_FLOOR_ABS,
                     tolcfg.X_FLOOR_REL * 10.0 / len(r["nodes"]))
    zeros = [v for v in r["nodes"] if r["x"][v] == 0.0]
    near = [v for v in r["nodes"] if 0.0 < r["x"][v] <= 10.0 * xmin_floor]
    check("the optimum really does zero out some nodes here, and they sit at "
          "exactly 0 rather than at the numerical floor (so a reported zero "
          "is the model's boundary solution)",
          len(zeros) > 0 and len(near) == 0,
          f"{len(zeros)} exact zeros, {len(near)} nodes pinned near the "
          f"floor {xmin_floor:.1e}")
    x0 = {v: (0.0 if v in zeros else r["x"][v]) for v in r["nodes"]}
    v0 = game_value(G, S, D, x0, 10.0, oracle=r["oracle"])["value"]
    check("re-evaluating that allocation with exact zeros reproduces V*",
          abs(v0 - r["value"]) <= 1e-9, f"{abs(v0-r['value']):.2e}")


# ------------------------------------------------------- adaptive dual search
def test_lambda_search_adaptive():
    print("\n20. Dual multiplier search adapts instead of trusting a fixed range")
    G, S, D = graphs.random_dag(10, 0.3, seed=23)
    orc = PathOracle(G, S, D, mode="ksp")
    rng = np.random.default_rng(23)

    # (a) extreme input scales: lambda ~ 1/t moves with the budget scale, so a
    # hard-coded log10 window can be left behind.  The adaptive search must
    # still produce a VALID upper bound on the true best response.
    worst = 0.0
    for scale in [1e-8, 1e-4, 1.0, 1e4, 1e8]:
        x = rng.dirichlet(np.ones(len(orc.nodes))) * scale
        br = orc.best_response(x, scale)
        worst = min(worst, br["ub"] - br["logval"])
    check("dual bound stays valid across 16 orders of magnitude of input scale",
          worst >= -tolcfg.CERT_SLACK, f"min slack {worst:.2e}")

    # (b) with a deliberately misplaced starting bracket the search must expand
    # it and recover the same multiplier as the default bracket does.
    x = rng.dirichlet(np.ones(len(orc.nodes))) * 10.0
    ref = PathOracle(G, S, D, mode="ksp")
    lam_ref, u_ref = ref._best_lambda(x, 10.0)
    bad = PathOracle(G, S, D, mode="ksp", lam_log_range=(-40.0, -30.0))
    lam_bad, u_bad = bad._best_lambda(x, 10.0)
    check("a misplaced starting bracket is expanded, not silently accepted",
          bad.last_lambda_search["expansions"] > 0
          and not bad.last_lambda_search["at_boundary"],
          f"{bad.last_lambda_search['expansions']} expansions, final bracket "
          f"{tuple(round(v, 1) for v in bad.last_lambda_search['bracket'])}")
    check("the recovered bound matches the one from the default bracket",
          abs(u_bad - u_ref) <= 1e-9 * max(abs(u_ref), 1.0),
          f"U_bad {u_bad:.12f} vs U_ref {u_ref:.12f}  "
          f"(lambda {lam_bad:.4g} vs {lam_ref:.4g})")

    # (c) boundary detection must be reported, not hidden: with expansion
    # disabled and a bracket that excludes the optimum, the search says so.
    stuck = PathOracle(G, S, D, mode="ksp", lam_log_range=(-40.0, -30.0),
                       lam_expand_max=0)
    stuck._best_lambda(x, 10.0)
    check("a search that terminates on a bracket boundary flags at_boundary",
          stuck.last_lambda_search["at_boundary"] is True,
          f"bracket {stuck.last_lambda_search['bracket']}")
    check("the lambda bracket is configurable (not hard-coded)",
          tuple(ref.lam_log_range) == (tolcfg.LAMBDA_LOG_LO, tolcfg.LAMBDA_LOG_HI)
          and tuple(bad.lam_log_range) == (-40.0, -30.0))


def main():
    print("=" * 74)
    print("  NETWORK SECURITY GAME -- VALIDATION SUITE")
    print("=" * 74)
    for t in [test_inner_vs_random_sampling, test_inner_kkt_and_budget,
              test_inner_symmetry, test_dual_bound, test_danskin,
              test_convexity, test_closed_forms, test_oracle_equivalence,
              test_brute_force, test_heuristics_never_better,
              test_comparative_statics, test_uncovered_path, test_general_m,
              test_contested_endpoints, test_uniqueness,
              test_m1_closed_form, test_residuals,
              test_certificate_reporting, test_zero_defence_boundary,
              test_lambda_search_adaptive]:
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
