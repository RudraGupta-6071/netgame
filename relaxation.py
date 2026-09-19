"""
relaxation.py -- a VALID lower bound for the m > 1 regime, where the convexity
theorem fails and the ordinary cutting-plane lower bound is not usable.

THE PROBLEM
-----------
For 0 < m <= 1, FORMULATION.md S3 proves G(x) = max_P g_P(x) is convex, which is
what makes the Kelley cuts global under-estimators and the reported UB - LB gap
an optimality certificate.  Proposition 5 shows convexity provably FAILS for
m > 1 (explicit counterexample on the diamond at m = 2), so for m > 1 the cuts
are not valid under-estimators, the lower bound is meaningless, and the solver
can only report an upper bound with no idea how far off it is.  Everything the
repository says about m > 1 is therefore "exploratory / non-certified".

THE FIX IMPLEMENTED HERE
------------------------
A valid lower bound is still obtainable, via a convexifying change of variables
plus a convex relaxation of the budget set.  Substitute

    u_i = x_i^m .

Then each path objective becomes

    h(u, y) = sum_i [ m log y_i - log(u_i + y_i^m) ] ,

and u -> -log(u + c) is convex for every c > 0, with NO restriction on m.  So

    G~(u) := max_P sup_y h(u, y)

is convex in u for EVERY m > 0: a supremum of convex functions over a set that
does not depend on u, then a finite max.  What breaks for m > 1 is not the
objective but the feasible set: the budget simplex {x >= 0, sum x_i = XA} maps to

    U = { u >= 0 : sum_i u_i^(1/m) = XA } ,

which is non-convex for m > 1.  Relax it to its convex hull.

    Lemma (the hull is a simplex).  For m >= 1,
        conv{ u >= 0 : sum_i u_i^(1/m) <= XA }  =  { u >= 0 : sum_i u_i <= XA^m }.

    Proof.  (subset) If sum_i u_i^(1/m) <= XA, put v_i = u_i^(1/m) >= 0, so
    sum_i v_i <= XA.  For m >= 1 the l_m norm is dominated by the l_1 norm,
    (sum_i v_i^m)^(1/m) <= sum_i v_i, hence sum_i u_i = sum_i v_i^m <= XA^m.
    The right-hand set is convex, so it contains the convex hull.
    (superset) Each vertex XA^m * e_j of the right-hand simplex lies in the
    left-hand set, since (XA^m)^(1/m) = XA.  A convex set containing all
    vertices of a simplex contains the simplex.  []

Minimising a convex function over a superset of the true feasible set gives a
valid lower bound:

    min_{u in hull} G~(u)   <=   min_{u in U} G~(u)   =   min_x G(x)   =   log V* .

That minimisation is an ordinary convex program -- convex objective over a
simplex -- so the same Kelley machinery applies, and its own UB/LB bracket
certifies it.  The subgradient is clean: by Danskin, at the inner optimum y*,

    d G~ / d u_i  =  -1 / (u_i + (y*_i)^m) ,

which is just the derivative of -log(u + y^m) and needs no chain rule.

WHAT THIS BUYS, AND WHAT IT DOES NOT
------------------------------------
It converts "m > 1: value reported, no idea how good" into a genuine two-sided
bracket [relaxed LB, attained UB] on log V*.  That is a WEAKER certificate than
the m <= 1 one: the bound is loose by however much the convex hull enlarges the
feasible set, and that slack does not go to zero as the solver converges.  It
does NOT restore the m <= 1 guarantee, does not prove the reported allocation is
optimal, and must never be reported as one.  A tight gap here is informative; a
loose gap means the relaxation, not necessarily the allocation, is loose.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linprog

import tolerances as tolcfg
from core import PathOracle, certificate

__all__ = ["relaxed_lower_bound", "bracket_m_gt_1"]


def relaxed_lower_bound(G, S, D, XA, XB, m=2.0, oracle=None,
                        contest_endpoints=False, max_iter=200, tol=None,
                        max_cuts=400, verbose=False):
    """Valid LOWER bound on log V* via the u = x^m convexification (see module doc).

    Returns a dict with `log_lb` (the rigorous lower bound on log V*), the
    relaxation's own internal certificate, and the u-space minimiser.  Valid for
    every m >= 1; for m <= 1 the ordinary solver already gives a tighter bound,
    so this is intended for the m > 1 regime.
    """
    tol = tolcfg.GAP_TOL if tol is None else tol
    t0 = time.time()
    if m < 1.0:
        raise ValueError("relaxed_lower_bound assumes m >= 1 (the hull lemma "
                         "needs the l_m <= l_1 norm inequality); for m <= 1 use "
                         "core.solve_defender, whose bound is already valid")
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    nodes = oracle.nodes
    n = len(nodes)
    if n == 0:
        return {"log_lb": 0.0, "value_lb": 1.0, "n": 0, "iterations": 0,
                "time": 0.0, "valid": True}

    U_budget = XA ** m                      # the hull's budget, see the Lemma
    umin = max(tolcfg.X_FLOOR_ABS, tolcfg.X_FLOOR_REL * U_budget / n)

    def eval_at(u):
        """G~(u) and a subgradient in u-space, via the existing x-space oracle."""
        u = np.maximum(np.asarray(u, float), 0.0)
        x = u ** (1.0 / m)                  # back to x-space for the oracle
        br = oracle.best_response(x, XB)
        y = np.zeros(n)
        for v, yy in zip(br["interior"], br["y"]):
            if v in oracle.index:
                y[oracle.index[v]] = yy
        # dG~/du_i = -1/(u_i + y_i^m) on the maximising path, 0 elsewhere
        s = np.zeros(n)
        for v in br["interior"]:
            if v in oracle.index:
                j = oracle.index[v]
                s[j] = -1.0 / max(u[j] + y[j] ** m, tolcfg.LOG_FLOOR)
        return br["logval"], s

    u = np.full(n, U_budget / n)
    A_ub, b_ub = [], []
    LB, UB = -np.inf, 0.0
    history = []

    for it in range(1, max_iter + 1):
        val, s = eval_at(u)
        UB = min(UB, val)
        b = val - float(s @ u)
        A_ub.append(np.append(s, -1.0))
        b_ub.append(-b)
        if len(A_ub) > max_cuts:
            A_ub, b_ub = A_ub[-max_cuts:], b_ub[-max_cuts:]

        c = np.append(np.zeros(n), 1.0)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.append(np.ones(n), 0.0).reshape(1, -1),
                      b_eq=[U_budget],
                      bounds=[(0.0, U_budget)] * n + [(-1e9, 0.0)],
                      method="highs",
                      options={"primal_feasibility_tolerance": tolcfg.LP_FEAS_TOL,
                               "dual_feasibility_tolerance": tolcfg.LP_FEAS_TOL})
        if not res.success:
            break
        LB = max(LB, float(res.x[-1]))
        history.append({"iter": it, "lb": LB, "ub": UB, "gap": UB - LB})
        if verbose:
            print(f"    relax it {it:3d}  LB={LB: .8f}  UB={UB: .8f}")
        if UB - LB <= tol:
            break
        u = np.maximum(res.x[:n], umin)
        u *= U_budget / u.sum()

    cert = certificate(min(LB, UB), UB, tol, iterations=len(history),
                       n_cuts=len(A_ub), runtime=time.time() - t0,
                       settings={"m": m, "XA": XA, "XB": XB,
                                 "hull_budget_XA^m": U_budget,
                                 "relaxation": "u = x^m, budget set relaxed to "
                                               "its convex hull {sum u <= XA^m}"})
    return {"log_lb": float(min(LB, UB)), "value_lb": math.exp(min(LB, UB)),
            "u": u, "x_equivalent": oracle.to_dict(np.maximum(u, 0.0) ** (1.0 / m)),
            "relaxation_certificate": cert, "iterations": len(history),
            "n_cuts": len(A_ub), "time": time.time() - t0, "n": n,
            "valid": True,
            "note": "rigorous LOWER bound on log V*; the relaxation gap does "
                    "not vanish with convergence, so a loose bracket may mean a "
                    "loose relaxation rather than a suboptimal allocation"}


def bracket_m_gt_1(G, S, D, XA, XB, m=2.0, contest_endpoints=False,
                   max_iter=200, tol=None, verbose=False):
    """Two-sided bracket on V* for m > 1: attained UB, relaxed LB.

    The upper bound is the value of the best allocation the (non-certified)
    solver finds -- rigorous as an upper bound simply because it is attained at
    a feasible point.  The lower bound is `relaxed_lower_bound`.  Together they
    bracket V* even though convexity fails, which is strictly more than the
    "exploratory, no bound" status m > 1 had before.

    The returned `relative_slack` is (V_ub - V_lb)/V_lb: how much room the
    bracket leaves, NOT a claim that the allocation is that close to optimal.
    """
    from core import solve_defender

    tol = tolcfg.GAP_TOL if tol is None else tol
    oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    up = solve_defender(G, S, D, XA, XB, m=m, tol=tol, max_iter=max_iter,
                        oracle=oracle, contest_endpoints=contest_endpoints)
    lo = relaxed_lower_bound(G, S, D, XA, XB, m=m, oracle=oracle,
                             contest_endpoints=contest_endpoints,
                             max_iter=max_iter, tol=tol, verbose=verbose)
    log_ub = up["log_ub"]
    log_lb = lo["log_lb"]
    v_ub, v_lb = math.exp(log_ub), math.exp(log_lb)
    return {"m": m, "value_ub": v_ub, "value_lb": v_lb,
            "log_ub": log_ub, "log_lb": log_lb,
            "abs_gap": max(log_ub - log_lb, 0.0),
            "relative_slack": (v_ub - v_lb) / max(v_lb, 1e-300),
            "x": up["x"], "bracket_valid": log_lb <= log_ub + 1e-9,
            "upper": up, "lower": lo,
            "certified_optimal": False,
            "status": ("valid two-sided bracket via convex relaxation; NOT an "
                       "optimality certificate (convexity fails for m > 1, so "
                       "the allocation is not proved optimal)")}
