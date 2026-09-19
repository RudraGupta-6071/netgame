"""
core.py -- Network Security Game on a directed graph (Tullock path contest).
============================================================================

MODEL
-----
Directed graph G=(V,E), source S, terminus D.

  Defender A :  x >= 0 on the contested nodes,  sum_i x_i = XA
  Evader   B :  chooses an S-D path P and y >= 0 on P, sum_{i in P} y_i = XB

  Node contest (ratio-form / Tullock CSF with contest intensity m > 0):

        p_i = y_i^m / (x_i^m + y_i^m)          (probability B gets through i)

  B must survive every contested node of P (series system):

        U_B(P,x,y) = prod_{i in P} p_i

  Stackelberg / minimax problem solved here:

        V* = min_x max_P max_y  U_B(P,x,y)

Everything is computed in LOG space,  g(x) = log U_B, which is where the
problem is convex in x (see FORMULATION.md S3).

CONVENTIONS
-----------
* x_i = 0  =>  node i is undefended, p_i = 1 and B spends y_i = 0 there.
  This is the continuous extension of the CSF (p_i -> 1 as x_i -> 0+ along
  the evader's optimal response), so the value function stays continuous.
  x_i = 0 is a legal point of the MODEL.  Any small positive floor appearing
  below (tolerances.X_FLOOR_REL) is a NUMERICAL safeguard against the
  1/(x_i + y_i) singularity of the subgradient, never a model constraint;
  the solver also evaluates exactly-sparse points, so a reported optimum with
  x_i = 0 is the model's boundary solution rather than an artefact of the
  floor.
* Source and terminus are uncontested by default (contest_endpoints=False).
  Set contest_endpoints=True to force S and D onto every path.  ONE convention
  governs both the contest product and the budget allocation: a node is
  contested (enters the product over P) exactly when it is eligible for
  defender budget and for the evader's split.  `relevant_nodes` and
  `path_interior` are the single source of truth for that set.

KEY FACTS USED (proved in FORMULATION.md)
-----------------------------------------
P1  The inner problem is strictly concave in y on the nodes with x_i > 0, so
    the interior response there is unique; it is solved as a KKT system
    reduced to one monotone scalar root.  Nodes with x_i = 0 are handled by
    the convention above (p_i = 1, y_i = 0), not by the KKT system, and the
    strict-concavity/uniqueness statement is not extended to them.
P2  g_P(x) = max_y h(x,y) is CONVEX in x for 0 < m <= 1, because h(.,y) is
    convex in x for every fixed y and a pointwise sup of convex functions is
    convex.  G(x) = max_P g_P(x) is therefore convex too.  This is convexity,
    NOT strict convexity: g_P ignores the coordinates off P, so G is flat in
    those directions and uniqueness of the minimiser does not follow.
P3  Danskin: s_i = d g_P / d x_i = -m x_i^(m-1) / (x_i^m + y_i*^m).  This is
    asserted where the inner maximiser y* is unique and strictly positive on
    the active set and x_i > 0; it is not claimed at the nonsmooth boundary.
P4  Any FEASIBLE y (not only the optimal one) yields a VALID cut, because
    g_P(x) >= h(x,y) >= tangent of h(.,y).  Lower bounds are therefore robust
    to inner-solver error.
P5  Lagrangian/weak duality: for every lam > 0,
        g_P(x) <= lam*XB + sum_{i in P} psi(x_i, lam),
    with equality at the optimal lam.  psi is separable => the bound is a
    SHORTEST-PATH problem => certified stopping rule for path generation.

WHAT "CERTIFIED" MEANS HERE
---------------------------
`solve_defender` returns a rigorous bracket [LB, UB] on log V*.  A run carries
a global optimality certificate ONLY when UB - LB closes within the requested
tolerance.  The returned dict reports UB, LB, the absolute gap, the relative
gap (with its denominator stated), the requested tolerance, the iteration and
cut counts, the runtime and a boolean `certified`.  Runs that stop at the
iteration cap with a nonzero gap are reported as NOT certified.  The
certificate is meaningful only because the implemented upper and lower bounds
refer to the same optimization problem, and it is stated net of the numerical
tolerances listed in `tolerances.py`.

The path-selection subproblem is closely related to maximum-reliability path
problems known to be NP-hard; consequently we use a certified path-generation
procedure (P5) rather than claiming a polynomial-time exact oracle.  No
reduction establishing NP-hardness of the exact continuous Tullock-budget
model solved here is supplied, and none is claimed.
"""

from __future__ import annotations

import math
import time
from itertools import islice

import networkx as nx
import numpy as np
from scipy.optimize import linprog

import tolerances as tolcfg

__all__ = [
    "relevant_nodes", "enumerate_paths", "path_interior",
    "evader_alloc", "alloc_residuals", "psi", "dual_bound_path",
    "PathOracle", "game_value", "solve_defender", "certificate",
    "heuristic_allocation", "brute_force_grid", "sparsify_alloc",
]

# ----------------------------------------------------------------------------
# 0.  Graph helpers
# ----------------------------------------------------------------------------


def relevant_nodes(G, S, D, contest_endpoints=False):
    """Nodes lying on at least one S-D path (excluding S, D unless asked).

    Allocating defender budget anywhere else is provably wasted: such a node
    belongs to no S-D path, so it never enters any payoff.
    """
    fwd = nx.descendants(G, S) | {S}
    bwd = nx.ancestors(G, D) | {D}
    core = fwd & bwd
    if not contest_endpoints:
        core = core - {S, D}
    return sorted(core, key=str)


def path_interior(path, S, D, contest_endpoints=False):
    """Contested nodes of a path."""
    if contest_endpoints:
        return list(path)
    return [v for v in path if v not in (S, D)]


def enumerate_paths(G, S, D, limit=200000):
    """All simple S-D paths (truncated at `limit`).  Returns (paths, complete)."""
    it = nx.all_simple_paths(G, S, D)
    paths = list(islice(it, limit + 1))
    complete = len(paths) <= limit
    return paths[:limit], complete


# ----------------------------------------------------------------------------
# 1.  Inner problem: evader's budget split on a FIXED path
#     (solved NUMERICALLY to tolerance -- not a symbolic/exact solution)
# ----------------------------------------------------------------------------


def _y_of_t(xv, t, m=1.0, root_tol=None, max_iter=None):
    """Stationary y_i as a function of t = 1/lambda.

    KKT:  m x^m / (y (x^m + y^m)) = lambda   <=>   y (x^m + y^m) = t m x^m.
    The left side is strictly increasing in y, so the root is unique.

    For m = 1 this is the quadratic y^2 + x y - t x = 0, whose positive root is
    returned in the cancellation-free form  y = 2xt / (x + sqrt(x^2 + 4xt))
    (algebraically identical to (-x + sqrt(x^2 + 4xt))/2, but stable when
    4xt << x^2).  That branch is a closed form evaluated in floating point.

    For m != 1 the root is found by safeguarded Newton, i.e. NUMERICALLY, to
    the relative tolerance `root_tol`, or until `max_iter` iterations have been
    taken.  The stopping criterion is the relative step test
    |y_new - y| <= root_tol * y.
    """
    root_tol = tolcfg.ROOT_TOL if root_tol is None else root_tol
    max_iter = tolcfg.ROOT_MAX_ITER if max_iter is None else max_iter
    xv = np.asarray(xv, dtype=float)
    if m == 1.0:
        return 2.0 * xv * t / (xv + np.sqrt(xv * xv + 4.0 * xv * t))
    xm = xv ** m
    target = t * m * xm
    # F(y) = y*x^m + y^(m+1) - target is increasing and convex on y > 0, so
    # Newton started from any point with F <= 0 converges monotonically.
    y = 0.5 * np.minimum(target / np.maximum(xm, tolcfg.LOG_FLOOR),
                         target ** (1.0 / (m + 1.0)))
    y = np.maximum(y, tolcfg.LOG_FLOOR)
    for _ in range(max_iter):
        F = y * xm + y ** (m + 1.0) - target
        dF = xm + (m + 1.0) * y ** m
        step = F / dF
        y_new = np.maximum(y - step, 0.5 * y)          # safeguard
        if np.all(np.abs(y_new - y)
                  <= root_tol * np.maximum(y, tolcfg.LOG_FLOOR)):
            y = y_new
            break
        y = y_new
    return y


def _root_residual(xv, yv, t, m=1.0):
    """Relative residual of the stationarity root  y (x^m + y^m) - t m x^m = 0.

    Reported per node so that a numerically solved y_i can be audited rather
    than assumed correct.
    """
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    xm = xv ** m
    lhs = yv * (xm + yv ** m)
    rhs = t * m * xm
    scale = np.maximum(np.abs(lhs) + np.abs(rhs), tolcfg.LOG_FLOOR)
    return np.abs(lhs - rhs) / scale


def _p_of_t(xv, yv, t, m=1.0):
    """p_i at the stationary point (cancellation-free closed form for m = 1).

    For m = 1 this is  p_i(t) = 2t / (x_i + 2t + sqrt(x_i^2 + 4 x_i t)),
    algebraically the same as y_i/(x_i + y_i) evaluated at the stationary
    y_i(t).  `tests.test_m1_closed_form` checks numerically that the two agree
    over randomised positive x_i and t.
    """
    xv = np.asarray(xv, dtype=float)
    if m == 1.0:
        return 2.0 * t / (xv + 2.0 * t + np.sqrt(xv * xv + 4.0 * xv * t))
    ym = yv ** m
    return ym / (xv ** m + ym)


def alloc_residuals(x_path, sol, XB, m=1.0):
    """Audit a returned evader allocation against the conditions it should meet.

    Returns a dict with

      budget_abs / budget_rel   |sum y_i - XB| and its relative form
      kkt_spread                relative spread of the stationarity multipliers
                                m x_i^m / (y_i (x_i^m + y_i^m)) over the ACTIVE
                                nodes (x_i > 0); this is the KKT/stationarity
                                residual, and it is 0 at an exact solution
      root_max                  max relative residual of the scalar root
                                y (x^m + y^m) = t m x^m over the active nodes
      n_active                  number of nodes with x_i > 0

    Inactive nodes (x_i = 0) are excluded deliberately: there the model sets
    p_i = 1 and y_i = 0 by convention (C1) and no stationarity condition is
    claimed, so including them would report a meaningless residual.
    """
    x_path = np.asarray(x_path, dtype=float)
    y = np.asarray(sol["y"], dtype=float)
    act = x_path > 0.0
    out = {"budget_abs": float(abs(y.sum() - XB)),
           "budget_rel": float(abs(y.sum() - XB) / max(abs(XB), 1e-300)),
           "n_active": int(act.sum()), "kkt_spread": 0.0, "root_max": 0.0}
    if act.sum() == 0:
        return out
    xa = x_path[act]
    ya = np.maximum(y[act], tolcfg.Y_FLOOR)
    lam = m * xa ** m / (ya * (xa ** m + ya ** m))
    out["kkt_spread"] = float((lam.max() - lam.min()) / max(lam.mean(), 1e-300))
    t = sol.get("t", 0.0)
    if t > 0:
        out["root_max"] = float(_root_residual(xa, ya, t, m).max())
    return out


def evader_alloc(x_path, XB, m=1.0, iters=None, root_tol=None,
                 diagnostics=False):
    """High-precision NUMERICAL solution of
           max_y sum log p_i   s.t.  sum y_i = XB,  y >= 0.

    The KKT system is reduced to the single monotone equation
    S(t) = sum_i y_i(t) = XB, that root is solved numerically (safeguarded
    Newton inside a maintained bracket), and the resulting y is then rescaled
    to satisfy the budget in floating point.  The returned vector is therefore
    a floating-point solution accurate to the tolerances configured in
    `tolerances.py` -- it is numerically solved to tolerance, not an exact
    symbolic solution.  Use `alloc_residuals`, or `diagnostics=True`, to audit
    it rather than assuming it.

    Parameters
    ----------
    x_path : array of defender allocations on the contested nodes of one path
    XB     : evader budget
    m      : contest intensity
    iters  : cap on iterations of the outer root S(t) = XB
             (default tolerances.ROOT_MAX_ITER)
    root_tol : relative stopping tolerance (default tolerances.ROOT_TOL)
    diagnostics : when True, add a `solver` sub-dict recording the root-solving
             tolerance, the iteration cap, the iterations actually used, the
             stopping criterion, the final root residual |S(t) - XB|/XB before
             rescaling, and the post-rescale budget and KKT residuals.

    Returns dict with keys y, p, logval, lam (= multiplier), t (= 1/lam), plus
    `solver` when diagnostics=True.  `y` is budget feasible to floating-point
    accuracy and logval is computed from that same `y`, so logval is always an
    attainable (lower) value -- see P4.
    """
    iters = tolcfg.ROOT_MAX_ITER if iters is None else iters
    root_tol = tolcfg.ROOT_TOL if root_tol is None else root_tol
    x_path = np.asarray(x_path, dtype=float)
    n = x_path.size
    y = np.zeros(n)
    p = np.ones(n)

    def _trivial(res):
        if diagnostics:
            res["solver"] = {"root_tol": root_tol, "max_iter": iters,
                             "iterations": 0,
                             "stopping_criterion": "trivial (no active node)",
                             "root_residual_rel": 0.0,
                             "budget_residual_abs": 0.0,
                             "budget_residual_rel": 0.0,
                             "kkt_spread": 0.0, "root_max": 0.0}
        return res

    if n == 0 or XB <= 0:
        return _trivial({"y": y, "p": p,
                         "logval": 0.0 if XB > 0 or n == 0 else -np.inf,
                         "lam": np.inf, "t": 0.0})

    act = x_path > 0.0
    if not act.any():                          # whole path undefended
        return _trivial({"y": y, "p": p, "logval": 0.0, "lam": np.inf,
                         "t": 0.0})

    xa = x_path[act]

    # --- bracket t so that sum y(t) = XB (sum y is strictly increasing in t)
    t_lo, t_hi = 1e-30, 1.0
    for _ in range(400):
        if _y_of_t(xa, t_hi, m, root_tol, iters).sum() >= XB:
            break
        t_hi *= 4.0
    for _ in range(400):
        if _y_of_t(xa, t_lo, m, root_tol, iters).sum() <= XB:
            break
        t_lo /= 4.0

    # --- safeguarded Newton on  S(t) = sum_i y_i(t) = XB
    #     S'(t) = sum_i m x^m / (x^m + (m+1) y^m)      (analytic derivative)
    #     Stopping criterion: |S(t) - XB| <= root_tol * XB, else `iters` steps.
    t = math.sqrt(t_lo * t_hi)
    xam = xa ** m
    used = 0
    stop = f"iteration cap ({iters}) reached"
    for _ in range(iters):
        used += 1
        ya = _y_of_t(xa, t, m, root_tol, iters)
        S = ya.sum()
        if abs(S - XB) <= root_tol * XB:
            stop = f"|S(t) - XB| <= root_tol * XB (root_tol = {root_tol:g})"
            break
        if S < XB:
            t_lo = t
        else:
            t_hi = t
        dS = (m * xam / (xam + (m + 1.0) * ya ** m)).sum()
        t_new = t + (XB - S) / dS if dS > 0 else math.sqrt(t_lo * t_hi)
        if not (t_lo < t_new < t_hi):
            t_new = math.sqrt(t_lo * t_hi)
        t = t_new

    ya = _y_of_t(xa, t, m, root_tol, iters)
    root_res = float(abs(ya.sum() - XB) / max(XB, 1e-300))
    s = ya.sum()
    if s > 0:
        # Rescaling to hit the budget exactly in floating point is what makes
        # the answer a numerical solution rather than a solution of the KKT
        # system alone; the residuals below quantify what that costs.
        ya = ya * (XB / s)
    ya = np.maximum(ya, tolcfg.Y_FLOOR)
    pa = (ya ** m) / (xa ** m + ya ** m)

    y[act] = ya
    p[act] = pa
    out = {"y": y, "p": p, "logval": float(np.log(p).sum()),
           "lam": 1.0 / t, "t": t}
    if diagnostics:
        res = alloc_residuals(x_path, out, XB, m)
        out["solver"] = {"root_tol": root_tol, "max_iter": iters,
                         "iterations": used, "stopping_criterion": stop,
                         "root_residual_rel": root_res,
                         "budget_residual_abs": res["budget_abs"],
                         "budget_residual_rel": res["budget_rel"],
                         "kkt_spread": res["kkt_spread"],
                         "root_max": res["root_max"]}
    return out


def cut_from(x_path, y, m=1.0):
    """Danskin subgradient of g_P at x_path (P3), for any budget-feasible y.

    Assumptions under which this is the derivative of g_P.  Danskin's theorem
    gives  d g_P / d x_i = -m x_i^(m-1) / (x_i^m + y_i*^m)  (= -1/(x_i + y_i*)
    at m = 1) when the inner maximiser y* is UNIQUE and the objective is
    differentiable in x at the point in question -- i.e. at x_i > 0, with y*
    strictly positive and interior on the active set.  At x_i = 0 the model's
    convention (C1) takes over, g_P is nonsmooth there, and the expression
    below is NOT claimed to be the derivative: it is clamped by the floors and
    the solver never builds cuts at exactly-sparse points (see
    `solve_defender`).  For any feasible (not necessarily optimal) y the
    returned vector is still a valid subgradient of h(., y) <= g_P, which is
    all the cutting-plane lower bound needs (P4).
    """
    x_path = np.asarray(x_path, dtype=float)
    y = np.asarray(y, dtype=float)
    if m == 1.0:
        denom = x_path + y
    else:
        denom = ((x_path ** m + y ** m)
                 / np.maximum(m * x_path ** (m - 1.0), tolcfg.LOG_FLOOR))
    return -1.0 / np.maximum(denom, tolcfg.LOG_FLOOR)


# ----------------------------------------------------------------------------
# 2.  Lagrangian relaxation of the inner problem  (separable => shortest path)
# ----------------------------------------------------------------------------


def psi(xv, lam, m=1.0):
    """psi(x,lam) = max_{y>=0} [ log(y^m/(x^m+y^m)) - lam y ]   (<= 0, = 0 if x = 0)."""
    xv = np.asarray(xv, dtype=float)
    out = np.zeros_like(xv)
    act = xv > 0.0
    if act.any():
        xa = xv[act]
        t = 1.0 / lam
        ya = _y_of_t(xa, t, m)
        pa = _p_of_t(xa, ya, t, m)
        out[act] = np.log(np.maximum(pa, tolcfg.LOG_FLOOR)) - lam * ya
    return out


def dual_bound_path(x_path, XB, lam, m=1.0):
    """Valid UPPER bound on g_P(x) for any lam > 0 (P5); tight at lam*."""
    return lam * XB + float(psi(x_path, lam, m).sum())


# ----------------------------------------------------------------------------
# 3.  Evader best response over paths
# ----------------------------------------------------------------------------


class PathOracle:
    """Evader best-response engine.

    mode = 'enum' : enumerates every simple S-D path, so the maximisation over
                    paths is complete by construction (the per-path allocations
                    are still numerical).
    mode = 'ksp'  : complete *with a certificate*.  Paths are generated in
                    increasing Lagrangian cost order (Yen's algorithm); since
                    g_P <= lam*XB - c_lam(P) and c_lam is non-decreasing along
                    the enumeration, the search can stop as soon as
                    lam*XB - c_lam(P_k) <= best value found.
    mode = 'auto' : 'enum' when the path count is small, else 'ksp'.
    """

    def __init__(self, G, S, D, m=1.0, contest_endpoints=False,
                 mode="auto", enum_limit=20000, ksp_limit=4000,
                 lam_log_range=None, lam_expand_max=None):
        self.G, self.S, self.D, self.m = G, S, D, m
        self.contest_endpoints = contest_endpoints
        self.ksp_limit = ksp_limit
        # Search bracket on log10(lambda).  lambda > 0 carries no upper bound
        # mathematically, so this is only a STARTING bracket: `_best_lambda`
        # expands it adaptively whenever the minimiser lands on an endpoint.
        self.lam_log_range = tuple(lam_log_range) if lam_log_range is not None \
            else (tolcfg.LAMBDA_LOG_LO, tolcfg.LAMBDA_LOG_HI)
        self.lam_expand_max = (tolcfg.LAMBDA_EXPAND_MAX
                               if lam_expand_max is None else lam_expand_max)
        #: diagnostics from the most recent dual search (see `_best_lambda`)
        self.last_lambda_search = None
        self.nodes = relevant_nodes(G, S, D, contest_endpoints)
        self.index = {v: i for i, v in enumerate(self.nodes)}

        self.paths, complete = enumerate_paths(G, S, D, limit=enum_limit)
        self.n_paths_known = len(self.paths)
        self.paths_complete = complete
        if mode == "auto":
            mode = "enum" if complete else "ksp"
        if mode == "enum" and not complete:
            raise ValueError("too many S-D paths for mode='enum'")
        self.mode = mode
        self.interiors = [path_interior(p, S, D, contest_endpoints)
                          for p in self.paths] if complete else None

        # trivial-value detection: a path with no contested node
        self.trivial = complete and any(len(q) == 0 for q in (self.interiors or []))

        # working copy for weighted shortest paths
        self.H = nx.DiGraph()
        self.H.add_nodes_from(G.nodes())
        self.H.add_edges_from(G.edges())

    # -- vector helpers ------------------------------------------------------
    def to_vec(self, x_dict):
        return np.array([float(x_dict.get(v, 0.0)) for v in self.nodes])

    def to_dict(self, x_vec):
        return {v: float(x_vec[i]) for i, v in enumerate(self.nodes)}

    def _sub(self, x_vec, nodes):
        return np.array([x_vec[self.index[v]] if v in self.index else 0.0
                         for v in nodes])

    # -- Lagrangian shortest path -------------------------------------------
    def _set_weights(self, x_vec, lam):
        """Put each node's Lagrangian cost on its INCOMING edges.

        A path's cost is then the sum over its nodes -- except the source,
        which has no incoming edge on an S-D path.  `self._src_cost` carries
        that missing term; it must be added to every path cost.  It is zero
        unless contest_endpoints=True, in which case omitting it would leave
        the dual bound valid but uniformly loose (S lies on every S-D path,
        so the omission is a constant shift that cannot change which path is
        shortest, but does slacken the bound and mis-locate the optimal lam).
        """
        cost = {}
        c = -psi(x_vec, lam, self.m)            # >= 0
        for i, v in enumerate(self.nodes):
            cost[v] = float(c[i])
        for u, v in self.H.edges():
            self.H[u][v]["w"] = cost.get(v, 0.0)
        self._src_cost = cost.get(self.S, 0.0)
        return cost

    def _dual_at(self, x_vec, lam):
        """U(lam) = lam*XB - min_P c_lam(P);  convex in lam, >= max_P g_P(x)."""
        self._set_weights(x_vec, lam)
        length, path = nx.single_source_dijkstra(self.H, self.S, self.D, weight="w")
        return length + self._src_cost, path

    def _U(self, x_vec, log_lam, XB):
        """U(lam) = lam*XB - SP(lam) at lam = 10**log_lam (convex in lam)."""
        lam = 10.0 ** log_lam
        return lam * XB - self._dual_at(x_vec, lam)[0]

    def _best_lambda(self, x_vec, XB, grid=None, golden=None):
        """Minimise U(lam) = lam*XB - SP(lam) over lam > 0 (U is convex).

        lambda ranges over (0, infinity); no finite bracket is implied by the
        mathematics.  A fixed window on log10(lambda) can therefore miss the
        useful multiplier entirely at extreme input scales, which shows up as a
        loose upper bound rather than as an error.  This routine starts from
        the configurable window `self.lam_log_range` and EXPANDS it adaptively:
        while the discrete minimiser sits on an endpoint, that endpoint is
        pushed outward (doubling the window width, up to `lam_expand_max`
        rounds) until the minimiser is interior.

        `self.last_lambda_search` records the final bracket, whether the search
        was still at a boundary when it stopped, and how many expansions were
        used, so a boundary-limited bound can be detected instead of silently
        accepted.
        """
        grid = tolcfg.LAMBDA_GRID if grid is None else grid
        golden = tolcfg.LAMBDA_GOLDEN_ITERS if golden is None else golden
        lo, hi = self.lam_log_range            # log10 lam
        expansions, at_boundary = 0, False

        for _ in range(self.lam_expand_max + 1):
            ls = np.linspace(lo, hi, grid)
            vals = [self._U(x_vec, L, XB) for L in ls]
            k = int(np.argmin(vals))
            if k == 0:                          # optimum pushed against lo
                width = hi - lo
                lo, hi = lo - width, ls[min(1, grid - 1)]
                expansions += 1
                at_boundary = True
                continue
            if k == grid - 1:                   # optimum pushed against hi
                width = hi - lo
                lo, hi = ls[max(grid - 2, 0)], hi + width
                expansions += 1
                at_boundary = True
                continue
            at_boundary = False
            break

        a = ls[max(k - 1, 0)]
        b = ls[min(k + 1, grid - 1)]
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c_, d_ = b - gr * (b - a), a + gr * (b - a)
        fc = self._U(x_vec, c_, XB)
        fd = self._U(x_vec, d_, XB)
        for _ in range(golden):
            if fc < fd:
                b, d_, fd = d_, c_, fc
                c_ = b - gr * (b - a)
                fc = self._U(x_vec, c_, XB)
            else:
                a, c_, fc = c_, d_, fd
                d_ = a + gr * (b - a)
                fd = self._U(x_vec, d_, XB)
        log_lam = 0.5 * (a + b)
        lam = 10.0 ** log_lam
        self.last_lambda_search = {
            "log10_lambda": float(log_lam),
            "bracket": (float(lo), float(hi)),
            "initial_bracket": tuple(float(v) for v in self.lam_log_range),
            "expansions": expansions,
            "at_boundary": bool(at_boundary),
            "grid": grid, "golden_iters": golden,
        }
        return lam, self._U(x_vec, log_lam, XB)

    # -- best response -------------------------------------------------------
    def best_response(self, x_vec, XB, pool=1):
        """Evader best response at x.

        Returns dict with path / interior / y / logval (attainable value, a
        LOWER bound on G) / ub (rigorous UPPER bound on G) / certified /
        n_evaluated, plus `pool`: up to `pool` near-binding paths, each of
        which yields an extra valid cut for the master problem.

        `certified` here means the path-generation stopping rule fired, i.e.
        no unexamined path can beat the incumbent to within
        tolerances.CERT_SLACK.  It is a statement about the inner maximisation
        over paths at this x, NOT about global optimality of x -- that is the
        UB/LB gap reported by `solve_defender`.

        In 'ksp' mode the result also carries `lambda_search`, the diagnostics
        of the dual-multiplier search (final bracket, expansions used, and
        whether the search terminated against a bracket boundary).
        """
        x_vec = np.asarray(x_vec, dtype=float)
        cands = []

        def keep(path, inter, sol):
            cands.append({"path": path, "interior": inter, "y": sol["y"],
                          "logval": sol["logval"], "lam": sol["lam"]})

        if self.mode == "enum":
            for path, inter in zip(self.paths, self.interiors):
                keep(path, inter, evader_alloc(self._sub(x_vec, inter), XB, self.m))
            cands.sort(key=lambda d: -d["logval"])
            cands = cands[:max(pool, 1)]
            best = dict(cands[0])
            best["ub"] = best["logval"]
            best["certified"] = True
            best["n_evaluated"] = len(self.paths)
            best["pool"] = cands
            best["lambda_search"] = None
            return best

        # --- certified k-shortest-path generation
        lam, _ = self._best_lambda(x_vec, XB)
        self._set_weights(x_vec, lam)
        best_val, k, ub = -np.inf, 0, 0.0
        for path in nx.shortest_simple_paths(self.H, self.S, self.D, weight="w"):
            k += 1
            cost = (sum(self.H[u][v]["w"] for u, v in zip(path[:-1], path[1:]))
                    + self._src_cost)          # source has no incoming edge
            bound_rest = lam * XB - cost       # bound for this and ALL later paths
            if cands and bound_rest <= best_val + tolcfg.CERT_SLACK * 1e-3:
                ub = best_val                  # certified: nothing left can beat it
                break
            inter = path_interior(path, self.S, self.D, self.contest_endpoints)
            sol = evader_alloc(self._sub(x_vec, inter), XB, self.m)
            keep(path, inter, sol)
            best_val = max(best_val, sol["logval"])
            ub = max(best_val, bound_rest)
            if k >= self.ksp_limit or len(cands) >= max(pool, 1) * 8:
                break
        cands.sort(key=lambda d: -d["logval"])
        cands = cands[:max(pool, 1)]
        best = dict(cands[0])
        best["ub"] = min(ub, 0.0)
        best["certified"] = bool(best["ub"] <= best["logval"] + tolcfg.CERT_SLACK)
        best["n_evaluated"] = k
        best["pool"] = cands
        best["lambda_search"] = dict(self.last_lambda_search or {})
        return best


def game_value(G, S, D, x, XB, m=1.0, contest_endpoints=False, oracle=None,
               mode="auto"):
    """Evaluate max_P max_y U_B for a GIVEN defender allocation x (dict or vec)."""
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints,
                            mode=mode)
    x_vec = oracle.to_vec(x) if isinstance(x, dict) else np.asarray(x, float)
    br = oracle.best_response(x_vec, XB)
    br["value"] = math.exp(br["logval"])
    br["value_ub"] = math.exp(br["ub"])
    return br


# ----------------------------------------------------------------------------
# 4.  Defender problem: convex cutting-plane (Kelley) method with certificates
# ----------------------------------------------------------------------------


def certificate(log_lb, log_ub, tol, iterations=None, n_cuts=None,
                runtime=None, settings=None):
    """Assemble the optimality-certificate record for one solve.

    Everything a reader needs to check the claim, in one place:

      log_ub / log_lb     the bracket in LOG space
      value_ub / value_lb the same bracket on V* itself
      abs_gap             log_ub - log_lb, clipped at 0 (see `abs_gap_raw`)
      rel_gap             abs_gap / max(|log_ub|, 1); the denominator is stated
                          explicitly because a relative gap is meaningless
                          without it.  `rel_gap_value` is the corresponding
                          multiplicative gap on V* itself, expm1(abs_gap),
                          i.e. (V_ub - V_lb) / V_lb.
      tolerance           the requested tolerance the gap is compared against
      certified           True IFF abs_gap <= tolerance.  A result must not be
                          described as certified when this is False.
      status              "certified" or "not certified (gap > tolerance)"

    The certificate is valid only because the upper and lower bounds refer to
    the same optimization problem (same graph, budgets, m and node convention),
    and it is stated net of the numerical tolerances in `tolerances.py`: the LP
    lower bound is solved to LP_FEAS_TOL and the inner allocations to ROOT_TOL,
    so a gap at or below those scales is at the resolution limit of the
    arithmetic, not a sharper claim.
    """
    gap_raw = log_ub - log_lb
    gap = max(gap_raw, 0.0)
    ok = bool(gap <= tol)
    rec = {"log_ub": float(log_ub), "log_lb": float(log_lb),
           "value_ub": math.exp(log_ub), "value_lb": math.exp(log_lb),
           "abs_gap": float(gap), "abs_gap_raw": float(gap_raw),
           "rel_gap": float(gap / max(abs(log_ub), 1.0)),
           "rel_gap_denominator": "max(|log_ub|, 1)",
           "rel_gap_value": float(math.expm1(gap)),
           "rel_gap_value_denominator": "V_lb  (so rel_gap_value = V_ub/V_lb - 1)",
           "tolerance": float(tol),
           "certified": ok,
           "status": "certified" if ok else "not certified (gap > tolerance)"}
    if iterations is not None:
        rec["iterations"] = int(iterations)
    if n_cuts is not None:
        rec["n_cuts"] = int(n_cuts)
    if runtime is not None:
        rec["runtime_sec"] = float(runtime)
    if settings:
        rec["solver_settings"] = dict(settings)
    return rec


def sparsify_alloc(v, XA, rel):
    """Zero out negligible components of an allocation and re-spend the budget.

    Iterates are kept at x_i >= x_min for numerical safety (the subgradient
    -1/(x_i + y_i) blows up as x_i -> 0), but x_min is NOT part of the model:
    x_i = 0 is a legal allocation and convention (C1) gives p_i = 1 there.
    Holding x_i >= x_min biases the UPPER bound upward by roughly
    sqrt(x_min / t) per node that ought to be zero, so the solver also
    evaluates these exactly-sparse points.  A reported optimum with x_i = 0 is
    therefore the model's boundary solution, not an artefact of the floor.

    Returns None when everything would be zeroed out.
    """
    v = np.asarray(v, dtype=float)
    w = np.where(v < rel * XA, 0.0, v)
    if w.sum() <= 0:
        return None
    return w * (XA / w.sum())


def solve_defender(G, S, D, XA, XB, m=1.0, contest_endpoints=False,
                   mode="auto", max_iter=200, tol=None, x0=None,
                   damping=True, pool=6, max_cuts=500, verbose=False,
                   oracle=None, master="boxstep", trust0=0.25):
    """Solve  V* = min_x max_{P,y} prod y_i^m/(x_i^m + y_i^m).

    Method
    ------
    Work with G(x) = max_P g_P(x) (log space), which is convex for 0 < m <= 1
    (P2).  Iterate:
        1. evader best response at x_k  -> value and a Danskin subgradient
        2. add the linear cut  theta >= g_k + s_k^T (x - x_k)
        3. master LP  min{theta : cuts, sum x = XA, x >= 0}  -> x_{k+1}, LB
        4. UB = best rigorous upper bound on G seen so far
    Stop when UB - LB <= tol.  LB is always valid (cuts underestimate G, P4)
    and UB is always valid (enumeration, or the Lagrangian bound P5), so the
    reported gap is a genuine bracket on the optimum.

    Certification
    -------------
    The result carries a global optimality certificate ONLY when that gap
    closes within `tol`.  The returned dict always contains a `certificate`
    sub-dict (see `certificate`) with UB, LB, absolute gap, relative gap and
    its denominator, the requested tolerance, iteration and cut counts,
    runtime, solver settings, and a `certified` flag / `status` string.  A run
    that exits at `max_iter` with a nonzero gap is reported as NOT certified;
    its value may still be accurate, but it is not proved optimal.

    For 0 < m <= 1 the convexity result (P2) is what makes the LB valid.  For
    m > 1 convexity provably fails, so cuts are no longer global
    under-estimators: `certificate["convexity_proved"]` is False there and the
    run is flagged exploratory / non-certified regardless of the gap.
    A VALID (if weaker) lower bound for m > 1 is available separately via
    `relaxation.relaxed_lower_bound`.

    master : "kelley" | "boxstep"
        Which master problem picks the next iterate.

        "kelley" is the plain cutting-plane master: the next iterate is the
        unrestricted LP minimiser.  Simple, but Kelley's method is known to
        converge slowly in higher dimension because the LP minimiser can jump to
        a far corner of the simplex on every iteration; that is exactly why the
        certificate used not to close within the iteration cap on the widest
        instances.

        "boxstep" (the DEFAULT) adds the classical BOXSTEP stabilisation
        (Marsten, Hogan & Blankenship): the next iterate is chosen from the same
        cut model but restricted to a box (trust region) of radius `trust`
        around the current incumbent, which stops the iterate oscillating
        between corners.  The radius is adapted -- expanded on a successful
        step, contracted on an unsuccessful one.  On the layered family of E7
        this closes every certificate that plain Kelley left open (including the
        48-contested-node instance, gap 2e-3 -> 7e-10) in roughly a quarter of
        the iterations, and it agrees with Kelley's value to ~2e-10 wherever
        Kelley does converge.

        The LOWER bound is taken from the UNRESTRICTED master in both cases.
        This matters for correctness: restricting the master shrinks its
        feasible set and would raise its optimal value, so a boxed LP value is
        NOT a valid lower bound.  "boxstep" therefore solves two LPs per
        iteration -- the unrestricted one for the rigorous LB, the boxed one for
        the next iterate -- and the reported certificate means exactly what it
        means under "kelley".

    trust0 : initial boxstep radius as a fraction of XA (ignored for "kelley").
    """
    t0 = time.time()
    tol = tolcfg.GAP_TOL if tol is None else tol
    convex_regime = bool(0.0 < m <= 1.0)
    settings = {"mode": mode, "max_iter": max_iter, "tol": tol,
                "damping": damping, "pool": pool, "max_cuts": max_cuts,
                "master": master, "trust0": trust0,
                "m": m, "XA": XA, "XB": XB,
                "contest_endpoints": contest_endpoints,
                "lp_method": "highs", "lp_feas_tol": tolcfg.LP_FEAS_TOL,
                "root_tol": tolcfg.ROOT_TOL,
                "x_floor_rel": tolcfg.X_FLOOR_REL}
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints,
                            mode=mode)
    nodes = oracle.nodes
    n = len(nodes)

    if n == 0 or getattr(oracle, "trivial", False):
        # some S-D path has no contested node -> the evader gets through surely
        cert = certificate(0.0, 0.0, tol, iterations=0, n_cuts=0,
                           runtime=time.time() - t0, settings=settings)
        cert["convexity_proved"] = convex_regime
        cert["certified"] = bool(cert["certified"] and convex_regime)
        cert["status"] = cert["status"] if convex_regime else \
            "not certified (m > 1: convexity of the objective fails)"
        return {"x": {v: XA / max(n, 1) for v in nodes}, "log_lb": 0.0,
                "log_ub": 0.0, "value": 1.0, "gap": 0.0, "iterations": 0,
                "history": [], "nodes": nodes, "trivial": True,
                "time": time.time() - t0, "oracle": oracle,
                "best_path": oracle.paths[0] if oracle.paths else None,
                "y": None, "certified": cert["certified"],
                "certificate": cert, "n_cuts": 0, "residuals": {}}

    # x_min is a NUMERICAL floor, not a model constraint (see `sparsify_alloc`).
    xmin = max(tolcfg.X_FLOOR_ABS, tolcfg.X_FLOOR_REL * XA / n)
    x = (np.full(n, XA / n) if x0 is None else
         np.asarray([x0[v] for v in nodes], float))
    x = np.maximum(x, xmin)
    x *= XA / x.sum()

    A_ub, b_ub = [], []
    LB, UB = -np.inf, 0.0
    x_best, br_best = x.copy(), None
    history = []
    lam_boundary_hits = 0
    trust = float(trust0) * XA          # boxstep radius (unused for "kelley")
    UB_prev = UB

    def try_ub(xp):
        """Record xp as an incumbent if its (rigorous) value beats UB."""
        nonlocal UB, x_best, br_best, lam_boundary_hits
        br = oracle.best_response(xp, XB, pool=pool)
        ls = br.get("lambda_search")
        if ls and ls.get("at_boundary"):
            lam_boundary_hits += 1
        if br["ub"] < UB:
            UB, x_best, br_best = br["ub"], np.asarray(xp, float).copy(), br
        return br

    for it in range(1, max_iter + 1):
        pts = [x] if not damping or br_best is None else [x, 0.5 * (x + x_best)]
        # Sparsified probes improve the UPPER bound only.  They are never used
        # to build cuts: at x_i = 0 the subgradient -1/(x_i + y_i) is unbounded
        # and Danskin's theorem is not being invoked there (see `cut_from`).
        for rel in (1e-6,):
            sp = sparsify_alloc(x, XA, rel)
            if sp is not None and not np.allclose(sp, x):
                try_ub(sp)

        for xp in pts:
            br = try_ub(xp)
            for cand in br["pool"]:
                s_loc = cut_from(oracle._sub(xp, cand["interior"]), cand["y"], m)
                s = np.zeros(n)
                for v, sv in zip(cand["interior"], s_loc):
                    if v in oracle.index:
                        s[oracle.index[v]] += sv
                b = cand["logval"] - float(s @ xp)     # theta >= s^T x + b
                A_ub.append(np.append(s, -1.0))
                b_ub.append(-b)

        if len(A_ub) > max_cuts:                      # keep the tightest cuts
            Am, bm = np.array(A_ub), np.array(b_ub)
            slack = bm - Am @ np.append(x_best, UB)
            keep = np.argsort(slack)[:max_cuts - 40]
            keep = np.union1d(keep, np.arange(len(A_ub) - 40, len(A_ub)))
            A_ub = [A_ub[i] for i in keep]
            b_ub = [b_ub[i] for i in keep]

        c = np.append(np.zeros(n), 1.0)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.append(np.ones(n), 0.0).reshape(1, -1),
                      b_eq=[XA],
                      # x >= 0 (NOT x >= xmin): the cuts underestimate G on the
                      # whole closed simplex, so the LP value is a rigorous LB
                      # for the unrestricted problem.  The iterate is clamped to
                      # xmin afterwards purely for numerical safety.
                      bounds=[(0.0, XA)] * n + [(-1e9, 0.0)],
                      method="highs",
                      # cut slopes -1/(x_i+y_i) can reach ~1e4, so the default
                      # feasibility tolerances would leave ~1e-6 of slack in the
                      # LOWER bound.  Tighten them (tolerances.LP_FEAS_TOL).
                      options={"primal_feasibility_tolerance": tolcfg.LP_FEAS_TOL,
                               "dual_feasibility_tolerance": tolcfg.LP_FEAS_TOL})
        if not res.success:
            break
        # The LOWER bound always comes from the UNRESTRICTED master: it is the
        # only one whose optimal value under-estimates G on the whole simplex.
        LB = max(LB, float(res.x[-1]))
        x_new = np.maximum(res.x[:n], xmin)
        x_new *= XA / x_new.sum()

        if master == "boxstep":
            # Re-solve the SAME cut model inside a box around the incumbent to
            # pick the next iterate.  Used only to choose where to evaluate
            # next; never to produce a bound.
            lo = np.maximum(x_best - trust, 0.0)
            hi = np.minimum(x_best + trust, XA)
            res_b = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                            A_eq=np.append(np.ones(n), 0.0).reshape(1, -1),
                            b_eq=[XA],
                            bounds=[(float(a), float(bb))
                                    for a, bb in zip(lo, hi)]
                                   + [(-1e9, 0.0)],
                            method="highs",
                            options={"primal_feasibility_tolerance":
                                     tolcfg.LP_FEAS_TOL,
                                     "dual_feasibility_tolerance":
                                     tolcfg.LP_FEAS_TOL})
            if res_b.success:
                xb = np.maximum(res_b.x[:n], xmin)
                ssum = xb.sum()
                if ssum > 0:
                    x_new = xb * (XA / ssum)
            # adapt the radius: grow it when the incumbent improved, shrink it
            # when the step bought nothing.
            if UB < UB_prev - 1e-14:
                trust = min(trust * 1.6, XA)
            else:
                trust = max(trust * 0.6, 1e-6 * XA)
            UB_prev = UB

        history.append({"iter": it, "lb": LB, "ub": UB, "gap": UB - LB,
                        "n_cuts": len(A_ub), "time": time.time() - t0})
        if verbose:
            print(f"  it {it:3d}  LB={LB: .8f}  UB={UB: .8f}  "
                  f"gap={UB-LB:.2e}  cuts={len(A_ub)}")
        if UB - LB <= tol:
            break
        x = x_new

    # ---- final polish: try exactly-sparse versions of the incumbent --------
    for rel in (1e-9, 1e-7, 1e-5, 1e-4, 1e-3):
        sp = sparsify_alloc(x_best, XA, rel)
        if sp is not None:
            try_ub(sp)

    # The LOWER bound comes from an LP solved to a finite tolerance while the
    # UPPER bound is a direct evaluation, so UB - LB can come out very slightly
    # negative once both agree to ~1e-9.  Report that honestly (gap_raw) rather
    # than pretending the bracket is exact.
    gap_raw = UB - LB
    LB_rep = min(LB, UB)

    br_fin = oracle.best_response(x_best, XB)
    runtime = time.time() - t0

    cert = certificate(LB_rep, UB, tol, iterations=len(history),
                       n_cuts=len(A_ub), runtime=runtime, settings=settings)
    cert["abs_gap_raw"] = float(gap_raw)
    cert["convexity_proved"] = convex_regime
    cert["path_oracle_certified"] = bool(br_fin["certified"])
    cert["lambda_boundary_hits"] = lam_boundary_hits
    if not convex_regime:
        # Convexity provably fails for m > 1 (FORMULATION.md S3, Prop. 5), so
        # the cuts are not global under-estimators and LB is not a valid bound.
        # Such runs are exploratory, never certified.
        cert["certified"] = False
        cert["status"] = ("not certified (m > 1: convexity of the objective "
                          "provably fails, so the lower bound is not valid)")
    elif lam_boundary_hits:
        cert["status"] += " [dual search hit a bracket boundary]"

    # ---- residual checks (audit trail, not a claim) ------------------------
    y_fin = np.asarray(br_fin["y"], float)
    x_fin_path = oracle._sub(x_best, br_fin["interior"])
    lam_fin = br_fin.get("lam", np.inf)
    t_fin = 1.0 / lam_fin if np.isfinite(lam_fin) and lam_fin > 0 else 0.0
    resid = alloc_residuals(x_fin_path, {"y": y_fin, "t": t_fin}, XB, m)
    resid["defender_budget_abs"] = float(abs(x_best.sum() - XA))
    resid["certificate_abs_gap"] = float(max(gap_raw, 0.0))
    resid["path_cost_consistency"] = _path_cost_consistency(oracle, x_best, XB)
    cert["residuals"] = resid

    return {"x": oracle.to_dict(x_best), "x_vec": x_best,
            "log_lb": LB_rep, "log_lb_raw": LB, "log_ub": UB,
            "value": math.exp(UB),
            "value_lb": math.exp(LB_rep), "gap": max(gap_raw, 0.0),
            "gap_raw": gap_raw,
            "rel_gap": math.expm1(max(gap_raw, 0.0)), "iterations": len(history),
            "n_cuts": len(A_ub), "tolerance": tol,
            "history": history, "nodes": nodes, "trivial": False,
            "best_path": br_fin["path"],
            "y": {v: float(yy) for v, yy in zip(br_fin["interior"], br_fin["y"])},
            "certified": cert["certified"], "certificate": cert,
            "residuals": resid,
            "path_oracle_certified": bool(br_fin["certified"]),
            "time": runtime,
            "oracle": oracle, "n_paths": oracle.n_paths_known,
            "paths_complete": oracle.paths_complete}


def _path_cost_consistency(oracle, x_vec, XB):
    """Check the dual relaxation's shortest-path step against direct evaluation.

    For the multiplier lambda chosen by the dual search, the Dijkstra path cost
    must equal the sum of the node costs c_i(lambda) = -psi(x_i, lambda) along
    the returned path (plus the source term when endpoints are contested).
    Any mismatch would mean the shortest-path reduction and the node/path
    convention have drifted apart.  Returns the absolute discrepancy, or None
    when the oracle is in enumeration mode and never builds the relaxation.
    """
    if oracle.mode != "ksp":
        return None
    try:
        lam, _ = oracle._best_lambda(x_vec, XB)
        length, path = oracle._dual_at(x_vec, lam)
        inter = path_interior(path, oracle.S, oracle.D, oracle.contest_endpoints)
        direct = float(-psi(oracle._sub(x_vec, inter), lam, oracle.m).sum())
        return float(abs(length - direct))
    except Exception:                                   # pragma: no cover
        return None


# ----------------------------------------------------------------------------
# 5.  Heuristic / baseline defender strategies (for comparison ONLY)
# ----------------------------------------------------------------------------


def heuristic_allocation(G, S, D, XA, kind="uniform", XB=None, m=1.0,
                         contest_endpoints=False, oracle=None, steps=200):
    """Baselines.  None of these is optimal in general -- they are baselines."""
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    nodes = oracle.nodes
    n = len(nodes)
    if n == 0:
        return {}

    if kind == "uniform":
        w = np.ones(n)

    elif kind == "pathcount":
        if oracle.paths_complete:
            cnt = {v: 0 for v in nodes}
            for inter in oracle.interiors:
                for v in inter:
                    if v in cnt:
                        cnt[v] += 1
            w = np.array([cnt[v] for v in nodes], float)
        else:                                   # DAG dynamic programming
            order = list(nx.topological_sort(G))
            fwd = {v: 0.0 for v in G}; fwd[S] = 1.0
            for u in order:
                for v in G.successors(u):
                    fwd[v] += fwd[u]
            bwd = {v: 0.0 for v in G}; bwd[D] = 1.0
            for u in reversed(order):
                for v in G.successors(u):
                    bwd[u] += bwd[v]
            w = np.array([fwd[v] * bwd[v] for v in nodes], float)
        w = np.maximum(w, 0.0)

    elif kind == "mincut":
        try:
            cut = nx.minimum_node_cut(G, S, D)
        except Exception:
            cut = set(nodes)
        cut = {v for v in cut if v in oracle.index} or set(nodes)
        w = np.array([1.0 if v in cut else 0.0 for v in nodes])

    elif kind == "betweenness":
        bc = nx.betweenness_centrality_subset(G, sources=[S], targets=[D],
                                              normalized=True)
        w = np.array([max(bc.get(v, 0.0), 0.0) for v in nodes])

    elif kind == "greedy":
        # myopic: repeatedly give a budget slice to the most valuable node of
        # the CURRENT worst-case path (largest |dg/dx_i| = 1/(x_i+y_i))
        x = np.full(n, tolcfg.X_FLOOR_REL * XA / n)
        inc = XA / steps
        for _ in range(steps):
            br = oracle.best_response(x, XB)
            s_loc = cut_from(oracle._sub(x, br["interior"]), br["y"], m)
            j = int(np.argmin(s_loc))           # most negative slope
            v = br["interior"][j]
            if v in oracle.index:
                x[oracle.index[v]] += inc
        x *= XA / x.sum()
        return oracle.to_dict(x)

    else:
        raise ValueError(f"unknown heuristic {kind!r}")

    if w.sum() <= 0:
        w = np.ones(n)
    x = XA * w / w.sum()
    x = np.maximum(x, tolcfg.X_FLOOR_ABS * XA / n)
    x *= XA / x.sum()
    return oracle.to_dict(x)


def brute_force_grid(G, S, D, XA, XB, m=1.0, steps=40, contest_endpoints=False,
                     max_points=2_000_000):
    """Independent reference for TINY instances: grid search over the simplex."""
    oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    nodes = oracle.nodes
    n = len(nodes)
    if n > 1:
        pts = math.comb(steps + n - 1, n - 1)
        if pts > max_points:
            raise ValueError(f"grid too large: {pts:.3g} points for n={n}, "
                             f"steps={steps}")
    best = (np.inf, None)

    def rec(i, left, acc):
        nonlocal best
        if i == n - 1:
            alloc = acc + [left]
            x = np.maximum(np.array(alloc, float) * XA / steps,
                           tolcfg.X_FLOOR_ABS)
            v = oracle.best_response(x, XB)["logval"]
            if v < best[0]:
                best = (v, x.copy())
            return
        for k in range(left + 1):
            rec(i + 1, left - k, acc + [k])

    if n == 0:
        return {"logval": 0.0, "value": 1.0, "x": {}}
    rec(0, steps, [])
    return {"logval": best[0], "value": math.exp(best[0]),
            "x": oracle.to_dict(best[1])}
