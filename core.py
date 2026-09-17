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
problem is convex in x (see FORMULATION / REPORT).

CONVENTIONS
-----------
* x_i = 0  =>  node i is undefended, p_i = 1 and B spends y_i = 0 there.
  This is the continuous extension of the CSF (p_i -> 1 as x_i -> 0+ along
  the evader's optimal response), so the value function stays continuous.
* Source and terminus are uncontested by default (contest_endpoints=False).
  Set contest_endpoints=True to force S and D onto every path.

KEY FACTS USED (proved in REPORT.md)
------------------------------------
P1  Inner problem is strictly concave in y  -> unique optimum, KKT + monotone
    bisection on the multiplier.
P2  g_P(x) = max_y h(x,y) is CONVEX in x for 0 < m <= 1, because h(.,y) is
    convex in x for every fixed y and a pointwise sup of convex functions is
    convex.  G(x) = max_P g_P(x) is therefore convex too.
P3  Danskin: s_i = d g_P / d x_i = -m x_i^(m-1) / (x_i^m + y_i*^m).
P4  Any FEASIBLE y (not only the optimal one) yields a VALID cut, because
    g_P(x) >= h(x,y) >= tangent of h(.,y).  Lower bounds are therefore robust
    to inner-solver error.
P5  Lagrangian/weak duality: for every lam > 0,
        g_P(x) <= lam*XB + sum_{i in P} psi(x_i, lam),
    with equality at the optimal lam.  psi is separable => the bound is a
    SHORTEST-PATH problem => certified stopping rule for path generation.
"""

from __future__ import annotations

import math
import time
from itertools import islice

import networkx as nx
import numpy as np
from scipy.optimize import linprog

__all__ = [
    "relevant_nodes", "enumerate_paths", "path_interior",
    "evader_alloc", "psi", "dual_bound_path",
    "PathOracle", "game_value", "solve_defender",
    "heuristic_allocation", "brute_force_grid",
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
# 1.  Inner problem: evader's budget split on a FIXED path  (exact)
# ----------------------------------------------------------------------------


def _y_of_t(xv, t, m=1.0):
    """Stationary y_i as a function of t = 1/lambda.

    KKT:  m x^m / (y (x^m + y^m)) = lambda   <=>   y (x^m + y^m) = t m x^m.
    The left side is strictly increasing in y, so the root is unique.
    For m = 1 this is the quadratic y^2 + x y - t x = 0 with the
    cancellation-free root  y = 2xt / (x + sqrt(x^2 + 4xt)).
    """
    xv = np.asarray(xv, dtype=float)
    if m == 1.0:
        return 2.0 * xv * t / (xv + np.sqrt(xv * xv + 4.0 * xv * t))
    xm = xv ** m
    target = t * m * xm
    # F(y) = y*x^m + y^(m+1) - target is increasing and convex on y > 0, so
    # Newton started from any point with F <= 0 converges monotonically.
    y = 0.5 * np.minimum(target / np.maximum(xm, 1e-300),
                         target ** (1.0 / (m + 1.0)))
    y = np.maximum(y, 1e-300)
    for _ in range(60):
        F = y * xm + y ** (m + 1.0) - target
        dF = xm + (m + 1.0) * y ** m
        step = F / dF
        y_new = np.maximum(y - step, 0.5 * y)          # safeguard
        if np.all(np.abs(y_new - y) <= 1e-14 * np.maximum(y, 1e-300)):
            y = y_new
            break
        y = y_new
    return y


def _p_of_t(xv, yv, t, m=1.0):
    """p_i at the stationary point (cancellation-free closed form for m = 1)."""
    xv = np.asarray(xv, dtype=float)
    if m == 1.0:
        return 2.0 * t / (xv + 2.0 * t + np.sqrt(xv * xv + 4.0 * xv * t))
    ym = yv ** m
    return ym / (xv ** m + ym)


def evader_alloc(x_path, XB, m=1.0, iters=60):
    """Exact solution of   max_y sum log p_i   s.t.  sum y_i = XB,  y >= 0.

    Parameters
    ----------
    x_path : array of defender allocations on the contested nodes of one path
    XB     : evader budget
    m      : contest intensity

    Returns dict with keys y, p, logval, lam (= multiplier), t (= 1/lam).
    `y` is exactly budget feasible, and logval is computed from that `y`,
    so logval is always an attainable (lower) value -- see P4.
    """
    x_path = np.asarray(x_path, dtype=float)
    n = x_path.size
    y = np.zeros(n)
    p = np.ones(n)
    if n == 0 or XB <= 0:
        return {"y": y, "p": p, "logval": 0.0 if XB > 0 or n == 0 else -np.inf,
                "lam": np.inf, "t": 0.0}

    act = x_path > 0.0
    if not act.any():                          # whole path undefended
        return {"y": y, "p": p, "logval": 0.0, "lam": np.inf, "t": 0.0}

    xa = x_path[act]

    # --- bracket t so that sum y(t) = XB (sum y is strictly increasing in t)
    t_lo, t_hi = 1e-30, 1.0
    for _ in range(400):
        if _y_of_t(xa, t_hi, m).sum() >= XB:
            break
        t_hi *= 4.0
    for _ in range(400):
        if _y_of_t(xa, t_lo, m).sum() <= XB:
            break
        t_lo /= 4.0

    # --- safeguarded Newton on  S(t) = sum_i y_i(t) = XB
    #     S'(t) = sum_i m x^m / (x^m + (m+1) y^m)      (exact)
    t = math.sqrt(t_lo * t_hi)
    xam = xa ** m
    for _ in range(iters):
        ya = _y_of_t(xa, t, m)
        S = ya.sum()
        if abs(S - XB) <= 1e-15 * XB:
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

    ya = _y_of_t(xa, t, m)
    s = ya.sum()
    if s > 0:
        ya = ya * (XB / s)                     # exact budget feasibility
    ya = np.maximum(ya, 1e-300)
    pa = (ya ** m) / (xa ** m + ya ** m)

    y[act] = ya
    p[act] = pa
    return {"y": y, "p": p, "logval": float(np.log(p).sum()),
            "lam": 1.0 / t, "t": t}


def cut_from(x_path, y, m=1.0):
    """Danskin subgradient of g_P at x_path (P3).  Valid for any feasible y."""
    x_path = np.asarray(x_path, dtype=float)
    y = np.asarray(y, dtype=float)
    if m == 1.0:
        denom = x_path + y
    else:
        denom = (x_path ** m + y ** m) / np.maximum(m * x_path ** (m - 1.0), 1e-300)
    return -1.0 / np.maximum(denom, 1e-300)


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
        out[act] = np.log(np.maximum(pa, 1e-300)) - lam * ya
    return out


def dual_bound_path(x_path, XB, lam, m=1.0):
    """Valid UPPER bound on g_P(x) for any lam > 0 (P5); tight at lam*."""
    return lam * XB + float(psi(x_path, lam, m).sum())


# ----------------------------------------------------------------------------
# 3.  Evader best response over paths
# ----------------------------------------------------------------------------


class PathOracle:
    """Evader best-response engine.

    mode = 'enum' : exact, enumerates every simple S-D path.
    mode = 'ksp'  : exact *with a certificate*.  Paths are generated in
                    increasing Lagrangian cost order (Yen's algorithm); since
                    g_P <= lam*XB - c_lam(P) and c_lam is non-decreasing along
                    the enumeration, the search can stop as soon as
                    lam*XB - c_lam(P_k) <= best value found.
    mode = 'auto' : 'enum' when the path count is small, else 'ksp'.
    """

    def __init__(self, G, S, D, m=1.0, contest_endpoints=False,
                 mode="auto", enum_limit=20000, ksp_limit=4000):
        self.G, self.S, self.D, self.m = G, S, D, m
        self.contest_endpoints = contest_endpoints
        self.ksp_limit = ksp_limit
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

    def _best_lambda(self, x_vec, XB, grid=25, golden=40):
        """Minimise U(lam) = lam*XB - SP(lam) over lam > 0 (U is convex)."""
        lo, hi = -12.0, 12.0                    # log10 lam
        ls = np.linspace(lo, hi, grid)
        vals = []
        for L in ls:
            lam = 10.0 ** L
            length, _ = self._dual_at(x_vec, lam)
            vals.append(lam * XB - length)
        k = int(np.argmin(vals))
        a = ls[max(k - 1, 0)]
        b = ls[min(k + 1, grid - 1)]
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c_, d_ = b - gr * (b - a), a + gr * (b - a)
        fc = 10.0 ** c_ * XB - self._dual_at(x_vec, 10.0 ** c_)[0]
        fd = 10.0 ** d_ * XB - self._dual_at(x_vec, 10.0 ** d_)[0]
        for _ in range(golden):
            if fc < fd:
                b, d_, fd = d_, c_, fc
                c_ = b - gr * (b - a)
                fc = 10.0 ** c_ * XB - self._dual_at(x_vec, 10.0 ** c_)[0]
            else:
                a, c_, fc = c_, d_, fd
                d_ = a + gr * (b - a)
                fd = 10.0 ** d_ * XB - self._dual_at(x_vec, 10.0 ** d_)[0]
        lam = 10.0 ** (0.5 * (a + b))
        return lam, lam * XB - self._dual_at(x_vec, lam)[0]

    # -- best response -------------------------------------------------------
    def best_response(self, x_vec, XB, pool=1):
        """Evader best response at x.

        Returns dict with path / interior / y / logval (attainable value, a
        LOWER bound on G) / ub (rigorous UPPER bound on G) / certified /
        n_evaluated, plus `pool`: up to `pool` near-binding paths, each of
        which yields an extra valid cut for the master problem.
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
            if cands and bound_rest <= best_val + 1e-12:
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
        best["certified"] = bool(best["ub"] <= best["logval"] + 1e-9)
        best["n_evaluated"] = k
        best["pool"] = cands
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


def solve_defender(G, S, D, XA, XB, m=1.0, contest_endpoints=False,
                   mode="auto", max_iter=200, tol=1e-7, x0=None,
                   damping=True, pool=6, max_cuts=500, verbose=False,
                   oracle=None):
    """Solve  V* = min_x max_{P,y} prod y_i/(x_i+y_i).

    Method
    ------
    Work with G(x) = max_P g_P(x) (log space), which is convex (P2).
    Iterate:
        1. evader best response at x_k  -> exact g and a Danskin subgradient
        2. add the linear cut  theta >= g_k + s_k^T (x - x_k)
        3. master LP  min{theta : cuts, sum x = XA, x >= xmin}  -> x_{k+1}, LB
        4. UB = best rigorous upper bound on G seen so far
    Stop when UB - LB <= tol.  LB is always valid (cuts underestimate G, P4),
    UB is always valid (exact enumeration, or the Lagrangian bound P5), so the
    reported gap is a genuine optimality certificate.
    """
    t0 = time.time()
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints,
                            mode=mode)
    nodes = oracle.nodes
    n = len(nodes)

    if n == 0 or getattr(oracle, "trivial", False):
        # some S-D path has no contested node -> the evader gets through surely
        return {"x": {v: XA / max(n, 1) for v in nodes}, "log_lb": 0.0,
                "log_ub": 0.0, "value": 1.0, "gap": 0.0, "iterations": 0,
                "history": [], "nodes": nodes, "trivial": True,
                "time": time.time() - t0, "oracle": oracle,
                "best_path": oracle.paths[0] if oracle.paths else None,
                "y": None, "certified": True}

    xmin = max(1e-12, 1e-9 * XA / n)
    x = (np.full(n, XA / n) if x0 is None else
         np.asarray([x0[v] for v in nodes], float))
    x = np.maximum(x, xmin)
    x *= XA / x.sum()

    A_ub, b_ub = [], []
    LB, UB = -np.inf, 0.0
    x_best, br_best = x.copy(), None
    history = []

    def sparsify(v, rel):
        """Zero out negligible components and re-spend the budget.

        The iterates are kept at x_i >= xmin for numerical safety, which costs
        the UPPER bound about sqrt(xmin/t) per node that should really be 0.
        Evaluating the exactly-sparse point removes that bias: x_i = 0 is handled
        by the p_i = 1 convention, with no 1/(x_i+y_i) blow-up.
        """
        w = np.where(v < rel * XA, 0.0, v)
        if w.sum() <= 0:
            return None
        return w * (XA / w.sum())

    def try_ub(xp):
        """Record xp as an incumbent if its (rigorous) value beats UB."""
        nonlocal UB, x_best, br_best
        br = oracle.best_response(xp, XB, pool=pool)
        if br["ub"] < UB:
            UB, x_best, br_best = br["ub"], np.asarray(xp, float).copy(), br
        return br

    for it in range(1, max_iter + 1):
        pts = [x] if not damping or br_best is None else [x, 0.5 * (x + x_best)]
        # Sparsified probes improve the UPPER bound only.  They are never used
        # to build cuts: at x_i = 0 the subgradient -1/(x_i+y_i) is unbounded.
        for rel in (1e-6,):
            sp = sparsify(x, rel)
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
                      # LOWER bound.  Tighten them.
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
        if not res.success:
            break
        LB = max(LB, float(res.x[-1]))
        x_new = np.maximum(res.x[:n], xmin)
        x_new *= XA / x_new.sum()

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
        sp = sparsify(x_best, rel)
        if sp is not None:
            try_ub(sp)

    # The LOWER bound comes from an LP solved to a finite tolerance while the
    # UPPER bound is an exact evaluation, so UB - LB can come out very slightly
    # negative once both agree to ~1e-9.  Report that honestly rather than
    # pretending the bracket is exact.
    gap_raw = UB - LB
    LB_rep = min(LB, UB)

    br_fin = oracle.best_response(x_best, XB)
    return {"x": oracle.to_dict(x_best), "x_vec": x_best,
            "log_lb": LB_rep, "log_lb_raw": LB, "log_ub": UB,
            "value": math.exp(UB),
            "value_lb": math.exp(LB_rep), "gap": max(gap_raw, 0.0),
            "gap_raw": gap_raw,
            "rel_gap": math.expm1(max(gap_raw, 0.0)), "iterations": len(history),
            "history": history, "nodes": nodes, "trivial": False,
            "best_path": br_fin["path"],
            "y": {v: float(yy) for v, yy in zip(br_fin["interior"], br_fin["y"])},
            "certified": br_fin["certified"], "time": time.time() - t0,
            "oracle": oracle, "n_paths": oracle.n_paths_known,
            "paths_complete": oracle.paths_complete}


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
        x = np.full(n, 1e-9 * XA / n)
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
    x = np.maximum(x, 1e-12 * XA / n)
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
            x = np.maximum(np.array(alloc, float) * XA / steps, 1e-12)
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
