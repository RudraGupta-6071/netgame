"""
baselines.py -- head-to-head comparison against the solution METHODS of the two
nearest papers, as opposed to the generic centrality/cut heuristics in core.py.

WHY THIS MODULE EXISTS
----------------------
`core.heuristic_allocation` compares the certified optimum against generic rules
(uniform, min vertex cut, path count, betweenness, greedy).  That is not the
comparison a reader of the related-work section will ask for.  This project
positions itself between two papers, so it should be compared against what those
papers actually *do*:

  * Ramirez-Marquez, Rocco & Levitin (2011) solve their protection problem with
    an EVOLUTIONARY ALGORITHM carrying no optimality guarantee.
  * Nguyen, Song & Smith (2023) solve theirs by CONSTRAINT GENERATION over
    paths, where the separation problem is turned into a shortest-path problem
    by log-linearising a product of survival probabilities.

HONESTY ABOUT WHAT THESE ARE
----------------------------
Neither original method applies verbatim to the model in FORMULATION.md S1, so
neither function below is a reimplementation of a published algorithm, and
neither should be described as one.  They are ADAPTATIONS, and the adaptation is
stated explicitly in each docstring:

  * `rrl_evolutionary` ports the *search strategy* (a population-based
    evolutionary search over defender allocations with no optimality
    certificate) onto this model's objective.  RRL's own contest function is the
    same ratio form we use, so the objective transfers; what does not transfer is
    their fixed, finitely-enumerated attacker scenarios, since here the evader
    re-optimises continuously.  We therefore hold the EVALUATION oracle fixed
    (both methods are scored by the same exact best-response oracle) so the
    comparison isolates the defender-side search method, which is the thing
    actually being compared.

  * `nss_style_oracle` ports the *separation strategy*: generate paths by a
    shortest-path computation on linearised node costs rather than on this
    project's Lagrangian-tight costs.  NSS's setting is discrete and
    simultaneous, so an apples-to-apples run of their algorithm is not possible;
    what is comparable is the CONVERGENCE BEHAVIOUR of the two separation rules
    on identical instances -- paths evaluated, and iterations/cuts needed to
    reach a given gap -- which is what E9 reports.

Both comparisons are therefore about method, on identical instances, with every
other component held fixed.  Results are reported with the same
excess-success-probability metric defined in experiments.py:

    excess success probability (%) = 100 * (V_method / V* - 1)
"""

from __future__ import annotations

import math
import time

import numpy as np

import tolerances as tolcfg
from core import PathOracle, evader_alloc, path_interior, psi

__all__ = ["rrl_evolutionary", "nss_style_oracle", "nss_constraint_generation"]


# ----------------------------------------------------------------------------
# 1.  Ramirez-Marquez / Rocco / Levitin -- evolutionary search over allocations
# ----------------------------------------------------------------------------


def rrl_evolutionary(G, S, D, XA, XB, m=1.0, oracle=None, contest_endpoints=False,
                     pop_size=60, generations=250, elite=4, p_mutate=0.30,
                     sigma0=0.45, tournament=3, seed=0, verbose=False):
    """Evolutionary (genetic) search for the defender allocation, RRL-style.

    ADAPTATION, NOT A REIMPLEMENTATION.  Ramirez-Marquez, Rocco & Levitin (2011)
    optimise protection of a system whose components have the same ratio-form
    vulnerability `v = T^m/(T^m + t^m)` used here, but against a fixed menu of
    named interdiction scenarios, using an evolutionary algorithm that returns no
    optimality bound.  This function keeps the part that is actually theirs --
    population-based stochastic search over the defender's continuous allocation
    with no certificate -- and evaluates each candidate against THIS model's
    fully-optimising evader, using the same exact best-response oracle the
    certified solver uses.

    Holding the evaluation oracle fixed is deliberate: it makes the comparison a
    statement about the SEARCH METHOD (evolutionary vs. convex cutting planes),
    not about who has the better inner solver.  Any gap in the reported value is
    therefore attributable to the optimiser, not to a difference in how the
    objective was measured.

    Encoding.  A genome is a weight vector `w >= 0` on the relevant nodes,
    decoded as `x = XA * w / sum(w)`, so every candidate is budget-feasible by
    construction.  Operators: tournament selection, arithmetic (blend)
    crossover, multiplicative log-normal mutation with a geometrically cooled
    step size, and elitism.

    Returns a dict with the best allocation found, its value, the generation-by-
    generation best (for convergence plots), the number of objective
    evaluations, runtime and the seed.  It carries NO optimality certificate --
    that is the point of the comparison -- and the returned dict says so.
    """
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    nodes = oracle.nodes
    n = len(nodes)
    t0 = time.time()
    if n == 0:
        return {"x": {}, "value": 1.0, "logval": 0.0, "history": [],
                "evaluations": 0, "time": 0.0, "seed": seed,
                "certified": False, "method": "RRL-style evolutionary search"}

    rng = np.random.default_rng(seed)
    n_eval = 0

    def decode(w):
        w = np.maximum(w, 0.0)
        tot = w.sum()
        if tot <= 0:
            w = np.ones(n)
            tot = float(n)
        return XA * w / tot

    def fitness(w):
        nonlocal n_eval
        n_eval += 1
        return oracle.best_response(decode(w), XB)["logval"]   # lower is better

    # initial population: uniform, plus Dirichlet draws at several concentrations
    pop = [np.ones(n)]
    while len(pop) < pop_size:
        alpha = float(10.0 ** rng.uniform(-1.0, 1.0))
        pop.append(rng.dirichlet(np.full(n, alpha)))
    pop = [np.asarray(w, float) for w in pop]
    fit = [fitness(w) for w in pop]

    history = []
    for gen in range(generations):
        order = np.argsort(fit)
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        history.append({"generation": gen, "best_logval": fit[0],
                        "best_value": math.exp(fit[0]),
                        "evaluations": n_eval})
        if verbose and gen % 25 == 0:
            print(f"    gen {gen:4d}  best V = {math.exp(fit[0]):.8f}")

        sigma = sigma0 * (0.02 / sigma0) ** (gen / max(generations - 1, 1))
        new_pop = [w.copy() for w in pop[:elite]]
        new_fit = list(fit[:elite])
        while len(new_pop) < pop_size:
            def pick():
                idx = rng.integers(0, pop_size, tournament)
                return pop[min(idx, key=lambda i: fit[i])]
            a, b = pick(), pick()
            beta = rng.uniform(-0.25, 1.25)                  # blend crossover
            child = np.clip(beta * a + (1.0 - beta) * b, 0.0, None)
            if rng.random() < p_mutate:                      # log-normal mutation
                k = max(1, int(rng.integers(1, n + 1)))
                idx = rng.choice(n, size=k, replace=False)
                child[idx] *= np.exp(rng.normal(0.0, sigma, size=k))
            if child.sum() <= 0:
                child = np.ones(n)
            new_pop.append(child)
            new_fit.append(fitness(child))
        pop, fit = new_pop, new_fit

    order = np.argsort(fit)
    best_w = pop[order[0]]
    best_logval = fit[order[0]]
    x = decode(best_w)
    return {"x": oracle.to_dict(x), "x_vec": x, "value": math.exp(best_logval),
            "logval": best_logval, "history": history, "evaluations": n_eval,
            "time": time.time() - t0, "seed": seed, "generations": generations,
            "pop_size": pop_size,
            "certified": False,
            "certificate_note": "evolutionary search returns no optimality "
                                "bound; the value is an upper bound on V* only "
                                "because it is attained at a feasible point",
            "method": "RRL-style evolutionary search (adaptation)"}


# ----------------------------------------------------------------------------
# 2.  Nguyen / Song / Smith -- constraint generation with linearised separation
# ----------------------------------------------------------------------------


def nss_style_oracle(oracle, x_vec, XB, max_paths=None, nominal_len=None):
    """Evader best response by NSS-style LINEARISED path separation.

    ADAPTATION.  Nguyen, Song & Smith turn their constraint-generation
    separation problem into a shortest-path problem by log-linearising a product
    of survival probabilities (their eqs. 11-14), i.e. by scoring each node with
    a cost that does not depend on the attacker's re-optimisation.  The analogue
    here is to score node `i` with

        c_i  =  -log p_i(x_i, y_nominal)

    for a FIXED nominal evader split `y_nominal = XB / nominal_len` -- a
    first-order stand-in for the evader's actual response -- run Dijkstra, and
    then evaluate the resulting path exactly.

    The contrast with `PathOracle.best_response` is precisely the one claimed in
    FORMULATION.md S4: our Lagrangian costs `c_i(lambda) = -psi(x_i, lambda)`
    come from a relaxation that is TIGHT at the optimal multiplier, so the
    generated order supports a stopping rule that certifies no unexamined path
    can win.  The linearised costs support no such rule, so this routine must
    keep generating paths and can only report the best found.

    Returns the same fields as `best_response` plus `certified=False` and the
    number of paths evaluated, so E9 can compare separation rules on equal
    terms.
    """
    import networkx as nx

    x_vec = np.asarray(x_vec, dtype=float)
    max_paths = oracle.ksp_limit if max_paths is None else max_paths
    if nominal_len is None:
        # a nominal path length: the median contested-path length if known,
        # else the graph's shortest S-D hop count
        if oracle.paths_complete and oracle.interiors:
            lens = [max(len(q), 1) for q in oracle.interiors]
            nominal_len = float(np.median(lens))
        else:
            nominal_len = max(
                float(nx.shortest_path_length(oracle.G, oracle.S, oracle.D)) - 1.0,
                1.0)
    y_nom = XB / max(nominal_len, 1.0)

    # linearised node costs: -log p_i at the fixed nominal split
    m = oracle.m
    xm = np.maximum(x_vec, 0.0) ** m
    p_nom = (y_nom ** m) / (xm + y_nom ** m)
    cost = {v: float(-math.log(max(p_nom[i], tolcfg.LOG_FLOOR)))
            for i, v in enumerate(oracle.nodes)}
    H = oracle.H
    for u, v in H.edges():
        H[u][v]["w"] = cost.get(v, 0.0)

    best = None
    k = 0
    for path in nx.shortest_simple_paths(H, oracle.S, oracle.D, weight="w"):
        k += 1
        inter = path_interior(path, oracle.S, oracle.D, oracle.contest_endpoints)
        sol = evader_alloc(oracle._sub(x_vec, inter), XB, m)
        if best is None or sol["logval"] > best["logval"]:
            best = {"path": path, "interior": inter, "y": sol["y"],
                    "logval": sol["logval"], "lam": sol["lam"]}
        if k >= max_paths:
            break
    best = dict(best)
    best["n_evaluated"] = k
    best["ub"] = 0.0             # linearised separation supports no valid bound
    best["certified"] = False
    best["pool"] = [best]
    best["method"] = "NSS-style linearised separation (adaptation)"
    return best


def nss_constraint_generation(G, S, D, XA, XB, m=1.0, oracle=None,
                              contest_endpoints=False, max_iter=200,
                              tol=None, sep_paths=40, pool=6, max_cuts=500,
                              verbose=False):
    """Cutting-plane solve whose PATH SEPARATION is NSS-style, not Lagrangian.

    Everything except the separation rule is held identical to
    `core.solve_defender`: same master LP, same Danskin cuts, same tolerances,
    same incumbent bookkeeping.  Only the rule that decides WHICH path to cut on
    differs -- `nss_style_oracle` (linearised costs, no stopping rule) instead of
    the Lagrangian-tight oracle with its certificate.

    That isolation is the point.  Because the linearised separation cannot
    certify that the generated path is the evader's true best response, the
    UPPER bound produced here is NOT rigorous, and the run therefore carries no
    optimality certificate at all: the reported `gap` is indicative only, and the
    dict says so.  E9 reports iterations, cuts and paths evaluated to reach a
    given gap, which is the comparison that is meaningful between the two.
    """
    from scipy.optimize import linprog
    from core import cut_from

    tol = tolcfg.GAP_TOL if tol is None else tol
    t0 = time.time()
    if oracle is None:
        oracle = PathOracle(G, S, D, m=m, contest_endpoints=contest_endpoints)
    nodes = oracle.nodes
    n = len(nodes)
    if n == 0:
        return {"x": {}, "value": 1.0, "iterations": 0, "n_cuts": 0,
                "paths_evaluated": 0, "time": 0.0, "certified": False}

    xmin = max(tolcfg.X_FLOOR_ABS, tolcfg.X_FLOOR_REL * XA / n)
    x = np.full(n, XA / n)
    A_ub, b_ub = [], []
    LB, UB = -np.inf, 0.0
    x_best = x.copy()
    history = []
    paths_evaluated = 0

    for it in range(1, max_iter + 1):
        br = nss_style_oracle(oracle, x, XB, max_paths=sep_paths)
        paths_evaluated += br["n_evaluated"]
        if br["logval"] < UB:
            UB, x_best = br["logval"], x.copy()

        s_loc = cut_from(oracle._sub(x, br["interior"]), br["y"], m)
        s = np.zeros(n)
        for v, sv in zip(br["interior"], s_loc):
            if v in oracle.index:
                s[oracle.index[v]] += sv
        b = br["logval"] - float(s @ x)
        A_ub.append(np.append(s, -1.0))
        b_ub.append(-b)

        if len(A_ub) > max_cuts:
            A_ub, b_ub = A_ub[-max_cuts:], b_ub[-max_cuts:]

        c = np.append(np.zeros(n), 1.0)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.append(np.ones(n), 0.0).reshape(1, -1), b_eq=[XA],
                      bounds=[(0.0, XA)] * n + [(-1e9, 0.0)], method="highs",
                      options={"primal_feasibility_tolerance": tolcfg.LP_FEAS_TOL,
                               "dual_feasibility_tolerance": tolcfg.LP_FEAS_TOL})
        if not res.success:
            break
        LB = max(LB, float(res.x[-1]))
        history.append({"iter": it, "lb": LB, "ub": UB, "gap": UB - LB,
                        "n_cuts": len(A_ub), "paths_evaluated": paths_evaluated,
                        "time": time.time() - t0})
        if verbose:
            print(f"    nss it {it:3d}  LB={LB: .8f} UB={UB: .8f} "
                  f"gap={UB-LB:.2e}")
        if UB - LB <= tol:
            break
        x = np.maximum(res.x[:n], xmin)
        x *= XA / x.sum()

    return {"x": oracle.to_dict(x_best), "x_vec": x_best,
            "value": math.exp(UB), "log_ub": UB, "log_lb": min(LB, UB),
            "gap": max(UB - LB, 0.0), "iterations": len(history),
            "n_cuts": len(A_ub), "paths_evaluated": paths_evaluated,
            "history": history, "time": time.time() - t0,
            "certified": False,
            "certificate_note": "linearised separation gives no rigorous upper "
                                "bound on max_P g_P, so this gap is indicative "
                                "only and is NOT an optimality certificate",
            "method": "NSS-style constraint generation (adaptation)"}
