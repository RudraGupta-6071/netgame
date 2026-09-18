"""
experiments.py -- numerical study of the network security game.

Run:  python experiments.py            (all experiments)
      python experiments.py E1 E4      (a subset)

Everything is written to out/ :  *.csv tables, *.json raw results, *.png figures.
All randomness is seeded, so the whole study is reproducible.

TERMINOLOGY AND METRIC DEFINITIONS (used consistently in every table)

  game value V*      the optimum of  min_x max_{P,y} prod y_i^m/(x_i^m + y_i^m).
                     Reported as V_ub, the value of the best defender
                     allocation actually found, which is a rigorous upper
                     bound on V*.
  log_lb / log_ub    rigorous bracket on log V* produced by the cutting plane
  abs_gap            log_ub - log_lb, in log space
  rel_gap            abs_gap / max(|log_ub|, 1)  -- denominator stated because
                     a relative gap without one is ambiguous
  tol                the requested gap tolerance for that run
  certified          TRUE only when abs_gap <= tol.  A row with certified=no is
                     an accurate-looking value that has NOT been proved
                     optimal; it must not be described as certified.
  baseline           a NON-optimal defender rule, evaluated by the same oracle

  excess success probability (%)
                     = 100 * (V_heuristic / V* - 1)

                     i.e. the percentage by which a heuristic allocation raises
                     the evader's success probability above the optimum.  It is
                     a RATIO minus one, not an absolute difference and not a
                     difference of percentage points: "+211%" means the evader
                     succeeds 3.11x as often, not "211 percentage points more".
                     V* is taken as V_ub, so the figure is conservative
                     (understated) by at most the certified gap.

Every run writes out/RUN_METADATA.json (seeds, package versions, repository
commit, hardware, runtime methodology, tolerance configuration) so the numbers
below can be reproduced and the certificate claims checked -- see repro.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import closed_form as cf
import graphs
import repro
import tolerances as tolcfg
from core import (PathOracle, enumerate_paths, evader_alloc, game_value,
                  heuristic_allocation, relevant_nodes, solve_defender)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
C = ["#2B4C7E", "#D1622B", "#3E8E5A", "#A53F4B", "#6B5B95", "#8A7A62"]
BASELINES = ["uniform", "mincut", "pathcount", "betweenness", "greedy"]


# ----------------------------------------------------------------- utilities
def write_csv(name, rows, header=None):
    path = os.path.join(OUT, name)
    if not rows:
        return path
    header = header or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def write_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def table(rows, cols, title=""):
    if title:
        print(f"\n{title}")
    widths = [max(len(c), *(len(f"{r.get(c,'')}") for r in rows)) for c in cols]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(f"{r.get(c,'')}".ljust(w)
                               for c, w in zip(cols, widths)))


def fmt(v, d=6):
    return f"{v:.{d}f}"


def stats(G, S, D):
    orc = PathOracle(G, S, D)
    return len(G), G.number_of_edges(), len(orc.nodes), orc.n_paths_known


def cert_cols(r):
    """The certificate fields every claimed result must carry (item 24).

    UB, LB, absolute gap, relative gap (with its denominator named in the
    module docstring), the requested tolerance, the certificate status, the
    number of cutting-plane iterations and cuts, and the runtime.
    """
    c = r.get("certificate", {})
    return dict(
        log_ub=f"{c.get('log_ub', r.get('log_ub', float('nan'))):.10f}",
        log_lb=f"{c.get('log_lb', r.get('log_lb', float('nan'))):.10f}",
        V_ub=f"{c.get('value_ub', r.get('value', float('nan'))):.8g}",
        V_lb=f"{c.get('value_lb', float('nan')):.8g}",
        abs_gap=f"{c.get('abs_gap', r.get('gap', float('nan'))):.2e}",
        rel_gap=f"{c.get('rel_gap', float('nan')):.2e}",
        tol=f"{c.get('tolerance', r.get('tolerance', float('nan'))):.1e}",
        certified="yes" if c.get("certified") else "no",
        cert_status=c.get("status", "unknown"),
        iters=r.get("iterations", ""), cuts=r.get("n_cuts", ""),
        sec=f"{r.get('time', float('nan')):.2f}")


CERT_HEADER = ["log_ub", "log_lb", "V_ub", "V_lb", "abs_gap", "rel_gap",
               "tol", "certified", "iters", "cuts", "sec"]


def excess_pct(v_heuristic, v_star):
    """Excess success probability (%) = 100 * (V_heuristic / V* - 1).

    Defined once, here, so every table means the same thing by "excess".
    """
    return 100.0 * (v_heuristic / v_star - 1.0)


# =========================================================== E1  benchmarks
def E1_benchmarks():
    """Solver against the four closed-form families derived in FORMULATION.md."""
    print("\n" + "=" * 78)
    print("E1  ANALYTICAL BENCHMARKS  (solver vs closed form)")
    print("=" * 78)
    rows = []

    for n in [1, 2, 3, 4, 6]:
        for XA, XB in [(10.0, 10.0), (30.0, 5.0)]:
            r = solve_defender(*graphs.chain(n), XA, XB, tol=1e-11)
            rows.append(dict(family=f"chain n={n}", XA=XA, XB=XB,
                             solver=fmt(r["value"], 9),
                             closed_form=fmt(cf.chain_value(n, XA, XB), 9),
                             abs_err=f"{abs(r['value']-cf.chain_value(n,XA,XB)):.2e}",
                             **cert_cols(r)))

    for k, n in [(2, 2), (3, 2), (4, 3)]:
        r = solve_defender(*graphs.parallel_chains(k, n), 12.0, 8.0, tol=1e-11)
        t = cf.parallel_value(k, n, 12.0, 8.0)
        rows.append(dict(family=f"{k} parallel chains n={n}", XA=12.0, XB=8.0,
                         solver=fmt(r["value"], 9), closed_form=fmt(t, 9),
                         abs_err=f"{abs(r['value']-t):.2e}", **cert_cols(r)))

    for L, w in [(2, 3), (3, 3), (4, 2), (3, 4)]:
        r = solve_defender(*graphs.layered(L, w), 10.0, 10.0, tol=1e-11)
        t = cf.layered_value(L, w, 10.0, 10.0)
        rows.append(dict(family=f"layered L={L} w={w} ({w**L} paths)",
                         XA=10.0, XB=10.0, solver=fmt(r["value"], 9),
                         closed_form=fmt(t, 9),
                         abs_err=f"{abs(r['value']-t):.2e}", **cert_cols(r)))

    for Ls in [[1, 2], [1, 2, 4], [2, 3, 5]]:
        r = solve_defender(*graphs.unequal_branches(Ls), 10.0, 10.0,
                           tol=1e-11, max_iter=400)
        V, a = cf.unequal_branches_value(Ls, 10.0, 10.0)
        got = [sum(v for kk, v in r["x"].items() if kk.startswith(f"b{j}_"))
               for j in range(len(Ls))]
        rows.append(dict(family=f"branches {Ls}", XA=10.0, XB=10.0,
                         solver=fmt(r["value"], 9), closed_form=fmt(V, 9),
                         abs_err=f"{abs(r['value']-V):.2e}", **cert_cols(r),
                         note="budgets " + "/".join(fmt(g, 3) for g in got)
                              + " vs " + "/".join(fmt(t, 3) for t in a)))

    table(rows, ["family", "XA", "XB", "solver", "closed_form", "abs_err",
                 "abs_gap", "rel_gap", "tol", "certified", "iters"],
          "Closed-form validation (V* here is V_ub; see the certificate columns)")
    for r in rows:
        if r.get("note"):
            print(f"    {r['family']}: {r['note']}")
    write_csv("E1_benchmarks.csv", rows,
              ["family", "XA", "XB", "solver", "closed_form", "abs_err"]
              + CERT_HEADER + ["cert_status", "note"])
    worst = max(float(r["abs_err"]) for r in rows)
    n_cert = sum(r["certified"] == "yes" for r in rows)
    print(f"\n  worst absolute deviation from theory: {worst:.2e}")
    print(f"  optimality certificate closed on {n_cert}/{len(rows)} runs "
          f"(rows with certified=no are accurate values that are NOT proved "
          f"optimal)")
    return rows


# ====================================================== E2  worked example
def E2_lecture():
    """The professor's topology, solved and reported node by node."""
    print("\n" + "=" * 78)
    print("E2  WORKED EXAMPLE  (topology from the problem statement)")
    print("=" * 78)
    G, S, D = graphs.lecture_example()
    XA = XB = 10.0
    r = solve_defender(G, S, D, XA, XB, tol=1e-12, max_iter=400)
    orc = r["oracle"]

    print(f"\n  edges           : {sorted(G.edges())}")
    print(f"  contested nodes : {orc.nodes}")
    print(f"  S-D paths       : {[' -> '.join(p) for p in orc.paths]}")
    print(f"\n  OPTIMAL DEFENCE x*  (XA = {XA})")
    for v in orc.nodes:
        print(f"      x*[{v}] = {r['x'][v]:.6f}")
    c = r["certificate"]
    print(f"\n  game value V* = {r['value']:.9f}   (this is V_ub)")
    print(f"      bracket        [{c['value_lb']:.9f}, {c['value_ub']:.9f}]")
    print(f"      log bracket    [{c['log_lb']:.10f}, {c['log_ub']:.10f}]")
    print(f"      abs gap        {c['abs_gap']:.2e}      "
          f"rel gap {c['rel_gap']:.2e}  (/{c['rel_gap_denominator']})")
    print(f"      tolerance      {c['tolerance']:.1e}")
    print(f"      certificate    {c['status'].upper()}   "
          f"({c['iterations']} iterations, {c['n_cuts']} cuts, "
          f"{c['runtime_sec']:.2f}s)")
    print(f"  evader best response    : {' -> '.join(r['best_path'])}")
    print(f"  evader split y*         : "
          + ", ".join(f"{k}={v:.4f}" for k, v in r["y"].items()))

    print("\n  EVERY PATH AT x*  (equalisation across the paths the evader uses)")
    rows = []
    for p, inter in zip(orc.paths, orc.interiors):
        sol = evader_alloc(orc._sub(r["x_vec"], inter), XB)
        rows.append(dict(path=" -> ".join(p), contested=len(inter),
                         value=fmt(math.exp(sol["logval"]), 9),
                         y=", ".join(f"{v}:{yy:.3f}"
                                     for v, yy in zip(inter, sol["y"]))))
    table(rows, ["path", "contested", "value", "y"])

    base = []
    for kind in BASELINES:
        xh = heuristic_allocation(G, S, D, XA, kind, XB=XB, oracle=orc)
        vh = game_value(G, S, D, xh, XB, oracle=orc)["value"]
        base.append(dict(strategy=kind, value=fmt(vh, 9),
                         excess=f"{max(excess_pct(vh, r['value']), 0.0):.2f}%",
                         x=", ".join(f"{k}:{v:.3f}" for k, v in xh.items())))
    base.insert(0, dict(strategy="OPTIMAL", value=fmt(r["value"], 9),
                        excess="0.00%",
                        x=", ".join(f"{k}:{v:.3f}" for k, v in r["x"].items())))
    table(base, ["strategy", "value", "excess", "x"],
          "Optimal defence vs baselines. excess = excess success probability "
          "(%) = 100*(V_heuristic/V* - 1)")

    write_json("E2_lecture.json", {"x": r["x"], "y": r["y"],
                                   "value": r["value"],
                                   "certificate": r["certificate"],
                                   "residuals": r["residuals"],
                                   "best_path": r["best_path"],
                                   "excess_metric":
                                       "100 * (V_heuristic / V_star - 1)",
                                   "paths": rows, "baselines": base})
    _draw(G, S, D, r["x"], "E2_lecture_graph.png",
          f"Optimal defence on the lecture topology  (V* = {r['value']:.4f})")
    return r


# ===================================================== E3  topology study
def E3_topologies():
    print("\n" + "=" * 78)
    print("E3  TOPOLOGY STUDY  (XA = XB = 10)")
    print("=" * 78)
    XA = XB = 10.0
    rows, keep = [], {}
    for name, build in graphs.CATALOG.items():
        G, S, D = build()
        nV, nE, nC, nP = stats(G, S, D)
        t0 = time.time()
        r = solve_defender(G, S, D, XA, XB, tol=1e-9, max_iter=400)
        row = dict(graph=name, nodes=nV, edges=nE, contested=nC, paths=nP,
                   V_star=fmt(r["value"], 8), **cert_cols(r),
                   worst_path=" -> ".join(r["best_path"]))
        for kind in BASELINES:
            xh = heuristic_allocation(G, S, D, XA, kind, XB=XB,
                                      oracle=r["oracle"])
            vh = game_value(G, S, D, xh, XB, oracle=r["oracle"])["value"]
            e = excess_pct(vh, r["value"])
            row[kind] = f"{0.0 if abs(e) < 1e-6 else e:.1f}%"
        rows.append(row)
        keep[name] = r
        print(f"  solved {name:<14s} V*={r['value']:.6f}  "
              f"abs_gap={r['gap']:.1e}  certified={row['certified']}  "
              f"{row['sec']}s")

    table(rows, ["graph", "nodes", "edges", "contested", "paths", "V_star",
                 "abs_gap", "rel_gap", "tol", "certified", "iters", "cuts",
                 "sec"],
          "Optima with their certificates (V_star is V_ub; a row with "
          "certified=no is NOT proved optimal)")
    table(rows, ["graph", "V_star"] + BASELINES,
          "Excess success probability (%) = 100*(V_heuristic/V* - 1) "
          "(0% = matches the optimum, higher = worse for the defender)")
    write_csv("E3_topologies.csv", rows,
              ["graph", "nodes", "edges", "contested", "paths", "V_star"]
              + CERT_HEADER + ["cert_status", "worst_path"] + BASELINES)

    # allocation profiles for three instructive instances
    picks = ["grid-4x4", "grid-bypass", "random-12"]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 3.8))
    for ax, name in zip(np.atleast_1d(axes), picks):
        r = keep[name]
        nodes = r["nodes"]
        xs = np.array([r["x"][v] for v in nodes])
        uni = np.full(len(nodes), XA / len(nodes))
        idx = np.arange(len(nodes))
        ax.bar(idx - 0.2, xs, 0.4, color=C[0], label="optimal $x^*$")
        ax.bar(idx + 0.2, uni, 0.4, color=C[1], alpha=0.75, label="uniform")
        ax.set_title(f"{name}   $V^*$ = {r['value']:.4f}")
        ax.set_xticks(idx)
        ax.set_xticklabels(nodes, rotation=90, fontsize=6)
        ax.set_ylabel("defender budget")
        ax.legend(fontsize=8)
    fig.suptitle("Optimal defence concentrates on structural bottlenecks, "
                 "not uniformly", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E3_allocations.png"), bbox_inches="tight")
    plt.close(fig)

    _draw(*graphs.grid_bypass(4, 4, 2), keep["grid-bypass"]["x"],
          "E3_gridbypass_graph.png",
          "grid + short bypass: the defender guards the cheap route "
          f"(V* = {keep['grid-bypass']['value']:.4f})")
    return rows, keep


# ================================================== E4  budget sensitivity
def E4_budget_sweep():
    print("\n" + "=" * 78)
    print("E4  BUDGET SENSITIVITY   V* as a function of XA/XB")
    print("=" * 78)
    XB = 10.0
    ratios = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    picks = {"chain-4": graphs.CATALOG["chain-4"],
             "parallel-3x3": graphs.CATALOG["parallel-3x3"],
             "grid-4x4": graphs.CATALOG["grid-4x4"],
             "layered-3x3": graphs.CATALOG["layered-3x3"],
             "random-12": graphs.CATALOG["random-12"]}
    rows, curves = [], {}
    for name, build in picks.items():
        G, S, D = build()
        orc = PathOracle(G, S, D)
        vals = []
        for rr in ratios:
            r = solve_defender(G, S, D, float(rr * XB), XB, tol=1e-9,
                               max_iter=300, oracle=orc)
            vals.append(r["value"])
            rows.append(dict(graph=name, ratio=rr, XA=rr * XB, XB=XB,
                             V_star=fmt(r["value"], 8), **cert_cols(r)))
        curves[name] = vals
        print(f"  {name:<14s} " + "  ".join(f"{v:.4f}" for v in vals))

    write_csv("E4_budget_sweep.csv", rows,
              ["graph", "ratio", "XA", "XB", "V_star"] + CERT_HEADER
              + ["cert_status"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, vals) in enumerate(curves.items()):
        axes[0].plot(ratios, vals, "o-", color=C[i % len(C)], label=name, ms=4)
        axes[1].semilogy(ratios, np.maximum(vals, 1e-12), "o-",
                         color=C[i % len(C)], label=name, ms=4)
    for ax, ttl in zip(axes, ["linear scale", "log scale"]):
        ax.set_xlabel(r"budget ratio  $\bar x_A/\bar x_B$")
        ax.set_ylabel(r"game value  $V^*$")
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    fig.suptitle("Evader success falls with defender budget — how fast depends "
                 "on topology, not on path count", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E4_budget_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    return rows


# ====================================================== E5  baseline study
def E5_baselines():
    print("\n" + "=" * 78)
    print("E5  BASELINE COMPARISON  (20 random DAGs, XA = XB = 10)")
    print("=" * 78)
    XA = XB = 10.0
    exc = {k: [] for k in BASELINES}
    rows = []
    for seed in range(20):
        G, S, D = graphs.random_dag(12, 0.26, seed=seed)
        r = solve_defender(G, S, D, XA, XB, tol=1e-9, max_iter=300)
        row = dict(seed=seed, paths=r["n_paths"], V_star=fmt(r["value"], 8),
                   **cert_cols(r))
        for kind in BASELINES:
            xh = heuristic_allocation(G, S, D, XA, kind, XB=XB,
                                      oracle=r["oracle"])
            vh = game_value(G, S, D, xh, XB, oracle=r["oracle"])["value"]
            e = excess_pct(vh, r["value"])
            e = 0.0 if abs(e) < 1e-6 else e
            exc[kind].append(e)
            row[kind] = f"{e:.1f}%"
        rows.append(row)
    table(rows, ["seed", "paths", "V_star", "abs_gap", "tol", "certified"]
          + BASELINES)
    n_cert = sum(r["certified"] == "yes" for r in rows)
    print(f"\n  certificate closed on {n_cert}/{len(rows)} instances at "
          f"tol {rows[0]['tol']}")
    write_csv("E5_baselines.csv", rows,
              ["seed", "paths", "V_star"] + CERT_HEADER + ["cert_status"]
              + BASELINES)

    summary = [dict(strategy=k, mean=f"{np.mean(v):.1f}%",
                    median=f"{np.median(v):.1f}%", worst=f"{max(v):.1f}%",
                    best=f"{min(v):.1f}%",
                    metric="100*(V_heuristic/V_star - 1)")
               for k, v in exc.items()]
    table(summary, ["strategy", "mean", "median", "best", "worst"],
          "Excess success probability (%) = 100*(V_heuristic/V* - 1), over the "
          "20 instances above.  V* is V_ub, so these are conservative.")
    write_csv("E5_baseline_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        ax.boxplot([exc[k] for k in BASELINES], tick_labels=BASELINES,
                   showmeans=True)
    except TypeError:                                  # matplotlib < 3.9
        ax.boxplot([exc[k] for k in BASELINES], labels=BASELINES, showmeans=True)
    ax.axhline(0, color=C[3], lw=1.4, label="optimum $V^*$")
    ax.set_ylabel(r"excess success probability  $100(V_{\rm heur}/V^*-1)$  [%]")
    ax.set_title("No heuristic rule recovers the optimal defence\n"
                 "(20 random DAGs, 12 interior nodes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E5_baselines.png"), bbox_inches="tight")
    plt.close(fig)
    return summary


# ======================================================= E6  convergence
def E6_convergence():
    print("\n" + "=" * 78)
    print("E6  CUTTING-PLANE CONVERGENCE  (certified LB/UB bracket)")
    print("=" * 78)
    picks = [("unequal-1-2-4", graphs.unequal_branches([1, 2, 4])),
             ("random-12", graphs.random_dag(12, 0.25, seed=7)),
             ("grid-bypass", graphs.grid_bypass(4, 4, 2))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for i, (name, (G, S, D)) in enumerate(picks):
        r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-10, max_iter=120)
        h = r["history"]
        it = [d["iter"] for d in h]
        axes[0].plot(it, [math.exp(d["ub"]) for d in h], "-",
                     color=C[i], label=f"{name}: UB")
        axes[0].plot(it, [math.exp(max(d["lb"], -50)) for d in h], "--",
                     color=C[i], label=f"{name}: LB")
        axes[1].semilogy(it, [max(d["gap"], 1e-16) for d in h], "o-",
                         color=C[i], ms=3, label=name)
        rows.append(dict(graph=name, V_star=fmt(r["value"], 9),
                         **cert_cols(r)))
        print(f"  {name:<14s} {r['iterations']:3d} iters, "
              f"{h[-1]['n_cuts']:3d} cuts, abs_gap {r['gap']:.2e}, "
              f"certified={'yes' if r['certified'] else 'no'}")
    axes[0].set_xlabel("cutting-plane iteration")
    axes[0].set_ylabel("bracket on $V^*$")
    axes[0].set_title("upper / lower bound")
    axes[0].legend(fontsize=7, ncol=1)
    axes[1].set_xlabel("cutting-plane iteration")
    axes[1].set_ylabel(r"$\log$-gap  UB $-$ LB")
    axes[1].set_title("certified optimality gap")
    axes[1].legend(fontsize=8)
    fig.suptitle("A rigorous UB/LB bracket, not a 'the iterates stopped "
                 "moving' criterion", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E6_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    table(rows, ["graph", "V_star", "log_ub", "log_lb", "abs_gap", "rel_gap",
                 "tol", "certified", "iters", "cuts", "sec"],
          "A global optimality certificate is claimed only where "
          "abs_gap <= tol (certified=yes)")
    write_csv("E6_convergence.csv", rows,
              ["graph", "V_star"] + CERT_HEADER + ["cert_status"])
    return rows


# ========================================================== E7  scaling
def E7_scaling():
    """Observed performance on layered graphs, where the path count explodes.

    SCOPE.  What follows is measured behaviour on ONE tested graph family
    (complete layered DAGs, L layers of width w) at the instance sizes listed,
    plus random DAGs up to 30 interior nodes.  It shows how this implementation
    behaved on those instances on the hardware recorded in
    out/RUN_METADATA.json.  It is NOT a complexity result and does not
    generalise to arbitrary graphs: no claim is made here about worst-case
    running time, and the path-selection subproblem is closely related to
    maximum-reliability path problems known to be NP-hard.
    """
    print("\n" + "=" * 78)
    print("E7  OBSERVED SCALING on the tested layered family")
    print("=" * 78)
    print("  Measured performance on the graph family and instance sizes")
    print("  listed below -- not a general complexity claim.")
    rows = []
    cases = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (8, 3), (10, 3),
             (6, 4), (8, 4), (10, 4), (12, 4), (8, 5), (10, 5)]
    for L, w in cases:
        G, S, D = graphs.layered(L, w)
        n_paths = w ** L
        orc = PathOracle(G, S, D, mode="ksp")
        t0 = time.time()
        r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-9, max_iter=200,
                           oracle=orc)
        dt = time.time() - t0
        exact = cf.layered_value(L, w, 10.0, 10.0)
        touched = orc.best_response(r["x_vec"], 10.0)["n_evaluated"]
        cc = cert_cols(r)
        cc["sec"] = f"{dt:.2f}"
        rows.append(dict(graph=f"layered L={L} w={w}", nodes=len(G),
                         paths=f"{n_paths:,}", contested=L * w,
                         V_star=f"{r['value']:.3e}", closed_form=f"{exact:.3e}",
                         abs_err=f"{abs(r['value']-exact):.1e}",
                         paths_touched=touched, **cc))
        print(f"  L={L} w={w}: {n_paths:>12,} S-D paths, solver evaluated "
              f"{touched} of them, V*={r['value']:.3e}, {dt:.2f}s, "
              f"certified={cc['certified']}")
    table(rows, ["graph", "nodes", "contested", "paths", "V_star",
                 "closed_form", "abs_err", "abs_gap", "tol", "certified",
                 "paths_touched", "iters", "sec"])
    print("  READING THIS TABLE")
    print("  * V_star is the UPPER bound; on these instances it matches the")
    print("    closed form to machine precision (abs_err column).")
    print("  * abs_gap is the residual optimality CERTIFICATE.  On the widest")
    print("    instances Kelley's lower bound has not finished closing within")
    print("    the 200-iteration cap, so those rows read certified=no: an")
    print("    accurate value that is NOT proved optimal.")
    print("  * paths_touched and sec are observed behaviour on this graph")
    print("    family at these sizes, on the hardware in RUN_METADATA.json.")
    print("    They are not a complexity result and do not generalise.")
    write_csv("E7_scaling.csv", rows,
              ["graph", "nodes", "contested", "paths", "V_star", "closed_form",
               "abs_err", "paths_touched"] + CERT_HEADER + ["cert_status"])

    np_ = [float(r["paths"].replace(",", "")) for r in rows]
    sec = [float(r["sec"]) for r in rows]
    touched = [r["paths_touched"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(np_, sec, "o", color=C[0], ms=6)
    axes[0].set_xlabel("number of distinct S-D paths")
    axes[0].set_ylabel("solve time [s]")
    axes[0].set_title("observed runtime vs path count\n(tested layered family)")
    axes[1].loglog(np_, touched, "o", color=C[1], ms=6, label="paths evaluated")
    axes[1].loglog(np_, np_, "--", color="grey", lw=1, label="full enumeration")
    axes[1].set_xlabel("number of distinct S-D paths")
    axes[1].set_ylabel("paths actually evaluated")
    axes[1].set_title("path generation evaluates few paths here")
    axes[1].legend(fontsize=8)
    fig.suptitle("Observed scaling on the tested layered family — measured "
                 "behaviour at these sizes, not a complexity result",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E7_scaling.png"), bbox_inches="tight")
    plt.close(fig)

    # size scaling on random DAGs
    rows2 = []
    for n in [8, 12, 16, 20, 25, 30]:
        ts, vs, its, cs = [], [], [], []
        for seed in range(3):
            G, S, D = graphs.random_dag(n, 0.22, seed=100 + seed)
            t0 = time.time()
            r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-8, max_iter=250)
            ts.append(time.time() - t0)
            vs.append(r["value"])
            its.append(r["iterations"])
            cs.append(bool(r["certified"]))
        rows2.append(dict(n_interior=n, mean_sec=f"{np.mean(ts):.2f}",
                          mean_iters=f"{np.mean(its):.1f}",
                          mean_V=f"{np.mean(vs):.4f}",
                          n_certified=f"{sum(cs)}/3", tol="1.0e-08",
                          seeds="100,101,102"))
        print(f"  random DAG n={n}: {np.mean(ts):.2f}s, "
              f"{np.mean(its):.1f} iterations")
    table(rows2, ["n_interior", "mean_iters", "mean_sec", "mean_V",
                  "n_certified"],
          "Random DAGs, mean of 3 seeds (observed on the tested instances)")
    write_csv("E7_random_scaling.csv", rows2)
    return rows, rows2


# ================================================= E8  contest intensity
def E8_intensity():
    print("\n" + "=" * 78)
    print("E8  CONTEST INTENSITY m   (ratio-form CSF  p = y^m/(x^m+y^m))")
    print("=" * 78)
    print("  Convexity -- and therefore the validity of the lower bound, and")
    print("  so the optimality certificate -- is proved only for 0 < m <= 1.")
    print("  For m > 1 convexity provably FAILS (FORMULATION.md S3, Prop. 5),")
    print("  so those rows are EXPLORATORY / NON-CERTIFIED: the solver still")
    print("  runs, but no optimality claim is attached to its output until a")
    print("  separate global method and proof are supplied.")
    ms = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    picks = ["chain-4", "parallel-3x3", "grid-4x4"]
    rows, curves = [], {}
    for name in picks:
        G, S, D = graphs.CATALOG[name]()
        vals = []
        for m in ms:
            orc = PathOracle(G, S, D, m=m)
            r = solve_defender(G, S, D, 10.0, 10.0, m=m, tol=1e-8,
                               max_iter=150, oracle=orc)
            vals.append(r["value"])
            rows.append(dict(graph=name, m=m, V=fmt(r["value"], 8),
                             **cert_cols(r),
                             regime="certifiable (convexity proved)" if m <= 1.0
                             else "EXPLORATORY / NON-CERTIFIED "
                                  "(convexity provably fails)"))
        curves[name] = vals
        print(f"  {name:<14s} " + "  ".join(f"{v:.4f}" for v in vals))
    table(rows, ["graph", "m", "V", "abs_gap", "tol", "certified", "regime"],
          "Contest intensity sweep.  m > 1 rows are exploratory: convexity "
          "fails there, so no certificate is claimed at any gap.")
    write_csv("E8_intensity.csv", rows,
              ["graph", "m", "V"] + CERT_HEADER + ["cert_status", "regime"])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, vals) in enumerate(curves.items()):
        ax.plot(ms, vals, "o-", color=C[i], label=name, ms=5)
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.text(1.02, 0.95, "convexity proof valid to the left;\n"
                        "m > 1 exploratory / non-certified",
            transform=ax.get_xaxis_transform(), fontsize=8, va="top")
    ax.set_xlabel("contest intensity $m$")
    ax.set_ylabel("game value $V^*$")
    ax.set_title("Contest intensity amplifies the CONCENTRATOR's advantage\n"
                 "(equal budgets; flat on a single chain, rising where the\n"
                 "defender must spread over parallel routes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E8_intensity.png"), bbox_inches="tight")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- drawing
def _draw(G, S, D, x, fname, title):
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.multipartite_layout(
                G, subset_key={v: i for i, v in
                               enumerate(nx.topological_sort(G))})
        except Exception:
            pos = nx.spring_layout(G, seed=3)
        pos = nx.spring_layout(G, pos=pos, seed=3, iterations=60)
    xs = {v: float(x.get(v, 0.0)) for v in G}
    mx = max(xs.values()) or 1.0
    sizes = [160 + 1500 * xs[v] / mx for v in G]
    cols = ["#D1622B" if v in (S, D) else
            plt.cm.Blues(0.25 + 0.7 * xs[v] / mx) for v in G]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.45,
                           node_size=600, width=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=cols,
                           edgecolors="#333333", linewidths=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    for v in G:
        if v not in (S, D) and xs[v] > 1e-6:
            ax.annotate(f"{xs[v]:.2f}", pos[v], textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=6.5,
                        color="#8A2B00")
    ax.set_title(title + "\n(node size and shade = defender budget $x^*_i$)")
    ax.axis("off")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- main
ALL = {"E1": E1_benchmarks, "E2": E2_lecture, "E3": E3_topologies,
       "E4": E4_budget_sweep, "E5": E5_baselines, "E6": E6_convergence,
       "E7": E7_scaling, "E8": E8_intensity}


def main(argv):
    want = [a.upper() for a in argv[1:]] or list(ALL)
    t0 = time.time()
    for key in want:
        if key not in ALL:
            print(f"unknown experiment {key}")
            continue
        ALL[key]()
    meta = repro.write_metadata(OUT, extra={"experiments_run": want,
                                            "total_sec": time.time() - t0})
    print("\n" + "=" * 78)
    print(f"  all requested experiments finished in {time.time()-t0:.1f}s")
    print(f"  tables and figures written to {OUT}")
    print(f"  reproducibility metadata written to {meta}")
    print("    (random seeds, package versions, repository commit, hardware,")
    print("     runtime methodology, and the full tolerance configuration)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
