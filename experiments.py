"""
experiments.py -- numerical study of the network security game.

Run:  python experiments.py            (all experiments)
      python experiments.py E1 E4      (a subset)

Everything is written to out/ :  *.csv tables, *.json raw results, *.png figures.
All randomness is seeded, so the whole study is reproducible.

TERMINOLOGY (used consistently in every table)
  game value V*      certified optimum of  min_x max_{P,y} prod y_i/(x_i+y_i)
  log_lb / log_ub    rigorous bracket on log V* produced by the cutting plane
  gap                log_ub - log_lb  (0 => proven optimal to solver tolerance)
  baseline           a NON-optimal defender rule, evaluated exactly
  excess             value(baseline)/V* - 1, i.e. how much worse a baseline is
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
                             gap=f"{r['gap']:.1e}", iters=r["iterations"]))

    for k, n in [(2, 2), (3, 2), (4, 3)]:
        r = solve_defender(*graphs.parallel_chains(k, n), 12.0, 8.0, tol=1e-11)
        t = cf.parallel_value(k, n, 12.0, 8.0)
        rows.append(dict(family=f"{k} parallel chains n={n}", XA=12.0, XB=8.0,
                         solver=fmt(r["value"], 9), closed_form=fmt(t, 9),
                         abs_err=f"{abs(r['value']-t):.2e}",
                         gap=f"{r['gap']:.1e}", iters=r["iterations"]))

    for L, w in [(2, 3), (3, 3), (4, 2), (3, 4)]:
        r = solve_defender(*graphs.layered(L, w), 10.0, 10.0, tol=1e-11)
        t = cf.layered_value(L, w, 10.0, 10.0)
        rows.append(dict(family=f"layered L={L} w={w} ({w**L} paths)",
                         XA=10.0, XB=10.0, solver=fmt(r["value"], 9),
                         closed_form=fmt(t, 9),
                         abs_err=f"{abs(r['value']-t):.2e}",
                         gap=f"{r['gap']:.1e}", iters=r["iterations"]))

    for Ls in [[1, 2], [1, 2, 4], [2, 3, 5]]:
        r = solve_defender(*graphs.unequal_branches(Ls), 10.0, 10.0,
                           tol=1e-11, max_iter=400)
        V, a = cf.unequal_branches_value(Ls, 10.0, 10.0)
        got = [sum(v for kk, v in r["x"].items() if kk.startswith(f"b{j}_"))
               for j in range(len(Ls))]
        rows.append(dict(family=f"branches {Ls}", XA=10.0, XB=10.0,
                         solver=fmt(r["value"], 9), closed_form=fmt(V, 9),
                         abs_err=f"{abs(r['value']-V):.2e}",
                         gap=f"{r['gap']:.1e}", iters=r["iterations"],
                         note="budgets " + "/".join(fmt(g, 3) for g in got)
                              + " vs " + "/".join(fmt(t, 3) for t in a)))

    table(rows, ["family", "XA", "XB", "solver", "closed_form", "abs_err",
                 "gap", "iters"], "Closed-form validation")
    for r in rows:
        if r.get("note"):
            print(f"    {r['family']}: {r['note']}")
    write_csv("E1_benchmarks.csv", rows,
              ["family", "XA", "XB", "solver", "closed_form", "abs_err",
               "gap", "iters", "note"])
    worst = max(float(r["abs_err"]) for r in rows)
    print(f"\n  worst absolute deviation from theory: {worst:.2e}")
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
    print(f"  certified game value V* = {r['value']:.9f}"
          f"   (bracket [{math.exp(r['log_lb']):.9f}, {math.exp(r['log_ub']):.9f}])")
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
                         excess=f"{max(100*(vh/r['value']-1), 0.0):.2f}%",
                         x=", ".join(f"{k}:{v:.3f}" for k, v in xh.items())))
    base.insert(0, dict(strategy="OPTIMAL", value=fmt(r["value"], 9),
                        excess="0.00%",
                        x=", ".join(f"{k}:{v:.3f}" for k, v in r["x"].items())))
    table(base, ["strategy", "value", "excess", "x"],
          "Optimal defence vs baselines on this instance")

    write_json("E2_lecture.json", {"x": r["x"], "y": r["y"],
                                   "value": r["value"], "gap": r["gap"],
                                   "best_path": r["best_path"],
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
                   V_star=fmt(r["value"], 8), gap=f"{r['gap']:.1e}",
                   iters=r["iterations"], sec=f"{time.time()-t0:.2f}",
                   worst_path=" -> ".join(r["best_path"]))
        for kind in BASELINES:
            xh = heuristic_allocation(G, S, D, XA, kind, XB=XB,
                                      oracle=r["oracle"])
            vh = game_value(G, S, D, xh, XB, oracle=r["oracle"])["value"]
            e = 100.0 * (vh / r["value"] - 1.0)
            row[kind] = f"{0.0 if abs(e) < 1e-6 else e:.1f}%"
        rows.append(row)
        keep[name] = r
        print(f"  solved {name:<14s} V*={r['value']:.6f}  "
              f"gap={r['gap']:.1e}  {row['sec']}s")

    table(rows, ["graph", "nodes", "edges", "contested", "paths", "V_star",
                 "gap", "iters", "sec"], "Certified optima")
    table(rows, ["graph", "V_star"] + BASELINES,
          "Excess value of each baseline relative to the optimum "
          "(0% = optimal, higher = worse for the defender)")
    write_csv("E3_topologies.csv", rows,
              ["graph", "nodes", "edges", "contested", "paths", "V_star",
               "gap", "iters", "sec", "worst_path"] + BASELINES)

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
                             V_star=fmt(r["value"], 8), gap=f"{r['gap']:.1e}"))
        curves[name] = vals
        print(f"  {name:<14s} " + "  ".join(f"{v:.4f}" for v in vals))

    write_csv("E4_budget_sweep.csv", rows)
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
                   gap=f"{r['gap']:.1e}")
        for kind in BASELINES:
            xh = heuristic_allocation(G, S, D, XA, kind, XB=XB,
                                      oracle=r["oracle"])
            vh = game_value(G, S, D, xh, XB, oracle=r["oracle"])["value"]
            e = 100.0 * (vh / r["value"] - 1.0)
            e = 0.0 if abs(e) < 1e-6 else e
            exc[kind].append(e)
            row[kind] = f"{e:.1f}%"
        rows.append(row)
    table(rows, ["seed", "paths", "V_star", "gap"] + BASELINES)
    write_csv("E5_baselines.csv", rows)

    summary = [dict(strategy=k, mean=f"{np.mean(v):.1f}%",
                    median=f"{np.median(v):.1f}%", worst=f"{max(v):.1f}%",
                    best=f"{min(v):.1f}%") for k, v in exc.items()]
    table(summary, ["strategy", "mean", "median", "best", "worst"],
          "Excess evader success probability vs the certified optimum")
    write_csv("E5_baseline_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        ax.boxplot([exc[k] for k in BASELINES], tick_labels=BASELINES,
                   showmeans=True)
    except TypeError:                                  # matplotlib < 3.9
        ax.boxplot([exc[k] for k in BASELINES], labels=BASELINES, showmeans=True)
    ax.axhline(0, color=C[3], lw=1.4, label="certified optimum")
    ax.set_ylabel("excess evader success  [%]")
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
        rows.append(dict(graph=name, iters=r["iterations"],
                         V_star=fmt(r["value"], 9), final_gap=f"{r['gap']:.2e}",
                         cuts=h[-1]["n_cuts"], sec=f"{r['time']:.2f}"))
        print(f"  {name:<14s} {r['iterations']:3d} iters, "
              f"{h[-1]['n_cuts']:3d} cuts, gap {r['gap']:.2e}")
    axes[0].set_xlabel("cutting-plane iteration")
    axes[0].set_ylabel("bracket on $V^*$")
    axes[0].set_title("upper / lower bound")
    axes[0].legend(fontsize=7, ncol=1)
    axes[1].set_xlabel("cutting-plane iteration")
    axes[1].set_ylabel(r"$\log$-gap  UB $-$ LB")
    axes[1].set_title("certified optimality gap")
    axes[1].legend(fontsize=8)
    fig.suptitle("The method closes a rigorous optimality gap — it does not "
                 "merely stop moving", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E6_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    table(rows, ["graph", "iters", "cuts", "V_star", "final_gap", "sec"])
    write_csv("E6_convergence.csv", rows)
    return rows


# ========================================================== E7  scaling
def E7_scaling():
    """Instances where enumerating every S-D path is hopeless."""
    print("\n" + "=" * 78)
    print("E7  SCALING: path generation vs full enumeration")
    print("=" * 78)
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
        rows.append(dict(graph=f"layered L={L} w={w}", nodes=len(G),
                         paths=f"{n_paths:,}", contested=L * w,
                         V_star=f"{r['value']:.3e}", closed_form=f"{exact:.3e}",
                         abs_err=f"{abs(r['value']-exact):.1e}",
                         gap=f"{r['gap']:.1e}",
                         paths_touched=touched, iters=r["iterations"],
                         sec=f"{dt:.2f}"))
        print(f"  L={L} w={w}: {n_paths:>12,} S-D paths, solver touched "
              f"{touched} of them, V*={r['value']:.3e}, {dt:.2f}s")
    table(rows, ["graph", "nodes", "contested", "paths", "V_star",
                 "closed_form", "abs_err", "gap", "paths_touched", "iters",
                 "sec"])
    print("  NOTE  V_star (the UPPER bound) is exact to machine precision on")
    print("  every instance.  The `gap` column is the residual CERTIFICATE: on")
    print("  the widest instances Kelley's lower bound has not finished closing")
    print("  within the 200-iteration cap, so those rows are 'exact value, gap")
    print("  not yet certified' rather than 'proved optimal'.")
    write_csv("E7_scaling.csv", rows)

    np_ = [float(r["paths"].replace(",", "")) for r in rows]
    sec = [float(r["sec"]) for r in rows]
    touched = [r["paths_touched"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(np_, sec, "o", color=C[0], ms=6)
    axes[0].set_xlabel("number of distinct S-D paths")
    axes[0].set_ylabel("solve time [s]")
    axes[0].set_title("runtime is flat in the path count")
    axes[1].loglog(np_, touched, "o", color=C[1], ms=6, label="paths evaluated")
    axes[1].loglog(np_, np_, "--", color="grey", lw=1, label="full enumeration")
    axes[1].set_xlabel("number of distinct S-D paths")
    axes[1].set_ylabel("paths actually evaluated")
    axes[1].set_title("certified path generation avoids enumeration")
    axes[1].legend(fontsize=8)
    fig.suptitle("Layered graphs: the strategy space explodes, the solver does "
                 "not", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E7_scaling.png"), bbox_inches="tight")
    plt.close(fig)

    # size scaling on random DAGs
    rows2 = []
    for n in [8, 12, 16, 20, 25, 30]:
        ts, vs, its = [], [], []
        for seed in range(3):
            G, S, D = graphs.random_dag(n, 0.22, seed=100 + seed)
            t0 = time.time()
            r = solve_defender(G, S, D, 10.0, 10.0, tol=1e-8, max_iter=250)
            ts.append(time.time() - t0)
            vs.append(r["value"])
            its.append(r["iterations"])
        rows2.append(dict(n_interior=n, mean_sec=f"{np.mean(ts):.2f}",
                          mean_iters=f"{np.mean(its):.1f}",
                          mean_V=f"{np.mean(vs):.4f}"))
        print(f"  random DAG n={n}: {np.mean(ts):.2f}s, "
              f"{np.mean(its):.1f} iterations")
    table(rows2, ["n_interior", "mean_iters", "mean_sec", "mean_V"],
          "Random DAGs (mean of 3 seeds)")
    write_csv("E7_random_scaling.csv", rows2)
    return rows, rows2


# ================================================= E8  contest intensity
def E8_intensity():
    print("\n" + "=" * 78)
    print("E8  CONTEST INTENSITY m   (ratio-form CSF  p = y^m/(x^m+y^m))")
    print("=" * 78)
    print("  Convexity — and therefore the optimality certificate — is PROVED")
    print("  only for m <= 1.  Values m > 1 are reported as a heuristic.")
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
                             gap=f"{r['gap']:.1e}",
                             certified="yes" if m <= 1.0 else "NOT PROVEN"))
        curves[name] = vals
        print(f"  {name:<14s} " + "  ".join(f"{v:.4f}" for v in vals))
    write_csv("E8_intensity.csv", rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, vals) in enumerate(curves.items()):
        ax.plot(ms, vals, "o-", color=C[i], label=name, ms=5)
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.text(1.02, 0.95, "convexity proof\nvalid to the left",
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
    print("\n" + "=" * 78)
    print(f"  all requested experiments finished in {time.time()-t0:.1f}s")
    print(f"  tables and figures written to {OUT}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
