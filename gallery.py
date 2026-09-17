"""Gallery: assorted topologies, their optimal defence and the evader's best path."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import graphs
from core import solve_defender

XA = XB = 10.0


def build_cases():
    cases = []

    G = nx.DiGraph(); nx.add_path(G, ["S", "v1", "v2", "v3", "D"])
    cases.append(("Chain (3 nodes)\nuniform IS optimal", G, "S", "D", None))

    G, S, D = graphs.diamond()
    cases.append(("Diamond\ntwo symmetric branches", G, S, D, None))

    G, S, D = graphs.lecture_example()
    cases.append(("Lecture example\nasymmetric branches", G, S, D,
                  {"S": (0, 1), "i1": (1, 2), "i2": (2, 2), "j1": (1.5, 0), "D": (3, 1)}))

    G, S, D = graphs.unequal_branches([1, 2, 4])
    cases.append(("Unequal branches 1/2/4\nshort branch costs most", G, S, D, None))

    G, S, D = graphs.layered(3, 3)
    cases.append(("Layered 3x3\n27 paths, uniform optimal", G, S, D, None))

    G, S, D = graphs.grid_dag(3, 3)
    cases.append(("Grid 3x3\nregular -> uniform", G, S, D, None))

    G, S, D = graphs.grid_bypass(3, 3, 2)
    cases.append(("Grid + short bypass\nbypass distorts defence", G, S, D, None))

    G, S, D = graphs.random_dag(9, 0.3, seed=5)
    cases.append(("Random DAG (9 nodes)\nirregular", G, S, D, None))

    G = nx.DiGraph()
    nx.add_path(G, ["S", "a", "b", "c", "D"])
    G.add_edge("c", "a"); G.add_edge("a", "c")
    cases.append(("CYCLIC graph\ncycle a->b->c->a", G, "S", "D",
                  {"S": (0, 0), "a": (1, 0), "b": (2, 1), "c": (2, -1), "D": (3, 0)}))

    return cases


def layout(G, S, D, fixed):
    if fixed:
        return fixed
    try:
        # layer by longest distance from S -> reads left to right
        depth = {}
        for v in nx.topological_sort(G):
            preds = list(G.predecessors(v))
            depth[v] = 0 if not preds else max(depth[p] for p in preds) + 1
        buckets = {}
        for v, d in depth.items():
            buckets.setdefault(d, []).append(v)
        pos = {}
        for d, vs in buckets.items():
            for k, v in enumerate(sorted(vs, key=str)):
                pos[v] = (d, k - (len(vs) - 1) / 2.0)
        return pos
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G, seed=1)


fig, axes = plt.subplots(3, 3, figsize=(17, 15))
rows = []

for ax, (title, G, S, D, fixed) in zip(axes.ravel(), build_cases()):
    r = solve_defender(G, S, D, XA, XB, tol=1e-9, max_iter=300)
    pos = layout(G, S, D, fixed)
    best = r["best_path"]
    best_edges = set(zip(best[:-1], best[1:])) if best else set()

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b6bcc4", arrows=True,
                           arrowsize=13, width=1.2,
                           connectionstyle="arc3,rad=0.06")
    if best_edges:
        nx.draw_networkx_edges(G, pos, edgelist=list(best_edges), ax=ax,
                               edge_color="#d62839", arrows=True, arrowsize=18,
                               width=2.6, connectionstyle="arc3,rad=0.06")

    colors, sizes, labels = [], [], {}
    for v in G.nodes():
        if v in (S, D):
            colors.append("#2B4C7E"); sizes.append(520); labels[v] = v
        else:
            xv = r["x"].get(v, 0.0)
            colors.append("#E08A3C"); sizes.append(260 + 2100 * (xv / XA))
            labels[v] = f"{xv:.2f}" if xv > 0.005 else "0"

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5,
                            font_color="white", font_weight="bold")

    npaths = r.get("n_paths", 0)
    ax.set_title(f"{title}\n$V^*$ = {r['value']:.6f}   ({npaths} paths, "
                 f"gap {r['gap']:.0e})", fontsize=10)
    ax.axis("off")

    rows.append((title.split("\n")[0], len(r["nodes"]), npaths, r["value"],
                 r["gap"], " -> ".join(best) if best else "-"))

fig.suptitle("Optimal defence across topologies  "
             "(node label & size = defence budget;  red = evader's best path;  "
             "$\\bar{x}_A=\\bar{x}_B=10$)", fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig("out/GALLERY.png", dpi=135)
print("saved out/GALLERY.png\n")

print(f"{'topology':32s} {'contested':>9s} {'paths':>7s} {'V*':>11s} {'gap':>9s}  best path")
print("-" * 118)
for name, nc, npth, v, gap, bp in rows:
    print(f"{name:32s} {nc:9d} {npth:7d} {v:11.6f} {gap:9.1e}  {bp}")
