"""
graphs.py -- test topologies for the network security game.

Every builder returns (G, S, D) with G a networkx.DiGraph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = ["chain", "parallel_chains", "unequal_branches", "layered", "grid_dag",
           "grid_bypass", "random_dag", "lecture_example", "diamond", "CATALOG"]


def chain(n):
    """S -> 1 -> 2 -> ... -> n -> D.  One path, n contested nodes."""
    G = nx.DiGraph()
    nodes = ["S"] + [f"v{i}" for i in range(1, n + 1)] + ["D"]
    nx.add_path(G, nodes)
    return G, "S", "D"


def parallel_chains(k, n):
    """k node-disjoint chains of length n in parallel between S and D."""
    G = nx.DiGraph()
    for j in range(k):
        nodes = ["S"] + [f"b{j}_{i}" for i in range(1, n + 1)] + ["D"]
        nx.add_path(G, nodes)
    return G, "S", "D"


def unequal_branches(lengths):
    """Node-disjoint parallel branches with the given numbers of contested nodes."""
    G = nx.DiGraph()
    for j, L in enumerate(lengths):
        nodes = ["S"] + [f"b{j}_{i}" for i in range(1, L + 1)] + ["D"]
        nx.add_path(G, nodes)
    return G, "S", "D"


def layered(L, w):
    """Complete layered DAG: L layers of width w, consecutive layers fully
    connected.  Contains w**L distinct S-D paths but only L*w nodes."""
    G = nx.DiGraph()
    layers = [[f"L{l}_{j}" for j in range(w)] for l in range(L)]
    for v in layers[0]:
        G.add_edge("S", v)
    for l in range(L - 1):
        for u in layers[l]:
            for v in layers[l + 1]:
                G.add_edge(u, v)
    for v in layers[-1]:
        G.add_edge(v, "D")
    return G, "S", "D"


def grid_dag(r, c, wide=True):
    """r x c grid, edges directed right and down.

    wide=True  : S feeds the whole left column and the whole right column
                 feeds D (no articulation node, many S-D paths).
    wide=False : S -> top-left, bottom-right -> D.
    """
    G = nx.DiGraph()

    def name(i, j):
        return f"g{i}_{j}"

    for i in range(r):
        for j in range(c):
            if j + 1 < c:
                G.add_edge(name(i, j), name(i, j + 1))
            if i + 1 < r:
                G.add_edge(name(i, j), name(i + 1, j))
    if wide:
        for i in range(r):
            G.add_edge("S", name(i, 0))
            G.add_edge(name(i, c - 1), "D")
    else:
        G.add_edge("S", name(0, 0))
        G.add_edge(name(r - 1, c - 1), "D")
    return G, "S", "D"


def random_dag(n, p=0.25, seed=0, ensure_len=2):
    """Random DAG on n interior nodes (topological order = index order).

    S connects to a random non-empty subset of the first few nodes, the last
    few connect to D, and every node is forced onto at least one S-D path.
    `ensure_len` guarantees no path shorter than `ensure_len` contested nodes,
    so the game value is non-trivial.
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    V = [f"n{i}" for i in range(n)]
    G.add_nodes_from(V)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(V[i], V[j])
    # guarantee connectivity through the topological order
    for i in range(n - 1):
        if not any(True for _ in G.successors(V[i])):
            G.add_edge(V[i], V[i + 1])
    for i in range(1, n):
        if G.in_degree(V[i]) == 0:
            G.add_edge(V[i - 1], V[i])
    heads = [v for v in V if G.in_degree(v) == 0] or [V[0]]
    tails = [v for v in V if G.out_degree(v) == 0] or [V[-1]]
    for v in heads:
        G.add_edge("S", v)
    for v in tails:
        G.add_edge(v, "D")
    # no S->D shortcut and no 1-node path if ensure_len >= 2
    if ensure_len >= 2:
        for v in list(G.successors("S")):
            if G.has_edge(v, "D") and G.out_degree(v) == 1:
                w = f"pad_{v}"
                G.remove_edge(v, "D")
                G.add_edge(v, w)
                G.add_edge(w, "D")
    return G, "S", "D"


def grid_bypass(r=4, c=4, bypass=2):
    """Wide grid plus a SHORT unguarded bypass S -> h1 -> ... -> D.

    Breaks the symmetry of the plain grid: the bypass is much shorter than any
    grid route, so the defender must over-invest on it.
    """
    G, S, D = grid_dag(r, c, wide=True)
    nodes = ["S"] + [f"h{i}" for i in range(1, bypass + 1)] + ["D"]
    nx.add_path(G, nodes)
    return G, S, D


def diamond():
    """S->A->D, S->B->D : two symmetric single-node branches."""
    G = nx.DiGraph()
    nx.add_path(G, ["S", "A", "D"])
    nx.add_path(G, ["S", "B", "D"])
    return G, "S", "D"


def lecture_example():
    """The topology sketched in the problem statement:
       a two-node branch  S -> i1 -> i2 -> D
       and a one-node branch S -> j1 -> D,
       plus a cross link i1 -> j1 that makes the branches non-disjoint.
    """
    G = nx.DiGraph()
    nx.add_path(G, ["S", "i1", "i2", "D"])
    nx.add_path(G, ["S", "j1", "D"])
    G.add_edge("i1", "j1")
    return G, "S", "D"


CATALOG = {
    "chain-4": lambda: chain(4),
    "diamond": diamond,
    "lecture": lecture_example,
    "parallel-3x3": lambda: parallel_chains(3, 3),
    "unequal-1-2-4": lambda: unequal_branches([1, 2, 4]),
    "layered-3x3": lambda: layered(3, 3),
    "grid-4x4": lambda: grid_dag(4, 4),
    "grid-bypass": lambda: grid_bypass(4, 4, 2),
    "random-12": lambda: random_dag(12, 0.25, seed=7),
}
