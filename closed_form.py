"""
closed_form.py -- exact analytical solutions of the game on special topologies.

These are DERIVED in REPORT.md and are used as ground truth for the solver.
All formulas are for contest intensity m = 1.
"""

from __future__ import annotations

import math

__all__ = ["single_node", "chain_value", "parallel_value", "layered_value",
           "unequal_branches_value"]


def single_node(XA, XB):
    """S -> v -> D.   V* = XB / (XA + XB)."""
    return XB / (XA + XB)


def chain_value(n, XA, XB):
    """Single chain with n contested nodes.

    The value function is convex and symmetric in (x_1..x_n), hence minimised
    at x_i = XA/n; by symmetry y_i = XB/n, so

        V* = ( XB / (XA + XB) ) ** n .
    """
    if n == 0:
        return 1.0
    return (XB / (XA + XB)) ** n


def parallel_value(k, n, XA, XB):
    """k node-disjoint identical chains of n contested nodes.

    Budget is split evenly across branches (equalisation of branch values), so
    each branch is a chain defended with XA/k:

        V* = ( XB / (XA/k + XB) ) ** n .
    """
    return (XB / (XA / k + XB)) ** n


def layered_value(L, w, XA, XB):
    """Complete layered DAG, L layers of width w  (w**L paths).

    Symmetry within a layer + the chain reduction give

        V* = ( XB / (XA/w + XB) ) ** L ,

    i.e. only the LAYER WIDTH matters, not the number of paths.
    """
    return (XB / (XA / w + XB)) ** L


def unequal_branches_value(lengths, XA, XB, tol=1e-15):
    """Node-disjoint parallel branches of lengths L_1..L_r.

    Every branch must receive budget (an undefended branch has value 1), and at
    the optimum all branch values are equal to V:

        a_j = XB ( V^(-1/L_j) - 1 ),      sum_j ( V^(-1/L_j) - 1 ) = XA / XB .

    The left side is strictly decreasing in V, so bisect on V in (0,1).
    Returns (V, [a_1..a_r]).
    """
    target = XA / XB

    def f(V):
        return sum(V ** (-1.0 / L) - 1.0 for L in lengths)

    lo, hi = 1e-300, 1.0
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    V = math.sqrt(lo * hi)
    a = [XB * (V ** (-1.0 / L) - 1.0) for L in lengths]
    return V, a
