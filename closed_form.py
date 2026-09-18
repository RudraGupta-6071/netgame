"""
closed_form.py -- closed-form solutions of the game on special topologies.

These are DERIVED in FORMULATION.md S6 (there is no REPORT.md in this
repository; `make_report.py` generates out/REPORT.html, which is a rendering of
the results, not the source of the derivations).  They are independent of the
solver and are used as analytical ground truth for it in `tests.py` [T-7] and
in experiment E1.

All formulas are for contest intensity m = 1.  Each carries the assumptions it
needs in its own docstring; they are closed forms of the model, evaluated here
in floating point.
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

    Assumptions: the path has n relevant (contested) nodes and the defence is
    symmetric on them.  The value function is convex and symmetric in
    (x_1..x_n), so averaging over the symmetric group cannot hurt the defender
    (Jensen) and x_i = XA/n attains the minimum; by symmetry the evader's
    optimal split is then y_i = XB/n, giving

        V* = ( XB / (XA + XB) ) ** n .

    Convexity plus symmetry gives that a symmetric allocation is optimal; it
    does not assert that it is the only optimal one.
    """
    if n == 0:
        return 1.0
    return (XB / (XA + XB)) ** n


def parallel_value(k, n, XA, XB):
    """k node-disjoint identical chains of n contested nodes.

    Assumptions: the branches are node-disjoint, identical, and every branch
    must be defended (an undefended branch has value 1).  Branch values are
    equalised at the optimum, so the budget is split evenly across branches and
    each branch is a chain defended with XA/k:

        V* = ( XB / (XA/k + XB) ) ** n .
    """
    return (XB / (XA / k + XB)) ** n


def layered_value(L, w, XA, XB):
    """Complete layered DAG, L layers of width w  (w**L paths).

    Assumptions: consecutive layers are fully connected, so every S-D path
    meets exactly one node per layer, and the objective is invariant under
    permuting nodes within a layer.  Symmetry within a layer (plus convexity)
    makes a layer-uniform allocation optimal, and the chain reduction then
    gives

        V* = ( XB / (XA/w + XB) ) ** L ,

    i.e. only the LAYER WIDTH matters, not the number of paths.
    """
    return (XB / (XA / w + XB)) ** L


def unequal_branches_value(lengths, XA, XB, tol=1e-15):
    """Node-disjoint parallel branches of lengths L_1..L_r.

    Every branch must receive budget (an undefended branch has value 1), and at
    the optimum all branch values are equal to V:

        a_j = XB ( V^(-1/L_j) - 1 ),      sum_j ( V^(-1/L_j) - 1 ) = XA / XB .

    The left side is strictly decreasing in V, so the root is unique and is
    located here by bisection -- i.e. V is computed NUMERICALLY to the stated
    tolerance, not in closed form (unlike the three families above, which are
    explicit expressions).  Returns (V, [a_1..a_r]).
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
