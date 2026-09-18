"""
tolerances.py -- one place for every numerical threshold used in this project.

Rationale
---------
Scattered hard-coded tolerances make test failures and certificate claims hard
to interpret: a check that passes at `1e-9` in one file and fails at `1e-12` in
another says nothing about the method, only about which literal was typed where.
Every solver, test and experiment in this repository therefore imports its
thresholds from here.

A note on what these numbers are and are not
--------------------------------------------
None of these constants is part of the *mathematical* model.  The model is the
continuous program

    V* = min_x max_P max_y  prod_{i in P} y_i^m / (x_i^m + y_i^m),

in which `x_i = 0` is a perfectly legal allocation (convention C1 of
FORMULATION.md S1.1: `p_i = 1` there).  `X_FLOOR_REL` below is a purely
*numerical* safeguard used to keep intermediate iterates away from the
`1/(x_i + y_i)` singularity of the subgradient; it is not a constraint of the
model, and the solver deliberately evaluates exactly-sparse points (`x_i = 0`)
so that reported optima are the model's boundary solutions rather than
artefacts of the floor.  See `core.sparsify_alloc`.
"""

from __future__ import annotations

__all__ = [
    "ROOT_TOL", "ROOT_MAX_ITER", "BUDGET_TOL", "KKT_TOL",
    "LAMBDA_LOG_LO", "LAMBDA_LOG_HI", "LAMBDA_EXPAND_MAX",
    "LAMBDA_GOLDEN_ITERS", "LAMBDA_GRID",
    "GAP_TOL", "CERT_SLACK", "LP_FEAS_TOL",
    "X_FLOOR_REL", "X_FLOOR_ABS", "Y_FLOOR", "LOG_FLOOR",
    "CONVEXITY_TOL", "TEST_TOL", "as_dict",
]

# --- inner problem: root solving for the evader allocation -------------------
#: relative stopping tolerance for the scalar root  sum_i y_i(t) = XB
ROOT_TOL = 1e-15
#: hard cap on safeguarded-Newton iterations for that root
ROOT_MAX_ITER = 60
#: acceptable relative residual of the budget constraint  |sum y_i - XB| / XB
BUDGET_TOL = 1e-12
#: acceptable relative spread of the KKT multipliers across active nodes
KKT_TOL = 1e-8

# --- Lagrangian dual: search over the multiplier lambda ----------------------
#: initial bracket on log10(lambda).  lambda > 0 is unbounded mathematically,
#: so this is only a STARTING bracket -- `core.PathOracle._best_lambda`
#: expands it adaptively when the minimiser sits on a boundary.
LAMBDA_LOG_LO = -12.0
LAMBDA_LOG_HI = 12.0
#: maximum number of doublings applied when expanding that bracket
LAMBDA_EXPAND_MAX = 40
#: golden-section refinement steps once the bracket is interior
LAMBDA_GOLDEN_ITERS = 40
#: coarse grid points used to locate the minimiser inside the bracket
LAMBDA_GRID = 25

# --- outer problem: cutting planes and the optimality certificate ------------
#: default requested certificate tolerance, in LOG space (UB - LB <= GAP_TOL)
GAP_TOL = 1e-7
#: slack allowed when declaring the path-generation stopping rule satisfied
CERT_SLACK = 1e-9
#: primal/dual feasibility tolerances handed to the master LP.  Cut slopes
#: -1/(x_i + y_i) reach ~1e4, so the HiGHS defaults would leave ~1e-6 of slack
#: in the LOWER bound and hence in the reported gap.
LP_FEAS_TOL = 1e-10

# --- numerical floors (NOT model constraints -- see the module docstring) ----
#: iterate floor as a fraction of the per-node average budget XA/n
X_FLOOR_REL = 1e-9
#: absolute lower limit on that floor
X_FLOOR_ABS = 1e-12
#: floor applied to y before taking logs, to avoid log(0)
Y_FLOOR = 1e-300
#: floor applied to probabilities before taking logs
LOG_FLOOR = 1e-300

# --- validation ---------------------------------------------------------------
#: tolerance for the randomised convexity checks (chord violations)
CONVEXITY_TOL = 1e-9
#: generic tolerance for equality assertions in the validation suite
TEST_TOL = 1e-9


def as_dict():
    """All tolerances as a plain dict, for embedding in result files."""
    return {k: globals()[k] for k in __all__ if k != "as_dict"}
