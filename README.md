# A path-contest network security game

Defender **A** spreads a budget over the nodes of a directed graph. Evader **B**
picks a source→terminus path and spreads a budget over the nodes of *that*
path. At each node the outcome is a ratio-form (Tullock) contest, and B must
survive every node on the path:

```
p_i = y_i / (x_i + y_i)            U_B(P, x, y) = ∏_{i∈P} p_i

V* = min_x max_P max_y ∏_{i∈P} y_i/(x_i + y_i)
     s.t.  Σ_i x_i = x̄_A,  Σ_{i∈P} y_i = x̄_B,  x, y ≥ 0
```

This repository solves that problem **to certified global optimality**, plus a
numerical study of how the optimal defence depends on graph topology.

---

## Quick start

```bash
pip install numpy scipy networkx matplotlib     # nothing else is needed
python run_all.py                               # validation + all experiments (~2 min)
```

Or piecemeal:

```bash
python tests.py                 # 36 validation checks
python experiments.py           # all experiments
python experiments.py E2 E7     # just the worked example and the scaling study
```

Minimal use:

```python
import graphs
from core import solve_defender

G, S, D = graphs.grid_dag(4, 4)
r = solve_defender(G, S, D, XA=10.0, XB=10.0)

r["value"]        # 0.40960000   certified game value V*
r["x"]            # optimal defender allocation, per node
r["best_path"]    # the evader's best response
r["gap"]          # UB - LB in log space: the optimality certificate
```

---

## How it is solved

The whole method rests on one structural fact, proved in `FORMULATION.md` §3:

> In log space, the defender's objective `G(x) = max_P max_y log U_B` is
> **convex** on the simplex for contest intensity `m ≤ 1`.

Because `h(x, y) = Σ[log y_i − log(x_i + y_i)]` is convex in `x` for every fixed
`y`, and a pointwise supremum of convex functions is convex. So the problem is a
convex program with a global optimum, not just a stationary point — and a
cutting-plane method solves it with a real certificate.

Three layers:

| Layer | Problem | Method |
|---|---|---|
| Inner | B's split on a fixed path | strictly concave; KKT reduces it to one scalar equation, solved by monotone bisection. Closed form at `m = 1`, in cancellation-free shape |
| Middle | B's choice of path | Lagrangian relaxation makes the inner problem separable ⇒ shortest path. Yen's k-shortest-paths in Lagrangian-cost order, with a **stopping rule that proves** no unexamined path can win |
| Outer | A's allocation | Kelley cutting planes. Danskin gives the subgradient; the LP master over `x ≥ 0` gives a rigorous lower bound, exact evaluation gives the upper bound |

`UB − LB` is a genuine optimality gap, not "the iterates stopped moving".

---

## What the numbers say

**Correctness.** Four closed-form families (single chain, parallel chains,
complete layered DAG, unequal branches) are derived analytically and matched by
the solver to a worst absolute error of **8.9e−10**. Brute-force grid search
never beats it. 36/36 validation checks pass. These include a numerical check
that the solver reaches the same allocation `x*` from very different starting
points — which is *evidence* consistent with a unique optimal allocation, not
a proof of one. Uniqueness of `x*` is an open question; see `FORMULATION.md` §3.

**The professor's topology** (`S→i1→i2→D`, `S→j1→D`, plus `i1→j1`), `x̄_A = x̄_B = 10`:

```
x*[i1] = x*[i2] = 1.513885        x*[j1] = 6.972244        V* = 0.589197293
```

Verified by hand: equalising the two active paths gives `a² + 15a − 25 = 0`, so
`a = (−15 + √325)/2 = 1.5139`. Exact match.

**Path count is the wrong complexity measure.** On a layered graph with
**16 777 216** distinct S–D paths, the oracle evaluates **2** of them and
recovers the closed-form value to machine precision, in a few seconds. Note the
outer cutting-plane certificate does *not* close on that instance (gap ≈ 2e−3
at the 200-iteration cap): the value is accurate, but not proved optimal.
And `parallel-3x3` (3 paths) and `layered-3x3` (27 paths) have *identical*
value `0.421875` — the value depends on layer width and depth, not on how many
paths exist.

**Heuristics are not close.** Mean excess evader success over 20 random DAGs,
relative to the certified optimum:

| rule | mean | worst |
|---|---|---|
| uniform | +211 % | +330 % |
| min vertex cut | +221 % | +579 % |
| path count | +125 % | +167 % |
| betweenness | +111 % | +340 % |
| greedy marginal | +10 % | +43 % |

Uniform allocation *is* exactly optimal on the symmetric families we prove
(chain, identical parallel chains, complete layered) — a theorem-level
counterpart to Ramirez-Marquez–Rocco–Levitin's "homogeneous vulnerability"
finding — and fails badly the moment the topology is irregular. It is also
observed optimal on the regular grid, but that case is *not* covered by any
of our proofs. `betweenness` scores `V = 1`, total defender failure,
on the lecture topology, because it starves a whole path.

**Structure.** Serial depth compounds the defender's advantage geometrically;
parallel redundancy forces the budget to spread. The optimal support must be a
vertex cut — but *only* defending a minimum cut is a bad rule (700 % worse than
optimal on a 4-chain).

---

## Files

| file | contents |
|---|---|
| `FORMULATION.md` | every mathematical claim with its proof, the closed forms, and the relation to the two supplied papers |
| `core.py` | inner solver, Lagrangian path oracle, cutting-plane defender solver, baselines, brute force |
| `graphs.py` | test topologies (chain, parallel, unequal branches, layered, grid, grid+bypass, random DAG, lecture example) |
| `closed_form.py` | the four analytical benchmark families |
| `tests.py` | 36 validation checks (analytical benchmarks, exhaustive enumeration, brute-force search, randomised sampling) |
| `experiments.py` | the eight numerical experiments E1–E8 |
| `run_all.py` | reproduce everything |
| `out/` | generated tables (`.csv`), raw results (`.json`), figures (`.png`), logs |

## Experiments

| id | question |
|---|---|
| E1 | does the solver match the closed forms? |
| E2 | the lecture topology, worked node by node |
| E3 | how does the optimum vary across topologies, and where do baselines fail? |
| E4 | how does `V*` respond to the budget ratio `x̄_A/x̄_B`? |
| E5 | baseline comparison over 20 random DAGs |
| E6 | does the certified gap actually close? |
| E7 | scaling: up to 1.7e7 paths, and random DAGs to 30 interior nodes |
| E8 | effect of contest intensity `m` |

## Known limitations

1. **`m > 1` is not certified.** Convexity provably fails (explicit
   counterexample on the diamond at `m = 2`); the solver still runs but results
   are labelled `NOT PROVEN`. Whether *any* equilibrium (pure or mixed) can be
   characterized there is open — flagged, not solved, in `FORMULATION.md` §3.
2. **Stackelberg, not simultaneous.** A commits first and B observes `x`. A
   simultaneous version would need mixed strategies over paths — the
   Nguyen–Song–Smith setting.
3. **The certificate does not always close within 200 iterations** on the widest
   layered instances (48–50 contested nodes): the value is exact to machine
   precision but Kelley's lower bound still has ~2e−3 of log-gap. A
   proximal-bundle or level-set master is the obvious fix. **No proven
   convergence rate exists for the cutting-plane method here** — only the
   empirical behaviour in E6/E7. Kelley's method has no rate guarantee in
   general without extra curvature assumptions; establishing one (e.g. from
   the strict convexity in `FORMULATION.md` Prop. 4b) is future work.
4. **Independence across nodes** is assumed (the payoff is a product).
5. **Node contests only**; arc contests would need each arc split into a node.
6. **The path oracle is certified but not polynomial** — the underlying
   max-reliability-path problem is NP-hard.
7. **Baselines compared are simple heuristics (uniform, mincut, betweenness,
   greedy), not re-implementations of Nguyen–Song–Smith's or
   Ramirez-Marquez–Rocco–Levitin's actual solution methods on the same
   instances.** A reviewer positioning this work "between" those two papers
   will expect that head-to-head comparison, not just a comparison against
   generic centrality/cut heuristics.
8. **Novelty has not been independently verified.** A literature check found
   no exact match, but also found a close relative (Bloch, Chatterjee & Dutta
   2023, *Attack and interception in networks*) that a reviewer will expect
   this work to be positioned against explicitly — see `FORMULATION.md` §8
   for the comparison. This checklist item should be confirmed with the
   supervising professor before any submission claim of novelty.
9. **Uniqueness of `x*` is NOT proved — open question.** An earlier version of
   this repo claimed a proof; it was wrong. The argument tried to deduce strict
   convexity of `G` in the full vector `x`, but `g_P` depends only on the nodes
   *on* `P`, so two allocations differing only off the maximising path give
   equality, not strict inequality. What survives is strict convexity restricted
   to the coordinates of the maximising path (`FORMULATION.md` §3). `[T-14]`
   shows the solver reaches the same `x*` from different starting points, which
   is evidence, not proof.
