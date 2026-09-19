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

This repository implements a convex cutting-plane method for that problem, plus
a numerical study of how the optimal defence depends on graph topology. **The
method provides a global optimality certificate when the computed upper and
lower bounds close within the stated tolerance.** Every result below reports
UB, LB, the absolute and relative gap, the tolerance and the resulting
certificate status; runs whose gap has not closed are labelled `certified=no`
and are *not* presented as proved optima.

---

## Quick start

```bash
pip install numpy scipy networkx matplotlib     # nothing else is needed
python run_all.py                               # validation + all experiments (~2 min)
```

Or piecemeal:

```bash
python tests.py                 # 80 validation checks
python experiments.py           # all experiments
python experiments.py E2 E7     # just the worked example and the scaling study
python repro.py                 # print the reproducibility metadata record
```

Minimal use:

```python
import graphs
from core import solve_defender

G, S, D = graphs.grid_dag(4, 4)
r = solve_defender(G, S, D, XA=10.0, XB=10.0)

r["value"]        # 0.40960000   game value, reported as the upper bound V_ub
r["x"]            # best defender allocation found, per node
r["best_path"]    # the evader's best response
r["gap"]          # UB − LB in log space
r["certified"]    # True only if that gap closed within the requested tolerance
r["certificate"]  # UB, LB, abs/rel gap (+ its denominator), tolerance, status,
                  # iterations, cuts, runtime, solver settings, residuals
r["residuals"]    # budget, KKT/stationarity, root and certificate residuals
```

`r["value"]` is an upper bound on `V*`. It is a *certified* optimum exactly when
`r["certified"]` is `True`. All numerical thresholds live in one place,
`tolerances.py`, rather than being hard-coded at each call site.

---

## How it is solved

The whole method rests on one structural fact, proved in `FORMULATION.md` §3:

> In log space, the defender's objective `G(x) = max_P max_y log U_B` is
> **convex** on the simplex for contest intensity `m ≤ 1`.

Because `h(x, y) = Σ[log y_i − log(x_i + y_i)]` is convex in `x` for every fixed
`y`, and a pointwise supremum of convex functions is convex. So the problem is a
convex program: any local minimum is global, and the cutting-plane lower bound
is valid.

**The objective is convex for `0 < m ≤ 1` under the stated model. This does not,
by itself, imply strict convexity or uniqueness of the global minimizer** — a
path objective `g_P` does not depend on the defence variables of nodes off `P`,
so `G` is flat in those directions.

Uniqueness nevertheless **holds**, and is now proved (`FORMULATION.md` §3.1) by
a different route: *Lemma 2* shows that at an optimum every node carrying budget
lies on an active path (empty a node that does not, spread the freed budget over
every other relevant node, and every active path strictly improves), and
*Proposition 4c* then needs strict convexity only along one active path through
one coordinate where two candidate optima differ. This supersedes an earlier,
**wrong** uniqueness proof that this repo retracted; the retraction is kept in
§3.1, and so is a warning that a proof is exactly the thing the test suite
cannot verify.

Three layers:

| Layer | Problem | Method |
|---|---|---|
| Inner | B's split on a fixed path | strictly concave on the nodes with `x_i > 0`; KKT reduces it to one monotone scalar equation, **solved numerically to tolerance** (not symbolically) and then rescaled onto the budget. Closed form at `m = 1`, in cancellation-free shape. Root tolerance, iteration cap, stopping criterion and the final budget / KKT / root residuals are all recorded |
| Middle | B's choice of path | Lagrangian relaxation makes the inner problem separable ⇒ shortest path with non-negative node costs. Yen's k-shortest-paths in Lagrangian-cost order, with a **stopping rule certifying** that no unexamined path can win. The multiplier `λ` comes from an **adaptive** search: the bracket on `log₁₀ λ` is configurable and is expanded automatically whenever the minimiser lands on a boundary |
| Outer | A's allocation | Kelley cutting planes. Danskin gives the subgradient; the LP master over `x ≥ 0` gives a rigorous lower bound, direct evaluation gives the upper bound |

`UB − LB` is a genuine optimality gap, not "the iterates stopped moving" — and
it becomes a *certificate* only once it closes within the requested tolerance.
Both bounds refer to the same optimization problem, and the gap is reported net
of the numerical tolerances configured in `tolerances.py`.

### On complexity

The path-selection subproblem is closely related to maximum-reliability path
problems known to be NP-hard; consequently we use a certified path-generation
procedure rather than claiming a polynomial-time exact oracle. **No reduction
establishing NP-hardness of the exact continuous Tullock-budget model solved
here is supplied, and none is claimed** — the literature result (Nguyen, Song &
Smith, Thm. 3) is for an analogous problem, and is cited as motivation, not as
a proof about this model.

---

## What the numbers say

All figures below were regenerated by `python run_all.py`; the accompanying
`out/RUN_METADATA.json` records the seeds, package versions, repository commit,
hardware, runtime methodology and tolerance configuration behind them.

**Correctness.** Four closed-form families (single chain, parallel chains,
complete layered DAG, unequal branches) are derived analytically and matched by
the solver to a worst absolute error of **5.6e−12**, with the certificate
closing on all 20 benchmark runs (E1). Brute-force grid search never beats the
solver. 80/80 validation checks pass, including explicit residual checks —
evader budget, KKT/stationarity, root equation, certificate gap, and
shortest-path cost consistency — against the tolerances in `tolerances.py`.

Two things those checks do **not** do:

* the randomised chord sampling **numerically validates convexity on sampled
  instances**; sampling cannot prove convexity, and the theorem rests on its
  proof in `FORMULATION.md` §3;
* likewise `[T-21]`/`[T-22]` hunt for counterexamples to Lemma 2 and
  Proposition 4c and find none over 35 instances — corroboration of a proof,
  never verification of one. This file has carried a wrong uniqueness proof
  before, so **the §3 and §3.1 proofs should be read by a second qualified
  person before submission** (see the checklist below).

**The professor's topology** (`S→i1→i2→D`, `S→j1→D`, plus `i1→j1`), `x̄_A = x̄_B = 10`:

```
x*[i1] = x*[i2] = 1.513878        x*[j1] = 6.972244        V* = 0.589197293
UB = LB = −0.5289941886     abs gap 1.6e−13     tol 1e−12     CERTIFIED
```

Cross-checked by hand: equalising the two active paths gives
`a² + 15a − 25 = 0`, so `a = (−15 + √325)/2 = 1.5139`, which the solver
reproduces.

**Path count is not what makes these instances hard — on the families tested.**
This is observed behaviour on the tested graph family and instance sizes, not a
complexity theorem. On the largest tested layered instance, with
**16 777 216** distinct S–D paths, the oracle evaluated **2** of them and
recovered the closed-form value to machine precision on the hardware recorded
in `out/RUN_METADATA.json`. Under the plain Kelley master the outer certificate
did *not* close there (abs gap ≈ 2.1e−3 at the 200-iteration cap), and only
5 of the 13 layered instances certified; with the stabilised boxstep master
that is now the default, **all 13 certify**, that instance included
(gap 7.2e−10, 62 iterations). E7 reports both masters side by side.

Structurally, `parallel-3x3` (3 paths) and `layered-3x3` (27 paths) have
*identical* value `0.421875`, and this is not a coincidence of the tested
instances: Corollary 1 in `FORMULATION.md` §6 proves `w` parallel chains and the
`w^L`-path layered DAG have the same value at every budget pair, so path count
provably cannot determine `V*`.

**Heuristics are not close.** The metric is

```
excess success probability (%) = 100 * (V_heuristic / V* - 1)
```

a ratio minus one — not an absolute difference and not a difference of
percentage points, so "+211 %" means the evader succeeds 3.11× as often. `V*`
is taken as the upper bound `V_ub`, so the figures are conservative. Over 20
random DAGs (E5; all 20 certified at tol 1e−9):

| rule | mean | worst |
|---|---|---|
| uniform | +211.0 % | +329.5 % |
| min vertex cut | +220.9 % | +578.7 % |
| path count | +125.0 % | +166.9 % |
| betweenness | +111.0 % | +339.6 % |
| greedy marginal | +10.3 % | +42.5 % |

Uniform allocation *is* optimal on the symmetric families proved in
`FORMULATION.md` §6 (chain, identical parallel chains, complete layered) — a
theorem-level counterpart to Ramirez-Marquez–Rocco–Levitin's "homogeneous
vulnerability" finding — and fails badly the moment the topology is irregular.
It is also observed optimal on the regular grid, but that case is *not* covered
by any of our proofs. `betweenness` scores `V = 1`, total defender failure, on
the lecture topology, because it starves a whole path.

**Against the nearest papers' own methods, not just generic heuristics** (E9).
The related-work section positions this project between two papers, so it is
compared against adaptations of what those papers actually *do* — an
evolutionary search (Ramirez-Marquez–Rocco–Levitin) and constraint generation
with a log-linearised shortest-path separation (Nguyen–Song–Smith). Both are
scored by the *same* exact best-response oracle, so the comparison isolates the
method. Over 10 random DAGs:

| method | mean excess | worst | certificate? |
|---|---|---|---|
| this work (certified cutting planes) | 0.00 % | 0.00 % | **yes** |
| NSS-style constraint generation (adapted) | +0.61 % | +3.03 % | no |
| RRL-style evolutionary search (adapted) | +3.64 % | +6.31 % | no |
| greedy marginal (generic heuristic) | +12.52 % | +42.45 % | no |
| uniform (generic heuristic) | +204.60 % | +265.71 % | no |

The honest reading: the two adapted methods land *close* on value — far closer
than generic heuristics — and **the difference that matters is that neither can
certify anything.** The same shows up in the path oracle (E9b): on identical
allocations the Lagrangian separation evaluates **2** paths and stops with a
proof, while the linearised separation finds the *same* best response but,
having no stopping rule, must exhaust all 13–57 paths. The contribution is the
certificate and the stopping rule, not a better answer. `baselines.py` states
exactly what was and was not carried over from each paper; neither is a rerun of
published code.

**`m > 1` is now bounded, not just flagged** (E10). Convexity provably fails for
`m > 1`, so the cutting-plane lower bound is invalid there and the regime used
to be reported with no bound at all. Substituting `u = x^m` makes the objective
convex for *every* `m`, and the budget set's convex hull is exactly the simplex
`{u ≥ 0 : Σu_i ≤ x̄_A^m}`, so minimising over the hull is a valid **lower**
bound. That gives a real two-sided bracket — diamond at `m = 2`:
`V* ∈ [0.667, 0.800]`. It is a **weaker** certificate: the relaxation gap does
not shrink as the solver converges, it widens with `m`, and it never certifies
the allocation is optimal.

**Structure.** Serial depth compounds the defender's advantage geometrically;
parallel redundancy forces the budget to spread. The optimal support must be a
vertex cut — but defending nothing *but* a minimum cut is a bad rule
(+700 % excess success probability on a 4-chain). And **path count provably does
not determine the value**: `w` parallel chains (`w` paths) and the complete
layered DAG with `L` layers of width `w` (`w^L` paths) have *identical* value at
every budget pair (`FORMULATION.md` §6, Corollary 1) — so no function of the
path count alone can predict `V*`.

---

## Files

| file | contents |
|---|---|
| `FORMULATION.md` | every mathematical claim with its proof, the closed forms, and the relation to the cited papers |
| `core.py` | inner solver, Lagrangian path oracle, cutting-plane defender solver, certificate assembly, residual audits, baselines, brute force |
| `tolerances.py` | every numerical threshold in one place: root, budget, KKT, certificate-gap and convexity tolerances, and the numerical floor |
| `baselines.py` | head-to-head comparison methods: an RRL-style evolutionary search and an NSS-style linearised constraint generation, both **adaptations** (what was and was not carried over is documented in the module) |
| `relaxation.py` | the `u = x^m` convex relaxation giving a **valid** lower bound for `m > 1`, where the ordinary bound is invalid |
| `repro.py` | reproducibility metadata: seeds, package versions, repo commit, hardware, runtime methodology |
| `graphs.py` | test topologies (chain, parallel, unequal branches, layered, grid, grid+bypass, random DAG, lecture example) |
| `closed_form.py` | the four analytical benchmark families (derived in `FORMULATION.md` §6) |
| `tests.py` | 80 validation checks (analytical benchmarks, exhaustive enumeration, brute-force search, randomised sampling, residuals, certificate reporting, boundary cases, the Lemma 2 / Prop. 4c / Corollary 1 counterexample hunts, arc contests, relaxation validity, head-to-head baselines, boxstep) |
| `experiments.py` | the eight numerical experiments E1–E8 |
| `run_all.py` | reproduce everything |
| `out/` | generated tables (`.csv`), raw results (`.json`), figures (`.png`), logs, and `RUN_METADATA.json` |

## Experiments

| id | question |
|---|---|
| E1 | does the solver match the closed forms, and does the certificate close? |
| E2 | the lecture topology, worked node by node |
| E3 | how does the optimum vary across topologies, and where do baselines fail? |
| E4 | how does `V*` respond to the budget ratio `x̄_A/x̄_B`? |
| E5 | baseline comparison over 20 random DAGs |
| E6 | does the certified gap actually close? |
| E7 | observed scaling on layered graphs up to 1.7e7 paths, and random DAGs to 30 interior nodes |
| E8 | effect of contest intensity `m` (`m > 1` reported as exploratory) |
| E9 | head-to-head against the two nearest papers' own methods (adapted), and against their path-separation strategy |
| E10 | a valid two-sided bracket for the `m > 1` regime via convex relaxation |

Every experiment table carries the certificate columns `log_ub`, `log_lb`,
`abs_gap`, `rel_gap`, `tol`, `certified`, `cert_status`, `iters`, `cuts`, `sec`.

## Before submitting

Two items on the pre-submission list are **not** things this repository can
close by itself.

1. **Have a second qualified reader check the proofs** — specifically
   Proposition 4 (convexity, `FORMULATION.md` §3) and now also Lemma 2 and
   Proposition 4c (uniqueness, §3.1). A test suite can find a counterexample —
   as it correctly does for `m > 1` — but it cannot confirm a universally
   quantified proof, and §3.1 has carried a wrong proof before. `[T-21]`,
   `[T-22]` and `[T-6]` hunt for counterexamples and find none; that is
   corroboration, not verification. Treat this as a hard prerequisite.

2. **Pick a venue and reformat.** `paper/paper.tex` is currently a
   single-column `article` draft (~20 pp) and is deliberately left that way
   until the target is chosen:

   | route | template | length | what to do |
   |---|---|---|---|
   | GameSec | LNCS, single column | 15–20 pp | keep the full numerical study; the E9 head-to-head is expected at this venue and now exists |
   | CDC / ACC | IEEE, two column | 6–8 pp | move the §6 closed-form derivations to an appendix; foreground the convexity theorem and the certified path oracle as the two headline contributions |

   The head-to-head comparison that used to block this decision (E9) is done, so
   the choice is now purely about format and length.

## Known limitations

1. **`m > 1` is exploratory, never certified.** Convexity provably fails
   (explicit counterexample on the diamond at `m = 2`), so the cutting-plane
   lower bound is not valid there. The solver still runs, and E8 reports those
   rows as `EXPLORATORY / NON-CERTIFIED` regardless of the gap they happen to
   reach. Whether *any* equilibrium (pure or mixed) can be characterized there
   is open — flagged, not solved, in `FORMULATION.md` §3.
2. **Stackelberg, not simultaneous.** A commits first and B observes `x`. A
   simultaneous version would need mixed strategies over paths — the
   Nguyen–Song–Smith setting.
3. **The certificate now closes on every instance we test** — this limitation
   used to say it did not. Plain Kelley left ~2e−3 of log-gap on the widest
   layered instances (48–50 contested nodes) at the 200-iteration cap. The fix
   the repo itself named, a stabilised master, is implemented:
   `solve_defender(master="boxstep")` picks each step from a trust region around
   the incumbent while still taking the **lower bound from the unrestricted
   master**, so the certificate means exactly what it did before. It certifies
   all 13 layered instances in roughly a quarter of the iterations, and is now
   the default (`master="kelley"` remains available, and E7 reports both).
   Still open: **no proven convergence rate exists for either master here** —
   only the measured behaviour in E6/E7, and nothing rules out a family on which
   boxstep also stalls.
4. **Independence across nodes** is assumed (the payoff is a product).
5. **Arc contests are implemented** (was: "node contests only").
   `graphs.arc_contest_graph` performs the line-graph relabelling — every arc
   becomes a node, `S–D` paths correspond bijectively — so every result applies
   after the transform; `[T-24]` checks the bijection and the resulting closed
   form. Simultaneous node *and* arc contests with separate budgets are not
   covered.
6. **The path oracle is certified but not polynomial.** The path-selection
   subproblem is closely related to maximum-reliability path problems known to
   be NP-hard, which is why we use certified path generation rather than claim
   a polynomial-time exact oracle. That literature result concerns an
   *analogous* problem; it is not a proof that the exact continuous model here
   is NP-hard, and no such reduction is supplied.
7. **Baselines compared are simple heuristics (uniform, mincut, betweenness,
   greedy), not re-implementations of Nguyen–Song–Smith's or
   Ramirez-Marquez–Rocco–Levitin's actual solution methods on the same
   instances.** A reviewer positioning this work "between" those two papers
   will expect that head-to-head comparison, not just a comparison against
   generic centrality/cut heuristics.
8. **No novelty claim is made.** This repository implements and studies the
   model above. A literature check run alongside the project found no exact
   match, and found a close relative (Bloch, Chatterjee & Dutta 2023, *Attack
   and interception in networks*) that is compared against explicitly in
   `FORMULATION.md` §8. To the best of that search the combination here is not
   duplicated, but a keyword search is not a systematic literature review, so
   no "first", "only" or "no prior work" claim is asserted anywhere in this
   repository. Confirm with the supervising professor before any submission
   claim of novelty.
9. **Uniqueness of `x*` is now proved — but the proof wants a second reader.**
   `FORMULATION.md` §3.1 proves it for `0 < m ≤ 1` when `V* < 1`, via Lemma 2
   (active cover) rather than via strict convexity of `G`, which genuinely
   fails. An earlier version of this repo claimed uniqueness by that wrong route
   and retracted it; the retraction is kept. Because a test suite can refute a
   proof but never confirm one, and because this exact claim has been wrong here
   before, treat it as provisional until independently checked.
10. **`m > 1` is bounded but still not characterised.** E10 gives a valid
   bracket, but the relaxation gap widens with `m` (a factor of 124 on a 3-node
   chain at `m = 3`) and never certifies the allocation. Whether a
   mixed-strategy equilibrium takes over there is open.
11. **Correlated node outcomes are still assumed away.** The payoff is a strict
   product. A copula- or Markov-based model would break the separability that
   the shortest-path reduction depends on, so it is a modelling project, not a
   relabelling.
