# The path-contest network security game: formulation, theory, algorithm

All statements below are either **proved** here or explicitly labelled as a
numerical observation. Every proposition has a matching automated check in
`tests.py`; the tag `[T-k]` gives the test number.

Two standing conventions about status. (i) Automated checks *validate* the
implementation against a claim; randomised sampling in particular can only show
that no violation was found on the instances sampled, so it never substitutes
for a proof. Where a claim is proved, the proof here is the basis for it. (ii)
Results derived in this document are labelled as ours; literature is cited for
what it actually establishes, and never as a proof of a statement about the
exact model defined in §1.

---

## 1. The model

Let `G = (V, E)` be a directed graph with a source `S` and a terminus `D`.
Write `P` for a simple `S–D` path and `P°` for its **contested** nodes (by
default `P \ {S, D}`; set `contest_endpoints=True` to contest them too).

**Defender A** commits to `x ≥ 0` on the contested nodes with

    Σ_{i ∈ V°} x_i = x̄_A .

**Evader B** observes `x`, chooses a path `P`, then splits `y ≥ 0` over `P°`:

    Σ_{i ∈ P°} y_i = x̄_B .

At each contested node the outcome is a **ratio-form (Tullock) contest**, with
contest intensity `m > 0`:

    p_i = y_i^m / (x_i^m + y_i^m) .

`m = 1` is the case in the problem statement. `m` is the parameter
Ramirez-Marquez–Rocco–Levitin call *contest intensity*: `m < 1` gives a
disproportionate advantage to the smaller investment, `m > 1` gives economies
of scale to the larger one.

B must survive **every** contested node of the chosen path (a series system),
so

    U_B(P, x, y) = Π_{i ∈ P°} p_i .

The Stackelberg / minimax problem is

    V* = min_x max_P max_y U_B(P, x, y).                                  (★)

Only `V*` is well defined as a *value*; A moves first and B best-responds, so
this is a Stackelberg game, not a simultaneous one. (See §9 for why that
distinction matters and when the two coincide.)

### 1.1 Two conventions, and why they are forced

**(C1) `x_i = 0` means free passage.** If `x_i = 0` then `p_i = y_i^m/y_i^m = 1`
for every `y_i > 0`, so B gets through node `i` for free and optimally spends
`y_i = 0` there. Taking `p_i = 1` at `x_i = 0` is the *continuous extension* of
the CSF along B's best response, and it is what keeps the value function
continuous at the boundary of the simplex. This is not a modelling choice made
for convenience — it is the only extension under which `G` below is continuous.

**(C2) Work in logs.** Everything is computed as

    g(x) = log U_B = Σ_{i ∈ P°} [ m log y_i − log(x_i^m + y_i^m) ] .

The product form is numerically hopeless on long paths (values reach `1e-300`);
the log form is where the problem is also *convex*, which is the entire reason
the game is solvable to global optimality.

### 1.2 A structural necessary condition

If the set `{i : x_i > 0}` is **not an `S–D` vertex cut**, then some path is
entirely undefended and `V* = 1`. So any sensible defence must at minimum
cover every `S–D` path. This is the link to the min-cut/max-flow view of
Kovenock–Roberson, and it is exactly why the `betweenness` baseline scores
`V = 1` on the lecture topology: it concentrates on one node and abandons a
whole path. `[T-12]`

---

## 2. The inner problem: B's split on a fixed path

Fix `P` and `x`. Let `φ_i(y) = m log y − log(x_i^m + y^m)` for `x_i > 0`.

**Proposition 1 (strict concavity, for `x_i > 0`).**  Fix a node with
`x_i > 0`. Then

    φ_i'(y) = m/y − m y^{m−1}/(x_i^m + y^m)
            = m x_i^m / [ y (x_i^m + y^m) ]  >  0 ,

and the denominator `y·x_i^m + y^{m+1}` is strictly increasing in `y`, hence
`φ_i'` is strictly decreasing and `φ_i` is strictly concave, for **every**
`m > 0`. ∎

**The hypothesis `x_i > 0` is load-bearing and is not dropped anywhere below.**
At `x_i = 0` the numerator `m x_i^m` vanishes, `φ_i' ≡ 0`, and `φ_i` is
constant — not strictly concave — which is exactly convention (C1): the node is
free, B gets through with probability 1 and spends nothing there. So the
strict-concavity and uniqueness statements are claims about the **active set**
`{i ∈ P° : x_i > 0}`.

On that active set, `max_y Σ φ_i(y_i)` subject to `Σ y_i = x̄_B`, `y ≥ 0` is a
strictly concave program with a unique global maximiser. The boundary nodes are
handled by (C1) rather than by the KKT system, and the implementation splits
them out explicitly (`core.evader_alloc`).

**Proposition 2 (KKT + a monotone one-dimensional root).** With multiplier `λ`
on the budget and `t = 1/λ`, stationarity reads

    m x_i^m / [ y_i (x_i^m + y_i^m) ] = λ    ⟺    y_i (x_i^m + y_i^m) = t · m x_i^m .

The left side is strictly increasing in `y_i`, so `y_i(t)` is unique and
strictly increasing in `t`; hence `Σ_i y_i(t)` is strictly increasing in `t`
and `t*` is the unique root of `Σ_i y_i(t) = x̄_B`. `[T-1, T-2]`

**What the implementation returns is a numerical solution, not a symbolic one.**
`core.evader_alloc` locates `t*` by safeguarded Newton inside a maintained
bracket, to the relative tolerance `tolerances.ROOT_TOL`, and then *rescales*
the resulting `y` so that the budget is met exactly in floating point. The
returned vector is therefore a high-precision numerical solution, accurate to
the configured tolerances — the phrase "exact solution" is not used of it.
`core.alloc_residuals` (and `evader_alloc(..., diagnostics=True)`) reports the
root-solving tolerance, the iteration cap, the iterations used, the stopping
criterion, the final root residual before rescaling, and the post-rescale
budget and KKT/stationarity residuals; `[T-17]` checks all of them against
`tolerances.py`.

For `m = 1` the stationarity condition is the quadratic `y² + x y − t x = 0`,
whose only positive root is written in cancellation-free form as

    y_i(t) = 2 x_i t / ( x_i + √(x_i² + 4 x_i t) ) ,
    p_i(t) = 2 t / ( x_i + 2t + √(x_i² + 4 x_i t) ) .

(The textbook form `y = (−x + √(x²+4xt))/2` loses precision when `4xt ≪ x²`;
the two expressions are algebraically identical, and the cancellation-free one
is what the implementation uses. `[T-16]` checks numerically, over randomised
positive `x_i` and `t`, that `p_i(t)` above agrees with `y_i/(x_i + y_i)`
evaluated at the computed `y_i(t)`, and that `y_i(t)` satisfies the
stationarity quadratic.)

**Symmetric special case.** Assume the path has `n` relevant nodes `P°`, that
the defence is symmetric on them (`x_i = c > 0` for all `i ∈ P°`), and that the
evader's budget is allocated at the optimum of the program above. Then by
symmetry and uniqueness on the active set `y_i = x̄_B/n` and

    U_B = ( x̄_B / (n c + x̄_B) )^n .                                       (1)

Note the exponent: **B's success decays geometrically in path length.** `[T-3]`

**Proposition 3 (Danskin / envelope).** Writing
`g_P(x) = max_y Σ φ_i(y_i)`, and **assuming** (a) `x_i > 0`, (b) the inner
maximiser `y*` is unique — which Prop. 1 gives on the active set — and (c) `y*`
is strictly positive there, so that the maximiser varies continuously and
`g_P` is differentiable at `x`, Danskin's theorem gives

    ∂ g_P / ∂ x_i  =  − m x_i^{m−1} / (x_i^m + y_i*^m)
                   =  − 1 / (x_i + y_i*)          for m = 1 . `[T-5]`

These assumptions are stated because they are not automatic. At `x_i = 0` the
function is **nonsmooth**: convention (C1) pins `p_i = 1` there, the expression
above diverges as `x_i → 0⁺` at `m = 1`, and no derivative is claimed. The
implementation never builds a cut at an exactly-sparse point for precisely this
reason (`core.cut_from`, `core.solve_defender`); sparse points are used only to
improve the upper bound, where no derivative is needed. Where `y` is merely
budget-feasible rather than optimal, the same vector is still a valid
subgradient of `h(·, y) ≤ g_P`, which is all the lower bound requires (§5).

---

## 3. The outer problem is convex — this is the key fact

Let `h(x, y) = Σ_{i ∈ P°} [ m log y_i − log(x_i^m + y_i^m) ]`.

**Proposition 4 (convexity).** For `0 < m ≤ 1`, `G(x) = max_P g_P(x)` is convex
on the closed simplex `{x ≥ 0, Σ x_i = x̄_A}`.

*Proof.* The argument is a composition chain; each link is needed.

1. `x_i ↦ x_i^m` is **concave** on `x_i ≥ 0` for `0 < m ≤ 1`.
2. For `u > 0`, `u ↦ −log(u + y_i^m)` is **convex and decreasing**.
3. A convex **decreasing** function composed with a **concave** function is
   convex; hence `x_i ↦ −log(x_i^m + y_i^m)` is convex. (The monotonicity in
   step 2 is what licenses step 3; without it the composition says nothing.)
   The `m log y_i` terms do not involve `x`, so `h(·, y)` is convex in `x` for
   every fixed feasible `y`.
4. The **supremum over the feasible `y`** preserves convexity:
   `g_P(x) = sup_{y ∈ Δ_P} h(x, y)` is a pointwise supremum of convex functions
   over a set `Δ_P` that does **not** depend on `x`, hence convex.
5. The **finite maximum over simple paths** preserves convexity:
   `G = max_P g_P` is a finite max of convex functions, hence convex.

Under convention (C1) the identity `−log(0 + y_i^m) + m log y_i = 0` holds, so
`h(·, y)` extends continuously and convexity holds up to the boundary. ∎
`[T-6]`

**What this establishes, and what it does not.** It establishes **convexity**
of `G` for `0 < m ≤ 1`, not strict convexity. Consequently **(★) is a convex
program**: any local minimum is global, and the cutting-plane lower bound is a
valid global under-estimator, so the method can return a *certified* optimum —
namely, whenever the reported `UB − LB` closes within the requested tolerance.
It does **not** follow that the global minimiser is unique; see §3.1.

`[T-6]` and `[T-13]` sample random chords and find no violation. That
**numerically validates convexity on the sampled instances** and sanity-checks
that the implementation matches the theorem; sampling cannot prove a
universally quantified statement, and the proof above is what carries the
claim.

**Proposition 5 (the proof fails for `m > 1`, and so does the conclusion).**
For `m = 2`, `d²/dx²[−log(x² + c)] = 2(x² − c)/(x² + c)²  < 0` whenever
`x² < c`, so `h(·, y)` is *concave* in `x` on `x_i < y_i` and the sup-of-convex
argument collapses. The substitution `u_i = x_i^m` restores convexity of the
objective but turns the budget set `{Σ u_i^{1/m} = x̄_A}` non-convex, so neither
coordinate system helps.

This is not merely a failed proof: on the diamond graph with `m = 2`,
`x̄_A = x̄_B = 10`, the restriction `f(s) = G(s, 10−s)` satisfies
`½f(2) + ½f(4) < f(3)` — an explicit convexity violation. `[T-13]` (One
counterexample *does* settle a universal claim, unlike the sampling in the
other direction.) This matches Kovenock–Roberson's comment on Hausken (2008),
that non-convexities at high contest intensity are what push such games into
mixed strategies.

**Consequence for `m > 1`.** Because convexity fails, the cutting planes are no
longer global under-estimators and the lower bound is not valid, so no gap —
however small — certifies anything. Numerical experiments with `m > 1` are
therefore labelled **exploratory / non-certified** throughout (E8,
`E8_intensity.csv`, and `certificate["convexity_proved"] = False` in the solver
output), and will stay so unless a separate global method and proof are
supplied.

### 3.1 Uniqueness of `x*` — RESOLVED (was open; a retracted proof is recorded below)

> **STATUS.** Uniqueness of the optimal defender allocation is **proved** below
> (Lemma 2 + Proposition 4c), for `0 < m ≤ 1` in the non-degenerate case
> `V* < 1`. An earlier version of this file claimed uniqueness via a *different*
> and **wrong** argument; that retraction is kept at the end of this subsection,
> both because the error is instructive and because the fix is precisely the
> step the retraction identified as missing.
>
> Like Proposition 4, this is a proof, and the repository cannot check a proof —
> only look for counterexamples. `[T-21]` and `[T-22]` search for
> counterexamples to Lemma 2 and to Proposition 4c and find none, which is
> corroboration, not verification. **This argument should be read by a second
> qualified reader before it is relied on in a submission** (see §9).

Throughout: `0 < m ≤ 1`, the defender's feasible set is
`Δ_A = {x ≥ 0 : Σ_i x_i = x̄_A}` over the relevant nodes `V°`, `n = |V°| ≥ 2`,
and `G(x) = max_P g_P(x)`. Call a path `P` **active at `x`** when it attains
`g_P(x) = G(x)`. Write `V* = exp(G(x*))` for the optimal value, and assume the
**non-degenerate case `V* < 1`**, i.e. `G(x*) < 0`. (If `V* = 1` the defender
cannot cover every path at all, `G ≡ 0` on a whole face, and uniqueness genuinely
fails; §1.2 characterises exactly when this happens.)

**Lemma 2 (active cover).** Let `x*` minimise `G` over `Δ_A`. If `x*_v > 0`
then `v` lies on at least one path active at `x*`.

*Proof.* Suppose not: every path through `v` is inactive, so

    δ  :=  G(x*) − max{ g_P(x*) : v ∈ P° }  >  0

(the max over an empty set is `−∞`, in which case any `δ > 0` works). For
`ε ∈ (0, x*_v)` define `x^ε` by moving `ε` off `v` and spreading it over *all*
the other relevant nodes:

    x^ε_v = x*_v − ε ,        x^ε_i = x*_i + ε/(n−1)   for i ∈ V° \ {v} .

`x^ε ∈ Δ_A` by construction. Take any path `P`.

* If `v ∈ P°`: `g_P` is continuous in `x`, so `g_P(x^ε) → g_P(x*) ≤ G(x*) − δ`
  as `ε → 0`. Hence there is `ε_P > 0` with `g_P(x^ε) < G(x*)` for `ε < ε_P`.
* If `v ∉ P°`: **every** node of `P°` is a node other than `v`, so every one of
  them strictly gains budget. `g_P` is strictly decreasing in each `x_i`,
  `i ∈ P°` (Prop. 3 gives `∂g_P/∂x_i < 0` for `x_i > 0`; at `x_i = 0` raising
  `x_i` off zero strictly lowers `p_i` from 1 by (C1)), and `P°  ≠ ∅` in the
  non-degenerate case. Hence `g_P(x^ε) < g_P(x*) ≤ G(x*)`.

There are finitely many simple paths, so taking `ε` below the minimum of the
finitely many `ε_P` gives `G(x^ε) = max_P g_P(x^ε) < G(x*)`, contradicting the
optimality of `x*`. ∎

This is exactly the step the retraction below flagged as missing: the freed
budget is spread over a set meeting **every** active path simultaneously, and
`V° \ {v}` is such a set precisely because `v` sits on no active path.

**Proposition 4c (uniqueness of the optimal allocation).** For `0 < m ≤ 1` and
`V* < 1`, the minimiser of `G` over `Δ_A` is unique.

*Proof.* Let `x` and `x'` both minimise `G`, and set `z = ½(x + x')`. By
convexity (Prop. 4), `G(z) ≤ ½G(x) + ½G(x') = G(x*)`, and `G(z) ≥ G(x*)`
because `x*` is a minimum; so **`z` is itself a minimiser** and
`G(z) = G(x*)`.

Suppose `x ≠ x'`, and pick `i` with `x_i ≠ x'_i`. At least one of `x_i, x'_i`
is positive, and both are `≥ 0`, so `z_i > 0`. By Lemma 2 applied to the
minimiser `z`, the node `i` lies on some path `P*` active at `z`. Let `y*` be
the evader's optimal split on `P*` at `z`, which is unique (Prop. 1 on the
active set).

Two facts about the coordinates of `P*°`:

* For `j ∈ P*°` with `z_j > 0` the evader must spend there (`y*_j = 0` against
  `z_j > 0` would mean `p_j = 0`, value `−∞`), so `y*_j > 0`.
* For `j ∈ P*°` with `z_j = 0` we get `y*_j = 0` and, since `z_j = ½(x_j+x'_j)`
  with both terms `≥ 0`, also `x_j = x'_j = 0`. So `x`, `x'` and `z` all vanish
  on those coordinates, and by (C1) each contributes `0` to `h(·, y*)` at all
  three points.

Because of the second fact, along the segment joining `x` and `x'` the function
`h(·, y*)` never evaluates a "`y_j = 0` against `x_j > 0`" term, so
`h(·, y*) ≤ g_{P*}` holds at `x`, `x'` and `z`, and `h(·, y*)` is convex there —
strictly convex in each coordinate `j` with `y*_j > 0`, since
`x_j ↦ −log(x_j^m + (y*_j)^m)` has

    d²/dx² [−log(x^m + c)] = −m x^{m−2} [ (m−1)c − x^m ] / (x^m + c)²  >  0

for `c > 0`, `x > 0` and `0 < m ≤ 1` (both bracketed terms are `≤ 0`, and
strictly so for the `−x^m` part), extended to `x = 0` by continuity. Our chosen
`i` has `z_i > 0`, hence `y*_i > 0`, and `x_i ≠ x'_i`. Therefore

    G(z) = g_{P*}(z) = h(z, y*)
         <  ½ h(x, y*) + ½ h(x', y*)      (strict in coordinate i)
         ≤  ½ g_{P*}(x) + ½ g_{P*}(x')
         ≤  ½ G(x) + ½ G(x')  =  G(x*) ,

contradicting `G(z) = G(x*)`. Hence `x = x'`. ∎

**What made the difference.** The retracted argument tried to get uniqueness
from strict convexity of `G` on the whole simplex, which is false. Proposition
4c never claims that. It only needs strict convexity *in the coordinates of one
active path through one coordinate where the two candidates differ*, and Lemma 2
is what guarantees such an active path exists. `G` is still **not** strictly
convex: two points differing only at a node that is off every active path and
carries zero budget give equality — Lemma 2 says such a node carries no budget
at an optimum, which is enough for uniqueness without strict convexity.

**Consequences for the rest of this document.**

* The four closed-form families of §6 have **unique** optimal allocations, so
  the symmetric allocations derived there are *the* optima, not merely *an*
  optimum. The hedge "convexity plus symmetry shows a symmetric allocation
  attains the optimum; it does not assert it is the only one" is no longer
  needed for `V* < 1`.
* `[T-14]`'s repeated-starting-point agreement is now a *consistency check on
  the solver* against a proved theorem, rather than the evidence the claim
  rests on.

---

> **RETRACTION (kept for the record).** An earlier version of this file
> contained a "Proposition 4b" claiming `G` is strictly convex and therefore
> `x*` is unique. **That proof was wrong.** It asserted that because
> `h(x,y) = Σ_{i∈P°} φ_i(x_i,y_i)` sums strictly-convex terms over disjoint
> coordinates, it is strictly convex *in the full vector `x`*. It is not: `h`
> (and hence `g_P`) depends **only on the coordinates lying on `P`**, so if `x`
> and `x'` differ only at nodes off `P` then `h_P(x,y) = h_P(x′,y)` — equality,
> not strict inequality — and the same failure propagates to `G = max_P g_P`.
> `G` is genuinely not strictly convex on the simplex, and
> "strictly convex ⇒ unique minimiser" does not apply to it.
>
> The retraction correctly identified what a repair would need: *"an argument
> that the freed budget can be spread over a set of nodes meeting every active
> path simultaneously (adding budget to a single node only lowers the paths
> through that node, so a single node does not suffice)."* Lemma 2 above is that
> argument — the set is `V° \ {v}`, and it works because a node off every active
> path can be emptied without raising any active path's value.

---

## 4. Certified evader best response without enumerating paths

`max_P g_P(x)` is a *maximum-reliability path* problem.

**A careful statement of the complexity situation.** The path-selection
subproblem is closely related to maximum-reliability path problems known to be
NP-hard — Nguyen–Song–Smith prove NP-hardness (their Theorem 3, via 3SAT) for
the **analogous** problem in their discrete setting. **That result is not a
proof that the exact continuous Tullock-budget model (★) defined in §1 is
NP-hard**, and no such reduction is supplied here or claimed anywhere in this
repository. What follows from the analogy is a design decision, not a theorem:
consequently we use a certified path-generation procedure rather than claiming
a polynomial-time exact oracle. Lagrangian duality is what makes that
certification possible.

**Proposition 6 (separable upper bound).** Define

    ψ(x_i, λ) = max_{y ≥ 0} [ log p_i(y) − λ y ] .

By concavity (Prop. 1) and Slater, for **every** `λ > 0`

    g_P(x)  ≤  λ x̄_B + Σ_{i ∈ P°} ψ(x_i, λ) ,                              (2)

with equality at the optimal `λ`. `[T-4]`

**Lemma 1 (sign of `ψ`, so that the node costs are non-negative).** For every
`x_i ≥ 0` and every `λ > 0`,

    ψ(x_i, λ) ≤ 0 ,     with equality exactly when x_i = 0 .

*Proof.* Fix `y ≥ 0`. Since `x_i^m ≥ 0`, the denominator dominates the
numerator,

    p_i(y) = y_i^m / (x_i^m + y_i^m) ≤ 1     ⟹     log p_i(y) ≤ 0 ,

and `λ y ≥ 0` because `λ > 0` and `y ≥ 0`. Hence the objective
`log p_i(y) − λ y ≤ 0` **pointwise in `y`**, so its supremum over `y ≥ 0`
satisfies `ψ(x_i, λ) ≤ 0`.

For equality: if `x_i = 0` then `p_i(y) = 1` for every `y > 0` by (C1), so the
objective is `−λ y`, whose supremum over `y ≥ 0` is `0`, attained at `y = 0`.
If `x_i > 0` then `log p_i(y) < 0` strictly for every finite `y`, and
`log p_i(y) − λ y → −∞` at both ends (`y → 0⁺` and `y → ∞`), so the continuous
objective attains its maximum at an interior point where it is strictly
negative; hence `ψ(x_i, λ) < 0`. ∎

This is what licenses the shortest-path reduction below: the node costs
`c_i(λ) = −ψ(x_i, λ)` are **non-negative**, so Dijkstra applies and no
general-weight (Bellman–Ford) shortest-path algorithm is needed. `[T-4, T-17]`
check the sign numerically; the proof above is what the claim rests on.

The right-hand side is **additive over the nodes of `P`**, so

    max_P g_P(x)  ≤  λ x̄_B − min_P Σ_{i ∈ P°} c_i(λ),   c_i(λ) = −ψ(x_i, λ) ≥ 0,

and the inner minimisation is a plain **shortest-path problem with
non-negative node costs** — Dijkstra.

*Node/path convention.* The set of nodes entering the product `Π_{i ∈ P°} p_i`,
the set eligible for defender budget, and the set carrying a Lagrangian cost
`c_i(λ)` are the **same** set, determined by `contest_endpoints`:
`core.relevant_nodes` and `core.path_interior` are the single source of truth
for it, and both the code and this document exclude `S` and `D` by default.
Because a node's cost is carried on its *incoming* edges, the source has no
incoming edge on an `S–D` path and its cost must be added separately — the
implementation does so (`_src_cost`), and `[T-15]` checks that omitting it
loosens the bound. `[T-17]` additionally checks that the Dijkstra path cost
equals the direct sum of `c_i(λ)` along the returned path.

*Choosing `λ`.* `λ ↦ λ x̄_B + Σ ψ` is a supremum of affine functions of `λ`,
hence convex, so the tightest bound is found by minimising over `λ > 0`.
Mathematically `λ` ranges over all of `(0, ∞)`; a fixed window on `log₁₀ λ`
could miss the useful multiplier at extreme input scales, producing a loose
upper bound with no visible symptom. The implementation therefore starts from a
**configurable** bracket (`tolerances.LAMBDA_LOG_LO/HI`, overridable per
oracle) and **expands it adaptively** whenever the discrete minimiser lands on
an endpoint, before refining by golden-section search; the final bracket, the
number of expansions and a boundary flag are recorded in
`oracle.last_lambda_search`, so a boundary-limited bound is detected rather
than silently accepted. `[T-20]`

**The certified stopping rule.** Fix the minimising `λ` and enumerate paths in
increasing Lagrangian cost `c_λ(P)` using Yen's algorithm. Because costs are
non-decreasing along that enumeration, once the `k`-th path satisfies

    λ x̄_B − c_λ(P_k)  ≤  best attainable value found so far,

**no later path can beat the incumbent** and the search stops with a proof.
`[T-8]`

Observed behaviour on the graph family tested: on complete layered graphs with
up to `1.7 × 10⁷` distinct `S–D` paths, the oracle evaluated **2** of them (E7).
That is a measurement on those instances at those sizes, not a bound — it says
nothing about worst-case behaviour on other families.

This is the same idea as the log-linearisation Nguyen–Song–Smith use to turn
their constraint-generation subproblem into a shortest-path problem (their
eqs. 11–14), but here the relaxation is tight at `λ*` (Prop. 6) rather than a
first-order approximation of `exp`.

---

## 5. Defender problem: cutting planes with a genuine certificate

**Valid cuts.** For any path `P` and **any budget-feasible** `y` (not
necessarily optimal), `h(·, y) ≤ g_P ≤ G`, and `h(·, y)` is convex. So its
tangent at the current iterate `x_k`,

    θ  ≥  h(x_k, y) + s_kᵀ (x − x_k),    s_k = ∇_x h(x_k, y) ,

is a **global** under-estimator of `G` on the whole simplex. Two consequences
that matter in practice:

* the lower bound is robust to inner-solver error — a slightly sub-optimal `y`
  gives a slightly weaker but still **valid** cut;
* because the cut is valid on `{x ≥ 0}`, the master LP may range over `x ≥ 0`
  and its optimum is a rigorous lower bound for the *unrestricted* problem.

**Master problem (an LP).**

    min_{x, θ}  θ
    s.t.        θ ≥ s_kᵀ x + b_k        for all cuts k
                Σ x_i = x̄_A ,  x ≥ 0 ,  θ ≤ 0 .

**Algorithm.**

```
x ← uniform
repeat
    evade  : best response at x (and at the damped midpoint) → value, cuts
    UB     ← min over all points evaluated so far          (rigorous)
    add cuts (the best response plus a small pool of near-binding paths)
    solve master LP → x_next, LB                            (rigorous)
until UB − LB ≤ tol
```

`UB` is a rigorous upper bound because it is a direct evaluation of `G` at a
feasible point; `LB` is rigorous by the paragraph above. `UB − LB` is therefore
a genuine bracket on the optimum, not a "the iterates stopped moving"
heuristic. `[T-9, T-11, T-18]`

**When this is a certificate.** The method provides a global optimality
certificate **when the computed upper and lower bounds close within the stated
tolerance** — and only then. The bracket is always valid; the *certificate* is
the extra statement `UB − LB ≤ tol`. Three conditions have to hold for it to
mean what it says:

1. `0 < m ≤ 1`, so that the convexity of §3 makes the cuts valid global
   under-estimators. For `m > 1` the lower bound is not valid at all.
2. Both bounds refer to the **same** optimization problem — same graph, same
   budgets, same `m`, same node/path convention (§4).
3. The numerical tolerances are accounted for: `LB` comes from an LP solved to
   `tolerances.LP_FEAS_TOL` and `UB` from allocations solved to
   `tolerances.ROOT_TOL`, so a gap at that scale is at the resolution limit of
   the arithmetic and not a sharper claim. (It is also why `UB − LB` can come
   out very slightly negative when both agree to ~1e−9; the solver reports the
   raw value as `gap_raw` rather than hiding it.)

`core.solve_defender` returns all of this in a `certificate` sub-dict: `log_ub`,
`log_lb`, `value_ub`, `value_lb`, `abs_gap`, `rel_gap` **with its denominator
named**, the requested `tolerance`, a boolean `certified`, a `status` string,
the iteration and cut counts, the runtime, the solver settings, and the
residuals of §5.1. Every experiment table carries the same columns, and a run
that exits at the iteration cap with a nonzero gap is reported `certified=no`.
A result is never described as certified when the gap criterion has not been
met.

**Two numerical details that are easy to get wrong.**

1. *The lower bound must be taken over `x ≥ 0`, not `x ≥ x_min`.* Restricting
   the master shrinks the feasible set and would make `LB` an over-estimate —
   destroying the certificate. Iterates are clamped to `x_min` only after the
   LP, for safety.
2. *The upper bound must be evaluated at exactly sparse points.* Holding
   `x_i ≥ x_min` biases `G` upward by roughly `√(x_min/t)` per node that ought
   to be zero — about `1e-5`, which swamps the target tolerance. The solver
   therefore also evaluates sparsified versions of the incumbent, where
   `x_i = 0` is handled by convention (C1) with no `1/(x_i + y_i)` blow-up.
   Before this fix the reported gaps went **negative** (down to `−1.8e−6`),
   which is the signature of exactly this bug.

### 5.1 Residual checks

The solver computes, and `[T-17]` enforces against `tolerances.py`:

| residual | definition |
|---|---|
| evader budget | `abs(Σ y_i − x̄_B)`, absolute and relative |
| KKT / stationarity | spread of `m x_i^m / (y_i(x_i^m + y_i^m))` across the **active** nodes `x_i > 0` (zero at an exact solution) |
| root equation | relative residual of `y(x^m + y^m) − t·m·x^m` for each numerically solved `y_i` |
| defender budget | `abs(Σ x_i − x̄_A)` |
| certificate | `UB − LB` |
| path-cost consistency | Dijkstra cost vs. the direct sum of `c_i(λ)` along the returned path |

Inactive nodes (`x_i = 0`) are excluded from the stationarity residual on
purpose: there the model sets `p_i = 1`, `y_i = 0` by (C1) and no stationarity
condition is claimed. Tests fail when a residual exceeds its configured
tolerance.

### 5.2 The numerical floor is not part of the model

`x_i = 0` is a legal allocation of the model (C1). The solver keeps its
*iterates* at `x_i ≥ x_min` with
`x_min = max(X_FLOOR_ABS, X_FLOOR_REL · x̄_A/n)` purely to stay away from the
`1/(x_i + y_i)` singularity of the subgradient. To keep that safeguard from
leaking into the answer, the solver also evaluates **exactly sparse** points
(`core.sparsify_alloc`), so a reported `x_i = 0` is the model's boundary
solution and not an artefact of the floor. `[T-19]` checks the boundary case
directly: exact zeros give `p_i = 1`; the value at `(0, x₂, x₃)` equals the
value of the sub-path `(x₂, x₃)`; the value at `x₁ = ε` converges to it as
`ε → 0`, confirming (C1) is the continuous extension; and on an instance whose
optimum genuinely zeroes out nodes, those coordinates come back as exact zeros
rather than pinned at `x_min`, and re-evaluating with exact zeros reproduces
`V*`.

---

## 6. Closed-form benchmarks (independent ground truth)

Derived here, implemented in `closed_form.py`, used as ground truth in `[T-7]`.
All for `m = 1`.

Each family below states the assumptions it needs: the path(s) have the stated
number of relevant nodes, the defence is symmetric on them, and the evader's
budget is allocated symmetrically at the optimum under those homogeneous
conditions. Convexity plus symmetry shows a symmetric allocation *attains* the
optimum; it does not assert it is the only one.

**(a) Single chain, `n` contested nodes.** The unique path's value is a
symmetric function of `(x_1,…,x_n)`, and `G` is convex; averaging over the
symmetric group can only improve the defender (Jensen), so `x_i = x̄_A/n`
attains the minimum. With (1):

    V* = ( x̄_B / (x̄_A + x̄_B) )^n .

**(b) `k` node-disjoint identical chains of length `n`.** Within a branch the
value is symmetric in its own `x`'s, so each branch spreads evenly; across
branches `V_j = (x̄_B/(a_j + x̄_B))^n` is decreasing in `a_j`, so `min max`
equalises:

    V* = ( x̄_B / (x̄_A/k + x̄_B) )^n .

**(c) Complete layered DAG, `L` layers of width `w`** (so `w^L` paths).
Consecutive layers are fully connected, so every `S–D` path meets exactly one
node per layer, and `G` is invariant under permuting nodes within a layer; by
convexity + symmetry a layer-uniform optimum exists, and the problem reduces to
a chain in the per-layer values, giving

    V* = ( x̄_B / (x̄_A/w + x̄_B) )^L .

**Within this family the value depends on layer width and depth only — not on
the number of paths.** This is not merely an observation on the tested
instances; for these families it is a corollary of (b) and (c), which is worth
stating precisely because "number of paths" is the first complexity measure a
reader reaches for.

**Corollary 1 (the path count does not determine the value).** Fix `w ≥ 1`,
`L ≥ 1` and budgets `x̄_A, x̄_B > 0`. Let

* `G₁` be `w` node-disjoint identical chains of length `L`, which has exactly
  **`w` distinct `S–D` paths**; and
* `G₂` be the complete layered DAG with `L` layers of width `w`, which has
  exactly **`w^L` distinct `S–D` paths**.

Then by (b) and (c) respectively,

    V*(G₁)  =  ( x̄_B / (x̄_A/w + x̄_B) )^L  =  V*(G₂) ,

while the ratio of their path counts is `w^{L−1}`, which is unbounded in `L`
for any `w ≥ 2`. Hence **no function of the path count alone can determine the
game value**: two graphs whose path counts differ by an arbitrarily large factor
have exactly the same value, at every budget pair. ∎  `[T-23]`

*What the corollary does and does not say.* It refutes "path count determines
the value" by explicit construction — one counterexample family suffices for a
negative claim. It does **not** say path count is irrelevant on every topology,
and it says nothing about *running time*: the empirical observation that the
oracle evaluates few paths on large layered instances (§4, E7) is separate, is
measured rather than proved, and is reported as observed performance on the
tested family. What the corollary does explain is *why* the two quantities come
apart at all: `V*` is governed by `(w, L)` — how wide the cuts are and how many
of them the evader must cross — while the path count `w^L` is a derived quantity
that varies independently of the pair `(w, L)` that actually sets the value.

**(d) Node-disjoint branches of unequal lengths `L_1,…,L_r`.** Every branch
must be defended (an undefended branch has value 1), so all branches are active
and equalised at the optimum:

    a_j = x̄_B ( V^{−1/L_j} − 1 ) ,     Σ_j ( V^{−1/L_j} − 1 ) = x̄_A / x̄_B ,

with the left side strictly decreasing in `V`, so the root is unique; unlike
(a)–(c) this one has no explicit closed form and `V` is obtained by bisection,
i.e. numerically. Both the value and the per-branch budgets are reproduced by
the solver to `< 4e−9` (E1; `[T-7]`).

---

## 7. What the experiments establish

Throughout this table, "excess" means the metric defined once in
`experiments.py`:

    excess success probability (%) = 100 · (V_heuristic / V* − 1)

i.e. a ratio minus one — not an absolute difference and not a difference of
percentage points. `V*` is taken as the upper bound `V_ub`, so the figures are
conservative. All runs cited as certified reported `certified=yes`, meaning the
gap closed within the stated tolerance.

| Claim | Evidence |
|---|---|
| Solver is correct | matches all four closed-form families to `5.6e−12` (E1, 20/20 certified at tol `1e−11`); never beaten by brute-force grid search (T-9) |
| The bracket is real and does close here | `UB − LB → 0` on the E6 instances, all three `certified=yes` at tol `1e−10`; all 9 E3 topologies and all 20 E5 instances certified |
| …but not always | 8 of the 13 layered instances in E7 exit at the 200-iteration cap with `certified=no`: accurate values, not proved optima |
| Path count is not what drives difficulty **on the families tested** | `1.7e7` paths, oracle evaluating 2, solved in 4.7 s (E7) — observed performance on that family and those sizes, not a complexity claim |
| Uniform defence is sometimes optimal | complete layered / regular grid graphs (E3) — the Ramirez-Marquez–Rocco–Levitin "homogeneous vulnerability" finding |
| …and sometimes badly wrong | irregular graphs: uniform has mean excess **+211.0 %** over 20 random DAGs, worst **+329.5 %** (E5) |
| Min-cut intuition is not enough | `mincut` baseline: mean excess **+220.9 %**, worst **+578.7 %** (E5) |
| Centrality can be catastrophic | `betweenness` scores `V = 1` on the lecture topology (E2) |
| Topology, not size, drives the value | E4: the `x̄_A/x̄_B` response curve differs sharply between chain, grid and random DAGs |
| Uniqueness of `x*` | **proved** (Lemma 2 + Prop. 4c, §3.1); `[T-21]` finds no counterexample to Lemma 2 over 35 instances, `[T-22]` confirms the solver lands on the same `x*` from 5 unrelated starts |
| Path count does not determine the value | **proved** (Corollary 1, §6): `w` parallel chains and the `w^L`-path layered DAG have identical value at every budget pair; `[T-23]` |
| Against the nearest papers' own methods | E9: an RRL-style evolutionary search lands **+3.64 %** above the optimum on average (worst +6.31 %) and an NSS-style linearised constraint generation **+0.61 %** (worst +3.03 %) — both far better than generic heuristics (uniform +204.60 %), and **neither produces any optimality certificate** |
| The certified oracle's advantage is the stopping rule, not the answer | E9(b): on the same instances the Lagrangian separation evaluates **2** paths and certifies; the linearised separation finds the *same* best response but, having no stopping rule, exhausts all 13–57 paths |
| `m > 1` is boundable even though it is not certifiable | E10: the `u = x^m` relaxation gives a **valid** two-sided bracket (e.g. diamond at `m = 2`: `V* ∈ [0.667, 0.800]`), which is not an optimality certificate |
| Stabilising the master closes the open certificates | E7: plain Kelley leaves 8 of 13 layered instances uncertified at the 200-iteration cap; the boxstep master certifies **all 13**, including the 48-contested-node instance (gap `2.1e−3 → 7.2e−10`) in ~¼ the iterations `[T-27]` |
| Arc contests are genuinely a relabelling | `[T-24]`: the line-graph transform sends a chain with `n` contested nodes to one with `n+1`, and the solver reproduces the corresponding closed form to `2.8e−17` |
| Contest intensity helps whoever concentrates | E8: `V*` is **flat in `m`** on a single chain (`0.0625` for every `m`, since `x̄_A = x̄_B` forces `p_i = ½` at any `m`), but rises from `0.118` to `0.940` on `grid-4x4` as `m` goes `0.25 → 3`. Where the defender must spread over parallel routes and the evader can concentrate, economies of scale favour the **evader**. The `m > 1` half of that sweep is **exploratory / non-certified** (§3) |

**Reproducibility.** Every experiment run writes `out/RUN_METADATA.json` with
the random seeds, package and interpreter versions, repository commit, hardware
description, runtime methodology and the full tolerance configuration. Runtimes
are hardware-dependent and are reported as observed performance on that
machine; the numerical results are seeded and regenerate exactly.

---

## 8. Relation to the cited literature

**How citations are used in this section.** A citation is attached to the
statement it actually supports in the cited work. Where a result is our own
theorem or derivation it is labelled as such (§2, §3, §6 are ours) rather than
given a literature citation that does not establish it. In particular the
NP-hardness result below is cited for the **analogous** problem it proves, and
never as proof of a statement about the exact model of §1. Citation details
(authors, title, venue, year) should be checked against the original
publications before any submission; this file uses one style throughout —
*Authors (year), Title*, venue.

**Nguyen, Song & Smith (2023), *A two-stage network interdiction-monitoring
game*.** Same defender/evader/`S–D`-path skeleton and the same
"product of per-arc survival along a path" payoff. Differences: their
defender's decisions are *discrete* (interdict arcs, then choose a monitoring
set) and the second stage is *simultaneous*, so they need mixed strategies and
an LP over path/monitoring distributions; ours is continuous and Stackelberg,
so a pure optimum exists. What transfers directly: constraint generation over
paths is our cutting plane; and their log reformulation of `Π(1−p_a)` into a
shortest-path subproblem is the discrete analogue of our Lagrangian oracle
(§4). Their Theorem 3 establishes NP-hardness for the **max-reliability path
problem in their setting**; that is the reason we adopt certified path
generation instead of claiming a polynomial exact oracle. It is *not* a proof
that (★) is NP-hard, and this document does not use it as one.

**Ramirez-Marquez, Rocco & Levitin (2011), *Optimal network protection against
diverse interdictor strategies*.** Their vulnerability
`v = T^m/(T^m + t^m)` is the same ratio-form CSF with contest intensity `m`,
and their central empirical finding — *equal resource allocation is optimal
under homogeneous component vulnerability* — has a counterpart here that we
derive as a **theorem for our model** (§6a–c, ours) rather than an observation,
with the boundary of its validity made precise: it holds on the symmetric
topologies of §6 under the assumptions stated there, and E3/E5 quantify how
badly it fails off them. Their solution method is an evolutionary
algorithm with no optimality guarantee; the convexity result (§3) lets us
replace it with a certified convex method for `m ≤ 1` — while §3's
counterexample explains why they needed a metaheuristic in the `m > 1` regime.

**Oruganti, Naghizadeh & Ahmed (2023), *The Impact of Network Design
Interventions on the Security of Interdependent Systems*, arXiv:2302.05411.**
Structurally the closest relative found in any of our searches, and the one a
reader from the attack-graph literature is most likely to raise. The skeleton is
the same as ours in four respects: it is a **Stackelberg** game with the
defender moving first; the defender spreads a **single budget** `Σ_i x_i ≤ B`
over the **nodes of a DAG**; the attacker's action is to **choose a path**; and
the payoff along a path is a **product of per-node terms**, giving a
`min_x max_P Π` objective of exactly our shape.

The decisive difference is that **their model contains no attacker-side
investment, and therefore no contest at any node.** Their per-node compromise
probability is a function of the *defender's* investment alone,

    p_i(x_i) = p_i⁰ · e^(−κ_i x_i) ,

with `p_i⁰ ∈ (0,1]` the undefended compromise probability and `κ_i ≥ 1` a
per-node sensitivity; the attacker picks a path and nothing else. In `netgame`
the second player is also budget-constrained and also invests **node by node**,
and `p_i = y_i^m/(x_i^m + y_i^m)` is a ratio-form (Tullock) **contest** between
the two investments. That single change is what generates everything specific to
this project: the inner maximisation over `y` (§2) and its KKT reduction, the
Lagrangian relaxation of that inner problem into a shortest-path oracle (§4),
and the convexity theorem (§3), whose whole content is about how `g_P` depends
on `x` *after* the evader has re-optimised against it. None of these has a
counterpart in a model where the attacker only chooses a path. Their objective
also sums a per-node loss `L_i` along the path (cumulative damage from a
stepping-stone attack) where ours is a single survival product, and their
exponential form makes each path objective log-**linear** in `x`, where ours is
log-concave-composed; the two convexity analyses are therefore not the same
argument. Their §II asserts strict convexity and uniqueness for their objective;
we make the corresponding claim for ours only through Lemma 2 and Prop. 4c
(§3.1), which do not rely on strict convexity of the full objective.

**Bloch, Chatterjee & Dutta (2023), *Attack and interception in networks*,
Theoretical Economics.** The paper closest in spirit found in a literature
check run alongside this project: a single attacker picks a target and a
path through a network, and nodes protect themselves. But the model differs
in every load-bearing respect: (i) it is **decentralized** — each node is its
own player minimising a private quadratic investment cost `x_i²/2` against
its own probability of being attacked, with **no shared budget constraint**
across nodes, whereas here a single defender allocates one fixed budget
`x̄_A` and the allocation problem is genuinely joint; (ii) interception
probability there is **linear**, `(1 − x_i)`, not a ratio-form contest
against an opposing investment — the attacker never chooses a continuous
effort level at all, only a *mixed strategy* over targets/paths; (iii) the
solution concept is **simultaneous Nash equilibrium**, not Stackelberg. Their
uniqueness result is for the (mixed-strategy) Nash equilibrium of a
decentralized game — a different object from the Stackelberg *allocation*
studied here, whose uniqueness is open (§3.1). This paper is
cited here as the nearest neighbour, not as a scoop: the two models answer
different questions (decentralized inspection economics vs. centralized
certified-optimal defense budgeting) and neither nests the other.

**Iliaev, Oren & Segev (2022/2023), *A Tullock-contest-based approach for
cyber security investments*, Annals of Operations Research.** Also uses a
Tullock CSF and compares sequential vs. simultaneous play, which overlaps
with the Stackelberg framing here — but their "assets" are **independent
targets with no graph/path structure**: there is no series system where an
attacker must survive multiple checkpoints in sequence, so the shortest-path
reformulation of §4 (the actual algorithmic contribution of this project)
has no counterpart there. Their contribution is about per-asset investment
incentives under Tullock competition, not about defending a network's
topology against a path-choosing adversary.

**Kovenock & Roberson (2018), *The Optimal Defense of Networks of Targets*,
Economic Inquiry 56(4), 2195–2211 — and the Colonel Blotto lineage.** Blotto
games are the closest *theoretical* cousin of Tullock contests: two
budget-constrained players simultaneously spread resources over battlefields,
which is the simultaneous, all-pay analogue of what happens node-by-node here.
Kovenock and Roberson's line of work on defending networks of targets — and on
weakest-link/series–parallel reliability structures, including their comment on
Hausken already cited in §3 — is the literature that establishes when such
games are forced into **mixed** strategies, which is exactly the regime our
`m > 1` counterexample (Prop. 5) runs into. The differences from this project
are structural rather than incidental: Blotto battlefields are **independent
targets with no path structure**, resolved by a simultaneous all-pay auction
rather than a ratio-form contest, and the solution concept is mixed-strategy
Nash rather than a pure Stackelberg allocation. There is no series system in
which the attacker must survive a *sequence* of contests it has itself chosen,
so the shortest-path reduction of §4 has no analogue. We cite this family as
the relevant theoretical neighbourhood for the `m > 1` regime and as the natural
comparison class for any future simultaneous-play extension (§9), not as a
method we outperform.

**On the novelty check itself.** No claim of the form "first", "only" or "no
prior work" is made anywhere in this repository, because none of them is
supported by a systematic literature review. What was done is narrower: a
search covering the two supplied papers, targeted queries on Tullock-contest
network defense, sequential network interdiction with ratio-form contests, and
Colonel Blotto network extensions, and a later pass that added the attack-graph
defense literature (Oruganti et al., above) and the Blotto-on-networks line
(Kovenock & Roberson, above). Bibliographic details for every work cited in this
section — authors, title, venue, year — were checked against the source, and
the model description attributed to Oruganti et al. was read off their
formulation section rather than their abstract. To the best of that search, no
exact match was found. It still did not, and cannot, replace a full
citation-tracing pass through each of the papers above (e.g. enumerating
everything that has cited Bloch et al. or Nguyen–Song–Smith since publication)
or a search restricted to operations research venues specifically. The accurate framing is
therefore "we study…" / "this repository implements…" / "to the best of our
search…". **Before treating the convexity result (§3) as a novel contribution
in a submission, this should be independently confirmed by the supervising
professor**, who will know this sub-literature (contest theory × network
interdiction) far better than a keyword search can approximate.

---

## 9. Limitations (stated plainly)

1. **`m > 1` carries no guarantee.** Convexity provably fails, so the lower
   bound is not valid and no gap certifies anything there. The solver still
   runs and the numbers look sensible, but those rows are labelled
   `EXPLORATORY / NON-CERTIFIED` in `E8_intensity.csv` and carry
   `convexity_proved = false` in the solver's certificate record.
2. **Stackelberg, not simultaneous.** A moves first and B sees `x`. If both
   moved simultaneously the pure equilibrium could fail to exist and mixed
   strategies over paths would be required — the Nguyen–Song–Smith setting, and
   the natural next step.
3. **The certificate now closes on every instance we test, after stabilising
   the master.** This limitation previously read "the certificate does not
   always close": plain Kelley left `~2e−3` of log-gap on the widest layered
   instances (48–50 contested nodes) at the 200-iteration cap. The named fix —
   a stabilised master — is implemented: `solve_defender(master="boxstep")`
   adds a boxstep/trust-region step selection (the LOWER bound is still taken
   from the *unrestricted* master, so the certificate means exactly what it did
   before), and it certifies all 13 layered instances in roughly a quarter of
   the iterations. It is now the default. What remains open is the *guarantee*:
   there is still **no proven convergence rate** for either master here, only
   the measured behaviour in E6/E7, and nothing rules out a family on which
   boxstep also stalls.
4. **Independence across nodes** is assumed (the payoff is a product). Real
   detection events on a path are usually correlated. See limitation 10.
5. **Arc contests are now implemented** (this limitation is closed). Contesting
   the *arcs* rather than the nodes is the line-graph relabelling
   `graphs.arc_contest_graph`: every arc becomes a node, head-to-tail arc pairs
   become edges, and `S–D` paths correspond bijectively, so every result here
   applies after the transform. `[T-24]` checks the bijection and reproduces the
   corresponding closed form. What is *not* covered is a model with contests on
   both nodes and arcs simultaneously with separate budgets.
6. **The path oracle is certified but not polynomial.** Worst-case it degrades
   to enumeration. The path-selection subproblem is closely related to
   maximum-reliability path problems known to be NP-hard, which is why we use
   certified path generation rather than claim a polynomial-time exact oracle;
   NP-hardness of the exact continuous model (★) itself is neither proved nor
   claimed here (§4).
7. **Uniqueness is proved, but the proof wants a second reader.** §3.1 proves
   the minimiser is unique for `0 < m ≤ 1` when `V* < 1`, via Lemma 2 rather
   than via strict convexity (which genuinely fails). The repository can only
   hunt for counterexamples to a proof, not verify one — and this file has
   carried a *wrong* uniqueness proof before. Treat Lemma 2, Prop. 4c and
   Prop. 4 alike: independently checked before they are relied on (see the
   status box in §3.1 and item 3 of the pre-submission checklist).
8. **Numerical results are numerical.** The inner allocation is solved to
   tolerance and rescaled, not solved symbolically; certificates are stated net
   of the tolerances in `tolerances.py`; and randomised checks validate the
   implementation on sampled instances rather than proving the underlying
   statements.
9. **The `m > 1` bracket is valid but loose, and is not a certificate.** E10
   brackets `V*` for `m > 1` via the `u = x^m` convexification, which is a real
   improvement on having no bound at all. But the relaxation gap does **not**
   shrink as the solver converges — it widens with `m`, reaching a factor of
   124 on a 3-node chain at `m = 3` — and it never certifies that the reported
   allocation is optimal. Characterising the `m > 1` regime properly (does a
   mixed-strategy equilibrium take over?) remains open.
10. **Correlated node outcomes are still assumed away.** The payoff is a strict
   product, i.e. independent detection events along a path. A copula- or
   Markov-based correlated model is a natural extension and is *not*
   implemented; it would change the inner problem's separability and hence the
   shortest-path reduction of §4, so it is a modelling project rather than a
   relabelling. (Arc contests, by contrast, *were* only a relabelling, and are
   now implemented — `graphs.arc_contest_graph`, `[T-24]`.)
11. **Simultaneous play remains a separate project.** See §9 limitation 2: a
   simultaneous version needs mixed strategies over both allocations and paths,
   since a pure equilibrium need not exist. Nothing in this repository
   addresses it, and the Blotto literature cited in §8 is the right starting
   point for it.
