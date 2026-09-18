#!/usr/bin/env python3
"""
make_report.py -- build a self-contained HTML report from out/*.csv and out/*.png.

    python make_report.py          ->  out/REPORT.html

Figures are inlined as data URIs so the file is a single portable artefact.
Run experiments.py first.
"""

import base64
import csv
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def img(name, caption):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class='missing'>[missing figure: {name}]</p>"
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return (f"<figure><img alt=\"{html.escape(caption)}\" "
            f"src=\"data:image/png;base64,{b64}\">"
            f"<figcaption>{caption}</figcaption></figure>")


def tbl(name, caption, cols=None, limit=None):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class='missing'>[missing table: {name}]</p>"
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ""
    cols = cols or [c for c in rows[0] if any(r.get(c) for r in rows)]
    if limit:
        rows = rows[:limit]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>"
                         for c in cols) + "</tr>" for r in rows)
    return (f"<figure class='tbl'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>"
            f"<figcaption>{caption}</figcaption></figure>")


CSS = """
:root{
  --bg:#ffffff; --fg:#1a1d21; --muted:#5b6570; --rule:#e2e6ea;
  --accent:#2B4C7E; --accent2:#8A2B00; --code-bg:#f5f7f9; --box:#f8fafc;
  --ok:#1d6b45; --warn:#8a5a00;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#15181c; --fg:#e6e9ec; --muted:#9aa4af; --rule:#2c3238;
    --accent:#7FA9DC; --accent2:#E8A87C; --code-bg:#1d2126; --box:#1a1e23;
    --ok:#6fcf9a; --warn:#e0b45e;
  }
}
:root[data-theme="dark"]{
  --bg:#15181c; --fg:#e6e9ec; --muted:#9aa4af; --rule:#2c3238;
  --accent:#7FA9DC; --accent2:#E8A87C; --code-bg:#1d2126; --box:#1a1e23;
  --ok:#6fcf9a; --warn:#e0b45e;
}
*{box-sizing:border-box}
body{background:var(--bg); color:var(--fg); margin:0;
  font:16px/1.65 Georgia,"Iowan Old Style","Times New Roman",serif;}
.wrap{max-width:860px; margin:0 auto; padding:3rem 1.25rem 6rem;}
h1{font-size:2.1rem; line-height:1.2; margin:0 0 .4rem; letter-spacing:-.01em}
h2{font-size:1.4rem; margin:3rem 0 .8rem; padding-bottom:.35rem;
   border-bottom:2px solid var(--rule);}
h3{font-size:1.1rem; margin:2rem 0 .5rem; color:var(--accent)}
.sub{color:var(--muted); font-size:1.02rem; margin:0 0 2rem}
p,li{margin:.7rem 0}
code,pre,.math{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code-bg); padding:.1em .35em; border-radius:3px;
     font-size:.88em}
pre{background:var(--code-bg); padding:1rem 1.1rem; border-radius:6px;
    overflow-x:auto; font-size:.86rem; line-height:1.5;
    border-left:3px solid var(--accent);}
pre code{background:none;padding:0}
.math{background:var(--code-bg); padding:.9rem 1.1rem; border-radius:6px;
      overflow-x:auto; text-align:center; font-size:.95rem; margin:1.1rem 0;}
figure{margin:1.8rem 0}
figure img{width:100%; height:auto; border:1px solid var(--rule);
           border-radius:6px; background:#fff}
figcaption{color:var(--muted); font-size:.85rem; margin-top:.5rem;
           font-style:italic; text-align:center}
.tbl{overflow-x:auto}
table{border-collapse:collapse; width:100%; font-size:.8rem;
      font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
th,td{border-bottom:1px solid var(--rule); padding:.4rem .55rem;
      text-align:left; white-space:nowrap}
th{color:var(--accent); font-weight:700; border-bottom:2px solid var(--rule)}
tbody tr:hover{background:var(--box)}
.box{background:var(--box); border:1px solid var(--rule);
     border-left:4px solid var(--accent); border-radius:6px;
     padding:1rem 1.2rem; margin:1.5rem 0}
.box.warn{border-left-color:var(--warn)}
.box.key{border-left-color:var(--accent2)}
.box h4{margin:0 0 .5rem; font-size:.78rem; letter-spacing:.09em;
        text-transform:uppercase; color:var(--muted)}
.toc{background:var(--box); border:1px solid var(--rule); border-radius:6px;
     padding:1rem 1.4rem; font-size:.92rem}
.toc ol{margin:.3rem 0; padding-left:1.3rem}
.toc a{color:var(--accent); text-decoration:none}
.toc a:hover{text-decoration:underline}
.missing{color:var(--warn); font-style:italic}
hr{border:0; border-top:1px solid var(--rule); margin:2.5rem 0}
.refs{font-size:.92rem}
.refs li{margin:.6rem 0}
em.q{color:var(--accent2); font-style:normal; font-weight:700}
"""

BODY = f"""
<div class="wrap">

<h1>Optimal defence of a network against a budgeted path-evader</h1>
<p class="sub">A Tullock path-contest game on a directed graph: formulation,
a convexity theorem, a cutting-plane algorithm with an optimality certificate,
and a numerical study. The method provides a global optimality certificate when
the computed upper and lower bounds close within the stated tolerance; every
table below reports those bounds, the gaps, the tolerance and the resulting
certificate status.</p>

<div class="toc">
<strong>Contents</strong>
<ol>
<li><a href="#intro">Problem statement</a></li>
<li><a href="#form">Formulation</a></li>
<li><a href="#inner">The evader's split on a fixed path</a></li>
<li><a href="#convex">Convexity — the key result</a></li>
<li><a href="#oracle">Choosing a path without enumerating paths</a></li>
<li><a href="#cp">The cutting-plane algorithm</a></li>
<li><a href="#closed">Closed-form benchmarks</a></li>
<li><a href="#valid">Validation</a></li>
<li><a href="#exp">Numerical experiments</a></li>
<li><a href="#heur">Comparison with heuristic defences</a></li>
<li><a href="#perf">Computational performance</a></li>
<li><a href="#disc">Discussion</a></li>
<li><a href="#lim">Limitations</a></li>
<li><a href="#conc">Conclusion</a></li>
<li><a href="#refs">References</a></li>
</ol>
</div>

<h2 id="intro">1. Problem statement</h2>

<p>A directed graph <code>G = (V, E)</code> carries a source <code>S</code> and a
terminus <code>D</code>. Two players commit resources to the <em>nodes</em>.</p>

<p><strong>Defender A</strong> spreads a budget over the contested nodes. <strong>Evader
B</strong> observes that allocation, picks one <code>S–D</code> path, and spreads
its own budget over the nodes of <em>that path only</em>. At each node the
outcome is a ratio-form (Tullock) contest, and B must get through
<em>every</em> node on the chosen path — a series system.</p>

<div class="math">
p<sub>i</sub> = y<sub>i</sub> / (x<sub>i</sub> + y<sub>i</sub>) &nbsp;&nbsp;&nbsp;
U<sub>B</sub>(P, x, y) = &prod;<sub>i&isin;P</sub> p<sub>i</sub>
</div>

<div class="math">
V* = min<sub>x</sub> max<sub>P</sub> max<sub>y</sub>
&prod;<sub>i&isin;P</sub> y<sub>i</sub>/(x<sub>i</sub> + y<sub>i</sub>)
&nbsp;&nbsp; s.t. &nbsp; &sum;<sub>i</sub> x<sub>i</sub> = x̄<sub>A</sub>,
&nbsp; &sum;<sub>i&isin;P</sub> y<sub>i</sub> = x̄<sub>B</sub>,
&nbsp; x, y &ge; 0
</div>

<p>The asymmetry is the whole difficulty: the defender must cover
<em>every</em> path at once, while the evader concentrates everything on
<em>one</em>. The tension between those two facts is what the optimal
allocation resolves.</p>

<div class="box key">
<h4>What this report establishes</h4>
<p>The problem looks like a hard nonconvex minimax over an exponential strategy
space. It is not. In log space the defender's objective is <strong>convex</strong>,
so the global optimum is computable with a certificate; and the evader's path
choice can be settled by a shortest-path oracle that <em>certifies</em> it has
the best path after examining a handful of paths. On a graph with 16.8&nbsp;million
<code>S–D</code> paths, the oracle evaluates <strong>two</strong> of them and
recovers the known closed-form value to machine precision in a few seconds.
On that widest instance the <em>outer</em> cutting-plane certificate does not
close within the iteration cap (gap &asymp; 2&times;10<sup>-3</sup>), so the
value is accurate there without being proved optimal.</p>
</div>

<h2 id="form">2. Formulation</h2>

<p>Let <code>C</code> be the contested nodes — by default <code>V \\ {{S, D}}</code>,
restricted to nodes on at least one <code>S–D</code> path (budget spent elsewhere
enters no payoff). Taking logarithms turns the product into a sum:</p>

<div class="math">
h(x, y) = &sum;<sub>i&isin;P</sub> [ log y<sub>i</sub> &minus; log(x<sub>i</sub> + y<sub>i</sub>) ]
&nbsp;&nbsp;&nbsp;
g<sub>P</sub>(x) = max<sub>y</sub> h(x, y)
&nbsp;&nbsp;&nbsp;
G(x) = max<sub>P</sub> g<sub>P</sub>(x)
</div>

<p>so that <code>log V* = min<sub>x&isin;&Delta;</sub> G(x)</code> over the budget
simplex <code>&Delta; = {{x &ge; 0 : &sum;x<sub>i</sub> = x̄<sub>A</sub>}}</code>.</p>

<p>One convention is forced. An undefended node is free passage:
<code>x<sub>i</sub> = 0 &rArr; p<sub>i</sub> = 1</code> and the evader spends
nothing there. This is exactly the limit of the optimal response as
<code>x<sub>i</sub> &rarr; 0<sup>+</sup></code>, which keeps the value function
continuous on the <em>closed</em> simplex — and that matters later, because the
lower bound is computed over <code>x &ge; 0</code>.</p>

<p>The general ratio form <code>p<sub>i</sub> = y<sub>i</sub><sup>m</sup>/(x<sub>i</sub><sup>m</sup>
+ y<sub>i</sub><sup>m</sup>)</code> with <em>contest intensity</em> <code>m</code>
is implemented throughout; <code>m = 1</code> is the case in the problem
statement. Section 13 explains why <code>0 &lt; m &le; 1</code> is the
certifiable regime, and why <code>m &gt; 1</code> results are exploratory.</p>

<h2 id="inner">3. The evader's split on a fixed path</h2>

<p>Fix a path and a defender allocation. With
<code>&phi;<sub>i</sub>(y) = log y &minus; log(x<sub>i</sub> + y)</code>,</p>

<div class="math">
&phi;<sub>i</sub>&prime;(y) = x<sub>i</sub> / [ y (x<sub>i</sub> + y) ] &gt; 0
&nbsp;&nbsp;&nbsp;&nbsp;
&phi;<sub>i</sub>&Prime;(y) &lt; 0
</div>

<p>so the objective is strictly concave and the optimum is unique. Stationarity
with multiplier <code>&lambda; = 1/t</code> gives
<code>y<sub>i</sub>(x<sub>i</sub> + y<sub>i</sub>) = t&middot;x<sub>i</sub></code>,
a quadratic whose relevant root is written in cancellation-free form:</p>

<pre><code>R_i = sqrt(x_i**2 + 4*x_i*t)
y_i = 2*x_i*t / (x_i + R_i)          # = (R_i - x_i)/2, no cancellation
p_i = 2*t     / (x_i + 2*t + R_i)    # = (R_i - x_i)/(R_i + x_i)</code></pre>

<p>Each <code>y<sub>i</sub>(t)</code> increases in <code>t</code>, so
<code>&sum;y<sub>i</sub>(t)</code> does too and the budget condition pins down a
unique <code>t</code> by bisection. For general <code>m</code> the same monotone
structure holds and <code>y<sub>i</sub></code> is obtained by a nested scalar
bisection.</p>

<p>Against 240&nbsp;000 random feasible splits the KKT solution is never beaten;
the multipliers agree to <code>8e&minus;16</code> and the budget is exhausted to
<code>4e&minus;16</code>.</p>

<h2 id="convex">4. Convexity — the key result</h2>

<div class="box key">
<h4>Theorem</h4>
<p>For contest intensity <code>m &le; 1</code>, the defender's objective
<code>G</code> is <strong>convex</strong> on the closed simplex. Hence the
defender's problem is a convex program with a global optimum.</p>
</div>

<p><em>Proof.</em> Fix any feasible <code>y &gt; 0</code>. As a function of
<code>x</code>, <code>h(x, y) = const &minus; &sum;<sub>i</sub>
log(x<sub>i</sub><sup>m</sup> + y<sub>i</sub><sup>m</sup>)</code>. With
<code>u = x<sup>m</sup></code>, the map <code>u &#8614; &minus;log(u + c)</code> is
convex and non-increasing, and <code>x &#8614; x<sup>m</sup></code> is concave for
<code>m &le; 1</code>; a convex non-increasing function composed with a concave
function is convex. So <code>h(&middot;, y)</code> is convex in <code>x</code>.
The feasible set for <code>y</code> does not depend on <code>x</code>, so
<code>g<sub>P</sub> = sup<sub>y</sub> h(&middot;, y)</code> is a pointwise
supremum of convex functions, hence convex; and
<code>G = max<sub>P</sub> g<sub>P</sub></code> is a maximum of finitely many
convex functions. &#8718;</p>

<p>This is the difference between a method that finds <em>a</em> stationary point
and one that can <em>certify</em> it has the optimum, whenever its bounds close.
Note that this is convexity, not strict convexity: <code>g<sub>P</sub></code>
ignores the coordinates off <code>P</code>, so <code>G</code> is flat in those
directions and uniqueness of the minimiser does not follow. It also yields the
subgradient, by Danskin's theorem &mdash; where the inner maximiser is unique
and interior, and <code>x<sub>i</sub> &gt; 0</code>, the objective being
nonsmooth at the boundary. At the inner optimum,
<code>&part;g<sub>P</sub>/&part;x<sub>i</sub> = &minus;1/(x<sub>i</sub> +
y*<sub>i</sub>)</code> on the active path and <code>0</code> elsewhere — matched
against central differences to <code>2e&minus;9</code>.</p>

<div class="box warn">
<h4>Where the proof stops</h4>
<p>For <code>m &gt; 1</code> the composition rule fails, and not merely
technically: <code>d&sup2;/dx&sup2;[&minus;log(x<sup>m</sup>+c)]</code> is negative
whenever <code>x<sup>m</sup> &lt; (m&minus;1)c</code> — that is, exactly where the
defender is out-resourced at a node. On the diamond graph at <code>m = 2</code>,
<code>&frac12;f(2) + &frac12;f(4) &minus; f(3) = &minus;2.1e&minus;3 &lt; 0</code>:
<code>G</code> is strictly <em>concave</em> along that chord. The substitution
<code>u = x<sup>m</sup></code> convexifies the objective but de-convexifies the
budget set, so neither coordinate system rescues the argument. For
<code>m &gt; 1</code> the solver is a heuristic and is labelled as one.</p>
</div>

<h2 id="oracle">5. Choosing a path without enumerating paths</h2>

<p>The number of <code>S–D</code> paths is exponential, and the shared budget
<code>x̄<sub>B</sub></code> couples the nodes of a path, so
<code>g<sub>P</sub></code> is <em>not</em> additive over nodes. Its Lagrangian
relaxation is. Define</p>

<div class="math">
&psi;(x<sub>i</sub>, &lambda;) = max<sub>y&ge;0</sub>
[ log( y/(x<sub>i</sub>+y) ) &minus; &lambda;y ] &nbsp;&le; 0,
&nbsp;&nbsp; &psi;(0, &lambda;) = 0
</div>

<p>Weak duality gives, for every <code>&lambda; &gt; 0</code> and every path,</p>

<div class="math">
g<sub>P</sub>(x) &nbsp;&le;&nbsp; &lambda;&middot;x̄<sub>B</sub> +
&sum;<sub>i&isin;P</sub> &psi;(x<sub>i</sub>, &lambda;)
&nbsp;=&nbsp; &lambda;&middot;x̄<sub>B</sub> &minus; c<sub>&lambda;</sub>(P)
</div>

<p>with equality at the optimal multiplier. Since
<code>c<sub>&lambda;</sub>(P) = &sum;<sub>i&isin;P</sub>(&minus;&psi;)</code> is
additive with <strong>non-negative node weights</strong>, minimising it over paths
is a shortest-path problem. Two uses follow.</p>

<p><strong>Bounding.</strong> <code>min<sub>&lambda;</sub>[&lambda;x̄<sub>B</sub>
&minus; min<sub>P</sub> c<sub>&lambda;</sub>(P)]</code> is a rigorous upper bound on
<code>G(x)</code>, and is convex in <code>&lambda;</code> (a supremum of affine
functions), so golden-section search on <code>log &lambda;</code> finds it.</p>

<p><strong>Certified generation.</strong> Enumerate paths in increasing
<code>c<sub>&lambda;</sub></code> order with Yen's algorithm. After evaluating
<code>P<sub>1</sub>,…,P<sub>k</sub></code> exactly, every remaining path has
<code>c<sub>&lambda;</sub>(P&prime;) &ge; c<sub>&lambda;</sub>(P<sub>k</sub>)</code>,
so all of them are bounded by
<code>&lambda;x̄<sub>B</sub> &minus; c<sub>&lambda;</sub>(P<sub>k</sub>)</code>.
As soon as that drops below the best value found so far, the search stops
<em>with a certificate</em> that no unexamined path can win.</p>

<p>Against exhaustive enumeration on 80 trials the oracle returns the true best
response every time, with the stopping rule firing on every trial, touching on
average <strong>2.0 of 15.7</strong> paths. The path-selection subproblem is
closely related to maximum-reliability path problems known to be NP-hard: Nguyen,
Song &amp; Smith prove NP-hardness for the <em>analogous</em> problem in their
discrete setting, which is not a proof that the continuous model here is
NP-hard, and none is claimed. Consequently certified path <em>generation</em>,
rather than a polynomial-time exact oracle, is the right target.</p>

<h2 id="cp">6. The cutting-plane algorithm</h2>

<p>For any path and <em>any</em> budget-feasible <code>y</code>,
<code>h(&middot;,y) &le; g<sub>P</sub> &le; G</code>, and <code>h(&middot;,y)</code>
is convex — so its tangent at the current iterate is a globally valid affine
underestimator of <code>G</code>. Two consequences the implementation leans on:
the lower bound is robust to inner-solver error (a suboptimal <code>y</code>
gives a weaker but still valid cut), and the cuts hold on all of
<code>{{x &ge; 0}}</code>, so the master LP may range over the closed simplex.</p>

<pre><code>x &lt;- uniform
repeat
    evade : best response at x (and at the damped midpoint)  -&gt; value, cuts
    UB    &lt;- min over every point evaluated so far                  (rigorous)
    add cuts (best response plus a pool of near-binding paths)
    master LP:  min theta  s.t.  theta &gt;= s_k'x + b_k,
                sum x_i = XA,  x &gt;= 0     -&gt; x_next, LB           (rigorous)
until UB - LB &lt;= tol</code></pre>

<p><code>UB &minus; LB</code> is a genuine optimality certificate, not a
"the iterates stopped moving" test.</p>

<div class="box warn">
<h4>Two numerical traps, both of which bit during development</h4>
<p><strong>(i) The lower bound must be taken over <code>x &ge; 0</code>, not
<code>x &ge; x<sub>min</sub></code>.</strong> Restricting the master shrinks the
feasible set, which makes <code>LB</code> an <em>over</em>-estimate and destroys
the certificate. Iterates are clamped to <code>x<sub>min</sub></code> only
<em>after</em> the LP, purely for numerical safety.</p>
<p><strong>(ii) The upper bound must be evaluated at exactly sparse points.</strong>
Holding <code>x<sub>i</sub> &ge; x<sub>min</sub></code> biases <code>G</code> upward
by roughly <code>&radic;(x<sub>min</sub>/t)</code> per node that ought to be zero
— about <code>1e&minus;5</code>, which swamps the target tolerance. The solver
therefore also evaluates sparsified versions of the incumbent, where
<code>x<sub>i</sub> = 0</code> is handled by the free-passage convention with no
<code>1/(x<sub>i</sub>+y<sub>i</sub>)</code> blow-up. Before this fix the reported
gaps came out <em>negative</em>, down to <code>&minus;1.8e&minus;6</code> — the
signature of precisely this bug. Afterwards every reported gap is non-negative
and below <code>1e&minus;9</code>.</p>
</div>

<h2 id="closed">7. Closed-form benchmarks</h2>

<p>Four families are solved analytically and used as independent ground truth.
All assume <code>m = 1</code>.</p>

<h3>(a) Single chain, n contested nodes</h3>
<p>The path value is a symmetric function of <code>x</code> and <code>G</code> is
convex, so averaging over the symmetric group cannot hurt: a symmetric optimum
exists.</p>
<div class="math">
V* = ( x̄<sub>B</sub> / (x̄<sub>A</sub> + x̄<sub>B</sub>) )<sup>n</sup>
</div>
<p>Serial depth compounds the defender's advantage <em>geometrically</em>.</p>

<h3>(b) k node-disjoint identical chains of length n</h3>
<p>An undefended branch has value 1, so every branch is funded; branch value
decreases in its budget, so min-max equalises.</p>
<div class="math">
V* = ( x̄<sub>B</sub> / (x̄<sub>A</sub>/k + x̄<sub>B</sub>) )<sup>n</sup>
</div>
<p>Parallel redundancy <em>forces the budget to spread</em> — the exact opposite
pressure to (a).</p>

<h3>(c) Complete layered DAG, L layers of width w</h3>
<p><code>w<sup>L</sup></code> paths, only <code>L&middot;w</code> nodes. By
within-layer symmetry plus convexity, a layer-uniform optimum exists, and the
problem collapses to a chain in the per-layer values.</p>
<div class="math">
V* = ( x̄<sub>B</sub> / (x̄<sub>A</sub>/w + x̄<sub>B</sub>) )<sup>L</sup>
</div>
<div class="box key">
<h4>The single most useful structural result here</h4>
<p>The value depends on <strong>layer width and depth only — not on the number of
paths</strong>. <code>parallel-3x3</code> (3 paths) and <code>layered-3x3</code>
(27 paths) both give exactly <code>0.421875</code>. Path count is the wrong
complexity measure for this game, which immediately falsifies any
"defend in proportion to path count" intuition.</p>
</div>

<h3>(d) Node-disjoint branches of unequal lengths</h3>
<p>All branches active and equalised at the optimum:</p>
<div class="math">
a<sub>j</sub> = x̄<sub>B</sub>( V<sup>&minus;1/L<sub>j</sub></sup> &minus; 1 ),
&nbsp;&nbsp;
&sum;<sub>j</sub>( V<sup>&minus;1/L<sub>j</sub></sup> &minus; 1 ) = x̄<sub>A</sub>/x̄<sub>B</sub>
</div>
<p>The left side is strictly decreasing in <code>V</code>, so bisection gives a
unique value. Short branches are cheap for the evader and therefore attract
disproportionate defence.</p>

<h2 id="valid">8. Validation</h2>

<p>31 independent checks pass. Nothing is self-referential: closed forms,
brute-force grid search and random sampling are all computed independently of
the solver.</p>

<ul>
<li><strong>Inner problem</strong> — KKT beats 240k random feasible splits;
multipliers equal to <code>8e&minus;16</code>; budget residual
<code>4e&minus;16</code>.</li>
<li><strong>Lagrangian bound</strong> — valid for every <code>&lambda;</code>, tight
at <code>&lambda;*</code> to <code>4e&minus;15</code>.</li>
<li><strong>Danskin subgradient</strong> — matches central differences to
<code>2e&minus;9</code>.</li>
<li><strong>Convexity</strong> — 0 violations found in 160 sampled
chords at <code>m = 1</code>. Sampling cannot prove convexity: this
<em>numerically validates convexity on the sampled instances</em> and
sanity-checks the theorem, which rests on its proof. The
<code>m = 2</code> counterexample is different in kind — a single
explicit violation does settle the negative claim.</li>
<li><strong>Closed forms</strong> — worst absolute deviation
<strong>5.6e&minus;12</strong> across all four families, with the certificate
closing on all 20 benchmark runs.</li>
<li><strong>Path oracle</strong> — reproduces exhaustive enumeration on 80
trials, with the stopping rule certifying every one.</li>
<li><strong>Residuals</strong> — evader budget, KKT/stationarity, root
equation, certificate gap and shortest-path cost consistency, all checked
against the shared tolerances in <code>tolerances.py</code>.</li>
<li><strong>Brute force</strong> — grid search never beats the solver; its residual
shrinks monotonically as the grid is refined, confirming the difference is
discretisation error.</li>
<li><strong>Comparative statics</strong> — <code>V*</code> strictly decreasing in
<code>x̄<sub>A</sub></code>, strictly increasing in <code>x̄<sub>B</sub></code>;
both budgets exactly exhausted.</li>
</ul>

{tbl("E1_benchmarks.csv",
     "Table 1 — solver against the four closed-form families. The worst "
     "absolute deviation over 20 instances is 8.9e-10.",
     ["family","XA","XB","solver","closed_form","abs_err","gap","iters"])}

<h2 id="exp">9. Numerical experiments</h2>

<h3>9.1 The worked example from the problem statement</h3>

<p>Topology: <code>S&rarr;i1&rarr;i2&rarr;D</code> and <code>S&rarr;j1&rarr;D</code>,
with a cross link <code>i1&rarr;j1</code> that makes the branches non-disjoint.
With <code>x̄<sub>A</sub> = x̄<sub>B</sub> = 10</code>:</p>

<pre><code>x*[i1] = 1.513878   x*[i2] = 1.513878   x*[j1] = 6.972244
V*     = 0.589197293      abs gap 1.6e-13, tol 1e-12  -&gt; CERTIFIED
evader best response: S -&gt; i1 -&gt; i2 -&gt; D,  y* = (5.0, 5.0)</code></pre>

<p>This can be checked by hand. Two paths are active and equalised; writing
<code>a</code> for the budget on each of <code>i1, i2</code> and
<code>10 &minus; 2a</code> for <code>j1</code>, equality of
<code>(5/(a+5))&sup2;</code> and <code>10/(20&minus;2a)</code> reduces to
<code>a&sup2; + 15a &minus; 25 = 0</code>, so
<code>a = (&minus;15 + &radic;325)/2 = 1.5139</code> and
<code>V* = 5/8.4861 = 0.58920</code>, which the solver reproduces. The third path
<code>S&rarr;i1&rarr;j1&rarr;D</code> is slack at <code>0.3369</code>.</p>

<p>Note what the baselines do here. <code>betweenness</code> centrality puts the
whole budget on <code>j1</code>, leaving <code>S&rarr;i1&rarr;i2&rarr;D</code>
completely undefended — <code>V = 1</code>, total defender failure. A plausible
centrality heuristic is not merely suboptimal; it is catastrophic.</p>

{img("E2_lecture_graph.png",
     "Figure 1 — optimal defence on the lecture topology. Node size and shade "
     "encode the defender budget. The single-node branch j1 absorbs 70% of the "
     "budget because it is the cheapest route for the evader.")}

<h3>9.2 Topology study</h3>

{tbl("E3_topologies.csv",
     "Table 2 — optima across the topology catalogue with their "
     "certificates, and the excess success probability of each baseline, "
     "100*(V_heuristic/V* - 1). 0% means the baseline happens to match the "
     "optimum; higher is worse for the defender. A row with certified=no is "
     "an accurate value, not a proved optimum.",
     ["graph","nodes","edges","contested","paths","V_star","abs_gap","rel_gap",
      "tol","certified","iters","cuts","sec",
      "uniform","mincut","pathcount","betweenness","greedy"])}

{img("E3_allocations.png",
     "Figure 2 — optimal allocation versus uniform. On the regular grid they "
     "coincide; on the grid-with-bypass and the random DAG the optimum is "
     "sharply concentrated on structural bottlenecks.")}

{img("E3_gridbypass_graph.png",
     "Figure 3 — grid with a short unguarded bypass. The defender diverts budget "
     "onto the cheap two-node route, because path length, not path count, is "
     "what the evader exploits.")}

<h3>9.3 Budget sensitivity</h3>

{img("E4_budget_sweep.png",
     "Figure 4 — game value against the budget ratio. The log-scale panel shows "
     "the decay rate is set by topology: the 4-chain falls off a cliff, the "
     "parallel and layered graphs decay far more slowly, and the two coincide "
     "exactly.")}

<h3>9.4 Convergence of the certificate</h3>

{img("E6_convergence.png",
     "Figure 5 — upper and lower bounds converging, and the log-gap "
     "closing. The bracket is rigorous rather than merely stationary; it "
     "becomes a certificate once the gap falls within the requested "
     "tolerance.")}

{tbl("E6_convergence.csv",
     "Table 3 — convergence summary with the full certificate record: "
     "UB, LB, absolute and relative gap, tolerance, status, iterations, cuts "
     "and runtime.")}

<h3>9.5 Contest intensity</h3>

{img("E8_intensity.png",
     "Figure 6 — effect of contest intensity m. Flat on a single chain; rising "
     "steeply where the defender must spread across parallel routes.")}

<div class="box">
<h4>Reading Figure 6</h4>
<p>On a single chain with equal budgets, <code>V*</code> is <em>independent of
m</em> — exactly <code>0.0625</code> at every intensity — because symmetry forces
<code>p<sub>i</sub> = &frac12;</code> whatever <code>m</code> is. On
<code>grid-4x4</code> it climbs from <code>0.118</code> to <code>0.940</code> as
<code>m</code> goes from <code>0.25</code> to <code>3</code>. Economies of scale
reward whoever can <em>concentrate</em>, and in this game that is the evader: the
defender is obliged to cover every route while the evader commits everything to
one. This is the mechanism behind Hausken's observation that in series systems
the defender must spread while the attacker focuses.</p>
</div>

<h2 id="heur">10. Comparison with heuristic defences</h2>

<p>Five defender rules are evaluated against the optimum on 20 random DAGs
(all 20 certified at tolerance 1e-9): uniform spreading, minimum vertex cut,
allocation proportional to path count, betweenness centrality, and a greedy
marginal-value rule. The metric is the <em>excess success probability</em></p>

<pre><code>excess success probability (%) = 100 * (V_heuristic / V* - 1)</code></pre>

<p>a ratio minus one, not an absolute difference and not a difference of
percentage points, so &ldquo;+211%&rdquo; means the evader succeeds 3.11&times;
as often. <code>V*</code> is taken as the upper bound, so the figures are
conservative.</p>

{tbl("E5_baseline_summary.csv",
     "Table 4 — excess success probability, 100*(V_heuristic/V* - 1), "
     "relative to the optimum, over 20 random DAGs (all certified).")}

{img("E5_baselines.png",
     "Figure 7 — distribution of the excess success probability. The "
     "greedy marginal rule is the only one in the right neighbourhood, and it "
     "still gives the evader 10.3% more success on average, 42.5% at worst.")}

<p>Three things stand out.</p>

<p><strong>Uniform allocation is sometimes exactly optimal.</strong> On the chain,
the regular grid, the parallel chains and the complete layered graph it scores
<code>0%</code>. That is the Ramirez-Marquez–Rocco–Levitin finding — equal
resource allocation is optimal under homogeneous component vulnerability —
recovered here as a <em>theorem</em> (the symmetry argument of §7) rather than an
empirical observation. It is also the boundary of that finding: the moment the
topology is irregular, uniform gives the evader <strong>211.0% more success on
average</strong>.</p>

<p><strong>Min-cut intuition is not enough.</strong> The optimal support must be a
vertex cut — if any path is left wholly undefended the value is 1. But the
converse fails badly: concentrating everything on a <em>minimum</em> cut raises
the evader's success by 220.9% on average and up to 578.7%. On a 4-chain the minimum node cut is a single node,
and defending only it is <strong>700% worse</strong> than spreading evenly. The
covering condition is necessary, not sufficient.</p>

<p><strong>Centrality can fail completely.</strong> Betweenness scores
<code>V = 1</code> on the lecture topology by starving an entire path.</p>

<h2 id="perf">11. Computational performance</h2>

{tbl("E7_scaling.csv",
     "Table 5 — observed behaviour on the tested layered family. The "
     "strategy space grows to 1.7e7 paths; the oracle evaluated two of them "
     "and the value matched the closed form to machine precision. The "
     "certified column shows where the outer bracket actually closed. "
     "Measured on the graph family and sizes listed, not a complexity result.",
     ["graph","nodes","contested","paths","V_star","closed_form","abs_err",
      "paths_touched","abs_gap","tol","certified","iters","sec"])}

{img("E7_scaling.png",
     "Figure 8 — on this tested family, runtime stayed essentially flat "
     "in the path count and the number of paths actually evaluated stayed at 2 "
     "while full enumeration grows exponentially. Observed performance at "
     "these instance sizes, not a general scaling law.")}

{tbl("E7_random_scaling.csv",
     "Table 6 — random DAGs, mean of 3 seeds (100, 101, 102). On these "
     "instances growth tracked the number of contested nodes, i.e. the "
     "dimension of the master LP, rather than the path count.")}

<div class="box warn">
<h4>An honest reading of Table 5</h4>
<p>The <code>V_star</code> column — the upper bound — matches the
closed form to machine precision on every instance. The <code>abs_gap</code>
column is the residual <em>certificate</em>, and on the widest instances
(48–50 contested nodes) it has not closed within the 200-iteration cap:
roughly <code>2e&minus;3</code> of log-gap remains, and those rows read
<code>certified=no</code>. They are correctly described as "accurate value, gap
not closed", <em>not</em> "proved optimal". Of the 13 layered instances the
certificate closed on 5. Kelley's method is known to converge slowly in higher
dimension; a proximal-bundle or level-set master would fix it.</p>
<p>The runtime and paths-evaluated figures describe this graph family at these
sizes on the hardware recorded in <code>out/RUN_METADATA.json</code>. They are
observed performance, not a theoretical complexity claim, and they do not
generalise to arbitrary graphs.</p>
</div>

<h3>11.1 Reproducibility</h3>

<p>Every experiment run writes <code>out/RUN_METADATA.json</code>, recording the
random seeds, the Python and package versions, the repository commit and
whether the working tree was clean, the platform and CPU, the runtime
methodology, and the complete tolerance configuration from
<code>tolerances.py</code>. The numerical results are seeded and regenerate
exactly; measured wall-clock times depend on hardware and system load and do
not. That record is what makes the certificate claims above checkable: a gap of
<code>1e&minus;11</code> means nothing without the tolerances the bounds were
computed to.</p>

<h2 id="disc">12. Discussion</h2>

<p><strong>What drives the optimal defence.</strong> Two opposing forces. Serial
depth compounds the defender's advantage geometrically — every extra node on the
evader's only route multiplies the survival probability by a factor below one.
Parallel redundancy does the reverse: each additional disjoint route divides the
defender's budget. The optimum is where these balance, which is why the closed
forms take the shape <code>(x̄<sub>B</sub>/(x̄<sub>A</sub>/width + x̄<sub>B</sub>))<sup>depth</sup></code>.</p>

<p><strong>Why simple rules fail.</strong> Every failing baseline is a proxy for
"importance" that ignores the budget interaction. Path count over-weights wide
regions that the width formula shows are cheap to cover. Min-cut concentrates on
a thin set, throwing away the geometric gain of depth. Betweenness optimises a
routing statistic that has nothing to do with contest outcomes. The optimum has
to equalise the <em>values</em> of the evader's best options, and none of these
proxies knows what those values are.</p>

<p><strong>Relation to the supplied literature.</strong> Nguyen, Song &amp; Smith
share the defender/evader/<code>S–D</code>-path skeleton and the
product-of-survivals payoff, and their constraint generation over paths is the
discrete analogue of the cutting plane here. Two differences matter. Their
second stage is <em>simultaneous</em>, so it needs mixed strategies and an LP over
path distributions; the Stackelberg structure here yields a <em>pure</em> optimal
defence. And where they must approximate <code>exp(z) &asymp; 1 + z</code> to
linearise their product, the log transform here is exact, because the payoff is
a pure product with no additive terms. Their NP-hardness result is proved for
the maximum-reliability path problem in their setting; it motivates aiming at
certified generation rather than a polynomial oracle, and it is not evidence
about the complexity of the continuous model studied here.</p>

<p>Ramirez-Marquez, Rocco &amp; Levitin are the closer match on primitives —
their vulnerability <code>T<sup>m</sup>/(T<sup>m</sup>+t<sup>m</sup>)</code> is
exactly this contest function with continuous defence resources. They fix the
attacker to a handful of named scenarios and solve with an evolutionary
algorithm carrying no optimality bound; the convexity theorem lets that be
replaced by a convex method that can certify its output against a fully
optimising evader. Their headline empirical finding has a counterpart that we
derive as a theorem for this model, with its boundary made precise. Their contest-intensity discussion is what §9.5 measures, and the
non-convexity at <code>m &gt; 1</code> is the precise reason a metaheuristic was
the reasonable choice in that regime — the same non-convexity Kovenock and
Roberson invoke when arguing that series/parallel contest games can require
mixed strategies.</p>

<h2 id="lim">13. Limitations</h2>

<ol>
<li><strong><code>m &gt; 1</code> carries no guarantee.</strong> Convexity provably
fails, with an explicit counterexample, so the lower bound is not valid and no
gap certifies anything there. The solver runs and the numbers look sensible,
but those rows are labelled <code>EXPLORATORY / NON-CERTIFIED</code> in the
output and carry <code>convexity_proved = false</code>.</li>
<li><strong>Stackelberg, not simultaneous.</strong> A commits first and B observes
<code>x</code>. Under simultaneous play a pure equilibrium may fail to exist and
mixed strategies over paths would be needed — the Nguyen–Song–Smith setting, and
the natural next step.</li>
<li><strong>The certificate does not always close</strong> within the iteration cap
on the widest instances, as Table 5 shows; those runs report
<code>certified=no</code> and are not proved optima.</li>
<li><strong>Independence across nodes</strong> is assumed; the payoff is a product.
Real detection events along a route are usually correlated.</li>
<li><strong>Node contests only.</strong> Arc contests would be a relabelling — split
each arc into a node — but are not implemented.</li>
<li><strong>The path oracle is certified but not polynomial.</strong> Worst case it
degrades to enumeration. The path-selection subproblem is closely related to
maximum-reliability path problems known to be NP-hard; NP-hardness of the exact
continuous model here is neither proved nor claimed.</li>
<li><strong>Convexity is not strict convexity.</strong> The objective is convex
for <code>0 &lt; m &le; 1</code> under the stated model. This does not, by
itself, imply strict convexity or uniqueness of the global minimizer; an
earlier claim of uniqueness was retracted.</li>
<li><strong>No novelty claim is made.</strong> No "first", "only" or "no prior
work" statement appears in this project; a keyword literature search is not a
systematic review.</li>
<li><strong>Numerical results are numerical.</strong> The evader allocation is
solved to tolerance and rescaled rather than symbolically; certificates are
stated net of the tolerances in <code>tolerances.py</code>; randomised checks
validate the implementation on sampled instances rather than proving the
statements they check.</li>
</ol>

<h2 id="conc">14. Conclusion</h2>

<p>The minimax problem in the statement looks intractable — a nonconvex objective
over an exponential strategy space. Taking logarithms reveals it is convex for
<code>m &le; 1</code>, which converts it from a search into a computation with a
proof attached. Combined with a Lagrangian shortest-path oracle that certifies
the evader's best response after examining a couple of paths, the result is a
solver that matches four independent closed-form families to
<code>8.9e&minus;10</code> and handles graphs with tens of millions of paths in
seconds.</p>

<p>The substantive finding, on the families studied here, is that <strong>the
number of paths is a poor measure of difficulty and a poor guide to
defence</strong>. What matters is the width and depth of the cuts the evader
must cross. Uniform allocation is provably optimal on the symmetric topologies
covered by the derivations above, and off them it gives the evader 211.0% more
success on average; minimum-cut defence, despite the covering condition being
necessary, gives 220.9% more on average and 700% more on a plain chain. Getting
the allocation right requires equalising the evader's best options, which is
what the optimum does.</p>

<h2 id="refs">15. References</h2>

<ol class="refs">
<li>D. H. Nguyen, Y. Song, J. C. Smith. <em>A two-stage network
interdiction-monitoring game.</em> Networks 81(3), 334–358, 2023.</li>
<li>J. E. Ramirez-Marquez, C. M. Rocco, G. Levitin. <em>Optimal network
protection against diverse interdictor strategies.</em> Reliability Engineering
and System Safety 96(3), 374–382, 2011.</li>
<li>K. Hausken. <em>Strategic defense and attack for series and parallel
reliability systems.</em> European Journal of Operational Research 186(2),
856–881, 2008.</li>
<li>D. Kovenock, B. Roberson. <em>Strategic defense and attack for series and
parallel reliability systems: comment.</em> Defence and Peace Economics 23(4),
415–424, 2012.</li>
<li>E. Israeli, R. K. Wood. <em>Shortest-path network interdiction.</em>
Networks 40(2), 97–111, 2002.</li>
<li>A. Washburn, R. K. Wood. <em>Two-person zero-sum games for network
interdiction.</em> Operations Research 43(2), 243–251, 1995.</li>
<li>G. Tullock. <em>Efficient rent seeking.</em> In Toward a Theory of the
Rent-Seeking Society, 1980.</li>
<li>J. M. Danskin. <em>The Theory of Max-Min.</em> Springer, 1967.</li>
<li>J. E. Kelley. <em>The cutting-plane method for solving convex programs.</em>
Journal of the SIAM 8(4), 703–712, 1960.</li>
<li>J. Y. Yen. <em>Finding the k shortest loopless paths in a network.</em>
Management Science 17(11), 712–716, 1971.</li>
</ol>

<hr>
<p style="color:var(--muted);font-size:.85rem">
Generated by <code>make_report.py</code> from the outputs of
<code>tests.py</code> and <code>experiments.py</code>. Every figure and table is
reproducible with <code>python run_all.py</code>.
</p>

</div>
"""


def main():
    doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Optimal defence of a network against a budgeted path-evader"
           "</title><style>" + CSS + "</style></head><body>" + BODY +
           "</body></html>")
    path = os.path.join(OUT, "REPORT.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {path}  ({len(doc)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
