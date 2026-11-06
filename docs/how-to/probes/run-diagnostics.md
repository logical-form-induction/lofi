# Run the packaged diagnostics

A `diagnostics` probe runs every diagnostic of the `lofi.diagnostic` catalog
whose required roles a held-out pass reaches, over the points of the roles it
declares. This guide declares the probe, restricts it to chosen roles, reads
what each diagnostic reports, and lists the catalog. Executing a probe and
reading its rows in general is covered in [Probe a trained point](run-probes.md).

## Declare the probe

With no `roles`, the probe reads every role the trained point publishes:

```yaml
probes:
  diag:
    kind: diagnostics
```

[`docs/examples/probes/probed-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/probed-parser.yaml)
declares it on the synthetic parser, and
[Probe a trained point](run-probes.md) trains that run and lists its rows.

`roles` restricts the probe to the points of the named [`Role`](../../reference/nn.md#lofi.nn.sites.Role) members. Probe
the trained run again over two roles:

```bash
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/marginals.json \
  --set 'probes.diag.roles=["CONSTITUENT_MARGINAL", "SPLIT_DISTRIBUTION"]'
```

```text
11 runs/probes/marginals.json
```

```python
from pathlib import Path

from lofi.run.probing import read_probe_rows

for row in read_probe_rows(Path("runs/probes/marginals.json")):
    print(row.measure, row.site, row.value, row.count)
```

```text
finite-readings/live parser/CONSTITUENT_MARGINAL/mu 6.0 1
finite-readings/live parser/SPLIT_DISTRIBUTION/alpha 4.0 1
finite-readings/log-zero parser/SPLIT_DISTRIBUTION/alpha 0.0 1
finite-readings/nonfinite parser/CONSTITUENT_MARGINAL/mu 0.0 1
finite-readings/nonfinite parser/SPLIT_DISTRIBUTION/alpha 0.0 1
finite-readings/nonfinite-share parser/CONSTITUENT_MARGINAL/mu 0.0 1
finite-readings/nonfinite-share parser/SPLIT_DISTRIBUTION/alpha 0.0 1
span-length-bias/normalized-span-length/uniform parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
span-length-bias/normalized-span-length/weighted parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
span-length-bias/spans parser/CONSTITUENT_MARGINAL/mu 2.0 1
cutpoint-entropy/refused None 1.0 1
```

Three diagnostics were selected: `finite-readings`, which requires no role,
`span-length-bias`, which requires `CONSTITUENT_MARGINAL`, and
`cutpoint-entropy`, which requires `SPLIT_DISTRIBUTION`. `split-posterior-kl`
also requires `INSIDE_SCORE` and `OUTSIDE_SCORE`, which the declaration leaves
out, so it is not run. A role name that is not a `Role` member is refused when
the document is validated, and a role no component publishes is refused when
the probe is composed.

## How the rows are formed

Each held-out pass is run once with every observable point of the selected
roles read. A measurement is reported as its mean over the passes that took it,
and `count` is the number of those passes. Where a pass reads exactly one
`CONSTITUENT_MARGINAL` point, its marginals weight the spans, and a quantity a
diagnostic aggregates over spans is reported twice: `…/weighted` under that
weighting and `…/uniform` under equal weights.

Each diagnostic's refused passes are counted under `<tag>/refused`, with the number of refused passes as the value;
`cutpoint-entropy` above refused the one pass it was selected for. Each
diagnostic's docstring in the [probe reference](../../reference/probe.md) lists
the conditions under which it refuses, such as `cutpoint-entropy` when no
eligible span remains.

## The catalog

Start with the behavior to inspect. `finite-readings` detects malformed
numerical values; distributional diagnostics describe how concentrated choices
are; representation diagnostics compare vectors. Low entropy measures concentration; evaluating decision correctness requires
reference outcomes. The [interpretation of probe results](../../explanation/probing-through-sites.md#what-a-probe-result-licenses)
requires a comparison or intervention tied to the proposed linguistic claim.
Each catalog entry is implemented through the [diagnostic API](../../reference/probe.md).

Each diagnostic declares the roles it requires and the scores it reports:

| Tag | Requires | Measures |
| --- | --- | --- |
| `finite-readings` | none | The live positions of each reading that hold no finite number. |
| `span-length-bias` | `CONSTITUENT_MARGINAL` | Normalized length of proper internal spans under the observed marginals. |
| `cutpoint-entropy` | `SPLIT_DISTRIBUTION` | Normalized entropy over non-forced, nonroot split choices. |
| `split-posterior-kl` | `INSIDE_SCORE`, `OUTSIDE_SCORE`, `SPLIT_DISTRIBUTION` | How far DIORA attention lies from its exact split posterior. |
| `action-margin` | `COMBINATOR_LOGITS` | The gap between the two highest action weights, averaged over rows. |
| `controller-choice` | `COMBINATOR_LOGITS` | Confidence, normalized entropy and top-two margin of each [controller](../models/configure-a-controller.md#scores-and-action-probabilities) choice. |
| `controller-relations` | `COMBINATOR_LOGITS` | Declared sequence and similar-input agreement at controller sites. |
| `atomic-complex-differentiation` | `NODE_ENCODING` | How far a [term encoder](../models/configure-term-encoders-and-decoders.md#choose-an-encoder) holds atomic and complex nodes apart. |
| `constructor-embedding` | `NODE_ENCODING` | Whether slots carrying one constructor are encoded alike. |
| `depth-graded-encoding` | `NODE_ENCODING` | How a node's likeness to the root above it moves with its depth. |
| `recursion-spread` | `NODE_ENCODING` | How far a term holds its own nodes apart as it takes on more. |
| `superposition-interference` | `TERM_ENCODING` | Angular interference as type encodings are superposed in row order. |
| `gate-patterns` | `GATE` | The first two moments of every observed recurrent gate activation. |
| `hidden-state-evolution` | `RECURRENT_HIDDEN` | Normalized hidden state displacement from each child to its parent. |
| `memory-cell-dynamics` | `GATE`, `RECURRENT_CELL`, `RECURRENT_HIDDEN` | Displacement and directional retention across memory cell edges. |
| `recurrent-connection-analysis` | `RECURRENT_HIDDEN` | Directional continuity between observed child and parent states. |
| `attention-patterns` | `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS` | Exact observed attention distributions, without labeling heads. |
| `feed-forward-specialization` | `FFN_PREACTIVATION` | Feature sparsity and declared group effects in feedforward preactivations. |
| `residual-contributions` | `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS`, `FFN_PREACTIVATION`, `RESIDUAL_STREAM` | Update and skip norm shares from paired residual stages. |

Which diagnostics a point can run depends on the roles its components publish
and its held-out passes reach. Each component kind has a declared role
inventory ([`lofi/probe/ROLES.md`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/ROLES.md)): a parser always publishes `CHART_CELL`, and
may publish the `INSIDE_SCORE`, `OUTSIDE_SCORE` and `SPLIT_DISTRIBUTION`
structure; a controller publishes `COMBINATOR_LOGITS`; a term encoder publishes
`TERM_ENCODING` and `NODE_ENCODING`, with optional attention, residual and
recurrence roles. On the synthetic parser above, the held-out passes reach only
the parser's sites. Print the catalog of the installed package:

```python
from lofi.probe import installed_diagnostics
from lofi.spec import load_plugins

load_plugins()
catalog = installed_diagnostics()
for tag in catalog.tags:
    kind = catalog.kind(tag)
    print(tag, sorted(role.name for role in kind.requires))
```

```text
action-margin ['COMBINATOR_LOGITS']
atomic-complex-differentiation ['NODE_ENCODING']
attention-patterns ['ATTENTION_LOGITS', 'ATTENTION_WEIGHTS']
constructor-embedding ['NODE_ENCODING']
controller-choice ['COMBINATOR_LOGITS']
controller-relations ['COMBINATOR_LOGITS']
cutpoint-entropy ['SPLIT_DISTRIBUTION']
depth-graded-encoding ['NODE_ENCODING']
feed-forward-specialization ['FFN_PREACTIVATION']
finite-readings []
gate-patterns ['GATE']
hidden-state-evolution ['RECURRENT_HIDDEN']
memory-cell-dynamics ['GATE', 'RECURRENT_CELL', 'RECURRENT_HIDDEN']
recurrent-connection-analysis ['RECURRENT_HIDDEN']
recursion-spread ['NODE_ENCODING']
residual-contributions ['ATTENTION_LOGITS', 'ATTENTION_WEIGHTS', 'FFN_PREACTIVATION', 'RESIDUAL_STREAM']
span-length-bias ['CONSTITUENT_MARGINAL']
split-posterior-kl ['INSIDE_SCORE', 'OUTSIDE_SCORE', 'SPLIT_DISTRIBUTION']
superposition-interference ['TERM_ENCODING']
```

## Mathematical definitions

The linguistic claim determines which inputs and comparisons support an
interpretation of these measurements, including interventions where
appropriate. [Probing through sites](../../explanation/probing-through-sites.md)
explains how to choose those comparisons.
All logarithms below are natural. A live position is one admitted by the
observation mask; missing positions are excluded from the relevant sums.
### Normalization and aggregation

For two hypothetical alternatives, probabilities $(0.5,0.5)$ express less
concentration than $(0.99,0.01)$. Entropy summarizes that spread under the
stated probability interpretation. A span-weighted diagnostic asks about spans
in proportion to their [constituent marginals](../../explanation/charts-and-approximation.md#inside-outside-and-marginals);
an unweighted one gives every included span equal weight. Holding the observed vectors fixed, these two rows can differ because they
weight spans differently.

For live alternatives $J$ and real logits $z_j$, distributional diagnostics
use $p(j)=\exp(z_j)/\sum_{k\in J}\exp(z_k)$. A point declared as probabilities
or log probabilities supplies $p(j)$ or $\log p(j)$ directly. These generic helpers check numeric ranges and masks and calculate directly
from the supplied probabilities. A PMF interpretation requires unit row sums. Attention diagnostics normalize their observed weights separately below. Normalized entropy
\citep{shannon1948}, confidence, and the probability margin are

$$
h(p)=\frac{-\sum_{j\in J}p(j)\log p(j)}{\log|J|},
\qquad c(p)=\max_{j\in J}p(j),
\qquad m(p)=p_{(1)}-p_{(2)},
$$

where $p_{(1)}\geq p_{(2)}$ are the two largest probabilities and
$0\log0=0$. The implementation clips normalized entropy to $[0,1]$;
this does not alter the formula for a PMF but can affect nonunit rows.
A single live alternative has entropy zero. A margin requires
two stored action columns. The probability margin orders the zero-padded
columns, so a single live action and a masked column give $p_{(1)}-0$.
`controller-choice` reports these three quantities.
`action-margin` instead subtracts the two largest **observed weights**,
$z_{(1)}-z_{(2)}$, so a log probability declaration yields a log ratio margin.
Log probability rows with at most one possible action are counted as
`certain` and excluded from the mean gap.

For eligible spans $S$, values $v_s$, and constituent marginals $\mu_s$,
the two within-pass aggregates are

$$
\bar v_{\mathrm{uniform}}=\frac{1}{|S|}\sum_{s\in S}v_s,
\qquad
\bar v_{\mathrm{weighted}}=\frac{\sum_{s\in S}\mu_s v_s}
{\sum_{s\in S}\mu_s}.
$$

Empty eligible sets or zero total weighting are refused where the diagnostic
requires them. Reported means average the per-pass measurements, giving each pass equal weight.

### Chart and controller diagnostics

`finite-readings` counts live entries, nonfinite entries, and their ratio.
For log probability observations, $-\infty$ denotes zero mass and is counted as
`log-zero`, separately from malformed nonfinite readings.
`span-length-bias` sets $v_s=(j-i)/n$ for proper internal spans $[i,j)$ of an
$n$-token sentence, excluding leaves and the root.
`cutpoint-entropy` applies $h$ to split probabilities only at nonroot spans
with at least two admitted split choices.

For DIORA attention $p_s(k)$ at span $s=[i,j)$ and child inside log scores
$a_{i,k},a_{k,j}$, `split-posterior-kl` forms

$$
q_s(k)=\frac{p_s(k)\exp(a_{i,k}+a_{k,j})}
{\sum_{u}p_s(u)\exp(a_{i,u}+a_{u,j})},
\qquad
D_{\mathrm{KL}}(p_s\|q_s)=\sum_kp_s(k)\log\frac{p_s(k)}{q_s(k)}.
$$

The outside score is shared across a span's alternatives and cancels in this
conditional distribution. The diagnostic reports raw KL divergence, averaged
over live split rows separately for each sentence length; it refuses positive
attention with no finite posterior mass. This compares two distributions in
the implemented DIORA computation \citep{drozdov-etal2019,kullback-leibler1951}.

For a declared sequence of [controller](../models/configure-a-controller.md#scores-and-action-probabilities) distributions $p_t$, consistency and
smoothness are

$$
c_{\mathrm{sequence}}=
\exp\!\left(-\frac1{|J|}\sum_{j\in J}\operatorname{Var}_t[p_t(j)]\right),
\qquad
s_{\mathrm{sequence}}=
\exp\!\left(-\operatorname{mean}_{t,j}|p_t(j)-p_{t-1}(j)|\right).
$$

Variance uses its population denominator over the live steps for each
outcome; smoothness averages only adjacent pairs where both positions are
live. No live transition gives smoothness one. `controller-relations`
averages these scores over declared sequences and applies the same
consistency equation to declared similar-input groups. An empty family of
relations receives one. The supplied relations determine which inputs these comparisons group together.
### Term representation diagnostics

Cosine compares directions, so multiplying a nonzero vector by a positive
constant leaves its cosine with another vector unchanged. Norm-based
measurements can detect that magnitude change. Choose cosine to study directional agreement and norms to study magnitude. [Term representations](../../explanation/term-layout-capabilities.md#terms-slots-and-vector-representations)
explains how symbolic expressions enter these learned vector spaces.

For nonzero vectors define
$\operatorname{cos}(\mathbf{u},\mathbf{v})=
\mathbf{u}^{\mathsf T}\mathbf{v}/(\|\mathbf{u}\|_2\|\mathbf{v}\|_2)$.
The node diagnostics assign zero cosine when either vector is zero. Their
Pearson correlation is

$$
\rho(\mathbf{x},\mathbf{y})=
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}},
$$

with zero returned for a constant or empty series. This last value is an
implementation convention for an undefined correlation.

| Diagnostic | Defined measurement |
| --- | --- |
| `atomic-complex-differentiation` | $(1-\operatorname{cos}(\bar{\mathbf{h}}_{\mathrm{atomic}},\bar{\mathbf{h}}_{\mathrm{complex}}))/2$, using the two groups' mean node vectors. A missing group yields counts without a separation score. |
| `constructor-embedding` | For every within-constructor pair cosine $a$ and between-constructor pair cosine $b$, average $\mathbf{1}[a>b]+\tfrac12\mathbf{1}[a=b]$ over their Cartesian product. Both pair sets must be nonempty. |
| `depth-graded-encoding` | Pearson correlation of node depth with cosine similarity to that term's root; the unit interval grading score is its absolute value. |
| `recursion-spread` | For a term with $n\geq2$ live nodes, average $(1-\operatorname{cos}(\mathbf{h}_i,\mathbf{h}_j))/2$ over $i<j$. Report mean spread, its correlation with node count across terms, and the absolute correlation. |
| `superposition-interference` | For successive prefixes of the first at most five term vectors, let $\mathbf{s}_m=\sum_{i=1}^m\mathbf{h}_i$. Report $(1-m^{-1}\sum_{i=1}^m\operatorname{cos}(\mathbf{h}_i,\mathbf{s}_m))/2$. Zero input vectors are refused; zero resultant sums are omitted. The measurement depends on row order. |

Angular and correlation measurements describe the observed encodings.
Assess constructor recovery with decoder evaluations and composition by
comparing the encoded meanings of parts and wholes.

### Recurrent state diagnostics

On each live child–parent edge with vectors $\mathbf{u},\mathbf{v}$, define

$$
d(\mathbf{u},\mathbf{v})=
\frac{\|\mathbf{v}-\mathbf{u}\|_2}{\|\mathbf{u}\|_2+\|\mathbf{v}\|_2}.
$$

`hidden-state-evolution` averages this displacement for hidden states.
`memory-cell-dynamics` reports it for cell states together with their mean
child–parent cosine. `recurrent-connection-analysis` reports mean hidden state
child–parent cosine. Displacement omits edges with two zero vectors; cosine
omits edges with either vector zero. These measurements summarize the states produced along the tree.

For $n$ live scalar gate activations $g_i$, `gate-patterns` reports
$\bar g=n^{-1}\sum_i g_i$ and population variance
$n^{-1}\sum_i(g_i-\bar g)^2$ separately for each gate site.

### Attention, feedforward, and residual diagnostics { #attention-feed-forward-and-residual-diagnostics }

For each query $i$, `attention-patterns` reads the **observed** nonnegative
attention weights $a_{ij}$, using finite logits to check their support $J_i$.
For each live row it normalizes

$$
p_i(j)=\frac{a_{ij}}{\sum_{k\in J_i}a_{ik}},\qquad j\in J_i.
$$

Live rows with zero total weight are refused.

It reports normalized
entropy, maximum mass, self mass $p_i(i)$, local mass
$\sum_{j:|j-i|\leq1}p_i(j)$, and significant mass coverage
$\sum_{j:p_i(j)\geq0.1}p_i(j)$. These quantities are averaged over live queries.
Cross-head consistency is one minus mean normalized Jensen–Shannon divergence
between pairs of heads in the same layer \citep{lin1991}:

$$
\operatorname{JS}(p,q)=\tfrac12D_{\mathrm{KL}}(p\|r)
+\tfrac12D_{\mathrm{KL}}(q\|r),\qquad r=\tfrac12(p+q),
\qquad c_{\mathrm{heads}}=1-\operatorname{mean}\frac{\operatorname{JS}(p,q)}{\log2}.
$$

With natural logarithms, $0\leq\operatorname{JS}(p,q)\leq\log 2$. Compared heads must have matching shapes and masks; only live query rows
contribute. No pair yields no cross-head score. Identifying a head's function requires testing what its computations
contribute to the model's predictions.

For a live feedforward preactivation vector $\mathbf{x}$ with feature set
$J$, the sparsity score is

$$
s(\mathbf{x})=\frac1{|J|}\sum_{j\in J}
\mathbf{1}\!\left[|x_j|\leq0.01\max_{k\in J}|x_k|\right].
$$

An all-zero vector scores one. For caller-declared groups $G_g$ of complete
rows, the feature-specific group effect is

$$
\eta_j^2=\frac{\sum_g |G_g|(\bar x_{g,j}-\bar x_j)^2}
{\sum_g\sum_{i\in G_g}(x_{ij}-\bar x_j)^2},
$$

where the grand mean is over grouped rows. A zero denominator yields zero;
`feed-forward-specialization` reports the mean and maximum effect over
features. These effects depend on the declared grouping.

For a paired residual stage with skip vector $\mathbf{x}$ and observed output
$\mathbf{y}$, let $\mathbf{u}=\mathbf{y}-\mathbf{x}$. The update and skip norm
shares are

$$
r_{\mathrm{update}}=\frac{\|\mathbf{u}\|_2}{\|\mathbf{u}\|_2+\|\mathbf{x}\|_2},
\qquad
r_{\mathrm{skip}}=\frac{\|\mathbf{x}\|_2}{\|\mathbf{u}\|_2+\|\mathbf{x}\|_2}.
$$

Both shares are one half when both norms vanish. Their sum is one by construction. They measure the two input norms; output
energy also depends on cancellation between the update and skip vectors.

## Add a diagnostic

A diagnostic is a class with a `tag`, a frozen set of required [`Role`](../../reference/nn.md#lofi.nn.sites.Role)s, a frozen
set of `scores`, and a `run(trace, *, context)` method returning a
[`DiagnosticResult`](../../reference/probe.md#lofi.probe.DiagnosticResult). It is installed by listing the class in the
`lofi.diagnostic` [entry point group](../../reference/entry-points.md#registered-groups-and-names) under its tag; a class listed under another
tag than the one it declares is refused when the catalog is read.
[Publish plugins](../extending/plugins.md) covers entry points.

## References

\bibliography
