# Experiments and commitments

This page explains what an experiment commits to before its results exist and
how LoFI holds it to those commitments: the preregistration, the freeze, stop
rules and decision points, multiple comparison corrections, and the contraction
ladder that is the only declared way to shrink an experiment once it is frozen.

## Why commitments are records

A factorial comparison of grammars produces many contrasts, and an analysis
chosen after the results are seen can make almost any of them look decisive.
LoFI's answer is to make each analytic choice a value written before the
points run and checked when their rows are ingested. A commitment is thus
not a convention a reader has to trust; it is a record that a later step reads
and refuses to contradict.

## Preregistration

A preregistration is a JSON document with a `grid`, the factors and their
levels, and `hypotheses`, each naming two arms, a family and optionally an
equivalence margin. The packaged example is
[`docs/examples/experiments/preregistration.json`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/experiments/preregistration.json),
which declares one factor and one hypothesis, `recovery`, comparing induced
against gold in the `primary` family.

Reading a preregistration refuses a duplicated level, a factor with no levels,
a duplicated hypothesis name, two hypotheses over the same unordered pair of
arms, a hypothesis comparing an arm with itself, a nonpositive margin, and a
document with no hypotheses. An experiment document must bind every
preregistered factor to a sweep axis, with every level mapped to a value the
axis takes (see [Source and learner declarations](synthetic-designs.md)). At
ingestion, each comparison a reducer produces is checked against the
preregistration: a comparison in an undeclared family is refused, as is one
whose family differs from its hypothesis's, and an audit records which declared
pairs were tested, which were not, and which tested pairs were never declared.

## The freeze

A freeze fixes the code and the sweep together. A freeze identifier has the
form `freeze/<n>`, and it names a git tag. Launching a frozen experiment
requires that the tag exist, that `HEAD` be the tagged commit, and that no
tracked file be modified; otherwise the launch is refused with
`UnfrozenRevisionError`. Rows scored outside a freeze carry the identifier
`UNFROZEN`.

Ingestion is where the freeze is enforced. It refuses an experiment whose
freeze differs from the freeze record, a sweep that is neither the frozen one
nor reachable from it by recorded amendments, a manifest whose directory name is
not its recomputed [point identifier](documents-are-the-dsl.md#point-identity)
or whose point lies outside the frozen sweep, a manifest whose seed was derived
from a different root or whose revision is not the tagged commit, and rows
scored under no freeze or another one. A freeze also admits only a priced
experiment; [`lofi price`](../reference/cli.md) times a point's stacks so that
an experiment's compute can be declared before it is admitted, and a pilot's
results can price the seed count an experiment needs.

## Stop rules and decision points

An experiment may declare stop rules, each a named predicate over its status
record and its report rows. A `points` predicate holds once at least a given
number of points are in the named statuses. A `metric` predicate reduces the
finite values of one evaluator's metric by mean, minimum or maximum and
compares the result strictly with a threshold, and it does not hold until a
declared minimum number of rows exist. A decision point names a date, a
signal and a fallback claim. Before its date it is pending; from its date it
reports whether its signal held or whether the experiment falls back.

Writing results records which stop rules fired and the state of the decision
point in the experiment's status record. The rules are declared before the
results, so whether one fired is a fact about the data rather than a judgment
made after seeing it.

Preregistration records a stopping rule; it does not by itself make repeated
looks at ordinary fixed-sample tests valid. A claim about error control also
requires a test whose assumptions cover the actual sampling and stopping rule.

## Paired tests and equivalence

For $n$ matched observations, let $d_i$ be the left arm score minus the
right arm score and let $t_{\mathrm{obs}}=n^{-1}\sum_i d_i$. A paired sign
randomization assigns a sign $\varepsilon_j\in\{-1,1\}$ to each exchangeable
unit $j$. With $g(i)$ the unit containing observation $i$, its statistic is

$$
t(\boldsymbol{\varepsilon})
=\frac{1}{n}\sum_{i=1}^n\varepsilon_{g(i)}d_i.
$$

A cluster shares one sign across its observations. Exact enumeration uses all
sign assignments; Monte Carlo randomization samples assignments. For $b$
reference statistics at least as large as the observed statistic among $m$
replicates, the upper-tail value is $b/m$ for an enumerated reference and
$(b+1)/(m+1)$ for a sampled reference. The lower tail is analogous, and the
two-sided value is the smaller of one and twice the smaller one-sided value.
The Monte Carlo correction prevents a finite sampled reference from reporting
zero \citep{phipson-smyth2010}. Validity requires the relevant exchangeability
or sign symmetry assumption under the null; matching observations does not
establish that assumption.

An equivalence margin $\delta>0$ specifies the alternative
$-\delta<\mathbb{E}[D]<\delta$, where $D$ is a population paired difference.
The two one-sided tests procedure rejects both boundary null hypotheses,
$\mathbb{E}[D]\leq-\delta$ and $\mathbb{E}[D]\geq\delta$
\citep{schuirmann1987}. LoFI applies its paired randomization test to differences
shifted by these boundaries and uses the larger of the two one-sided values.
Its calibration depends on the randomization assumptions at each boundary.
Failure to reject a zero difference null is not evidence of equivalence.

## Corrections

An experiment's evaluation section declares a significance level, a dependence
assumption among its tests (`independent`, `prds` or `arbitrary`) and a
multiple comparison correction: `bonferroni`, `holm` \citep{holm1979},
`benjamini-hochberg` \citep{benjamini-hochberg1995} or `benjamini-yekutieli`
\citep{benjamini-yekutieli2001}. The pairing is checked: Benjamini–Hochberg
has the stated false discovery rate guarantee under independence or positive
regression dependence on the true nulls, so the configuration refuses it under
`arbitrary` dependence,
while the other three are valid under every declared regime. The declared
correction is applied per preregistered family over the comparisons ingested.
The source comparison reducer applies the same declared correction, under the
same dependence assumption, within each of its own comparison groups, and
records both on every test row.

The dependence assumption is declared, not tested. A result corrected under
`prds` holds if the tests are positively dependent in the required sense, and
the code does not check that they are.

For a family of $m$ raw test values, write their ordered values as
$p_{(1)}\leq\cdots\leq p_{(m)}$. The adjusted values, restored to the original
test order after computation, are

$$
\begin{aligned}
\widetilde p_i^{\mathrm{Bonferroni}}&=\min(1,mp_i),\\
\widetilde p_{(i)}^{\mathrm{Holm}}&=\min\!\left(1,\max_{j\leq i}(m-j+1)p_{(j)}\right),\\
\widetilde p_{(i)}^{\mathrm{BH}}&=\min\!\left(1,\min_{j\geq i}\frac{m}{j}p_{(j)}\right),\\
\widetilde p_{(i)}^{\mathrm{BY}}&=\min\!\left(1,\min_{j\geq i}\frac{mh_m}{j}p_{(j)}\right),
\qquad h_m=\sum_{j=1}^{m}\frac1j.
\end{aligned}
$$

Here $p_i$ is a test value, not a probability mass function. If $V$ counts
false rejections and $R$ counts all rejections, the familywise error rate is
$\mathbb{P}(V\geq1)$ and the false discovery rate is
$\mathbb{E}[V/\max(R,1)]$. Bonferroni and Holm control the former; BH and BY
control the latter under their respective dependence assumptions
\citep{holm1979,benjamini-hochberg1995,benjamini-yekutieli2001}.
All four require valid input test values; a correction cannot repair a
misspecified null model or an invalid stopping rule.

## Contraction

A frozen experiment can turn out to cost more than its budget allows. Rather
than editing the sweep, which ingestion would refuse, an experiment declares a
contraction ladder in advance, and `lofi contract --experiment PATH --rung
RUNG` applies one rung at a time. The rungs are:

| Rung | What it removes |
| --- | --- |
| `unpreregistered-cells` | Axis values that realize no preregistered level. |
| `resolution-iv` | Reduces a two-level grid to a resolution-IV fraction by declared generators. |
| `languages`, `corpus-scale` | The next value of a named axis, in a declared order. |
| `seeds` | Seeds beyond a floor, which may not fall below the evaluation's minimum seed count. |
| `fallback` | No points; the experiment switches to its declared fallback claim. |

The ladder's order is binding. A rung earlier than the last one applied is
refused, as is an undeclared rung, a rung that would remove every value of an
axis, and any rung other than `fallback` that changes no point. Each
application records the rung, what it dropped, the previous and new sweep
identifiers, the remaining point count and the date in the experiment's
amendment ledger, and the ledger must chain. When the experiment is read again,
every recorded rung is reapplied and must drop exactly what it recorded.

What a contracted experiment licenses is the claim its remaining design
supports. Dropping a corpus scale removes the evidence at that scale; falling
back replaces the headline claim with the fallback one. The ledger makes the
contraction visible to every later reader, so that a reduced experiment is not
reported as the one preregistered. To apply a rung, see [Contract an
experiment](../how-to/experiments/contract-an-experiment.md).

## References

\bibliography
