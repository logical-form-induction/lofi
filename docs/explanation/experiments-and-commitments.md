# Experiments and commitments

This page explains what an experiment commits to before its results exist and
how LoFI holds it to those commitments: the preregistration, the freeze, stop
rules and decision points, multiple-comparison corrections, and the contraction
ladder that is the only declared way to shrink an experiment once it is frozen.

## Why commitments are records

A factorial comparison of grammars produces many contrasts, and an analysis
chosen after the results are seen can make almost any of them look decisive.
LoFI's answer is to make each analytic choice a value written before the
points run and checked when their rows are ingested. A commitment is therefore
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
arms, a hypothesis comparing an arm with itself, a non-positive margin, and a
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

## Corrections

An experiment's evaluation section declares a significance level, a dependence
assumption among its tests (`independent`, `prds` or `arbitrary`) and a
multiple-comparison correction: `bonferroni`, `holm` (Holm 1979),
`benjamini-hochberg` (Benjamini and Hochberg 1995) or `benjamini-yekutieli`
(Benjamini and Yekutieli 2001). The pairing is checked: Benjamini–Hochberg
controls the false discovery rate only under independent or positively
regression-dependent tests, so it is refused under `arbitrary` dependence,
while the other three are valid under every declared regime. The declared
correction is applied per preregistered family over the comparisons ingested.
The source-comparison reducer applies the same declared correction, under the
same dependence assumption, within each of its own comparison groups, and
records both on every test row.

The dependence assumption is declared, not tested. A result corrected under
`prds` holds if the tests are positively dependent in the required sense, and
the code does not check that they are.

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
every recorded rung is re-applied and must drop exactly what it recorded.

What a contracted experiment licenses is the claim its remaining design
supports. Dropping a corpus scale removes the evidence at that scale; falling
back replaces the headline claim with the fallback one. The ledger makes the
contraction visible to every later reader, so that a reduced experiment is not
reported as the one preregistered. To apply a rung, see [Contract an
experiment](../how-to/experiments/contract-an-experiment.md).

## References

- Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery
  rate: A practical and powerful approach to multiple testing. *Journal of the
  Royal Statistical Society: Series B (Methodological)*, 57(1):289–300.
  [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
- Yoav Benjamini and Daniel Yekutieli. 2001. The control of the false
  discovery rate in multiple testing under dependency. *The Annals of
  Statistics*, 29(4):1165–1188.
  [doi:10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998).
- Sture Holm. 1979. A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2):65–70.
