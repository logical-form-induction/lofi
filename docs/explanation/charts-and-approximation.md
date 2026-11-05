# Charts and approximation

This page explains how LoFI parses over spans, what the choice of semiring
decides, what pruning and A* widening give up, and what the residual-mass
budget bounds. It separates the approximations that are measured and checked
from those that are only flagged, since the two license different conclusions.

## Spans and the sweep

A chart is indexed by half-open spans `(i, j)` over a sentence of `n` tokens.
[`SpanIndex`](../reference/deduce.md) assigns each span a dense identifier,
grouped by width and then by start, so that the `n(n+1)/2` spans of a sentence
can be stored in flat tensors. Its `decompositions(region, arity)` enumerates
every way of cutting a span into `arity` contiguous non-empty parts, left to
right, so the sweep obtains the premises for any rule arity the grammar uses,
not only binary splits.

The packaged strategy is `cky`: an exhaustive bottom-up traversal from narrow
spans to wide ones, in which every span is visited after all the spans it
decomposes into. Within one width, cells are independent, and the native kernel
may fill up to the budget's `threads` of them concurrently. Every parse inside a
run takes this count: the counterfactual and synthetic-corpus parses, the A\*
decoding parse of held-out evaluation, and the derivation-count census. A grammar with a
unary rule runs unary-closure rounds in each cell, bounded by the budget's
`unary_rounds`. Spans are contiguous with fan-out one; the `RegionIndex`
protocol in [`lofi.formalism`](../reference/formalism.md) admits other fan-outs,
and a consumer that serves only fan-out one refuses a grammar that needs more
(see [Grammars as deductive systems](derivations-as-proofs.md)).

## What a semiring decides

One chart sweep computes different quantities under different semirings
(Goodman 1999). The weight of an item is the semiring sum, over its
derivations, of the semiring product of the weights of their steps.

| Semiring | Computes |
| --- | --- |
| `counting` | The number of derivations, exactly, promoting to arbitrary precision on overflow. |
| `boolean` | Whether an item is derivable at all. |
| `viterbi` | The log probability of the best derivation (max, +). |
| `log-inside` | The log of the summed probability of every derivation (log-add, +). |
| `k-best` | The `k` highest derivation log probabilities. |
| `expectation` | A probability paired with an expected value, for expected counts. |
| `forest` | Every derivation, as edge sequences. |

The run document's `parse.semiring` family registers every semiring above, so
any of them may be the run's parse; each is also available from
[`lofi.deduce`](../reference/deduce.md) for direct use, as [the semiring
tutorial](../tutorials/07-charts-and-semirings.md) shows. A declared semiring
states what its weights can serve beyond the semiring laws. `log-inside` and
`viterbi` weights are float log probabilities, which a beam can rank and A* can
bound; only `log-inside` weights sum over derivations, so only they yield
posterior marginals and goal probability mass. `counting`, `boolean`, `k-best`,
`expectation` and `forest` weights are neither. A component that needs a
capability the declared semiring lacks, such as the `top-k` pruner, A*
decoding, a minimum-Bayes-risk decode or the Bisk--Hockenmaier baseline, is
refused when the run is composed, with an error naming the capability. The
neural losses do not read the symbolic parse, so its semiring does not
constrain them.

The exported conformance properties check that plus is a commutative monoid
with zero as its identity, that times is a monoid, that times distributes over
plus on both sides, and that zero annihilates; every semiring above is tested
against them, the floating-point ones up to a tolerance. Commutativity of times
and idempotence of plus are not checked. Consumers that need more than these
laws say so: outside marginals and pruned goal mass require a semiring that
declares `log_probability`, that is, one whose weights are log probabilities
summed by log-sum-exp (among the built-in semirings, only `log-inside`), and
refuse any other; a normal form other than `none` makes a category's
weight a sum over its normal-form derivations only, so `counting` counts
those.

The choice of semiring is therefore a choice of question. `log-inside` asks how
much probability the model gives a string across all its analyses, which is
what a likelihood objective and constituent marginals need. `viterbi` asks for
the best single analysis, which is what decoding needs. A comparison of two
grammars under `viterbi` compares their best analyses, not their total
probability of the data.

## Inside, outside and marginals

The symbolic outside pass computes, for a goal, the goal-relative inside and
outside weights and the partition, and from them constituent and rule
marginals. It refuses a forest with a cycle. Rule weights in the outside pass
are either shared with the inside pass or declared independently; stacked
outside passes reweight edges by the inside mass they contributed, and stacking
over a chart-derived rule source is refused, because it would carry a span's
own inside weight into its outside weight.

The neural chart parsers in [`lofi.nn.parsers`](../reference/nn.md) use the
same span index and gather tables but compute over tensors, with the algebra
chosen by a log-weight reduction rather than a semiring object. Under
log-sum-exp reduction, marginals are inside plus outside minus the log
partition, and expected rule counts are obtained by differentiating the log
partition. A maximum reduction is not probabilistic and refuses any request
for an expected count. Span functions and chart compositions decide how a
parent's vector is built from its children's; they do not change the
traversal.

## Pruning

A cell pruner keeps a subsequence of a cell's items and drops the rest, with
every step that used one of them. The packaged pruner, `top-k`, is a cell beam:
it keeps the `k` items with the highest inside log weight, earlier items
winning ties. Ranking by a float log weight is meaningful for algebras in which
greater is better, as under `viterbi` and `log-inside`; an algebra weighted
otherwise needs its own pruner. One-token cells are not pruned unless
`prune_lexical` is set, so a beam never removes a lexical reading the source
admitted. Only a chart that retains its forest can be pruned; asking to prune
an inside-only chart is refused.

When a pruner, a unary round bound or a category bound drops anything, the
chart's `truncated` flag is set, and surviving inside weights are recomputed
over what remains. Restricting a chart to items that reach a root is not
truncation and does not set the flag, since it drops only what no root
derivation uses.

A pruned forest also records, beside the flag, what its pruner removed: for
each cell it pruned, the semiring sum of the inside weights the dropped items
held when the cell closed. Under `log-inside` this is the log of the dropped
inside mass, and under `viterbi` it is the best dropped log weight. The record
is local, since an item dropped from a narrow cell would also have contributed
to wider cells it no longer reaches, so the per-cell weights do not add up to
what the whole parse lost. A truncated `log-inside` chart therefore still
underestimates the goal mass by an amount these weights do not settle, and a
truncated `viterbi` chart may miss the best derivation.

The exact loss is measured only on request. A parse declaring
`measure_pruned_mass: true` repeats each pruned parse without its pruner and
records, on the pruned forest, the goal-relative log partition of both. Their
difference in probability is the exact goal mass the beam removed. The
measurement needs `log-inside` weights, since a partition is a sum over
derivations, and it costs one exhaustive parse per sentence, so it is a
diagnostic for calibrating a beam rather than a setting for every run.

The packaged `parse/exhaustive` fragment declares no pruner and `log-inside`;
`parse/viterbi-beam` declares `viterbi` with a beam of eight. The two answer
different questions at different costs, and a result reported under one does
not transfer to the other.

## A* widening

The supertag decoder parses with A* over a supertag-factored lexical source, in
the manner of Lewis and Steedman (2014) and after Klein and Manning (2003). Its
heuristic gives, for each position, the sum of the best lexical log weights at
every other position, which bounds the lexical score still to come. Widening
admits lexical candidates whose global score clears a threshold, keeps at least
one candidate per position, and reruns the parse at successively wider
thresholds, linearly or by binary search, until a spanning root appears. The
result records the attempts and widenings. The decoder stops at the first
threshold that parses; it does not check that the parse found equals the one an
unrestricted lexical beam would yield.

## The residual-mass budget

The residual-mass budget concerns term decoding, not the chart. Where a
component enumerates complete terms from a decoder's distribution, as an
exhaustive term readout or the truncated coherence estimator does, it stops
when the probability mass of the candidates still open falls to a declared
bound, or when a hard `cap` is reached. The mass it did not enumerate is the
residual.

A run declares the bound once, as `approximation.residual_mass`, in `[0, 1)`;
a truncating component may override it. A truncating component with neither a
run bound nor an override is refused when the run specification is resolved.
At run time the measured residual is checked against the bound, and a residual
that exceeds it raises, naming the component. A decoding that pruned nothing
and leaves a residual below `1e-6` is reported as exact rather than truncated.
When two frontiers are enumerated together, the union of their omitted masses
must meet the bound, so each receives `1 - sqrt(1 - r)` of a bound `r`. A
truncated frontier says nothing about a term it did not list, so two frontiers
are compared only over the terms both list, and the residual is the probability
that either decoder lies outside those terms. Where the frontiers list
different terms, this residual exceeds the union of their own omissions, and
the run-time check holds it to the bound.

This is the one approximation in the chart-to-loss path whose cost is declared
and checked. The residual is an upper bound on the probability mass an
objective did not see, so a result under a residual bound of `0.01` and one
under `0.5` are results about different estimators, and an experiment that
varies the bound varies the estimator.

To declare a parser, its semiring and its pruner, see [Configure a chart
parser](../how-to/models/configure-a-chart-parser.md).

## References

- Joshua Goodman. 1999. Semiring parsing. *Computational Linguistics*,
  25(4):573–606. [J99-4004](https://aclanthology.org/J99-4004/).
- Dan Klein and Christopher D. Manning. 2003. A* parsing: Fast exact Viterbi
  parse selection. In *Proceedings of the 2003 Human Language Technology
  Conference of the North American Chapter of the Association for
  Computational Linguistics*, pages 119–126.
  [N03-1016](https://aclanthology.org/N03-1016/).
- Mike Lewis and Mark Steedman. 2014. A* CCG parsing with a supertag-factored
  model. In *Proceedings of the 2014 Conference on Empirical Methods in Natural
  Language Processing*, pages 990–1000, Doha.
  [D14-1107](https://aclanthology.org/D14-1107/).
