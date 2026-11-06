# Charts and approximation

This page explains how LoFI parses over spans, what the choice of semiring
decides, what pruning and A* widening give up, and what the residual-mass
budget bounds. It separates the approximations that are measured and checked
from those that are only flagged, since the two license different conclusions.

## Spans and the sweep

For a string $w_1\cdots w_n$, a chart is indexed by half-open spans
$[i,j)$, where $0\leq i<j\leq n$. The span covers
$w_{i+1}\cdots w_j$, has width $j-i$, and admits binary split points
$i<k<j$.
[`SpanIndex`](../reference/deduce.md) assigns each span a dense identifier,
grouped by width and then by start, so that the $n(n+1)/2$ nonempty spans of a sentence
can be stored in flat tensors. Its `decompositions(region, arity)` enumerates
every way of cutting a span into `arity` contiguous non-empty parts, left to
right, so the sweep obtains the premises for any rule arity the grammar uses,
not only binary splits.

The packaged strategy is `cky`, a bottom-up traversal of the kind used in
Cocke–Kasami–Younger parsing \citep{younger1967,goodman1999}. It proceeds from narrow
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
\citep{goodman1999}. The weight of an item is the semiring sum, over its
derivations, of the semiring product of the weights of their steps.

Let $(K,\oplus,\otimes,0_K,1_K)$ be a semiring. In a finite acyclic
packed forest, an edge $e$ has parent $v$, an ordered list of children
$\operatorname{ch}(e)$, and weight $w_e\in K$. With lexical alternatives
represented as nullary edges, the inside recurrence is:

$$
\alpha_v=
\bigoplus_{e:\operatorname{pa}(e)=v}
\left(w_e\otimes\bigotimes_{u\in\operatorname{ch}(e)}\alpha_u\right).
$$

The empty product is $1_K$ and the empty sum is $0_K$. Child order matters
when multiplication is noncommutative. This finite recurrence does not by
itself define a convergent sum over infinitely many cyclic derivations.
Finite unary unrolling computes the derivations admitted by that unrolling,
not an unbounded unary closure \citep{goodman1999}.

| Semiring | Computes |
| --- | --- |
| `counting` | The number of derivations, exactly, promoting to arbitrary precision on overflow. |
| `boolean` | Whether an item is derivable at all. |
| `viterbi` | The largest derivation log weight, using $(\max,+)$. |
| `log-inside` | The log of the sum of derivation weights, using $(\operatorname{logsumexp},+)$. |
| `k-best` | The $k$ highest derivation log weights. |
| `expectation` | A total weight paired with an unnormalized weighted statistic; normalize to obtain an expectation. |
| `forest` | Every derivation, as edge sequences. |

The run document's `parse.semiring` family registers every semiring above, so
any of them may be the run's parse; each is also available from
[`lofi.deduce`](../reference/deduce.md) for direct use, as [the semiring
tutorial](../tutorials/07-charts-and-semirings.md) shows. A declared semiring
states what its weights can serve beyond the semiring laws. `log-inside` and
`viterbi` weights are floating-point log weights, which a beam can rank and A* can
bound; only `log-inside` weights sum over derivations, so only they yield
posterior marginals and summed goal weight through these interfaces. `counting`, `boolean`, `k-best`,
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
declares `log_probability`, the API capability for log weights summed by
log-sum-exp (among the built-in semirings, only `log-inside`), and
refuse any other; a normal form other than `none` makes a category's
weight a sum over its normal-form derivations only, so `counting` counts
those.

A log weight is not automatically a log probability. Write
$s(d)=\sum_{e\in d}s_e$ for a derivation's additive score, including its
root weight, and let $\mathcal D(w)$ contain the derivations admitted by the
forest and goal. For $0<z(w)<\infty$,

$$
z(w)=\sum_{d\in\mathcal D(w)}\exp s(d),
\qquad
p(d\mid w)=\frac{\exp s(d)}{z(w)}.
$$

This defines a conditional PMF over derivations. The scalar $z(w)$ is a
partition function; it is a string probability only under an additional
normalized generative model. Viterbi computes $\max_d s(d)$, whereas
log-inside computes $\log z(w)$. Their values answer different questions.
When $z(w)=0$, no accepted positive-weight derivation exists and the
conditional PMF is undefined; the outside API reports zero marginals.

For the first-order expectation semiring, a pair $(z,\boldsymbol{r})$
accumulates a total weight and a weighted additive statistic. Its operations are
\citep{li-eisner2009}:

$$
\begin{aligned}
(z,\boldsymbol{r})\oplus(z',\boldsymbol{r}')
  &=(z+z',\boldsymbol{r}+\boldsymbol{r}'),\\
(z,\boldsymbol{r})\otimes(z',\boldsymbol{r}')
  &=(zz',z\boldsymbol{r}'+z'\boldsymbol{r}).
\end{aligned}
$$

Initialize each edge with $(a_e,a_e\boldsymbol{f}_e)$, where
$a_e=\exp(s_e)$ and $\boldsymbol{f}_e$ is its statistic vector. Then
$\boldsymbol{f}(d)=\sum_{e\in d}\boldsymbol{f}_e$ is the additive
derivation statistic. If $D$ is a random derivation with PMF
$p(\cdot\mid w)$ and the final weight $z$ is positive, the expectation is
$\mathbb{E}[\boldsymbol{f}(D)\mid w]=\boldsymbol{r}/z$.
The second component is not already a normalized expectation.

## Inside, outside and marginals

The symbolic outside pass computes, for a goal, the goal-relative inside and
outside weights and the partition, and from them constituent and rule
marginals. It refuses a forest with a cycle. Rule weights in the outside pass
are either shared with the inside pass or declared independently; stacked
outside passes reweight edges by the inside mass they contributed, and stacking
over a chart-derived rule source is refused, because it would carry a span's
own inside weight into its outside weight.

With shared edge weights in the sum-product setting, write $\alpha_v$ and $\beta_v$ for inside
and outside weights in ordinary, nonlog units. For a root goal, outside
weights are initialized with the root factors; for each parent edge, a
child receives the parent's outside weight times the edge weight and the
other children's inside weights. The resulting constituent and edge
marginals are \citep{goodman1999}:

$$
\begin{aligned}
\mathbb{P}(v\in D\mid w)&=\frac{\alpha_v\beta_v}{z(w)},\\
\mathbb{P}(e\in D\mid w)&=
\frac{\beta_{\operatorname{pa}(e)}\exp(s_e)
      \prod_{u\in\operatorname{ch}(e)}\alpha_u}{z(w)}.
\end{aligned}
$$

These are occurrence probabilities for the acyclic, contiguous-span forest
where an item or edge occurs at most once in a derivation. For a selected
cover with fragment-root groups $\mathcal R_1,\ldots,\mathcal R_m$, the goal
partition is instead the product of group sums:

$$
z_{\mathrm{cover}}(w)=\exp(\operatorname{penalty}(m))
\prod_{r=1}^{m}\left(\sum_{v\in\mathcal R_r}
\exp(\omega_v)\alpha_v\right).
$$

Here $\omega_v$ is the root log weight, zero for the current cover goal.
The outside seed for one group includes the totals of the other groups.
This computes alternatives within the selected fragments, not a sum over
all tilings; see [Goals](derivations-as-proofs.md#goals).

This probability interpretation requires `SHARED_RULE_WEIGHTS`, or an
outside rule source assigning exactly the same edge weights. The API also
allows independently weighted outside propagation. In that case, its inside
weights and partition still come from the held forest, while its outside
messages use the independent source. The resulting ratios need not be
posteriors under a single PMF, despite the API's marginal terminology.

The neural chart parsers in [`lofi.nn.parsers`](../reference/nn.md) use the
same span index and gather tables but compute over tensors, with the algebra
chosen by a log-weight reduction rather than a semiring object. Under
log-sum-exp reduction, log marginals are log-inside plus log-outside minus
$\log z(w)$. If $s_e$ is an independently varied additive edge potential,
then $\partial\log z(w)/\partial s_e=\mathbb{E}[N_e(D)\mid w]$, where
$N_e(D)$ counts edge occurrences \citep{li-eisner2009}. A derivative through
additional neural dependencies is not automatically that edge marginal. A maximum reduction is not probabilistic and refuses any request
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
what the whole parse lost. For a fixed root goal, a truncated `log-inside` chart thus
underestimates the goal weight by an amount these weights do not settle, and a
truncated `viterbi` chart may miss the best derivation.

The exact loss is measured only on request. A parse declaring
`measure_pruned_mass: true` repeats each pruned parse without its pruner and
records, on the pruned forest, the goal-relative log partition of both. For a fixed root goal and unchanged edge weights, let $z$ and $\widetilde z$
be the unpruned and pruned partitions. Then

$$
0\leq\widetilde z\leq z,
\qquad
\delta=1-\frac{\widetilde z}{z}
       =1-\exp(\log\widetilde z-\log z)
\quad(z>0).
$$

The difference $z-\widetilde z$ is the lost unnormalized weight; $\delta$
is the probability of an omitted derivation under the reference forest's
conditional PMF. The reference parse still has the same category and unary
bounds. If reparsing changes the cover's selected tiling, the two partitions
need not describe nested derivation sets; the subset interpretation then
requires holding the selected cover fixed. The
measurement needs `log-inside` weights, since a partition is a sum over
derivations, and it costs one additional parse without the cell pruner per sentence, so it is a
diagnostic for calibrating a beam rather than a setting for every run.

The packaged `parse/exhaustive` fragment declares no pruner and `log-inside`;
`parse/viterbi-beam` declares `viterbi` with a beam of eight. The two answer
different questions at different costs, and a result reported under one does
not transfer to the other.

## A* widening

The supertag decoder uses an A*-inspired lexical bound to widen candidate
sets and rerun the chart parser. A* parsing uses bounds on unfinished
derivations to guide search \citep{klein-manning2003};
\citet{lewis-steedman2014} develop this approach for supertag-factored CCG.
LoFI's widening procedure does not implement their best-first A* agenda. Its
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
component enumerates complete terms from a decoder’s weighted frontier, as an
exhaustive term readout or the truncated coherence estimator does, it stops
when the probability mass of the candidates still open falls to a declared
bound, or when a hard `cap` is reached. The mass it did not enumerate is the
residual.

A run declares the bound once, as `approximation.residual_mass`, in `[0, 1)`;
a truncating component may override it. A truncating component with neither a
run bound nor an override is refused when the run specification is resolved.
At run time the measured residual is checked against the bound, and a residual
that exceeds it raises, naming the component. An exhausted decoding frontier that pruned nothing and leaves a residual
at most its `exact_residual` threshold (default $10^{-6}$) is reported as
exact rather than truncated.
For locally normalized decoding choices, let $p$ denote the induced
subprobability mass function on admitted complete terms. For a finite
enumerated set $S_p$, the residual is $1-p(S_p)$, with
$p(S_p)=\sum_{u\in S_p}p(u)$. It includes both unenumerated terms and
rejected or dead-end paths. A PMF over admitted complete terms requires
their total mass to be one, or an explicit renormalization.

Paired decoding compares two such term measures $p$ and $q$.
If term mappings are configured, each side first maps its terms and sums
the weights of terms with the same image. Here $p$, $q$, $S_p$, and $S_q$
refer to the resulting mapped distributions and enumerated supports.
LoFI compares the common set $S=S_p\cap S_q$ when either frontier is
truncated. If $S$ has positive mass on both sides, it evaluates the declared
divergence on:

$$
p_S(u)=\frac{p(u)}{p(S)},\qquad
q_S(u)=\frac{q(u)}{q(S)}\quad(u\in S),
\qquad
r=1-p(S)q(S).
$$

The residual $r$ is the probability that at least one of two independent
draws $U$ and $V$ lies outside $S$, after augmenting each term space
with a failure outcome carrying any missing total mass:
$\mathbb{P}(U\notin S\ \lor\ V\notin S)=r$. Independence specifies the
product measure underlying this diagnostic; it is not an assumption that
neural representations are statistically independent. An empty intersection
returns zero divergence and residual one, which fails any permitted run
bound below one.

Write $r_p=1-p(S_p)$ and $r_q=1-q(S_q)$ for the individual residuals.
For a paired residual budget $\varepsilon$, allocating each individual
frontier $1-\sqrt{1-\varepsilon}$ gives
$1-(1-r_p)(1-r_q)\leq\varepsilon$ when the individual residuals meet their
allocations. Different listed supports can lose additional mass on
intersection, so the implementation checks $1-p(S)q(S)$ afterward. For two
exact frontiers, it instead compares their union support, assigning zero
probability to terms absent from one side.

This bounds omitted probability mass, not divergence error. In particular,
a small omitted mass does not bound an omitted KL contribution when the
candidate probability can approach zero. A truncated objective is an
objective conditional on the enumerated common support, and changing the
budget changes that estimator. Chart pruning, unary unrolling, and paired
term truncation thus require separate diagnostics.

To declare a parser, its semiring and its pruner, see [Configure a chart
parser](../how-to/models/configure-a-chart-parser.md).

## References

\bibliography
