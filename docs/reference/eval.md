# Eval

This page renders `lofi.eval`: result values, metrics and alignment policies.

Held-out evaluator plugins receive one frozen evaluation context: the lowered
grammar and either (i) derivations with their declared interpretation and
logical form texts or (ii) labeled inference pair occurrences. A logical form
evaluator names a complete term readout and one member of the logical form
match family; it never infers gold forms from a private model layout.

## Inference evaluation

The `inference` evaluator reads the grounding's published label
distribution. Its response scale comes from `InferenceLabel`, the same owner
used by the data and grounding, and its pair identifiers name occurrences. It
thus does not collapse repeated pairs of premises and hypotheses.

On the native three-way scale, one evaluator emits accuracy, macro-F1,
precision, recall, and F1 for each class, plus the explicit abstention rate. It
emits no micro aggregate. An abstention remains in the denominator when
`coverage: zero` and is excluded from the classification rows when
`coverage: ignore`; the abstention row always reports it. Every defined row
carries the evaluator's declared bootstrap interval over inference pair
occurrences. The declaration must state whether labels are reduced:
`reduction: null` retains three classes, while a total map with exactly two
images declares a two-way analysis. There is no implicit two-way collapse.

The same evaluator reports decision coverage, expected calibration error, and
the multiclass Brier score. Calibration conditions on the represented mass of
decided forecasts and applies the declared label reduction before scoring.
The `bins` field fixes the calibration bin count. An ECE row is undefined when
fewer decided pairs than bins are available.

The majority class and hypothesis-only controls fit on `partitions.fit_on`
and remain frozen while scoring `partitions.score_on`. The hypothesis
featurizer receives only hypothesis tokens. Both names must select retained,
disjoint dataset partitions; their occurrence identifiers cannot overlap.

Setting `proof` to a budget and joint beam depth enables proof reproduction.
The evaluator reads the retained true premise and hypothesis forms and the
model's published form beams. It orders joint candidates by their summed log
weights and asks the interpretation's shared proof service both whether the
premise entails the hypothesis and whether it entails its negation. These
questions distinguish entailment, contradiction, neutral, and both-proved
cells. Both-proved and abstained decisions have separate report rows and do
not supply a three-way label. EPR compares the highest-ranked candidate's cell
to the retained reference cell; EPR@K permits any matching candidate among the
declared K. An unresolved candidate without a match leaves EPR@K undefined.
The relaxation gap is the absolute difference between published entailment
probability and the highest-ranked candidate's crisp entailment decision,
over decided nondegenerate candidates. Proof rows also report calls, memo
hits, questions, CPU and wall time, and abstention causes.

## Classification and calibration definitions

Let $n$ be the number of scored occurrences, $y_i$ the gold label,
$\widehat y_i$ the predicted label, and $p_i(k)$ the decided forecast PMF
after reduction and renormalization. Calibration uses only decided
occurrences: in its formulas below, $n$ is the number of those occurrences,
which can differ from the classification denominator. Accuracy is
$n^{-1}\sum_i\mathbf{1}\{\widehat y_i=y_i\}$. For class $k$, let
$t_k$, $f_k$, and $m_k$ count true positives, false positives, and false
negatives. Then

$$
\operatorname{precision}_k=\frac{t_k}{t_k+f_k},\quad
\operatorname{recall}_k=\frac{t_k}{t_k+m_k},\quad
\operatorname{f1}_k=\frac{2t_k}{2t_k+f_k+m_k}.
$$

A zero denominator makes the corresponding metric undefined. Macro-F1 is the
arithmetic mean of defined class F1 values; classes absent on both sides are
omitted. Under `coverage: zero`, an abstention counts as an incorrect
prediction and as a missed gold label. Under `ignore`, classification
conditions on non-abstention. Decision coverage is the fraction of all
occurrences with a decided forecast, and must accompany conditional scores.

For $c$ classes, the multiclass Brier score is

$$
\operatorname{brier}=\frac1n\sum_{i=1}^n\sum_{k=1}^{c}
\left(p_i(k)-\mathbf{1}\{y_i=k\}\right)^2.
$$

This is the summed, not class-averaged, convention \citep{brier1950}.
The category-specific version keeps one summand; in a binary task, the full
score is twice either category-specific score.

For top label calibration, define $u_i=\max_k p_i(k)$ and
$v_i=\mathbf{1}\{\arg\max_k p_i(k)=y_i\}$. Sort by $u_i$ and partition into
$b$ equal-count bins $B_1,\ldots,B_b$, whose sizes differ by at most one.
LoFI reports

$$
\operatorname{ece}=\sum_{j=1}^{b}\frac{|B_j|}{n}
\left|\frac1{|B_j|}\sum_{i\in B_j}u_i
-\frac1{|B_j|}\sum_{i\in B_j}v_i\right|.
$$

For a category-specific curve, use $u_i=p_i(k)$ and
$v_i=\mathbf{1}\{y_i=k\}$ instead. ECE depends on the binning convention;
LoFI uses equal-count bins, whereas equal-width bins are also common
\citep{guo-etal2017}. It is not a population calibration guarantee.

## Proof reproduction definitions

Let $c_i^\star$ be the retained reference proof cell and $c_{ij}$ the cell
of the $j$th joint candidate, ordered by the sum of its two log weights.
On decided, nondegenerate comparisons, EPR scores
$\mathbf{1}\{c_{i1}=c_i^\star\}$ and EPR@K scores
$\mathbf{1}\{\exists j\leq k:c_{ij}=c_i^\star\}$.
A match establishes the latter despite other undecided candidates; without
a match, an undecided candidate can leave it undefined. A zero score requires
a settled nondegenerate reference, at least one decided nondegenerate
candidate, and no undecided candidate; an entirely degenerate beam is
undefined. The aggregate averages
defined scores and reports their coverage. These are budget-dependent
reproduction measurements, not proofs that a decoded logical form is correct.

For a decided highest-ranked candidate with crisp entailment indicator
$e_i\in\{0,1\}$, let $r_i$ be the native published entailment mass before
label reduction or renormalization. The relaxation gap is $|r_i-e_i|$,
averaged over eligible occurrences.
Both-proved cases and abstentions have separate rows rather than being
silently assigned one of the three labels.

```yaml
evaluators:
  inference:
    kind: inference
    interval: {replicates: 1000, level: 0.95}
    reduction: null
    partitions: {fit_on: dev, score_on: test}
    bins: 10
    proof:
      depth: 5
      budget: {calls_per_step: 1000, wall_per_step: 30, cpu_seconds: 300}
    conventions:
      punctuation: keep
      unary_chains: keep
      trivial_spans: keep
      minimum_length: 0
      aggregation: corpus
      coverage: zero
      self_arcs: allow
      gold_derivation: canonical
```

::: lofi.eval

## Corpus-dependent evaluation

::: lofi.eval.corpora

## References

\bibliography
