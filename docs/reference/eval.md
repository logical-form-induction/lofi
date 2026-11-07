# Eval

A [bracketing evaluator](configuration/evaluation.md#evaluators) assesses
constituent structure; a logical-form evaluator compares meaning expressions;
an [inference evaluator](#inference-evaluation) assesses relations between
paired sentences. They require different reference information and answer
different questions about a trained model.

An occurrence is one dataset observation, even when its text repeats.
[Classification and calibration](#classification-and-calibration-definitions)
specifies which occurrences enter each denominator, including how abstentions
are counted. [Reading the notation](../explanation/reading-the-notation.md)
explains the indicators and sums.

This page renders [`lofi.eval`](#lofi.eval): result values, metrics and alignment policies.

Held-out evaluator plugins receive one frozen evaluation context: the lowered
grammar and either (i) derivations with their declared [interpretation](../explanation/logical-forms-as-types.md#category-to-type-interpretation) and
[logical form](../explanation/logical-forms-as-types.md#the-correspondence) texts or (ii) labeled inference pair occurrences. A logical form
evaluator names a complete term [readout](../how-to/models/configure-readouts.md) and one member of the logical form
match family. Gold forms come from the supplied reference texts.

## Inference evaluation

The `inference` evaluator reads the [grounding](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient)'s published label
distribution. Its response scale comes from [`InferenceLabel`](lf.md#lofi.lf.InferenceLabel), the same owner
used by the data and grounding. Pair identifiers distinguish each occurrence,
including repetitions of the same premise and hypothesis.

On the native [three-way scale](../explanation/inference-as-evidence.md#a-label-is-a-pair-of-proof-questions), one evaluator emits accuracy, macro-F1,
precision, recall, and F1 for each class, plus the explicit [abstention](../explanation/inference-as-evidence.md#deciding-a-question) rate. It
emits no micro aggregate. An abstention remains in the denominator when
`coverage: zero` and is excluded from the classification rows when
`coverage: ignore`; the abstention row always reports it. Every defined row
carries the evaluator's declared [bootstrap interval](stats.md#bootstrap-intervals) over inference pair
occurrences. The declaration must state whether labels are reduced:
`reduction: null` retains three classes, while a total map with exactly two
images declares a two-way analysis. There is no implicit two-way collapse.

The same evaluator reports decision coverage, expected [calibration](#classification-and-calibration-definitions) error, and
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
model's published form beams. It orders joint candidates by their summed [log
weights](../explanation/charts-and-approximation.md#what-a-semiring-decides) and asks the [interpretation](../explanation/logical-forms-as-types.md#category-to-type-interpretation)'s shared [proof service](../explanation/inference-as-evidence.md#deciding-a-question) both whether the
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

The denominator determines how an abstaining system is described. In an
illustrative set of ten inputs, suppose eight receive correct decisions and
two receive abstentions. Accuracy is $8/10=0.8$ under `coverage: zero` and
$8/8=1$ under `coverage: ignore`; decision coverage remains $8/10=0.8$.
Decision coverage records the share of inputs that received an answer.

Precision asks how often predictions of a class are correct; recall asks how
many observations of that class were found. Calibration instead compares
forecast probabilities with observed frequencies: among forecasts of 0.9,
calibration asks whether about nine in ten are correct in the evaluation
population. Accuracy and calibration thus assess different properties of
the predictions.

Let $n$ be the number of scored occurrences, $y_i$ the gold label,
$\widehat y_i$ the predicted label, and $p_i(k)$ the decided forecast [PMF](../explanation/reading-the-notation.md#sums-products-and-probabilities)
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
omitted. Under `coverage: zero`, an [abstention](../explanation/inference-as-evidence.md#deciding-a-question) counts as an incorrect
prediction and as a missed gold label. Under `ignore`, classification
conditions on non-abstention. Decision coverage is the fraction of all
occurrences with a decided forecast, and must accompany conditional scores.

For $c$ classes, the multiclass Brier score is

$$
\operatorname{brier}=\frac1n\sum_{i=1}^n\sum_{k=1}^{c}
\left(p_i(k)-\mathbf{1}\{y_i=k\}\right)^2.
$$

This convention sums over classes \citep{brier1950}.
The category-specific version keeps one summand; in a binary task, the full
score is twice either category-specific score.

For top label calibration, define $u_i=\max_k p_i(k)$ and
$v_i=\mathbf{1}\{\arg\max_k p_i(k)=y_i\}$. Sort by $u_i$ and partition into
$b$ equal-count bins $B_1,\ldots,B_b$, with $1\leq b\leq n$,
whose sizes differ by at most one.
LoFI reports

$$
\operatorname{ece}=\sum_{j=1}^{b}\frac{|B_j|}{n}
\left|\frac1{|B_j|}\sum_{i\in B_j}u_i
-\frac1{|B_j|}\sum_{i\in B_j}v_i\right|.
$$

An empty population returns no calibration value. With fewer observations
than requested bins, evaluators report an undefined metric;
the standalone [`expected_calibration_error`](#lofi.eval.expected_calibration_error)
function raises an error for a nonempty population.
For a category-specific curve, use $u_i=p_i(k)$ and
$v_i=\mathbf{1}\{y_i=k\}$ instead. ECE depends on the binning convention;
LoFI uses equal-count bins, whereas equal-width bins are also common
\citep{guo-etal2017}. The reported value describes calibration on the observed
sample under that binning convention.

## Proof reproduction definitions

Proof reproduction asks whether a decoded pair recovers the reference pair's
logical relation under the declared proof procedure. Several different formula
pairs can have that same relation, so the score assesses agreement at the
level of logical relations. The top
candidate and the first $K$ candidates answer distinct retrieval questions,
and undecided proof calls limit which comparisons have a defined score.

Let $c_i^\star$ be the retained reference proof cell and $c_{ij}$ the cell
of the $j$th joint candidate, ordered by the sum of its two log weights.
On decided, nondegenerate comparisons, EPR scores
$\mathbf{1}\{c_{i1}=c_i^\star\}$ and EPR@K scores
$\mathbf{1}\{\exists j\in\{1,\ldots,K\}:c_{ij}=c_i^\star\}$.
A match establishes the latter despite other undecided candidates; without
a match, an undecided candidate can leave it undefined. A zero score requires
a settled nondegenerate reference, at least one decided nondegenerate
candidate, and no undecided candidate; an entirely degenerate beam is
undefined. The aggregate averages
defined scores and reports their coverage. The proof budget determines which
comparisons between decoded [logical forms](../explanation/logical-forms-as-types.md#the-correspondence) can contribute to these reproduction measurements.

For a decided highest-ranked candidate with crisp entailment indicator
$e_i\in\{0,1\}$, let $r_i$ be the native published entailment mass before
label reduction or renormalization. The relaxation gap is $|r_i-e_i|$,
averaged over eligible occurrences.
Both-proved cases and abstentions are reported in separate rows, outside the
three-way label scale.

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
