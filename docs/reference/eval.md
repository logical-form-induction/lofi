# Eval

This page renders `lofi.eval`: result values, metrics and alignment policies.

Held-out evaluator plugins receive one frozen evaluation context: the lowered
grammar and either (i) derivations with their declared interpretation and
logical-form texts or (ii) labeled inference-pair occurrences. A logical-form
evaluator names a complete-term readout and one member of the logical-form
match family; it never infers gold forms from a private model layout.

## Inference evaluation

The `inference` evaluator reads the grounding's published label
distribution. Its response scale comes from `InferenceLabel`, the same owner
used by the data and grounding, and its pair identifiers name occurrences. It
thus does not collapse repeated premise-hypothesis pairs.

On the native three-way scale, one evaluator emits accuracy, macro-F1,
precision, recall, and F1 for each class, plus the explicit abstention rate. It
emits no micro aggregate. An abstention remains in the denominator when
`coverage: zero` and is excluded from the classification rows when
`coverage: ignore`; the abstention row always reports it. Every defined row
carries the evaluator's declared bootstrap interval over inference-pair
occurrences. The declaration must state whether labels are reduced:
`reduction: null` retains three classes, while a total map with exactly two
images declares a two-way analysis. There is no implicit two-way collapse.

The same evaluator reports decision coverage, expected calibration error, and
the multiclass Brier score. Calibration conditions on the represented mass of
decided forecasts and applies the declared label reduction before scoring.
The `bins` field fixes the calibration bin count. An ECE row is undefined when
fewer decided pairs than bins are available.

The majority-class and hypothesis-only controls fit on `partitions.fit_on`
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
