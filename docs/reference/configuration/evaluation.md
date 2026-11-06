# Evaluation declarations

A bracketing evaluator needs [reference constituent structure](../../explanation/derivations-as-proofs.md#from-words-to-a-derivation); an inference
evaluator needs reference relations between sentence pairs. The
[evaluator declarations](#evaluators) select that target and its conventions.
[Alignment and coverage](../eval.md#classification-and-calibration-definitions)
determine what enters the comparison and how unanswered observations are counted.

A [baseline](#baselines) supplies a comparison system. A [sink](#sinks) receives
computed measurements for storage or display. [Evaluation](../eval.md)
defines the reported [metrics](../eval.md#classification-and-calibration-definitions) and their denominators.

This page renders the classes that validate a run document's `evaluators` and `sinks` sections and a baseline fit document's baseline system.

## Evaluators

::: lofi.spec.evaluators.alignment

::: lofi.spec.evaluators.bracketing

::: lofi.spec.evaluators.category_side

::: lofi.spec.evaluators.coherence_diagnostics

::: lofi.spec.evaluators.coherence_square

::: lofi.spec.evaluators.constructor_diagnostics

::: lofi.spec.evaluators.conventions

::: lofi.spec.evaluators.decode

::: lofi.spec.evaluators.equivalence

::: lofi.spec.evaluators.expert

::: lofi.spec.evaluators.grammaticality

::: lofi.spec.evaluators.held_out_likelihood

::: lofi.spec.evaluators.interannotator

::: lofi.spec.evaluators.inference

::: lofi.spec.evaluators.lexical_diagnostics

::: lofi.spec.evaluators.lexical_types

::: lofi.spec.evaluators.logical_form

::: lofi.spec.evaluators.reachability

::: lofi.spec.evaluators.structural_diagnostics

## Baselines

A [baseline](../../how-to/evaluation/fit-a-baseline.md) is a comparison system fitted under its declared procedure. Its data partitions and reported [metrics](../eval.md#classification-and-calibration-definitions) must be compatible with the comparison being made.

::: lofi.spec.baselines

## Sinks

::: lofi.spec.sinks
