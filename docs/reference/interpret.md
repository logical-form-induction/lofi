# Interpret

An [interpretation](../explanation/logical-forms-as-types.md#category-to-type-interpretation)
assigns a reading to a derivation. For *John sleeps*, [lexical meanings](../explanation/logical-forms-as-types.md#category-to-type-interpretation) and the
action for backward application can yield
$\operatorname{sleep}(\operatorname{john})$.
[`MeaningReading`](#lofi.interpret.MeaningReading) supplies this kind of reading.

A [derivation fold](#lofi.interpret.read) computes from the leaves toward the root: each rule step
uses the readings of its premises. The
[interpretation guide](../how-to/grammars/declare-an-interpretation.md)
connects those semantic actions to a declaration. Whether the resulting formula
is true in a model is a separate [inference question](../explanation/inference-as-evidence.md#syntax-consequence-and-models).

This page renders [`lofi.interpret`](#lofi.interpret): interpretations, derivation folds and semantic readings.

::: lofi.interpret
