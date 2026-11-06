# Logical forms

A [logical-form language](configuration/language.md) supplies expressions for
candidate meanings. A [finite model](#finite-models) assigns denotations to
their symbols. A [prover](#provers) instead attempts a consequence question in
its supported logical fragment. Checking that John sleeps in one model differs
from proving that a hypothesis follows in every compatible model.

[Inference as evidence](../explanation/inference-as-evidence.md#deciding-a-question)
explains the resulting verdicts, including abstentions. The APIs here specify
the language, translation, model evaluation, and proof calls used to obtain them.

This page renders [`lofi.lf`](#lofi.lf), [`lofi.lf.models`](#lofi.lf.models) and [`lofi.lf.provers`](#lofi.lf.provers): [logical form](../explanation/logical-forms-as-types.md#the-correspondence)
signatures, binder presentation, [logic profiles](../explanation/inference-as-evidence.md#syntax-consequence-and-models), emission to the [prover
dialects](../how-to/training/reward-with-a-theorem-prover.md), finite models, and the entailment oracle with its call ledger.

::: lofi.lf

## Finite models

::: lofi.lf.models

## Provers

::: lofi.lf.provers
