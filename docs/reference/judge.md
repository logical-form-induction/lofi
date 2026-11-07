# Judge

Under [backward application](../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms),
an $\mathrm{NP}$ followed by an $\mathrm{S}\backslash\mathrm{NP}$ can produce
an $\mathrm{S}$. A [`Rule`](#lofi.judge.Rule) represents such a relation; a [rule
set](../explanation/derivations-as-proofs.md#rules-are-relations-over-terms) supplies the available relations, and a
[normal-form filter](../explanation/derivations-as-proofs.md#normal-forms)
restricts permitted derivation steps.

The name `judge` refers to these grammatical inference judgments. Participant
response models are in [`lofi.judgment`](judgment.md).

This page renders [`lofi.judge`](#lofi.judge): rules, rule sets, [normal forms](../explanation/derivations-as-proofs.md#normal-forms) and reports.

::: lofi.judge
