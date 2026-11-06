# Formalism

A [`Grammar`](#lofi.formalism.Grammar) packages the compiled grammatical choices
a parser consumes. Its [`DeductionProfile`](#lofi.formalism.DeductionProfile)
states requirements such as [rule arities](../explanation/derivations-as-proofs.md#what-the-compiler-derives). A rule with three premises needs a
consumer that supports combining three premises; a binary-only consumer must
refuse that profile.

[Grammar compilation](../explanation/derivations-as-proofs.md#what-the-compiler-derives)
derives these requirements, while a [budget](#lofi.formalism.Budget) bounds the
computation attempted. A mathematically valid grammar may still require
operations beyond a particular consumer's capabilities.

This page renders [`lofi.formalism`](#lofi.formalism): runtime grammar declarations, deduction profiles, budgets and [region protocols](#lofi.formalism.RegionIndex).

::: lofi.formalism
