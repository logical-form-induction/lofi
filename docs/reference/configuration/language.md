# Logical form language declarations

A language can declare an [individual type](../../explanation/logical-forms-as-types.md#semantic-typing) $e$, a truth-value type $t$, and a
predicate `sleep` of type $e\to t$. This permits meaning expressions such as
$\operatorname{sleep}(\operatorname{john})$ once the individual constant is
declared. A [model](../lf.md#finite-models) assigns the predicate its [denotation](../../explanation/inference-as-evidence.md#syntax-consequence-and-models)
and determines which individuals satisfy it.

[Semantic typing](../../explanation/logical-forms-as-types.md#semantic-typing)
explains the type restrictions; [binding](../../explanation/logical-forms-as-types.md#signatures-and-terms)
determines variable scope. [Reading the notation](../../explanation/reading-the-notation.md)
introduces function arrows and lambda expressions.

This page renders [`lofi.spec.language`](#lofi.spec.language), the classes that validate a [logical form](../../explanation/logical-forms-as-types.md#the-correspondence) language: its types, constructors, constants, binders and [logical roles](../lf.md#lofi.lf.LogicalRole).

::: lofi.spec.language
