# Type checking

A predicate of [semantic type](../explanation/logical-forms-as-types.md#semantic-typing)
$e\to t$ takes an individual of type $e$ and produces a truth value of type
$t$. Applying it to an expression of type $t$ fails that requirement. Type
checking establishes these compatibility conditions for an expression under
the declared language.

Constraints record the requirements introduced by subexpressions. [Unification](#lofi.typecheck.Unifier)
solves unknown types that must agree, as when a function's argument type must
match its supplied argument. The [typing rules](../explanation/logical-forms-as-types.md#semantic-typing)
state these conditions; [substitution and equality](../explanation/logical-forms-as-types.md#substitution-equality-and-matching)
explain the normalization used when comparing [logical forms](../explanation/logical-forms-as-types.md#the-correspondence).

This page renders [`lofi.typecheck`](#lofi.typecheck): constraint generation, unification,
solving, typing laws, and the [logical form chart](#lofi.typecheck.LogicalFormChart) quotiented by [normal form
meaning](../explanation/logical-forms-as-types.md#substitution-equality-and-matching).

::: lofi.typecheck
