# Terms

A [term](../explanation/logical-forms-as-types.md#signatures-and-terms) is a
structured symbolic expression. For instance, $\mathrm{NP}$ can be an atomic
category, while $\mathrm{S}\backslash\mathrm{NP}$ contains a slash constructor
with two category arguments. A [signature](../how-to/grammars/declare-a-signature.md) specifies which symbols and argument
sorts are allowed; [`TermStore`](#lofi.terms.TermStore) stores the resulting
expressions.

[Syntactic categories, semantic types, and logical forms](../explanation/logical-forms-as-types.md#the-correspondence)
use this machinery under separate signatures. The distinction matters when
reading a constructor's type: an expression of sort `Tm` still needs a semantic
[typing judgment](../explanation/logical-forms-as-types.md#semantic-typing) to establish whether it denotes an individual, a truth value,
or a function.

This page renders [`lofi.terms`](#lofi.terms): sorts, operators, signatures, terms and the term store.

::: lofi.terms
