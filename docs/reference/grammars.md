# Grammars

An [applicative grammar](#applicative-grammar) combines categories by
application. A [context-free grammar](#context-free-grammar) supplies production
rules; [type-logical grammar](#type-logical-grammar) expresses derivability in
a logic; [CCG](#ccg) supplies its declared [combinatory rules](../how-to/grammars/choose-combinatory-rules.md).

The families compile to a shared [`Grammar`](formalism.md#lofi.formalism.Grammar)
interface, so the same consumer can use different [rule inventories](../explanation/derivations-as-proofs.md#rules-are-relations-over-terms) when its
capabilities permit them. [Grammars as deductive systems](../explanation/derivations-as-proofs.md)
explains the formal distinctions. [Grammar declarations](configuration/grammar.md)
records the configuration fields.

This page renders [`lofi.grammars`](#lofi.grammars) and its applicative, context-free, type-logical and CCG declarations and compilers.

::: lofi.grammars

## Applicative grammar

::: lofi.grammars.ab

## Context-free grammar

::: lofi.grammars.cfg

## Type-logical grammar

::: lofi.grammars.tlg

## CCG

::: lofi.grammars.ccg
