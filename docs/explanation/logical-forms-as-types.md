# Logical forms as types

This page explains the reading under which LoFI treats a syntactic category as
a type, a derivation as a proof, and a logical form as that proof's term. It
distinguishes what the syntactic and semantic sides share, one term algebra,
from what they keep apart: their signatures, their stores and their imports.

## The correspondence

In the type-logical tradition a category is a formula and a derivation is a
proof of it from the categories of the words (Lambek 1958). Under the
formulae-as-types correspondence, a proof determines a term, and that term is
the derivation's meaning (Howard 1980). LoFI implements both sides over one
[term algebra](../reference/terms.md): categories are terms over a syntactic
signature, logical forms are terms over a semantic signature, and a derivation
built from lexical entries yields a category and, through an interpretation,
a logical form.

Sharing the algebra keeps the two sides' operations consistent. Equality,
substitution and proof-term construction are implemented once, so a property
established for terms holds for categories and logical forms alike. Sharing it
does not force the two signatures to agree: a semantic signature may draw
distinctions a syntactic one does not, and the reverse.

## Terms

A term is a value over a declared signature. When no operator of the
signature binds variables, the signature is a first-order algebra and the
binding machinery stays inert, so the same store implementation serves
categories and lambda terms, with one store instance for each signature.

A term holds no score, tensor or mutable child list. Scores belong to
deduction items and edges, and a decoder's state belongs in a `Frontier`
beside the term. Term equality therefore depends only on term structure, and a
term can serve as a key wherever a category or a meaning must be compared.

## Equality and matching

A term has one equality. A rule that ignores morphosyntactic features must
still read `NP[conj]` as an `NP`, so it reads its premises under a relation
coarser than identity. LoFI represents that relation as a named
[`Matching`](../reference/terms.md) with a key function, and each rule set
declares the matching it uses. A comparison that erases features therefore
stays distinct from structural equality, and two grammars that differ only in
their matching differ in a declared value (see [Grammars as deductive
systems](derivations-as-proofs.md)).

## A rule is a relation

A rule relates premises to a conclusion, and the relation need not be a
function. Type raising without a target set has infinitely many conclusions,
and a coercion may have several. A [rule](../reference/judge.md) therefore
declares which directions it computes: it may enumerate conclusions from
premises, validate a proposed step, or residuate a missing premise from a
conclusion and the other premises. `forward_is_complete` says whether forward
enumeration yields every conclusion. A consumer inspects it and `directions`
before choosing an operation, and a consumer that needs complete enumeration
does not assume it.

## Consequences

The induction packages are import-independent of the logical-form packages.
Terms, rules, lexical entries, spans and semirings do not import logical-form
typing, and an import contract enforces this (see [The layer
order](layering.md)). Category induction can therefore run, and be tested,
without any logical-form language declared.

The syntactic and semantic signatures stay separate and may encode different
distinctions. They share the term implementation, its equality and its
proof-term construction, but not a `TermStore` instance, so a category can
never be mistaken for a logical form that happens to have the same shape.

## References

- William A. Howard. 1980. The formulae-as-types notion of construction. In
  J. P. Seldin and J. R. Hindley, editors, *To H. B. Curry: Essays on
  Combinatory Logic, Lambda Calculus and Formalism*, pages 479–490. Academic
  Press.
- Joachim Lambek. 1958. The mathematics of sentence structure. *The American
  Mathematical Monthly*, 65(3):154–170.
  [doi:10.1080/00029890.1958.11989160](https://doi.org/10.1080/00029890.1958.11989160).
