# Grammars as deductive systems

This page explains how a grammar declaration becomes one deductive system: what
a signature, a rule inventory, a matching relation, a normal form and a goal
each contribute, what the compiler derives from them rather than reading from
the declaration, and why every formalism LoFI compares passes through the same
compiled value.

## One compiled value per grammar

A grammar declaration names its parts and nothing about how to parse with
them. Its complete form, [`GrammarDecl`](../reference/configuration/grammar.md)
(`formalism: declared`), holds a signature, a rule set, a matching relation, a
normal form, a goal, a lexicon, an approximation budget and, optionally, the
hypotheses of a capacity theorem. A concise declaration such as `formalism:
ab`, `cfg`, `tlg` or `undirected-ccg` is a plugin that expands into a
`GrammarDecl`; after that expansion every formalism is the same kind of value.

Compilation produces a [`Formalism`](../reference/formalism.md), and
`Formalism.compile` returns the run-scoped `Grammar` that every downstream
consumer reads. The pieces correspond to the parts of a deductive system:

| Part | Role in the deductive system |
| --- | --- |
| Signature | The formulas. Sorts and operators over which categories are terms in one store. |
| Lexicon | The axioms. Each entry pairs a form with a category and, optionally, a meaning. |
| Rule set | The inference rules. Each rule relates premise categories to a conclusion. |
| Matching | When two categories count as the same premise. |
| Normal form | Which derivation steps are admitted, so that equivalent proofs are not counted twice. |
| Goal | Which conclusions over which input regions count as a proof of the sentence. |
| Budget | How far an approximate consumer may go: term depth, beam, mass target, unary rounds, beta-reduction fuel. |

[Logical forms as types](logical-forms-as-types.md) gives the reading under
which a category is a type and a derivation a proof. This page is about the
machinery that makes that reading the same for every formalism.

## Rules are relations over terms

A rule is declared by its family (`application`, `composition`,
`substitution`, `type-raising`, `coordination`, `pattern` and others, each
registered in the `lofi.rule` entry-point group) and compiled to a value that
implements the [`Rule`](../reference/judge.md) protocol. The deduction engine
in `lofi.deduce` depends only on that protocol. It imports no formalism and no
grammar pack, and an import contract enforces this (see [The layer
order](layering.md)). One engine therefore serves CFG productions, AB
application, CCG combinators and type-logical patterns alike, and a difference
between two grammars' results cannot come from two different parsers.

The matching relation decides when a category fills a premise slot. Identity
matching compares terms structurally; feature erasure, subsumption, alpha
equivalence, product and modal matchings are coarser relations, each named and
registered in `lofi.matching`. Because the matching is a declared value rather
than a property of term equality, a grammar that ignores morphosyntactic
features and one that respects them differ in one field and are otherwise
compiled identically. The compiler refuses a goal whose target set was built
under a different matching from the rule set's, compared by identity.

## What the compiler derives

Three properties of a grammar are computed from its rules, not declared:

- **Ranks and unary closure.** `deduction_requirements` collects the arity of
  every rule. A rule set containing a unary rule requires unary-closure rounds
  in each chart cell, and a declaration whose budget allows zero such rounds is
  refused.
- **The reachable category set.** A closure report classifies the categories
  the rules can reach as `finite`, `infinite` or `unknown`, and lists the
  reached and the escaping categories. Every goal
  category must be reachable, or the grammar is refused.
- **A capacity report.** When a declaration supplies CCG-family evidence, the
  judge reports the weak generative capacity the theorem gives for that family
  together with its citation: bounded-degree pure CCG is a proper superclass of
  the context-free languages (Kuhlmann, Koller, and Satta 2010), and
  prefix-closed VW-CCG with target restrictions on application is
  TAG-equivalent (Kuhlmann, Koller, and Satta 2015). When the evidence is
  absent or a hypothesis fails (an unbounded composition degree, an inventory
  that is not exclusively generalized composition, a lexical-anchor
  restriction), the report is `not_characterized` and names the unmet
  hypothesis. An applicable theorem with no citation is refused. The report
  characterizes the rule family, not one concrete grammar.

The rule arities, unary closure, category-set class, fan-out and
lexicalization together form the [`DeductionProfile`](../reference/formalism.md).
A consumer that serves only some profiles, such as a neural chart parser built
for binary rules, declares a `ProfileBound`. `Formalism.compile` checks each
supplied bound and raises `ProfileUnsupportedError` naming the consumer, the
arities it serves and the arities the grammar uses. A grammar outside a
consumer's reach is therefore refused at compilation rather than parsed
incorrectly.

## Normal forms

Under composition and type raising, a CCG admits many derivations with the same
semantics, the so-called spurious ambiguity. A normal form is a filter on
derivation steps that keeps one representative of each equivalence class.
LoFI registers `none`, `eisner` (Eisner 1996), `undirected-eisner`,
`hockenmaier-bisk` (Hockenmaier and Bisk 2010) and `all-of`, which conjoins
several. The normal form is part of the rule set and of the grammar's
fingerprint. Whether a normal form preserves every reading depends on the
rule inventory it is paired with; the cited results state the inventories for
which each filter is complete, and the compiler does not check that pairing.

## Goals

A goal says which chart items count as a parse. The `root` goal accepts a
derivation of one of the target categories over the whole input, with optional
log weights per root category. The `cover` goal accepts a tiling of the input
by at most `max_fragments` fragments, with a declared penalty per fragment
after the first or for each fragment. A cover goal therefore scores strings the
grammar cannot derive whole, which a corpus with fragments may need and a
grammaticality objective may not want. The two goals ask different questions of
the same chart.

## Identity

A compiled grammar has a fingerprint: the SHA-256 of a canonical JSON record
of its name, signature digest, every rule's specification, the matching and
normal form, the closure report, the profile, the budget, the capacity report,
the clamps, the targets, the goal settings, the interpretations and the digest
of its checked theory. Two declarations that compile to the same parts have
the same fingerprint however they were spelled; a change to any part, the
budget included, changes it. [Documents are the declaration
language](documents-are-the-dsl.md) describes how that identity enters a
point's identity.

## What this licenses

Because every formalism compiles to one value read by one engine, a comparison
between two grammars varies only what their declarations vary. It does not
show that a difference in results reflects the grammars' generative capacity:
the budget, the normal form and the goal are part of each grammar and can
account for a difference on their own. A comparison that means to isolate the
rule inventory holds those three fixed. [Swappable grammatical
assumptions](swappable-assumptions.md) states this condition in general.

To declare each part, see [Declare a signature](../how-to/grammars/declare-a-signature.md),
[Choose combinatory rules](../how-to/grammars/choose-combinatory-rules.md) and
[Choose a normal form, matching and goal](../how-to/grammars/choose-normal-form-matching-goal.md).

## References

- Jason Eisner. 1996. Efficient normal-form parsing for Combinatory Categorial
  Grammar. In *Proceedings of the 34th Annual Meeting of the Association for
  Computational Linguistics*, pages 79–86, Santa Cruz, California.
  [P96-1011](https://aclanthology.org/P96-1011/).
- Julia Hockenmaier and Yonatan Bisk. 2010. Normal-form parsing for
  Combinatory Categorial Grammars with generalized composition and
  type-raising. In *Proceedings of the 23rd International Conference on
  Computational Linguistics (COLING 2010)*, pages 465–473, Beijing.
  [C10-1053](https://aclanthology.org/C10-1053/).
- Marco Kuhlmann, Alexander Koller, and Giorgio Satta. 2010. The importance of
  rule restrictions in CCG. In *Proceedings of the 48th Annual Meeting of the
  Association for Computational Linguistics*, pages 534–543, Uppsala.
  [P10-1055](https://aclanthology.org/P10-1055/).
- Marco Kuhlmann, Alexander Koller, and Giorgio Satta. 2015. Lexicalization
  and generative power in CCG. *Computational Linguistics*, 41(2):187–219.
  [J15-2002](https://aclanthology.org/J15-2002/).
