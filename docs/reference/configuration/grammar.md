# Grammar declarations

For *John sleeps*, a grammar declares [lexical entries](../lexicon.md#lofi.lexicon.Entry) for the two words, a rule
that combines their categories, and a [goal](../../explanation/derivations-as-proofs.md#goals) accepting the result as a sentence.
The [grammar tutorial](../../tutorials/02-declare-a-grammar.md) constructs such
an analysis; [Grammars as deductive systems](../../explanation/derivations-as-proofs.md)
defines its components.

A dataset's grammar can be the source that generated observations. The
[learner declaration](../../explanation/synthetic-designs.md#two-declarations-two-owners)
can make different grammatical assumptions. The [logical-form language](language.md)
separately declares expressions available for meanings.

This page renders [`lofi.spec.grammar`](#lofi.spec.grammar), the classes that validate a dataset's `grammar` section: its formalism, signature, rules, lexicon, [normal form](../../explanation/derivations-as-proofs.md#normal-forms), [matching](../../explanation/logical-forms-as-types.md#substitution-equality-and-matching), goal and interpretations.

::: lofi.spec.grammar
