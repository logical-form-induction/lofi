# Lexicon

A [lexical entry](#lofi.lexicon.Entry) pairs a form with a category and optionally
a meaning. The form *John* can have category `NP` and meaning `john`. Another
entry for the same form supplies an alternative analysis; the parser can
consider both.

[`LexicalSource`](#lofi.lexicon.LexicalSource) supplies these entries. A
[corpus reader](data.md#corpora) instead supplies the observations to be
analyzed. The [grammar tutorial](../tutorials/02-declare-a-grammar.md) shows
how a lexicon and rules together license a sentence.

This page renders [`lofi.lexicon`](#lofi.lexicon): lexical entries and the sources that supply them.

::: lofi.lexicon
