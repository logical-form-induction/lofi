# Tutorials

The tutorials form one learning path. It starts from an installed package and
ends with an experiment that asks whether the grammar that generated a corpus
fits it better than its rivals. Chapters 2 to 5 build one run document in
stages, and each stage is a short document under `docs/examples/tutorials/`
that names the previous one in its `defaults` list and changes one section of
it, so the effect of each change can be traced to the lines that made it.

1. [Your first run](01-first-run.md). Install LoFI, compose a complete run
   document, train one point on a small synthetic corpus, and read the records
   it writes.
2. [Declaring a grammar](02-declare-a-grammar.md). Write a categorial grammar
   by its signature, rules and lexicon, draw a corpus from it, and read the
   derivations the corpus holds.
3. [Inducing a lexicon](03-induce-a-lexicon.md). Hold out development and test
   sentences, train a chart parser on the strings alone, and score how well it
   recovers the brackets and the lexical categories.
4. [Declaring logical forms](04-declare-logical-forms.md). Add a logical form
   language and an interpretation, so that each drawn derivation carries a
   typed logical form.
5. [Inducing logical forms](05-induce-logical-forms.md). Learn lexical logical
   forms from labeled inference pairs under a prover-grounded objective, and
   score their recovery.
6. [Comparing grammars](06-compare-grammars.md). Cross a source grammar with
   candidate learner grammars in one experiment, price it, run it, and reduce
   its rows to a verdict.
7. [Under the hood: charts and semirings](07-charts-and-semirings.md). Build
   the span sweep the chart parsers of chapters 3 to 6 run, and evaluate it
   under a counting and a log inside semiring.

Chapters 1 to 6 need the `nn` extra, which installs PyTorch; every run in them
is sized to finish in under a few minutes on a laptop CPU. Chapter 7 needs only
the base install and can be read at any point. The
[how-to guides](../how-to/index.md) cover each task in isolation once the path
is familiar.
