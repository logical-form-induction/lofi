# Tutorials

The tutorials ask how much a learner can recover about language from examples.
We generate sentences from a known grammar, hide some of that grammar's
information from the learner, and compare what it learns with what generated
the sentences. A [**corpus**](../how-to/data/generate-a-synthetic-corpus.md#draw-the-corpus) is the collection of sentences. A [**grammar**](../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms)
specifies how expressions combine; a [**logical form**](../explanation/logical-forms-as-types.md#semantic-typing) represents an expression's
meaning explicitly enough to reason with it.

You do not need to know categorial grammar or lambda calculus before starting.
The chapters introduce their notation through sentences such as *john sleeps*
and *the dog chases mary*. The [notation guide](../explanation/reading-the-notation.md)
provides another way to read the mathematical symbols. For running the examples,
start with chapter 1's installation and command instructions. To understand the
grammar first, read chapter 2, then return to installation before executing code.

Chapters 2 and 4 declare known structures; chapters 3 and 5 ask a learner to
recover them. Each stage has a settings file under [`docs/examples/tutorials/`](https://github.com/logical-form-induction/lofi/tree/main/docs/examples/tutorials).
Its [`defaults`](../how-to/setup/configuration.md#select-fragments-with-defaults) list reuses the previous stage's settings, and its remaining
entries state what changes. We call these settings files [**run documents**](../how-to/setup/configuration.md#the-words-this-site-uses) and
one run at a particular configuration and seed a [**point**](../explanation/documents-are-the-dsl.md#point-identity).

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

Chapters 1 to 6 use the neural components installed by the source environment
in chapter 1, including the [`nn` extra](../how-to/setup/installing.md#add-an-extra) and PyTorch. Their small datasets and
models are intended for local exercises; elapsed time depends on the machine.
Run shell commands from the repository's root directory. Within each chapter,
run the Python blocks in order in the same Python session unless stated
otherwise: later blocks reuse the variables created earlier. Chapter 7 needs
only the base install and can be read at any point. The
[how-to guides](../how-to/index.md) cover each task in isolation once the path
is familiar.
