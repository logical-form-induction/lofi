# Declaring a grammar

This chapter writes a categorial grammar from nothing and draws a corpus from
it, without training anything: it declares the grammar's signature, rules and
[lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) in a [run document](../how-to/setup/configuration.md#the-words-this-site-uses), composes the document, draws the synthetic corpus
the run would train on, reads one drawn derivation node by node, and reads the
reports the draw measures about itself. Chapter 3 trains a parser on exactly
this corpus.

In chapter 1 the slash constructors came from the packaged `directed`
fragment, and the document only replaced the lexicon. Here the document owns
every part of the grammar, so nothing about it is inherited from a fragment.

## The grammar

We can read the grammar through one sentence: *the dog chases mary*. A
[category](../explanation/logical-forms-as-types.md#start-with-a-sentence) describes how an expression combines with its neighbors, rather
than what it means. `N` marks a noun, `NP` a noun phrase, and `S` a sentence.
Let `X` and `Y` stand for arbitrary categories. A slash category names
something that still needs an argument: `X/Y` needs
a `Y` on its right to produce `X`; `X\Y` needs a `Y` on its left.

The sentence is built in three combinations:

1. *the* has category `NP/N`. It combines with *dog*, of category `N`, to
   make the noun phrase *the dog*, of category `NP`.
2. *chases* has category `(S\NP)/NP`. It combines with *mary*, of category
   `NP`, to make *chases mary*, of category `S\NP`.
3. *chases mary* still needs an `NP` on its left. Combining it with *the dog*
   gives the complete sentence, of category `S`.

The brackets in `[[the dog] [chases mary]]` record those combinations. A
[derivation](../explanation/derivations-as-proofs.md#from-words-to-a-derivation) records both the combinations and the rules licensing them.
In a rule such as $\frac{X/Y\quad Y}{X}$, expressions above the line are the
inputs; the expression below is the resulting category. Thus this one rule
describes both the determiner–noun and verb–object steps above.

The full application derivation has this shape, with each word above its
[lexical category](../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms):

$$
\frac{
  \dfrac{
    \overset{\text{the}}{\mathrm{NP}/\mathrm{N}}
    \quad
    \overset{\text{dog}}{\mathrm{N}}
  }{\mathrm{NP}}
  \qquad
  \dfrac{
    \overset{\text{chases}}{(\mathrm{S}\backslash\mathrm{NP})/\mathrm{NP}}
    \quad
    \overset{\text{mary}}{\mathrm{NP}}
  }{\mathrm{S}\backslash\mathrm{NP}}
}{\mathrm{S}}
$$

Read each line as one combination. The two upper lines produce the
constituents *the dog* and *chases mary*; the lower line combines those
constituents. Only the last combination takes its argument from the left.

The document is
[`docs/examples/tutorials/declared-grammar.yaml`](../examples/tutorials/declared-grammar.yaml).
Its model, loss and trainer sections are those of chapter 1 at width 4, with a
[learned token encoder](../how-to/models/choose-a-token-encoder.md#learn-rows-from-scratch) in place of chapter 1's static vectors, since the new
forms have no pretrained vectors. Its grammar, declared under
[`data.datasets.train.dataset`](../reference/configuration/datasets.md#datasets), is this:

```yaml
grammar:
  formalism: declared
  name: tutorial
  signature:
    primitives: [S, NP, N]
    constructors:
      - {name: /, kind: [Cat, Cat, Cat], notation: {style: infix, spelling: /, precedence: 10, associativity: left}}
      - {name: '\', kind: [Cat, Cat, Cat], notation: {style: infix, spelling: '\', precedence: 10, associativity: left}}
    functors:
      - {name: /, constructor: /, argument_side: right, opposite: '\'}
      - {name: '\', constructor: '\', argument_side: left, opposite: /}
  rules:
    name: application
    rules:
      - {rule: application, view: /}
      - {rule: application, view: '\'}
  lexicon:
    kind: inline
    entries:
      - {form: john, category: NP}
      - {form: mary, category: NP}
      - {form: the, category: NP/N}
      - {form: dog, category: N}
      - {form: cat, category: N}
      - {form: sleeps, category: 'S\NP'}
      - {form: sees, category: 'S\NP/NP'}
      - {form: chases, category: 'S\NP/NP'}
  goal: {kind: root, categories: [S]}
  matching: {kind: feature-erasure}
  budget: {unary_rounds: 1}
```

A [signature](../how-to/grammars/declare-a-signature.md#declare-primitives-and-constructors) specifies the basic symbols and how larger categories can be
built. Its primitives are the basic categories; its constructors make complex
categories from smaller ones. `Cat` names the [sort of category objects](../explanation/logical-forms-as-types.md#signatures-and-terms), so
`kind: [Cat, Cat, Cat]` declares a constructor with two category inputs and a
category output. *Infix* means that the symbol appears between its inputs;
*left-associative* tells the reader to group `S\NP/NP` as `(S\NP)/NP`.
The [`functors` entries](../how-to/grammars/declare-a-signature.md#declare-functor-views) add argument direction to these constructors; the
field name here describes function categories in the grammar.

Reading the grammar line by line:

| Line | What it says |
|---|---|
| `primitives` | The atomic categories: sentences, noun phrases and nouns. |
| `constructors` | Two binary constructors over the category sort `Cat`, spelled infix and left-associative, so `S\NP/NP` reads as `(S\NP)/NP`. |
| `functors` | Each constructor builds a function category. `/` takes its argument on the right and `\` on the left, and each names the other as its opposite. |
| [`rules`](../how-to/grammars/choose-combinatory-rules.md#the-rule-families) | Forward and backward application, $\frac{X/Y\quad Y}{X}$ and $\frac{Y\quad X\backslash Y}{X}$, as two views of one rule. |
| `lexicon` | Eight forms, each with one category. A form may be listed more than once with different categories; none is here. |
| [`goal`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-goal) | A derivation of a whole sentence must end in `S`. |
| [`matching`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-matching-relation), [`budget`](../how-to/grammars/choose-normal-form-matching-goal.md#bound-the-compiled-grammar) | Categories match after erasing features, and the chart closes unary rules for at most one round. |

The corpus around the grammar is declared in the same section:

```yaml
size: 24
seed: 5
length: {kind: uniform, min_length: 2, max_length: 5}
```

with [`retained: {validation: 0.25}`](../how-to/data/generate-a-synthetic-corpus.md#set-the-size-seed-and-splits) holding a quarter of the sentences out of
training. These settings control generation. We already know the source
grammar, so we can generate both a sentence and its derivation. In the next
chapter we withhold that derivation from the learner while retaining it for
evaluation.

## Composing the document

Composition reads the grammar as a typed declaration, exactly as it read the
grammar of chapter 1:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("docs/examples/tutorials/declared-grammar.yaml"))

declaration = run.data.datasets["train"].dataset
grammar = declaration.grammar
print(grammar.name, grammar.signature.primitives, grammar.goal.categories)
print([rule.view for rule in grammar.rules.rules])
print(len(grammar.lexicon.entries), declaration.size, declaration.retained)
```

```text
tutorial ('S', 'NP', 'N') ('S',)
['/', '\\']
8 24 {'validation': 0.25}
```

`declaration` is the corpus declaration, a [`SyntheticCorpusDecl`](../reference/configuration/datasets.md#lofi.data.synthetic.spec.corpus.SyntheticCorpusDecl), and it holds
everything the draw needs: the grammar, the size, the seed, the [length law](../how-to/data/generate-a-synthetic-corpus.md#choose-a-length-distribution) and
the split fractions.

## Drawing the corpus

[`plan_training`](../reference/run.md#lofi.run.plan_training) checks a composed run without allocating a model, and the
plan's [`lower_dataset`](../reference/run.md#lofi.run.TrainingBlueprint.lower_dataset) draws one dataset role. The draw compiles the grammar,
draws derivations under the declared [length law](../how-to/data/generate-a-synthetic-corpus.md#choose-a-length-distribution), and certifies every draw by
parsing its string again under the policy the run's [`parse`](../reference/spec.md#lofi.spec.ParseSpec) section declares,
which here retains the [packed forest](../explanation/charts-and-approximation.md#reusing-partial-analyses) and builds every chart item:

```python
from lofi.run import plan_training

plan = plan_training(run)
corpus = plan.lower_dataset("train").corpus

print(len(corpus.examples))
for example in corpus.examples[:5]:
    print(example.item.split, example.item.text)
```

```text
24
train the dog chases mary
train john sleeps
train john sleeps
train the cat sees the dog
train the cat sleeps
```

Each example pairs an item (its tokens, its text and the split it was assigned
to) with its gold supervision. The split assignment is drawn from the same
seed, so six of the twenty-four sentences land in `validation`. Drawing the
same run again, in a fresh plan, yields the same corpus:

```python
again = plan_training(codec(Path("docs/examples/tutorials/declared-grammar.yaml")))
print(again.lower_dataset("train").corpus.examples[0].item.text)
```

```text
the dog chases mary
```

## Reading a derivation

The supervision of a drawn example is its gold derivation. Its [`preorder`](../reference/data.md#lofi.data.Derivation.preorder)
lists node identifiers from the root down, and each node records the span it
covers, its category and the rule that built it:

```python
derivation = corpus.examples[0].supervision.derivation

for node_id in derivation.preorder:
    entry = derivation.node(node_id)
    print(entry.span, entry.label, entry.rule)
```

```text
TokenSpan(0, 4) S application:\
TokenSpan(0, 2) NP application:/
TokenSpan(0, 1) NP/N None
TokenSpan(1, 2) N None
TokenSpan(2, 4) S\NP application:/
TokenSpan(2, 3) S\NP/NP None
TokenSpan(3, 4) NP None
```

Here **gold** names the reference derivation supplied by the generator under
this particular grammar of English.
This is the derivation of *the dog chases mary*. Number the boundaries around
its words as `0 the 1 dog 2 chases 3 mary 4`. The span `(0, 2)` covers *the dog*;
`(2, 4)` covers *chases mary*. These are half-open ranges: include the start
position, stop before the end position. [Chapter 7](07-charts-and-semirings.md)
uses the same convention.

The first printed row covers the whole sentence. The next three describe
its left constituent and two words; the final three describe its right
constituent and two words. The leaves carry no rule:
their categories come from the [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline). Each inner node names the rule and view
that built it, `application:/` for forward and `application:\` for backward
application, and these names are the keys the model's
[`combinators.decoders`](../how-to/models/configure-term-encoders-and-decoders.md#choose-decoders) map assigns decoders to.

## Reading the draw's reports

The corpus records the measurements of the draw. Its length report pairs the declared [length law](../how-to/data/generate-a-synthetic-corpus.md#choose-a-length-distribution) with the lengths achieved:

```python
print(corpus.length)
for category, histogram in corpus.lexicon.classes.items():
    print(category, histogram)
print(corpus.ambiguity)
```

```text
LengthReport(Histogram({2: 6, 3: 5, 4: 8, 5: 5}), resampled=0, distribution=UniformLength(2, 5))
N Histogram({10: 1, 11: 1})
S\NP/NP Histogram({7: 1, 8: 1})
NP Histogram({8: 1, 10: 1})
S\NP Histogram({9: 1})
NP/N Histogram({21: 1})
AmbiguityReport(Histogram({1: 84}), Histogram({1: 24}), rejected=0, target=None)
```

Reading the three reports:

| Report | What it shows |
|---|---|
| [`LengthReport`](../reference/data.md#lofi.data.synthetic.LengthReport) | Six sentences of two tokens, five of three, eight of four and five of five, all inside the declared bounds, with none resampled. |
| [`lexicon.classes`](../reference/data.md#lofi.data.synthetic.LexiconReport.classes) | For each category, how often each of its forms was drawn. The two forms of `N` were drawn ten and eleven times; the one form of `NP/N`, *the*, twenty-one times. |
| [`AmbiguityReport`](../reference/data.md#lofi.data.synthetic.AmbiguityReport) | All eighty-four tokens had one category and all twenty-four sentences had one derivation in this draw. These are separate observations: a grammar can permit several derivations even when each word has only one [lexical category](../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms). |

The run of chapter 1 recorded these same measurements under the `corpus` field
of its metrics record; here they are read before any model exists.

The [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) histograms summarize how often forms occur.
`N Histogram({10: 1, 11: 1})` means that one noun occurred ten times and one
occurred eleven times. By contrast, the length histogram's `{2: 6}` means six
sentences had length two. Read each histogram's keys in the units of its report.

## A refused grammar

A [lexical category](../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms) must be built from the declared primitives and
constructors. Adding an entry whose category names an undeclared primitive is
refused when the run is planned, before anything is drawn:

```python
broken = codec(
    Path("docs/examples/tutorials/declared-grammar.yaml"),
    overrides=(
        'data.datasets.train.dataset.grammar.lexicon.entries+=[{"form": "barks", "category": "S\\\\VP"}]',
    ),
)

try:
    plan_training(broken)
except Exception as refused:
    print(type(refused).__name__)
    print(str(refused)[:300])
```

```text
DeclarationError
data.datasets.train.dataset.grammar.lexicon was refused: cannot parse operator 'VP' at character offset 2: operator is not declared by the signature
```

The refusal names the section, the symbol and its offset in the category
text, which is read against the declared signature.

## Try this

- Give *sees* a second entry, `{form: sees, category: NP}`, and draw again.
  The lexical histogram of the ambiguity report now has a bucket above one,
  and [`corpus.lexicon.per_type_ambiguity`](../reference/data.md#lofi.data.synthetic.LexiconReport.per_type_ambiguity) counts one form with two categories.
- Raise `length.max_length` to 8 and compare the achieved length histogram
  with the declared uniform law.
- Add `{form: big, category: N/N}` and read a derivation of a sentence that
  uses it: the adjective's node combines with its noun by [forward application](../how-to/grammars/choose-combinatory-rules.md#the-rule-families)
  before the determiner applies.

## Next

[Chapter 3](03-induce-a-lexicon.md) trains a parser on this corpus from its
strings alone and scores how much of the grammar it recovers.
