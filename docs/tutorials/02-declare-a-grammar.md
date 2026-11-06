# Declaring a grammar

This chapter writes a categorial grammar from nothing and draws a corpus from
it, without training anything: it declares the grammar's signature, rules and
lexicon in a run document, composes the document, draws the synthetic corpus
the run would train on, reads one drawn derivation node by node, and reads the
reports the draw measures about itself. Chapter 3 trains a parser on exactly
this corpus.

In chapter 1 the slash constructors came from the packaged `directed`
fragment, and the document only replaced the lexicon. Here the document owns
every part of the grammar, so nothing about it is inherited from a fragment.

## The grammar

The document is
[`docs/examples/tutorials/declared-grammar.yaml`](../examples/tutorials/declared-grammar.yaml).
Its model, loss and trainer sections are those of chapter 1 at width 4, with a
learned token encoder in place of chapter 1's static vectors, since the new
forms have no pretrained vectors. Its grammar, declared under
`data.datasets.train.dataset`, is this:

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

Reading the grammar line by line:

| Line | What it says |
|---|---|
| `primitives` | The atomic categories: sentences, noun phrases and nouns. |
| `constructors` | Two binary constructors over the category sort `Cat`, spelled infix and left-associative, so `S\NP/NP` reads as `(S\NP)/NP`. |
| `functors` | Each constructor builds a function category. `/` takes its argument on the right and `\` on the left, and each names the other as its opposite. |
| `rules` | Forward and backward application, $\frac{X/Y\quad Y}{X}$ and $\frac{Y\quad X\backslash Y}{X}$, as two views of one rule. |
| `lexicon` | Eight forms, each with one category. A form may be listed more than once with different categories; none is here. |
| `goal` | A derivation of a whole sentence must end in `S`. |
| `matching`, `budget` | Categories match after erasing features, and the chart closes unary rules for at most one round. |

The corpus around the grammar is declared in the same section:

```yaml
size: 24
seed: 5
length: {kind: uniform, min_length: 2, max_length: 5}
```

with `retained: {validation: 0.25}` holding a quarter of the sentences out of
training.

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

`declaration` is the corpus declaration, a `SyntheticCorpusDecl`, and it holds
everything the draw needs: the grammar, the size, the seed, the length law and
the split fractions.

## Drawing the corpus

`plan_training` checks a composed run without allocating a model, and the
plan's `lower_dataset` draws one dataset role. The draw compiles the grammar,
draws derivations under the declared length law, and certifies every draw by
parsing its string again under the policy the run's `parse` section declares,
which here retains the packed forest and builds every chart item:

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

The supervision of a drawn example is its gold derivation. Its `preorder`
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

This is the derivation of *the dog chases mary*. Spans are half-open token
ranges, as in [chapter 7](07-charts-and-semirings.md). The leaves carry no rule:
their categories come from the lexicon. Each inner node names the rule and view
that built it, `application:/` for forward and `application:\` for backward
application, and these names are the keys the model's
`combinators.decoders` map assigns decoders to.

## Reading the draw's reports

The corpus measures what it drew instead of assuming it. Its length report
pairs the declared length law with the lengths achieved:

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
| `LengthReport` | Six sentences of two tokens, five of three, eight of four and five of five, all inside the declared bounds, with none resampled. |
| `lexicon.classes` | For each category, how often each of its forms was drawn. The two forms of `N` were drawn ten and eleven times; the one form of `NP/N`, *the*, twenty-one times. |
| `AmbiguityReport` | All eighty-four tokens had one category and all twenty-four sentences had one derivation. This lexicon assigns one category per form, so neither lexical nor derivational ambiguity arises. |

The run of chapter 1 recorded these same measurements under the `corpus` field
of its metrics record; here they are read before any model exists.

## A refused grammar

A lexical category must be built from the declared primitives and
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
  and `corpus.lexicon.per_type_ambiguity` counts one form with two categories.
- Raise `length.max_length` to 8 and compare the achieved length histogram
  with the declared uniform law.
- Add `{form: big, category: N/N}` and read a derivation of a sentence that
  uses it: the adjective's node combines with its noun by forward application
  before the determiner applies.

## Next

[Chapter 3](03-induce-a-lexicon.md) trains a parser on this corpus from its
strings alone and scores how much of the grammar it recovers.
