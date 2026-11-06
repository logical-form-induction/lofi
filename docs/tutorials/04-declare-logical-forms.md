# Declaring logical forms

This chapter gives chapter 3's grammar a semantics: it selects a [logical form
language](../how-to/grammars/declare-a-logical-form-language.md#use-a-registered-language), attaches a meaning to every lexical entry, declares how categories
erase to semantic types, draws the corpus again, and reads the typed logical
form each derivation now yields. As in chapter 2, nothing is trained; chapter
5 trains a learner on judgments computed from forms like these.

For *john sleeps*, the category derivation establishes that the words form a
sentence. Its [logical form](../explanation/logical-forms-as-types.md#semantic-typing), `sleep(john)`, represents what the sentence says.
We use `e` for the type of individuals and `t` for the type of truth values.
Thus `john` has type `e`, and `sleep` has type `e -> t`: it takes an individual
and produces a truth value. The categories organize combination in the
sentence; the semantic types check combination in its meaning.

The [lambda notation](../explanation/reading-the-notation.md#lambda-expressions-and-substitution) `\x. sleep(x)` means a function whose input is called `x`.
Applying it to `john` replaces that input with `john`. Written mathematically,
this reduction is $(\lambda x.\operatorname{sleep}(x))(\operatorname{john})
\to\operatorname{sleep}(\operatorname{john})$. The lambda symbol marks an
input waiting to be supplied; the variable name holds its place in the expression.

## The document

[`docs/examples/tutorials/interpreted-grammar.yaml`](../examples/tutorials/interpreted-grammar.yaml)
names chapter 3's document in its [`defaults`](../how-to/setup/configuration.md#select-fragments-with-defaults) list, selects the packaged
[`propositional` language](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/data/datasets/train/dataset/language/propositional.yaml) for the corpus, and changes two parts of the corpus
declaration. The [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) is restated with a [`meaning` on each entry](../how-to/grammars/declare-a-lexicon.md#list-entries-inline):

```yaml
defaults:
  - induce-lexicon
  - data.datasets.train.dataset.language: propositional
data:
  datasets:
    train:
      dataset:
        grammar:
          lexicon:
            entries:
              - {form: john, category: NP, meaning: john}
              - {form: mary, category: NP, meaning: mary}
              - {form: the, category: NP/N, meaning: '\p. the(p)'}
              - {form: dog, category: N, meaning: '\x. dog(x)'}
              - {form: cat, category: N, meaning: '\x. cat(x)'}
              - {form: sleeps, category: 'S\NP', meaning: '\x. sleep(x)'}
              - {form: sees, category: 'S\NP/NP', meaning: '\y. \x. see(x)(y)'}
              - {form: chases, category: 'S\NP/NP', meaning: '\y. \x. chase(x)(y)'}
```

and an [`interpretation`](../how-to/grammars/declare-an-interpretation.md#declare-the-datasets-interpretation) is added beside it:

```yaml
interpretation:
  erasure:
    kind: signature-morphism
    atoms: {S: t, NP: e, N: '⟨e,t⟩'}
  constants:
    - {name: john, type: e, classes: [individual]}
    - {name: mary, type: e, classes: [individual]}
    - {name: the, type: '(e -> t) -> e', classes: [determiner]}
    - {name: dog, type: 'e -> t', classes: [predicate]}
    # ... cat, sleep, see and chase
  meanings: {source: {kind: lexical}}
```

Reading the new parts line by line:

| Line | What it says |
|---|---|
| `language: propositional` | The packaged [logical form language](../how-to/grammars/declare-a-logical-form-language.md#declare-a-language-inline): the types `e`, `v` and `t`, the function type constructor `->`, and the connectives `not`, `and`, `or`, `if` and `iff`, decided by a truth table prover. |
| `meaning` | Each lexical entry's logical form, written in the language's surface syntax. `\x.` binds a variable, and application is curried, so `see(x)(y)` applies `see` to the subject and then the object. |
| `erasure` | The semantic type of each primitive category. `S` erases to a truth value, `NP` to an individual and `N` to a property, written in angle bracket type syntax. The image of a slash category follows from its functor: $\mathrm{S}\backslash\mathrm{NP}$ erases to $e\to t$. |
| [`constants`](../how-to/grammars/declare-an-interpretation.md#declare-the-datasets-interpretation) | The nonlogical constants the meanings are built from, each with a type and a role class. |
| [`meanings: {source: {kind: lexical}}`](../how-to/grammars/declare-an-interpretation.md#choose-a-meaning-source) | Read the meaning supplied by each lexical entry. |

[Erasure](../how-to/grammars/map-between-signatures.md#map-categories-onto-semantic-types) translates a syntactic category into a semantic type while
forgetting distinctions that the semantic type does not need. Both `S/NP`
and `S\NP` can map to `e -> t`, though they expect a noun phrase on different
sides. `signature-morphism` declares this translation systematically: its
mapping of primitive categories and treatment of constructors determine the
type of a larger category. Sentence meanings remain logical form terms.

The meaning of *sees* takes its inputs one at a time, which is called
**[currying](../explanation/logical-forms-as-types.md#semantic-typing)**. In `\y. \x. see(x)(y)`, `y` is the object supplied on the
right; `x` is the subject supplied on the left. Combining *sees mary* first
thus leaves `\x. see(x)(mary)`, waiting for a subject. Argument order in
the written predicate and the order in which syntax supplies arguments are
distinct, and the lambda expression relates them.

The package named `propositional` supplies a
[truth table prover](../explanation/inference-as-evidence.md#deciding-a-question). It treats atomic propositions as Boolean inputs
and evaluates their propositional combinations. Domain knowledge about
dogs or people would require additional axioms; quantified first-order
reasoning would require another prover. The constant `the` maps properties
to individuals in this tutorial. A uniqueness condition would require an
additional semantic assumption.

No semantic action is declared for the application rule. When a rule's action
is not listed under `actions`, the interpretation derives it from the rule by
[proof search](../how-to/grammars/declare-an-interpretation.md#derive-or-supply-the-rule-actions), so forward and backward application both apply the function's
meaning to the argument's.

## Composing the document

The language and the erasure are part of the corpus declaration once the
document is composed:

```python
from pathlib import Path

from lofi.run import plan_training
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("docs/examples/tutorials/interpreted-grammar.yaml"))

declaration = run.data.datasets["train"].dataset
print(
    declaration.language.name,
    [constant.name for constant in declaration.language.constants],
)
print(declaration.interpretation.erasure.atoms)
```

```text
propositional ['not', 'and', 'or', 'if', 'iff']
{'S': 't', 'NP': 'e', 'N': '⟨e,t⟩'}
```

The [language's own constants](../how-to/grammars/declare-a-logical-form-language.md#declare-a-language-inline) are its connectives. The [interpretation's
constants](../how-to/grammars/declare-an-interpretation.md#declare-the-datasets-interpretation) are added to them when the interpretation is lowered.

## Drawing forms

The draw is the one of chapter 2. With an interpretation declared, the corpus
also holds the [logical form of each grammatical item](../reference/data.md#lofi.data.synthetic.SyntheticCorpus.forms), keyed by the item's
identifier:

```python
corpus = plan_training(run).lower_dataset("train").corpus

print(len(corpus.examples), len(corpus.forms))
for example in corpus.examples[:5]:
    print(example.item.text, "=>", corpus.forms[example.item.item_id].text)
```

```text
40 40
the dog chases mary => chase(the(\x0. dog(x0)))(mary)
john sleeps => sleep(john)
john sleeps => sleep(john)
the cat sees the dog => see(the(\x0. cat(x0)))(the(\x0. dog(x0)))
the cat sleeps => sleep(the(\x0. cat(x0)))
```

The corpus holds forty sentences, chapter 3's size, and the same seed draws
the same first sentences as chapter 2. Each form is the derivation's meaning
after the actions have been applied and the result normalized: in
*the dog chases mary*, `chases` takes `mary` as its object and then `the dog`
as its subject, which $\lambda y.\lambda x.\operatorname{chase}(x)(y)$ places first. The printed form
names the bound variable of *dog*'s meaning `x0`.

[Normalization](../explanation/logical-forms-as-types.md#substitution-equality-and-matching) simplifies function applications such as the application
of `\x. sleep(x)` to `john`. Bound variables may also be renamed consistently:
`\x0. dog(x0)` and `\x. dog(x)` express the same function. The printed
variable names bind the same argument position.

## A refused meaning

Every meaning must have the [type its category erases to](../reference/data.md#lofi.data.synthetic.InterpretationFunction.check). Adding *naps* at
the category of *sleeps* but with the two-place meaning of *sees* is refused
when the interpretation is lowered, as the run is planned and before any
sentence is drawn:

```python
broken = codec(
    Path("docs/examples/tutorials/interpreted-grammar.yaml"),
    overrides=(
        'data.datasets.train.dataset.grammar.lexicon.entries+=[{"form": "naps", "category": "S\\\\NP", "meaning": "\\\\y. \\\\x. see(x)(y)"}]',
    ),
)

try:
    plan_training(broken)
except Exception as refused:
    print(type(refused).__name__)
    print(str(refused)[:400])
```

```text
DeclarationError
data.datasets.train.dataset was refused: the interpretation does not lower over 'tutorial': the meaning of 'naps' at 'S\\NP' does not type as the erasure of its category: logical-form type 'e -> e -> t' does not match category type 'e -> t'
```

The check requires every form in the drawn corpus to have the type obtained
from its derivation's category.

## Try this

- Add `{form: not-sleeps, category: 'S\NP', meaning: '\x. not(sleep(x))'}` to
  the [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline). The connective comes from the language, so it needs no entry
  under `constants`.
- [Change the erasure of `N`](../how-to/grammars/map-between-signatures.md#map-categories-onto-semantic-types) to `e` and read the refusal: every `N` meaning is
  now ill-typed.
- Swap the bound variables in the meaning of *sees* to $\lambda y.\lambda x.\operatorname{see}(y)(x)$
  and read how the drawn forms change.

## Next

[Chapter 5](05-induce-logical-forms.md) hides the forms from the learner and
asks it to recover the lexical meanings from inference judgments alone.
