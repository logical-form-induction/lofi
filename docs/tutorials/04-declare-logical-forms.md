# Declaring logical forms

This chapter gives chapter 3's grammar a semantics: it selects a logical-form
language, attaches a meaning to every lexical entry, declares how categories
erase to semantic types, draws the corpus again, and reads the typed logical
form each derivation now yields. As in chapter 2, nothing is trained; chapter
5 trains a learner on judgments computed from forms like these.

## The document

[`docs/examples/tutorials/interpreted-grammar.yaml`](../examples/tutorials/interpreted-grammar.yaml)
names chapter 3's document in its `defaults` list, selects the packaged
`propositional` language for the corpus, and changes two parts of the corpus
declaration. The lexicon is restated with a `meaning` on each entry:

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

and an `interpretation` is added beside it:

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
| `language: propositional` | The packaged logical-form language: the types `e`, `v` and `t`, the function-type constructor `->`, and the connectives `not`, `and`, `or`, `if` and `iff`, decided by a truth-table prover. |
| `meaning` | Each lexical entry's logical form, written in the language's surface syntax. `\x.` binds a variable, and application is curried, so `see(x)(y)` applies `see` to the subject and then the object. |
| `erasure` | The semantic type of each primitive category. `S` erases to a truth value, `NP` to an individual and `N` to a property, written in angle-bracket type syntax. The image of a slash category follows from its functor: $\mathrm{S}\backslash\mathrm{NP}$ erases to $e\to t$. |
| `constants` | The non-logical constants the meanings are built from, each with a type and a role class. |
| `meanings: {source: {kind: lexical}}` | Read the meanings from the lexical entries themselves, rather than generating them from a semantic structure. |

No semantic action is declared for the application rule. When a rule's action
is not listed under `actions`, the interpretation derives it from the rule by
proof search, so forward and backward application both apply the function's
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

The language's own constants are its connectives. The interpretation's
constants are added to them when the interpretation is lowered.

## Drawing forms

The draw is the one of chapter 2. With an interpretation declared, the corpus
also holds the logical form of each grammatical item, keyed by the item's
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

## A refused meaning

Every meaning must have the type its category erases to. Adding *naps* at
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

The check is the one that licenses calling these forms typed. A corpus drawn
under this interpretation cannot hold a form whose type disagrees with its
derivation's category.

## Try this

- Add `{form: not-sleeps, category: 'S\NP', meaning: '\x. not(sleep(x))'}` to
  the lexicon. The connective comes from the language, so it needs no entry
  under `constants`.
- Change the erasure of `N` to `e` and read the refusal: every `N` meaning is
  now ill-typed.
- Swap the bound variables in the meaning of *sees* to $\lambda y.\lambda x.\operatorname{see}(y)(x)$
  and read how the drawn forms change.

## Next

[Chapter 5](05-induce-logical-forms.md) hides the forms from the learner and
asks it to recover the lexical meanings from inference judgments alone.
