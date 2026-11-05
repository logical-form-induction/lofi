# Declare a logical-form language

A [`LanguageDecl`](../../reference/configuration/language.md) declares the
higher-order signature logical forms are written in: its atomic semantic
types, the type constructors (one of which is the function type), the typed
constants, the binder style, and the logic, prover and proof resources that
decide entailment between forms. A constant's `classes` select a logical role
from the `lofi.logical_role` group, and its `semantics` select a postulated
interpretation from the `lofi.logical_constant` group. This guide selects a
registered or packaged language, declares one inline, and lists the
refusals.

A grammar is always compiled inside a language: every example in the grammar
guides compiles against `propositional`. The dataset declares its language
under `data.datasets.<name>.dataset.language`, beside its grammar.

## Use a registered language

The `lofi.lf_language` group registers two languages by name:

```python
from lofi.spec import load_plugins
from lofi.spec.language import registered_language

load_plugins()
language = registered_language("neo-davidsonian")

print([atom.name for atom in language.atomic_types])
for constant in language.constants:
    print(f"{constant.name:7} {constant.type:16} {constant.classes}")
print(language.logic.order, language.logic.equality, language.logic.prover.kind)
```

```text
['e', 'v', 't']
not     t -> t           ('sentential', 'negation')
and     t -> t -> t      ('sentential', 'conjunction')
or      t -> t -> t      ('sentential', 'disjunction')
if      t -> t -> t      ('sentential', 'conditional')
iff     t -> t -> t      ('sentential', 'biconditional')
eq      a -> a -> t      ('sentential', 'equality')
forall  (a -> t) -> t    ('quantifier', 'universal')
exists  (a -> t) -> t    ('quantifier', 'existential')
first True truth-table
```

`neo-davidsonian` has individuals, events and propositions, the five
sentential connectives, polymorphic equality and the two quantifiers.
`propositional` is the same language without equality and the quantifiers.
`eq`, `forall` and `exists` quantify over the type variable `a`. The
CCGbank, PMB and collected-corpus datasets accept a registered name in place
of a declaration, as in `language: neo-davidsonian`.

## Compose a packaged language

The configuration library holds its language fragments in
`lofi/configs/data/datasets/train/dataset/language/`, each built from shared
pieces under `_shared/` (the types, each connective, each quantifier and a
`higher-order-logic` fragment with a THF prover). `propositional` and
`neo-davidsonian` are general-purpose languages; `choice` (a Hilbert choice
operator), `dynamic` (dynamic conjunction and existential over indices) and
`plurals` (sums and parthood with meaning postulates) serve specific
phenomena. A run document selects one in its `defaults` list:

```yaml
defaults:
  - data.datasets.train.dataset.language: propositional
```

`experiments/s1/s1-run.yaml` does this. Any of the fragments can also be
composed on its own:

```python
from pathlib import Path

from lofi.spec import SpecLoader
from lofi.spec.language import LanguageDecl

loader = SpecLoader()
fragment = Path("lofi/configs/data/datasets/train/dataset/language/propositional.yaml")
propositional = loader.compose_traced(fragment, schema=LanguageDecl).spec
print(propositional.name, [constant.name for constant in propositional.constants])

plurals = loader.compose_traced(fragment.with_name("plurals.yaml"), schema=LanguageDecl).spec
print(plurals.name, plurals.logic.order, plurals.logic.prover.kind, plurals.logic.domains)
print([(constant.name, constant.semantics) for constant in plurals.constants if constant.semantics])
print(len(plurals.logic.postulates))
```

```text
propositional ['not', 'and', 'or', 'if', 'iff']
plurals higher tptp {'e': 3}
[('sum', 'sum'), ('part', 'part-of'), ('atom', 'atom')]
6
```

The `plurals` fragment adds three constants whose semantics are postulated
by plugins, bounds the individual domain at three, and declares six meaning
postulates the prover assumes. It names the shared pieces in its
`defaults` list and appends its own constants with `constants+`, as the
[configuration
guide](../setup/configuration.md#extend-a-list-without-restating-it)
describes.

## Declare a language inline

Declare a language with a generalized quantifier whose semantics come from the
`most` plugin:

```python
from lofi.spec.language import LanguageDecl

declared = {
    "name": "generalized-quantifiers",
    "atomic_types": [{"name": "e"}, {"name": "t", "kind": "proposition"}],
    "constructors": [
        {"name": "->", "kind": ["Ty", "Ty", "Ty"],
         "notation": {"style": "infix", "spelling": "->", "precedence": 1, "associativity": "right"}},
    ],
    "constants": [
        {"name": "not", "type": "t -> t", "classes": ["sentential", "negation"],
         "notation": {"kind": "prefix", "ascii": "~", "unicode": "¬", "precedence": 6}},
        {"name": "most", "type": "(e -> t) -> (e -> t) -> t", "classes": ["quantifier"], "semantics": "most"},
        {"name": "dog", "type": "e -> t", "classes": ["predicate"]},
    ],
    "binders": {"style": "curry"},
    "logic": {"negation": "not", "domains": {"e": 3}},
}
language = LanguageDecl.model_validate(declared)
plugins = language.prepared_plugins
print(sorted(plugins.roles), sorted(plugins.semantics))
```

```text
['not'] ['most']
```

Validation resolves every role and semantics plugin once: `not` takes the
`negation` role because one of its classes names that role, and `most` takes
the `most` semantics. A class that names no role, such as `quantifier` or
`predicate`, is kept as a class; the [interpretation](declare-an-interpretation.md)
reads `individual` and `predicate` to build its symbol inventory.

The parts of the declaration:

| Field | What it declares |
|---|---|
| `atomic_types` | Each atomic type's `name` and resource `kind`, `individual` by default. The packaged languages give `t` the kind `proposition` and `v` the kind `event`. |
| `constructors` | Type constructors over the type sort, read in their declared notation by the same reader as category text. A token ending in a letter, digit or underscore matches only as a whole word, so a type named `toe` is not read as the token `to`. The one named by `function_type` (`->` by default) must be binary and infix. |
| `constants` | Each constant's `name`, `type` (or `infer`, to infer it from its uses), the type variables it `quantifies`, its `classes`, a surface `notation` (`prefix`, `infix` or `binder`, each with an ASCII and a Unicode token), and either a `definition` as a closed term or a postulated `semantics` plugin. |
| `leaf_lexicon` | A source of further non-logical constants, read once when the language is compiled. |
| `binders` | The binder `style`, `curry` (unannotated) or `church` (annotated lambdas). |
| `logic` | The logic's `order`, whether it has `functions` and `equality`, finite `domains` per type, meaning `postulates`, the variable inventory, the designated `negation`, the `label_map`, and the `prover` and `labeling` resources. |
| `bounds` | One finite representation bound per sort (`Ty`, `Tm` and `Cat`), used by the neural term encoders. |

The [theorem-prover guide](../training/reward-with-a-theorem-prover.md)
covers the `prover` declarations.

## Read constants from a leaf lexicon

A `leaf_lexicon` of kind `path` reads constants from a JSON, TOML or YAML
document; a relative path is read from `lofi/configs/leaf_lexicons/`, which
packages `extensional.json`, `predicate.json` and `superintensional.json`.
Kind `inline` lists the `entries` in the declaration itself:

```python
from lofi.spec.language import PathLeafLexiconSpec

leaves = PathLeafLexiconSpec(path="predicate.json").load()
for leaf in leaves:
    print(leaf.name, leaf.type, leaf.classes)
```

```text
jo e ('individual',)
sleep e -> v -> t ('predicate',)
love e -> e -> v -> t ('predicate',)
tall (e -> v -> t) -> e -> v -> t ('predicate',)
think (v -> t) -> e -> v -> t ('predicate',)
```

A leaf constant's type defaults to `infer`. Compiling a grammar against the
language appends the leaf constants to the declared ones.

## How a language is refused

```python
from copy import deepcopy


def refuse(edit):
    changed = deepcopy(declared)
    edit(changed)
    try:
        LanguageDecl.model_validate(changed)
    except Exception as refused:
        print(str(refused).splitlines()[1].strip()[:150])


refuse(lambda d: d["constants"][0].__setitem__("classes", ["negation", "conjunction"]))
refuse(lambda d: d["constants"][1].__setitem__("semantics", "few"))
refuse(lambda d: d.__setitem__("function_type", "=>"))
```

```text
<root>: logical constant 'not' declares conflicting roles ['conjunction', 'negation'] [validator_error]
<root>: logical constant 'most' requests semantics 'few'; installed semantics are atom, choice, most, part-of, sum [validator_error]
<root>: the function type '=>' names no declared binary type constructor [validator_error]
```

A language also refuses a repeated atomic type, constructor or constant, a
notation token assigned twice, a constant with both a `definition` and a
`semantics`, a type constructor whose kind leaves the type sort, and bounds
that do not cover exactly the language's three sorts.

## Add a role or a semantics

A plugin adds a logical role by registering a `LogicalRole` under
`lofi.logical_role`, and a constant semantics by registering a
`LogicalConstantSemantics` under `lofi.logical_constant`; in both cases the
member's `name` must equal its entry-point name. A whole language is
registered under `lofi.lf_language` as a function that returns its
`LanguageDecl`. The [entry-point reference](../../reference/entry-points.md)
lists the groups.

## Related

- [Declare an interpretation](declare-an-interpretation.md) to map a grammar's
  derivations into this language.
- [Logical forms as types](../../explanation/logical-forms-as-types.md) for
  how categories, types and terms share one store.
