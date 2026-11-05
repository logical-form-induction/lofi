# Choose a normal form, matching relation and goal

Four sections of a grammar declaration fix how its rules are run rather than
which rules there are. `normal_form` filters which derivation steps are
admitted, `matching` declares the congruence categories are compared under,
`goal` declares what a complete derivation ends in, and `budget` bounds the
compiled grammar's approximations and its structural closure. Each of the
first three selects a variant by its `kind` from an entry-point group. This
guide declares each, compiles the variants, and lists the refusals.

All examples start from the packaged `directed` grammar with first-order
composition added, so that a normal form has something to filter:

```python
from copy import deepcopy
import json
from pathlib import Path

from lofi.spec import SpecLoader, config_search_path, load_plugins
from lofi.spec.grammar import GrammarIso, GrammarSpec, declare_grammar
from lofi.spec.language import registered_language

load_plugins()
path = Path("lofi/configs/data/datasets/train/dataset/grammar/directed.yaml")
loader = SpecLoader(search_path=config_search_path(Path.cwd()))
directed = json.loads(loader.compose_traced(path, schema=GrammarSpec).spec.model_dump_json())
directed["rules"]["rules"] += [
    {"rule": "composition", "direction": "unrestricted", "order": 1, "view": "/"},
    {"rule": "composition", "direction": "unrestricted", "order": 1, "view": "\\"},
]
language = registered_language("propositional")


def compiled(**fields):
    document = {**deepcopy(directed), **fields}
    grammar = declare_grammar(GrammarSpec.model_validate(document))
    return GrammarIso().forward(grammar, language).grammar
```

## Declare a normal form

The `lofi.normal_form` group registers five variants:

| `kind` | Fields | Admits |
|---|---|---|
| `none` (default) | | Every derivation step. |
| `eisner` | `include_crossed` | Steps allowed by Eisner's composition normal form. |
| `undirected-eisner` | `include_crossed` | Eisner's form with both composition directions identified, for an undirected functor. |
| `hockenmaier-bisk` | `order_bound` (1), `track_child_order`, `composition_primary`, `unit_primary`, `raised_secondary`, `raised_application`, `coordination_raising` | Steps allowed by the Hockenmaier and Bisk normal form over every composition order up to the bound. |
| `all-of` | `normal_forms` | Steps every listed normal form admits. |

Compile each:

```python
for normal_form in (
    {"kind": "none"},
    {"kind": "eisner"},
    {"kind": "hockenmaier-bisk", "order_bound": 1},
    {"kind": "all-of", "normal_forms": [{"kind": "eisner"}, {"kind": "hockenmaier-bisk"}]},
):
    print(normal_form["kind"], type(compiled(normal_form=normal_form).normal_form).__name__)
```

```text
none NoNormalForm
eisner EisnerNormalForm
hockenmaier-bisk HockenmaierBiskNormalForm
all-of AllOf
```

A normal form reads each rule's structural role. The composition, raising and
coordination families declare theirs; a `pattern` rule declares one with its
`normal_form_role` field, as [Choose combinatory
rules](choose-combinatory-rules.md#write-a-pattern-rule) describes.
A pattern rule that declares no role is one no normal form constrains.
`hockenmaier-bisk` tags a type-raised category with the direction it was raised
in: forward, backward, or undirected when the view's two slashes coincide. A
composition constrains only primaries raised in its own direction.

## Declare a matching relation

The `lofi.matching` group registers six congruences:

| `kind` | Fields | Two categories match when |
|---|---|---|
| `identity` (default) | | They are the same term. |
| `alpha` | | They are the same up to renaming bound variables. |
| `feature-erasure` | | Their feature-erased cores are the same, so `S[dcl]` matches `S`. |
| `subsumption` | | Each subsumes the other, with free variables as placeholders. |
| `product` | `left`, `right` | Both declared relations hold. |
| `modal` | `modes`, `order` | They are the same term, qualified by a lattice of modes. The `tlg` formalism declares this one for you. |

```python
for matching in (
    {"kind": "identity"},
    {"kind": "feature-erasure"},
    {"kind": "product", "left": {"kind": "feature-erasure"}, "right": {"kind": "alpha"}},
):
    print(matching["kind"], compiled(matching=matching).matching.name.split(":")[0])
```

```text
identity identity
feature-erasure ccg-feature-erasure
product product(feature-erasure,alpha)
```

Over a signature with a conjunction inventory, feature erasure is compiled
together with that inventory, and its compiled name carries the inventory's
name and fingerprint.
A single rule may declare its own `matching`, which then replaces the
grammar's for that rule's premises; the packaged PMB grammars declare
`identity` matching on their coercion sets this way.

## Declare a goal

The `lofi.goal` group registers two goals. Both take the goal `categories`:

| `kind` | Fields | A derivation is complete when |
|---|---|---|
| `root` (default, `categories: [S]`) | `root_weights` | One derivation spans the sentence and ends in a goal category. `root_weights` gives one finite log weight per category. |
| `cover` | `max_fragments`, `penalty`, `penalty_form`, `tiling` | The sentence is tiled by at most `max_fragments` fragments, with `penalty` charged `after-first` fragment or for `each-fragment`, and the tiling chosen as `fewest-fragments` or `lexical`. |

```python
for goal in (
    {"kind": "root", "categories": ["S"]},
    {"kind": "root", "categories": ["S", "NP"], "root_weights": [0.0, -2.0]},
    {"kind": "cover", "categories": ["S"], "max_fragments": 3, "penalty": 1.0},
):
    print(goal["kind"], type(compiled(goal=goal).goal).__name__)
```

```text
root RootGoal
root RootGoal
cover CoverGoal
```

The `cfg` and `tlg` formalisms take their goal as one category name and build
the root goal themselves.

## Bound the compiled grammar

`budget` holds the grammar's approximation bounds and the limits of the
structural closure the compiler runs to check that the goal is reachable:

| Field | Default | Bounds |
|---|---|---|
| `depth` | 5 | Term depth. |
| `mass` | 1.0 | The probability-mass target, in (0, 1]. |
| `beam` | 8 | The search beam width. |
| `unary_rounds` | 1 | Rounds of unary closure. |
| `threads` | 1 | Chart-cell workers for every parse inside a run, held-out decoding included. |
| `beta_fuel` | 1000 | Beta-reduction steps. |
| `closure_rounds`, `closure_nodes` | 2, 2048 | Rounds and nodes of the structural closure. |
| `max_category_arity`, `max_category_size` | none | A category's functor-spine arity and node count. |

The compile report records what the closure reached. With no closure rounds,
the goal cannot be shown reachable and the grammar is refused:

```python
from lofi.spec import DeclarationError

for budget in ({"closure_rounds": 2}, {"closure_rounds": 1}, {"closure_rounds": 0}):
    document = {**deepcopy(directed), "budget": {**directed["budget"], **budget}}
    grammar = declare_grammar(GrammarSpec.model_validate(document))
    try:
        report = GrammarIso().forward(grammar, language).compile_report.measurements()
    except DeclarationError as refused:
        print(budget, str(refused).split(" in ClosureReport")[0])
    else:
        print(budget, report["closure-reached"], report["closure-saturated"], report["closure-category-set"])
```

```text
{'closure_rounds': 2} 4 True finite
{'closure_rounds': 1} 4 True finite
{'closure_rounds': 0} grammar.goal.categories was refused: targets ('S',) are unreachable
```

`closure-reached` counts the categories the closure reached,
`closure-saturated` says whether it stopped because nothing new was derivable,
and `closure-category-set` says whether the reached set is `finite`. The
packaged `ccgbank` grammar declares `closure_rounds: 0` with a goal that its
lexicon supplies directly.

## How these sections are refused

Each variant checks its own fields at validation:

```python
def refuse(**fields):
    try:
        GrammarSpec.model_validate({**deepcopy(directed), **fields})
    except Exception as refused:
        print(str(refused).splitlines()[1].strip())


refuse(goal={"kind": "root", "categories": ["S", "NP"], "root_weights": [0.0]})
refuse(goal={"kind": "cover", "categories": ["S"], "max_fragments": 0})
refuse(normal_form={"kind": "hockenmaier-bisk", "order_bound": 0})
refuse(budget={"mass": 1.5})
```

```text
goal: a grammar goal needs one root weight per category [axiom_failed]
goal: a fragment bound must be positive [axiom_failed]
normal_form: a Hockenmaier-Bisk order bound must be positive [axiom_failed]
budget: a mass target lies in (0, 1] [axiom_failed]
```

An unknown `kind` is refused with the list of registered variants, and a
variant that a plugin registers under one of these groups is validated
exactly as the packaged ones are.

## Related

- [Charts and approximation](../../explanation/charts-and-approximation.md)
  for what the budget's bounds trade.
- [Map between signatures](map-between-signatures.md) for feature erasure in
  detail.
