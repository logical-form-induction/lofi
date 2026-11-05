# Choose combinatory rules

A grammar's `rules` section is a
[`RuleSetSpec`](../../reference/spec.md): a named list of rule declarations,
each selecting a rule family by its `rule` tag. A `combinators` or `coercions`
entry in the same list is a concise inventory that expands, in its place, into
rules of those families. This guide selects packaged rule sets, declares the families one by
one, expands a concise inventory, and shows how a rule the grammar cannot run
is refused.

## The rule families

The `lofi.rule` entry-point group registers nine families, and two expanding
entries declare several of them at once:

| `rule` | Declares | Required fields |
|---|---|---|
| `application` | `X/Y Y ⇒ X` through one functor view. | `view` |
| `composition` | Generalized composition through a view, of one `order` or, with `order: null`, of every finite order above zero. `crossed: true` declares the crossed variant. | `view`, `order`, `direction` |
| `substitution` | Generalized substitution through a view, harmonic or `crossed`. | `view`, `order` |
| `type-raising` | Raising through a view of the listed `arguments`, by default `N` and `S`, optionally only to the listed `targets`; `arguments: null` raises every category. | `view` |
| `coordination` | A coordination step over the signature's conjunction inventory, in the `mark` phase (head policy `non-absorbing`) or the `merge` phase (head policy `left-conjunct`), relating conjuncts of the `same` or a `different` core. | `name`, `premises`, `conclusion`, `phase`, `core_relation`, `head_policy` |
| `absorption` | A binary rule in which one side, drawn from the `absorbing` categories, is absorbed. | `name`, `premises`, `conclusion`, `absorbing`, `side` |
| `clausal-adjunct` | An adjunct rule from a `source_view` to a `target_view` over `source_core` (`S`) and `argument` (`NP`). | `name`, `source_view`, `target_view` |
| `pattern` | Any rule written as premise patterns and a conclusion pattern. | `name`, `premises`, `conclusion` |
| `coercion` | One ground unary coercion, normally expanded from a `coercions` set. | `label`, `member`, `source`, `target` |
| `combinators` | An application, composition, substitution and type-raising inventory expanded over every functor view. | none |
| `coercions` | Ground unary coercions sharing one `label`, one `coercion` rule per `source => target` entry. | `label`, `entries` |

The composition `direction` restriction fixes which slashes the composed
spine may use. Under `steedman` every spine slash is the one the rule's view
names; under `kuhlmann` only the deepest slash, the one next to the shared
category, is fixed; under `unrestricted` no spine slash is constrained.

Every rule also accepts a `matching` relation for comparing its premises (the
grammar's own when absent), a `restriction` that limits its instances, and
`introduces` or `consumes` category lists. A restricted rule fires only when
the target of its primary premise, the atom reached by following functor
results, is one of its `primary_targets`, and its secondary premise is one of
its `secondary_patterns`. The primary premise of an application, composition
or substitution rule is its functor and the secondary its other premise; a
type-raising, coercion or unary pattern rule has a primary premise only; a
binary pattern rule names its primary premise through its composition role.
A restriction on any other rule, or a secondary restriction on a unary rule,
is refused when the grammar compiles, since the chart could not enforce it. An expanding entry hands each of
these to every rule it expands into. The `head_policy`, which selects the
lexical head a derivation step passes up, is fixed by the family for
application, composition, substitution, type raising, clausal adjuncts and
coercions, which therefore declare none; `pattern` and `absorption` rules
declare it with a default, and coordination declares it with its phase.

## Select a packaged rule set

Four rule sets are packaged under
`lofi/configs/data/datasets/train/dataset/grammar/rules/`. Each refinement
names `application` in its `defaults` list and appends to its rules with
`rules+`, as the [configuration guide](../setup/configuration.md) describes:

```yaml
defaults:
  - application
name: harmonic-composition
rules+:
  - {rule: composition, direction: unrestricted, order: 1, view: /}
  - {rule: composition, direction: unrestricted, order: 1, view: "\\"}
```

Compose each rule set and compile it into the packaged `directed` grammar:

```python
from pathlib import Path

from lofi.spec import RuleSetSpec, SpecLoader, load_plugins
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.language import registered_language

load_plugins()
loader = SpecLoader()
packaged = Path("lofi/configs/data/datasets/train/dataset/grammar")
directed = loader.compose_traced(packaged / "directed.yaml", schema=GrammarDecl).spec
language = registered_language("propositional")


def compiled_rules(rules):
    assembly = GrammarIso().forward(directed.with_(rules=rules), language)
    return [rule.name for rule in assembly.grammar.rules]


for name in ("application", "composition", "type-raising", "application-identity"):
    rules = loader.compose_traced(packaged / "rules" / f"{name}.yaml", schema=RuleSetSpec).spec
    print(f"{rules.name}: {compiled_rules(rules)}")
```

```text
application: ['application:/', 'application:\\']
harmonic-composition: ['application:/', 'application:\\', 'composition:/:1:harmonic:unrestricted', 'composition:\\:1:harmonic:unrestricted']
raised-composition: ['application:/', 'application:\\', 'composition:/:1:harmonic:unrestricted', 'type-raising:/:targets=S']
application-identity: ['application:/', 'application:\\', 'identity-combinator']
```

A compiled rule's name records its family, view, order, harmonic or crossed
form and direction restriction, so `composition:/:1:harmonic:unrestricted` is
first-order forward harmonic composition with no spine restriction. A model
that decodes a rule's action refers to it by this name; the
[first tutorial's](../../tutorials/01-first-run.md) document maps
`"application:/"` to a decoder under `model.spaces.categories.combinators`.

In a run document, select a rule set in the `defaults` list, as in
`data.datasets.train.dataset.grammar.rules: composition`. The [configuration
guide](../setup/configuration.md#extend-a-list-without-restating-it) shows a
project fragment that refines a packaged rule set the same way.

## Expand a concise inventory

A `combinators` entry declares a CCG rule inventory without listing each rule.
It expands, at its position in the list, over every functor view the
signature publishes:

```python
concise = RuleSetSpec.model_validate({
    "name": "concise",
    "rules": [{
        "rule": "combinators",
        "application": True,
        "composition": [1, 2],
        "crossed_composition": [1],
        "direction": "steedman",
        "type_raising": True,
        "raising_targets": ["S"],
    }],
})
for name in compiled_rules(concise):
    print(name)
```

```text
application:/
application:\
composition:/:1:harmonic:steedman
composition:/:2:harmonic:steedman
composition:\:1:harmonic:steedman
composition:\:2:harmonic:steedman
composition:/:1:crossed:steedman
composition:\:1:crossed:steedman
type-raising:/:targets=S:arguments=N,S
type-raising:\:targets=S:arguments=N,S
```

Each order listed under `composition`, `crossed_composition`, `substitution`
or `crossed_substitution` yields one rule per view; `null` in those lists
admits every finite order. `raising_targets` requires `type_raising: true`;
`raising_arguments` lists the categories that may raise, by default `N` and
`S`, and a raising rule declaring arguments names them after its targets.
The packaged `ccgbank` grammar lists one such entry after its explicit
pattern, absorption, clausal-adjunct and coordination rules:

```python
ccgbank = loader.compose_traced(packaged / "ccgbank.yaml", schema=GrammarDecl).spec
names = [rule.name for rule in GrammarIso().forward(ccgbank, language).grammar.rules]
print(len(names))
print([name for name in names if not name.startswith(("composition", "application"))])
```

```text
36
['n-to-np', 'punctuation-left', 'punctuation-right', 'clausal-adjunct-backward-backward', 'clausal-adjunct-backward-forward', 'clausal-adjunct-forward-backward', 'clausal-adjunct-forward-forward', 'appositive-np-comma-forward', 'appositive-comma-np-backward', 'appositive-comma-np-verbal', 'coordination-mark-conj', 'coordination-mark-period', 'coordination-mark-comma', 'coordination-mark-colon', 'coordination-mark-semicolon', 'coordination-mark-lrb', 'coordination-mark-rrb', 'coordination-merge', 'coordination-unlike', 'compound-conj-n', 'substitution:/:1:harmonic', 'substitution:\\:1:harmonic', 'substitution:/:1:crossed', 'substitution:\\:1:crossed', 'type-raising:/:targets=S,S[dcl]', 'type-raising:\\:targets=S,S[dcl]']
```

A `coercions` entry declares ground unary coercions that share one label, each
entry written `source => target`. The packaged PMB grammars under
`grammar/pmb/` append their lexical coercions to the shared PMB rules with
`rules+` this way.

## Write a pattern rule

A `pattern` rule states its premises and conclusion as term patterns: `term`
binds a variable of a sort, `constant` names a category constant, `node`
matches a constructor (`operator: {kind: fixed, name: /}`) with argument
patterns, and `spine` matches a functor spine between a `min_order` and a
`max_order`. The packaged
`application-identity` rule set adds a unary identity rule:

```yaml
rules+:
  - rule: pattern
    name: identity-combinator
    premises:
      - &category {pattern: term, name: X, sort: Cat}
    conclusion: *category
```

A pattern rule may also declare its `targets` and a `normal_form_role`, which
tells a [normal form](choose-normal-form-matching-goal.md) whether it acts as
composition, raising or coordination.

## How a rule is refused

A rule is checked twice: its own fields when the rule set is validated, and
its views and categories against the signature when the grammar is:

```python
def refuse(rules):
    try:
        directed.with_(rules=RuleSetSpec.model_validate({"name": "trial", "rules": rules}))
    except Exception as refused:
        print(str(refused).splitlines()[1].strip())


refuse([{"rule": "composition", "direction": "unrestricted", "order": 0, "view": "/"}])
refuse([{"rule": "composition", "direction": "unrestricted", "order": 1, "view": "|"}])
refuse([{"rule": "permutation", "view": "/"}])
```

```text
rules.0: composition order must be positive [axiom_failed]
<root>: a composition rule names '|', which is no functor constructor of signature 'directed' [validator_error]
rules: TaggedUnion 'RuleSpec' has no variant registered for rule='permutation'; expected one of ['absorption', 'application', 'clausal-adjunct', 'coercion', 'coercions', 'combinators', 'composition', 'coordination', 'pattern', 'substitution', 'type-raising']. [type_error]
```

Order zero is refused because it is application, which is declared on its
own. A coordination rule over a signature with no conjunction inventory is
refused in the same way; [Declare a category
signature](declare-a-signature.md#how-a-signature-is-refused) shows it.

## Check a rule family's laws

The `lofi.rule_laws` entry-point group publishes property-law cases for the
composition, substitution, raising and pattern families. Each case compiles a
declared grammar and checks the rule the compiler adopts, so the cases check
the same rules a grammar document declares. A plugin that adds a rule family
registers its plugin under `lofi.rule` and its law cases under
`lofi.rule_laws`; [Running the law suites](../extending/laws.md) shows how the
suites are run.

## Related

- [Choose a normal form, matching relation and goal](choose-normal-form-matching-goal.md)
  for the filters a rule set compiles under.
- [Derivations as proofs](../../explanation/derivations-as-proofs.md) for what
  a rule licenses.
