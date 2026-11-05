# Declare an interpretation

An interpretation maps a grammar's derivations to values. Two declarations
take part. A grammar's `interpretations` list binds names to derivation
readings from the `lofi.interpretation` group: the exact category at each
node, the surface string, the dependencies the rules' heads project, or the
logical form. A synthetic dataset's `interpretation` section then supplies
what the `meaning` reading needs: an erasure from categories to semantic
types, the typed constants, a source of lexical meanings, and the semantic
action of each rule. This guide declares both, lowers the result, and shows
how an interpretation that does not cover its grammar is refused.

## Name the derivation readings

A grammar that declares no `interpretations` gets the four packaged readings
under their own names. A grammar may bind other names, but each name once:

```yaml
interpretations:
  - {name: category, reading: {kind: category}}
  - {name: string, reading: {kind: string}}
  - {name: dependency, reading: {kind: dependency}}
  - {name: meaning, reading: {kind: meaning}}
```

| `kind` | A derivation's value |
|---|---|
| `category` | The exact category at each node. |
| `string` | The surface string the derivation yields. |
| `dependency` | The dependencies projected by each rule's head slot, as its `head_policy` selects. |
| `meaning` | The logical form the semantic actions build. |

Evaluators and readouts refer to a reading by its name; the packaged
category-side evaluator, for instance, declares
`dependency_interpretation: dependency`.

## Declare the dataset's interpretation

The packaged run document `lofi/configs/run/synthetic-parser-consistency.yaml`
extends the first tutorial's document with an interpretation:

```yaml
data:
  datasets:
    train:
      dataset:
        grammar:
          lexicon:
            entries:
              - {form: word-0, category: NP, meaning: a}
              - {form: word-1, category: "S\\NP", meaning: '\x. p(x)'}
              - {form: word-2, category: "S\\NP/NP", meaning: '\y. \x. r(x)(y)'}
              - {form: word-3, category: NP/NP, meaning: '\x. x'}
              # ... four more entries
        interpretation:
          erasure:
            kind: signature-morphism
            atoms: {S: t, NP: e, conj: e}
          constants:
            - {name: a, type: e, classes: [individual]}
            - {name: p, type: "e -> t", classes: [predicate]}
            - {name: r, type: "e -> e -> t", classes: [predicate]}
          meanings: {source: {kind: lexical}}
```

Reading the interpretation line by line:

| Line | What it says |
|---|---|
| `meaning` on each entry | The lexical logical form of that form and category, in the language's term syntax. |
| `erasure` | The image of each category atom as a semantic type. Functor images are derived from the signature's functor views, so `S\NP` erases to `e -> t`. |
| `constants` | The non-logical constants the meanings use, each typed and given at least one `classes` role for the symbol inventory. They are added to the language's own constants. |
| `meanings` | Read the meanings from the lexical entries. |

Compose the document and read the declarations back:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("lofi/configs/run/synthetic-parser-consistency.yaml"))
dataset = run.data.datasets["train"].dataset

print([(named.name, named.reading.kind) for named in dataset.grammar.interpretations])
interpretation = dataset.interpretation
print(interpretation.erasure.kind, interpretation.erasure.atoms)
print([(constant.name, constant.type) for constant in interpretation.constants])
print(interpretation.meanings.source.kind, interpretation.meanings.inhabitant_selection)
```

```text
[('category', 'category'), ('string', 'string'), ('dependency', 'dependency'), ('meaning', 'meaning')]
signature-morphism {'S': 't', 'NP': 'e', 'conj': 'e'}
[('a', 'e'), ('p', 'e -> t'), ('r', 'e -> e -> t')]
lexical unique
```

## Lower the interpretation

`lower_interpretation` compiles the grammar into the dataset's language with
the interpretation's constants added, resolves the lexical meanings, and
builds the interpretation function a draw applies:

```python
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.spec.grammar import GrammarIso, declare_grammar
from lofi.spec.language import LanguageIso

grammar = declare_grammar(dataset.grammar)
assembly, function = lower_interpretation(
    grammar, dataset.language, interpretation, GrammarIso(), LanguageIso()
)
print(sorted(function.interpretations))
print(dict(function.report()))
```

```text
['category', 'dependency', 'meaning', 'string']
{'actions': 2, 'uncovered': '[]', 'ambiguous': '[]', 'bound-exhausted': '{}'}
```

The report says every rule has an action: the grammar's two application
rules each received one, none is `uncovered` or `ambiguous`, and no proof
search ran out of its bound. The erasure the function holds maps each
category to its semantic type:

```python
for category in ("NP", "S\\NP", "S\\NP/NP", "NP/NP"):
    term = function.erasure(assembly.syntax.parse(assembly.store, category))
    print(f"{category:8} {assembly.language.print_type(term)}")
```

```text
NP       e
S\NP     e -> t
S\NP/NP  e -> e -> t
NP/NP    e -> e
```

## Derive or supply the rule actions

A rule's semantic action is derived by proof search over the erased types of
its premises and conclusion. `proof_search` declares the structural `logic`
(`relevant` by default, or `linear`, `affine` or `intuitionistic`) and
`max_uses`, which must be one for the linear and affine logics.
`meanings.inhabitant_selection` decides which inhabitants the search
returns become the action: `unique` (the default) takes the only one and
none if there are several, `first` takes the first in proof-search order,
`smallest` the smallest term, and `all` keeps every one.
`meanings.search_budget` (64) bounds how many inhabitants the search
collects. A rule with no unique derived action needs one in
`actions`, a closed term keyed by the rule's name and applied to its premises
in order; a listed action also overrides a derived one.

The other fields:

| Field | Default | What it declares |
|---|---|---|
| `entity`, `truth` | `e`, `t` | The types of an individual and of a formula, which must differ. |
| `type_syntax` | `angle` | How atom images are written: `⟨⟨e,t⟩,t⟩` in angle syntax, or the language's canonical s-expressions under `s-expression`. |
| `normalization` | | Beta traversal, the eager or lazy derivation policy, and quotient equations such as associative-commutative symbols. |
| `countermodels` | none | A countermodel filter over finite structures. |

## Choose a meaning source

`meanings.source` selects where the lexical meanings come from:

| `kind` | Meanings |
|---|---|
| `lexical` | The `meaning` on each lexical entry. At least one entry must carry one. |
| `structured` | Generated over the compiled lexicon from a semantic-lexicon `structure` declared inline, or a `preset` read from `semantic_lexicons/<name>.json` on the configuration search path. LoFI packages `natural`. |
| `resembling` | Drawn at a declared size to resemble a measured semantic lexicon. |

A generated source also generates the typed constants its meanings use.
[Control semantic distributions](../data/control-semantic-distributions.md)
declares the structure of a generated semantic lexicon.

## How an interpretation is refused

Lowering refuses an interpretation that does not cover its grammar. Strip the
meanings from the lexicon and lower again:

```python
from lofi.spec import DeclarationError

bare = grammar.with_(
    lexicon=grammar.lexicon.with_(
        entries=tuple(entry.with_(meaning=None) for entry in grammar.lexicon.entries)
    )
)
try:
    lower_interpretation(
        bare, dataset.language, interpretation, GrammarIso(), LanguageIso()
    )
except DeclarationError as refused:
    print(refused)
```

```text
the interpretation does not lower over 'synthetic-parser': a lexical meaning source needs a meaning on a lexical entry
```

Lowering also refuses a type term that does not read, a constant the language already declares, an
associative-commutative symbol that is not a closed binary operation, and a
language whose profile lies outside the declared prover's bound. A grammar
that binds one name to two readings is refused when the grammar itself is
validated.

## Related

- [Declare a logical-form language](declare-a-logical-form-language.md) for the
  language the meanings are written in.
- [Map between signatures](map-between-signatures.md) for the erasure in
  detail.
- [Logical forms as types](../../explanation/logical-forms-as-types.md) for
  why a rule's action is a proof term.
