# Declare a category signature

A grammar's `signature` declares the [term algebra](../../explanation/logical-forms-as-types.md#signatures-and-terms) its categories are built in:
the primitive categories, the constructors that build complex categories from
them, the functor views that tell the rules which constructor slot is the
result and which the argument, the featured categories that erase to a core,
and the conjunction markers coordination runs over. This guide declares each
part, compiles the result, and shows how a malformed signature is refused.

The [`SignatureSpec`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) class is the
declaration; every grammar section of a run document holds one under
`signature`.

For a hypothetical grammar, begin with the primitive categories `NP` and
`S`. A [constructor](../../explanation/logical-forms-as-types.md#signatures-and-terms) combines categories to form another category: `S\NP`
is a sentence missing a noun phrase on its left, and `(S\NP)/NP` also needs
a noun phrase on its right. The constructor's *arity* is its number of inputs;
each slash here has two. An operator's *sort* is the sort of its result;
its full *kind* records the argument sorts followed by the result sort.
Each slash here has kind `[Cat, Cat, Cat]`: two category inputs and a
category result.

A [term](../../explanation/logical-forms-as-types.md#signatures-and-terms) is an expression built from these symbols. A ground term has no
unfilled variables. The signature specifies which expressions can be written;
rules separately specify which combinations of expressions are licensed.
To use slash categories in application, declare the corresponding rule.

Write $\Sigma$ for the declared many-sorted signature and
$\mathcal T_\Sigma(\mathrm{Cat})$ for its ground category terms. A primitive
is a nullary symbol of sort $\mathrm{Cat}$; a constructor of arity $k$ has
profile

$$
c:\underbrace{\mathrm{Cat}\times\cdots\times\mathrm{Cat}}_{k\text{ arguments}}
\longrightarrow\mathrm{Cat}.
$$

The formation condition is recursive: if $A_1,\ldots,A_k$ are category
terms, then $c(A_1,\ldots,A_k)$ is a category term. This is a term algebra
\citep{meinke-tucker1992}; the configuration's [`functors`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) adds operational
information about argument direction and constructor slots. Here “functor”
denotes a grammatical category expecting an argument. See [Signatures and terms](../../explanation/logical-forms-as-types.md#signatures-and-terms)
for the general sorted and binding-aware definition.

## A complete signature

The example
[`docs/examples/grammars-data/grammars/featured.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/grammars-data/grammars/featured.yaml)
declares a small grammar whose signature uses every part this guide covers:

```yaml
signature:
  name: featured
  primitives: [S, NP, N, conj]
  constructors:
    - {name: /, kind: [Cat, Cat, Cat], notation: {style: infix, spelling: /, precedence: 10, associativity: left}}
    - {name: '\', kind: [Cat, Cat, Cat], notation: {style: infix, spelling: '\', precedence: 10, associativity: left}}
  functors:
    - {name: /, constructor: /, argument_side: right, opposite: '\'}
    - {name: '\', constructor: '\', argument_side: left, opposite: /}
  operators:
    - {name: 'S[dcl]', sort: Cat, erases_to: S}
    - {name: 'S[b]', sort: Cat, erases_to: S}
    - {name: 'NP[conj]', sort: Cat, erases_to: NP}
  conjunctions:
    markers: [conj]
    pairs:
      - {unmarked: NP, marked: 'NP[conj]'}
```

Reading the signature line by line:

| Line | What it says |
|---|---|
| [`primitives`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) | The atomic categories. Each is a constant of the category sort, `Cat` by default. |
| `constructors` | Two binary constructors of kind $\mathrm{Cat}\times\mathrm{Cat}\to\mathrm{Cat}$, written infix as `/` and `\`. The notation fixes how category text is read and printed. |
| `functors` | The two constructors as functor views. `/` takes its argument on the right and `\` on the left; each names the other as its `opposite`. Both default to result slot `0` and argument slot `1`. |
| `operators` | Featured categories. `S[dcl]` is `S` with the feature `dcl`; it takes every other property from `S`. |
| `conjunctions` | The marker `conj` and the pair relating `NP` to its marked form `NP[conj]`, which coordination rules produce and consume. |

## Declare primitives and constructors

List the atomic categories under [`primitives`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec). A category is any term built
from them with the declared `constructors`, and each constructor declares its
`kind` (its argument sorts followed by its result sort) and a `notation`:
`infix` with a spelling, precedence and associativity, `prefix` with a
spelling, or `mixfix` with one more part than it has arguments. An infix
constructor must be binary, and every category constructor must take category arguments and return
a category. Category text is read by the same notation reader that reads
a [logical form language](declare-a-logical-form-language.md#use-a-registered-language)'s [semantic types](../../explanation/logical-forms-as-types.md#semantic-typing): prefix and mixfix tokens are tried
longest first, and infix constructors bind by precedence and associativity. In
category text a constructor token also ends the atom before it, so `S\NP` needs
no spaces.

The `syntax` field selects how category text is read. Under the default,
`declared`, categories are written in the constructors' notation, as in
`(S[dcl]\NP)/NP`; this syntax needs at least one constructor. Under
`s-expression`, categories are written as canonical terms, as in
`("/_linear" ("#const[Cat]" "S") ("#const[Cat]" "NP"))`. The packaged
`undirected` and `nl-diamond` grammars use the second form.

## Declare functor views

There are two orders to keep apart: the order of the constructor's slots and
the order of words in a sentence. In `S\NP`, slot zero contains the result
`S` and slot one contains the required `NP`, but the NP phrase appears to the
left of the expression being completed. A functor view supplies that link
between stored structure and grammatical combination.

A rule such as application or composition names the functor view it acts
through (`view: /`), not the constructor. Each [`functors`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) row names a binary
constructor, the side its argument sits on (`left`, `right` or `either`), the
view that is its `opposite`, and the `result` and `argument` slots, which must
differ. Views come in two shapes:

- a forward and a backward view over two constructors, with sides `right` and
  `left`, as in the example;
- one self-opposite view with side `either` over a single constructor, as in
  the packaged undirected grammar.

A rule that names a view the signature does not declare is refused when the
grammar is validated; [Choose combinatory rules](choose-combinatory-rules.md)
shows that refusal.
## Declare featured categories and conjunctions

Suppose a rule requires `S` and its proposed argument has category `S[dcl]`.
[Identity matching](choose-normal-form-matching-goal.md#declare-a-matching-relation)
keeps these terms distinct; [feature erasure](../../explanation/logical-forms-as-types.md#substitution-equality-and-matching) compares both as `S`. Declaring
`dcl` in the signature preserves the annotation, while selecting feature
erasure permits that rule to ignore it.

A featured category is an [`operators`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) row whose `name` is its target's name
with one feature in the selected `feature_delimiter` syntax, `brackets`
(`S[dcl]`) by default or `colon` (`S:dcl`), and whose `erases_to` names the
target. It declares nothing else: its shape comes from the target. The
grammar's [matching relation](choose-normal-form-matching-goal.md) decides
whether a rule compares categories with or without their features.

The `conjunctions` block is optional. Without it the signature has no
conjunction marks, and a grammar that declares a coordination rule over it is
refused. Its `pairs` name finite marked forms. Its optional `marking` names a
unary category constructor, e.g. `'@[conj]'` declared with kind `[Cat, Cat]`,
whose application to any category is that category's marked form, so
coordination also marks and merges categories no pair names, raised and
composed ones among them. A `marking` that does not map one category to
another is refused when the grammar compiles.

## Inspect the declaration

Compose the example against [`GrammarSpec`](../../reference/configuration/grammar.md#lofi.spec.grammar.GrammarSpec) and read the parsed signature:

```python
from pathlib import Path

from lofi.spec import SpecLoader, load_plugins
from lofi.spec.grammar import GrammarIso, GrammarSpec, declare_grammar
from lofi.spec.language import registered_language

load_plugins()
loader = SpecLoader()
path = Path("docs/examples/grammars-data/grammars/featured.yaml")
grammar = declare_grammar(loader.compose_traced(path, schema=GrammarSpec).spec)
signature = grammar.signature

print(signature.primitives)
print([(operator.name, operator.erases_to) for operator in signature.operators])
print(signature.functor_slots)
```

```text
('S', 'NP', 'N', 'conj')
[('S[dcl]', 'S'), ('S[b]', 'S'), ('NP[conj]', 'NP')]
{'/': (0, 1), '\\': (0, 1)}
```

`functor_slots` maps each functor constructor to its result and argument
slots. Compiling the grammar against a [logical form language](declare-a-logical-form-language.md#use-a-registered-language) links the
signature into that language's term store:

```python
assembly = GrammarIso().forward(grammar, registered_language("propositional"))
measurements = assembly.compile_report.measurements()

print([rule.name for rule in assembly.grammar.rules])
print(measurements["congruence"].split(":")[:2])
print(measurements["diagnostics"])
```

```text
['application:/', 'application:\\', 'coordination-mark', 'coordination-merge']
['ccg-feature-erasure', 'conjunctions']
[]
```

The congruence is [feature erasure](../../explanation/logical-forms-as-types.md#substitution-equality-and-matching) over the declared conjunction inventory.
The compile report's `diagnostics` list holds an `E310` row for every operator
of the category sort whose arity no category reached by the [structural closure](../../explanation/derivations-as-proofs.md#what-the-compiler-derives)
uses; it is empty here.

## How a signature is refused

Every check runs when the composed declaration is validated, before any
language is linked or any corpus drawn. Four overrides on the example, each
refused:

```python
def refuse(*overrides):
    try:
        loader.compose_traced(path, schema=GrammarSpec, overrides=overrides).spec
    except Exception as refused:
        print(refused)


refuse("signature.conjunctions=null")
refuse('signature.primitives+=["NP"]')
refuse('signature.operators+=[{"name": "VP[ng]", "sort": "Cat", "erases_to": "VP"}]')
refuse(
    'signature.functors=[{"name": "/", "constructor": "/", "argument_side": "right", "opposite": "\\\\"}]'
)
```

```text
docs/examples/grammars-data/grammars/featured.yaml cannot be composed: a coordination rule needs a conjunction inventory, and this signature declares none
signature.primitives was refused: category primitive 'NP' is declared twice
signature was refused: featured operator 'VP[ng]' erases to 'VP', which the signature does not declare
signature was refused: functor view '/' has no mutual opposite
```

A refusal the whole declaration raises names the document; one a part of the
signature raises names that part.

The signature also refuses an operator that should be a primitive or a
constructor (an unfeatured nullary or categorial operator), a constructor whose
kind leaves the category sort, two functor views that disagree on slots, and a
category name declared by two sources.

## Other fields

| Field | Default | What it declares |
|---|---|---|
| `name` | `unnamed` | The signature's name, used in messages and in the linked store's signature name. |
| `sorts`, [`category_sort`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureSpec) | `("Cat",)`, `Cat` | The sorts the signature declares and the one categories belong to. Every sort other than the category sort must also be declared by the language the grammar is compiled against. |
| `vocabularies` | `{}` | Explicit constant rows per sort, beside the primitives. |
| `open_sorts`, `classifiers` | `()`, `{}` | Open sorts and a classifier table, each merged with the language signature's own when the grammar is linked. |
| `allows_vacuous_binding`, `holes` | `true`, `false` | Whether binding may be vacuous and whether the signature admits holes. The linked store allows vacuous binding only if both the language and the grammar do, and admits holes if either does. |

An operator row may also declare [`arg_sorts`](../../reference/configuration/grammar.md#lofi.spec.grammar.OperatorSpec), `binding` sites and a `family`
for a non-categorial operator; these extend the [term algebra](../../explanation/logical-forms-as-types.md#signatures-and-terms) rather than the
category inventory.

## Related

- [Choose a formalism](choose-a-formalism.md) for formalisms that generate the
  signature for you.
- [Map between signatures](map-between-signatures.md) for erasing features and
  mapping categories onto [semantic types](../../explanation/logical-forms-as-types.md#semantic-typing).
- [Derivations as proofs](../../explanation/derivations-as-proofs.md) for what
  a category and its constructors stand for.

## References

\bibliography
