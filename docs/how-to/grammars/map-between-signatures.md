# Map between signatures

LoFI relates one signature to another in three places. [Feature erasure](../../explanation/logical-forms-as-types.md#substitution-equality-and-matching) maps
each featured category to its core in the same signature, so `S[dcl]` and `S`
can match. A signature morphism erasure maps a [category signature](../../explanation/logical-forms-as-types.md#signatures-and-terms) onto the
type signature of a [logical form language](declare-a-logical-form-language.md#use-a-registered-language), so that every category has a
[semantic type](../../explanation/logical-forms-as-types.md#semantic-typing). And an evaluator's alignment maps a learner grammar's category
labels onto a source grammar's, so that a learner whose primitives are named
`P1` and `P2` can be scored against a source whose primitives are `NP` and
`N`. This guide declares each.
## Erase features

The three maps on this page answer different questions. [Feature erasure](../../explanation/logical-forms-as-types.md#substitution-equality-and-matching) asks
whether two syntactic categories agree after a distinction is forgotten.
[Category-to-type interpretation](../../explanation/logical-forms-as-types.md#category-to-type-interpretation) asks what kind of meaning a category carries.
[Evaluation alignment](../evaluation/evaluate-a-run.md) asks which reference label best corresponds to a learned
label. The same pair of labels can agree under one map and differ under another.

A featured category is declared in the signature with the core it
`erases_to`, as [Declare a category signature](declare-a-signature.md)
describes. The compiled store records each feature atom's core:

```python
from pathlib import Path

from lofi.spec import SpecLoader, config_search_path, load_plugins
from lofi.spec.grammar import GrammarIso, GrammarSpec, declare_grammar
from lofi.spec.language import registered_language

load_plugins()
path = Path("docs/examples/grammars-data/grammars/featured.yaml")
loader = SpecLoader(search_path=config_search_path(Path.cwd()))
grammar = declare_grammar(loader.compose_traced(path, schema=GrammarSpec).spec)
assembly = GrammarIso().forward(grammar, registered_language("propositional"))
signature = assembly.store.signature

for name in ("S[dcl]", "S[b]", "NP[conj]", "NP"):
    print(f"{name:9} {signature.constant_core('Cat', name)}")
```

```text
S[dcl]    S
S[b]      S
NP[conj]  NP
NP        NP
```

Whether rules see the features is the grammar's `matching` choice. Under
`identity`, `S[dcl]\NP` and `S\NP` are different categories; under
`feature-erasure`, a rule compares premises by their erased cores. A rule can
override the grammar's choice with its own `matching`.
[Choose a normal form, matching relation and goal](choose-normal-form-matching-goal.md#declare-a-matching-relation)
lists the relations.

Writing $F$ for feature erasure, matching compares

$$
A\sim_F B\quad\Longleftrightarrow\quad F(A)=F(B).
$$

Erasure acts recursively on category terms: it replaces featured atoms by
their declared cores and applies the same constructor to the erased
arguments. Thus $F(\mathrm S[\mathrm{dcl}]\backslash\mathrm{NP})
=\mathrm S\backslash\mathrm{NP}$. The equality on the right is equality
in the target [term algebra](../../explanation/logical-forms-as-types.md#signatures-and-terms). This definition explains why removing features
can admit rule instances that identity matching rejects.
## Map categories onto semantic types

For a hypothetical map with $E(\mathrm{NP})=e$ and $E(\mathrm S)=t$,
$E(\mathrm S\backslash\mathrm{NP})=e\to t$. Compute the argument's type
first and then the result's type. A forward slash with the same argument and
result has the same function type: semantic input and output remain, while the
requirement about which side supplies the phrase has been forgotten. This is
why the map can be useful without being one-to-one.

A [`SignatureMorphismErasureSpec`](../../reference/configuration/grammar.md#lofi.spec.grammar.SignatureMorphismErasureSpec)
declares the image of each category atom as a type, and the image of each
category constructor that is not a [functor view](declare-a-signature.md#declare-functor-views). A functor view's image is
derived: a category whose functor has result $X$ and argument $Y$ erases to
the function type $E(Y)\to E(X)$. Declare atom
images for the featured grammar and erase three categories:

```python
from lofi.spec.grammar import SignatureMorphismErasureSpec

erasure = SignatureMorphismErasureSpec(
    atoms={"S": "t", "NP": "e", "N": "⟨e,t⟩", "conj": "e"}
)
image = erasure.build(assembly, type_syntax="angle")

for category in ("S[dcl]\\NP", "(S[dcl]\\NP)/NP", "NP/N"):
    term = image(assembly.syntax.parse(assembly.store, category))
    print(f"{category:16} {assembly.language.print_type(term)}")
```

```text
S[dcl]\NP        e -> t
(S[dcl]\NP)/NP   e -> e -> t
NP/N             (e -> t) -> e
```

A featured category erases as its core does. Under `type_syntax: angle` an
image is written with angle brackets over the function type, so `⟨e,t⟩` denotes
$e\to t$; under `s-expression` it is written as a canonical term of the language's
signature. In a dataset this declaration is the `erasure` of the
[interpretation](declare-an-interpretation.md), and a learner grammar's
interpretation under `model.language` declares one of its own.

Restating a functor's image is refused, since the functor views already fix
it:

```python
from lofi.spec import DeclarationError

restating = SignatureMorphismErasureSpec(
    atoms={"S": "t", "NP": "e", "N": "⟨e,t⟩", "conj": "e"},
    constructors={"/": "a0 -> a1"},
)
try:
    restating.build(assembly, type_syntax="angle")
except DeclarationError as refused:
    print(refused)
```

```text
the erasure restates the functor constructors ('/',) of 'featured', whose images are derived from their functor roles
```

More generally, a constructor image $\tau_c(\alpha_1,\ldots,\alpha_k)$
extends the atom map recursively:

$$
E\bigl(c(A_1,\ldots,A_k)\bigr)
=\tau_c\bigl[E(A_1)/\alpha_1,\ldots,E(A_k)/\alpha_k\bigr].
$$

The configuration writes the positional metavariables $\alpha_i$ as `a0`,
`a1`, and so on. A category's depth thus determines a finite recursive
calculation of its image. This is the term level extension of the declared
signature mapping \citep{meinke-tucker1992}; distinct categories can share
an image.

### Declare constructor images

A signature whose constructors are not [functor views](declare-a-signature.md#declare-functor-views) needs an image for each.
The `tlg` formalism's expansion declares five prefix constructors and no
functor views, so an erasure over the packaged `nl-diamond` grammar that names
only atoms is refused, and one that names every constructor is accepted.
Each image is written over positional metavariables `a0`, `a1`, … for the
constructor's arguments in order:

```python
tlg_path = Path("lofi/configs/data/datasets/train/dataset/grammar/nl-diamond.yaml")
tlg = declare_grammar(loader.compose_traced(tlg_path, schema=GrammarSpec).spec)
tlg_assembly = GrammarIso().forward(tlg, registered_language("propositional"))

atoms_only = SignatureMorphismErasureSpec(atoms={"S": "t", "NP": "e"})
try:
    atoms_only.build(tlg_assembly, type_syntax="angle")
except DeclarationError as refused:
    print(refused)

images = SignatureMorphismErasureSpec(
    atoms={"S": "t", "NP": "e"},
    constructors={
        "/_linear": "a1 -> a0",
        "\\_linear": "a1 -> a0",
        "dot_linear": "a0",
        "diamond_linear": "a0",
        "box-down_linear": "a0",
    },
)
image = images.build(tlg_assembly, type_syntax="angle")
category = '("/_linear" ("/_linear" ("#const[Cat]" "S") ("#const[Cat]" "NP")) ("#const[Cat]" "NP"))'
print(
    tlg_assembly.language.print_type(
        image(tlg_assembly.syntax.parse(tlg_assembly.store, category))
    )
)
```

```text
constructor image coverage has missing ('/_linear', '\\_linear', 'box-down_linear', 'diamond_linear', 'dot_linear') and extra ()
e -> e -> t
```

The slashes take their result in slot `a0` and their argument in `a1`, and
the two modalities are mapped to the type of their argument. The language
declares no product type, so this example maps `dot_linear` to its first
factor; a language with a product constructor would map it to that. An image
for a constructor the signature does not declare is reported as `extra` and
refused the same way.

The product projection in this example deliberately loses the second
factor: $E(A\mathbin{\cdot}B)=E(A)$. The resulting type map preserves
only the first factor's semantic type. Lowering the interpretation separately
checks whether each rule's premises and conclusion admit an appropriate
semantic action.

## Score a learner grammar against a source grammar

Suppose, hypothetically, two induced labels both occur most often where the
reference annotation has `NP`. A many-to-one alignment can map both to `NP`.
A one-to-one alignment must assign distinct reference labels where assignments
are made, so their counts compete. The one-to-one restriction penalizes a learner that splits one reference
category across several induced labels. Fit the
map on one split and assess it on another so the evaluated answers have not
already determined the relabeling.

A learner declares its own grammar under `model.language`, either taken from
the data (`from: data`, as in the first tutorial) or written inline
(`from: inline`) with its own language, grammar and optional
`interpretation`. The S2 experiment's candidates under
[`experiments/s2/model/language/`](https://github.com/logical-form-induction/lofi/tree/main/experiments/s2/model/language/) are inline learner grammars over primitives
`S` and `P1` up to `P15`, crossed with a source grammar over `S` and `P1`.
The learner and source can use different label inventories. An evaluator
that compares them declares an `alignment`, fitted on one retained split
and frozen for another. The
packaged category side evaluator,
[`lofi/configs/evaluators/category-side/category-side.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/evaluators/category-side/category-side.yaml), declares:

```yaml
alignment:
  kind: one-to-one
  tie_break: least-gold-label
  fit_on: dev
  score_on: test
  random_floor: {replicates: 1000, level: 0.95}
  stability: {replicates: 1000, level: 0.95}
```

| `kind` | The correspondence fitted on `fit_on` |
|---|---|
| `identity` | Literal label identity, for a learner whose labels are the source's. |
| `many-to-one` | Each induced label is mapped to the gold label it co-occurs with most, independently. |
| `one-to-one` | An injective assignment between the two inventories with the largest total count. |

Let $\mathbf N$ be the contingency matrix on `fit_on`, with scalar entry
$n_{ab}$ counting co-occurrences of induced label $a$ and gold label $b$.
Many-to-one fitting chooses
$\pi(a)\in\operatorname*{arg\,max}_b n_{ab}$ independently for each $a$.
One-to-one fitting instead maximizes

$$
\sum_{a\in\operatorname{dom}(\pi)}n_{a,\pi(a)}
$$

over injective assignments between the inventories, using a partial
assignment if their sizes differ. Ties follow the declared policy. These
assignments maximize agreement between labels. Preservation of category
constructors and derivability requires the corresponding signature and rule
conditions in addition to label agreement.

`fit_on` and `score_on` must differ, so the alignment is never scored on the
split it was fitted on. The evaluator reports the aligned accuracy of
primitives and constructors beside a random relabeling floor computed over
`random_floor.replicates` relabelings. The `lexical-types` evaluator declares
the same `alignment` block; [Evaluate a run](../evaluation/evaluate-a-run.md)
covers the evaluators.

## Related

- [Declare an interpretation](declare-an-interpretation.md) for the erasure's
  place in a dataset.
- [Swappable assumptions](../../explanation/swappable-assumptions.md) for what
  a comparison between a learner and a source grammar licenses.

## References

\bibliography
