# Declare a logical form language

A [`LanguageDecl`](../../reference/configuration/language.md#lofi.spec.language.LanguageDecl) declares the
higher-order signature [logical forms](../../explanation/logical-forms-as-types.md#semantic-typing) are written in: its atomic semantic
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

A grammar describes how words combine; a [logical form language](../../explanation/logical-forms-as-types.md#semantic-typing) describes the
expressions used to state their meanings. In a hypothetical interpretation,
`ada` names an individual of type $e$, `sleep` has type $e\to t$, and
`sleep(ada)` has type $t$. The function type says what input is required and
what kind of result it produces. It does not say whether the resulting claim
is true. A [binder](../../explanation/reading-the-notation.md#lambda-expressions-and-substitution), such as a quantifier or lambda, introduces a variable with
a specified scope.

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
CCGbank, PMB and collected corpus datasets accept a registered name in place
of a declaration, as in `language: neo-davidsonian`.

## Compose a packaged language

The configuration library holds its language fragments in
[`lofi/configs/data/datasets/train/dataset/language/`](https://github.com/logical-form-induction/lofi/tree/main/lofi/configs/data/datasets/train/dataset/language/), each built from shared
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

[`experiments/s1/s1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/s1-run.yaml) does this. Any of the fragments can also be
composed on its own:

```python
from pathlib import Path

from lofi.spec import SpecLoader
from lofi.spec.language import LanguageDecl

loader = SpecLoader()
fragment = Path("lofi/configs/data/datasets/train/dataset/language/propositional.yaml")
propositional = loader.compose_traced(fragment, schema=LanguageDecl).spec
print(propositional.name, [constant.name for constant in propositional.constants])

plurals = loader.compose_traced(
    fragment.with_name("plurals.yaml"), schema=LanguageDecl
).spec
print(
    plurals.name, plurals.logic.order, plurals.logic.prover.kind, plurals.logic.domains
)
print(
    [
        (constant.name, constant.semantics)
        for constant in plurals.constants
        if constant.semantics
    ]
)
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

## Semantics of the specialized fragments

The [type and term rules](../../explanation/logical-forms-as-types.md) give
these declarations their shared syntax. The following equations specify
additional interpretations or postulates; a fragment name alone does not
supply every assumption used in the associated linguistic literature.

### Choice

Think of a predicate as identifying members of a domain that satisfy a
condition. A choice operation returns a member satisfying that condition when
there is one. If nobody satisfies it, the choice condition below does not
identify a preferred individual. This empty case is why the implementation's
completion rule must be stated separately from the logical condition.

For each instantiated type $\alpha$, the choice constant has type
$\varepsilon_\alpha:(\alpha\to t)\to\alpha$. On a nonempty domain
$D_\alpha$, its defining condition is

$$
(\exists x\in D_\alpha\;P(x))\;\Longrightarrow\;
P(\varepsilon_\alpha(P)).
$$

Thus a nonempty predicate extension supplies a witness. For the empty
extension, the condition imposes no restriction beyond membership in
$D_\alpha$. This is Hilbert's choice condition \citep{hilbert-bernays1970}.
LoFI’s finite choice resource requires known crisp predicate facts, validates
explicitly supplied choices and
otherwise selects the first satisfying member in domain order, or the first
domain member when there is none. This deterministic completion is not a
uniform draw from the predicate extension. The prover translation uses the
higher-order choice binder; not every first-order backend supports it.
### Dynamic meanings

A [dynamic meaning](#dynamic-meanings) relates an incoming discourse state to possible outgoing
states. Combining two meanings asks whether the first can reach a state from
which the second can continue. The intermediate state in the equation below
connects those two steps. In this fragment, introducing an individual can make
it available to a later pronoun; the declared state mechanism determines how
much discourse information can be retained.

Let $D_s$ be the state domain. A dynamic proposition has type
$s\to s\to t$ and denotes a relation on $D_s$. The packaged `d_and` is
relational composition, and `d_exists` quantifies an individual argument:

$$
\begin{aligned}
(P;Q)(i,o)&\;\Longleftrightarrow\;
\exists m\in D_s\,[P(i,m)\land Q(m,o)],\\
(\exists^{\mathrm{dyn}}x\,P)(i,o)&\;\Longleftrightarrow\;
\exists x\in D_e\,P(x,i,o).
\end{aligned}
$$

Here $i,m,o$ denote input, intermediate, and output states, and $P,Q$ are
relations between states. Relational update semantics is
central to dynamic predicate logic \citep{groenendijk-stokhof1991}.
LoFI's fragment uses a single retrievable referent, with functions
$\operatorname{assign}:D_e\times D_s\to D_s$ and
$\operatorname{referent}:D_s\to D_e$ constrained by
$\operatorname{referent}(\operatorname{assign}(x,i))=x$.
For unary individual predicates $R,U$ and a binary state relation $P$,
its remaining definitions are

$$
\begin{aligned}
\operatorname{introduce}(R,U)(i,o)
&\Longleftrightarrow\exists x\,[R(x)\land U(x)\land
 o=\operatorname{assign}(x,i)],\\
\operatorname{pronoun}(U)(i,o)
&\Longleftrightarrow U(\operatorname{referent}(i))\land o=i,\\
\operatorname{close}(P)&\Longleftrightarrow\exists i\,\exists o\,P(i,o).
\end{aligned}
$$

Each state makes one referent retrievable through these functions. The
packaged finite domains each have two members, bounding the states and
individuals considered during model search.
### Plural individuals

For a hypothetical atomic set containing Ada and Bea, the nonempty plural
individuals are Ada, Bea, and their sum. Joining Ada with the sum does not add
a second copy of Ada: join is idempotent. A predicate such as `laughed` can be
postulated to hold of the sum exactly when it holds of both individuals.
A collective predicate such as `gathered` need not have that behavior.

The plural fragment supplies a join-semilattice $(D_e,\oplus)$, with
commutativity, associativity, and idempotence:

$$
x\oplus y=y\oplus x,\qquad
(x\oplus y)\oplus z=x\oplus(y\oplus z),\qquad x\oplus x=x.
$$

Its parthood and atomicity predicates are

$$
x\leq y\;\Longleftrightarrow\;x\oplus y=y,\qquad
\operatorname{atom}(x)\;\Longleftrightarrow\;
\forall y\,[y\leq x\Rightarrow y=x].
$$

The atomicity definition uses minimal elements of the domain of nonempty plural
individuals, with singleton sets as the minimal nonempty values. This algebraic
approach to plurality follows \citet{link1983}. The finite runtime realizes this as $\mathcal{P}(A)\setminus\{\varnothing\}$
for an atomic set $A$, with union as join, inclusion as parthood, and singleton
sets as atoms. Thus its domain size is $2^{|A|}-1$; the packaged size of
three represents two atoms and their sum. It is more specific than an
arbitrary finite join-semilattice. LoFI separately postulates
$\operatorname{laughed}(x\oplus y)\Longleftrightarrow
\operatorname{laughed}(x)\land\operatorname{laughed}(y)$.
This distributivity postulate applies specifically to `laughed`; the
collective predicate `gathered` has no corresponding axiom. Model search
covers the declared domain of three plural individuals, leaving larger
models outside its scope.

### Events and generalized quantifiers

The registered `neo-davidsonian` language has an event sort $v$ in addition
to $e$ and $t$. An event analysis can represent a sentence with a formula such
as $\exists v\,[\operatorname{sleep}(v)\land\operatorname{agent}(v,a)]$,
provided its lexical declaration supplies those typed predicates
\citep{parsons1990}. The sort alone supplies neither thematic role axioms
nor tense or aspect. The similarly named packaged YAML fragment currently
inherits only the shared individual and truth sorts; inspect the resolved
declaration rather than inferring its inventory from the name.

For a hypothetical majority judgment, suppose three of five dogs sleep.
The restrictor is the set of dogs; the scope is the set of sleepers. Three
members of the restrictor satisfy the scope and two do not, so strict majority
holds. Both counts are restricted to dogs.

For a finite individual domain and restrictor and scope extensions
$A,B\subseteq D_e$, the inline `most` semantics is strict majority:

$$
\operatorname{most}(A,B)\;\Longleftrightarrow\;
|A\cap B|>|A\setminus B|.
$$

This is a generalized quantifier of type $(e\to t)\to(e\to t)\to t$
\citep{barwise-cooper1981}. An empty restrictor makes it false. The finite
plugin requires known crisp facts on the same finite domain.

## Declare a language inline

Declare a language with a generalized quantifier whose semantics come from the
`most` plugin:

```python
from lofi.spec.language import LanguageDecl

declared = {
    "name": "generalized-quantifiers",
    "atomic_types": [{"name": "e"}, {"name": "t", "kind": "proposition"}],
    "constructors": [
        {
            "name": "->",
            "kind": ["Ty", "Ty", "Ty"],
            "notation": {
                "style": "infix",
                "spelling": "->",
                "precedence": 1,
                "associativity": "right",
            },
        },
    ],
    "constants": [
        {
            "name": "not",
            "type": "t -> t",
            "classes": ["sentential", "negation"],
            "notation": {
                "kind": "prefix",
                "ascii": "~",
                "unicode": "¬",
                "precedence": 6,
            },
        },
        {
            "name": "most",
            "type": "(e -> t) -> (e -> t) -> t",
            "classes": ["quantifier"],
            "semantics": "most",
        },
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
| [`atomic_types`](../../reference/configuration/language.md#lofi.spec.language.LanguageDecl) | Each atomic type's `name` and resource `kind`, `individual` by default. The registered neo-Davidsonian language declares `v` with kind `event`; its packaged shared YAML declares `e` and `t`, with `t` of kind `proposition`. |
| `constructors` | Type constructors over the type sort, read in their declared notation by the same reader as category text. A token ending in a letter, digit or underscore matches only as a whole word, so a type named `toe` is not read as the token `to`. The one named by `function_type` (`->` by default) must be binary and infix. |
| `constants` | Each constant's `name`, `type` (or `infer`, to infer it from its uses), the type variables it `quantifies`, its `classes`, a surface `notation` (`prefix`, `infix` or `binder`, each with an ASCII and a Unicode token), and either a `definition` as a closed term or a postulated `semantics` plugin. |
| `leaf_lexicon` | A source of further nonlogical constants, read once when the language is compiled. |
| `binders` | The binder `style`, `curry` (unannotated) or `church` (annotated lambdas). |
| `logic` | The logic's `order`, whether it has `functions` and `equality`, finite `domains` per type, meaning `postulates`, the variable inventory, the designated `negation`, the [`label_map`](../../reference/configuration/language.md#lofi.spec.language.LogicSpec), and the `prover` and `labeling` resources. |
| `bounds` | One finite representation bound per sort (`Ty`, `Tm` and `Cat`), used by the neural [term encoders](../models/configure-term-encoders-and-decoders.md#choose-an-encoder). |

The [theorem prover guide](../training/reward-with-a-theorem-prover.md)
covers the `prover` declarations.

## Read constants from a leaf lexicon

A `leaf_lexicon` of kind `path` reads constants from a JSON, TOML or YAML
document; a relative path is read from [`lofi/configs/leaf_lexicons/`](https://github.com/logical-form-induction/lofi/tree/main/lofi/configs/leaf_lexicons/), which
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

A plugin adds a logical role by registering a [`LogicalRole`](../../reference/lf.md#lofi.lf.LogicalRole) under
`lofi.logical_role`, and a constant semantics by registering a
[`LogicalConstantSemantics`](../../reference/lf.md#lofi.lf.LogicalConstantSemantics) under `lofi.logical_constant`; in both cases the
member's `name` must equal its entry point name. A whole language is
registered under `lofi.lf_language` as a function that returns its
[`LanguageDecl`](../../reference/configuration/language.md#lofi.spec.language.LanguageDecl). The [entry point reference](../../reference/entry-points.md)
lists the groups.

## Related

- [Declare an interpretation](declare-an-interpretation.md) to map a grammar's
  derivations into this language.
- [Categories, semantic types, and logical forms](../../explanation/logical-forms-as-types.md) for
  how categories, [semantic types](../../explanation/logical-forms-as-types.md#semantic-typing), and logical forms share a term representation.

## References

\bibliography
