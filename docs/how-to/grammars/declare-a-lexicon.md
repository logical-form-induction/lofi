# Declare a lexicon

A grammar's `lexicon` assigns categories to surface forms, and its `kind`
selects how. An `inline` lexicon lists every entry. A `structured` lexicon
draws entries over named categories from three separately declared laws: a category
marginal, an ambiguity law and a rank law over form frequency. A `generated`
lexicon draws the same way over categories it grows from the grammar's goal
and rules, so that no category is named. This guide declares each kind,
builds it inside the packaged `directed` grammar, and lists the refusals.

The declarations are [`InlineLexiconSpec`, `LexiconStructure` and
`GeneratedLexiconSpec`](../../reference/configuration/grammar.md). Every
example below builds its lexicon with one helper, which replaces the
lexicon of `directed`, compiles the grammar, and prints each form's total mass
and the probability it gives each of its categories:

```python
from copy import deepcopy
import json
from math import exp
from pathlib import Path

import yaml

from lofi.spec import SpecLoader, config_search_path, load_plugins
from lofi.spec.grammar import GrammarIso, GrammarSpec, declare_grammar
from lofi.spec.language import registered_language

load_plugins()
packaged = Path("lofi/configs/data/datasets/train/dataset/grammar")
loader = SpecLoader(search_path=config_search_path(Path.cwd()))
directed = json.loads(
    loader.compose_traced(
        packaged / "directed.yaml", schema=GrammarSpec
    ).spec.model_dump_json()
)
language = registered_language("propositional")


def show(lexicon):
    document = {**deepcopy(directed), "lexicon": lexicon}
    grammar = declare_grammar(GrammarSpec.model_validate(document))
    assembly = GrammarIso().forward(grammar, language)
    for form in assembly.lexicon.forms:
        categories = [
            (
                assembly.syntax.render(assembly.store, entry.category),
                round(exp(entry.weight), 3),
            )
            for entry in assembly.lexicon.entries(form)
        ]
        print(form, round(exp(assembly.lexicon.log_mass(form)), 3), categories)
```

The compiled lexicon normalizes each form's entry weights into a distribution
over its categories; `log_mass` returns the total the form's entries were
declared or drawn with.

For a surface form $w$, let $\mathcal E_w$ be its entries and $\ell_e$ an
entry's declared log weight. Compilation computes

$$
z_w=\sum_{e\in\mathcal E_w}\exp(\ell_e),\qquad
\operatorname{log\_mass}(w)=\log z_w,\qquad
p(e\mid w)=\frac{\exp(\ell_e)}{z_w}.
$$

Thus `log_mass` is a log mass, not necessarily the logarithm of a normalized
form PMF: the inline example assigns mass two to each ambiguous form.
Normalizing over the finite vocabulary $\mathcal V$ would give
$p(w)=z_w/\sum_{v\in\mathcal V}z_v$. Entries with the same category and
meaning are merged by summing their masses before normalization. If several
meanings share a category $A$, its conditional probability is the sum
$p(A\mid w)=\sum_{e:\operatorname{category}(e)=A}p(e\mid w)$.

## List entries inline

An inline lexicon is a list of `entries`, each a `form`, a `category` in the
signature's syntax, an optional `weight` (a finite log weight, `0.0` by
default) and an optional `meaning`, a logical form the [interpretation](declare-an-interpretation.md)
reads. A form listed twice is ambiguous. The packaged fragment
`grammar/lexicon/ambiguous.yaml` gives four of its five forms two categories:

```python
show(yaml.safe_load((packaged / "lexicon/ambiguous.yaml").read_text()))
```

```text
w0 2.0 [('NP', 0.5), ('N', 0.5)]
w1 2.0 [('S\\NP', 0.5), ('S\\NP/NP', 0.5)]
w2 2.0 [('S\\NP/NP', 0.5), ('S\\NP/NP/NP', 0.5)]
w3 2.0 [('NP/N', 0.5), ('N/N', 0.5)]
w4 1.0 [('N', 1.0)]
```

A run document selects it in the `defaults` list under the slot
`data.datasets.train.dataset.grammar.lexicon`; `unambiguous.yaml` is its
one-category-per-form counterpart. Because the lexicon is a group slot, a
sweep can vary it as an axis: `experiments/s5/anchoring.yaml` crosses three
lexicon fragments of the S5 source grammar, each of which sets only
`coupling`.

## Draw a structured lexicon

A structured lexicon declares how many `forms` it holds, a `joint` giving each
named category its share of token mass, an `ambiguity` law over how many
categories each form bears, a `rank_law` over the forms' frequency ranks, and
a `seed`:

```python
structured = {
    "kind": "structured",
    "forms": 6,
    "joint": {
        "spelling": "tabular",
        "rows": [
            {"category": "NP", "share": 3.0},
            {"category": "S\\NP", "share": 2.0},
            {"category": "S\\NP/NP", "share": 1.0},
        ],
    },
    "ambiguity": {
        "spelling": "histogram",
        "maximum": 2,
        "placement": 0.0,
        "buckets": [{"categories": 1, "share": 2.0}, {"categories": 2, "share": 1.0}],
    },
    "rank_law": {"kind": "power-law", "exponent": 1.0},
    "seed": 7,
}
show(structured)
```

```text
w0 0.102 [('NP', 0.234), ('S\\NP/NP', 0.766)]
w1 0.408 [('NP', 1.0)]
w2 0.136 [('S\\NP/NP', 0.65), ('S\\NP', 0.35)]
w3 0.082 [('S\\NP', 1.0)]
w4 0.204 [('S\\NP', 1.0)]
w5 0.068 [('NP', 1.0)]
```

The histogram gives one form in three two categories, the power law gives
the form of rank $r$ mass in proportion to $r^{-1}$, and the cells are fitted so
that the target token masses of $\mathrm{NP}$, $\mathrm S\backslash\mathrm{NP}$,
and $(\mathrm S\backslash\mathrm{NP})/\mathrm{NP}$ have ratio $3:2:1$.
The three laws have these spellings:

| Law | Spelling | Fields |
|---|---|---|
| `joint` | `tabular` | `rows` of `category` and relative `share`. |
| `joint` | `parametric` | `categories` and a `category_shares` rank law over them. |
| `ambiguity` | `parametric` | `mean` $\mu\in[1,m]$ and `dispersion` $\delta\in[0,1]$, with $m$ the `maximum`: at zero dispersion the mass sits on the integers bracketing the mean, at one on 1 and `maximum`, with the same mean. |
| `ambiguity` | `histogram` | `buckets` of `categories` per form and relative `share` of forms. |
| `rank_law` | `uniform` | None. Every rank weighs alike. |
| `rank_law` | `power-law` | `exponent` $\alpha$: rank $r$ weighs in proportion to $r^{-\alpha}$. |

Both ambiguity spellings also take `maximum`, a hard bound on categories per
form, and `placement` in [-1, 1], the signed association between a form's
ambiguity and its frequency rank: negative values put ambiguous forms lower in
the frequency order, zero places them independently, and positive values put
them higher.

For $n$ ranks and exponent $\alpha$, the normalized rank PMF is

$$
p_R(r)=\frac{r^{-\alpha}}{\sum_{j=1}^{n}j^{-\alpha}},
\qquad r\in\{1,\ldots,n\}.
$$

Uniform ranks have $p_R(r)=1/n$. The helper returns weights with mean one;
the draw divides by $n$ to obtain this PMF before assigning ranks to forms.

For parametric ambiguity, let $K\in\{1,\ldots,m\}$ count a form's
categories, with requested mean $\mu$ and dispersion $\delta$. Write
$a=\lfloor\mu\rfloor$ and $b=\lceil\mu\rceil$. When $a<b$, the
minimum-variance PMF has $q(a)=b-\mu$ and $q(b)=\mu-a$; when $a=b$, it
places all mass at $a$. For $m>1$, the endpoint PMF has
$r(1)=(m-\mu)/(m-1)$ and $r(m)=(\mu-1)/(m-1)$. All unlisted probabilities
are zero. The declared law is

$$
p_K(k)=(1-\delta)q(k)+\delta r(k),\qquad
\mathbb{E}[K]=\mu.
$$

When $m=1$, both component PMFs put all mass at one. The implementation
apportions finitely many forms to these probabilities and adjusts the total
category-slot count toward $\operatorname{round}(n\mu)$; rounding and the
available support can prevent exact realization of the target mean.
`placement` controls the assignment of frequency ranks to those realized
ambiguity counts, rather than changing $p_K$.

The target joint masses $q_{wA}$ form a nonnegative matrix $\mathbf Q$ whose
row and column marginals are the requested form masses and category shares:

$$
\sum_A q_{wA}=p(w),\qquad \sum_w q_{wA}=p(A),\qquad
\bigl|\{A:q_{wA}>0\}\bigr|=k_w.
$$

The support is selected subject to each realized ambiguity $k_w$, and cell
masses are fitted on that support. These equations state fitting targets;
the finite alternating-scaling procedure need not satisfy both marginals
exactly after its last update. Inspect the compiled and sampled reports
rather than treating requested marginals as measured ones.

## Generate the categories

A generated lexicon names no category. It grows an inventory of exactly
`categories` lexical categories from the goal through the inverse of the
declared rules, then draws `forms` entries over them as a structured lexicon
does, with `category_shares` as the rank law over the generated categories in
the order they were admitted and `coupling` as the target association between
a category's class size and the mean token mass of its forms:

```python
generated = {
    "kind": "generated",
    "forms": 6,
    "categories": 4,
    "category_shares": {"kind": "uniform"},
    "coupling": 0.0,
    "ambiguity": {
        "spelling": "parametric",
        "mean": 1.0,
        "dispersion": 0.0,
        "maximum": 1,
        "placement": 0.0,
    },
    "rank_law": {"kind": "uniform"},
    "seed": 3,
    "max_depth": 2,
}
show(generated)
```

```text
w0 0.167 [('N', 1.0)]
w1 0.167 [('S/N\\N', 1.0)]
w2 0.167 [('S/N', 1.0)]
w3 0.167 [('S/(S/N)', 1.0)]
w4 0.167 [('S/N', 1.0)]
w5 0.167 [('N', 1.0)]
```

With category shares $p(A)$ held fixed, generated-lexicon coupling
$\kappa\ne1$ targets class sizes proportional to
$p(A)^{1/(1-\kappa)}$. These targets are apportioned to integers, bounded by
one and the number of forms, and constrained by available form masses. At
$\kappa=1$ the implementation uses the finite exponent equal to the number
of forms times the number of categories. This is an allocation convention;
the realized association remains a reported property of the built lexicon.

Every generated category is a leaf of a derivation of the goal whose other
leaves are lexical too, so a corpus drawn from the grammar can use each one.
The generation is bounded by these fields:

| Field | Default | Bounds |
|---|---|---|
| `max_depth` | 3 | How deeply constructors nest in a lexical category; `(S/NP)/NP` has depth two. |
| `max_arguments` | none | A category's functor-spine argument count, as `{value: n}`. |
| `max_order` | none | A category's order under the signature's functor views, as `{value: n}`. |
| `derivation_depth` | 4 | Steps from the root of a generating derivation to a leaf. |
| `expansion` | 0.5 | The chance a node between the root and the depth bound is expanded rather than left a leaf. |
| `patience` | 1000 | Consecutive refused derivations before the declaration is refused. |

The premises generation draws from are the signature's primitives less those
it reserves for its rules, such as the conjunction markers, so changing the
signature's `primitives` changes what the inventory is built from. The
[synthetic-generated example](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-generated.yaml)
declares a generated lexicon over primitives `S`, `P1`, `P2` and `P3`.

## Seed a drawn lexicon

The `seed` of a structured or generated lexicon is a nonnegative integer or
`derived`. A fixed seed draws the same lexicon in every point; `derived` takes
the seed each point derives for the lexicon's path in the run document, so the
points of a sweep draw different lexicons from one declaration. Equal
declarations over equal grammars with equal seeds build equal lexicons.

## How a lexicon is refused

```python
def refuse(lexicon):
    try:
        GrammarSpec.model_validate({**deepcopy(directed), "lexicon": lexicon})
    except Exception as refused:
        print(str(refused).splitlines()[1].strip())


refuse(
    {**generated, "ambiguity": {**generated["ambiguity"], "maximum": 6, "mean": 2.0}}
)
refuse({**structured, "forms": 1})
refuse({"kind": "inline", "entries": []})
```

```text
lexicon: ambiguity cannot exceed the category inventory [axiom_failed]
lexicon: the ambiguity law supplies 1 category slots for 3 declared categories [validator_error]
lexicon: an inline lexicon needs an entry [axiom_failed]
```

A structured or generated lexicon whose ambiguity law supplies fewer category
slots than there are categories is refused at validation, since some category
would bear no form. A generated lexicon whose rules stop yielding new
categories within its bounds before the inventory is full is refused when the
grammar is compiled.

## Related

- [Generate a synthetic corpus](../data/generate-a-synthetic-corpus.md) for
  drawing sentences from the grammar the lexicon belongs to.
- [Synthetic designs](../../explanation/synthetic-designs.md) for why the
  category, ambiguity and frequency laws are declared independently.
