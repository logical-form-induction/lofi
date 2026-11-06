# Control semantic distributions

A synthetic design controls its logical forms at two levels. A structured
meaning source draws the semantic lexicon itself: how many forms each logical
class has, how token mass falls across classes and within them, which classes
each category may take, and how many meanings a form carries. Complexity and
scope targets then filter the drawn strings by measurements of their forms.
This guide declares each control, compares two semantic profiles from one
experiment, and aims an inference draw at a complexity band.

## Draw a structured semantic lexicon

An interpretation's `meanings.source` is `lexical` when the lexicon's entries
spell their meanings, and `structured` when the meanings are drawn. The L3
source interpretation,
`experiments/l3/data/datasets/train/dataset/source/interpretation/l3-interpretation.yaml`,
is structured:

```yaml
meanings:
  source:
    kind: structured
    structure:
      joint:
        spelling: parametric
        members: [predicate, constant]
        largest_class: 6
        class_sizes: {kind: uniform}
        coupling: 0.0
      rank_law: {kind: uniform}
      arities:
        arities:
          - {arity: 1, share: 0.5, rank_law: {kind: uniform}}
          - {arity: 2, share: 0.5, rank_law: {kind: uniform}}
      assignment:
        kind: tabular
        rows:
          - {category: NP, classes: [{logical_class: constant}]}
          - {category: "S\\NP", classes: [{logical_class: predicate, arity: 1}]}
          - {category: "S\\NP/NP", classes: [{logical_class: predicate, arity: 2}]}
      fragment: {order: first, quantifiers: [], connectives: [], functions: false, equality: false}
      seed: 0
      polysemy: {spelling: parametric, mean: 1.0, dispersion: 0.0, maximum: 3, placement: 0.0}
```

Reading the structure:

| Field | What it says |
| --- | --- |
| `joint` | The logical classes, their sizes and their shares of token mass. `parametric` gives the first member `largest_class` forms and shrinks the rest by `class_sizes`; `tabular` lists each class's `size` and `share`. |
| `coupling` | The exponent relating a class's token share to its size: at zero every form has the same mean mass whatever its class, a positive coupling gives the forms of small classes more mass, and a negative one gives the forms of large classes more. |
| `rank_law` | How token mass falls along the forms of a class other than the predicate class: `uniform` or `power-law` with an `exponent`. |
| `arities` | How the predicate forms split across arities, with a rank law within each. Required when the joint has a predicate class. |
| `assignment` | Which classes each category may draw: `tabular` rows per category, or `independent` of the category. |
| `fragment` | The logic forms are drawn within: the quantifiers and connectives the quantifier and connective classes may use, and whether functions and equality are admitted. An omitted field takes the language's own. |
| `seed` | The seed of the draw. Equal seeds over equal lexicons draw equal assignments. |
| `constants` | The total inventory size, or nothing to use the joint's sizes. |
| `polysemy` | The law over meanings per form-category item and where extra meanings fall in the frequency ranking. |

The logical classes are the `lofi.lf_class` entries: `quantifier`,
`connective`, `predicate`, `identity` and `constant`. A structure may also be
loaded by name with `preset`, which reads `semantic_lexicons/<name>.json` from
the configuration search path; LoFI ships `natural`.

For a parametric joint, let class $c$ contain $n_c>0$ symbols and let
$\kappa$ be `coupling`. Before category-specific assignment restrictions,
its normalized token share is

$$
p_C(c)=\frac{n_c^{1-\kappa}}{\sum_{d\in\mathcal C}n_d^{1-\kappa}},
\qquad
\frac{p_C(c)}{n_c}
=\frac{n_c^{-\kappa}}{\sum_{d\in\mathcal C}n_d^{1-\kappa}}.
$$

The second expression is the class's mean share per symbol. Thus
$\kappa=0$ equalizes that mean, while $\kappa=1$ equalizes class shares.
Within-class frequency need not be uniform: a power law with exponent
$\alpha$ assigns symbol rank $r$ probability proportional to $r^{-\alpha}$
within its eligible inventory.

A category $A$ draws an assignment row $j$ with declared weight $a_{Aj}$ and
class $c_j$. Its first row draw has PMF

$$
p_J(j\mid A)
=\frac{a_{Aj}p_C(c_j)}{\sum_{h\in\mathcal J_A}a_{Ah}p_C(c_h)}.
$$

It then draws a symbol from that row's class, restricting to the required
arity if one is declared, and renormalizes the eligible symbol weights.
The `independent` assignment supplies one unit-weight row per class, giving
$p_C(c\mid A)=p_C(c)$. With polysemy, meanings are drawn without replacement:
already selected symbols are excluded and exhausted assignment rows are
removed before subsequent draws. Thus the displayed PMFs describe the
initial draw, not independent repeated draws of every meaning.

The requested class shares, conditional assignment laws, realized lexical
assignment, and corpus frequencies are separate objects. Restricting a
category to one class can change the realized marginal, and finite draws
need not reproduce its expectation. The `polysemy` count law uses the same
bounded mixture construction as [lexical ambiguity](../grammars/declare-a-lexicon.md#draw-a-structured-lexicon).

## Compare semantic profiles

`experiments/l3/distribution.yaml` sweeps the joint over twelve profiles
packaged beside the run document, named for inventory size (`i6`, `i12`),
class-size law (`du` uniform, `dp` power-law) and coupling (`cn` negative,
`c0` zero, `cp` positive). Select two of them by group override and read the
semantic report of each draw:

```python
from pathlib import Path

from lofi.data.synthetic.spec.inference import PLUGIN
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
policy = ParsePolicy(Retain.FOREST, Reachability.ALL)


def draw(run):
    declaration = run.data.datasets["train"].dataset
    return PLUGIN.lower(PLUGIN.check(declaration), policy).inference


joint = (
    "data/datasets/train/dataset/source/interpretation/meanings/source/structure/joint"
)

for profile in ("i6-du-c0", "i12-dp-cp"):
    run = codec(Path("experiments/l3/l3-run.yaml"), overrides=(f"{joint}={profile}",))
    semantic = draw(run).semantic
    print(
        profile,
        dict(semantic.lexicon.sizes),
        dict(semantic.symbols_used),
        semantic.concordance,
    )
```

```text
i6-du-c0 {'predicate': 6, 'constant': 6} {'predicate': 4, 'constant': 2} 1.0
i12-dp-cp {'predicate': 12, 'constant': 8} {'predicate': 4, 'constant': 2} 1.0
```

The report separates what was declared from what the corpus used.
`lexicon.sizes` is the drawn inventory per class, `symbols_used` counts the
inventory symbols that occur in the corpus, and `concordance` is the
normalized mutual information of category and class, which is one here
because the tabular assignment fixes the class of each category and both
variables vary. At the 12 pairs this run draws, both profiles use the same
number of symbols. Equal support counts do not show that inventory size has
no effect: the identities, frequencies, assignments, or inference labels may
already differ. The report also gives
the per-class frequency histograms, the item shares and the meanings per form,
which is what L3 reads to separate inventory size from frequency.

Let $n_{ac}$ count the report's category-class occurrences, let
$n=\sum_{a,c}n_{ac}>0$, and set $\widehat p_{AC}(a,c)=n_{ac}/n$.
The empirical marginal PMFs follow by summation. Using natural logarithms,
the report computes

$$
\begin{aligned}
H(A)&=-\sum_a\widehat p_A(a)\log\widehat p_A(a),\\
H(C)&=-\sum_c\widehat p_C(c)\log\widehat p_C(c),\\
I(A;C)&=\sum_{a,c:\,n_{ac}>0}\widehat p_{AC}(a,c)
\log\frac{\widehat p_{AC}(a,c)}{\widehat p_A(a)\widehat p_C(c)},\\
\operatorname{concordance}(A,C)&=\frac{I(A;C)}{\min\{H(A),H(C)\}}.
\end{aligned}
$$

Entropy and mutual information have their standard information-theoretic
meaning \citep{shannon1948}; division by the smaller entropy is this
report's specified normalization. Concordance is undefined, reported as
`None`, when there are no observations or either variable has zero entropy.
When both entropies are positive, a deterministic mapping in either
direction gives concordance one. This statistic describes the recorded
category-class counts, not the size of the declared symbol inventory.

## Aim at a complexity band

An inference draw's `complexity` block rejects candidate pairs until each
side's form falls inside declared bands. Controls are keyed by
`lofi.lf_complexity` entry:

| Control | Measures |
| --- | --- |
| `connective_count` | language-declared connective occurrences |
| `operator_nesting` | the greatest nesting of declared connective roles |
| `predicate_count` | declared predicate occurrences |
| `quantifier_depth` | the greatest quantifier nesting |
| `variable_count` | binder occurrences plus distinct free variables |
| `meaning_depth` | the greatest edge depth of the meaning term |

`premise` and `hypothesis` each take a `controls` table; `overlap` bands the
number of predicate symbols the two sides share. Aim the L4 draw at premises
with exactly one connective and pairs that share exactly one predicate:

```python
controlled = codec(
    Path("experiments/l4/l4-run.yaml"),
    overrides=(
        'data.datasets.train.dataset.complexity={"premise": {"controls": {"connective_count": {"value": 1, "tolerance": 0}}}, "overlap": {"value": 1, "tolerance": 0}}',
    ),
)
complexity = draw(controlled).complexity
print(complexity.premise["connective_count"], complexity.hypothesis["connective_count"])
print(complexity.overlap, complexity.rejected)
```

```text
Histogram({1: 12}) Histogram({0: 1, 1: 11})
Histogram({1: 12}) 219
```

Every premise has one connective and every pair shares one predicate; the
hypothesis side, which is not controlled, is reported but not banded. The
report counts the candidate pairs rejected to meet the bands, so a band that
is expensive to meet is visible before training.

For a complexity measurement $c(t)$ with target $v$ and tolerance
$\epsilon\geq0$, acceptance requires $|c(t)-v|\leq\epsilon$.
If $S(t)$ is the set of predicate symbols in $t$, the overlap measurement is
$|S(\varphi)\cap S(\psi)|$. All declared bands must hold together.
Conditioning on these acceptances changes the distribution of sampled pairs;
it does not merely annotate an otherwise unchanged draw. The
[synthetic-design definition](../../explanation/synthetic-designs.md)
formalizes the resulting rejection-sampling law.

## Count scope readings

A `synthetic` or `synthetic-inference` declaration's `scope` block measures
how many readings its bounded equivalence procedure distinguishes:

| Field | What it says |
| --- | --- |
| `readings` | A `{value, tolerance}` band on readings per string, or nothing to measure without banding. |
| `derivations` | The most derivations one string's readings are read from. A string with more is drawn again. |
| `attempts` | How many draws one accepted string may take. |
| `undecided` | Whether an equivalence the prover cannot decide keeps readings apart (`distinct`, the default) or counts them as one. |

Scope requires an interpretation, since readings are read from forms, and each
equivalence decision runs under the interpretation's clock. The
corpus's `scope` report lists the derivations and readings counted for each
item. `experiments/l3/ambiguity.yaml` crosses a scope-reading target with
lexical polysemy.

The ideal equivalence relation is

$$
\varphi\equiv_\Gamma\psi
\quad\Longleftrightarrow\quad
\Gamma,\varphi\vdash\psi\ \text{and}\ \Gamma,\psi\vdash\varphi.
$$

With exhaustive derivations and a complete decision procedure, counting
readings would count the quotient of the derived forms by this relation.
The implementation instead compares each new form with previously retained
representatives under one shared per-string prover budget. It joins a form
when both entailments are proved; undecided comparisons follow `undecided`.
Thus `distinct` can split a true equivalence class, whereas `equivalent` can
merge logically distinct forms. Enumeration bounds, abstentions, and
representative order can affect the reported count. Read it together with
the number of interpreted derivations and undecided comparisons, rather than
as an unconditional count of semantic equivalence classes.

## Refusals

- A complexity control that is not installed is refused, with the installed
  names:

```python
from lofi.spec import ConfigValueError

try:
    codec(
        Path("experiments/l4/l4-run.yaml"),
        overrides=(
            'data.datasets.train.dataset.complexity={"premise": {"controls": {"clause_count": {"value": 1, "tolerance": 0}}}}',
        ),
    )
except ConfigValueError as refused:
    print(str(refused)[:240])
```

```text
data.datasets.train.dataset.complexity.premise.controls was refused: complexity controls are not installed: ['clause_count']; installed controls are ['connective_count', 'meaning_depth', 'operator_nesting', 'predicate_count', 'quantifier_de
```

- A `controls` table that names no control, a negative tolerance, or a
  `scope` block on a corpus with no interpretation is refused.
- A class size that is not positive, a class share that is not positive, or a
  non-positive `constants` count is refused.
- A band that no candidate meets within `attempts` stops the draw.

[Source and learner declarations](../../explanation/synthetic-designs.md)
explains how these controls become factors of an experiment.

## References

\bibliography
