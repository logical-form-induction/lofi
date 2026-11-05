# Generate ill-typed forms

An `ill_typing` block corrupts the logical forms of a declared share of a
corpus's grammatical items into forms the type checker refuses, so a learner
sees negative logical forms beside the well-typed ones. This guide declares
the operators and proportion, draws an interpreted corpus that carries them,
and reads the typing report.

Ill-typed forms are corrupted from gold forms, so the corpus must declare an
interpretation that gives every grammatical item a form. A `synthetic`
corpus and a `synthetic-inference` dataset both take the block.

## Declare an ill-typing

The block has the same shape as a string corruption:

```yaml
ill_typing:
  proportion: {value: 0.25, draw: exact}
  operators:
    entries:
      - operator: {kind: argument-swap, edits: 1}
        weight: 1.0
  attempts: 64
  regime: labeled-negatives
```

| Field | What it says |
| --- | --- |
| `proportion` | The share of the grammatical items whose form is corrupted, and whether the count is `exact` or `bernoulli`. |
| `operators.entries` | Installed operators and their relative draw weights, each kind once. |
| `attempts` | How many candidate forms one injected form may take. Defaults to 64. |
| `regime` | How an ill-typed form is observed: `labeled-negatives` (the default) keeps its label, `positive-only` observes every one as well-typed, and `label-flips` observes each as well-typed with probability `noise`. |
| `noise` | The flip probability, which only the `label-flips` regime takes. |

Each operator is one entry in the `lofi.ill_typing` group and edits a form's
applications or bound variables:

| Operator | Edit |
| --- | --- |
| `argument-swap` | Exchange the function and argument of one application whose two sides differ. |
| `over-application` | Apply one application to its own argument again. |
| `sort-mismatch` | Replace a term under a binder with the variable that binder binds, where the two are assigned different types. |

A candidate is kept only when the type checker refuses it, and the refusal is
kept as the form's certificate.

The packaged fragments are `data.datasets.train.dataset.ill_typing.operators:
argument-swap` and `data.datasets.train.dataset.ill_typing.proportion:
quarter`.

## Draw an interpreted corpus

[`docs/examples/tutorials/interpreted-grammar.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/tutorials/interpreted-grammar.yaml)
declares an eight-word lexicon with meanings and a lexical interpretation into
the `propositional` language. [Declare an interpretation](../grammars/declare-an-interpretation.md)
covers its interpretation block. Select the ill-typing fragments as overrides,
draw the corpus, and read the report:

```python
from pathlib import Path

from lofi.run import plan_training
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/tutorials/interpreted-grammar.yaml")
ill_typing = (
    "data/datasets/train/dataset/ill_typing/operators=argument-swap",
    "data/datasets/train/dataset/ill_typing/proportion=quarter",
    "data.datasets.train.dataset.size=48",
)
corpus = plan_training(codec(document, overrides=ill_typing)).lower_dataset("train").corpus

report = corpus.ill_typing
print(len(corpus.forms), len(corpus.ill_typed))
print(dict(report.injection.accepted), dict(report.certificates), dict(report.observed))

refused = corpus.ill_typed[0]
print(refused.kind, refused.source, "->", refused.text, refused.observed_label)
print(refused.certificate.conflicts[0].kind)
```

```text
48 12
{'argument-swap': 12} {'unification': 13} {'well-typed': 0, 'ill-typed': 12}
argument-swap sleep(john) -> john(sleep) ill-typed
unification
```

Every one of the 48 grammatical items has a gold form, and a quarter of them,
12, also yield an ill-typed form. Each ill-typed form records the operator,
the source form, the corrupted text, and the typing failure that refutes it:
here a unification conflict, since `john` is an individual and cannot be
applied. `corpus.form_typing` holds, per item, the form-typing examples a
learner trains on, with their true and observed labels.

## Mix operators and flip labels

Append an operator with a `+` list key, and select the `label-flips` regime
with its rate:

```python
mixed = codec(
    document,
    overrides=(
        *ill_typing,
        'data.datasets.train.dataset.ill_typing.operators.entries+=[{"operator": {"kind": "over-application"}, "weight": 1.0}]',
        "data.datasets.train.dataset.ill_typing.regime=label-flips",
        "data.datasets.train.dataset.ill_typing.noise=0.5",
    ),
)
report = plan_training(mixed).lower_dataset("train").corpus.ill_typing
print(dict(report.injection.accepted), dict(report.observed), report.achieved_noise)
```

```text
{'argument-swap': 4, 'over-application': 8} {'well-typed': 6, 'ill-typed': 6} 0.5
```

The twelve injections are drawn using the operator weights, and half of the
ill-typed forms are observed as well-typed. The report gives the achieved flip
rate, so it can be compared with the declared one.

## Refusals

An ill-typing over a corpus with no interpretation is refused when the
document is validated:

```python
from lofi.spec import ConfigValueError

try:
    codec(
        Path("docs/examples/synthetic-parser.yaml"),
        overrides=(
            "data/datasets/train/dataset/ill_typing/operators=argument-swap",
            "data/datasets/train/dataset/ill_typing/proportion=quarter",
        ),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
data.datasets.train.dataset was refused: ill-typed forms are injected under an interpretation
```

The block is also refused for a proportion outside zero to one, a
non-positive `attempts`, an empty or repeating operator list, or a `noise`
rate outside the `label-flips` regime. An injection that exhausts its
attempts for one form stops the draw.

[Logical forms as types](../../explanation/logical-forms-as-types.md)
explains what the type checker decides.
