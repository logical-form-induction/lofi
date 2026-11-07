# Combine losses

A run's `loss` section declares the training objective as a list of weighted
terms. Each term pairs one objective from the `lofi.loss` [entry point group](../../reference/entry-points.md#registered-groups-and-names)
with a nonnegative weight, an optional name, and an optional ramp of that
weight over the first [optimizer steps](configure-a-trainer.md#optimization-units). The trainer minimizes the weighted sum
of the terms and logs each one under its own name.
## Declare the terms

A [loss](../../explanation/induction-objectives.md#what-training-is-asked-to-improve) is a numerical quantity the trainer tries to reduce. Different terms
can express different aims: fitting observed labels, making two learned
representations agree, or preferring less complex expressions. The weighted
sum sets the contribution of each aim to the training objective.

In an illustrative calculation, losses of 2 and 0.5 with weights 1 and 0.1
contribute $2+0.1\times0.5=2.05$. Each term's influence also depends on its
scale and gradient. Read
the separately logged terms to see which parts of the objective are changing.

The checked-in [`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
declares two terms:

```yaml
loss:
  terms:
    - objective:
        kind: grammaticality
        readout: sentences
        noise: 0.0
      weight: 1.0
    - objective:
        kind: coherence
        readout: category-terms
        space: categories
        estimator: truncated
        divergence:
          kind: kl
        warmup: null
      weight: 0.1
```

Compose the document and list its terms:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")

run = codec(document)
for term in run.loss.terms:
    print(term.key, term.weight, term.schedule)
```

```text
grammaticality 1.0 None
coherence 0.1 None
```

`term.key` is the name the term is built, reported and logged under. A term
that declares no `name` takes its objective's `kind`, so these two terms log
`loss/grammaticality` and `loss/coherence`, and [early stopping](configure-a-trainer.md#early-stopping) refers to them
by those keys.

Each term has four fields:

| Field | What it says |
| --- | --- |
| `objective` | One loss declaration, selected by its `kind`. |
| `weight` | A finite, nonnegative multiplier. The default is `1.0`. |
| `name` | The key the term is logged under, or `null` to use the objective's `kind`. A name is one nonempty segment and contains no slash. |
| `schedule` | `{start, steps}`: the weight moves linearly from `start` to `weight` over the first `steps` [optimizer steps](configure-a-trainer.md#optimization-units) and keeps `weight` afterwards. `null` keeps `weight` throughout. |

## Add a term with a warmup { #add-a-term-with-a-warm-up }

A [warmup](../../reference/configuration/training.md) gradually introduces one term's influence. This can let the model
first develop useful predictions before a regularizer applies at full weight.
The schedule is indexed by [optimizer steps](configure-a-trainer.md#optimization-units), so changing batch accumulation can
change how many examples have been seen when a ramp ends. Specify the schedule
in the units the trainer actually advances.

Append a term from the command line with `loss.terms+=[...]` rather than
restating the list. This override adds a simplicity regularizer whose weight ramps
from zero to `0.01` over the first hundred optimizer steps:

```python
extended = codec(
    document,
    overrides=(
        'loss.terms+=[{"objective": {"kind": "simplicity", "readout": "category-terms"},'
        ' "weight": 0.01, "schedule": {"start": 0.0, "steps": 100}}]',
    ),
)
term = extended.loss.terms[-1]
print(term.key, term.weight, term.schedule)
print(term.objective)
```

```text
simplicity 0.01 WeightScheduleSpec(start=0.0, steps=100)
SimplicityLossSpec(kind='simplicity', readout='category-terms', scope=<SimplicityScope.LEXICON: 'lexicon'>, symbol_cost=UniformSymbolCostSpec(kind='uniform', cost=1.0))
```

The printed objective shows the defaults the schema filled in: the regularizer
prices lexical cells only, at a uniform cost of one per constructor. On the
command line the same override is
`--set 'loss.terms+=[{"objective": ...}]'`; the
[configuration guide](../setup/configuration.md) describes list extension.

The term level `schedule` ramps a term's weight. Two objectives also carry
their own warmup, which is a different thing: a coherence term's `warmup`
counts the steps over which marginal weighting is introduced inside the
objective (`null` or `0` turns it off, and the default is `1000`), and a
token-warm-up training step is declared on the trainer rather than in the loss
section. See [Configure a trainer](configure-a-trainer.md) for the latter.

The [objective definitions](../../explanation/induction-objectives.md) give
an equation for each family, including masks, averaging denominators,
normalization, and gradient detachment. Each term retains its own observation
unit and normalization when added to the weighted objective.
## Choose objectives

Start from the signal available to the learner. Observed grammaticality calls
for a term that reads grammaticality evidence; [partial category labels](supervise-and-constrain.md) call
for a supervision term; a relation between [representation spaces](../models/configure-term-encoders-and-decoders.md#declare-a-space) calls for a
term that compares those spaces. The named role connects each objective to
that information. The constructed model must publish the quantity required
by each objective at its named role.

The packaged `lofi.loss` group registers thirteen objectives. Each reads a
role the model declares, named by `readout`, `space` or `decoder`, so the role
must exist under [`model.readouts`](../../reference/spec.md#lofi.spec.ReadoutSpec), [`model.spaces`](../../reference/spec.md#lofi.spec.SpaceSpec) or `model.decoders`.

| `kind` | Scores | Required fields | Settings and defaults |
| --- | --- | --- | --- |
| `grammaticality` | Binary grammaticality of each string against a published root mass. | `readout` | `noise` (default `0.0`) is the probability that an observed label was flipped, in `[0, 0.5)`. |
| `coherence` | The divergence between a parent cell's term distribution and the distribution its combinators compose. | `readout`, `space` | `estimator`: `exact` (default) or `truncated`; `residual_mass` overrides the truncation bound; `divergence`: `kl` (default), `cross-entropy`, or `fenchel-young` with `alpha` at least one (default `1.5`); `warmup` (default `1000`). |
| `chart-scale` | A penalty on chart cells whose Euclidean norm departs from a target. | none | `target` (default `1.0`, positive); `penalty`: `squared-ratio` (default), `squared-distance` or `absolute-distance`. |
| `simplicity` | Expected declared constructor cost under the readout. | `readout` | `scope`: `lexicon` (default) or `all-cells`; `symbol_cost`: `uniform` with `cost` (default `1.0`), `log-alphabet`, or `per-constructor` with a `costs` map. |
| `derivation-entropy` | The entropy of the parser's tree distribution. | `direction`: `minimize` or `maximize` | none |
| `span-reconstruction` | Reconstruction of each span's words from its inside cell. | `decoder` | `weighting`: `uniform` (default) or `marginal`; `max_span_length` (default `null`, every length); `normalization`: `span` (default) or `token`. |
| `outside-reconstruction` | Reconstruction of each word from its outside vector, the context outside its own leaf span. | `decoder` | none |
| `form-typing` | Binary typing supervision over one complete [term readout](../models/configure-readouts.md#choose-a-route). | `readout` | none |
| `form-supervision` | The negative log likelihood of each retained gold [logical form](../../explanation/logical-forms-as-types.md#semantic-typing). | `readout` | Requires a dataset that supplies gold forms. |
| `inference` | A loss over the model's grounded inference distribution. | `signal`: `likelihood` or `reward`; `response` | `auxiliary_weight` (default `0.0`). Reads the grounding declared under `model.grounding`. |
| `shaped-proof` | Proof-shaped credit over decoded forms against their gold forms. | `readout`, `shaping`, `budget` | `shaping` pays `ill_formed`, `well_formed`, `type_checks` and `proved`, with a `length` cost. |
| `pinning` | The mass a readout assigns to the targets of selected constraints, or the exclusion of [negative brackets](supervise-and-constrain.md#choose-the-negative-bracket-route). | none | `readout` (default `null`). |
| `vector-anchor` | The distance from constrained chart cells to the [term encoder](../models/configure-term-encoders-and-decoders.md#choose-an-encoder)'s vectors for their targets. | `readout` | `distance`: `cosine` (default) or `squared-euclidean`; `aggregate`: `nearest` (default) or `mean`; `margin` (default `1.0`). |

The `inference` term is covered in [Reward with a theorem
prover](reward-with-a-theorem-prover.md) and [Reward with model
checking](reward-with-model-checking.md), and `pinning` and `vector-anchor` in
[Supervise and constrain a learner](supervise-and-constrain.md). The class
reference is under [Training declarations](../../reference/configuration/training.md).

## Start from a packaged loss fragment

[`lofi/configs/loss/`](https://github.com/logical-form-induction/lofi/tree/main/lofi/configs/loss/) holds one fragment per common objective. Each, except
`inference.yaml`, declares a complete `terms` list. Select one in a run
document's `defaults` list:

```yaml
defaults:
  - loss: pinning
```

`loss/pinning.yaml` is the three-term objective the pinning run documents
use:

```yaml
terms:
  - objective: {kind: grammaticality, readout: sentences}
  - objective: {kind: coherence, readout: category-terms, space: categories,
      estimator: truncated, warmup: null}
    weight: 0.1
  - objective: {kind: pinning, readout: types}
```

A `loss` section written in the run document itself is more specific than a
selected fragment, so its `terms` list replaces the fragment's. Extend the
fragment's list with `terms+:` instead when the document should keep it.

## Run two terms of one kind

Two terms of one kind must name themselves apart. Without a name, the second
coherence term collides with the first:

```python
def refusal(overrides):
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused).splitlines()[-1])


refusal(
    (
        'loss.terms+=[{"objective": {"kind": "coherence",'
        ' "readout": "category-terms", "space": "categories"}}]',
    )
)
```

```text
ConfigValueError | loss.terms was refused: two loss terms are named 'coherence'; give each term of one kind its own name
```

Give the second term `name: coherence-alternate`, and it logs under
`loss/coherence-alternate`.

## How a declaration is refused

Every check below runs at composition, before any data is drawn or any model
is built:

```python
refusal(('loss.terms+=[{"objective": {"kind": "chart-scale"}, "weight": -1.0}]',))
refusal(
    ('loss.terms+=[{"objective": {"kind": "chart-scale"}, "name": "scale/cells"}]',)
)
refusal(
    (
        'loss.terms+=[{"objective": {"kind": "chart-scale"},'
        ' "schedule": {"start": 0.0, "steps": 0}}]',
    )
)
```

```text
ConfigValueError | loss.terms.2.weight was refused: an objective weight is negative: -1.0
ConfigValueError | loss.terms.2.name was refused: a loss term name contains no slash: 'scale/cells'
ConfigValueError | loss.terms.2.schedule.steps was refused: a schedule's steps is not positive: 0
```

An unknown `kind` is refused with the list of installed objectives:

```python
try:
    codec(document, overrides=('loss.terms+=[{"objective": {"kind": "chart-scal"}}]',))
except Exception as refused:
    print(str(refused).split("LossSpec registers: ")[1])
```

```text
['chart-scale', 'coherence', 'derivation-entropy', 'form-supervision', 'form-typing', 'grammaticality', 'inference', 'outside-reconstruction', 'pinning', 'shaped-proof', 'simplicity', 'span-reconstruction', 'vector-anchor']
```

A loss section also refuses an empty `terms` list, and grounded objectives that
ask incompatible proof questions. Composition does not check that a term's
named role exists: a simplicity term reading `readout: missing` composes, and
the role is resolved only when the objective is built against the model.
Check role names with `show-config` before a long run.

[Induction objectives](../../explanation/induction-objectives.md) explains
what each family of terms asks of a learner.
