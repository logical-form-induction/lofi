# Package a plugin

A LoFI plugin is an installed distribution whose entry points add variants to
LoFI's families: an entry point in the group `lofi.<family>` registers one
variant under the entry's name, and a `lofi.config_dirs` entry contributes
configuration fragments. This guide registers a component, a trainer and a
declaration-only variant, contributes fragments, and checks that a fresh
process discovers them. Installing the distribution is the whole
registration; nothing in LoFI is edited.

## How families are discovered

[`load_plugins`](../../reference/spec.md) loads every installed entry point
in three passes, each in stable name order:

1. `lofi.families`, whose entries name the root declaration class of each
   component family, for instance `term_encoder =
   "lofi.spec.components.model:TermEncoderSpec"`;
2. the declaration families that have no built component (rules, matching,
   normal forms, goals, formalisms, semirings, A* choices, fuzzy logics,
   knowledge-base readers, reducers, corrections, launchers, sweep designs and
   the others listed in `lofi.spec._plugins`), whose entries name declaration
   classes;
3. the group of every component family named in the first pass, whose entries
   name plugin objects.

A family root loads before its variants, so one distribution may contribute
both a new family and its variants. The installed counts:

```python
from importlib.metadata import entry_points

from lofi.spec import component_groups, load_plugins

loaded = load_plugins()
print(len(loaded), len(component_groups()))
print(sorted(point.name for point in entry_points(group="lofi.term_encoder")))
```

```text
320 28
['hrr', 'linear-rnn', 'transformer', 'tree-rnn']
```

## Register a component

A component variant is a declaration class and a
[`ComponentPlugin`](../../reference/registry.md) that pairs the class with the
builder of its component. LoFI's holographic term encoder is one. Its
declaration, in `lofi/spec/encoders/hrr.py`, subclasses the family root,
fixes its `kind`, holds every default, and returns a settings value:

```python
from lofi.spec.encoders.hrr import HrrTermEncoderSpec

spec = HrrTermEncoderSpec(width=8)
print(spec.settings())
```

```text
HrrTermEncoderSettings(width=8, magnitude=1.0, floor=0.001)
```

The [declaration module](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/encoders/hrr.py)
contains the complete class and its validation rules. Import the installed
class when exploring it; defining another class with the same `kind` would
register that tag twice.


The runtime module, `lofi/nn/encoders/hrr.py`, builds the component from the
settings and a `BuildContext`, and exports the plugin:

```python
from typing import Final

from lofi.nn.build import BuildContext
from lofi.nn.encoders.hrr import HolographicEncoderSettings, HrrTermEncoder
from lofi.nn.policy import place
from lofi.registry import ComponentPlugin
from lofi.spec.encoders.hrr import HrrTermEncoderSettings


def _build(
    settings: HrrTermEncoderSettings, context: BuildContext, /
) -> HrrTermEncoder:
    return place(
        HrrTermEncoder(
            codec=context.codec,
            component=context.component,
            settings=HolographicEncoderSettings(
                width=settings.width,
                magnitude=settings.magnitude,
                floor=settings.floor,
            ),
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=HrrTermEncoderSpec, build=_build)
```

The distribution lists the plugin under the family's group, with the variant's
`kind` as the entry name:

```toml
[project.entry-points."lofi.term_encoder"]
hrr = "lofi.nn.encoders.hrr:PLUGIN"
```

Loading the group defines the declaration class, which adds the variant to
the family's tagged union, and registers the builder under the same tag. The
settings value, the builder and the component's constructor take every value
as a required argument, so a default lives only on the declaration.

A model component's `BuildContext` carries the codec view, the vocabulary,
the action inventory, the producers wired to the component's ports, the
target sets, the pretrained loader and the device policy. The other component
families pass the run resources their components read: a loss takes the
built components it names by role, a featurizer the dataset's resources, a
type sampler the term store, a pair sampler the finite type inventory and
declared rules, an optimizer the selected parameters, and a schedule the
optimizer and its training horizon. Sinks, datasets and evaluators are
component families too, listed under `lofi.sink`, `lofi.dataset` and
`lofi.evaluator`.

A `lofi.dataset` entry is a `DatasetPlugin`. Beside `check`, `lower` and
`validate`, it builds the token vocabulary and the featurizer resources from a
lowered dataset and the `ReservedForms` its dataset section declares, and it
supplies the dataset's training routes. `sentences` returns a
`SentenceTraining`, whose `epochs` partitions the dataset's sentence
occurrences into seeded training and retained batches; `pairs` returns a
`PairTraining`, which does the same for labeled inference pairs. A dataset kind
that supplies neither route passes `RefusedTraining(reason)`, so a run that
trains on it is refused with that reason. Training reads epochs only through
these routes, so a new dataset kind trains without a change to `lofi.train`.

## Register a trainer

A trainer is a `TrainerSpec` variant paired with a `TrainerPlugin` in the
`lofi.trainer` group. The declaration states what a run must hold for it:

- `required_roles` and `forbidden_roles` return the `RunRole` values the
  trainer trains and those it does not consume, such as
  `RunRole.TERM_ENCODERS` or `RunRole.PARSER`;
- `dataset_requirements` maps each dataset role to the `TrainingBatchKind`
  values (`SENTENCES`, `INFERENCE_PAIRS`, `GOLD_FORMS`, or `JUDGMENTS`) it
  must yield.

A run that breaks any of these is refused when it is composed, naming the
role or the batch kind. The plugin pairs the declaration with four
callables: `plan`, which returns the sampler streams the run draws from
without allocating anything; `present`, which names the supervision supplied
by each dataset role; `build`, which prepares training from a
`TrainerBuildContext`; and `price`, which returns the `TrainingLength` an
experiment over the trainer is priced at. The packaged distance trainer, in
`lofi/training_plugins/distance.py`:

```python
from collections.abc import Mapping

from lofi.data import SupervisionArm
from lofi.spec import RunSpec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.training_plugins import (
    TrainerBuilderTarget,
    TrainerPlugin,
    TrainingLength,
    TrainingUnit,
    training_sampler_stream,
)


def _plan(declaration: DistanceTrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
    return (training_sampler_stream(declaration.dataset, "distance-batches"),)


def _present(
    declaration: DistanceTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    return {}


def _price(declaration: DistanceTrainerSpec, /) -> TrainingLength:
    return TrainingLength(declaration.steps, TrainingUnit.STEPS)


PLUGIN: Final = TrainerPlugin(
    spec=DistanceTrainerSpec,
    plan=_plan,
    present=_present,
    build=TrainerBuilderTarget("lofi.train._assembly:_build_distance_training"),
    price=_price,
)
```

`trainer_plugin` resolves a declaration's plugin:

```python
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.training_plugins import trainer_plugin

declaration = DistanceTrainerSpec(
    dataset="train",
    encoder="terms",
    sort="T",
    steps=200,
    batch_size=8,
    checkpoints=CheckpointPolicySpec(interval=50, keep=1),
    inverse_distance_power=0.0,
    embedding_penalty=0.0,
)
plugin = trainer_plugin(declaration)
print(plugin.spec.__name__, plugin.price(declaration))
print(sorted(role.name for role in declaration.required_roles()))
```

```text
DistanceTrainerSpec TrainingLength(count=200, unit=<TrainingUnit.STEPS: 'steps'>)
['TERM_ENCODERS', 'TYPE_SAMPLER']
```

The prepared training a builder returns fits under one of the two fitting
plans the training package runs, parser epochs or distance steps.

## Register a declaration-only variant

A family without a built component registers its value directly. The packaged
fuzzy logics are three entries naming module values:

```toml
[project.entry-points."lofi.fuzzy_logic"]
product = "lofi.lf.models:PRODUCT"
goedel = "lofi.lf.models:GOEDEL"
lukasiewicz = "lofi.lf.models:LUKASIEWICZ"
```

These groups load before any document is validated, so a third-party
tagged-union variant is available at the boundary that names it.

A new fuzzy logic needs no more than its t-norm. A
[`TNorm`](../../reference/lf.md) writes its conjunction, dual t-conorm and
residuum once against an `Arithmetic`, and `FuzzyAlgebra` evaluates it on
floats for model checking and on tensors for learned structures:

```python
from lofi.lf.models import (
    CRISP,
    FLOAT_ARITHMETIC,
    MATERIAL,
    TNORM_QUANTIFIER,
    Arithmetic,
    FuzzyAlgebra,
)


class Hamacher:
    @property
    def name(self) -> str:
        return "hamacher"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        product = arithmetic.multiply(left, right)
        sum_ = arithmetic.subtract(arithmetic.add(left, right), product)
        guarded = arithmetic.where_at_most(
            sum_, arithmetic.constant(0.0, left), arithmetic.constant(1.0, left), sum_
        )
        return arithmetic.divide(product, guarded)

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        return arithmetic.subtract(
            one,
            self.conjunction(
                arithmetic,
                arithmetic.subtract(one, left),
                arithmetic.subtract(one, right),
            ),
        )

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        return arithmetic.where_at_most(left, right, one, right)


algebra = FuzzyAlgebra(
    Hamacher(),
    implication=MATERIAL,
    quantifier=TNORM_QUANTIFIER,
    equality=CRISP,
    margin=0.0,
)
print(algebra.name, round(algebra.conjunction(0.5, 0.5), 4))
print(round(algebra.conjoin(FLOAT_ARITHMETIC, 0.5, 0.5), 4))
```

```text
hamacher 0.3333
0.3333
```

Its entry point names a factory whose `build` returns that algebra.

## Contribute configuration fragments

A `lofi.config_dirs` entry names an importable package, not an attribute:

```toml
[project.entry-points."lofi.config_dirs"]
my-lofi-plugin = "my_lofi_plugin.configs"
```

The package's directory uses the slot layout of `lofi/configs/`, so
`model/encoders/terms/tree.yaml` under it contributes the option `tree` to the
`model.encoders.terms` slot. Plugin directories come last on the
[search path](../setup/configuration.md), after the project's `configs/` and
the packaged library, so a project or packaged option of the same name shadows
a plugin's. Choose option names specific to the distribution.

## Check the installed contribution

Build and install the distribution, then check discovery in a new process.
[`plugin_for`](../../reference/registry.md) returns the plugin installed for a
declaration's family and tag:

```python
from lofi.registry import plugin_for
from lofi.spec.encoders.hrr import HrrTermEncoderSpec

print(plugin_for(HrrTermEncoderSpec(width=8)))
```

```text
ComponentPlugin(spec=HrrTermEncoderSpec)
```

Every family resolves its tags the same way, through
[`Registry`](../../reference/registry.md) over the family's entry-point group:
`Registry("matching").get("alpha")` returns the object the `alpha` entry of
`lofi.matching` names. An entry point is the only way to register a component;
there is no in-process registration.

Building a changed declaration with the plugin's `construct` and checking that
the component reflects the change catches an entry point that names
source-tree code, stale build output or the wrong callable. Two entries under
one tag raise `TagCollisionError` for the whole family and an unregistered tag
raises `UnknownTagError`; neither is resolved by installation order. Run the exported
[law suites](laws.md) for every protocol the plugin implements.
