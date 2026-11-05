"""Typed planning, construction and pricing plugins for trainer variants.

A trainer is a declaration variant paired, in its own module, with a plugin
that plans its run without allocating anything, prepares its training at run
time and states the training length its run is priced at. The run roles it
requires and refuses, and the training batches it needs from a dataset, are
declarations of the variant itself, checked when a run is composed. The
supervision arm it presents to each featurized dataset is a declaration of the
plugin, checked when the run is planned against the arms each declared
featurizer accepts.
"""

from enum import StrEnum
from importlib.metadata import EntryPoint
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, override

from lofi._frozen import FrozenValue, Unpicklable
from lofi.data.featurizers import featurizer_plugin
from lofi.registry import Registry, RegistryError
from lofi.spec.components.trainer import TrainerSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SupervisionArm
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.data.datasets import (
        CheckedDataset,
        DatasetVocabulary,
        LoweredDataset,
        PairTraining,
        SentenceTraining,
    )
    from lofi.data.synthetic import InterpretationFunction
    from lofi.judge import ActionSet
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.spec import RunSpec
    from lofi.spec.grammar import GrammarAssembly

__all__ = [
    "FeaturizerMismatchError",
    "LearnerAssembly",
    "Trainer",
    "TrainerBuildContext",
    "TrainerBuilder",
    "TrainerBuilderTarget",
    "TrainerPlanner",
    "TrainerPlugin",
    "TrainerPresenter",
    "TrainerPricer",
    "TrainingBlueprint",
    "TrainingLength",
    "TrainingUnit",
    "trainer_plugin",
    "training_sampler_stream",
]


def training_sampler_stream(dataset: str, purpose: str, /) -> str:
    """Return one sampler stream owned by a named training dataset."""
    return f"training:{dataset}:{purpose}"


class FeaturizerMismatchError(ValueError):
    """An error for a featurizer that does not accept what its trainer presents.

    Parameters
    ----------
    trainer : str
        The trainer's tag.
    role : str
        The featurized dataset's role.
    featurizer : str
        The featurizer's tag.
    presented : SupervisionArm
        The supervision arm the trainer presents to the featurizer.
    accepted : frozenset[SupervisionArm]
        The supervision arms the declared featurizer accepts.
    """

    def __init__(
        self,
        trainer: str,
        role: str,
        featurizer: str,
        presented: SupervisionArm,
        accepted: frozenset[SupervisionArm],
    ) -> None:
        names = ", ".join(sorted(arm.__name__ for arm in accepted)) or "nothing"
        super().__init__(
            f"data.datasets.{role}.featurizer: the {trainer!r} trainer presents "
            f"{presented.__name__} examples, but the {featurizer!r} featurizer "
            f"accepts {names}"
        )

        #: The trainer's tag.
        self.trainer = trainer

        #: The featurized dataset's role.
        self.role = role

        #: The featurizer's tag.
        self.featurizer = featurizer

        #: The supervision arm the trainer presents.
        self.presented = presented

        #: The supervision arms the declared featurizer accepts.
        self.accepted = accepted


class TrainingUnit(StrEnum):
    """The unit a run declares its training length in."""

    EPOCHS = "epochs"
    STEPS = "steps"


class TrainingLength(NamedTuple):
    """The training length one run declares, as a count of one unit."""

    count: int
    unit: TrainingUnit


class LearnerAssembly(Protocol):
    """Expose the one symbolic assembly owned by a model."""

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the learner grammar."""
        ...

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the optional learner interpretation."""
        ...


class TrainingBlueprint(Protocol):
    """Checked run facts a trainer builder consumes."""

    @property
    def run_spec(self) -> RunSpec:
        """Return the validated run declaration."""
        ...

    @property
    def checked_datasets(self) -> Mapping[str, CheckedDataset]:
        """Return checked dataset inputs keyed by their declared role."""
        ...

    def checked_dataset(self, role: str, /) -> CheckedDataset:
        """Return the checked inputs for one declared dataset role."""
        ...

    def lower_dataset(self, role: str, /) -> LoweredDataset:
        """Lower and validate one checked dataset."""
        ...

    def dataset_vocabulary(
        self, role: str, dataset: LoweredDataset, /
    ) -> DatasetVocabulary:
        """Return one lowered dataset's token vocabulary."""
        ...

    def featurizer_resources(
        self, role: str, dataset: LoweredDataset, tensorizer: CodeTensorizer, /
    ) -> FeaturizerResources:
        """Return the resources a featurizer over one lowered dataset is built from."""
        ...

    def sentence_training(
        self, role: str, dataset: LoweredDataset, /
    ) -> SentenceTraining:
        """Return the sentence training one lowered dataset's plugin supplies."""
        ...

    def pair_training(self, role: str, dataset: LoweredDataset, /) -> PairTraining:
        """Return the inference-pair training one lowered dataset's plugin supplies."""
        ...

    @property
    def sampler_streams(self) -> tuple[str, ...]:
        """Return the named sampler streams the trainer declared."""
        ...

    @property
    def actions(self) -> ActionSet:
        """Return the actions derived from the run's grammar."""
        ...

    @property
    def learner(self) -> LearnerAssembly:
        """Return the model-owned learner assembly."""
        ...


class TrainerBuildContext(FrozenValue, Unpicklable):
    """Runtime policies and checked inputs supplied to one trainer builder."""

    __slots__ = ("_blueprint", "_policy", "_pretrained", "_run_spec", "_streams")

    _noun = "trainer build context"
    _because = "holds live random streams and device policy"

    def __init__(
        self,
        blueprint: TrainingBlueprint,
        run_spec: RunSpec,
        streams: object,
        policy: DevicePolicy,
        /,
        *,
        pretrained_resources: Mapping[str, PretrainedLoader],
    ) -> None:
        object.__setattr__(self, "_blueprint", blueprint)
        object.__setattr__(self, "_run_spec", run_spec)
        object.__setattr__(self, "_streams", streams)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(
            self,
            "_pretrained",
            MappingProxyType(dict(pretrained_resources)),
        )

    @property
    def blueprint(self) -> TrainingBlueprint:
        """Return the checked run inputs."""
        return self._blueprint

    @property
    def streams(self) -> object:
        """Return the run's restorable random streams."""
        return self._streams

    @property
    def run_spec(self) -> RunSpec:
        """Return the validated run declaration."""
        return self._run_spec

    @property
    def policy(self) -> DevicePolicy:
        """Return the resolved device policy."""
        return self._policy

    @property
    def pretrained_resources(self) -> Mapping[str, PretrainedLoader]:
        """Return pretrained loaders keyed by component role."""
        return self._pretrained


class TrainerPlanner[S: TrainerSpec](Protocol):
    """Checks one trainer's component graph without allocating runtime state."""

    def __call__(self, declaration: S, run_spec: RunSpec, /) -> tuple[str, ...]:
        """Return the sampler streams the checked trainer consumes."""
        ...


class TrainerPresenter[S: TrainerSpec](Protocol):
    """States the supervision arms one trainer presents to its featurizers."""

    def __call__(
        self, declaration: S, run_spec: RunSpec, /
    ) -> Mapping[str, frozenset[SupervisionArm]]:
        """Return the arms each featurized dataset role is presented with."""
        ...


class TrainerBuilder[S: TrainerSpec](Protocol):
    """Builds one prepared training run from a checked trainer declaration."""

    def __call__(self, declaration: S, context: TrainerBuildContext, /) -> object:
        """Return prepared training for the declaration and runtime context."""
        ...


class TrainerPricer[S: TrainerSpec](Protocol):
    """States the training length one trainer declaration runs for."""

    def __call__(self, declaration: S, /) -> TrainingLength | None:
        """Return the declared length, or None for a trainer that declares none."""
        ...


class Trainer(Protocol):
    """Plans, prepares and prices the training one declaration describes."""

    def plan(self, declaration: TrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
        """Check the run's component graph and return its sampler streams.

        Parameters
        ----------
        declaration : TrainerSpec
            The run's trainer declaration.
        run_spec : RunSpec
            The composed run, whose roles and dataset the declaration has
            already accepted.
        """
        ...

    def prepare(
        self, declaration: TrainerSpec, context: TrainerBuildContext, /
    ) -> object:
        """Build prepared training from the checked declaration and context."""
        ...

    def price(self, declaration: TrainerSpec, /) -> TrainingLength | None:
        """Return the training length the run is priced at, if it declares one."""
        ...


class TrainerBuilderTarget[S: TrainerSpec](FrozenValue):
    """A trainer builder imported only when runtime construction begins."""

    __slots__ = ("_target",)

    _noun = "trainer builder target"

    def __init__(self, target: str, /) -> None:
        if not target:
            msg = "a trainer builder target cannot be empty"
            raise ValueError(msg)

        object.__setattr__(self, "_target", target)

    def __call__(self, declaration: S, context: TrainerBuildContext, /) -> object:
        """Load and invoke the runtime builder named by this target."""
        point = EntryPoint(name="builder", value=self._target, group="lofi.trainer")
        builder = cast("TrainerBuilder[S]", point.load())

        return builder(declaration, context)

    @override
    def __repr__(self) -> str:
        """Return the installed builder target."""
        return f"TrainerBuilderTarget({self._target!r})"


class TrainerPlugin[S: TrainerSpec](FrozenValue):
    """Pair a trainer declaration with its planner, presenter, builder and pricer."""

    __slots__ = ("_build", "_plan", "_present", "_price", "_spec")

    _noun = "trainer plugin"

    def __init__(
        self,
        *,
        spec: type[S],
        plan: TrainerPlanner[S],
        present: TrainerPresenter[S],
        build: TrainerBuilder[S],
        price: TrainerPricer[S],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_present", present)
        object.__setattr__(self, "_build", build)
        object.__setattr__(self, "_price", price)

    @property
    def spec(self) -> type[S]:
        """Return the declaration class this plugin implements."""
        return self._spec

    @property
    def planner(self) -> TrainerPlanner[S]:
        """Return the allocation-free planner."""
        return self._plan

    @property
    def presenter(self) -> TrainerPresenter[S]:
        """Return the declaration of the arms presented to each featurizer."""
        return self._present

    @property
    def builder(self) -> TrainerBuilder[S]:
        """Return the runtime builder."""
        return self._build

    @property
    def pricer(self) -> TrainerPricer[S]:
        """Return the training-length declaration the run is priced at."""
        return self._price

    def _declaration(self, declaration: TrainerSpec, /) -> S:
        """Return a declaration of this plugin's exact variant."""
        if type(declaration) is not self._spec:
            msg = (
                f"the trainer plugin builds {self._spec.__name__}, not "
                f"{type(declaration).__name__}"
            )
            raise RegistryError(msg, family="trainer")

        return cast("S", declaration)

    def plan(self, declaration: TrainerSpec, run_spec: RunSpec, /) -> tuple[str, ...]:
        """Check the trainer graph and return its sampler streams.

        Raises
        ------
        FeaturizerMismatchError
            If a featurized dataset's declared featurizer does not accept an
            arm the trainer presents to it.
        ValueError
            If the trainer presents examples to a dataset that declares no
            featurizer, or its planner refuses the run's wiring.
        """
        checked = self._declaration(declaration)

        for role, presented in self._present(checked, run_spec).items():
            featurizer = run_spec.data.datasets[role].featurizer

            if featurizer is None:
                msg = (
                    f"data.datasets.{role}.featurizer: the "
                    f"{cast('str', checked.kind)!r} "
                    "trainer presents examples to this dataset, which declares "
                    "no featurizer"
                )
                raise ValueError(msg)

            accepted = featurizer_plugin(featurizer).accepted(featurizer)
            refused = sorted(presented - accepted, key=lambda arm: arm.__name__)

            if refused:
                raise FeaturizerMismatchError(
                    cast("str", checked.kind),
                    role,
                    cast("str", featurizer.kind),
                    refused[0],
                    accepted,
                )

        return self._plan(checked, run_spec)

    def prepare(
        self, declaration: TrainerSpec, context: TrainerBuildContext, /
    ) -> object:
        """Build prepared training from the checked declaration."""
        return self._build(self._declaration(declaration), context)

    def price(self, declaration: TrainerSpec, /) -> TrainingLength | None:
        """Return the training length the declaration runs for, if declared."""
        return self._price(self._declaration(declaration))

    @override
    def __repr__(self) -> str:
        """Return the declaration class this plugin implements."""
        return f"TrainerPlugin(spec={self._spec.__name__})"


def trainer_plugin(declaration: TrainerSpec, /) -> TrainerPlugin[TrainerSpec]:
    """Return the installed plugin for one trainer declaration.

    Parameters
    ----------
    declaration : TrainerSpec
        The trainer declaration whose tag names the plugin's entry point.

    Raises
    ------
    RegistryError
        If no plugin is installed under the declaration's tag, or the entry
        names an object that is not a plugin for the declaration's variant.
    """
    tag = cast("str", declaration.kind)
    member = Registry[object]("trainer").get(tag)

    if not isinstance(member, TrainerPlugin):
        msg = f"the trainer entry {tag!r} names {member!r}, not a plugin"
        raise RegistryError(msg, family="trainer")

    plugin = cast("TrainerPlugin[TrainerSpec]", member)

    if plugin.spec is not type(declaration):
        msg = (
            f"the trainer entry {tag!r} builds "
            f"{plugin.spec.__name__}, not {type(declaration).__name__}"
        )
        raise RegistryError(msg, family="trainer")

    return plugin
