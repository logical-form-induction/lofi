"""Seeded assembly of configured training graphs."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, Protocol, cast, final, override

from torch import nn

from lofi._frozen import FrozenValue
from lofi.train._autocast import OptimizationState
from lofi.train._capacity import with_parameter_capacity
from lofi.train._grouping import GroupResolution, resolve_groups
from lofi.train._groups import ActiveGroups, ParameterGroup, ParameterGroups
from lofi.train._resume import _copy_state_tree, _validate_compatible_state
from lofi.train._workload import AlternatingStack, ParserStack, StepStack

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.data import ConstraintScope, SelectionReport
    from lofi.data._epochs import DatasetEpoch, EpochSource
    from lofi.data.datasets import LoweredDataset
    from lofi.data.synthetic import SyntheticCorpus
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic.training import (
        InferenceTrainingBatch,
        PairCorpus,
        SyntheticTrainingBatch,
    )
    from lofi.judgment.bead import TrainedPointPredictor
    from lofi.lexicon import ClampPolicy, ClampReport
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.grounding import GroundingMetadata, GroundingTelemetry
    from lofi.nn.loss._constraint import (
        BatchLocalConstraint,
        BatchLocalNegativeBracket,
    )
    from lofi.nn.policy import DevicePolicy
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term
    from lofi.train._alternating import AlternatingStep
    from lofi.train._autocast import AutocastPolicy
    from lofi.train._checkpoint import CheckpointPolicy
    from lofi.train._early_stopping import AccuracyStopping, EarlyStopping
    from lofi.train._grounded import GroundedForward, GroundedStep
    from lofi.train._parsing import ChartForward, ParserStep
    from lofi.train._pretraining import Draw
    from lofi.train._staging import Stage
    from lofi.train._steps import EvalResult, StepOutcome, TrainResult
    from lofi.train._store import StateTree
    from lofi.train._workload import TimedStack
    from lofi.training_plugins.steps import FittingTask

__all__ = [
    "ConfiguredModel",
    "GeneratedStep",
    "ParserFittingPlan",
    "PreparedTraining",
    "ResidualNames",
    "StepFittingPlan",
]

type _ParserBatch = SyntheticTrainingBatch[
    BatchLocalConstraint[Term, ConstraintScope]
    | BatchLocalNegativeBracket[ConstraintScope]
]
type _InferenceBatch = InferenceTrainingBatch
type _TrainingBatch = _ParserBatch | _InferenceBatch
type _TrainingForward = ChartForward[_ParserBatch] | GroundedForward[_InferenceBatch]
type _TrainingStep = ParserStep[_ParserBatch] | GroundedStep[_InferenceBatch]


class ResidualNames(Protocol):
    """Names the truncation residual each objective of an assembly reports."""

    @property
    def residual_keys(self) -> Mapping[str, str]:
        """Return the telemetry key of each residual, by objective role."""
        ...


class GeneratedStep(Protocol):
    """Fits and evaluates one generated batch drawn under a request."""

    @property
    def assembly(self) -> ResidualNames:
        """Return the weighted objectives the step trains."""
        ...

    def run(self, draw: Draw, /) -> StepOutcome[object, TrainResult]:
        """Draw one batch and take one optimizer step on it."""
        ...

    def evaluate(self, draw: Draw, /) -> EvalResult:
        """Draw one held-out batch and evaluate it without fitting it."""
        ...


@final
class _ConfiguredTopology:
    """One canonical module tree for group ownership and checkpoint state.

    A role whose own module was built with parameters none of which take a
    gradient declared itself frozen, and stays frozen whatever groups a run
    activates. Parameters a nested role owns are the nested role's, and do not
    count.
    """

    __slots__ = ("_aggregate", "_frozen", "_modules", "_omitted", "_paths")

    def __init__(
        self,
        modules: Mapping[str, nn.Module],
        /,
        *,
        target_parameter_count: int | None = None,
    ) -> None:
        held = dict(with_parameter_capacity(modules, target_parameter_count))
        nested = {
            nested_role
            for nested_role, nested_module in held.items()
            for owner_role, owner_module in held.items()
            if owner_role != nested_role
            if any(
                path and candidate is nested_module
                for path, candidate in owner_module.named_modules(
                    remove_duplicate=False
                )
            )
        }
        aggregate = nn.Module()

        for position, role in enumerate(sorted(set(held) - nested)):
            aggregate.add_module(f"component_{position}", held[role])

        paths: dict[str, str] = {}

        for role, module in held.items():
            candidates = tuple(
                path
                for path, candidate in aggregate.named_modules(remove_duplicate=False)
                if path and candidate is module
            )

            if len(candidates) != 1:
                msg = (
                    f"component role {role!r} has {len(candidates)} canonical "
                    "locations in the configured module topology"
                )
                raise ValueError(msg)

            paths[role] = candidates[0]

        omitted: dict[str, tuple[str, ...]] = {}

        for role, path in paths.items():
            prefixes = {
                nested_path.removeprefix(f"{path}.")
                for nested_role, nested_path in paths.items()
                if nested_role != role and nested_path.startswith(f"{path}.")
            }
            omitted[role] = tuple(
                sorted(
                    prefix
                    for prefix in prefixes
                    if not any(
                        prefix.startswith(f"{other}.")
                        for other in prefixes
                        if other != prefix
                    )
                )
            )

        frozen: set[str] = set()

        for role, module in held.items():
            own = tuple(
                parameter
                for name, parameter in module.named_parameters()
                if not any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in omitted[role]
                )
            )

            if own and not any(parameter.requires_grad for parameter in own):
                frozen.add(role)

        object.__setattr__(self, "_aggregate", aggregate)
        object.__setattr__(self, "_frozen", frozenset(frozen))
        object.__setattr__(self, "_modules", MappingProxyType(held))
        object.__setattr__(self, "_paths", MappingProxyType(paths))
        object.__setattr__(self, "_omitted", MappingProxyType(omitted))

    @property
    def modules(self) -> Mapping[str, nn.Module]:
        """Return each declared component role and its exact live module."""
        return self._modules

    @property
    def named_parameters(self) -> Mapping[str, nn.Parameter]:
        """Return the canonical aggregate parameter namespace."""
        return MappingProxyType(dict(self._aggregate.named_parameters()))

    @property
    def frozen(self) -> frozenset[str]:
        """Return the roles whose modules were built with no trainable parameter."""
        return self._frozen

    def owns_parameters(self, role: str, /) -> bool:
        """Return whether a role holds a parameter outside every nested role."""
        omitted = self._omitted[role]
        return any(
            not any(
                name == prefix or name.startswith(f"{prefix}.") for prefix in omitted
            )
            for name, _ in self._modules[role].named_parameters()
        )

    def resolve(self, active: tuple[str, ...] | None, /) -> GroupResolution:
        """Resolve most-specific role ownership over the canonical topology.

        Parameters
        ----------
        active : tuple[str, ...] | None
            The roles the run trains, or None for every declared group. A
            frozen role's group is never active, so its parameters take no
            gradient even where ``active`` names it, and a component declared
            frozen is never trained.

        Returns
        -------
        GroupResolution
            The parameters an optimizer steps over and what each role holds.

        Raises
        ------
        ValueError
            If the resolution leaves no parameter trainable.
        """
        groups = ParameterGroups(
            ParameterGroup(role, self._paths[role])
            for role in self._modules
            if self.owns_parameters(role)
        )
        enabled = groups.names if active is None else frozenset(active)
        resolution = resolve_groups(
            groups,
            ActiveGroups(enabled - self._frozen),
            self._aggregate,
        )

        if not resolution.parameters:
            msg = "configured training selected no free parameter"
            raise ValueError(msg)

        return resolution

    def role_state(self, role: str, /) -> dict[str, StateTree]:
        """Return one role's state without state owned by a nested role."""
        omitted = self._omitted[role]

        return cast(
            "dict[str, StateTree]",
            {
                name: value
                for name, value in self._modules[role].state_dict().items()
                if not any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in omitted
                )
            },
        )


def _untrained(
    role: str, topology: _ConfiguredTopology, groups: GroupResolution, /
) -> bool:
    """Return whether a role owns parameters and none of them takes a gradient.

    A role is untrained when its module was built frozen or when the run's
    active groups leave its group out, so its parameters keep the values they
    were loaded with for the whole run.

    Parameters
    ----------
    role : str
        The component role.
    topology : _ConfiguredTopology
        The canonical module tree.
    groups : GroupResolution
        What the run's active groups select.
    """
    if not topology.owns_parameters(role):
        return False

    owned = groups.members(role)

    return bool(owned) and owned.isdisjoint(groups.trainable)


@final
class ConfiguredModel(FrozenValue):
    """One live training graph with complete role-keyed component state."""

    _noun = "configured model"

    __slots__ = (
        "_checkpoint_labels",
        "_codec",
        "_groups",
        "_pinned",
        "_policy",
        "_topology",
    )

    def __init__(
        self,
        topology: _ConfiguredTopology,
        *,
        codec: NeuralCodecView,
        groups: GroupResolution,
        policy: DevicePolicy,
        checkpoint_labels: Mapping[str, Mapping[str, int]],
        pretrained: Mapping[str, str],
    ) -> None:
        absent = sorted(set(pretrained) - set(topology.modules))

        if absent:
            msg = f"pretrained weights are named for absent component roles {absent}"
            raise ValueError(msg)

        object.__setattr__(self, "_topology", topology)
        object.__setattr__(
            self,
            "_pinned",
            MappingProxyType(
                {
                    role: identity
                    for role, identity in sorted(pretrained.items())
                    if _untrained(role, topology, groups)
                }
            ),
        )
        object.__setattr__(self, "_codec", codec)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(
            self,
            "_checkpoint_labels",
            MappingProxyType(
                {
                    role: MappingProxyType(dict(labels))
                    for role, labels in checkpoint_labels.items()
                }
            ),
        )

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        msg = "a configured model is immutable"
        raise AttributeError(msg)

    @property
    def modules(self) -> Mapping[str, nn.Module]:
        """Return every stateful component keyed by its configured role."""
        return self._topology.modules

    @property
    def codec(self) -> NeuralCodecView:
        """Return the run-wide neural codec view."""
        return self._codec

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Return each optimizer parameter exactly once."""
        return self._groups.parameters

    @property
    def groups(self) -> GroupResolution:
        """Return most-specific parameter ownership for every declared role."""
        return self._groups

    @property
    def named_parameters(self) -> Mapping[str, nn.Parameter]:
        """Return every parameter under its canonical aggregate name."""
        return self._topology.named_parameters

    @property
    def policy(self) -> DevicePolicy:
        """Return the exact caller-supplied device policy."""
        return self._policy

    @property
    def checkpoint_labels(self) -> Mapping[str, Mapping[str, int]]:
        """Return output-column labels keyed by checkpoint role."""
        return self._checkpoint_labels

    @property
    def pinned(self) -> Mapping[str, str]:
        """Return the pinned pretrained checkpoint of every untrained loaded role.

        A role is pinned when its weights come from a pinned pretrained
        checkpoint and none of its parameters take a gradient, whether it was
        built frozen or the run's active groups leave it out, so the checkpoint
        identity reproduces them exactly.
        """
        return self._pinned

    def role_parameters(self) -> Mapping[str, tuple[nn.Parameter, ...]]:
        """Return the parameters each component role owns, in registration order.

        A parameter belongs to the most specific role whose module holds it, so
        a role nested inside another keeps its own parameters and the roles hold
        disjoint sets. A role whose module holds no parameter owns none.
        """
        named = self._topology.named_parameters
        owned: dict[str, tuple[nn.Parameter, ...]] = {}

        for role in self._topology.modules:
            if not self._topology.owns_parameters(role):
                owned[role] = ()
                continue

            members = self._groups.members(role)
            owned[role] = tuple(
                parameter for name, parameter in named.items() if name in members
            )

        return MappingProxyType(owned)

    def activate(self, roles: tuple[str, ...], /) -> GroupResolution:
        """Leave exactly the declared component roles trainable for one stage.

        The optimizer owns the complete parameter set of a staged run. This
        operation changes only gradient admission: inactive parameters remain
        in the optimizer and checkpoint topology but receive no gradient and
        thus cannot move during the stage.
        """
        resolution = self._topology.resolve(roles)

        for name, parameter in self._topology.named_parameters.items():
            parameter.grad = None
            parameter.requires_grad_(name in resolution.trainable)

        return resolution

    def _pinned_parameters(self, role: str, /) -> frozenset[str]:
        """Return the state names of a role's parameters a checkpoint omits."""
        if role not in self._pinned:
            return frozenset()

        return frozenset(
            name
            for name, _ in self._topology.modules[role].named_parameters(
                remove_duplicate=False
            )
        )

    def _live_state(self, role: str, /, *, pinned: bool) -> dict[str, StateTree]:
        """Return one role's live state, without its pinned parameters if asked."""
        live = self._topology.role_state(role)

        if not pinned:
            return live

        omitted = self._pinned_parameters(role)

        return {name: value for name, value in live.items() if name not in omitted}

    def state(self) -> Mapping[str, dict[str, StateTree]]:
        """Return detached complete state for every component role."""
        return MappingProxyType(
            {
                role: cast(
                    "dict[str, StateTree]",
                    _copy_state_tree(
                        cast("StateTree", self._live_state(role, pinned=False))
                    ),
                )
                for role in self.modules
            }
        )

    def checkpoint_state(self) -> Mapping[str, dict[str, StateTree]]:
        """Return detached checkpoint state for every component role.

        A pinned role's parameters are left out, since its manifest record
        names the pinned checkpoint they are loaded from.
        """
        return MappingProxyType(
            {
                role: cast(
                    "dict[str, StateTree]",
                    _copy_state_tree(
                        cast("StateTree", self._live_state(role, pinned=True))
                    ),
                )
                for role in self.modules
            }
        )

    def validate_restore(
        self, states: Mapping[str, Mapping[str, StateTree]], /
    ) -> None:
        """Refuse incomplete or incompatible role-keyed component state.

        A pinned role's state may be complete or may leave out every one of its
        parameters, as a checkpoint does.
        """
        expected = frozenset(self.modules)
        found = frozenset(states)

        if found != expected:
            msg = (
                f"component state roles are {sorted(found)}, not the complete "
                f"configured roles {sorted(expected)}"
            )
            raise ValueError(msg)

        for role in self.modules:
            saved = dict(states[role])
            live = self._live_state(
                role,
                pinned=self._pinned_parameters(role).isdisjoint(saved),
            )
            _validate_compatible_state(
                f"component role {role!r}",
                cast("StateTree", saved),
                cast("StateTree", live),
            )

    def prepare_restore(
        self, states: Mapping[str, Mapping[str, StateTree]], /
    ) -> Callable[[], None]:
        """Return a restore operation only after exact structural preflight."""
        self.validate_restore(states)
        prepared = {
            role: cast(
                "dict[str, StateTree]",
                _copy_state_tree(cast("StateTree", dict(states[role]))),
            )
            for role in self.modules
        }

        def restore() -> None:
            for role, module in self.modules.items():
                module.load_state_dict(prepared[role], strict=False)

        return restore

    def restore(self, states: Mapping[str, Mapping[str, StateTree]], /) -> None:
        """Restore complete role-keyed component state after full preflight."""
        self.prepare_restore(states)()


@final
class ParserFittingPlan(FrozenValue):
    """Immutable parser epochs and validation policy."""

    _noun = "parser fitting plan"

    __slots__ = (
        "_alternation",
        "_checkpoints",
        "_clamp_policy",
        "_clamp_report",
        "_constraint_selection",
        "_corpus",
        "_dataset",
        "_early_stopping",
        "_epoch_source",
        "_epochs",
        "_fit_split",
        "_forward",
        "_grammar",
        "_interpretation",
        "_pairs",
        "_stage_tasks",
        "_stages",
        "_step",
        "_steps_per_epoch",
        "_validation_stage",
    )

    def __init__(
        self,
        forward: _TrainingForward,
        step: _TrainingStep,
        epoch_source: EpochSource[_TrainingBatch],
        early_stopping: EarlyStopping,
        constraint_selection: SelectionReport[str] | None,
        clamp_policy: ClampPolicy[Term] | None,
        clamp_report: ClampReport | None,
        dataset: LoweredDataset,
        *,
        grammar: GrammarAssembly,
        corpus: SyntheticCorpus | None,
        pairs: PairCorpus | None,
        interpretation: InterpretationFunction | None,
        epochs: int,
        steps_per_epoch: int,
        checkpoints: CheckpointPolicy,
        fit_split: str,
        stages: tuple[Stage, ...] = (),
        stage_tasks: tuple[FittingTask, ...] = (),
        validation_stage: str | None = None,
        alternation: AlternatingStep | None = None,
    ) -> None:
        if alternation is not None and stages:
            msg = "a parser plan alternates its tasks or runs stages, not both"
            raise ValueError(msg)

        object.__setattr__(self, "_alternation", alternation)
        object.__setattr__(self, "_forward", forward)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_epoch_source", epoch_source)
        object.__setattr__(self, "_early_stopping", early_stopping)
        object.__setattr__(self, "_constraint_selection", constraint_selection)
        object.__setattr__(self, "_clamp_policy", clamp_policy)
        object.__setattr__(self, "_clamp_report", clamp_report)
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_corpus", corpus)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_interpretation", interpretation)
        object.__setattr__(self, "_epochs", epochs)
        object.__setattr__(self, "_steps_per_epoch", steps_per_epoch)
        object.__setattr__(self, "_checkpoints", checkpoints)
        object.__setattr__(self, "_fit_split", fit_split)
        object.__setattr__(self, "_stages", stages)
        object.__setattr__(self, "_stage_tasks", stage_tasks)
        object.__setattr__(self, "_validation_stage", validation_stage)

    @property
    def forward(self) -> _TrainingForward:
        """Return the bound parser forward pass."""
        return self._forward

    @property
    def grounding_metadata(self) -> GroundingMetadata | None:
        """Return the parser pass's stable grounding identity, when present."""
        return self._forward.grounding_metadata

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return the parser pass's latest detached grounding telemetry."""
        return self._step.grounding_telemetry

    @property
    def step(self) -> _TrainingStep:
        """Return the bound parser optimization step."""
        return self._step

    @property
    def batches(self) -> tuple[_TrainingBatch, ...]:
        """Return the first seeded training epoch."""
        return self.epoch(0).training

    @property
    def fit_batches(self) -> tuple[_TrainingBatch, ...]:
        """Return the first epoch's retained partition used during fitting."""
        return self.epoch(0).retained[self._fit_split]

    @property
    def retained_batches(self) -> Mapping[str, tuple[_TrainingBatch, ...]]:
        """Return the first epoch's exact retained partitions."""
        return self.epoch(0).retained

    @property
    def fit_split(self) -> str:
        """Return the retained partition used for epoch selection."""
        return self._fit_split

    def epoch(self, index: int, /) -> DatasetEpoch[_TrainingBatch]:
        """Return the deterministically selected batches for one declared epoch."""
        if type(index) is not int:
            msg = f"a parser epoch index is not an integer: {index!r}"
            raise TypeError(msg)

        if index < 0 or index >= self._epochs:
            msg = (
                f"parser epoch index {index} lies outside a horizon of "
                f"{self._epochs} epochs"
            )
            raise ValueError(msg)

        return self._epoch_source.at(index)

    @property
    def early_stopping(self) -> EarlyStopping:
        """Return the validation stopping rule."""
        return self._early_stopping

    @property
    def constraint_selection(self) -> SelectionReport[str] | None:
        """Return the achieved constraint-selection budget, if configured."""
        return self._constraint_selection

    @property
    def clamp_policy(self) -> ClampPolicy[Term] | None:
        """Return the configured clamp policy, if constraints were selected."""
        return self._clamp_policy

    @property
    def clamp_report(self) -> ClampReport | None:
        """Return the outcomes of applying the configured clamp policy."""
        return self._clamp_report

    @property
    def dataset(self) -> LoweredDataset:
        """Return the lowered source corpus both epochs are drawn from."""
        return self._dataset

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the model-owned learner grammar used by this plan."""
        return self._grammar

    @property
    def corpus(self) -> SyntheticCorpus | None:
        """Return the generated sentence corpus, when this is sentence training."""
        return self._corpus

    @property
    def pairs(self) -> PairCorpus | None:
        """Return the labeled pair corpus, when this is pair training."""
        return self._pairs

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the parser data's lowered interpretation, when declared."""
        return self._interpretation

    @property
    def items_per_epoch(self) -> int:
        """Return training occurrences consumed by one epoch."""
        return sum(map(len, self.batches))

    @property
    def epochs(self) -> int:
        """Return the epoch horizon."""
        return self._epochs

    @property
    def steps_per_epoch(self) -> int:
        """Return optimizer steps per epoch."""
        if self._alternation is not None:
            return len(self._alternation.schedule)

        return self._steps_per_epoch

    @property
    def alternation(self) -> AlternatingStep | None:
        """Return the interleaved tasks every epoch runs, if the plan alternates."""
        return self._alternation

    @property
    def checkpoints(self) -> CheckpointPolicy:
        """Return the checkpoint cadence, in completed epochs, and retention."""
        return self._checkpoints

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Return the declared active-group stages, empty for ordinary fitting."""
        return self._stages

    def stage_at(self, epoch: int, /) -> tuple[int, Stage, FittingTask] | None:
        """Return the stage and concrete task containing one zero-based epoch."""
        cursor = 0

        for position, (stage, task) in enumerate(
            zip(self._stages, self._stage_tasks, strict=True)
        ):
            cursor += stage.epochs

            if epoch < cursor:
                return position, stage, task

        return None

    def validates(self, epoch: int, /) -> bool:
        """Return whether one epoch owns parser validation and selection."""
        staged = self.stage_at(epoch)

        return staged is None or staged[1].name == self._validation_stage

    def with_stages(
        self,
        stages: tuple[Stage, ...],
        tasks: tuple[FittingTask, ...],
        validation_stage: str | None,
        /,
    ) -> ParserFittingPlan:
        """Return this parser plan under one complete concrete stage schedule."""
        if sum(stage.epochs for stage in stages) != self._epochs:
            msg = (
                f"stages cover {sum(stage.epochs for stage in stages)} epochs, "
                f"not the parser horizon of {self._epochs}"
            )
            raise ValueError(msg)

        if len(tasks) != len(stages):
            msg = (
                f"staged fitting declares {len(stages)} stages and "
                f"{len(tasks)} concrete tasks"
            )
            raise ValueError(msg)

        if validation_stage is not None and (
            not stages or stages[-1].name != validation_stage
        ):
            msg = (
                f"validation stage {validation_stage!r} is not the final concrete stage"
            )
            raise ValueError(msg)

        return ParserFittingPlan(
            self._forward,
            self._step,
            self._epoch_source,
            self._early_stopping,
            self._constraint_selection,
            self._clamp_policy,
            self._clamp_report,
            self._dataset,
            grammar=self._grammar,
            corpus=self._corpus,
            pairs=self._pairs,
            interpretation=self._interpretation,
            epochs=self._epochs,
            steps_per_epoch=self._steps_per_epoch,
            checkpoints=self._checkpoints,
            fit_split=self._fit_split,
            stages=stages,
            stage_tasks=tasks,
            validation_stage=validation_stage,
        )

    def with_alternation(self, alternation: AlternatingStep, /) -> ParserFittingPlan:
        """Return this parser plan with every epoch running interleaved tasks."""
        return ParserFittingPlan(
            self._forward,
            self._step,
            self._epoch_source,
            self._early_stopping,
            self._constraint_selection,
            self._clamp_policy,
            self._clamp_report,
            self._dataset,
            grammar=self._grammar,
            corpus=self._corpus,
            pairs=self._pairs,
            interpretation=self._interpretation,
            epochs=self._epochs,
            steps_per_epoch=self._steps_per_epoch,
            checkpoints=self._checkpoints,
            fit_split=self._fit_split,
            alternation=alternation,
        )

    def parser_plan(self) -> ParserFittingPlan:
        """Return this plan, which fits a parser."""
        return self

    def held_out_parser(self) -> ParserFittingPlan:
        """Return this plan, whose parser holds out its retained partitions."""
        return self

    @property
    def residual_keys(self) -> Mapping[str, str]:
        """Return the parser objective's residual names."""
        return self._step.assembly.residual_keys

    def timed_stack(self) -> TimedStack:
        """Return the parser steps of one epoch, or its alternating task steps."""
        alternation = self._alternation

        if alternation is not None:
            return AlternatingStack(alternation)

        return ParserStack(self)


@final
class StepFittingPlan(FrozenValue):
    """Immutable generated-batch step horizon, target and checkpoint policy.

    Parameters
    ----------
    trainers : tuple[GeneratedStep, ...]
        The steps the horizon visits in rotation, one per optimizer step.
    batch_seed : Callable[[int, int], int]
        The seed of the batch at one position of a stream, from the stream's
        seed and the position.
    stream_seed : int
        The training-batch stream seed.
    validation_seed : int | None
        The held-out batch stream seed, held exactly when ``stop`` is.
    stop : AccuracyStopping | None
        The held-out accuracy target, when declared.
    steps : int
        The optimizer-step horizon.
    batch_size : int
        The items drawn for one optimizer step.
    checkpoints : CheckpointPolicy
        The checkpoint cadence, in completed steps, and retention.
    dataset : LoweredDataset
        The dataset whose learner language supplies the generated terms.

    Raises
    ------
    ValueError
        If no trainer is given, or exactly one of the target and its held-out
        stream is.
    """

    _noun = "step fitting plan"

    __slots__ = (
        "_batch_seed",
        "_batch_size",
        "_checkpoints",
        "_dataset",
        "_steps",
        "_stop",
        "_stream_seed",
        "_trainers",
        "_validation_seed",
    )

    def __init__(
        self,
        trainers: tuple[GeneratedStep, ...],
        batch_seed: Callable[[int, int], int],
        stream_seed: int,
        *,
        validation_seed: int | None,
        stop: AccuracyStopping | None,
        steps: int,
        batch_size: int,
        checkpoints: CheckpointPolicy,
        dataset: LoweredDataset,
    ) -> None:
        if not trainers:
            msg = "a step fitting plan requires at least one step"
            raise ValueError(msg)

        if (stop is None) != (validation_seed is None):
            msg = "a step fitting plan holds a target exactly with its held-out stream"
            raise ValueError(msg)

        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_trainers", trainers)
        object.__setattr__(self, "_batch_seed", batch_seed)
        object.__setattr__(self, "_stream_seed", stream_seed)
        object.__setattr__(self, "_validation_seed", validation_seed)
        object.__setattr__(self, "_stop", stop)
        object.__setattr__(self, "_steps", steps)
        object.__setattr__(self, "_batch_size", batch_size)
        object.__setattr__(self, "_checkpoints", checkpoints)

    @property
    def trainers(self) -> tuple[GeneratedStep, ...]:
        """Return the steps the horizon visits in rotation."""
        return self._trainers

    def trainer_at(self, position: int, /) -> GeneratedStep:
        """Return the step scheduled at one position."""
        return self._trainers[position % len(self._trainers)]

    def batch_seed(self, stream_seed: int, position: int, /) -> int:
        """Return the seed of the batch at one position of a stream."""
        return self._batch_seed(stream_seed, position)

    @property
    def dataset(self) -> LoweredDataset:
        """Return the dataset whose learner language supplies generated terms."""
        return self._dataset

    @property
    def stream_seed(self) -> int:
        """Return the training-batch stream seed."""
        return self._stream_seed

    @property
    def validation_seed(self) -> int | None:
        """Return the held-out batch seed, when a target is declared."""
        return self._validation_seed

    @property
    def stop(self) -> AccuracyStopping | None:
        """Return the held-out accuracy stopping rule, when declared."""
        return self._stop

    @property
    def steps(self) -> int:
        """Return the optimizer-step horizon."""
        return self._steps

    @property
    def batch_size(self) -> int:
        """Return the items drawn for one optimizer step."""
        return self._batch_size

    @property
    def checkpoints(self) -> CheckpointPolicy:
        """Return the checkpoint cadence in completed optimizer steps."""
        return self._checkpoints

    @property
    def grounding_metadata(self) -> GroundingMetadata | None:
        """Return no grounding identity for generated-batch fitting."""
        return None

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return no grounding telemetry for generated-batch fitting."""
        return None

    def parser_plan(self) -> Never:
        """Refuse, since a plan counted in generated-batch steps fits no parser.

        Raises
        ------
        ValueError
            Always.
        """
        msg = "a step-counted plan fits no parser"
        raise ValueError(msg)

    def held_out_parser(self) -> None:
        """Return no parser, since generated draws hold out no partition."""

    @property
    def residual_keys(self) -> Mapping[str, str]:
        """Return the compatible residual names across the rotated steps.

        Raises
        ------
        ValueError
            If two steps assign one residual role to different names.
        """
        merged: dict[str, str] = {}

        for trainer in self.trainers:
            for role, key in trainer.assembly.residual_keys.items():
                previous = merged.get(role)

                if previous is not None and previous != key:
                    msg = (
                        f"step objectives assign residual role {role!r} to both "
                        f"{previous!r} and {key!r}"
                    )
                    raise ValueError(msg)

                merged[role] = key

        return MappingProxyType(merged)

    def timed_stack(self) -> TimedStack:
        """Return the generated-batch step scheduled at each horizon position."""
        return StepStack(self)


type FittingPlan = ParserFittingPlan | StepFittingPlan


@final
class PreparedTraining(FrozenValue):
    """Declaration-free model, fitting plan and optimizer for one run."""

    _noun = "prepared training"

    __slots__ = (
        "_model",
        "_optimization",
        "_plan",
        "_prediction",
    )

    def __init__(
        self,
        model: ConfiguredModel,
        optimization: AutocastPolicy,
        plan: FittingPlan,
        prediction: TrainedPointPredictor | None,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_optimization", optimization)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_prediction", prediction)

    @property
    def model(self) -> ConfiguredModel:
        """Return the configured model graph."""
        return self._model

    @property
    def optimization(self) -> AutocastPolicy:
        """Return the optimization policy."""
        return self._optimization

    @property
    def plan(self) -> FittingPlan:
        """Return the closed parser-epoch or distance-step fitting plan."""
        return self._plan

    @property
    def prediction(self) -> TrainedPointPredictor | None:
        """Return arbitrary inference prediction when the run prepares it."""
        return self._prediction

    @property
    def grounding_metadata(self) -> GroundingMetadata | None:
        """Return the prepared plan's stable grounding identity, when present."""
        return self._plan.grounding_metadata

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return the prepared plan's latest detached grounding telemetry."""
        return self._plan.grounding_telemetry

    def state(self) -> tuple[Mapping[str, dict[str, StateTree]], OptimizationState]:
        """Return detached component and optimization checkpoint state."""
        optimization = self._optimization.state()

        return (
            self._model.checkpoint_state(),
            OptimizationState(
                cast(
                    "dict[str, StateTree]",
                    _copy_state_tree(cast("StateTree", optimization.optimizer)),
                ),
                scaler=(
                    None
                    if optimization.scaler is None
                    else cast(
                        "dict[str, StateTree]",
                        _copy_state_tree(cast("StateTree", optimization.scaler)),
                    )
                ),
                scheduler=(
                    None
                    if optimization.scheduler is None
                    else cast(
                        "dict[str, StateTree]",
                        _copy_state_tree(cast("StateTree", optimization.scheduler)),
                    )
                ),
            ),
        )

    def restore(
        self,
        components: Mapping[str, Mapping[str, StateTree]],
        optimization: OptimizationState,
        /,
    ) -> None:
        """Restore components and optimization after both surfaces preflight."""
        restore_components = self._model.prepare_restore(components)
        restore_optimization = self._optimization.prepare_restore(optimization)
        restore_components()
        restore_optimization()
