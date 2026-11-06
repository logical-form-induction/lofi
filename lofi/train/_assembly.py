"""Configured training assembly.

A validated run declaration becomes prepared training: built components, a
weighted objective, selected constraints and a fitting plan. A checkpoint
manifest resumes prepared training at its recorded position.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, fsum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Protocol,
    assert_never,
    cast,
    final,
    override,
    runtime_checkable,
)

import torch
from torch import nn

from lofi._frozen import FrozenValue, Unpicklable
from lofi._types import ComponentId, ItemId, ReadoutId
from lofi.data import (
    ConstraintScope,
    Example,
    HashedResources,
    Item,
    ItemLookup,
    PairDistribution,
    Relational,
    TypeDistribution,
)
from lofi.data._epochs import EpochBatching
from lofi.data._selection_factory import build_selection_policy
from lofi.data.synthetic import (
    ConstraintSourceContext,
    RulePairSamplerContext,
    TypeSamplerContext,
    build_arity_branching,
    build_child_depth,
    synthetic_inference_item,
)
from lofi.data.synthetic.training import InferenceTrainingBatch, SyntheticTrainingBatch
from lofi.deduce import SpanIndex
from lofi.lexicon import (
    Clamp,
    ClampMode,
    ClampReport,
    ClampScope,
    UnsatisfiableAction,
    UnsatisfiableClampError,
)
from lofi.nn._inference_response import resolve_response
from lofi.nn._snapshots import artifact_fetcher
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.grounding import GroundingQuery, InferenceGrounding, Phase
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._autoencoder import TypeAutoencoderObjective
from lofi.nn.loss._combinator import CombinatorSupervisionObjective
from lofi.nn.loss._constraint import (
    BatchLocalConstraint,
    BatchLocalNegativeBracket,
    Constraint,
    ConstraintOccurrence,
    NegativeBracketConstraint,
)
from lofi.nn.loss._controller import ControllerSupervisionObjective
from lofi.nn.loss._distance import TypeDistanceObjective
from lofi.nn.loss._ports import (
    CombinatorScorer,
    Objective,
)
from lofi.nn.loss._schedule import TeacherForcingSchedule
from lofi.nn.loss._warmup import TokenWarmupObjective
from lofi.nn.parsers import MaximumReduction, ScoreChannels, TaggedInsideKernel
from lofi.nn.ports import (
    Controller,
    SubwordPooler,
    TeacherForcedDecoder,
    TermEncoder,
    TokenEncoder,
)
from lofi.nn.schedules import ScheduleContext
from lofi.nn.sharing import ModuleSharingSubject, module_sharing_matches
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    TokenVocabulary,
)
from lofi.nn.tokens.pretrained import HuggingFacePretrainedLoader
from lofi.registry import Registry, plugin_for
from lofi.spec._wiring import (
    check_codec_identity,
    check_realized_sharing,
    resolve_dependencies,
    resolve_sharing,
)
from lofi.spec.components.data import (
    HeldOutDatasetSpec,
    RetainedDatasetSpec,
)
from lofi.spec.components.model import ModelRole
from lofi.spec.constraint_policy import PolicyDomain
from lofi.spec.readouts import TargetSource
from lofi.spec.trainers.generation import autoencoding_objective, combinator_objective
from lofi.terms import EncodingOverflowError, TargetSet, Term, TermCodec
from lofi.train._alternating import (
    AlternatingStep,
    AlternatingTask,
    DatasetTask,
    StepTask,
)
from lofi.train._autocast import (
    AutocastPolicy,
    GradScaling,
    OptimizationState,
    StepOptions,
)
from lofi.train._boundary import CheckpointManifest, disagreements
from lofi.train._checkpoint import (
    Checkpoint,
    CheckpointPolicy,
    FingerprintMismatchError,
)
from lofi.train._configured import (
    ConfiguredModel,
    ParserFittingPlan,
    PreparedTraining,
    StepFittingPlan,
    _ConfiguredTopology,
)
from lofi.train._early_stopping import AccuracyStopping, EarlyStopping
from lofi.train._grounded import GroundedForward, GroundedStep
from lofi.train._groups import ActiveGroups
from lofi.train._losses import (
    LossAssembly,
    LossBindingContext,
    ProofRewardContext,
    WeightedObjective,
    WeightSchedule,
)
from lofi.train._parsing import ChartForward, ParserStep
from lofi.train._pretraining import (
    ControllerStep,
    DistanceBatcher,
    DistanceEncoderStep,
    Draw,
    LinearDepthCurriculum,
    PairDraws,
    RuleBatch,
    TermsBatch,
    TermsStep,
    TypeBatch,
    distance_batch_seed,
    terms_batch_seed,
)
from lofi.train._resume import (
    ResumePosition,
    _copy_state_tree,
    _restore_checkpoint,
    _validate_state_tree,
)
from lofi.train._spaces import (
    assemble_space_graph,
    build_space_encoders,
    wire_spaces,
)
from lofi.train._staging import Stage
from lofi.train._term_pairs import MeaningPairSampler, MorphismPairSampler
from lofi.train._warmup import TokenWarmupStep, Utterances, WarmupBatch
from lofi.training_plugins import (
    StepBuildContext,
    TrainerBuildContext,
    trainer_plugin,
    training_sampler_stream,
)
from lofi.training_plugins.mixing import mixing
from lofi.training_plugins.steps import step_plugin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from lofi._types import SubmoduleDeclaration
    from lofi.data import (
        ConstraintScope,
        ConstraintSelection,
        ConstraintSource,
        InventoriedTypeSampler,
        PairSampler,
        SelectionPolicy,
        SelectionReport,
        SpanConstraint,
        Supervision,
    )
    from lofi.data._ports import CodeTensorizer, Featurizer, FeaturizerResources
    from lofi.data._supervision import Unlabeled
    from lofi.data.datasets import (
        CheckedDataset,
        DatasetVocabulary,
        LoweredDataset,
        PairTraining,
        SentenceTraining,
    )
    from lofi.data.featurizers._core import FeaturizedBatch
    from lofi.data.sampler import BatchSampler
    from lofi.data.synthetic import SyntheticCorpus
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.judge import Action, ActionSet
    from lofi.judgment.bead import InferenceText
    from lofi.lexicon import ClampPolicy
    from lofi.lf import InferenceLabel
    from lofi.nn.loss._ports import (
        PublishedOutputs,
        SpanTokenDecoder,
    )
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.policy import DevicePolicy
    from lofi.nn.ports import ChartScorer
    from lofi.nn.sharing import (
        ModuleSharingProducer,
        StateTensor,
        SubmodulePublication,
    )
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.nn.tokens.pretrained import PretrainedLoader
    from lofi.registry import Builder
    from lofi.spec import (
        ComponentSpec,
        ConfiguredSpec,
        LossSection,
        ModelSection,
        RunSpec,
        SchedulerSpec,
        WeightedLossSpec,
    )
    from lofi.spec._ports import Dependency
    from lofi.spec.components.model import GroundingSpec, ParserSpec
    from lofi.spec.constraint_policy import ConstraintPolicySpec
    from lofi.spec.constraint_sources import GoldConstraintSourceSettings
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.losses.inference import InferenceLossSettings
    from lofi.spec.spaces import CombinatorFamilySpec, SpaceSpec
    from lofi.spec.trainers.composed import AlternatingTrainerSpec, StagedTrainerSpec
    from lofi.spec.trainers.distance import DistanceTrainerSpec
    from lofi.spec.trainers.generation import TermGenerationSpec
    from lofi.spec.trainers.parser import ParserTrainerSpec
    from lofi.spec.trainers.steps import (
        AlternatingStepSpec,
        ControllerStepSpec,
        GroundedStepSpec,
        JudgmentStepSpec,
        ParserStepSpec,
        StepSpec,
        SupervisedWarmStartStepSpec,
        TermsStepSpec,
        TokenWarmupStepSpec,
    )
    from lofi.spec.trainers.terms import TermsTrainerSpec
    from lofi.terms import ChoiceIndex
    from lofi.train._losses import LossPlugin, LossRole
    from lofi.train._ports import Forward, RestorableRandomStreams
    from lofi.training_plugins.steps import FittingTask
from lofi.train._store import StateTree

__all__ = [
    "ComponentGraph",
    "constraint_source",
    "prepare_training",
    "resolve_loss_section",
    "resume_from_checkpoint",
    "select_constraints",
    "wire_components",
]

type _AlternatingBatch = (
    SyntheticTrainingBatch[
        BatchLocalConstraint[Term, ConstraintScope]
        | BatchLocalNegativeBracket[ConstraintScope]
    ]
    | InferenceTrainingBatch
)


class _TrainingBlueprint(Protocol):
    """Read-only planning values consumed by seeded preparation."""

    @property
    def run_spec(self) -> RunSpec:
        """Return the validated run declaration."""
        ...

    @property
    def checked_datasets(self) -> Mapping[str, CheckedDataset]:
        """Return checked dataset inputs keyed by their declared role."""
        ...

    def checked_dataset(self, role: str, /) -> CheckedDataset:
        """Return checked inputs for one declared dataset role."""
        ...

    def lower_dataset(self, role: str, /) -> LoweredDataset:
        """Lower and validate one checked dataset."""
        ...

    def featurizer_resources(
        self, role: str, dataset: LoweredDataset, tensorizer: CodeTensorizer, /
    ) -> FeaturizerResources:
        """Return the resources a featurizer over one lowered dataset is built from."""
        ...

    def dataset_vocabulary(
        self, role: str, dataset: LoweredDataset, /
    ) -> DatasetVocabulary:
        """Return one lowered dataset's token vocabulary."""
        ...

    def sentence_training(
        self, role: str, dataset: LoweredDataset, /
    ) -> SentenceTraining:
        """Return the sentence training one lowered dataset's plugin supplies."""
        ...

    def pair_training(self, role: str, dataset: LoweredDataset, /) -> PairTraining:
        """Return the inference pair training one lowered dataset's plugin supplies."""
        ...

    @property
    def sampler_streams(self) -> tuple[str, ...]:
        """Return the named sampler streams preparation consumes."""
        ...

    @property
    def actions(self) -> ActionSet:
        """Return the controller actions the run's grammar defines."""
        ...

    @property
    def learner(self) -> _LearnerAssembly:
        """Return the model-owned learner grammar and interpretation."""
        ...


class _LearnerAssembly(Protocol):
    """Expose the resolved model-owned symbolic assembly."""

    @property
    def grammar(self) -> GrammarAssembly:
        """Return the learner grammar."""
        ...

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the optional learner interpretation."""
        ...


@runtime_checkable
class _UnaryClosureDeclaration(Protocol):
    """Expose the unary closure bound of a parser that applies grammar rules."""

    def unary_closure_depth(self) -> int:
        """Return the number of unary rule rounds the parser applies per cell."""
        ...


@runtime_checkable
class _CategoryTransitionDeclaration(Protocol):
    """Expose that a parser admits only grammar-licensed category transitions."""

    def category_transition_reachability(self) -> bool:
        """Return whether category reachability follows the grammar transitions."""
        ...


def _parser_unary_depth(parser: ParserSpec | None, /) -> int:
    """Return a parser's published unary closure bound, or zero without closure."""
    return (
        parser.unary_closure_depth()
        if isinstance(parser, _UnaryClosureDeclaration)
        else 0
    )


def _parser_uses_category_transitions(parser: ParserSpec | None, /) -> bool:
    """Return a parser's published category-transition capability."""
    return isinstance(parser, _CategoryTransitionDeclaration) and (
        parser.category_transition_reachability()
    )


def _approximation(run_spec: RunSpec, /) -> ApproximationBound | None:
    """Return the run level residual budget as a runtime checker."""
    declared = run_spec.approximation
    return None if declared is None else ApproximationBound(declared.residual_mass)


type _ComponentStatePreparer = Callable[
    [Mapping[str, Mapping[str, StateTree]]], Callable[[], None]
]
"""Nonmutating preparation of one complete configured component state."""


class _ConstraintCorpus(Protocol):
    """The sentence corpus a constraint source selects over."""

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the sentence corpus."""
        ...

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the corpus was read under."""
        ...

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation the corpus was read under, when declared."""
        ...


def _sentences(
    blueprint: _TrainingBlueprint, role: str, /
) -> tuple[LoweredDataset, SentenceTraining]:
    """Return one lowered dataset and the sentence training its plugin supplies."""
    dataset = blueprint.lower_dataset(role)

    return dataset, blueprint.sentence_training(role, dataset)


def _pairs(
    blueprint: _TrainingBlueprint, role: str, /
) -> tuple[LoweredDataset, PairTraining]:
    """Return one lowered dataset and the pair training its plugin supplies."""
    dataset = blueprint.lower_dataset(role)

    return dataset, blueprint.pair_training(role, dataset)


def _batching(
    trainer: ParserTrainerSpec,
    streams: RestorableRandomStreams,
    role: str,
    /,
) -> EpochBatching:
    """Return the declared batching of one dataset role's epochs."""
    return EpochBatching(
        cast(
            "BatchSampler",
            _construct(cast("ComponentSpec", trainer.batch_sampler), None),
        ),
        batch_size=trainer.batch_size,
        training_seed=streams.sampler_seed(
            training_sampler_stream(role, "training-batches")
        ),
        validation_seed=streams.sampler_seed(
            training_sampler_stream(role, "validation-batches")
        ),
    )


def _token_vocabulary(
    blueprint: _TrainingBlueprint, role: str, dataset: LoweredDataset, /
) -> TokenVocabulary:
    """Return the neural token vocabulary declared by the dataset plugin."""
    declared = blueprint.dataset_vocabulary(role, dataset)
    return TokenVocabulary(
        declared.forms,
        reserved=ReservedTokens(declared.reserved),
        unknown=declared.unknown,
    )


def _state_mapping(name: str, state: StateTree) -> dict[str, StateTree]:
    """Return one state block keyed by state name.

    Parameters
    ----------
    name : str
        The role or run state part the block belongs to.
    state : StateTree
        The loaded state block.

    Raises
    ------
    ValueError
        If the block is not keyed only by state names.
    """
    _validate_state_tree(name, state)

    if not isinstance(state, dict) or any(type(key) is not str for key in state):
        msg = f"state for {name!r} is not keyed by state names"

        raise ValueError(msg)

    return cast("dict[str, StateTree]", state)


def _resume_component_states(
    checkpoint: Checkpoint[StateTree],
    /,
) -> dict[str, dict[str, StateTree]]:
    """Return detached component state keyed by every declared role.

    Parameters
    ----------
    checkpoint : Checkpoint[StateTree]
        The loaded checkpoint.

    Raises
    ------
    ValueError
        If any component block is not keyed by state names.
    """
    prepared: dict[str, dict[str, StateTree]] = {}

    for role in sorted(checkpoint.manifest.roles):
        state = _state_mapping(role, checkpoint.states[role])
        prepared[role] = cast(
            "dict[str, StateTree]", _copy_state_tree(cast("StateTree", state))
        )

    return prepared


def _resume_optimization_state(
    checkpoint: Checkpoint[StateTree],
) -> OptimizationState:
    """Return complete optimizer, scaler and scheduler state.

    Parameters
    ----------
    checkpoint : Checkpoint[StateTree]
        The loaded checkpoint.

    Raises
    ------
    ValueError
        If the manifest names no optimizer or the run state blocks do not cover
        exactly its declared topology.
    """
    manifest = checkpoint.manifest
    expected_keys = frozenset(manifest.references)
    found_keys = frozenset(checkpoint.run_state)
    missing_keys = sorted(expected_keys - found_keys)
    extra_keys = sorted(found_keys - expected_keys)

    if missing_keys or extra_keys:
        msg = f"run state has missing keys {missing_keys} and extra keys {extra_keys}"

        raise ValueError(msg)

    optimizer = manifest.optimizer

    if optimizer is None:
        msg = "a resumable checkpoint names no optimizer state"

        raise ValueError(msg)

    scaler = manifest.scaler
    scheduler = manifest.scheduler

    return OptimizationState(
        _state_mapping("optimizer", checkpoint.run_state[optimizer.key]),
        scaler=(
            None
            if scaler is None
            else _state_mapping("scaler", checkpoint.run_state[scaler.key])
        ),
        scheduler=(
            None
            if scheduler is None
            else _state_mapping("scheduler", checkpoint.run_state[scheduler.key])
        ),
    )


def _resume_random_state(
    checkpoint: Checkpoint[StateTree],
    streams: RestorableRandomStreams,
) -> dict[str, str]:
    """Return complete random-stream state for the declared streams.

    Parameters
    ----------
    checkpoint : Checkpoint[StateTree]
        The loaded checkpoint.
    streams : RestorableRandomStreams
        The run's initialized random streams.

    Raises
    ------
    ValueError
        If the checkpoint omits a declared stream or names an extra one.
    """
    state = checkpoint.manifest.rng.streams
    expected_keys = frozenset(streams.snapshot())
    found_keys = frozenset(state)
    missing_keys = sorted(expected_keys - found_keys)
    extra_keys = sorted(found_keys - expected_keys)

    if missing_keys or extra_keys:
        msg = (
            f"random-stream state has missing keys {missing_keys} and extra keys "
            f"{extra_keys}"
        )

        raise ValueError(msg)

    return dict(state)


def resume_from_checkpoint(
    checkpoint: Checkpoint[StateTree],
    expected: CheckpointManifest,
    prepare_components: _ComponentStatePreparer,
    prepare_optimization: Callable[[OptimizationState], Callable[[], None]],
    streams: RestorableRandomStreams,
    /,
) -> ResumePosition:
    """Restore one completed optimizer-step checkpoint after a complete preflight.

    Parameters
    ----------
    checkpoint : Checkpoint[StateTree]
        The loaded checkpoint to restore.
    expected : CheckpointManifest
        The manifest the live run would write for itself.
    prepare_components : _ComponentStatePreparer
        The configured component graph's nonmutating restore preparation operation.
    prepare_optimization : Callable[[OptimizationState], Callable[[], None]]
        The optimization policy's nonmutating restore preparation operation.
    streams : RestorableRandomStreams
        Initialized random streams declared for the run.

    Returns
    -------
    ResumePosition
        The completed step and next data position recovered from the checkpoint.

    Raises
    ------
    FingerprintMismatchError
        If the checkpoint manifest is incompatible with the live run.
    ValueError
        If component, run state or random-stream coverage is incomplete.
    """
    reasons = disagreements(expected, checkpoint.manifest)

    if reasons:
        raise FingerprintMismatchError(checkpoint.checkpoint, reasons)

    component_states = _resume_component_states(checkpoint)
    optimization_state = _resume_optimization_state(checkpoint)
    random_state = _resume_random_state(checkpoint, streams)
    restore_components = prepare_components(component_states)
    restore_optimization = prepare_optimization(optimization_state)
    restore_random = streams.prepare_restore(random_state)
    position = ResumePosition(
        checkpoint.manifest.step,
        checkpoint.manifest.data_position,
    )

    return _restore_checkpoint(
        restore_components,
        restore_optimization,
        restore_random,
        position,
    )


def _built_loss[BatchT](
    term: WeightedLossSpec,
    registry: Registry[LossPlugin[BatchT]],
    context: LossBindingContext,
    /,
) -> Objective[PublishedOutputs[BatchT]]:
    """Return one declared objective built through the plugin its tag names.

    Parameters
    ----------
    term : WeightedLossSpec
        Validated objective declaration, weight and name.
    registry : Registry[LossPlugin[BatchT]]
        Registry resolving the declaration's tag to its plugin.
    context : LossBindingContext
        The term's name and the built components the objective may read, by role.
    """
    plugin = registry.get(term.tag)

    return plugin.construct(cast("ConfiguredSpec[object]", term.objective), context)


def resolve_loss_section[BatchT](
    section: LossSection,
    registry: Registry[LossPlugin[BatchT]],
    /,
    *,
    published: frozenset[str] | None = None,
    unavailable_outputs: Mapping[str, str] | None = None,
    readouts: Mapping[str, TermsReadout],
    spaces: Mapping[str, Space],
    readout_spaces: Mapping[str, Space],
    relations: Mapping[str, SpaceRelationBinding],
    span_decoders: Mapping[str, SpanTokenDecoder],
    proof_reward: ProofRewardContext | None,
) -> LossAssembly[BatchT]:
    """Build the declared loss terms through the loss registry.

    Parameters
    ----------
    section : LossSection
        The validated loss section of a run.
    registry : Registry[LossPlugin[BatchT]]
        The registry that resolves each term's tag to its loss plugin.
    published : frozenset[str] | None
        The names the configured graph publishes, or None to accept any objective
        whose declared names are in the published vocabulary.
    unavailable_outputs : Mapping[str, str] | None
        Configuration diagnostics keyed by outputs the graph deliberately does not
        publish, or None when no absent output has a more specific explanation.
    readouts : Mapping[str, TermsReadout]
        Built term readouts keyed by their declared roles.
    spaces : Mapping[str, Space]
        Built represented spaces keyed by their declared roles.
    readout_spaces : Mapping[str, Space]
        The represented space each term readout observes.
    relations : Mapping[str, SpaceRelationBinding]
        Cross-space relations keyed by source-space role.
    span_decoders : Mapping[str, SpanTokenDecoder]
        Built span decoders keyed by span-decoder role.
    proof_reward : ProofRewardContext | None
        Logical form language and shared proof service for proof-shaped credit,
        or None when the selected data declares no interpretation.

    Returns
    -------
    LossAssembly[BatchT]
        The built objectives under their names and weights, in section order.

    Raises
    ------
    UnknownTagError
        If a term's tag resolves to no plugin.
    MisnamedObjectiveError
        If a built objective does not report its term's declared name.
    MissingOutputError
        If an objective declares a name the configured graph does not publish.
    ValueError
        If an objective requires an output the graph deliberately withholds under
        the selected configuration.
    """
    built_readouts: Mapping[str, LossRole] = readouts
    built_spaces: Mapping[str, LossRole] = spaces
    entries = tuple(
        WeightedObjective(
            term.key,
            term.weight,
            schedule=(
                None
                if term.schedule is None
                else WeightSchedule(term.schedule.start, term.schedule.steps)
            ),
            objective=_built_loss(
                term,
                registry,
                LossBindingContext(
                    name=term.key,
                    readouts=built_readouts,
                    spaces=built_spaces,
                    readout_spaces=readout_spaces,
                    relations=relations,
                    span_decoders=span_decoders,
                    proof_reward=proof_reward,
                ),
            ),
        )
        for term in section.terms
    )

    if published is not None and unavailable_outputs is not None:
        for entry in entries:
            missing = entry.objective.requires - published

            for output in sorted(missing):
                reason = unavailable_outputs.get(output)

                if reason is not None:
                    msg = f"loss term {entry.name!r} cannot read {output}: {reason}"
                    raise ValueError(msg)

    return LossAssembly(entries, published=published)


def _step_options(
    run_spec: RunSpec,
    /,
    *,
    accumulation_steps: int = 1,
    max_grad_norm: float | None,
) -> StepOptions:
    """Return a trainer's step options under the run's declared gradient scaler.

    Parameters
    ----------
    run_spec : RunSpec
        The run whose device policy declares the gradient scaler, if any.
    accumulation_steps : int
        How many accumulation units make one optimizer step.
    max_grad_norm : float | None
        The norm gradients are clipped to, or None to leave them unclipped.

    Returns
    -------
    StepOptions
        Accumulation, clipping, and the declared dynamic gradient scaling.
    """
    scaler = run_spec.device_policy.scaler

    return StepOptions(
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        scaling=(
            None
            if scaler is None
            else GradScaling(
                init_scale=scaler.init_scale,
                growth_factor=scaler.growth_factor,
                backoff_factor=scaler.backoff_factor,
                growth_interval=scaler.growth_interval,
            )
        ),
    )


def _construct[C](declaration: ComponentSpec, context: C, /) -> object:
    """Build one declared component through the plugin its family lists.

    Parameters
    ----------
    declaration : ComponentSpec
        A validated declaration of a component family whose entries are plugins.
    context : C
        What the family's builders take beside the declaration's settings.

    Raises
    ------
    UnknownTagError
        If the declaration's family lists no plugin under its tag.
    RegistryError
        If the entry names something other than the declaration's plugin.
    """
    configured = cast("ConfiguredSpec[object]", declaration)

    return plugin_for(configured).construct(configured, context)


type _Subject = ModuleSharingSubject[nn.Parameter, StateTensor]
type _Producer = ModuleSharingProducer[nn.Parameter, StateTensor]


@final
class _Unregistered:
    """A built component that registers no parameters or saved state of its own."""

    __slots__ = ()

    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Return no parameters."""
        return ()

    def state_dict(self) -> Mapping[str, StateTensor]:
        """Return no saved state."""
        return MappingProxyType({})

    @property
    def shared(self) -> Mapping[str, SubmodulePublication[nn.Parameter, StateTensor]]:
        """Return no publications."""
        return MappingProxyType({})


def _registered(component: object, /) -> _Producer:
    """Return a component as what it registers, or as registering nothing.

    A component that is not a module, such as a readout or a decoder family,
    holds what it reads without registering it, so it realizes no sharing.
    """
    if isinstance(component, ModuleSharingSubject):
        return cast("_Producer", component)

    return _Unregistered()


def _module_sharing_matches(
    consumer: object, producer: object, declaration: SubmoduleDeclaration | None
) -> bool:
    """Compare the sharing two built components realize with a declaration."""
    return module_sharing_matches(
        cast("_Subject", _registered(consumer)), _registered(producer), declaration
    )


@runtime_checkable
class _CodecReader(Protocol):
    """A built component that reads and writes terms through one codec view."""

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view the component was built with."""
        ...


@final
class ComponentGraph(FrozenValue, Unpicklable):
    """The built components of one wired graph and the edges between them.

    Parameters
    ----------
    order : tuple[str, ...]
        Every role, each after the roles it depends on.
    components : Mapping[str, object]
        The built component filling each role.
    producers : Mapping[tuple[str, str], str]
        The producer role supplying each consumer role and input port.
    sharing : Mapping[tuple[str, str], SubmoduleDeclaration]
        The submodule each consumer port holds from its producer. A port absent
        here shares nothing.
    """

    _noun = "component graph"
    _because = "holds built modules"

    __slots__ = ("_components", "_order", "_producers", "_sharing")

    def __init__(
        self,
        *,
        order: tuple[str, ...],
        components: Mapping[str, object],
        producers: Mapping[tuple[str, str], str],
        sharing: Mapping[tuple[str, str], SubmoduleDeclaration],
    ) -> None:
        object.__setattr__(self, "_order", order)
        object.__setattr__(
            self,
            "_components",
            MappingProxyType({role: components[role] for role in order}),
        )
        object.__setattr__(self, "_producers", MappingProxyType(dict(producers)))
        object.__setattr__(self, "_sharing", MappingProxyType(dict(sharing)))

    @property
    def order(self) -> tuple[str, ...]:
        """Return the roles in construction order."""
        return self._order

    @property
    def components(self) -> Mapping[str, object]:
        """Return the built components in construction order."""
        return self._components

    def dependency(self, consumer: str, port: str, /) -> object:
        """Return the component wired to one consumer port.

        Parameters
        ----------
        consumer : str
            The consuming role.
        port : str
            The port the consumer reads its producer under.

        Raises
        ------
        KeyError
            If no dependency arrives on that port.
        """
        return self._components[self._producers[consumer, port]]

    def sharing(self, consumer: str, port: str, /) -> SubmoduleDeclaration | None:
        """Return the submodule one consumer port holds, if it holds one.

        Parameters
        ----------
        consumer : str
            The consuming role.
        port : str
            The port the consumer reads its producer under.
        """
        return self._sharing.get((consumer, port))


@final
class _DerivedDependencies(FrozenValue):
    """One declaration with the producer edges its run derives beside it.

    Parameters
    ----------
    declaration : ComponentSpec
        The component declaration.
    derived : tuple[Dependency, ...]
        The edges the run derives from the rest of the model, after the
        declaration's own.
    """

    _noun = "derived component dependency"

    __slots__ = ("_declaration", "_derived")

    def __init__(
        self, declaration: ComponentSpec, derived: tuple[Dependency, ...], /
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_derived", derived)

    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared dependencies, then the derived ones."""
        return (*self._declaration.dependencies(), *self._derived)

    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the declaration's submodule sharing."""
        return self._declaration.sharing()

    def publications(self) -> frozenset[str]:
        """Return the declaration's lent submodules."""
        return self._declaration.publications()


def wire_components(
    specifications: Mapping[str, ComponentSpec],
    run: BuildContext,
    /,
    *,
    pretrained_resources: Mapping[str, PretrainedLoader],
    built: Mapping[str, object],
    derived: Mapping[str, tuple[Dependency, ...]],
    role_targets: Mapping[str, Mapping[str, TargetSet[Term]]] | None = None,
) -> ComponentGraph:
    """Build a declared component graph through each declaration's plugin.

    Dependencies and submodule sharing are resolved from the declarations before
    anything is built. Each role is then built in dependency order with its
    producers. A built component that holds a codec view other than the run's is
    refused before any consumer is built from it, and the built graph is refused
    unless every edge realizes exactly its declared sharing. A component that is
    not a module registers nothing it holds, so each of its edges must declare no
    sharing.

    Parameters
    ----------
    specifications : Mapping[str, ComponentSpec]
        The declarations of the graph, by role.
    run : BuildContext
        The run-wide resources every component is built with.
    pretrained_resources : Mapping[str, PretrainedLoader]
        The loader each role's pretrained weights come through, where it is not
        the run's.
    built : Mapping[str, object]
        Components of declared roles that were built under their own context
        before the graph, such as term encoders bound to their represented
        sort. They are producers only: no edge may arrive at one.
    derived : Mapping[str, tuple[Dependency, ...]]
        Producer edges a role reads beside its declared ones, derived from the
        rest of the model, by consumer role.
    role_targets : Mapping[str, Mapping[str, TargetSet[Term]]] | None
        The target sets each role scores, by source name, where they are not the
        run's. None gives every role the run's.

    Raises
    ------
    ValueError
        If a pretrained resource names a role the graph does not hold, a role
        whose component loads no pretrained weights or a role built before the
        graph; if a built component, a derived edge or a target override names
        a role the graph does not hold; or if an edge arrives at a built role.
    WiringError
        If the declared dependencies or sharing do not resolve, or the built
        components do not realize the declared sharing.
    LinearizationDisagreementError
        If a built component holds a codec view other than the run's.
    """
    for what, roles in (("built components", built), ("derived edges", derived)):
        absent = sorted(set(roles) - set(specifications))

        if absent:
            msg = f"{what} name absent component roles {absent}"
            raise ValueError(msg)

    sources = {
        role: (
            _DerivedDependencies(declaration, derived[role])
            if role in derived
            else declaration
        )
        for role, declaration in specifications.items()
    }
    dependency_plan = resolve_dependencies(sources)
    order = dependency_plan.order
    edges = dependency_plan.producers
    sharing = resolve_sharing(sources, edges)
    fed = sorted({consumer for consumer, _ in edges if consumer in built})

    if fed:
        msg = f"components built before the graph cannot read producers: {fed}"
        raise ValueError(msg)

    extra_resources = sorted(set(pretrained_resources) - set(specifications))

    if extra_resources:
        msg = f"construction resources name absent component roles {extra_resources}"
        raise ValueError(msg)

    prebuilt_resources = sorted(set(pretrained_resources) & set(built))

    if prebuilt_resources:
        msg = (
            "construction resources name components built before the graph "
            f"{prebuilt_resources}"
        )
        raise ValueError(msg)

    unread_resources = sorted(
        role
        for role in pretrained_resources
        if specifications[role].pretrained_weights() is None
    )

    if unread_resources:
        msg = (
            f"construction resources name roles whose components load no "
            f"pretrained weights {unread_resources}"
        )
        raise ValueError(msg)

    targets = {} if role_targets is None else role_targets
    untargeted = sorted(set(targets) - set(specifications))

    if untargeted:
        msg = f"target overrides name absent component roles {untargeted}"
        raise ValueError(msg)

    components: dict[str, object] = {}

    for role in order:
        if role in built:
            components[role] = built[role]
            continue

        producers = {
            port: components[producer]
            for (consumer, port), producer in edges.items()
            if consumer == role
        }
        context = run.for_component(ComponentId(role), producers)

        if role in pretrained_resources:
            context = context.with_loader(pretrained_resources[role])

        if role in targets:
            context = context.with_targets(targets[role])

        component = build_component(
            cast("ConfiguredSpec[object]", specifications[role]), context
        )

        if isinstance(component, _CodecReader):
            check_codec_identity(str(run.component), run.codec, {role: component.codec})

        components[role] = component

    check_realized_sharing(components, edges, sharing, _module_sharing_matches)

    return ComponentGraph(
        order=order, components=components, producers=edges, sharing=sharing
    )


type _LocalConstraint = (
    BatchLocalConstraint[Term, ConstraintScope]
    | BatchLocalNegativeBracket[ConstraintScope]
)
type _TargetConstraint = tuple[ReadoutId, Constraint[Term, ConstraintScope]]
type _GlobalConstraint = _TargetConstraint | NegativeBracketConstraint[ConstraintScope]


class _ConstraintSourceDeclaration(Protocol):
    """A validated constraint-source declaration with runtime settings."""

    def settings(self) -> GoldConstraintSourceSettings:
        """Return the settings passed to the source plugin."""


@final
class _SyntheticConstraintBatcher:
    """Localize selected synthetic constraints to each parser minibatch."""

    __slots__ = ("_constraints",)

    def __init__(
        self,
        constraints: tuple[_GlobalConstraint, ...],
    ) -> None:
        self._constraints = constraints

    def for_items(
        self, item_ids: Sequence[ItemId], /
    ) -> tuple[tuple[_LocalConstraint, ...], ...]:
        """Return selected constraints aligned to items in this batch."""
        rows: list[list[_LocalConstraint]] = [[] for _ in item_ids]

        for constraint in self._constraints:
            if isinstance(constraint, NegativeBracketConstraint):
                localized_constraints: tuple[_LocalConstraint, ...] = (
                    constraint.to_spans(item_ids)
                )
            else:
                readout, target_constraint = constraint
                localized_constraints = target_constraint.to_spans(
                    item_ids, readout=readout
                )

            for localized in localized_constraints:
                for position in localized.item_positions:
                    rows[position].append(localized)

        return tuple(tuple(row) for row in rows)


@final
class _LearnerReachabilityChart(FrozenValue, Unpicklable):
    """Category reachability of one neural chart length."""

    __slots__ = ("_index", "_rows")
    _noun = "learner reachability chart"
    _because = "holds symbolic category identities"

    def __init__(
        self,
        index: SpanIndex,
        rows: tuple[tuple[Term, ...], ...],
        /,
    ) -> None:
        if len(rows) != index.num_spans:
            msg = (
                f"a reachability chart for length {index.n} has {len(rows)} rows, "
                f"not {index.num_spans}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_rows", rows)

    def categories_at(self, span: tuple[int, int], /) -> tuple[tuple[Term, float], ...]:
        """Return the categories the neural transition table reaches at a span."""
        row = self._rows[self._index.span_id(*span)]

        return tuple((category, 0.0) for category in row)


@final
class _LearnerCategorySupport(FrozenValue, Unpicklable):
    """Finite learner-owned categories and charts shared by sentence length."""

    __slots__ = ("_brackets", "_categories", "_charts")
    _noun = "learner category support"
    _because = "holds symbolic chart identities"

    def __init__(
        self,
        categories: tuple[Term, ...],
        charts: Mapping[int, _LearnerReachabilityChart],
        brackets: _MandatoryBrackets,
        /,
    ) -> None:
        object.__setattr__(self, "_categories", categories)
        object.__setattr__(self, "_charts", MappingProxyType(dict(charts)))
        object.__setattr__(self, "_brackets", brackets)

    @property
    def categories(self) -> tuple[Term, ...]:
        """Return codec-representable category classes in stable order."""
        return self._categories

    @property
    def charts(self) -> Mapping[int, _LearnerReachabilityChart]:
        """Return one exhaustive neural reachability chart per input length."""
        return self._charts

    @property
    def brackets(self) -> _MandatoryBrackets:
        """Return the decision of which spans every derivation brackets."""
        return self._brackets


class _MandatoryBrackets(Protocol):
    """Decide which spans every derivation the parser counts brackets."""

    def mandatory(self, length: int, span: tuple[int, int], /) -> bool:
        """Return whether every counted derivation of an input brackets a span."""


@final
class _TerminalBrackets(FrozenValue):
    """The spans every derivation brackets whatever its grammar."""

    __slots__ = ()
    _noun = "terminal bracket decision"

    def mandatory(self, length: int, span: tuple[int, int], /) -> bool:
        """Return whether the span is one token or the whole input."""
        return span[1] - span[0] == 1 or span == (0, length)


@final
class _DerivedBrackets(FrozenValue, Unpicklable):
    """The spans every goal derivation of a tagged transition table brackets."""

    __slots__ = ("_goals", "_kernel")
    _noun = "derived bracket decision"
    _because = "holds symbolic category identities"

    def __init__(
        self, kernel: TaggedInsideKernel[Term], goals: tuple[Term, ...], /
    ) -> None:
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_goals", goals)

    def mandatory(self, length: int, span: tuple[int, int], /) -> bool:
        """Return whether no goal derivation of the length omits the span.

        A length with no goal derivation at all has none that omits it.
        """
        if span[1] - span[0] == 1 or span == (0, length):
            return True

        index = SpanIndex(length)

        with torch.inference_mode():
            required = self._kernel.requires_span(
                _unweighted_channels(index, self._kernel),
                index,
                span,
                goals=self._goals,
            )

        return bool(required[0])


def _learner_category_inventory(
    learner: GrammarAssembly,
    codec: NeuralCodecView[TermCodec],
    /,
) -> tuple[Term, ...]:
    """Return the codec-representable classes of the learner's finite closure."""
    grammar = learner.grammar
    matching = grammar.matching
    held: list[Term] = []
    representable: list[Term] = []
    overflow: list[tuple[Term, EncodingOverflowError]] = []

    for category in grammar.closure.reached:
        try:
            codec.codec.encode(category)
        except EncodingOverflowError as error:
            overflow.append((category, error))
            continue

        representable.append(category)

        if not any(matching.equivalent(category, prior) for prior in held):
            held.append(category)

    for goal in grammar.targets.categories:
        exact_error = next(
            (error for category, error in overflow if category == goal),
            None,
        )
        if exact_error is not None:
            raise exact_error

        if not any(category == goal for category in representable):
            codec.codec.encode(goal)

        if any(matching.equivalent(goal, category) for category in held):
            continue

        msg = f"grammar goal {goal!r} is outside its declared finite closure"
        raise ValueError(msg)

    return tuple(held)


def _unweighted_channels(
    index: SpanIndex, kernel: TaggedInsideKernel[Term], /
) -> ScoreChannels:
    """Return one item's channels admitting every category, step and lift."""
    transitions = kernel.transitions
    unary = (
        None
        if not transitions.unary_rules
        else torch.zeros((1, index.num_spans, len(transitions.unary_rules)))
    )

    return ScoreChannels(
        torch.zeros((1, index.n, len(transitions.categories))),
        torch.zeros((1, index.num_spans, index.max_splits, len(transitions.rules))),
        unary,
    )


def _learner_reachability_chart(
    length: int,
    kernel: TaggedInsideKernel[Term],
    goals: tuple[Term, ...],
    /,
) -> _LearnerReachabilityChart:
    """Compile the categories some goal derivation uses at each span."""
    index = SpanIndex(length)
    transitions = kernel.transitions

    with torch.inference_mode():
        support = kernel.goal_support(
            _unweighted_channels(index, kernel), index, goals=goals
        )

    rows = tuple(
        tuple(
            category
            for column, category in enumerate(transitions.categories)
            if bool(support[0, index.span_id(*span), column].any())
        )
        for span in index.spans_by_length()
    )

    return _LearnerReachabilityChart(index, rows)


def _dense_category_chart(
    length: int,
    categories: tuple[Term, ...],
    /,
) -> _LearnerReachabilityChart:
    """Return dense readout support for a parser without category state."""
    index = SpanIndex(length)

    return _LearnerReachabilityChart(
        index,
        tuple(categories for _ in range(index.num_spans)),
    )


def _learner_category_support(
    items: Sequence[Item],
    learner: GrammarAssembly,
    categories: tuple[Term, ...],
    parser: ParserSpec | None,
    /,
) -> _LearnerCategorySupport:
    """Return the learner inventory and exact neural support by input length."""
    lengths = tuple(dict.fromkeys(len(item.tokens) for item in items))
    if _parser_uses_category_transitions(parser):
        kernel = TaggedInsideKernel(
            MaximumReduction(),
            learner.grammar.rules,
            categories,
            unary_depth=_parser_unary_depth(parser),
        )
        goals = tuple(learner.grammar.targets.categories)
        charts = {
            length: _learner_reachability_chart(length, kernel, goals)
            for length in lengths
        }

        return _LearnerCategorySupport(
            categories, charts, _DerivedBrackets(kernel, goals)
        )

    charts = {length: _dense_category_chart(length, categories) for length in lengths}

    return _LearnerCategorySupport(categories, charts, _TerminalBrackets())


def _constraint_readout_role(run_spec: RunSpec, /) -> str | None:
    """Return the one readout role declared for selected constraints."""
    if run_spec.loss is None:
        return None

    roles = tuple(
        dependency.role
        for term in run_spec.loss.terms
        for dependency in term.objective.dependencies()
        if dependency.port == "constraint-readout"
    )

    return None if not roles else roles[0]


def _constraint_evidence(
    corpus: SyntheticCorpus, /
) -> tuple[Example[Supervision[str]], ...]:
    """Return observed items with gold derivations retained where available."""
    gold = {example.item.item_id: example for example in corpus.examples}

    return tuple(
        cast(
            "Example[Supervision[str]]",
            gold.get(example.item.item_id, example),
        )
        for example in corpus.observed()
    )


def _constraint_catalog(
    selection: ConstraintSelection[str, str],
    grammar: GrammarAssembly,
    support: _LearnerCategorySupport,
    /,
    *,
    domain: PolicyDomain,
    policy: ClampPolicy[Term],
) -> tuple[
    Mapping[str, TargetSet[Term]],
    _SyntheticConstraintBatcher,
    ClampReport,
]:
    """Build one stable target catalog and its dataset-global constraints."""
    if not policy.require_constituent:
        active_brackets: list[
            tuple[
                Example[Supervision[str]],
                SpanConstraint[str],
                tuple[tuple[int, int], ...],
            ]
        ] = []
        dropped_clamps = 0
        skipped_items: set[ItemId] = set()
        decided: dict[tuple[int, tuple[int, int]], bool] = {}

        def mandatory(length: int, span: tuple[int, int], /) -> bool:
            """Return the support's bracket decision, once per length and span."""
            key = (length, span)

            if key not in decided:
                decided[key] = support.brackets.mandatory(length, span)

            return decided[key]

        for example in selection.examples:
            length = len(example.item.tokens)

            for constraint in example.supervision.constraints:
                span = constraint.span
                spans = (
                    tuple(
                        (position, position + 1)
                        for position in range(span.start, span.end)
                    )
                    if policy.scope is ClampScope.OCCURRENCE
                    else ((span.start, span.end),)
                )
                guaranteed = next(
                    (held for held in spans if mandatory(length, held)),
                    None,
                )

                if guaranteed is None:
                    active_brackets.append((example, constraint, spans))
                    continue

                action = policy.on_unsatisfiable.action
                if action is UnsatisfiableAction.ERROR:
                    position = guaranteed[0]
                    form = (
                        example.item.tokens[position].surface
                        if guaranteed[1] - guaranteed[0] == 1
                        else str(example.item.item_id)
                    )
                    raise UnsatisfiableClampError(
                        clamp=Clamp(
                            guaranteed,
                            mode=policy.mode,
                            penalty=policy.penalty,
                            scope=policy.scope,
                            require_constituent=False,
                        ),
                        position=position,
                        form=form,
                        offered=(),
                    )

                if action is UnsatisfiableAction.DROP_CLAMP:
                    dropped_clamps += 1
                    continue

                skipped_items.add(example.item.item_id)

        brackets = tuple(
            NegativeBracketConstraint(
                tuple(
                    ConstraintOccurrence(example.item.item_id, span) for span in spans
                ),
                constraint.scope,
            )
            for example, constraint, spans in active_brackets
            if example.item.item_id not in skipped_items
        )

        return (
            MappingProxyType({}),
            _SyntheticConstraintBatcher(brackets),
            ClampReport(dropped_clamps, len(skipped_items)),
        )

    def read_category(text: str, /) -> Term:
        """Read category text into the grammar's store."""
        return grammar.syntax.parse(grammar.store, text)

    match domain:
        case PolicyDomain.SYNTACTIC:
            parse_label = read_category
            matching = policy.matching or grammar.grammar.matching
        case PolicyDomain.SEMANTIC:
            parse_label = grammar.language.read
            matching = policy.meaning_matching
        case unreachable:
            assert_never(unreachable)

    categories = support.categories
    category_matching = grammar.grammar.matching

    def is_supported(category: Term, /) -> bool:
        """Return whether the finite parser state owns this category class."""
        return any(
            category_matching.equivalent(category, admitted) for admitted in categories
        )

    def categories_at(
        example: Example[Supervision[str]], span: tuple[int, int], /
    ) -> tuple[Term, ...]:
        """Return learner-derived categories at one constrained span."""
        chart = support.charts[len(example.item.tokens)]

        return tuple(
            category
            for category, _ in chart.categories_at(span)
            if is_supported(category)
        )

    occurrences = tuple(
        (
            example,
            constraint,
            frozenset(parse_label(label) for label in constraint.permitted),
        )
        for example in selection.examples
        for constraint in example.supervision.constraints
    )
    active: list[
        tuple[Example[Supervision[str]], SpanConstraint[str], frozenset[Term]]
    ] = []
    dropped_clamps = 0
    skipped_items: set[ItemId] = set()

    for example, constraint, declared in occurrences:
        unsatisfied: tuple[int, str, tuple[Term, ...]] | None = None
        permitted = declared

        if domain is PolicyDomain.SYNTACTIC:
            permitted = frozenset(
                admitted
                for admitted in categories
                if any(matching.equivalent(admitted, wanted) for wanted in declared)
            )

            if not permitted:
                if policy.negate:
                    continue

                unsatisfied = (
                    constraint.span.start,
                    str(example.item.item_id),
                    categories,
                )

        if (
            unsatisfied is None
            and policy.mode is ClampMode.HARD
            and domain is PolicyDomain.SYNTACTIC
        ):
            matching = policy.matching or grammar.grammar.matching
            span = constraint.span
            token_level = policy.scope is ClampScope.OCCURRENCE or (
                span.end - span.start == 1
            )
            candidates = (
                tuple(
                    (
                        position,
                        example.item.tokens[position].surface,
                        categories_at(example, (position, position + 1)),
                    )
                    for position in range(span.start, span.end)
                )
                if token_level
                else (
                    (
                        span.start,
                        str(example.item.item_id),
                        categories_at(example, (span.start, span.end)),
                    ),
                )
            )

            for position, form, offered in candidates:
                admitted = tuple(
                    candidate
                    for candidate in offered
                    if any(
                        matching.equivalent(candidate, wanted) for wanted in permitted
                    )
                    is not policy.negate
                )

                if not admitted:
                    unsatisfied = position, form, offered
                    break

        if unsatisfied is None:
            active.append((example, constraint, permitted))
            continue

        action = policy.on_unsatisfiable.action

        if action is UnsatisfiableAction.ERROR:
            position, form, offered = unsatisfied
            raise UnsatisfiableClampError(
                clamp=Clamp(
                    (constraint.span.start, constraint.span.end),
                    permitted,
                    mode=policy.mode,
                    penalty=policy.penalty,
                    scope=policy.scope,
                    require_constituent=policy.require_constituent,
                    negate=policy.negate,
                ),
                position=position,
                form=form,
                offered=offered,
            )

        if action is UnsatisfiableAction.DROP_CLAMP:
            dropped_clamps += 1
            continue

        skipped_items.add(example.item.item_id)

    if skipped_items:
        active = [
            occurrence
            for occurrence in active
            if occurrence[0].item.item_id not in skipped_items
        ]

    parsed = {permitted for _, _, permitted in active}
    ordered = tuple(
        sorted(
            parsed,
            key=lambda categories: tuple(
                category.key()
                for category in sorted(categories, key=lambda category: category.key())
            ),
        )
    )
    named = {
        categories: ReadoutId(f"pinning-{position:06d}")
        for position, categories in enumerate(ordered)
    }
    targets = {
        str(named[categories]): TargetSet(
            sorted(categories, key=lambda category: category.key()),
            matching,
        )
        for categories in ordered
    }
    constraints = tuple(
        (
            named[categories],
            Constraint(
                targets[str(named[categories])],
                tuple(
                    ConstraintOccurrence(item_id, span)
                    for span in (
                        tuple(
                            (position, position + 1)
                            for position in range(
                                constraint.span.start, constraint.span.end
                            )
                        )
                        if policy.scope is ClampScope.OCCURRENCE
                        else ((constraint.span.start, constraint.span.end),)
                    )
                ),
                constraint.scope,
                clamp_scope=policy.scope,
                mode=policy.mode,
                penalty=policy.penalty,
                negate=policy.negate,
            ),
        )
        for example, constraint, categories in active
        for item_id in (example.item.item_id,)
    )

    return (
        MappingProxyType(targets),
        _SyntheticConstraintBatcher(constraints),
        ClampReport(dropped_clamps, len(skipped_items)),
    )


def _selection_policy(
    declaration: ConstraintPolicySpec, /
) -> SelectionPolicy[str, str]:
    """Return the selection policy one constraint-policy declaration describes.

    Parameters
    ----------
    declaration : ConstraintPolicySpec
        A validated syntactic or semantic constraint-selection policy.
    """
    return cast(
        "SelectionPolicy[str, str]", build_selection_policy(declaration.settings())
    )


def constraint_source(
    run_spec: RunSpec, dataset_role: str, dataset: _ConstraintCorpus, /
) -> ConstraintSource[str, str] | None:
    """Return the run's declared constraint source, or None where none is declared.

    The source selects under the run policy of the label domain it declares.

    Parameters
    ----------
    run_spec : RunSpec
        The validated run declaration.
    dataset : _ConstraintCorpus
        The corpus the source selects over, with its grammar and interpretation.
    """
    data = run_spec.data.datasets[dataset_role]
    declaration = data.constraint_source

    if declaration is None:
        return None

    policies = data.constraint_policy
    grammar = dataset.source_grammar

    return cast(
        "ConstraintSource[str, str]",
        _construct(
            declaration,
            ConstraintSourceContext(
                syntactic=_selection_policy(policies.syntactic),
                semantic=_selection_policy(policies.semantic),
                lexicon=grammar.lexicon,
                render=lambda category: grammar.syntax.render(grammar.store, category),
                interpretation=dataset.source_interpretation,
            ),
        ),
    )


def select_constraints(
    run_spec: RunSpec, dataset_role: str, dataset: _ConstraintCorpus, /
) -> ConstraintSelection[str, str] | None:
    """Return what the run's declared constraint source selects on its corpus.

    Parameters
    ----------
    run_spec : RunSpec
        The validated run declaration.
    dataset : _ConstraintCorpus
        The corpus the source selects over.

    Returns
    -------
    ConstraintSelection[str, str] | None
        The constrained observed items, the load report and the achieved
        budget, or None where the run declares no source.
    """
    source = constraint_source(run_spec, dataset_role, dataset)

    if source is None:
        return None

    return source.select(_constraint_evidence(dataset.corpus))


def _selected_constraints(
    run_spec: RunSpec,
    dataset_role: str,
    dataset: _ConstraintCorpus,
    learner: GrammarAssembly,
    categories: tuple[Term, ...],
    /,
) -> tuple[
    SelectionReport[str] | None,
    Mapping[str, TargetSet[Term]],
    _SyntheticConstraintBatcher | None,
    ClampPolicy[Term] | None,
    ClampReport | None,
]:
    """Select configured constraints once and lower their target catalog."""
    selection = select_constraints(run_spec, dataset_role, dataset)

    if selection is None:
        return None, MappingProxyType({}), None, None, None

    declaration = cast(
        "_ConstraintSourceDeclaration",
        run_spec.data.datasets[dataset_role].constraint_source,
    )
    settings = declaration.settings()
    support = _learner_category_support(
        tuple(example.item for example in selection.examples),
        learner,
        categories,
        run_spec.model.parser,
    )
    targets, batcher, report = _constraint_catalog(
        selection,
        learner,
        support,
        domain=settings.policy,
        policy=settings.clamp,
    )

    return selection.budget, targets, batcher, settings.clamp, report


def _pretrained_weights(
    specifications: Mapping[str, ComponentSpec], /
) -> dict[str, str]:
    """Return the pinned checkpoint of every role whose weights are loaded.

    Parameters
    ----------
    specifications : Mapping[str, ComponentSpec]
        The declarations of the wired roles.
    """
    return {
        role: identity
        for role, declaration in specifications.items()
        if (identity := declaration.pretrained_weights()) is not None
    }


def _pretrained_loader(run_spec: RunSpec, /) -> PretrainedLoader | None:
    """Return the loader the run's pretrained weights are fetched through.

    Parameters
    ----------
    run_spec : RunSpec
        The run, whose artifacts cache every pinned release is fetched into.

    Returns
    -------
    PretrainedLoader | None
        A loader reading every release from the directory the run's fetcher
        materializes, or None for a run that declares no artifacts cache.
    """
    cache = run_spec.artifacts

    if cache is None:
        return None

    return HuggingFacePretrainedLoader(artifact_fetcher(cache.settings()))


def _scheduler(
    declaration: SchedulerSpec | None, context: ScheduleContext, /
) -> LRScheduler | None:
    """Return the declared learning rate schedule, or None where none is declared.

    Parameters
    ----------
    declaration : SchedulerSpec | None
        The run's schedule declaration, if it declares one.
    context : ScheduleContext
        The optimizer the schedule drives and the horizon it spans.
    """
    if declaration is None:
        return None

    return cast("LRScheduler", _construct(declaration, context))


def _parser_base_components(
    model: ModelSection,
    run: BuildContext,
    encoders: Mapping[str, TermEncoder],
    resources: Mapping[str, PretrainedLoader],
    category_sort: str,
    /,
) -> dict[str, object]:
    """Wire the components declared beside the represented spaces.

    The term encoders are already built against their sorts, so they enter the
    graph as producers. A controller reads the cells of the one represented
    category space's encoder, an edge derived from the declared spaces.
    """
    specifications = model.base_components()
    controller = model.controller
    derived: dict[str, tuple[Dependency, ...]] = {}

    if controller is not None:
        category_spaces = tuple(
            role
            for role, declaration in model.spaces.items()
            if declaration.sort == category_sort
        )

        if len(category_spaces) != 1:
            msg = (
                "a controller-backed parser requires exactly one represented "
                f"category space; found {category_spaces!r}"
            )
            raise ValueError(msg)

        encoder = model.spaces[category_spaces[0]].encoder
        derived[ModelRole.CONTROLLER] = (controller.encoder_dependency(encoder),)

    graph = wire_components(
        {**specifications, **{role: model.encoders[role] for role in encoders}},
        run,
        pretrained_resources=resources,
        built=encoders,
        derived=derived,
    )

    return {
        role: component
        for role, component in graph.components.items()
        if role not in encoders
    }


def _retained_fit_split(run_spec: RunSpec, dataset_role: str, /) -> str:
    """Return the exact retained partition used for epoch selection."""
    declaration = run_spec.data.datasets[dataset_role].dataset
    if not isinstance(declaration, RetainedDatasetSpec):
        msg = (
            f"parser dataset {dataset_role!r} does not declare named retained "
            "partitions"
        )
        raise TypeError(msg)
    retained = declaration.retained_splits()
    requested = frozenset(
        split
        for evaluator in run_spec.evaluators.values()
        if (split := evaluator.fit_split()) is not None
    )
    if len(requested) > 1:
        msg = (
            f"evaluators fit on incompatible retained partitions: {sorted(requested)!r}"
        )
        raise ValueError(msg)
    if requested:
        selected = next(iter(requested))
        if selected not in retained:
            msg = (
                f"evaluators fit on {selected!r}, but dataset {dataset_role!r} "
                f"retains {sorted(retained)!r}"
            )
            raise ValueError(msg)
        return selected
    if isinstance(declaration, HeldOutDatasetSpec):
        selected = declaration.held_out_split()
        if selected not in retained:
            msg = (
                f"dataset {dataset_role!r} selects {selected!r} for fitted "
                f"validation, but retains {sorted(retained)!r}"
            )
            raise ValueError(msg)
        return selected
    if len(retained) != 1:
        msg = (
            f"dataset {dataset_role!r} retains {sorted(retained)!r}; an evaluator "
            "must declare which partition fits epoch selection"
        )
        raise ValueError(msg)
    return next(iter(retained))


def _build_grounding(
    declaration: ComponentSpec,
    run: BuildContext,
    pairs: PairTraining,
    interpretation: InterpretationFunction,
    readout_spaces: Mapping[str, Space],
    pooler: SubwordPooler,
    /,
) -> tuple[InferenceGrounding, Mapping[str, object]]:
    """Build one grounding and its independently declared nested components."""
    nested_declarations = cast("GroundingSpec", declaration).nested_components()
    nested = {
        role: _construct(
            component,
            run.for_component(
                ComponentId(f"grounding.{role}"),
                {
                    **readout_spaces,
                    "normalization": interpretation.normalization,
                    "service": interpretation.service,
                },
            ),
        )
        for role, component in nested_declarations.items()
    }
    producers: dict[str, object] = {
        **readout_spaces,
        **nested,
        "labeling": pairs.labeling,
        "sentences": pooler,
    }
    built = _construct(
        declaration,
        run.for_component(ComponentId("grounding"), producers),
    )

    if not isinstance(built, InferenceGrounding):
        msg = f"a grounding declaration built {built!r}, not an InferenceGrounding"
        raise TypeError(msg)

    return built, MappingProxyType(nested)


class _InferenceObjective(Protocol):
    """A grounded objective that lowers its response declaration to settings."""

    def settings(self) -> InferenceLossSettings:
        """Return the response settings consumed by the objective."""
        ...


def _response_rows(
    loss: LossSection,
    labels: tuple[InferenceLabel, ...],
    /,
) -> tuple[tuple[float, ...], ...]:
    """Return response probabilities by grounded-label row."""
    grounded = tuple(
        term.objective
        for term in loss.terms
        if term.objective.grounding_query() is not None
    )

    if len(grounded) != 1:
        msg = (
            "trained-point prediction requires exactly one grounded response, "
            f"not {len(grounded)}"
        )
        raise ValueError(msg)

    response = cast("_InferenceObjective", grounded[0]).settings().response

    return resolve_response(response).response_rows(labels)


@final
class _InferencePoint:
    """Predict arbitrary inference text through one configured grounded graph."""

    __slots__ = (
        "_build_featurizer",
        "_forward",
        "_hasher",
        "_model",
        "_relation_labels",
        "_response_rows",
        "_settings",
        "_tensorizer",
    )

    def __init__(
        self,
        model: ConfiguredModel,
        forward: GroundedForward[InferenceTrainingBatch],
        relation_labels: tuple[InferenceLabel, ...],
        response_rows: tuple[tuple[float, ...], ...],
        build_featurizer: Builder[
            object,
            FeaturizerResources,
            object,
        ],
        settings: object,
        resources: FeaturizerResources,
        /,
    ) -> None:
        if len(response_rows) != len(relation_labels):
            msg = (
                f"a response instrument has {len(response_rows)} rows for "
                f"{len(relation_labels)} grounded labels"
            )
            raise ValueError(msg)

        widths = {len(row) for row in response_rows}

        if len(widths) != 1 or not widths or next(iter(widths)) < 2:
            msg = "a trained-point response instrument has one width of at least two"
            raise ValueError(msg)

        self._model = model
        self._forward = forward
        self._relation_labels = relation_labels
        self._response_rows = response_rows
        self._build_featurizer = build_featurizer
        self._settings = settings
        self._hasher = resources.hasher
        self._tensorizer = resources.tensorizer

    @property
    def response_levels(self) -> int:
        """Return the width of the declared response distribution."""
        return len(self._response_rows[0])

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        """Return grounded-label probability columns in model order."""
        return self._relation_labels

    def _batch(self, pairs: Sequence[InferenceText], /) -> InferenceTrainingBatch:
        items: list[Item] = []
        examples: list[Example[Relational]] = []
        pair_ids: list[ItemId] = []
        labels: list[InferenceLabel] = []
        reference = self._relation_labels[0]

        for position, pair in enumerate(pairs):
            premise = synthetic_inference_item(
                ItemId(f"prediction/{position}/premise/{pair.item_id}"),
                pair.premise,
            )
            hypothesis = synthetic_inference_item(
                ItemId(f"prediction/{position}/hypothesis/{pair.item_id}"),
                pair.hypothesis,
            )
            items.extend((premise, hypothesis))
            examples.append(
                Example(premise, Relational(hypothesis.item_id, reference.value))
            )
            pair_ids.append(ItemId(pair.item_id))
            labels.append(reference)

        resources = HashedResources(
            self._hasher,
            self._tensorizer,
            ItemLookup(items),
        )
        featurizer = cast(
            "Featurizer[Relational, FeaturizedBatch[Relational]]",
            self._build_featurizer(self._settings, resources),
        )
        featured = featurizer(examples)
        by_id = {item.item_id: item for item in items}
        words = tuple(
            tuple(token.surface for token in by_id[item_id].tokens)
            for item_id in featured.batch.item_ids
        )

        return InferenceTrainingBatch(
            featured,
            words,
            pair_ids,
            labels,
            (),
            step=0,
        )

    def _relation_probabilities(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        if not pairs:
            return ()

        batch = self._batch(pairs)
        seen: set[int] = set()
        collected: list[nn.Module] = []

        for root in self._model.modules.values():
            for module in root.modules():
                identity = id(module)

                if identity not in seen:
                    seen.add(identity)
                    collected.append(module)

        modules = tuple(collected)
        modes = tuple(module.training for module in modules)

        try:
            for module in modules:
                module.eval()

            with torch.inference_mode():
                outputs = self._forward.publish(batch)
        finally:
            for module, mode in zip(modules, modes, strict=True):
                module.train(mode)

        grounded = outputs.entailment

        if grounded is None:
            msg = "the trained-point prediction forward published no entailment"
            raise RuntimeError(msg)

        probabilities = grounded.log_probabilities.tensor.detach().exp().cpu()
        return tuple(
            tuple(float(value) for value in probabilities[row])
            for row in range(len(pairs))
        )

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        """Return grounded-label probabilities for arbitrary inference text."""
        return self._relation_probabilities(pairs)

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        """Marginalize grounded-label mass through the declared response."""
        return tuple(
            tuple(
                fsum(
                    probability * instrument[level]
                    for probability, instrument in zip(
                        relation_row, self._response_rows, strict=True
                    )
                )
                for level in range(self.response_levels)
            )
            for relation_row in self._relation_probabilities(pairs)
        )


def _prepare_grounded_training(
    blueprint: _TrainingBlueprint,
    run_spec: RunSpec,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
    *,
    trainer: ParserTrainerSpec,
    pretrained_resources: Mapping[str, PretrainedLoader],
) -> PreparedTraining:
    """Assemble parser and inference grounding over labeled inference pairs."""
    dataset, pairs = _pairs(blueprint, trainer.dataset)
    grammar = blueprint.learner.grammar
    interpretation = blueprint.learner.interpretation
    if interpretation is None:
        msg = "grounded training requires a learner interpretation"
        raise ValueError(msg)
    language = grammar.language
    codec = NeuralCodecView(language.codec.at(language.category_sort))
    model = run_spec.model
    categories = _learner_category_inventory(grammar, codec)
    vocabulary = _token_vocabulary(blueprint, trainer.dataset, dataset)
    actions = blueprint.actions
    grammar_targets = {TargetSource.GRAMMAR_GOAL: grammar.grammar.targets}
    run = BuildContext(
        component=ComponentId("model"),
        policy=policy,
        language=language,
        codec=codec,
        vocabulary=vocabulary,
        actions=actions,
        rules=grammar.grammar.rules,
        categories=categories,
        producers={},
        targets=grammar_targets,
        loader=_pretrained_loader(run_spec),
        meaning=blueprint.learner.interpretation,
        approximation=_approximation(run_spec),
    )
    declaration = model.grounding

    if declaration is None:
        msg = "grounded training requires a model grounding declaration"
        raise ValueError(msg)

    base_specifications = model.base_components()
    base_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in base_specifications
    }
    encoder_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in model.encoders
    }
    encoders = build_space_encoders(model, run, encoder_resources)
    base_components = _parser_base_components(
        model,
        run,
        encoders,
        base_resources,
        language.category_sort,
    )
    scorer = cast("ChartScorer", base_components[ModelRole.PARSER])
    decoder_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in model.decoders
    }
    expected_resources = (
        set(base_resources) | set(encoder_resources) | set(decoder_resources)
    )
    extra_resources = sorted(set(pretrained_resources) - expected_resources)

    if extra_resources:
        msg = f"construction resources name absent component roles {extra_resources}"
        raise ValueError(msg)

    space_graph = assemble_space_graph(
        model,
        run,
        encoders,
        parser=scorer,
        pretrained_resources=decoder_resources,
        erasure=interpretation.erasure,
    )
    components = {**base_components, **space_graph.components}
    tokens = cast("TokenEncoder", components[ModelRole.TOKENS_ENCODER])
    pooler = cast("SubwordPooler", components[ModelRole.SUBWORD_POOLER])
    spaces = dict(space_graph.spaces)
    readout_spaces = {
        role: spaces[readout.space] for role, readout in model.readouts.items()
    }
    grounding, nested = _build_grounding(
        declaration,
        run,
        pairs,
        interpretation,
        readout_spaces,
        pooler,
    )
    components.update(nested)
    components["grounding"] = grounding
    readouts = dict(space_graph.readouts)
    chart = ChartForward[InferenceTrainingBatch](
        tokens,
        pooler,
        scorer,
        readouts=tuple(readouts.values()),
        actions=actions,
    )
    form_readouts = declaration.form_readouts()
    query = cast("LossSection", run_spec.loss).grounding_query()

    if query is None:
        msg = "grounding assembly requires one declared inference response"
        raise ValueError(msg)

    premise_readout = None if form_readouts is None else ReadoutId(form_readouts[0])
    hypothesis_readout = None if form_readouts is None else ReadoutId(form_readouts[1])
    forward = GroundedForward(
        chart,
        grounding,
        query=GroundingQuery(query),
        premise_readout=premise_readout,
        hypothesis_readout=hypothesis_readout,
    )
    prediction_forward = GroundedForward(
        chart,
        grounding,
        query=GroundingQuery.FULL,
        premise_readout=premise_readout,
        hypothesis_readout=hypothesis_readout,
        phase=Phase.EVALUATE,
    )
    losses = resolve_loss_section(
        cast("LossSection", run_spec.loss),
        Registry("loss"),
        published=forward.publishes,
        unavailable_outputs=chart.unavailable_outputs,
        readouts=readouts,
        spaces=spaces,
        readout_spaces=readout_spaces,
        relations=space_graph.relations,
        span_decoders={},
        proof_reward=ProofRewardContext(
            interpretation.language,
            interpretation.service,
        ),
    )
    modules = {
        role: component
        for role, component in components.items()
        if isinstance(component, nn.Module)
    }
    topology = _ConfiguredTopology(
        modules,
        target_parameter_count=(
            None if model.capacity is None else model.capacity.target
        ),
    )
    groups = topology.resolve(trainer.active_groups)
    optimizer = cast("Optimizer", _construct(run_spec.optimizer, groups.parameters))
    configured_featurizer = cast(
        "ConfiguredSpec[object]",
        run_spec.data.datasets[trainer.dataset].featurizer,
    )
    featurizer_plugin = plugin_for(configured_featurizer)
    featurizer_settings = configured_featurizer.settings()
    featurizer_resources = blueprint.featurizer_resources(
        trainer.dataset, dataset, TorchCodeTensorizer()
    )
    build_featurizer = cast(
        "Builder[object, FeaturizerResources, object]",
        featurizer_plugin.build,
    )
    featurizer = cast(
        "Featurizer[Relational, FeaturizedBatch[Relational]]",
        build_featurizer(featurizer_settings, featurizer_resources),
    )
    epoch_source = pairs.epochs(
        featurizer, _batching(trainer, streams, trainer.dataset)
    )
    first_epoch = epoch_source.at(0)
    accumulation = trainer.accumulation_steps
    steps_per_epoch = len(first_epoch.training) // accumulation
    scheduler = _scheduler(
        run_spec.scheduler,
        ScheduleContext(
            optimizer,
            epochs=trainer.epochs,
            steps_per_epoch=steps_per_epoch,
        ),
    )
    optimization = AutocastPolicy(
        optimizer,
        policy,
        _step_options(
            run_spec,
            accumulation_steps=accumulation,
            max_grad_norm=trainer.max_grad_norm,
        ),
        scheduler=scheduler,
    )
    configured = ConfiguredModel(
        topology,
        codec=codec,
        groups=groups,
        policy=policy,
        checkpoint_labels={
            ModelRole.CONTROLLER: {
                action.name: column for column, action in enumerate(actions.actions)
            }
        },
        pretrained=_pretrained_weights(model.components()),
    )

    prediction = _InferencePoint(
        configured,
        prediction_forward,
        pairs.labeling.labels,
        _response_rows(cast("LossSection", run_spec.loss), pairs.labeling.labels),
        build_featurizer,
        featurizer_settings,
        featurizer_resources,
    )

    return PreparedTraining(
        configured,
        optimization,
        ParserFittingPlan(
            forward,
            GroundedStep(forward, losses, optimization, role=ModelRole.PARSER),
            epoch_source,
            EarlyStopping(
                trainer.early_stopping.metric,
                trainer.early_stopping.mode,
                trainer.early_stopping.patience,
                trainer.early_stopping.minimum_delta,
            ),
            None,
            None,
            None,
            dataset,
            grammar=grammar,
            corpus=None,
            pairs=pairs.corpus,
            interpretation=interpretation,
            epochs=trainer.epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoints=CheckpointPolicy(
                trainer.checkpoints.interval,
                trainer.checkpoints.keep,
                trainer.early_stopping.keep_best,
            ),
            fit_split=_retained_fit_split(run_spec, trainer.dataset),
        ),
        prediction,
    )


def _prepare_parser_training(
    blueprint: _TrainingBlueprint,
    run_spec: RunSpec,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
    *,
    trainer: ParserTrainerSpec,
    pretrained_resources: Mapping[str, PretrainedLoader],
) -> PreparedTraining:
    """Lower and assemble one checked run under seeded runtime policies."""
    if run_spec.model.grounding is not None:
        return _prepare_grounded_training(
            blueprint,
            run_spec,
            streams,
            policy,
            trainer=trainer,
            pretrained_resources=pretrained_resources,
        )

    dataset, sentences = _sentences(blueprint, trainer.dataset)
    grammar = blueprint.learner.grammar
    language = grammar.language
    model = run_spec.model
    codec = NeuralCodecView(language.codec.at(language.category_sort))
    categories = _learner_category_inventory(grammar, codec)
    (
        selection_report,
        constraint_targets,
        constraint_batcher,
        clamp_policy,
        clamp_report,
    ) = _selected_constraints(
        run_spec,
        trainer.dataset,
        sentences,
        grammar,
        categories,
    )
    constraint_readout = _constraint_readout_role(run_spec)
    vocabulary = _token_vocabulary(blueprint, trainer.dataset, dataset)
    actions = blueprint.actions
    grammar_targets = {TargetSource.GRAMMAR_GOAL: grammar.grammar.targets}
    interpretation = blueprint.learner.interpretation
    run = BuildContext(
        component=ComponentId("model"),
        policy=policy,
        language=language,
        codec=codec,
        vocabulary=vocabulary,
        actions=actions,
        rules=grammar.grammar.rules,
        categories=categories,
        producers={},
        targets=grammar_targets,
        loader=_pretrained_loader(run_spec),
        meaning=blueprint.learner.interpretation,
        approximation=_approximation(run_spec),
    )
    base_specifications = model.base_components()
    base_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in base_specifications
    }
    encoder_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in model.encoders
    }
    encoders = build_space_encoders(model, run, encoder_resources)
    base_components = _parser_base_components(
        model,
        run,
        encoders,
        base_resources,
        language.category_sort,
    )
    scorer = cast("ChartScorer", base_components[ModelRole.PARSER])
    decoder_resources = {
        role: loader
        for role, loader in pretrained_resources.items()
        if role in model.decoders
    }
    expected_resources = (
        set(base_resources) | set(encoder_resources) | set(decoder_resources)
    )
    extra_resources = sorted(set(pretrained_resources) - expected_resources)

    if extra_resources:
        msg = f"construction resources name absent component roles {extra_resources}"
        raise ValueError(msg)

    space_graph = assemble_space_graph(
        model,
        run,
        encoders,
        parser=scorer,
        pretrained_resources=decoder_resources,
        erasure=None if interpretation is None else interpretation.erasure,
        role_targets=(
            None
            if constraint_readout is None
            else {constraint_readout: constraint_targets}
        ),
    )
    components = {**base_components, **space_graph.components}
    tokens = cast("TokenEncoder", components[ModelRole.TOKENS_ENCODER])
    pooler = cast("SubwordPooler", components[ModelRole.SUBWORD_POOLER])
    span_decoders = {
        role: cast("SpanTokenDecoder", components[role]) for role in model.span_decoders
    }
    readouts = dict(space_graph.readouts)
    spaces = dict(space_graph.spaces)
    readout_spaces = {
        role: spaces[declaration.space] for role, declaration in model.readouts.items()
    }
    forward = ChartForward[SyntheticTrainingBatch[_LocalConstraint]](
        tokens,
        pooler,
        scorer,
        readouts=tuple(readouts.values()),
        actions=actions,
    )
    losses = resolve_loss_section(
        cast("LossSection", run_spec.loss),
        Registry("loss"),
        published=forward.publishes,
        unavailable_outputs=forward.unavailable_outputs,
        readouts=readouts,
        spaces=spaces,
        readout_spaces=readout_spaces,
        relations=space_graph.relations,
        span_decoders=span_decoders,
        proof_reward=(
            None
            if interpretation is None
            else ProofRewardContext(
                interpretation.language,
                interpretation.service,
            )
        ),
    )
    modules = {
        role: component
        for role, component in components.items()
        if isinstance(component, nn.Module)
    }
    active_roles = trainer.active_groups
    topology = _ConfiguredTopology(
        modules,
        target_parameter_count=(
            None if model.capacity is None else model.capacity.target
        ),
    )
    groups = topology.resolve(active_roles)
    optimizer = cast("Optimizer", _construct(run_spec.optimizer, groups.parameters))
    featurizer = cast(
        "Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]]",
        _construct(
            cast(
                "ComponentSpec",
                run_spec.data.datasets[trainer.dataset].featurizer,
            ),
            blueprint.featurizer_resources(
                trainer.dataset, dataset, TorchCodeTensorizer()
            ),
        ),
    )
    epoch_source = sentences.epochs(
        featurizer,
        constraint_batcher,
        interpretation,
        _batching(trainer, streams, trainer.dataset),
    )
    first_epoch = epoch_source.at(0)
    accumulation = trainer.accumulation_steps
    steps_per_epoch = len(first_epoch.training) // accumulation
    scheduler = _scheduler(
        run_spec.scheduler,
        ScheduleContext(
            optimizer, epochs=trainer.epochs, steps_per_epoch=steps_per_epoch
        ),
    )
    optimization = AutocastPolicy(
        optimizer,
        policy,
        _step_options(
            run_spec,
            accumulation_steps=accumulation,
            max_grad_norm=trainer.max_grad_norm,
        ),
        scheduler=scheduler,
    )
    configured = ConfiguredModel(
        topology,
        codec=codec,
        groups=groups,
        policy=policy,
        checkpoint_labels={
            ModelRole.CONTROLLER: {
                action.name: column for column, action in enumerate(actions.actions)
            }
        },
        pretrained=_pretrained_weights(model.components()),
    )

    return PreparedTraining(
        configured,
        optimization,
        ParserFittingPlan(
            forward,
            ParserStep(forward, losses, optimization, role=ModelRole.PARSER),
            epoch_source,
            EarlyStopping(
                trainer.early_stopping.metric,
                trainer.early_stopping.mode,
                trainer.early_stopping.patience,
                trainer.early_stopping.minimum_delta,
            ),
            selection_report,
            clamp_policy,
            clamp_report,
            dataset,
            grammar=grammar,
            corpus=sentences.corpus,
            pairs=None,
            interpretation=interpretation,
            epochs=trainer.epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoints=CheckpointPolicy(
                trainer.checkpoints.interval,
                trainer.checkpoints.keep,
                trainer.early_stopping.keep_best,
            ),
            fit_split=_retained_fit_split(run_spec, trainer.dataset),
        ),
        None,
    )


def _prepare_distance_training(
    blueprint: _TrainingBlueprint,
    run_spec: RunSpec,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
    trainer: DistanceTrainerSpec,
    pretrained_resources: Mapping[str, PretrainedLoader],
) -> PreparedTraining:
    """Lower and assemble one checked distance-pretraining run."""
    stream_name = training_sampler_stream(trainer.dataset, "distance-batches")
    stream_seed = streams.sampler_seed(stream_name)
    dataset = blueprint.lower_dataset(trainer.dataset)
    grammar = blueprint.learner.grammar
    language = grammar.language
    codec = NeuralCodecView(language.codec.at(trainer.sort))
    category_codec = NeuralCodecView(language.codec.at(language.category_sort))
    vocabulary = _token_vocabulary(blueprint, trainer.dataset, dataset)
    run = BuildContext(
        component=ComponentId("model"),
        policy=policy,
        language=language,
        codec=codec,
        vocabulary=vocabulary,
        actions=blueprint.actions,
        rules=grammar.grammar.rules,
        categories=_learner_category_inventory(grammar, category_codec),
        producers={},
        targets={TargetSource.GRAMMAR_GOAL: grammar.grammar.targets},
        loader=_pretrained_loader(run_spec),
        approximation=_approximation(run_spec),
    )
    declaration = run_spec.model.encoders.get(trainer.encoder)

    if declaration is None:
        msg = (
            f"distance trainer names encoder {trainer.encoder!r}; declared encoders: "
            f"{sorted(run_spec.model.encoders)}"
        )
        raise ValueError(msg)

    unexpected = sorted(set(pretrained_resources) - {trainer.encoder})

    if unexpected:
        msg = f"distance training received resources for unused roles {unexpected}"
        raise ValueError(msg)

    context = run.for_component(ComponentId(trainer.encoder), {})

    if trainer.encoder in pretrained_resources:
        context = context.with_loader(pretrained_resources[trainer.encoder])

    encoder = build_component(cast("ConfiguredSpec[object]", declaration), context)

    if not isinstance(encoder, TermEncoder):
        msg = (
            f"distance encoder {trainer.encoder!r} built "
            f"{type(encoder).__name__}, not a term encoder"
        )
        raise TypeError(msg)

    components = {trainer.encoder: encoder}
    topology = _ConfiguredTopology(
        {
            role: component
            for role, component in components.items()
            if isinstance(component, nn.Module)
        },
        target_parameter_count=(
            None if run_spec.model.capacity is None else run_spec.model.capacity.target
        ),
    )
    groups = topology.resolve(None)
    optimizer = cast("Optimizer", _construct(run_spec.optimizer, groups.parameters))
    scheduler = _scheduler(
        run_spec.scheduler,
        ScheduleContext(optimizer, epochs=1, steps_per_epoch=trainer.steps),
    )
    optimization = AutocastPolicy(
        optimizer,
        policy,
        _step_options(run_spec, max_grad_norm=trainer.max_grad_norm),
        scheduler=scheduler,
    )
    sampler = cast(
        "InventoriedTypeSampler",
        _construct(
            cast("ComponentSpec", run_spec.type_sampler),
            TypeSamplerContext(grammar.store, trainer.sort),
        ),
    )
    objective = TypeDistanceObjective(
        encoder,
        inverse_distance_power=trainer.inverse_distance_power,
        embedding_penalty=trainer.embedding_penalty,
    )
    losses = cast(
        "LossAssembly[TypeBatch]",
        LossAssembly(
            (WeightedObjective(objective.name, 1.0, objective),),
            published=frozenset({"batch"}),
        ),
    )
    configured = ConfiguredModel(
        topology,
        codec=codec,
        groups=groups,
        policy=policy,
        checkpoint_labels={},
        pretrained=_pretrained_weights({trainer.encoder: declaration}),
    )
    step = DistanceEncoderStep(
        DistanceBatcher(
            sampler,
            radius=trainer.ball_radius,
            curriculum=(
                None
                if trainer.depth_curriculum is None
                else LinearDepthCurriculum(
                    trainer.depth_curriculum.start_depth,
                    trainer.depth_curriculum.steps,
                )
            ),
            branching=build_arity_branching(trainer.branching.settings()),
            child_depth=build_child_depth(trainer.child_depth.settings()),
            constructor_weights=trainer.constructor_weights,
        ),
        language.codec.at(trainer.sort),
        losses,
        optimization,
        role=trainer.encoder,
    )

    return PreparedTraining(
        configured,
        optimization,
        StepFittingPlan(
            (step,),
            distance_batch_seed,
            stream_seed,
            validation_seed=None,
            stop=None,
            steps=trainer.steps,
            batch_size=trainer.batch_size,
            checkpoints=CheckpointPolicy(
                trainer.checkpoints.interval, trainer.checkpoints.keep, 0
            ),
            dataset=dataset,
        ),
        None,
    )


def _pair_sampler_for_sort(
    sort: str,
    sampler: InventoriedTypeSampler,
    grammar: GrammarAssembly,
    interpretation: InterpretationFunction | None,
    actions: ActionSet,
    type_sampler: ComponentSpec,
    pair_sampler: ComponentSpec,
    /,
    *,
    composes: bool,
) -> PairSampler[Term, Action] | None:
    """Return generated combinator instances for one represented sort."""
    if not composes:
        return None

    language = grammar.language

    if sort == language.category_sort:
        return cast(
            "PairSampler[Term, Action]",
            _construct(
                pair_sampler,
                RulePairSamplerContext(sampler, grammar.grammar.rules, actions),
            ),
        )

    if interpretation is None:
        msg = (
            f"represented sort {sort!r} composes meanings, but the "
            "learner declares no interpretation"
        )
        raise ValueError(msg)

    if sort == language.type_sort:
        categories = cast(
            "InventoriedTypeSampler",
            _construct(
                type_sampler,
                TypeSamplerContext(grammar.store, language.category_sort),
            ),
        )
        source = cast(
            "PairSampler[Term, Action]",
            _construct(
                pair_sampler,
                RulePairSamplerContext(
                    categories,
                    grammar.grammar.rules,
                    actions,
                ),
            ),
        )
        return cast(
            "PairSampler[Term, Action]",
            MorphismPairSampler(source, interpretation.erasure),
        )

    if sort == language.term_sort:
        return cast(
            "PairSampler[Term, Action]",
            MeaningPairSampler(
                sampler,
                actions,
                interpretation.actions,
                tuple(grammar.grammar.rules),
                normalization=interpretation.normalization,
            ),
        )

    msg = f"represented sort {sort!r} has no declared combinator-pair semantics"
    raise ValueError(msg)


def _terms_pair_sampler(
    space: Space,
    sampler: InventoriedTypeSampler,
    grammar: GrammarAssembly,
    interpretation: InterpretationFunction | None,
    actions: ActionSet,
    type_sampler: ComponentSpec,
    pair_sampler: ComponentSpec,
    /,
) -> PairSampler[Term, Action] | None:
    """Return generated combinator instances for one represented space."""
    return _pair_sampler_for_sort(
        space.sort,
        sampler,
        grammar,
        interpretation,
        actions,
        type_sampler,
        pair_sampler,
        composes=space.combinators is not None,
    )


def _terms_step(
    space: str,
    *,
    roles: Sequence[str],
    encoder: TermEncoder,
    identity: object,
    choices: ChoiceIndex,
    scorers: Mapping[Action, CombinatorScorer] | None,
    sampler: InventoriedTypeSampler,
    pairs: PairSampler[Term, Action] | None,
    codec: TermCodec,
    generation: TermGenerationSpec,
    optimization: AutocastPolicy,
    name: str,
) -> TermsStep:
    """Build one represented space's generated-term step.

    Parameters
    ----------
    space : str
        The represented space's role, which names its objectives.
    roles : Sequence[str]
        The component roles the step trains.
    encoder : TermEncoder
        The space's term encoder.
    identity : object
        The space's identity decoder, which must provide teacher-forced rows.
    choices : ChoiceIndex
        The learner language's choice index.
    scorers : Mapping[Action, CombinatorScorer] | None
        The combinator scorer of each composing action, or None to train no
        combinators.
    sampler : InventoriedTypeSampler
        What draws the space's terms.
    pairs : PairSampler[Term, Action] | None
        What draws rule instances, held exactly when ``scorers`` is.
    codec : TermCodec
        The codec of the space's sort.
    generation : TermGenerationSpec
        The declared term and rule instance distribution.
    optimization : AutocastPolicy
        What runs the backward pass and the optimizer.
    name : str
        The step's name.

    Raises
    ------
    TypeError
        If the identity decoder provides no teacher-forced rows.
    ValueError
        If combinators are trained without a pair sampler, or the reverse.
    """
    if not isinstance(identity, TeacherForcedDecoder):
        msg = f"space {space!r} identity does not provide teacher-forced rows"
        raise TypeError(msg)

    if (scorers is None) != (pairs is None):
        msg = f"space {space!r} trains combinators exactly when it draws rule instances"
        raise ValueError(msg)

    autoencoder = TypeAutoencoderObjective[TermsBatch](
        encoder, identity, choices, name=autoencoding_objective(space)
    )
    objectives: list[WeightedObjective[TermsBatch]] = [
        WeightedObjective(autoencoder.name, 1.0, autoencoder)
    ]

    if scorers is not None:
        combinators = CombinatorSupervisionObjective[TermsBatch](
            scorers,
            schedule=TeacherForcingSchedule(generation.teacher_forcing_saturation),
            name=combinator_objective(space),
        )
        objectives.append(WeightedObjective(combinators.name, 1.0, combinators))

    distribution = TypeDistribution(
        generation.max_depth,
        generation.constructor_weights,
        branching=build_arity_branching(generation.branching.settings()),
        child_depth=build_child_depth(generation.child_depth.settings()),
        max_depth=generation.max_depth,
    )

    return TermsStep(
        sampler,
        distribution,
        pairs,
        None if pairs is None else PairDistribution(max_depth=generation.max_depth),
        codec,
        LossAssembly(tuple(objectives), published=frozenset({"batch"})),
        optimization,
        roles=tuple(dict.fromkeys(roles)),
        name=name,
    )


def _prepare_terms_training(
    blueprint: _TrainingBlueprint,
    run_spec: RunSpec,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
    trainer: TermsTrainerSpec,
    pretrained_resources: Mapping[str, PretrainedLoader],
) -> PreparedTraining:
    """Lower and assemble generated-term and combinator-pair pretraining."""
    horizon = trainer.step_horizon()
    stream_seed = streams.sampler_seed(
        training_sampler_stream(trainer.dataset, "terms-batches")
    )
    validation_seed = (
        None
        if trainer.stop is None
        else streams.sampler_seed(
            training_sampler_stream(trainer.dataset, "terms-held-out-batches")
        )
    )
    dataset = blueprint.lower_dataset(trainer.dataset)
    grammar = blueprint.learner.grammar
    language = grammar.language
    first_space = run_spec.model.spaces[trainer.spaces[0]]
    codec = NeuralCodecView(language.codec.at(first_space.sort))
    category_codec = NeuralCodecView(language.codec.at(language.category_sort))
    vocabulary = _token_vocabulary(blueprint, trainer.dataset, dataset)
    run = BuildContext(
        component=ComponentId("model"),
        policy=policy,
        language=language,
        codec=codec,
        vocabulary=vocabulary,
        actions=blueprint.actions,
        rules=grammar.grammar.rules,
        categories=_learner_category_inventory(grammar, category_codec),
        producers={},
        targets={TargetSource.GRAMMAR_GOAL: grammar.grammar.targets},
        loader=_pretrained_loader(run_spec),
        meaning=blueprint.learner.interpretation,
        approximation=_approximation(run_spec),
    )
    interpretation = blueprint.learner.interpretation
    graph = wire_spaces(
        run_spec.model,
        run,
        parser=None,
        pretrained_resources=pretrained_resources,
        erasure=None if interpretation is None else interpretation.erasure,
    )
    topology = _ConfiguredTopology(
        {
            role: component
            for role, component in graph.components.items()
            if isinstance(component, nn.Module)
        },
        target_parameter_count=(
            None if run_spec.model.capacity is None else run_spec.model.capacity.target
        ),
    )
    active_roles: list[str] = []

    for space_role in trainer.spaces:
        declaration = run_spec.model.spaces[space_role]
        active_roles.extend((declaration.encoder, declaration.identity))

        if declaration.combinators is not None:
            active_roles.extend(declaration.combinators.decoders.values())

    active = tuple(dict.fromkeys(active_roles))
    groups = topology.resolve(active)
    optimizer = cast("Optimizer", _construct(run_spec.optimizer, groups.parameters))
    scheduler = _scheduler(
        run_spec.scheduler,
        ScheduleContext(optimizer, epochs=1, steps_per_epoch=horizon),
    )
    optimization = AutocastPolicy(
        optimizer,
        policy,
        _step_options(run_spec, max_grad_norm=trainer.max_grad_norm),
        scheduler=scheduler,
    )
    type_sampler = cast("ComponentSpec", run_spec.type_sampler)
    pair_sampler = cast("ComponentSpec", run_spec.pair_sampler)
    steps: list[TermsStep] = []

    for space_role in trainer.spaces:
        declaration = run_spec.model.spaces[space_role]
        space = graph.spaces[space_role]
        sampler = cast(
            "InventoriedTypeSampler",
            _construct(type_sampler, TypeSamplerContext(language.store, space.sort)),
        )
        family = space.combinators
        steps.append(
            _terms_step(
                space_role,
                roles=(
                    declaration.encoder,
                    declaration.identity,
                    *(
                        ()
                        if declaration.combinators is None
                        else declaration.combinators.decoders.values()
                    ),
                ),
                encoder=space.encoder,
                identity=space.identity,
                choices=space.choices,
                scorers=None if family is None else family.scorers,
                sampler=sampler,
                pairs=_terms_pair_sampler(
                    space,
                    sampler,
                    grammar,
                    interpretation,
                    blueprint.actions,
                    type_sampler,
                    pair_sampler,
                ),
                codec=language.codec.at(space.sort),
                generation=trainer.generation,
                optimization=optimization,
                name=space_role,
            )
        )

    configured = ConfiguredModel(
        topology,
        codec=codec,
        groups=groups,
        policy=policy,
        checkpoint_labels={},
        pretrained=_pretrained_weights(run_spec.model.components()),
    )

    return PreparedTraining(
        configured,
        optimization,
        StepFittingPlan(
            tuple(steps),
            terms_batch_seed,
            stream_seed,
            validation_seed=validation_seed,
            stop=(
                None
                if trainer.stop is None
                else AccuracyStopping(
                    trainer.stop.metric,
                    trainer.stop.target,
                    trainer.stop.max_steps,
                )
            ),
            steps=horizon,
            batch_size=trainer.batch_size,
            checkpoints=CheckpointPolicy(
                trainer.checkpoints.interval, trainer.checkpoints.keep, 0
            ),
            dataset=dataset,
        ),
        None,
    )


def _build_parser_training(
    declaration: ParserTrainerSpec, context: TrainerBuildContext, /
) -> PreparedTraining:
    """Assemble parser training under one trainer build context."""
    blueprint = context.blueprint

    return _prepare_parser_training(
        blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
        context.policy,
        trainer=declaration,
        pretrained_resources=context.pretrained_resources,
    )


def _task_loss(
    declaration: StepSpec,
    source: LossAssembly[_AlternatingBatch],
    /,
) -> LossAssembly[_AlternatingBatch]:
    """Return one task's declared objective view of the run's loss."""
    selected = declaration.objectives

    if not selected:
        return source

    by_name = {entry.name: entry for entry in source}

    return LossAssembly(tuple(by_name[name] for name in selected))


def _position_draw(draw: Draw, step: int, /) -> Draw:
    """Return one generated-batch request at its global optimizer position."""
    return Draw(draw.seed, draw.count, step=step, horizon=draw.horizon)


def _position_pair_draw(inputs: PairDraws, step: int, /) -> PairDraws:
    """Return one generated-rule request at its global optimizer position."""
    draw = inputs.draw
    return PairDraws(
        inputs.pair_distribution,
        Draw(draw.seed, draw.count, step=step, horizon=draw.horizon),
    )


def _position_utterances(inputs: Utterances, step: int, /) -> Utterances:
    """Return one lexical warmup request at its global optimizer position."""
    return Utterances(inputs.items, inputs.candidates, step)


@final
class _AlternatingStepContext(StepBuildContext):
    """Build installed step kinds over one shared configured parser graph."""

    __slots__ = (
        "_base",
        "_blueprint",
        "_declaration",
        "_optimization",
        "_prepared",
        "_run_spec",
        "_streams",
    )

    def __init__(
        self,
        declaration: AlternatingTrainerSpec | StagedTrainerSpec,
        prepared: PreparedTraining,
        blueprint: _TrainingBlueprint,
        run_spec: RunSpec,
        streams: RestorableRandomStreams,
        /,
    ) -> None:
        self._declaration = declaration
        self._prepared = prepared
        self._base = cast("ParserFittingPlan", prepared.plan)
        self._blueprint = blueprint
        self._run_spec = run_spec
        self._streams = streams
        self._optimization = prepared.optimization

    @property
    def _trainer(self) -> ParserTrainerSpec:
        """Return the common parser-training horizon and batching declaration."""
        return self._declaration.base

    def _loss(self, declaration: StepSpec) -> LossAssembly[_AlternatingBatch]:
        """Return one task's objective view of the configured run loss."""
        return _task_loss(
            declaration,
            cast("LossAssembly[_AlternatingBatch]", self._base.step.assembly),
        )

    def _sentence_batches(
        self, declaration: StepSpec, /
    ) -> tuple[_AlternatingBatch, ...]:
        """Build one deterministic sentence epoch for a declared task."""
        role = declaration.dataset
        dataset, sentences = _sentences(self._blueprint, role)
        data = self._run_spec.data.datasets[role]
        featurizer = cast(
            "Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]]",
            _construct(
                cast("ComponentSpec", data.featurizer),
                self._blueprint.featurizer_resources(
                    role, dataset, TorchCodeTensorizer()
                ),
            ),
        )
        grammar = self._blueprint.learner.grammar
        language = grammar.language
        categories = _learner_category_inventory(
            grammar,
            NeuralCodecView(language.codec.at(language.category_sort)),
        )
        _, _, batcher, _, _ = _selected_constraints(
            self._run_spec,
            role,
            sentences,
            grammar,
            categories,
        )
        source = sentences.epochs(
            featurizer,
            batcher,
            self._blueprint.learner.interpretation,
            _batching(self._trainer, self._streams, role),
        )
        return source.at(0).training

    def _inference_batches(
        self, declaration: StepSpec, /
    ) -> tuple[_AlternatingBatch, ...]:
        """Build one deterministic labeled-inference epoch for a declared task."""
        role = declaration.dataset
        dataset, pairs = _pairs(self._blueprint, role)
        data = self._run_spec.data.datasets[role]
        featurizer = cast(
            "Featurizer[Relational, FeaturizedBatch[Relational]]",
            _construct(
                cast("ComponentSpec", data.featurizer),
                self._blueprint.featurizer_resources(
                    role, dataset, TorchCodeTensorizer()
                ),
            ),
        )
        source = pairs.epochs(featurizer, _batching(self._trainer, self._streams, role))
        return source.at(0).training

    def _parser_task(self, declaration: StepSpec, /) -> FittingTask:
        """Build a sentence task against the parser's ungrounded head."""
        forward = self._base.forward

        if isinstance(forward, GroundedForward):
            forward = forward.inner

        return DatasetTask(
            self._sentence_batches(declaration),
            cast("Forward[_AlternatingBatch]", forward),
            self._loss(declaration),
            self._optimization,
            name=declaration.name,
        )

    def _grounded_task(self, declaration: StepSpec, /) -> FittingTask:
        """Build an inference task against the configured grounded head."""
        forward = self._base.forward

        if not isinstance(forward, GroundedForward):
            msg = (
                f"training step {declaration.name!r} requires a declared model "
                "grounding"
            )
            raise TypeError(msg)

        return DatasetTask(
            self._inference_batches(declaration),
            cast("Forward[_AlternatingBatch]", forward),
            self._loss(declaration),
            self._optimization,
            name=declaration.name,
        )

    @override
    def parser(self, declaration: ParserStepSpec, /) -> FittingTask:
        """Build parser loss over sentence batches."""
        return self._parser_task(declaration)

    @override
    def supervised_warm_start(
        self, declaration: SupervisedWarmStartStepSpec, /
    ) -> FittingTask:
        """Build gold-form supervision through the parser head."""
        return self._parser_task(declaration)

    @override
    def judgment(self, declaration: JudgmentStepSpec, /) -> FittingTask:
        """Build declared judgment objectives over inference pairs."""
        return self._grounded_task(declaration)

    @override
    def grounded(self, declaration: GroundedStepSpec, /) -> FittingTask:
        """Build declared grounded objectives over inference pairs."""
        return self._grounded_task(declaration)

    def _space(
        self, role: str, dataset_role: str, /
    ) -> tuple[SpaceSpec, LoweredDataset]:
        """Return one represented space and the dataset named by its task."""
        declaration = self._run_spec.model.spaces.get(role)

        if declaration is None:
            msg = (
                f"training step names represented space {role!r}; declared spaces "
                f"are {sorted(self._run_spec.model.spaces)}"
            )
            raise ValueError(msg)

        return declaration, self._blueprint.lower_dataset(dataset_role)

    def _module(self, role: str, /) -> nn.Module:
        """Return one exact configured component module by declared role."""
        module = self._prepared.model.modules.get(role)

        if module is None:
            msg = f"training step names absent configured component role {role!r}"
            raise ValueError(msg)

        return module

    def _encoder(self, role: str, /) -> TermEncoder:
        """Return one configured term encoder after capability validation."""
        module = self._module(role)

        if not isinstance(module, TermEncoder):
            msg = f"component role {role!r} is not a term encoder"
            raise TypeError(msg)

        return module

    def _teacher_decoder(self, role: str, /) -> TeacherForcedDecoder:
        """Return one configured decoder providing teacher-forced rows."""
        module = self._module(role)

        if not isinstance(module, TeacherForcedDecoder):
            msg = f"component role {role!r} does not provide teacher-forced rows"
            raise TypeError(msg)

        return module

    def _type_sampler(self, sort: str, /) -> InventoriedTypeSampler:
        """Build the declared type sampler for one represented sort."""
        declaration = self._run_spec.type_sampler

        if declaration is None:
            msg = "generated-type training requires a declared type_sampler"
            raise ValueError(msg)

        return cast(
            "InventoriedTypeSampler",
            _construct(
                cast("ComponentSpec", declaration),
                TypeSamplerContext(self._blueprint.learner.grammar.store, sort),
            ),
        )

    def _draws(self, role: str, purpose: str, /) -> tuple[Draw, ...]:
        """Return one deterministic generated request per sentence minibatch."""
        _, sentences = _sentences(self._blueprint, role)
        batches = ceil(len(sentences.corpus.observed()) / self._trainer.batch_size)
        horizon = max(1, batches * self._trainer.epochs)
        stream = self._streams.sampler_seed(training_sampler_stream(role, purpose))
        return tuple(
            Draw(
                terms_batch_seed(stream, position),
                self._trainer.batch_size,
                step=0,
                horizon=horizon,
            )
            for position in range(batches)
        )

    @override
    def token_warmup(self, declaration: TokenWarmupStepSpec, /) -> FittingTask:
        """Build lexical token warmup over selected occurrence constraints."""
        space, _ = self._space(declaration.space, declaration.dataset)
        _, sentences = _sentences(self._blueprint, declaration.dataset)
        encoder = self._encoder(space.encoder)
        decoder = self._teacher_decoder(space.identity)
        tokens = self._module(ModelRole.TOKENS_ENCODER)
        pooler = self._module(ModelRole.SUBWORD_POOLER)

        if not isinstance(tokens, TokenEncoder):
            msg = "component role 'token_encoder' is not a token encoder"
            raise TypeError(msg)

        if not isinstance(pooler, SubwordPooler):
            msg = "component role 'subword_pooler' is not a subword pooler"
            raise TypeError(msg)

        objective = TokenWarmupObjective[WarmupBatch](
            tokens,
            pooler,
            encoder,
            decoder,
            self._blueprint.learner.grammar.language.codec.choices,
        )
        assembly = LossAssembly[WarmupBatch](
            (WeightedObjective(objective.name, 1.0, objective),),
            published=frozenset({"batch"}),
        )
        step = TokenWarmupStep(
            self._blueprint.learner.grammar.language.codec.at(space.sort),
            assembly,
            self._optimization,
            role=ModelRole.TOKENS_ENCODER,
            name=declaration.name,
        )
        selection = select_constraints(self._run_spec, declaration.dataset, sentences)

        if selection is None:
            msg = (
                f"token warm-up step {declaration.name!r} requires a declared "
                "constraint source"
            )
            raise ValueError(msg)

        grammar = self._blueprint.learner.grammar
        items: list[tuple[tuple[str, ...], tuple[tuple[Term, ...], ...]]] = []

        for example in selection.examples:
            words = tuple(token.surface for token in example.item.tokens)
            admitted: list[set[Term]] = [set() for _ in words]

            for constraint in example.supervision.constraints:
                span = constraint.span

                if span.end - span.start != 1:
                    msg = (
                        f"token warm-up step {declaration.name!r} received a "
                        f"nonlexical constraint over [{span.start}, {span.end})"
                    )
                    raise ValueError(msg)

                admitted[span.start].update(
                    grammar.syntax.parse(grammar.store, label)
                    for label in constraint.permitted
                )

            items.append(
                (
                    words,
                    tuple(
                        tuple(sorted(candidates, key=lambda term: term.key()))
                        for candidates in admitted
                    ),
                )
            )

        size = self._trainer.batch_size
        requests = tuple(
            Utterances(
                tuple(words for words, _ in items[start : start + size]),
                tuple(candidates for _, candidates in items[start : start + size]),
                0,
            )
            for start in range(0, len(items), size)
        )
        return StepTask(
            requests,
            step,
            _position_utterances,
            name=declaration.name,
            optimizer_steps=sum(
                any(candidates for item in request.candidates for candidates in item)
                for request in requests
            ),
        )

    def _scorers(
        self, family: CombinatorFamilySpec, /
    ) -> Mapping[Action, CombinatorScorer]:
        """Return the combinator scorer of each composing action of one family."""
        by_name = {action.name: action for action in self._blueprint.actions.actions}
        scorers: dict[Action, CombinatorScorer] = {}

        for action_name, role in family.decoders.items():
            action = by_name.get(action_name)

            if action is None:
                msg = f"combinator decoder names absent action {action_name!r}"
                raise ValueError(msg)

            module = self._module(role)

            if not isinstance(module, CombinatorScorer):
                msg = f"component role {role!r} is not a combinator scorer"
                raise TypeError(msg)

            scorers[action] = module

        return scorers

    @override
    def terms(self, declaration: TermsStepSpec, /) -> FittingTask:
        """Build generated-term autoencoding and any combinator decoding."""
        if declaration.combinators:
            space, _, sampler, pairs = self._pair_training(declaration)
        else:
            space, _ = self._space(declaration.space, declaration.dataset)
            sampler = self._type_sampler(space.sort)
            pairs = None

        family = space.combinators if declaration.combinators else None
        step = _terms_step(
            declaration.space,
            roles=(
                space.encoder,
                space.identity,
                *(() if family is None else family.decoders.values()),
            ),
            encoder=self._encoder(space.encoder),
            identity=self._module(space.identity),
            choices=self._blueprint.learner.grammar.language.codec.choices,
            scorers=None if family is None else self._scorers(family),
            sampler=sampler,
            pairs=pairs,
            codec=self._blueprint.learner.grammar.language.codec.at(space.sort),
            generation=declaration.generation,
            optimization=self._optimization,
            name=declaration.name,
        )
        requests = self._draws(declaration.dataset, "terms-batches")
        target = declaration.stop
        stop = (
            None
            if target is None
            else AccuracyStopping(target.metric, target.target, target.max_steps)
        )
        validation = (
            ()
            if stop is None
            else (
                Draw(
                    self._streams.sampler_seed(
                        training_sampler_stream(
                            declaration.dataset, "terms-held-out-batches"
                        )
                    ),
                    self._trainer.batch_size,
                    step=0,
                    horizon=stop.max_steps,
                ),
            )
        )
        return StepTask(
            requests,
            step,
            _position_draw,
            name=declaration.name,
            optimizer_steps=len(requests),
            validation=validation,
            evaluate=None if stop is None else step.evaluate,
            stop=stop,
        )

    def _pair_training(
        self, declaration: TermsStepSpec | ControllerStepSpec, /
    ) -> tuple[
        SpaceSpec,
        LoweredDataset,
        InventoriedTypeSampler,
        PairSampler[Term, Action],
    ]:
        """Return one represented space and its declared rule instance sampler."""
        space, dataset = self._space(declaration.space, declaration.dataset)
        sampler = self._type_sampler(space.sort)
        pair_declaration = self._run_spec.pair_sampler
        type_declaration = self._run_spec.type_sampler

        if pair_declaration is None or type_declaration is None:
            msg = "rule-instance training requires pair_sampler and type_sampler"
            raise ValueError(msg)

        pairs = _pair_sampler_for_sort(
            space.sort,
            sampler,
            self._blueprint.learner.grammar,
            self._blueprint.learner.interpretation,
            self._blueprint.actions,
            cast("ComponentSpec", type_declaration),
            cast("ComponentSpec", pair_declaration),
            composes=space.combinators is not None,
        )

        if pairs is None:
            msg = f"represented space {declaration.space!r} declares no combinators"
            raise ValueError(msg)

        return space, dataset, sampler, pairs

    @override
    def controller(self, declaration: ControllerStepSpec, /) -> FittingTask:
        """Build labeled parser-controller supervision."""
        space, _, _, pairs = self._pair_training(declaration)
        module = self._module(ModelRole.CONTROLLER)

        if not isinstance(module, Controller):
            msg = "component role 'controller' is not a controller"
            raise TypeError(msg)

        objective = ControllerSupervisionObjective[RuleBatch](
            self._encoder(space.encoder),
            self._blueprint.learner.grammar.language.codec.at(space.sort),
            module,
            name=declaration.name,
        )
        assembly = LossAssembly[RuleBatch](
            (WeightedObjective(objective.name, 1.0, objective),),
            published=frozenset({"batch"}),
        )
        step = ControllerStep(
            pairs,
            self._blueprint.actions,
            assembly,
            self._optimization,
            role=ModelRole.CONTROLLER,
            name=declaration.name,
        )
        distribution = PairDistribution(max_depth=declaration.max_depth)
        requests = tuple(
            PairDraws(distribution, draw)
            for draw in self._draws(declaration.dataset, "controller-batches")
        )
        stop = (
            None
            if declaration.stop is None
            else AccuracyStopping(
                declaration.stop.metric,
                declaration.stop.target,
                declaration.stop.max_steps,
            )
        )
        validation = (
            ()
            if stop is None
            else (
                PairDraws(
                    distribution,
                    Draw(
                        self._streams.sampler_seed(
                            training_sampler_stream(
                                declaration.dataset, "controller-held-out-batches"
                            )
                        ),
                        self._trainer.batch_size,
                        step=0,
                        horizon=stop.max_steps,
                    ),
                ),
            )
        )
        return StepTask(
            requests,
            step,
            _position_pair_draw,
            name=declaration.name,
            optimizer_steps=len(requests),
            validation=validation,
            evaluate=None if stop is None else step.evaluate,
            stop=stop,
        )

    @override
    def alternating(self, declaration: AlternatingStepSpec, /) -> FittingTask:
        """Build one nested alternating task from installed child plugins."""
        tasks = tuple(
            step_plugin(task).prepare(task, self) for task in declaration.tasks
        )
        nested = AlternatingStep(
            tasks,
            mixing(declaration.mixing),
            role=ModelRole.PARSER,
            name=declaration.name,
        )
        return AlternatingTask(nested, name=declaration.name)


def _build_alternating_training(
    declaration: AlternatingTrainerSpec,
    context: TrainerBuildContext,
    /,
) -> PreparedTraining:
    """Assemble named parser-objective tasks under one mixing declaration."""
    prepared = _prepare_parser_training(
        context.blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
        context.policy,
        trainer=declaration.base,
        pretrained_resources=context.pretrained_resources,
    )
    base = cast("ParserFittingPlan", prepared.plan)
    task_context = _AlternatingStepContext(
        declaration,
        prepared,
        context.blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
    )
    tasks = tuple(
        step_plugin(task).prepare(task, task_context) for task in declaration.tasks
    )
    step = AlternatingStep(
        tasks,
        mixing(declaration.mixing),
        role=ModelRole.PARSER,
    )

    return PreparedTraining(
        prepared.model,
        prepared.optimization,
        base.with_alternation(step),
        prepared.prediction,
    )


def _build_staged_training(
    declaration: StagedTrainerSpec,
    context: TrainerBuildContext,
    /,
) -> PreparedTraining:
    """Assemble one parser graph under declared active-group stages."""
    prepared = _prepare_parser_training(
        context.blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
        context.policy,
        trainer=declaration.base,
        pretrained_resources=context.pretrained_resources,
    )
    plan = cast("ParserFittingPlan", prepared.plan)
    task_context = _AlternatingStepContext(
        declaration,
        prepared,
        context.blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
    )
    stages = tuple(
        Stage(stage.name, ActiveGroups(stage.active_groups), stage.epochs)
        for stage in declaration.stages
    )
    tasks = tuple(
        step_plugin(stage.step).prepare(stage.step, task_context)
        for stage in declaration.stages
    )

    for stage, task in zip(declaration.stages, tasks, strict=True):
        stop = task.stop if isinstance(task, StepTask) else None

        if stop is not None and stop.max_steps > stage.epochs * task.optimizer_steps:
            msg = (
                f"stage {stage.name!r} provides "
                f"{stage.epochs * task.optimizer_steps} "
                f"optimizer steps, fewer than stop.max_steps={stop.max_steps}"
            )
            raise ValueError(msg)

    return PreparedTraining(
        prepared.model,
        prepared.optimization,
        plan.with_stages(stages, tasks, declaration.validation_stage),
        prepared.prediction,
    )


def _build_distance_training(
    declaration: DistanceTrainerSpec, context: TrainerBuildContext, /
) -> PreparedTraining:
    """Assemble distance pretraining under one trainer build context."""
    blueprint = context.blueprint

    return _prepare_distance_training(
        blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
        context.policy,
        trainer=declaration,
        pretrained_resources=context.pretrained_resources,
    )


def _build_terms_training(
    declaration: TermsTrainerSpec, context: TrainerBuildContext, /
) -> PreparedTraining:
    """Assemble generated-term pretraining under one trainer build context."""
    blueprint = context.blueprint

    return _prepare_terms_training(
        blueprint,
        context.run_spec,
        cast("RestorableRandomStreams", context.streams),
        context.policy,
        trainer=declaration,
        pretrained_resources=context.pretrained_resources,
    )


def prepare_training(
    blueprint: _TrainingBlueprint,
    streams: RestorableRandomStreams,
    policy: DevicePolicy,
    /,
    *,
    pretrained_resources: Mapping[str, PretrainedLoader] | None = None,
) -> PreparedTraining:
    """Lower and assemble one checked run under seeded runtime policies."""
    run_spec = blueprint.run_spec
    trainer = run_spec.trainer
    resources = {} if pretrained_resources is None else pretrained_resources
    context = TrainerBuildContext(
        blueprint,
        run_spec,
        streams,
        policy,
        pretrained_resources=resources,
    )

    return cast("PreparedTraining", trainer_plugin(trainer).prepare(trainer, context))
