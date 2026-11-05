"""Training-layer protocols."""

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from lofi.nn._inference_ports import InferencePairs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Generator

    from lofi._types import ItemId, JsonValue, ReadoutId
    from lofi.lf import ArgumentRenaming
    from lofi.nn._values import Encoded
    from lofi.nn.loss._proof import ShapedOutcome
    from lofi.nn.loss._values import ObjectiveOutputs
    from lofi.terms import Term
    from lofi.train._boundary import CheckpointManifest
    from lofi.train._checkpoint import Checkpoint, CheckpointId
    from lofi.train._grouping import GroupResolution
    from lofi.train._groups import ParameterGroups
    from lofi.train._optimization import OptimizationResult
    from lofi.train._steps import EvalResult, StepOutcome, TrainResult

__all__ = [
    "CheckpointStore",
    "EvalStep",
    "FormReward",
    "Forward",
    "GroundedPairs",
    "MetricLog",
    "OptimizationPolicy",
    "ParameterDeclaration",
    "PositionedBatch",
    "RestorableRandomStreams",
    "SidedPairs",
    "StageFitting",
    "StepDeclaration",
    "TrainStep",
    "ValueCodec",
]


@runtime_checkable
class RestorableRandomStreams(Protocol):
    """Named random streams whose advancing state can be checkpointed."""

    @property
    def seed_policy(self) -> Mapping[str, JsonValue]:
        """Return canonical immutable policy metadata."""
        ...

    def sampler_seed(self, name: str, /) -> int:
        """Return the seed assigned to one declared sampler.

        Parameters
        ----------
        name : str
            The sampler stream name.
        """
        ...

    def loader_generator(self, name: str, /) -> Generator:
        """Return the retained generator assigned to one declared loader.

        Parameters
        ----------
        name : str
            The loader stream name.
        """
        ...

    def worker_initializer(self, name: str, /) -> Callable[[int], None]:
        """Return the initializer assigned to one declared worker family.

        Parameters
        ----------
        name : str
            The worker-family name.
        """
        ...

    def snapshot(self) -> Mapping[str, str]:
        """Return every advancing stream in its safe textual wire form."""
        ...

    def validate(self, streams: Mapping[str, str], /) -> None:
        """Refuse a snapshot that cannot be restored without changing state.

        Parameters
        ----------
        streams : Mapping[str, str]
            Encoded state keyed by stream name.
        """
        ...

    def prepare_restore(self, streams: Mapping[str, str], /) -> Callable[[], None]:
        """Validate state and return its non-validating restore operation.

        Parameters
        ----------
        streams : Mapping[str, str]
            Encoded state keyed by stream name.
        """
        ...

    def restore(self, streams: Mapping[str, str], /) -> None:
        """Restore a complete previously validated snapshot.

        Parameters
        ----------
        streams : Mapping[str, str]
            Encoded state keyed by stream name.
        """
        ...

    def reseed(self, attempt: int, /) -> None:
        """Restart every stream at a deterministic retry seed."""
        ...


class PositionedBatch(Protocol):
    """An immutable minibatch that can be placed at an optimizer step."""

    @property
    def item_ids(self) -> tuple[ItemId, ...]:
        """Return item identities in within-batch order."""
        ...

    def at_step(self, step: int, /) -> Self:
        """Return this batch at the requested optimizer position."""
        ...


class MetricLog(Protocol):
    """Receives the scalar metrics fitting reports as it goes."""

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        """Log named scalar values at one global optimizer step.

        Parameters
        ----------
        mapping : Mapping[str, float]
            The metrics, by the name each is published under.
        step : int
            The number of optimizer steps the run had taken when the metrics
            were reported.
        """


@runtime_checkable
class CheckpointStore[StateT](Protocol):
    """Checkpoint storage."""

    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, StateT],
        /,
    ) -> CheckpointId:
        """Write a checkpoint and return the handle it was written under.

        Parameters
        ----------
        manifest : CheckpointManifest
            What the checkpoint records about the run it is written from.
        states : Mapping[str, StateT]
            The state to write for each component, keyed by the role it fills.

        Returns
        -------
        CheckpointId
            The handle the checkpoint may be read back under.
        """
        ...

    def load(
        self,
        checkpoint: CheckpointId,
        expected: CheckpointManifest,
        /,
    ) -> Checkpoint[StateT]:
        """Load a checkpoint.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.
        expected : CheckpointManifest
            The manifest the reading run would write for itself.

        Returns
        -------
        Checkpoint[StateT]
            The saved manifest and the state of each role.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        FingerprintMismatchError
            If the saved manifest disagrees with the expected one, listing every
            disagreement rather than the first.
        """
        ...

    def list(self) -> tuple[CheckpointId, ...]:
        """Return checkpoint identifiers in storage order.

        Returns
        -------
        tuple[CheckpointId, ...]
            The handles, oldest first.
        """
        ...

    def manifest(self, checkpoint: CheckpointId, /) -> CheckpointManifest:
        """Return the manifest a checkpoint was written with, without its state.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        """
        ...

    def remove(self, checkpoint: CheckpointId, /) -> None:
        """Remove a checkpoint, so it is neither listed nor read again.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        """
        ...


@runtime_checkable
class OptimizationPolicy[LossT](Protocol):
    """Optimization state and complete training steps."""

    def step(self, forward: Callable[[], LossT], /) -> OptimizationResult:
        """Run one accumulation unit and report its outcome.

        Parameters
        ----------
        forward : Callable[[], LossT]
            The computation that produces the loss.
        """
        ...


@runtime_checkable
class FormReward(Protocol):
    """The shaped credit one sampled logical form earns.

    The credit is a function of the form and of the form the item is judged by,
    and is constant with respect to every model parameter, which is what lets a
    score-function estimator treat it as a reward rather than as part of the
    objective.
    """

    def score(self, form: Term, /, *, reference: Term | None) -> ShapedOutcome:
        """Pay one form against the form its item is judged by.

        Parameters
        ----------
        form : Term
            The logical form a draw returned.
        reference : Term | None
            The form the proof is against, or None where the item has none, in
            which case the payment stops short of a proof.

        Returns
        -------
        ShapedOutcome
            What the form was paid, or the stopping condition the prover named
            where it reached no answer at all.
        """
        ...


@runtime_checkable
class SidedPairs(Protocol):
    """A pair batch carrying the vector each of its sides is decoded from.

    A side is a premise or a hypothesis, named by the id its scored-forms beam is
    published under. One vector stands for each named side and no side is named
    twice, so a sentence serving as the premise of two pairs is decoded once and
    both pairs read the same beam.

    What stands beside the pairs is the batch's own business: a corpus with
    relations established by proof carries them, and a collection of human
    judgments carries the responses instead.
    """

    @property
    def premises(self) -> tuple[ReadoutId, ...]:
        """Return the id each pair's premise beam is published under, by pair."""

    @property
    def hypotheses(self) -> tuple[ReadoutId, ...]:
        """Return the id each pair's hypothesis beam is published under, by pair."""

    @property
    def alignments(self) -> tuple[tuple[ArgumentRenaming, ...], ...]:
        """Return the argument alignments each pair declares, at least one per pair."""

    @property
    def sides(self) -> tuple[ReadoutId, ...]:
        """Return the id each conditioning vector's beam is published under, by row."""

    @property
    def vectors(self) -> Encoded:
        """Return one vector per side, in the order ``sides`` names them."""


@runtime_checkable
class GroundedPairs(InferencePairs, SidedPairs, Protocol):
    """A pair batch whose every pair declares the relation a proof established."""


@runtime_checkable
class ParameterDeclaration(Protocol):
    """A component's own account of the parameter groups it holds."""

    @property
    def parameter_groups(self) -> ParameterGroups:
        """Return the groups the component declares over its own modules."""


@runtime_checkable
class StageFitting[StateT](Protocol):
    """What a staged protocol fits at each stage and checkpoints it from."""

    def fit(self, resolution: GroupResolution, step: int, /) -> TrainResult:
        """Fit the parameters one stage left active for one epoch.

        Parameters
        ----------
        resolution : GroupResolution
            The parameters the stage trains, beside which every other parameter
            of the model requires no gradient.
        step : int
            The count of optimizer steps the run has taken before this epoch.

        Returns
        -------
        TrainResult
            What the epoch reported, at the count of optimizer steps the run has
            taken once it is done.
        """
        ...

    def checkpoint(
        self, step: int, /
    ) -> tuple[CheckpointManifest, Mapping[str, StateT]]:
        """Return what a stage's checkpoint is written from.

        Parameters
        ----------
        step : int
            The count of optimizer steps the run has taken.

        Returns
        -------
        tuple[CheckpointManifest, Mapping[str, StateT]]
            The manifest the run would write for itself, and the state to write
            for each component keyed by the role it fills.
        """
        ...


@runtime_checkable
class StepDeclaration(Protocol):
    """A step declaration."""

    @property
    def name(self) -> str:
        """Return the step name."""

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""

    @property
    def produces(self) -> frozenset[str]:
        """Return produced value names."""


@runtime_checkable
class TrainStep[InputsT, OutputsT](StepDeclaration, Protocol):
    """A training step."""

    def run(self, inputs: InputsT, /) -> StepOutcome[OutputsT, TrainResult]:
        """Run the step once and return what it published and what it reported.

        Parameters
        ----------
        inputs : InputsT
            The values the step reads, in the type its family declares.

        Returns
        -------
        StepOutcome[OutputsT, TrainResult]
            What the step published, and the numbers it reported for this optimizer
            step.
        """
        ...


@runtime_checkable
class EvalStep[InputsT, OutputsT](StepDeclaration, Protocol):
    """An evaluation step."""

    def run(self, inputs: InputsT, /) -> StepOutcome[OutputsT, EvalResult]:
        """Run the step once and return what it published and what it reported.

        Parameters
        ----------
        inputs : InputsT
            The values the step reads, in the type its family declares.

        Returns
        -------
        StepOutcome[OutputsT, EvalResult]
            What the step published, and the numbers it reported over the items it
            scored.
        """
        ...


@runtime_checkable
class Forward[BatchT](Protocol):
    """Publishes the outputs objectives read for one batch."""

    @property
    def publishes(self) -> frozenset[str]:
        """Return the names of the outputs every published bundle fills."""

    def publish(self, batch: BatchT, /) -> ObjectiveOutputs[BatchT]:
        """Return what the graph publishes for one batch.

        Parameters
        ----------
        batch : BatchT
            What the featurizer handed the step.

        Returns
        -------
        ObjectiveOutputs[BatchT]
            The bundle, with every name in ``publishes`` filled and every other
            output absent.
        """
        ...


@runtime_checkable
class ValueCodec[V](Protocol):
    """A JSON wire form for the values that flow along step-graph edges."""

    def encode(self, value: V, /) -> JsonValue:
        """Return the value as JSON-shaped data.

        Parameters
        ----------
        value : V
            A value a step published or a run supplied.

        Returns
        -------
        JsonValue
            Data from which ``decode`` rebuilds an equal value, identical across
            processes for equal values.
        """
        ...

    def decode(self, data: JsonValue, /) -> V:
        """Return the value some JSON-shaped data encodes.

        Parameters
        ----------
        data : JsonValue
            What ``encode`` returned.

        Returns
        -------
        V
            A value equal to the one encoded.

        Raises
        ------
        ValueError
            If the data encodes no value.
        """
        ...
