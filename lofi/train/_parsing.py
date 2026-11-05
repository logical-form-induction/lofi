"""The parser training step, and the chart forward pass it trains through."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name, ensure_non_negative, refuse_reassignment
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import (
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._values import (
    OUTSIDE_CHART,
    CellChart,
    ObjectiveOutputs,
)
from lofi.spec.readouts import ReadoutSource
from lofi.train._optimizing import Optimizing, stepped_result
from lofi.train._pretraining import Trained
from lofi.train._steps import StepOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._types import ReadoutId
    from lofi.judge import Action, ActionSet
    from lofi.nn._values import ChartActionScores, LogDistribution
    from lofi.nn.grounding import GroundingMetadata, GroundingTelemetry
    from lofi.nn.loss._ports import SurfaceItems
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.loss._values import SpanReadout, SpanTerms
    from lofi.nn.ports import ChartScorer, SubwordPooler, TokenEncoder
    from lofi.terms import TargetSet, Term
    from lofi.train._losses import LossAssembly
    from lofi.train._optimization import OptimizationResult
    from lofi.train._ports import Forward, OptimizationPolicy
    from lofi.train._steps import TrainResult

__all__ = ["ChartForward", "Featurized", "ParserStep"]


class Featurized[BatchT]:
    """The values a parser step reads.

    Parameters
    ----------
    batch : BatchT
        What the featurizer handed the step.
    step : int
        The count of optimizer steps the run has taken before this one.

    Raises
    ------
    TypeError
        If the step is not an integer.
    ValueError
        If the step is negative.
    """

    __slots__ = ("_batch", "_step")

    def __init__(self, batch: BatchT, step: int) -> None:
        object.__setattr__(self, "_batch", batch)
        object.__setattr__(self, "_step", ensure_non_negative("step", step))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("featurized batch")

    @property
    def batch(self) -> BatchT:
        """Return the batch."""
        return self._batch

    @property
    def step(self) -> int:
        """Return the count of optimizer steps taken before this one."""
        return self._step

    @override
    def __repr__(self) -> str:
        """Return the batch and the position."""
        return f"Featurized({self._batch!r}, step={self._step!r})"


def _published_names(
    declared: frozenset[str],
    *,
    readouts: bool,
    terms: bool,
    actions: bool,
) -> frozenset[str]:
    """Return the outputs a chart forward pass fills from what its scorer declares.

    Parameters
    ----------
    declared : frozenset[str]
        The readouts the scorer fills, by the name each has on its record.
    readouts : bool
        Whether a term readout publishes distributions over the cells.
    terms : bool
        Whether a term readout decodes weighted complete terms from the cells.
    actions : bool
        Whether a complete chart-action inventory is published.

    Returns
    -------
    frozenset[str]
        The names an assembly may declare against this pass.
    """
    names = {"spans", "chart", "batch"}
    derived = (
        ("outside", OUTSIDE_CHART),
        ("alpha", "splits"),
        ("log_z", "scores"),
        ("cells", "cells"),
        ("outside_cells", "outside_cells"),
        ("entropy", "entropy"),
        ("mbr", "mbr"),
        ("derivations", "derivations"),
    )
    names.update(name for readout, name in derived if readout in declared)

    if readouts:
        names.update(("readouts", "choice_rows"))

    if terms:
        names.add("terms")

    if {"alpha", "mu"} <= declared:
        names.add("marginals")

    if actions:
        names.add("actions")

    return frozenset(names)


def _marginals(
    mu: Encoded | None,
    splits: SplitPosterior | None,
    /,
    *,
    words: tuple[tuple[str, ...], ...],
) -> SpanMarginals | None:
    """Return constituent and rule marginals when the chart published both."""
    if mu is None or splits is None:
        return None

    valid = splits.valid
    constituent = Encoded(mu.tensor, mu.axes, valid.any(dim=-1))
    conditional = splits.log_probabilities.tensor.exp().masked_fill(~valid, 0.0)
    rule = Encoded(
        mu.tensor.unsqueeze(-1) * conditional,
        splits.log_probabilities.axes,
        valid,
    )
    lengths = mu.tensor.new_tensor([len(item) for item in words]).long()

    return SpanMarginals(splits.spans, constituent, rule, lengths)


@final
class ChartForward[BatchT: SurfaceItems]:
    """Runs the token encoder, the pooler and the chart over a batch and publishes.

    The pass publishes the span index, the cell chart and the batch on every run,
    and beside them whichever of the outside chart, the split posterior, the item
    scores and the cells the scorer declares. A term readout publishes over the
    cells, and an action set assigns every chart-action score column to its action.

    Parameters
    ----------
    tokens : TokenEncoder
        What encodes each item's words to rows.
    pooler : SubwordPooler
        What pools those rows to one vector per word.
    scorer : ChartScorer
        What computes the chart over the pooled words.
    readouts : Sequence[TermsReadout]
        What publishes a term distribution over the cells, each under its own id.
    actions : ActionSet | None
        The exact ordered actions the scorer's action inventory must carry, or None
        where the scorer declares no inventory.

    Raises
    ------
    ValueError
        If the scorer does not declare the inside chart, if a readout is given over a
        scorer that declares no cells, if the scorer declares action scores without
        an action set, or if two readouts publish one id.
    """

    __slots__ = (
        "_actions",
        "_pooler",
        "_publishes",
        "_readouts",
        "_scorer",
        "_tokens",
        "_unavailable_outputs",
    )

    def __init__(
        self,
        tokens: TokenEncoder,
        pooler: SubwordPooler,
        scorer: ChartScorer,
        *,
        readouts: Sequence[TermsReadout] = (),
        actions: ActionSet | None = None,
    ) -> None:
        declared = scorer.publishes
        declares_actions = "actions" in declared

        if "inside" not in declared:
            msg = (
                f"{type(scorer).__name__} declares {sorted(declared)} and no chart "
                "trains without the inside pass"
            )
            raise ValueError(msg)

        required_vectors = {
            "cells" if readout.reads is ReadoutSource.INSIDE else "spans"
            for readout in readouts
        }
        missing_vectors = sorted(required_vectors - declared)

        if missing_vectors:
            msg = (
                f"term readouts require chart vectors {missing_vectors}, and "
                f"{type(scorer).__name__} declares {sorted(declared)}"
            )
            raise ValueError(msg)

        missing_meaning_outputs = sorted(
            {
                output
                for readout in readouts
                for output in readout.meanings.chart_outputs()
            }
            - declared
        )

        if missing_meaning_outputs:
            msg = (
                "meaning induction requires chart outputs "
                f"{missing_meaning_outputs}, and {type(scorer).__name__} declares "
                f"{sorted(declared)}"
            )
            raise ValueError(msg)

        if declares_actions and actions is None:
            msg = (
                f"{type(scorer).__name__} declares a chart-action inventory, and no "
                "action set names its columns"
            )
            raise ValueError(msg)

        keys = [key for readout in readouts for key in (readout.key, *readout.targets)]

        if len(set(keys)) != len(keys):
            msg = f"each readout id is published once, and these repeat: {keys}"
            raise ValueError(msg)

        self._tokens = tokens
        self._pooler = pooler
        self._scorer = scorer
        self._readouts = tuple(readouts)
        self._actions = actions if declares_actions else None
        self._publishes = _published_names(
            declared,
            readouts=any(readout.decoding is None for readout in self._readouts),
            terms=any(readout.decoding is not None for readout in self._readouts),
            actions=self._actions is not None,
        )
        unavailable: dict[str, str] = {}

        if "spans" in declared and "outside_cells" not in declared:
            unavailable["outside_cells"] = (
                "model.parser.outside_weights is 'reuse'; an objective reading "
                "outside vectors alone requires outside_weights='contextual'"
            )

        self._unavailable_outputs: Mapping[str, str] = MappingProxyType(unavailable)

    @property
    def publishes(self) -> frozenset[str]:
        """Return the names of the outputs every published bundle fills."""
        return self._publishes

    @property
    def unavailable_outputs(self) -> Mapping[str, str]:
        """Return configuration diagnostics for deliberately absent outputs."""
        return self._unavailable_outputs

    @property
    def target_sets(self) -> Mapping[ReadoutId, TargetSet[Term]]:
        """Return symbolic target sets keyed by their published mass id."""
        return MappingProxyType(
            {
                key: targets
                for readout in self._readouts
                for key, targets in readout.target_sets.items()
            }
        )

    @property
    def grounding_metadata(self) -> GroundingMetadata | None:
        """Return no grounding identity for an ordinary chart pass."""
        return None

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return no grounding telemetry for an ordinary chart pass."""
        return None

    def _actions_of(
        self, inventory: ChartActionScores | None
    ) -> Mapping[Action, LogDistribution]:
        """Return every score keyed by its exact inventory action, where declared.

        Parameters
        ----------
        inventory : ChartActionScores | None
            The scorer's complete action scores, or None where it declares none.

        Raises
        ------
        ValueError
            If a declared inventory is absent, an undeclared one is present, or the
            returned inventory is not the exact object declared at wiring.
        """
        expected = self._actions

        if expected is None:
            if inventory is not None:
                msg = (
                    f"{type(self._scorer).__name__} filled a chart-action inventory "
                    "after declaring none"
                )
                raise ValueError(msg)

            return MappingProxyType({})

        if inventory is None:
            msg = (
                f"{type(self._scorer).__name__} declared a chart-action inventory "
                "and filled none"
            )
            raise ValueError(msg)

        if inventory.action_set is not expected:
            wanted = tuple(action.name for action in expected.actions)
            received = tuple(action.name for action in inventory.action_set.actions)
            msg = (
                f"{type(self._scorer).__name__} returned the action inventory "
                f"{received}, not the exact declared inventory {wanted}"
            )
            raise ValueError(msg)

        return inventory.by_action

    def publish(self, batch: BatchT, /) -> ObjectiveOutputs[BatchT]:
        """Return what the chart publishes for one batch.

        Parameters
        ----------
        batch : BatchT
            What the featurizer handed the step.

        Returns
        -------
        ObjectiveOutputs[BatchT]
            The bundle over a span index as long as the batch's longest item, with
            every name in ``publishes`` filled.

        Raises
        ------
        ValueError
            If the scorer declared a readout the pass did not fill.
        """
        encoded = self._tokens.encode_words(batch.words)
        leaves = self._pooler.pool(encoded)
        computed = self._scorer.readouts(leaves)
        spans = SpanIndex(leaves.size("word"))
        inside = computed.inside

        if inside is None:
            msg = (
                f"{type(self._scorer).__name__} declared the inside pass and "
                "filled none"
            )
            raise ValueError(msg)

        cells = computed.cells
        span_cells = computed.spans
        outside_cells = computed.outside_cells

        if "outside_cells" in self._scorer.publishes and outside_cells is None:
            msg = (
                f"{type(self._scorer).__name__} declared contextual outside "
                "vectors and filled none"
            )
            raise ValueError(msg)

        for readout in self._readouts:
            selected = cells if readout.reads is ReadoutSource.INSIDE else span_cells

            if selected is None:
                msg = (
                    f"{type(self._scorer).__name__} declared {readout.reads.value!r} "
                    "vectors and filled none"
                )
                raise ValueError(msg)

        alpha = computed.alpha
        splits = None if alpha is None else SplitPosterior(spans, alpha)
        marginals = _marginals(computed.mu, splits, words=batch.words)
        published: dict[ReadoutId, SpanReadout] = {}
        choice_rows: dict[ReadoutId, SpanReadout] = {}
        decoded: dict[ReadoutId, SpanTerms] = {}
        action_weights = self._actions_of(computed.actions)

        for readout in self._readouts:
            selected = cells if readout.reads is ReadoutSource.INSIDE else span_cells

            if selected is None:
                continue

            if readout.decoding is None:
                held = readout.publish_with_choices(
                    selected, spans=spans, marginals=marginals
                )
                published.update(held.readouts)
                choice_rows[readout.key] = held.choices
            else:
                decoded[readout.key] = readout.decode(
                    selected,
                    spans=spans,
                    splits=splits,
                    actions=action_weights,
                )

        return ObjectiveOutputs(
            spans=spans,
            chart=CellChart(inside, computed.outside),
            batch=batch,
            splits=splits,
            scores=computed.log_z,
            marginals=marginals,
            actions=action_weights,
            readouts=published,
            choice_rows=choice_rows,
            terms=decoded,
            cells=cells,
            outside_cells=outside_cells,
            entropy=computed.entropy,
            mbr=computed.mbr,
            derivations=computed.derivations,
            truncated=encoded.truncated,
        )

    @override
    def __repr__(self) -> str:
        """Return the three components' types and what the pass publishes."""
        return (
            f"ChartForward(tokens={type(self._tokens).__name__}, "
            f"pooler={type(self._pooler).__name__}, "
            f"scorer={type(self._scorer).__name__}, "
            f"publishes={sorted(self._publishes)!r})"
        )


@final
class ParserStep[BatchT: SurfaceItems]:
    """Trains the parser on featurized batches through a published forward pass.

    Each run publishes the forward pass over the batch, evaluates the assembled loss
    on what was published, and takes one optimizer step on the total.

    Parameters
    ----------
    forward : Forward[BatchT]
        What publishes the outputs the objectives read.
    assembly : LossAssembly[BatchT]
        The weighted objectives the parser trains under.
    policy : OptimizationPolicy[Tensor]
        What runs the backward pass and the optimizer.
    role : str
        The role the parser's state is checkpointed under.
    name : str | None
        The step's name, or None to name it after the role.

    Raises
    ------
    TypeError
        If the role or the name is not a string.
    ValueError
        If the role or the name is empty.
    MissingOutputError
        If an objective declares a name the forward pass does not publish.
    """

    __slots__ = ("_forward", "_name", "_optimizing", "_role")

    def __init__(
        self,
        forward: Forward[BatchT],
        assembly: LossAssembly[BatchT],
        policy: OptimizationPolicy[Tensor],
        *,
        role: str,
        name: str | None = None,
    ) -> None:
        for entry in assembly:
            missing = entry.objective.requires - forward.publishes

            if missing:
                raise MissingOutputError(
                    objective=entry.name,
                    missing=missing,
                    published=forward.publishes,
                )

        self._forward = forward
        self._optimizing: Optimizing[BatchT] = Optimizing(assembly, policy)
        self._role = ensure_name("role", role)
        self._name = ensure_name("name", f"train:{role}" if name is None else name)

    @property
    def name(self) -> str:
        """Return the step name."""
        return self._name

    @property
    def role(self) -> str:
        """Return the checkpoint role."""
        return self._role

    @property
    def requires(self) -> frozenset[str]:
        """Return required value names."""
        return frozenset({"batch", "step"})

    @property
    def produces(self) -> frozenset[str]:
        """Return produced value names."""
        return frozenset({self._role})

    @property
    def forward(self) -> Forward[BatchT]:
        """Return the forward pass."""
        return self._forward

    @property
    def assembly(self) -> LossAssembly[BatchT]:
        """Return the weighted objectives."""
        return self._optimizing.assembly

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return no grounding telemetry for an ordinary parser step."""
        return None

    def run(self, inputs: Featurized[BatchT], /) -> StepOutcome[Trained, TrainResult]:
        """Publish the forward pass over one batch and take one optimizer step on it.

        Parameters
        ----------
        inputs : Featurized[BatchT]
            The batch and the position in the run.

        Returns
        -------
        StepOutcome[Trained, TrainResult]
            The checkpoint role, and the result over the batch's items with the
            optimizer and loss scalars beside ``tokens/truncated_items``, how many
            items the token encoder encoded only a prefix of. A batch of no items
            takes no step.
        """
        batch = inputs.batch
        step = inputs.step
        items = len(batch.words)
        metrics: dict[str, float] = {}
        reported: OptimizationResult | None = None

        if items:
            outputs = self._forward.publish(batch)
            reported, metrics = self._optimizing.run(outputs, step=step)
            metrics["tokens/truncated_items"] = float(
                sum(1 for dropped in outputs.truncated if dropped)
            )

        return StepOutcome(
            Trained(self._role), stepped_result(step, items, reported, metrics)
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the role, the forward pass and the assembly."""
        return (
            f"ParserStep(name={self._name!r}, role={self._role!r}, "
            f"forward={self._forward!r}, assembly={self.assembly!r})"
        )
