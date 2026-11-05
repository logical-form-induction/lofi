"""The single grounded-inference batch, forward pass, and training step."""

from typing import TYPE_CHECKING, Protocol, cast, final, override

import torch

from lofi._checks import ensure_name
from lofi.lf import InferenceCell, InferenceLabel
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.entailment import PairedVectors
from lofi.nn.grounding import (
    GroundedPairs,
    GroundingInput,
    GroundingMetadata,
    GroundingQuery,
    GroundingTelemetry,
    Phase,
)
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._values import ObjectiveOutputs
from lofi.train._optimizing import Optimizing, stepped_result
from lofi.train._pretraining import Trained
from lofi.train._steps import StepOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi._types import ReadoutId
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding import GroundedBatch, InferenceGrounding
    from lofi.train._losses import LossAssembly
    from lofi.train._parsing import Featurized
    from lofi.train._ports import Forward, OptimizationPolicy
    from lofi.train._steps import TrainResult

__all__ = [
    "GroundedForward",
    "GroundedStep",
    "grounding_metrics",
]


class GroundingTrainingBatch(GroundedPairs, Protocol):
    """A grounded pair batch over featurized premise and hypothesis rows."""

    @property
    def words(self) -> tuple[tuple[str, ...], ...]: ...

    @property
    def premise_rows(self) -> tuple[int, ...]: ...

    @property
    def hypothesis_rows(self) -> tuple[int, ...]: ...


def _root_rows(
    batch: GroundingTrainingBatch, outputs: ObjectiveOutputs[object], /
) -> Encoded:
    """Return one chart root vector per featurized side row."""
    cells = outputs.cells

    if cells is None:
        raise MissingOutputError(
            objective="inference grounding",
            missing=frozenset({"cells"}),
            published=outputs.published,
        )

    roots = tuple(outputs.spans.span_id(0, len(words)) for words in batch.words)
    rows = torch.arange(len(roots), device=cells.tensor.device)
    columns = torch.tensor(roots, device=cells.tensor.device)
    return Encoded(cells.tensor[rows, columns], ("item", "width"))


def _sentence_pairs(
    batch: GroundingTrainingBatch, outputs: ObjectiveOutputs[object], /
) -> PairedVectors:
    """Resolve paired vectors from each side's chart-root representation."""
    vectors = _root_rows(batch, outputs)
    premise = torch.stack([vectors.tensor[row] for row in batch.premise_rows])
    hypothesis = torch.stack([vectors.tensor[row] for row in batch.hypothesis_rows])

    return PairedVectors(
        Encoded(premise, ("pair", "width")),
        Encoded(hypothesis, ("pair", "width")),
    )


def _grounding_forms(
    batch: GroundingTrainingBatch,
    outputs: ObjectiveOutputs[object],
    /,
    *,
    premise_readout: ReadoutId,
    hypothesis_readout: ReadoutId,
) -> Mapping[ReadoutId, WeightedTerms]:
    """Return root-span forms under the side ids consumed by the grounding."""
    try:
        premises = outputs.terms[premise_readout]
        hypotheses = outputs.terms[hypothesis_readout]
    except KeyError as error:
        offered = tuple(sorted(outputs.terms))
        msg = (
            f"inference grounding reads decoded forms from {error.args[0]!r}; "
            f"the forward pass published {offered!r}"
        )
        raise ValueError(msg) from error

    held: dict[ReadoutId, WeightedTerms] = {}

    for side, row in zip(batch.premises, batch.premise_rows, strict=True):
        root = outputs.spans.span_id(0, len(batch.words[row]))
        held[side] = premises.at(row, root)

    for side, row in zip(batch.hypotheses, batch.hypothesis_rows, strict=True):
        root = outputs.spans.span_id(0, len(batch.words[row]))
        held[side] = hypotheses.at(row, root)

    return held


def _grounding_choice_rows(
    batch: GroundingTrainingBatch,
    outputs: ObjectiveOutputs[object],
    /,
    *,
    premise_readout: ReadoutId,
    hypothesis_readout: ReadoutId,
) -> Mapping[ReadoutId, LogDistribution]:
    """Return root-span choice marginals under the pair-side ids."""
    held: dict[ReadoutId, LogDistribution] = {}

    for sides, rows, source in (
        (batch.premises, batch.premise_rows, premise_readout),
        (batch.hypotheses, batch.hypothesis_rows, hypothesis_readout),
    ):
        try:
            published = outputs.choice_rows[source]
        except KeyError:
            offered = tuple(sorted(outputs.choice_rows))
            msg = (
                f"inference grounding reads choice rows from {source!r}; "
                f"the forward pass published {offered!r}"
            )
            raise ValueError(msg) from None

        for side, row in zip(sides, rows, strict=True):
            cell = published.at(row, 0, len(batch.words[row]))
            held[side] = LogDistribution(
                cell.tensor.unsqueeze(0),
                ("term", *cell.axes),
                normalized_over=cell.normalized_over,
            )

    return held


def _label_agreement(
    grounded: GroundedBatch,
    labels: tuple[InferenceLabel, ...],
    /,
) -> float | None:
    """Return argmax agreement on the pairs for which grounding decided."""
    pairs = len(grounded.decided)

    if len(labels) != pairs:
        msg = f"grounding produced {pairs} pairs for {len(labels)} symbolic labels"
        raise ValueError(msg)

    decided = grounded.decided

    if not bool(decided.any()):
        return None

    predicted = grounded.log_probabilities.tensor.detach().argmax(dim=1)
    targets = torch.tensor(
        [grounded.labels.index(label) for label in labels],
        device=predicted.device,
    )
    return float((predicted[decided] == targets[decided]).float().mean())


def grounding_metrics(
    grounded: GroundedBatch,
    labels: tuple[InferenceLabel, ...],
    /,
) -> dict[str, float]:
    """Return the complete per-step grounding telemetry."""
    ledger = grounded.ledger
    pairs = len(grounded.decided)
    decided = int(grounded.decided.sum())
    variance = grounded.variance
    agreement = _label_agreement(grounded, labels)
    degenerate = grounded.cells.tensor[
        :, tuple(InferenceCell).index(InferenceCell.DEGENERATE)
    ]
    metrics = {
        "grounded/gradient_path": float(
            tuple(type(grounded.gradient_path)).index(grounded.gradient_path)
        ),
        "grounded/pairs": float(pairs),
        "grounded/decided": float(decided),
        "grounded/abstention_rate": (pairs - decided) / pairs if pairs else 0.0,
        "grounded/degenerate_mass": (
            float(degenerate.detach().mean()) if pairs else 0.0
        ),
        "grounded/calls": float(ledger.calls),
        "grounded/questions": float(ledger.questions),
        "grounded/questions_per_second": ledger.questions_per_second,
        "grounded/memo_hits": float(ledger.memo_hits),
        "grounded/memo_hit_rate": ledger.memo_hit_rate,
        "grounded/cpu_seconds": ledger.cpu_seconds,
        "grounded/wall_seconds": ledger.wall_seconds,
        "grounded/process_starts": float(ledger.process_starts or 0),
        "grounded/label_agreement": 0.0 if agreement is None else agreement,
        "grounded/verified_share": ledger.verified_share or 0.0,
        "grounded/out_of_fragment_share": (
            ledger.abstentions.get("out-of-fragment", 0) / ledger.questions
            if ledger.questions
            else 0.0
        ),
        "grounded/variance": 0.0 if variance is None else float(variance.mean()),
    }

    for cause in sorted(ledger.abstentions):
        metrics[f"grounded/abstention/{cause}"] = float(ledger.abstentions[cause])

    return metrics


def _grounding_telemetry(
    grounded: GroundedBatch,
    labels: tuple[InferenceLabel, ...],
    /,
) -> GroundingTelemetry:
    """Detach one grounded batch into checkpoint-safe scalar telemetry."""
    ledger = grounded.ledger
    pairs = len(labels)
    decided = grounded.decided
    rows = torch.arange(pairs, device=decided.device)
    columns = torch.tensor(
        [grounded.labels.index(label) for label in labels],
        device=decided.device,
    )
    probabilities = (
        grounded.log_probabilities.tensor[rows, columns].detach().exp()[decided]
    )
    reward = grounded.reward
    rewards = (
        probabilities.new_empty((0,))
        if reward is None
        else probabilities.mul(reward.proved - reward.refuted).add(reward.refuted)
    )
    degenerate = grounded.cells.tensor[
        :, tuple(InferenceCell).index(InferenceCell.DEGENERATE)
    ]
    variance = grounded.variance

    return GroundingTelemetry(
        grounded.gradient_path,
        {str(cause): count for cause, count in ledger.abstentions.items()},
        calls=ledger.calls,
        questions=ledger.questions,
        questions_per_second=ledger.questions_per_second,
        memo_hits=ledger.memo_hits,
        memo_hit_rate=ledger.memo_hit_rate,
        cpu_seconds=ledger.cpu_seconds,
        wall_seconds=ledger.wall_seconds,
        process_starts=ledger.process_starts,
        label_agreement=_label_agreement(grounded, labels),
        degenerate_mass=float(degenerate.detach().mean()) if pairs else 0.0,
        malformed_mass=0.0,
        reward_mean=(None if not rewards.numel() else float(rewards.mean())),
        variance=(None if variance is None else float(variance.detach().mean())),
        verified_share=ledger.verified_share,
        out_of_fragment_share=(
            ledger.abstentions.get("out-of-fragment", 0) / ledger.questions
            if ledger.questions
            else 0.0
        ),
    )


@final
class GroundedForward[BatchT: GroundingTrainingBatch]:
    """Publish a model pass and its one batched inference grounding."""

    __slots__ = (
        "_grounding",
        "_hypothesis_readout",
        "_inner",
        "_latest",
        "_phase",
        "_premise_readout",
        "_publishes",
        "_query",
    )

    def __init__(
        self,
        inner: Forward[BatchT],
        grounding: InferenceGrounding,
        /,
        *,
        query: GroundingQuery,
        premise_readout: ReadoutId | None = None,
        hypothesis_readout: ReadoutId | None = None,
        phase: Phase = Phase.TRAIN,
    ) -> None:
        if not isinstance(phase, Phase):
            msg = f"a grounded forward phase is Phase, not {phase!r}"
            raise TypeError(msg)

        if not isinstance(query, GroundingQuery):
            msg = f"a grounded forward query is GroundingQuery, not {query!r}"
            raise TypeError(msg)

        self._inner = inner
        self._grounding = grounding
        self._premise_readout = premise_readout
        self._hypothesis_readout = hypothesis_readout
        self._phase = phase
        self._query = query
        self._latest: GroundingTelemetry | None = None
        self._publishes = (
            inner.publishes
            | {"entailment"}
            | ({"forms"} if premise_readout is not None else set())
        )

    @property
    def publishes(self) -> frozenset[str]:
        return self._publishes

    @property
    def grounding(self) -> InferenceGrounding:
        return self._grounding

    @property
    def inner(self) -> Forward[BatchT]:
        return self._inner

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def grounding_metadata(self) -> GroundingMetadata:
        """Return the grounding's stable checkpoint identity."""
        return self._grounding.grounding_metadata

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return detached telemetry from the most recent published batch."""
        return self._latest

    def publish(self, batch: BatchT, /) -> ObjectiveOutputs[BatchT]:
        outputs = self._inner.publish(batch)
        available = (
            outputs.published
            | (
                {"forms"}
                if self._premise_readout is not None
                and self._hypothesis_readout is not None
                else set()
            )
            | ({"sentences"} if outputs.cells is not None else set())
        )
        missing = self._grounding.requires - available

        if missing:
            raise MissingOutputError(
                objective="inference grounding",
                missing=missing,
                published=outputs.published,
            )

        sentences = (
            _sentence_pairs(batch, cast("ObjectiveOutputs[object]", outputs))
            if "sentences" in self._grounding.requires
            else None
        )
        forms = outputs.forms
        choice_rows: Mapping[ReadoutId, LogDistribution] = {}

        if "forms" in self._grounding.requires:
            premise_readout = self._premise_readout
            hypothesis_readout = self._hypothesis_readout

            if premise_readout is None or hypothesis_readout is None:
                msg = "form grounding requires premise and hypothesis readouts"
                raise RuntimeError(msg)

            forms = _grounding_forms(
                batch,
                cast("ObjectiveOutputs[object]", outputs),
                premise_readout=premise_readout,
                hypothesis_readout=hypothesis_readout,
            )

        if "choice_rows" in self._grounding.requires:
            premise_readout = self._premise_readout
            hypothesis_readout = self._hypothesis_readout

            if premise_readout is None or hypothesis_readout is None:
                msg = "choice-row grounding requires premise and hypothesis readouts"
                raise RuntimeError(msg)

            choice_rows = _grounding_choice_rows(
                batch,
                cast("ObjectiveOutputs[object]", outputs),
                premise_readout=premise_readout,
                hypothesis_readout=hypothesis_readout,
            )
        grounded = self._grounding.ground(
            GroundingInput(
                forms,
                choice_rows=choice_rows,
                pairs=batch,
                query=self._query,
                sentences=sentences,
                phase=self._phase,
            )
        )
        self._latest = _grounding_telemetry(grounded, batch.labels)

        return ObjectiveOutputs(
            spans=outputs.spans,
            chart=outputs.chart,
            batch=outputs.batch,
            splits=outputs.splits,
            scores=outputs.scores,
            marginals=outputs.marginals,
            actions=outputs.actions,
            readouts=outputs.readouts,
            choice_rows=outputs.choice_rows,
            terms=outputs.terms,
            forms=outputs.forms,
            entailment=grounded,
            cells=outputs.cells,
            outside_cells=outputs.outside_cells,
            truncated=outputs.truncated,
        )

    def open_step(self) -> None:
        """Reset step telemetry while preserving scorer memo state."""
        self._grounding.open_step()

    @override
    def __repr__(self) -> str:
        return (
            f"GroundedForward(inner={self._inner!r}, grounding={self._grounding!r}, "
            f"phase={self._phase.value!r})"
        )


@final
class GroundedStep[BatchT: GroundingTrainingBatch]:
    """Optimize every declared objective over one grounded pair batch."""

    __slots__ = ("_forward", "_latest", "_name", "_optimizing", "_role")

    def __init__(
        self,
        forward: GroundedForward[BatchT],
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
        self._latest: GroundingTelemetry | None = None
        self._role = ensure_name("role", role)
        self._name = ensure_name("name", f"grounded:{role}" if name is None else name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"batch", "step"})

    @property
    def produces(self) -> frozenset[str]:
        return frozenset({self._role})

    @property
    def forward(self) -> GroundedForward[BatchT]:
        return self._forward

    @property
    def assembly(self) -> LossAssembly[BatchT]:
        """Return the weighted objective optimized by this step."""
        return self._optimizing.assembly

    @property
    def grounding_telemetry(self) -> GroundingTelemetry | None:
        """Return telemetry from the most recent completed optimizer batch."""
        return self._latest

    def run(self, inputs: Featurized[BatchT], /) -> StepOutcome[Trained, TrainResult]:
        batch = inputs.batch
        pairs = len(batch.labels)

        if not pairs:
            return StepOutcome(
                Trained(self._role),
                stepped_result(inputs.step, 0, None, {}, dropped=0),
            )

        outputs = self._forward.publish(batch)
        grounded = outputs.entailment

        if grounded is None:
            msg = "a grounded forward published no entailment batch"
            raise RuntimeError(msg)

        reported, metrics = self._optimizing.run(outputs, step=inputs.step)
        metrics.update(grounding_metrics(grounded, batch.labels))
        decided = int(grounded.decided.sum())
        self._latest = self._forward.grounding_telemetry
        self._forward.open_step()

        return StepOutcome(
            Trained(self._role),
            stepped_result(
                inputs.step,
                decided,
                reported,
                metrics,
                dropped=pairs - decided,
            ),
        )

    @override
    def __repr__(self) -> str:
        return f"GroundedStep(name={self._name!r}, role={self._role!r})"
