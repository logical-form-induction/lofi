"""Validated epoch fitting with distinct best and resumable checkpoints."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from math import fsum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Never, NewType, cast, final, override

import torch

from lofi._checks import NumericalFailureError, ensure_finite
from lofi._frozen import FrozenValue
from lofi.train._alternating import StepTask
from lofi.train._boundary import (
    CheckpointManifest,
    ParserFittingRecord,
    RngRecord,
    StateReference,
    StepFittingRecord,
    grounding_record,
    staged_manifest,
)
from lofi.train._checkpoint import CheckpointId
from lofi.train._configured import ParserFittingPlan, StepFittingPlan
from lofi.train._early_stopping import AccuracyStoppingDecision, EarlyStoppingState
from lofi.train._parsing import Featurized
from lofi.train._pretraining import Draw
from lofi.train._resume import ResumePosition
from lofi.train._steps import TrainResult
from lofi.train._validation import validate_epoch
from lofi.training_plugins import TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.data import SelectionReport
    from lofi.train._configured import PreparedTraining
    from lofi.train._losses import LossAssembly
    from lofi.train._ports import (
        CheckpointStore,
        Forward,
        MetricLog,
        PositionedBatch,
        RestorableRandomStreams,
    )
    from lofi.train._steps import EvalResult, StepOutcome
    from lofi.train._store import StateTree
    from lofi.train._validation import _ValidationBatch
    from lofi.training_plugins.steps import FittingTask

__all__ = [
    "BestCheckpointId",
    "EpochReport",
    "FittingReport",
    "ResumeCheckpointId",
    "fit_training",
    "fitting_remains",
]

BestCheckpointId = NewType("BestCheckpointId", CheckpointId)
"""A checkpoint selected by the global validation argbest."""

ResumeCheckpointId = NewType("ResumeCheckpointId", CheckpointId)
"""The latest checkpoint at an eligible completed optimizer-step boundary."""


def _log(
    sinks: Sequence[MetricLog], metrics: Mapping[str, float], step: int, /
) -> None:
    """Log one set of metrics at one global optimizer step to every sink."""
    for sink in sinks:
        sink.log_scalars(metrics, step)


def _step_metrics(window: Sequence[TrainResult], /) -> dict[str, float]:
    """Return the metrics of one optimizer step from its accumulation window.

    A metric under ``optimizer/`` describes the step itself and is read from
    the window's last microbatch, the one that stepped. Every other metric is
    the mean over the microbatches that report it, as the accumulated
    gradient is the mean of the microbatches' gradients.

    Parameters
    ----------
    window : Sequence[TrainResult]
        The microbatch results the step accumulated, in order, ending with the
        one that stepped.
    """
    last = window[-1].metrics
    merged: dict[str, float] = {}

    for name in sorted({name for result in window for name in result.metrics}):
        if name.startswith("optimizer/"):
            if name in last:
                merged[name] = last[name]

            continue

        values = [result.metrics[name] for result in window if name in result.metrics]
        merged[name] = fsum(values) / len(values)

    return merged


def _batch_order(batches: Sequence[PositionedBatch], /) -> float:
    """Return an exact scalar fingerprint of one ordered batch window."""
    digest = sha256()

    for batch in batches:
        digest.update(len(batch.item_ids).to_bytes(8, "big"))

        for item_id in batch.item_ids:
            payload = str(item_id).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)

    return float(int.from_bytes(digest.digest()[:6]))


def _positioned[BatchT: PositionedBatch](batch: BatchT, step: int, /) -> BatchT:
    """Return the immutable batch at its current optimizer-step position."""
    return batch.at_step(step)


@final
class EpochReport(FrozenValue):
    """Training steps and validation metrics from one completed epoch."""

    _noun = "configured epoch report"

    __slots__ = ("_epoch", "_results", "_validation")

    def __init__(
        self,
        epoch: int,
        results: tuple[TrainResult, ...],
        validation: EvalResult | None,
    ) -> None:
        object.__setattr__(self, "_epoch", epoch)
        object.__setattr__(self, "_results", results)
        object.__setattr__(self, "_validation", validation)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        msg = "a configured epoch report is immutable"
        raise AttributeError(msg)

    @property
    def epoch(self) -> int:
        """Return the zero-based completed epoch."""
        return self._epoch

    @property
    def results(self) -> tuple[TrainResult, ...]:
        """Return its microbatch reports in iteration order."""
        return self._results

    @property
    def validation(self) -> EvalResult | None:
        """Return validation metrics, or none before the owning stage."""
        return self._validation

    @property
    def optimizer_steps(self) -> int:
        """Return how many optimizer steps the epoch closed."""
        return sum(int(result.metrics["optimizer/stepped"]) for result in self._results)


@final
class FittingReport(FrozenValue):
    """Reports completed work, counters and checkpoint selections."""

    _noun = "configured fitting report"

    __slots__ = (
        "_achieved_accuracy",
        "_best_checkpoint",
        "_constraint_selection",
        "_data_position",
        "_epochs",
        "_latest_checkpoint",
        "_optimizer_steps",
        "_results",
        "_stop_reason",
        "_stopped_early",
        "_unit",
    )

    def __init__(
        self,
        epochs: tuple[EpochReport, ...],
        results: tuple[TrainResult, ...],
        optimizer_steps: int,
        data_position: int,
        *,
        unit: TrainingUnit,
        constraint_selection: SelectionReport[str] | None,
        best_checkpoint: BestCheckpointId | None,
        latest_checkpoint: ResumeCheckpointId,
        stopped_early: bool,
        stop_reason: Literal[
            "horizon", "early-stopping", "target-met", "target-not-met"
        ]
        | None = None,
        achieved_accuracy: float | None = None,
    ) -> None:
        object.__setattr__(self, "_epochs", epochs)
        object.__setattr__(self, "_results", results)
        object.__setattr__(self, "_optimizer_steps", optimizer_steps)
        object.__setattr__(self, "_data_position", data_position)
        object.__setattr__(self, "_best_checkpoint", best_checkpoint)
        object.__setattr__(self, "_latest_checkpoint", latest_checkpoint)
        object.__setattr__(self, "_stopped_early", stopped_early)
        reason = (
            ("early-stopping" if stopped_early else "horizon")
            if stop_reason is None
            else stop_reason
        )
        target_stopping = reason in {"target-met", "target-not-met"}

        if target_stopping != (achieved_accuracy is not None):
            msg = "target stopping reports both its reason and achieved accuracy"
            raise ValueError(msg)

        if achieved_accuracy is not None:
            held = ensure_finite("achieved_accuracy", achieved_accuracy)

            if not 0.0 <= held <= 1.0:
                msg = f"achieved_accuracy is outside [0, 1]: {achieved_accuracy!r}"
                raise ValueError(msg)

        object.__setattr__(self, "_stop_reason", reason)
        object.__setattr__(self, "_achieved_accuracy", achieved_accuracy)
        object.__setattr__(self, "_unit", unit)
        object.__setattr__(self, "_constraint_selection", constraint_selection)

    @property
    def unit(self) -> TrainingUnit:
        """Return the unit the fitted plan's horizon is counted in."""
        return self._unit

    @property
    def epochs(self) -> tuple[EpochReport, ...]:
        """Return epochs completed by this fitting invocation."""
        return self._epochs

    @property
    def results(self) -> tuple[TrainResult, ...]:
        """Return step-counted results in optimizer order."""
        return self._results

    @property
    def optimizer_steps(self) -> int:
        """Return the total number of completed optimizer steps."""
        return self._optimizer_steps

    @property
    def data_position(self) -> int:
        """Return the global count of consumed training occurrences."""
        return self._data_position

    @property
    def best_checkpoint(self) -> BestCheckpointId | None:
        """Return the validation-best handle, or none for step-counted fitting."""
        return self._best_checkpoint

    @property
    def latest_checkpoint(self) -> ResumeCheckpointId:
        """Return the latest resumable completed-step checkpoint handle."""
        return self._latest_checkpoint

    @property
    def stopped_early(self) -> bool:
        """Return whether patience ended fitting before the epoch horizon."""
        return self._stopped_early

    @property
    def stop_reason(
        self,
    ) -> Literal["horizon", "early-stopping", "target-met", "target-not-met"]:
        """Return why this fitting invocation ended."""
        return self._stop_reason

    @property
    def achieved_accuracy(self) -> float | None:
        """Return the final held-out accuracy of target stopping, when used."""
        return self._achieved_accuracy

    @property
    def constraint_selection(self) -> SelectionReport[str] | None:
        """Return the configured parser selection budget, if one exists."""
        return self._constraint_selection


@final
class _CheckpointSnapshot(FrozenValue):
    """Detached state and random streams at one completed epoch."""

    _noun = "checkpoint snapshot"

    __slots__ = ("_blocks", "_rng")

    def __init__(
        self,
        blocks: Mapping[str, StateTree],
        rng: Mapping[str, str],
    ) -> None:
        object.__setattr__(self, "_blocks", MappingProxyType(dict(blocks)))
        object.__setattr__(self, "_rng", MappingProxyType(dict(rng)))

    @property
    def blocks(self) -> Mapping[str, StateTree]:
        """Return component and run state blocks by manifest key."""
        return self._blocks

    @property
    def rng(self) -> Mapping[str, str]:
        """Return advancing random streams by declared name."""
        return self._rng


def _snapshot(
    prepared: PreparedTraining,
    streams: RestorableRandomStreams,
    template: CheckpointManifest,
    /,
) -> _CheckpointSnapshot:
    """Return detached complete state at a completed optimizer step."""
    components, optimization = prepared.state()
    blocks: dict[str, StateTree] = {
        role: cast("StateTree", state) for role, state in components.items()
    }

    if template.optimizer is None:
        msg = "the manifest has no optimizer state reference"
        raise ValueError(msg)

    blocks[template.optimizer.key] = cast("StateTree", optimization.optimizer)

    if optimization.scaler is not None:
        if template.scaler is None:
            msg = "the manifest has no scaler state reference"
            raise ValueError(msg)

        blocks[template.scaler.key] = cast("StateTree", optimization.scaler)

    if optimization.scheduler is not None:
        if template.scheduler is None:
            msg = "the manifest has no scheduler state reference"
            raise ValueError(msg)

        blocks[template.scheduler.key] = cast("StateTree", optimization.scheduler)

    return _CheckpointSnapshot(blocks, streams.snapshot())


def _digest(state: StateTree, /) -> str:
    """Return the digest of one weights-only state serialization."""
    serialized = BytesIO()
    torch.save(state, serialized)

    return sha256(serialized.getvalue()).hexdigest()


def _reference(
    reference: StateReference,
    blocks: Mapping[str, StateTree],
    /,
    *,
    block_key: str | None = None,
) -> StateReference:
    """Return a reference carrying the current block digest."""
    key = reference.key if block_key is None else block_key

    return StateReference(key=reference.key, digest=_digest(blocks[key]))


def _parser_manifest(
    template: CheckpointManifest,
    snapshot: _CheckpointSnapshot,
    state: EarlyStoppingState,
    /,
    *,
    kind: Literal["validation-best", "resume-latest"],
    completed_epochs: int,
    step: int,
    data_position: int,
    metric: str,
    best_checkpoint: BestCheckpointId | None,
    stop: Literal["target-met", "target-not-met"] | None = None,
    achieved_accuracy: float | None = None,
) -> CheckpointManifest:
    """Return a manifest for one detached completed epoch snapshot."""
    fitting = ParserFittingRecord(
        kind=kind,
        completed_epochs=completed_epochs,
        metric=metric,
        best=state.best,
        failures=state.failures,
        best_checkpoint=(None if best_checkpoint is None else str(best_checkpoint)),
        stop=stop,
        achieved_accuracy=achieved_accuracy,
    )
    components = {
        role: record.with_(
            state=_reference(record.state, snapshot.blocks, block_key=role)
        )
        for role, record in template.components.items()
    }

    return template.with_(
        step=step,
        data_position=data_position,
        components=components,
        rng=RngRecord(streams=dict(snapshot.rng)),
        optimizer=(
            None
            if template.optimizer is None
            else _reference(template.optimizer, snapshot.blocks)
        ),
        scaler=(
            None
            if template.scaler is None
            else _reference(template.scaler, snapshot.blocks)
        ),
        scheduler=(
            None
            if template.scheduler is None
            else _reference(template.scheduler, snapshot.blocks)
        ),
        fitting=fitting,
    )


def _initial_state(
    prepared: PreparedTraining,
    plan: ParserFittingPlan,
    template: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    /,
) -> tuple[int, EarlyStoppingState, BestCheckpointId | None]:
    """Validate fitting inputs and return recovered epoch-selection state."""
    expected_roles = frozenset(prepared.model.modules)

    if template.roles != expected_roles:
        msg = (
            f"the manifest names component roles {sorted(template.roles)}, not the "
            f"configured roles {sorted(expected_roles)}"
        )
        raise ValueError(msg)

    _, optimization = prepared.state()
    expected_presence = (
        True,
        optimization.scaler is not None,
        optimization.scheduler is not None,
    )
    found_presence = (
        template.optimizer is not None,
        template.scaler is not None,
        template.scheduler is not None,
    )

    if found_presence != expected_presence:
        msg = (
            "the manifest run-state presence "
            f"{found_presence!r} differs from the configured presence "
            f"{expected_presence!r} for optimizer, scaler and scheduler"
        )
        raise ValueError(msg)

    if frozenset(template.rng.streams) != frozenset(streams.snapshot()):
        msg = "the manifest random streams differ from the supplied random streams"
        raise ValueError(msg)

    if (template.step, template.data_position) != (
        position.step,
        position.data_position,
    ):
        msg = (
            f"the manifest position {(template.step, template.data_position)!r} "
            f"differs from {(position.step, position.data_position)!r}"
        )
        raise ValueError(msg)

    fitting = template.fitting

    if fitting is None:
        if position != ResumePosition(0, 0):
            msg = "a nonzero fitting position has no saved validation state"
            raise ValueError(msg)

        return 0, EarlyStoppingState(), None

    if fitting.unit != TrainingUnit.EPOCHS or fitting.kind != "resume-latest":
        msg = "configured fitting resumes only from a resume-latest checkpoint"
        raise ValueError(msg)

    parser_fitting = cast("ParserFittingRecord", fitting)

    if parser_fitting.stop is not None:
        msg = f"configured fitting already stopped with {parser_fitting.stop}"
        raise ValueError(msg)

    if plan.stages:
        expected = plan.stage_at(parser_fitting.completed_epochs - 1)

        if expected is None:
            msg = "a staged checkpoint lies outside the declared stage schedule"
            raise ValueError(msg)

        position_index, stage, _ = expected
        recorded = template.stage

        if recorded is None or (
            recorded.name,
            recorded.active,
            recorded.position,
        ) != (stage.name, tuple(sorted(stage.active.names)), position_index):
            msg = "the checkpoint stage differs from the declared stage schedule"
            raise ValueError(msg)

    elif template.stage is not None:
        msg = "an ordinary parser plan cannot resume a staged checkpoint"
        raise ValueError(msg)

    if parser_fitting.metric != plan.early_stopping.metric:
        msg = (
            f"the checkpoint monitored {parser_fitting.metric!r}, not "
            f"{plan.early_stopping.metric!r}"
        )
        raise ValueError(msg)

    epochs = parser_fitting.completed_epochs
    expected_step = (
        sum(
            staged[2].optimizer_steps
            for epoch in range(epochs)
            if (staged := plan.stage_at(epoch)) is not None
        )
        if plan.stages
        else epochs * plan.steps_per_epoch
    )

    if position.step != expected_step:
        msg = (
            f"completed epoch {epochs} implies optimizer step "
            f"{expected_step}, not {position.step}"
        )
        raise ValueError(msg)

    if not plan.stages and plan.alternation is None:
        expected_position = epochs * plan.items_per_epoch

        if position.data_position != expected_position:
            msg = (
                f"completed epoch {epochs} implies data position "
                f"{expected_position}, not {position.data_position}"
            )
            raise ValueError(msg)

    checkpoint = parser_fitting.best_checkpoint

    return (
        epochs,
        EarlyStoppingState(parser_fitting.best, parser_fitting.failures),
        (None if checkpoint is None else BestCheckpointId(CheckpointId(checkpoint))),
    )


def _evaluate_staged_target(
    prepared: PreparedTraining,
    task: StepTask[object, object],
    step: int,
    /,
) -> EvalResult:
    """Evaluate one targeted task with model modes and autocast restored."""
    reachable = tuple(
        dict.fromkeys(
            module
            for root in prepared.model.modules.values()
            for module in root.modules()
        )
    )
    modes = tuple(module.training for module in reachable)

    try:
        for module in reachable:
            module.eval()

        with torch.no_grad(), prepared.model.policy.autocast():
            return task.evaluate(step)

    finally:
        for module, training in zip(reachable, modes, strict=True):
            module.train(training)


def _fit_staged_task(
    prepared: PreparedTraining,
    task: FittingTask,
    step: int,
    target_steps: int,
    /,
) -> tuple[
    TrainResult,
    AccuracyStoppingDecision | None,
    float | None,
    Mapping[str, float],
]:
    """Fit one stage-task epoch, stopping after the first met target."""
    taken = step
    items = 0
    reported: dict[str, list[float]] = {}
    decision: AccuracyStoppingDecision | None = None
    achieved_accuracy: float | None = None
    validation: Mapping[str, float] = MappingProxyType({})
    targeted = task if isinstance(task, StepTask) and task.stop is not None else None

    for position in range(task.batches):
        fitted = task.fit(position, taken)

        if fitted.result is not None:
            if not fitted.result.loss_finite:
                quantity = f"the staged task {task.name!r} loss"
                raise NumericalFailureError(quantity, None)

            taken += int(fitted.result.stepped)

        items += fitted.items

        for name, value in fitted.metrics.items():
            reported.setdefault(name, []).append(value)

        if targeted is not None and fitted.result is not None and fitted.result.stepped:
            evaluated = _evaluate_staged_target(prepared, targeted, taken)
            validation = evaluated.metrics
            stop = targeted.stop

            if stop is None:
                msg = "a targeted step task lost its accuracy stopping rule"
                raise RuntimeError(msg)

            try:
                achieved_accuracy = validation[stop.metric]

            except KeyError:
                known = sorted(validation)
                msg = (
                    f"held-out task evaluation reported metrics {known}, not target "
                    f"metric {stop.metric!r}"
                )
                raise ValueError(msg) from None

            decision = stop.observe(
                achieved_accuracy,
                target_steps + taken - step,
            )

            if decision.stop:
                break

    metrics = {
        f"task/{task.name}/{name}": fsum(values) / len(values)
        for name, values in reported.items()
    }
    metrics[f"task/{task.name}/steps"] = float(taken - step)
    metrics[f"task/{task.name}/items"] = float(items)
    metrics["optimizer/stepped"] = float(taken - step)

    return (
        TrainResult(taken, items, metrics=metrics),
        decision,
        achieved_accuracy,
        validation,
    )


def _fit_epochs(
    prepared: PreparedTraining,
    plan: ParserFittingPlan,
    store: CheckpointStore[StateTree],
    manifest: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    sinks: Sequence[MetricLog],
    /,
) -> FittingReport:
    """Fit, validate and checkpoint configured epochs from one recovered position.

    Each optimizer step's metrics are logged to every sink at the global step
    count once it closes, and each epoch's validation metrics, prefixed with
    ``validation/``, at the step count the epoch ends on.

    An improving epoch writes a validation-best checkpoint. A resumable
    checkpoint is written after an improving epoch, after the epoch that stops
    the run and at every epoch the plan's checkpoint cadence falls on, so the
    newest resumable checkpoint always names the newest best one. After each
    epoch the store is pruned to the plan's retention.

    Parameters
    ----------
    prepared : PreparedTraining
        Configured parser graph, batches, objectives and optimizer.
    store : CheckpointStore[StateTree]
        Existing checkpoint store bound to the run.
    manifest : CheckpointManifest
        Compatible template, or the latest loaded manifest when resuming.
    streams : RestorableRandomStreams
        Exact initialized or restored streams for this run.
    position : ResumePosition
        Completed optimizer steps and consumed training occurrences.
    sinks : Sequence[MetricLog]
        Where every step's and every epoch's metrics are logged.
    """
    first_epoch, stopping_state, best_checkpoint = _initial_state(
        prepared,
        plan,
        manifest,
        streams,
        position,
    )

    if first_epoch >= plan.epochs:
        msg = (
            f"the run has completed {first_epoch} epochs against a horizon of "
            f"{plan.epochs}"
        )
        raise ValueError(msg)

    completed = position.step
    data_position = position.data_position
    epochs: list[EpochReport] = []
    latest_checkpoint: ResumeCheckpointId | None = None
    stopped_early = False
    stop_reason: Literal[
        "horizon", "early-stopping", "target-met", "target-not-met"
    ] = "horizon"
    achieved_accuracy: float | None = None
    window: list[TrainResult] = []
    batch_window: list[PositionedBatch] = []

    for epoch in range(first_epoch, plan.epochs):
        staged = plan.stage_at(epoch)
        stage_task: FittingTask | None = None

        if staged is not None:
            _, stage, stage_task = staged
            prepared.model.activate(tuple(stage.active.names))

        epoch_batches = plan.epoch(epoch)
        target_decision: AccuracyStoppingDecision | None = None

        alternation = plan.alternation

        if alternation is not None:
            result = alternation.run(completed).result
            expected_step = completed + len(alternation.schedule)

            if result.step != expected_step:
                msg = (
                    f"alternating epoch {epoch} ended at step {result.step}, "
                    f"not its declared horizon {expected_step}"
                )
                raise RuntimeError(msg)

            results = [result]
            completed = result.step
            data_position += result.items
            _log(sinks, result.metrics, completed)

        elif stage_task is not None:
            if staged is None:
                msg = "a staged task has no stage declaration"
                raise RuntimeError(msg)

            stage_position, _, _ = staged
            target_steps = sum(
                earlier[2].optimizer_steps
                for earlier_epoch in range(epoch)
                if (earlier := plan.stage_at(earlier_epoch)) is not None
                and earlier[0] == stage_position
            )
            result, target_decision, achieved_accuracy, target_validation = (
                _fit_staged_task(
                    prepared,
                    stage_task,
                    completed,
                    target_steps,
                )
            )
            results = [result]
            completed = result.step
            data_position += result.items
            _log(sinks, result.metrics, completed)

            if target_validation:
                _log(
                    sinks,
                    {
                        f"validation/{name}": value
                        for name, value in target_validation.items()
                    },
                    completed,
                )

        else:
            results = []

            for batch in epoch_batches.training:
                positioned = _positioned(batch, completed)
                run_step = cast(
                    "Callable[[object], StepOutcome[object, TrainResult]]",
                    plan.step.run,
                )
                result = run_step(Featurized(positioned, completed)).result

                if result.metrics["optimizer/loss_finite"] != 1.0:
                    quantity = f"the parser loss in epoch {epoch}"
                    raise NumericalFailureError(quantity, None)

                window.append(result)
                batch_window.append(positioned)

                if result.step > completed:
                    metrics = _step_metrics(window)
                    metrics["data/batch-order"] = _batch_order(batch_window)
                    _log(sinks, metrics, result.step)
                    window.clear()
                    batch_window.clear()

                completed = result.step
                data_position += result.items
                results.append(result)

        validation: EvalResult | None = None
        decision = None

        if plan.validates(epoch):
            validation = validate_epoch(
                cast("Forward[_ValidationBatch]", plan.forward),
                cast("LossAssembly[_ValidationBatch]", plan.step.assembly),
                cast(
                    "Sequence[_ValidationBatch]",
                    epoch_batches.retained[plan.fit_split],
                ),
                prepared.model.modules,
                prepared.model.policy,
                step=completed,
            )
            _log(
                sinks,
                {
                    f"validation/{name}": value
                    for name, value in validation.metrics.items()
                },
                completed,
            )

            try:
                monitored = validation.metrics[plan.early_stopping.metric]

            except KeyError:
                known = sorted(validation.metrics)
                msg = (
                    f"validation reported metrics {known}, not monitored metric "
                    f"{plan.early_stopping.metric!r}"
                )
                raise ValueError(msg) from None

            decision = plan.early_stopping.observe(stopping_state, monitored)
            stopping_state = decision.state

        report = EpochReport(epoch, tuple(results), validation)

        expected_steps = (
            stage_task.optimizer_steps
            if stage_task is not None
            else plan.steps_per_epoch
        )

        if report.optimizer_steps != expected_steps and not (
            target_decision is not None and target_decision.stop
        ):
            msg = (
                f"epoch {epoch} closed {report.optimizer_steps} optimizer steps, "
                f"not its derived horizon {expected_steps}"
            )
            raise RuntimeError(msg)

        completed_epochs = epoch + 1

        if target_decision is not None and target_decision.stop:
            reason = target_decision.reason

            if reason is None:
                msg = "a stopped accuracy target has no stopping reason"
                raise RuntimeError(msg)

            stop_reason = reason
        snapshot = _snapshot(prepared, streams, manifest)
        metadata = plan.grounding_metadata
        checkpoint_template = manifest.with_(
            grounding=(
                None
                if metadata is None
                else grounding_record(metadata, plan.grounding_telemetry)
            )
        )

        if staged is not None:
            stage_position, stage, _ = staged
            checkpoint_template = staged_manifest(
                checkpoint_template,
                name=stage.name,
                active=stage.active.names,
                position=stage_position,
            )

        if decision is not None and decision.improved:
            best_manifest = _parser_manifest(
                checkpoint_template,
                snapshot,
                stopping_state,
                kind="validation-best",
                completed_epochs=completed_epochs,
                step=completed,
                data_position=data_position,
                metric=plan.early_stopping.metric,
                best_checkpoint=None,
            )
            best_checkpoint = BestCheckpointId(
                store.save(best_manifest, snapshot.blocks)
            )

        stage_ends = staged is not None and (
            epoch + 1 == plan.epochs or plan.stage_at(epoch + 1) != staged
        )

        if (
            (decision is not None and (decision.improved or decision.stop))
            or (target_decision is not None and target_decision.stop)
            or stage_ends
            or plan.checkpoints.due(completed_epochs, plan.epochs)
        ):
            latest_manifest = _parser_manifest(
                checkpoint_template,
                snapshot,
                stopping_state,
                kind="resume-latest",
                completed_epochs=completed_epochs,
                step=completed,
                data_position=data_position,
                metric=plan.early_stopping.metric,
                best_checkpoint=best_checkpoint,
                stop=(
                    stop_reason
                    if stop_reason in {"target-met", "target-not-met"}
                    else None
                ),
                achieved_accuracy=(
                    achieved_accuracy
                    if stop_reason in {"target-met", "target-not-met"}
                    else None
                ),
            )
            latest_checkpoint = ResumeCheckpointId(
                store.save(latest_manifest, snapshot.blocks)
            )

        plan.checkpoints.prune(store)
        epochs.append(report)

        if target_decision is not None and target_decision.stop:
            stopped_early = (
                stop_reason == "target-met" and completed_epochs < plan.epochs
            )
            break

        if decision is not None and decision.stop:
            stopped_early = completed_epochs < plan.epochs
            stop_reason = "early-stopping" if stopped_early else "horizon"
            break

    if latest_checkpoint is None:
        msg = "configured fitting completed no epoch checkpoint"
        raise RuntimeError(msg)

    return FittingReport(
        tuple(epochs),
        (),
        completed,
        data_position,
        unit=TrainingUnit.EPOCHS,
        constraint_selection=plan.constraint_selection,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        achieved_accuracy=(
            achieved_accuracy
            if stop_reason in {"target-met", "target-not-met"}
            else None
        ),
    )


def _step_manifest(
    template: CheckpointManifest,
    snapshot: _CheckpointSnapshot,
    /,
    *,
    completed_steps: int,
    data_position: int,
    stop: Literal["target-met", "target-not-met"] | None = None,
    achieved_accuracy: float | None = None,
) -> CheckpointManifest:
    """Return a resume-latest manifest for a completed optimizer step."""
    components = {
        role: record.with_(
            state=_reference(record.state, snapshot.blocks, block_key=role)
        )
        for role, record in template.components.items()
    }

    return template.with_(
        step=completed_steps,
        data_position=data_position,
        components=components,
        rng=RngRecord(streams=dict(snapshot.rng)),
        optimizer=(
            None
            if template.optimizer is None
            else _reference(template.optimizer, snapshot.blocks)
        ),
        scaler=(
            None
            if template.scaler is None
            else _reference(template.scaler, snapshot.blocks)
        ),
        scheduler=(
            None
            if template.scheduler is None
            else _reference(template.scheduler, snapshot.blocks)
        ),
        fitting=StepFittingRecord(
            completed_steps=completed_steps,
            stop=stop,
            achieved_accuracy=achieved_accuracy,
        ),
    )


def _step_start(
    prepared: PreparedTraining,
    plan: StepFittingPlan,
    template: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    /,
) -> int:
    """Validate and return the completed optimizer-step position."""
    expected_roles = frozenset(prepared.model.modules)

    if template.roles != expected_roles:
        msg = (
            f"the manifest names component roles {sorted(template.roles)}, not the "
            f"configured roles {sorted(expected_roles)}"
        )
        raise ValueError(msg)

    _, optimization = prepared.state()
    expected_presence = (
        True,
        optimization.scaler is not None,
        optimization.scheduler is not None,
    )
    found_presence = (
        template.optimizer is not None,
        template.scaler is not None,
        template.scheduler is not None,
    )

    if found_presence != expected_presence:
        msg = (
            "the manifest run-state presence "
            f"{found_presence!r} differs from the configured presence "
            f"{expected_presence!r} for optimizer, scaler and scheduler"
        )
        raise ValueError(msg)

    if frozenset(template.rng.streams) != frozenset(streams.snapshot()):
        msg = "the manifest random streams differ from the supplied random streams"
        raise ValueError(msg)

    if (template.step, template.data_position) != (
        position.step,
        position.data_position,
    ):
        msg = (
            f"the manifest position {(template.step, template.data_position)!r} "
            f"differs from {(position.step, position.data_position)!r}"
        )
        raise ValueError(msg)

    expected_position = position.step * plan.batch_size

    if position.data_position != expected_position:
        msg = (
            f"completed step {position.step} implies item position "
            f"{expected_position}, not {position.data_position}"
        )
        raise ValueError(msg)

    fitting = template.fitting

    if fitting is None:
        if position != ResumePosition(0, 0):
            msg = "a nonzero step position has no saved fitting progress"
            raise ValueError(msg)

        return 0

    if fitting.unit != TrainingUnit.STEPS or fitting.kind != "resume-latest":
        msg = "step fitting resumes only from a step resume-latest checkpoint"
        raise ValueError(msg)

    step_fitting = cast("StepFittingRecord", fitting)

    if step_fitting.completed_steps != position.step:
        msg = (
            f"the checkpoint records {step_fitting.completed_steps} completed "
            f"steps at optimizer position {position.step}"
        )
        raise ValueError(msg)

    return position.step


def _validate_step_target(
    prepared: PreparedTraining,
    plan: StepFittingPlan,
    step: int,
    /,
) -> dict[str, float]:
    """Evaluate the fixed held-out generated batches for accuracy stopping."""
    seed = plan.validation_seed

    if seed is None:
        msg = "a target-stopped step plan has no held-out sampler seed"
        raise RuntimeError(msg)

    reachable: list[torch.nn.Module] = []
    seen: set[int] = set()

    for root in prepared.model.modules.values():
        for module in root.modules():
            if id(module) not in seen:
                reachable.append(module)
                seen.add(id(module))

    modes = tuple(module.training for module in reachable)
    metrics: dict[str, float] = {}

    try:
        for module in reachable:
            module.eval()

        with torch.no_grad(), prepared.model.policy.autocast():
            for position, trainer in enumerate(plan.trainers):
                draw = Draw(
                    plan.batch_seed(seed, position),
                    plan.batch_size,
                    step=step,
                    horizon=plan.steps,
                )
                evaluated = trainer.evaluate(draw)
                repeated = sorted(set(metrics).intersection(evaluated.metrics))

                if repeated:
                    msg = f"held-out steps report repeated metrics {repeated}"
                    raise ValueError(msg)

                metrics.update(evaluated.metrics)

    finally:
        for module, training in zip(reachable, modes, strict=True):
            module.train(training)

    return metrics


def _fit_steps(
    prepared: PreparedTraining,
    plan: StepFittingPlan,
    store: CheckpointStore[StateTree],
    manifest: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    sinks: Sequence[MetricLog],
    /,
) -> FittingReport:
    """Fit and checkpoint generated-batch steps, logging each step's metrics.

    A resumable checkpoint is written at every step the plan's checkpoint
    cadence falls on, at the final step and at the step an accuracy target
    stops, and the store is pruned to the plan's retention after each one.
    """
    completed = _step_start(prepared, plan, manifest, streams, position)

    if completed >= plan.steps:
        msg = (
            f"the run has completed {completed} steps against a horizon of {plan.steps}"
        )
        raise ValueError(msg)

    data_position = position.data_position
    results: list[TrainResult] = []
    latest_checkpoint: ResumeCheckpointId | None = None
    stop_reason: Literal["horizon", "target-met", "target-not-met"] = "horizon"
    achieved_accuracy: float | None = None

    while completed < plan.steps:
        seed = plan.batch_seed(plan.stream_seed, completed)
        result = (
            plan.trainer_at(completed)
            .run(Draw(seed, plan.batch_size, step=completed, horizon=plan.steps))
            .result
        )

        if result.metrics["optimizer/loss_finite"] != 1.0:
            quantity = f"the generated-batch loss at step {completed}"
            raise NumericalFailureError(quantity, None)

        if result.step != completed + 1:
            msg = f"step {completed} did not close one optimizer step"
            raise RuntimeError(msg)

        if result.items != plan.batch_size:
            msg = (
                f"step {completed} consumed {result.items} items, not {plan.batch_size}"
            )
            raise RuntimeError(msg)

        _log(sinks, result.metrics, result.step)
        completed = result.step
        data_position += result.items
        results.append(result)

        target_decision = None

        if plan.stop is not None:
            validation = _validate_step_target(prepared, plan, completed)
            _log(
                sinks,
                {f"validation/{name}": value for name, value in validation.items()},
                completed,
            )

            try:
                achieved_accuracy = validation[plan.stop.metric]
            except KeyError:
                known = sorted(validation)
                msg = (
                    f"held-out step evaluation reported metrics {known}, not "
                    f"target metric {plan.stop.metric!r}"
                )
                raise ValueError(msg) from None

            target_decision = plan.stop.observe(achieved_accuracy, completed)

            if target_decision.reason is not None:
                stop_reason = target_decision.reason

        if plan.checkpoints.due(completed, plan.steps) or (
            target_decision is not None and target_decision.stop
        ):
            snapshot = _snapshot(prepared, streams, manifest)
            latest_checkpoint = ResumeCheckpointId(
                store.save(
                    _step_manifest(
                        manifest,
                        snapshot,
                        completed_steps=completed,
                        data_position=data_position,
                        stop=(
                            None if target_decision is None else target_decision.reason
                        ),
                        achieved_accuracy=achieved_accuracy,
                    ),
                    snapshot.blocks,
                )
            )
            plan.checkpoints.prune(store)

        if target_decision is not None and target_decision.stop:
            break

    if latest_checkpoint is None:
        msg = "configured step fitting completed no checkpoint"
        raise RuntimeError(msg)

    return FittingReport(
        (),
        tuple(results),
        completed,
        data_position,
        unit=TrainingUnit.STEPS,
        constraint_selection=None,
        best_checkpoint=None,
        latest_checkpoint=latest_checkpoint,
        stopped_early=stop_reason == "target-met" and completed < plan.steps,
        stop_reason=stop_reason,
        achieved_accuracy=achieved_accuracy,
    )


def fit_training(
    prepared: PreparedTraining,
    store: CheckpointStore[StateTree],
    manifest: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    /,
    *,
    sinks: Sequence[MetricLog],
) -> FittingReport:
    """Fit one plan through the loop of its training unit.

    Parameters
    ----------
    prepared : PreparedTraining
        The prepared run whose plan is fitted.
    store : CheckpointStore[StateTree]
        The run's checkpoint store.
    manifest : CheckpointManifest
        The template of a fresh run, or the manifest of the checkpoint it
        resumes from.
    streams : RestorableRandomStreams
        The run's applied or restored random streams.
    position : ResumePosition
        The completed optimizer steps and consumed occurrences recovered.
    sinks : Sequence[MetricLog]
        Where the metrics of every optimizer step, and the validation metrics
        of every epoch or held-out target evaluation, are logged at the global
        step count.

    Returns
    -------
    FittingReport
        The work completed, its counters and its checkpoint selections.
    """
    plan = prepared.plan

    match plan:
        case ParserFittingPlan():
            return _fit_epochs(
                prepared, plan, store, manifest, streams, position, sinks
            )
        case StepFittingPlan():
            return _fit_steps(prepared, plan, store, manifest, streams, position, sinks)


def fitting_remains(
    prepared: PreparedTraining,
    manifest: CheckpointManifest,
    streams: RestorableRandomStreams,
    position: ResumePosition,
    /,
) -> bool:
    """Return whether a run resumed from a checkpoint has fitting left to do.

    The checkpoint is validated against the prepared run, its streams and its
    position exactly as ``fit_training`` validates it before fitting. An
    epoch-counted run is finished once it has completed its epoch horizon, its
    patience is spent or an accuracy target has stopped it, and a step-counted
    run once it has completed its step horizon or a target has stopped it. A run
    resumed from a checkpoint written at that point is complete as it stands,
    so it is not fitted again.

    Parameters
    ----------
    prepared : PreparedTraining
        The prepared run whose plan sets the horizon.
    manifest : CheckpointManifest
        The template of a fresh run, or the manifest of the checkpoint it
        resumes from.
    streams : RestorableRandomStreams
        The run's applied or restored random streams.
    position : ResumePosition
        The completed optimizer steps and consumed occurrences recovered.

    Raises
    ------
    ValueError
        If the checkpoint disagrees with the prepared run, its streams, its
        position or its monitored metric.
    """
    fitting = manifest.fitting
    stopped = fitting is not None and fitting.stop is not None
    plan = prepared.plan

    match plan:
        case ParserFittingPlan():
            if stopped:
                return False

            epochs, stopping, _ = _initial_state(
                prepared, plan, manifest, streams, position
            )

            return (
                epochs < plan.epochs
                and stopping.failures < plan.early_stopping.patience
            )
        case StepFittingPlan():
            completed = _step_start(prepared, plan, manifest, streams, position)

            return completed < plan.steps and not stopped
