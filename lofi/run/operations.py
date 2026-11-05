"""The concrete training operations that prepare, fit, score and probe a point.

``TRAINING_OPERATIONS`` is what ``lofi run`` and the sweep launcher hand a point
executor. A caller composing its own executor, or a script that runs one
document's grammar comparison, uses the same operations, so its points train
exactly as the command line trains them. ``CHECKPOINT_RECORDS`` names the record
types these operations write a point's checkpoint manifests with.
"""

import gc
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from typing import TYPE_CHECKING, cast, final

import torch

from lofi import train
from lofi.nn.policy import DeviceKind
from lofi.probe._probing import PROBE_RECORDS_NAME, write_probe_records
from lofi.run._boundary.manifest import (
    _CheckpointRecordTypes,
    _training_checkpoint_manifest,
)
from lofi.run._executor import CHECKPOINTS_NAME, PreparationError
from lofi.run._ports import ReportingComponent
from lofi.run._values import RunPhase
from lofi.run.held_out import evaluate_held_out
from lofi.run.hub import newest_checkpoint
from lofi.run.probing import (
    PROBES_NAME,
    ProbeExecution,
    report_probe_rows,
    write_probe_rows,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import ReportValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run._ports import HeldOutScoring, MetricSink, PointRecorder
    from lofi.run._training import TrainingBlueprint

__all__ = ["CHECKPOINT_RECORDS", "TRAINING_OPERATIONS"]


def _state_digest(state: train.StateTree, /) -> str:
    """Return the digest of one weights-only state serialization."""
    serialized = BytesIO()
    torch.save(state, serialized)

    return sha256(serialized.getvalue()).hexdigest()


CHECKPOINT_RECORDS = _CheckpointRecordTypes(
    checkpoint_manifest=train.CheckpointManifest,
    component_record=train.ComponentRecord,
    precision_record=train.PrecisionRecord,
    rng_record=train.RngRecord,
    state_reference=train.StateReference,
    state_digest=_state_digest,
    grounding_record=train.grounding_record,
)
"""The checkpoint record types training operations write a point's manifest with."""


def _latest_resume(
    store: train.DirectoryStore,
    template: train.CheckpointManifest,
    /,
) -> train.Checkpoint[train.StateTree] | None:
    """Return the newest checkpoint explicitly marked resume-latest."""
    newest = newest_checkpoint(store, "resume-latest")

    if newest is None:
        return None

    handle, _ = newest

    return store.load(handle, template)


def _log_fitting(sinks: Sequence[MetricSink], report: train.FittingReport, /) -> None:
    """Log what one fitting invocation completed, at the step it ended on."""
    summary = {
        "fitting/epochs": float(len(report.epochs)),
        "fitting/optimizer_steps": float(report.optimizer_steps),
        "fitting/data_position": float(report.data_position),
        "fitting/stopped_early": float(report.stopped_early),
    }

    if report.achieved_accuracy is not None:
        summary["fitting/achieved_accuracy"] = report.achieved_accuracy

    for sink in sinks:
        sink.log_scalars(summary, report.optimizer_steps)


def _release_accelerator_memory(policy: DevicePolicy, /) -> None:
    """Collect unreachable training state and empty the run device's cache."""
    gc.collect()

    match policy.kind:
        case DeviceKind.CUDA:
            torch.cuda.empty_cache()

        case DeviceKind.MPS:
            torch.mps.empty_cache()

        case DeviceKind.XPU:
            torch.xpu.empty_cache()

        case DeviceKind.CPU | DeviceKind.META:
            pass


@final
class _ProductionTrainingOperations:
    """Concrete training operations composed only by a process entry point."""

    __slots__ = ()

    def execute(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: train.RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        """Prepare, recover and fit one training point, then score it.

        Fitting logs every optimizer step's metrics and every epoch's
        validation metrics to the sinks, and what it completed once it ends.
        A run recovered from a checkpoint written once its fitting finished is
        complete as it stands, so it is scored without being fitted again.

        The recorder is told what preparation built, the wall time of each
        phase, what fitting completed, the report of every component that
        reports on itself, and, once the point closes, the peak memory the run
        device held from the end of preparation on.

        Each training step is measured against the
        ``activation_memory_bytes`` bound the run's device policy declares, on
        a device whose allocator keeps a peak count. An accelerator that runs
        out of memory is reported as a ``MemoryError``, and a step above the
        bound as an ``ActivationMemoryExceededError``, which is one; the
        executor treats either as exhausted memory under the experiment's
        failure policy. The failed attempt's model, optimizer and tensors are
        released first: the error, whose traceback holds them, is dropped,
        garbage is collected and the accelerator's cached allocations are
        emptied, so the next attempt starts from the memory the process held
        before this one.

        Raises
        ------
        ActivationMemoryExceededError
            If a training step holds more memory than its device policy's
            bound allows.
        MemoryError
            If the accelerator runs out of memory.
        PreparationError
            If preparation refuses the configuration.
        """
        exhausted: str | None = None
        bounded: tuple[str, int, int] | None = None

        try:
            self._train(
                blueprint,
                directory,
                streams,
                policy,
                scoring=scoring,
                sinks=sinks,
                recorder=recorder,
            )

        except torch.OutOfMemoryError as error:
            exhausted = str(error)

        except train.ActivationMemoryExceededError as error:
            bounded = (error.device, error.peak_bytes, error.max_bytes)

        if exhausted is not None:
            _release_accelerator_memory(policy)
            raise MemoryError(exhausted)

        if bounded is not None:
            _release_accelerator_memory(policy)
            device, peak_bytes, max_bytes = bounded
            raise train.ActivationMemoryExceededError(
                device, peak_bytes=peak_bytes, max_bytes=max_bytes
            )

    def _train(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: train.RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        """Prepare, recover, fit what remains, score and probe one training point.

        The declared probes are composed against the prepared model before
        anything is fitted, so a probe naming a role or a site the model does not
        publish refuses the configuration. Once fitting and scoring end, the
        probes' rows are written to the run directory and sent to the sinks.

        Raises
        ------
        PreparationError
            If preparation refuses the configuration, or a declared probe cannot
            be composed against the prepared model.
        """
        started = perf_counter()

        try:
            prepared = train.prepare_training(blueprint, streams, policy)
            probes = ProbeExecution(blueprint.run_spec, prepared)

        except ValueError as error:
            raise PreparationError(str(error)) from error

        memory = train.StepMemory(
            policy, blueprint.run_spec.device_policy.activation_memory_bytes
        )
        memory.reset()

        checkpoints = train.DirectoryStore(directory / CHECKPOINTS_NAME)
        template = _training_checkpoint_manifest(
            blueprint,
            prepared,
            streams,
            CHECKPOINT_RECORDS,
            step=0,
            data_position=0,
        )
        recorder.timed(RunPhase.PREPARATION, perf_counter() - started)
        recorder.prepared(blueprint, prepared)

        started = perf_counter()
        selected = _latest_resume(checkpoints, template)

        if selected is None:
            manifest = template
            position = train.ResumePosition(0, 0)
        else:
            manifest = selected.manifest
            position = train.resume_from_checkpoint(
                selected,
                template,
                prepared.model.prepare_restore,
                prepared.optimization.prepare_restore,
                streams,
            )

        grounding = manifest.grounding

        recorder.timed(RunPhase.RECOVERY, perf_counter() - started)

        if train.fitting_remains(prepared, manifest, streams, position):
            started = perf_counter()
            with prepared.optimization.metered(memory):
                report = train.fit_training(
                    prepared,
                    checkpoints,
                    manifest,
                    streams,
                    position,
                    sinks=sinks,
                )

            _log_fitting(sinks, report)
            recorder.fitted(report, position)
            recorder.timed(RunPhase.FITTING, perf_counter() - started)
            grounding = checkpoints.manifest(report.latest_checkpoint).grounding

        if grounding is not None:
            recorder.contribute(
                "grounding",
                cast(
                    "Mapping[str, ReportValue]",
                    grounding.model_dump(exclude_none=True),
                ),
            )

        if scoring is not None:
            started = perf_counter()
            evaluate_held_out(
                prepared,
                blueprint.run_spec,
                freeze_id=scoring.freeze_id,
                seed=scoring.seed,
                sink=scoring.sink,
            )
            recorder.timed(RunPhase.SCORING, perf_counter() - started)

        for role, module in prepared.model.modules.items():
            if isinstance(module, ReportingComponent):
                recorder.contribute(role, module.report())

        recorder.closed(peak_memory_bytes=memory.high_water())

        if probes.roles:
            rows, records = probes.execute()
            write_probe_rows(directory / PROBES_NAME, rows)
            report_probe_rows(sinks, rows)

            if records:
                write_probe_records(directory / PROBE_RECORDS_NAME, records)


TRAINING_OPERATIONS = _ProductionTrainingOperations()
"""The training operations ``lofi run`` and the sweep launcher execute points with."""
