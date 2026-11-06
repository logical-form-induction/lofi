"""The ``lofi`` command line.

Every command composes its run specification through the process-entry
codec from a configuration document, an optional profile, extra overlay
documents and ``key=value`` overrides, in that ascending precedence, so
nothing the command line accepts is dropped before validation.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, get_args

from lofi.run._array import SubmissionError
from lofi.run._boundary.active import awaited_batch
from lofi.run._boundary.experiment import FailurePolicySpec
from lofi.run._boundary.site import RunStatus
from lofi.run._boundary.sweep import SweepPoint
from lofi.run._compose import Selection, resolved_document, trace
from lofi.run._configs import packaged_groups, packaged_options
from lofi.run._environment import (
    SourceRevisionError,
    UnfrozenRevisionError,
    execution_environment,
    package_checkout,
    require_frozen_revision,
)
from lofi.run._executor import PointExecutionError, PointExecutor
from lofi.run._experiment import (
    ComputeEstimate,
    ExecutionContext,
    ExperimentError,
    Unpriced,
    amend_experiment,
    amendment_ledger,
    read_experiment,
)
from lofi.run._launch import SWEEPS_NAME, read_points, task_points
from lofi.run._manifest import PRICE_NAME, RESOLVED_NAME, write_price
from lofi.run._ports import ReleaseAuthorization, ReleasePublication
from lofi.run._pricing import price_experiment
from lofi.run._seed import ExperimentSeed
from lofi.run._serial import SerialLauncher
from lofi.run._store import ExperimentStore, ReseededRunError, RunStore
from lofi.run._sweep import UnexpectedPointCountError
from lofi.run._values import HOUR_DECIMALS
from lofi.spec import (
    ConfigDirectoryError,
    ConfigDocumentError,
    ConfigError,
    MalformedOverrideError,
    RunContext,
    RunSpecCodec,
    UndeclaredLeafError,
    config_search_path,
    run_context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

    from lofi._types import JsonValue
    from lofi.run._experiment import Experiment
    from lofi.run._ports import (
        ActiveLearningRun,
        BeadPrediction,
        ExperimentReporting,
        FreezeMinting,
        ModelReleasing,
        PointExport,
        PointPublication,
        ResultIngestion,
        SeedBudgeting,
        TrainingOperations,
    )
    from lofi.run._pricing import StepTimer
    from lofi.run._serial import ExecutePoint
    from lofi.run._values import ExperimentPrice
    from lofi.run.launching import ExperimentLaunching
    from lofi.run.probing import PointProbe
    from lofi.spec import LeafOrigin, Provenance

__all__ = ["build_parser", "main", "render_annotated"]

_REFUSALS: Final = (
    ExperimentError,
    ConfigDirectoryError,
    ConfigDocumentError,
    ConfigError,
    MalformedOverrideError,
    NotADirectoryError,
    ReseededRunError,
    UndeclaredLeafError,
    UnexpectedPointCountError,
)
"""The errors that refuse a command's input, reported with exit status 2."""


class _Context(NamedTuple):
    """The collaborators every command handler runs against."""

    training: TrainingOperations | None
    launching: ExperimentLaunching | None
    freeze: FreezeMinting | None
    seed_budget: SeedBudgeting | None
    release: ModelReleasing | None
    timer: StepTimer | None
    export: PointExport | None
    bead_predict: BeadPrediction | None
    probe: PointProbe | None
    publish: PointPublication | None
    ingest: ResultIngestion | None
    reports: ExperimentReporting | None
    active_learn: ActiveLearningRun | None
    environ: Mapping[str, str]
    stdout: TextIO
    stderr: TextIO
    cwd: Path


def _selection(namespace: argparse.Namespace, cwd: Path) -> Selection:
    """Return the selection a parsed command line makes."""
    return Selection(
        config=namespace.config,
        profile=namespace.profile,
        overrides=tuple(namespace.overrides),
        overlays=tuple(namespace.overlays),
        search_path=config_search_path(cwd),
    )


def _annotation(origin: LeafOrigin) -> str:
    """Return the comment text that names where one leaf came from."""
    text = str(origin.source.value)

    if origin.layer:
        text = f"{text} {origin.layer}"

    if origin.expression is not None:
        text = f"{text}, from {origin.expression}"

    return text


def _render(
    lines: list[str],
    mapping: Mapping[str, JsonValue],
    provenance: Provenance,
    *,
    prefix: str,
    depth: int,
) -> None:
    """Append the annotated lines of one mapping."""
    indent = "  " * depth

    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) and value:
            lines.append(f"{indent}{key}:")
            _render(lines, value, provenance, prefix=path, depth=depth + 1)
            continue

        if isinstance(value, list) and value:
            for index, element in enumerate(value):
                item = f"{key}[{index}]"
                item_path = f"{path}[{index}]"
                text = json.dumps(element, allow_nan=False)
                lines.append(
                    f"{indent}{item}: {text}  # "
                    f"{_annotation(provenance.origin_of(item_path))}"
                )
            continue

        text = json.dumps(value, allow_nan=False)
        lines.append(
            f"{indent}{key}: {text}  # {_annotation(provenance.origin_of(path))}"
        )


def render_annotated(
    document: Mapping[str, JsonValue], provenance: Provenance, /
) -> str:
    """Return a resolved document with each leaf's source beside it.

    Parameters
    ----------
    document : Mapping[str, JsonValue]
        The resolved document, nested as its specification dumps it.
    provenance : Provenance
        The source of each leaf, keyed by dotted path.

    Returns
    -------
    str
        One line per leaf, nested by indentation, with every list element
        addressed by its bracketed index and every leaf followed by a comment
        naming its source, its layer and the expression it was resolved from.

    Raises
    ------
    UnknownSettingError
        If the provenance lacks a leaf of the document.
    """
    lines: list[str] = []
    _render(lines, document, provenance, prefix="", depth=0)

    return "\n".join(lines) + "\n"


def _root(namespace: argparse.Namespace, context: _Context) -> Path:
    """Return the store root a command runs under."""
    root: Path | None = namespace.root

    return context.cwd if root is None else root


def _point(namespace: argparse.Namespace, context: _Context) -> SweepPoint:
    """Return the one point a run or resume command composes."""
    selection = _selection(namespace, context.cwd)

    with run_context(RunContext(context.cwd, namespace.seed)):
        composition = trace(selection)

    return SweepPoint(
        resolved_document(composition.spec),
        overrides=selection.overrides,
        provenance=composition.provenance,
        seed=namespace.seed,
    )


def _failure_policy(namespace: argparse.Namespace, /) -> FailurePolicySpec:
    """Return the failure policy a development point declares on its command line.

    A field the command line leaves out takes the default the failure policy
    declares.
    """
    declared = {
        field: getattr(namespace, field)
        for field in FailurePolicySpec.__field_specs__
        if getattr(namespace, field) is not None
    }

    return FailurePolicySpec(**declared)


def _training(context: _Context, /) -> TrainingOperations:
    """Return the training operations a training command executes through.

    Raises
    ------
    PointExecutionError
        If the command was given none.
    """
    if context.training is None:
        msg = "a training command was given no training operations"
        raise PointExecutionError(msg)

    return context.training


def _develop(
    store: RunStore,
    point: SweepPoint,
    namespace: argparse.Namespace,
    context: _Context,
) -> int:
    """Train one development point under the failure policy its options declare.

    Raises
    ------
    PointExecutionError
        If no training operations are given.
    """
    failure_policy = _failure_policy(namespace)
    execute = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=_training(context),
        infrastructure_attempts=failure_policy.infrastructure_attempts,
        failure_policy=failure_policy,
    )

    return _submit(store, (point,), execute, context)


def _submit(
    store: RunStore,
    points: Sequence[SweepPoint],
    execute: ExecutePoint,
    context: _Context,
) -> int:
    """Execute points in this process and report each one's status."""
    launcher = SerialLauncher(store, execute=execute)
    handles = {handle.point_id: handle for handle in launcher.submit(points)}
    failed = 0

    for point in points:
        handle = handles.get(point.point_id)
        status = RunStatus.COMPLETED if handle is None else launcher.status(handle)
        failed += status is not RunStatus.COMPLETED
        context.stdout.write(
            f"{point.point_id} {status.value} {store.directory(point)}\n"
        )

    return int(failed > 0)


def _run(namespace: argparse.Namespace, context: _Context) -> int:
    """Train one composed point, or report it cached.

    Raises
    ------
    ReseededRunError
        If the point's directory records a run seeded under another
        experiment root than the one given.
    """
    store = RunStore(
        _root(namespace, context), freeze_id=None, seed=namespace.experiment_seed
    )

    return _develop(store, _point(namespace, context), namespace, context)


def _resume(namespace: argparse.Namespace, context: _Context) -> int:
    """Train one composed point again, requiring a run directory to resume.

    Raises
    ------
    ReseededRunError
        If the recorded run, completed or not, was seeded under a derivation
        other than the one the given experiment root derives for the point.
    """
    store = RunStore(
        _root(namespace, context), freeze_id=None, seed=namespace.experiment_seed
    )
    point = _point(namespace, context)

    if store.manifest(point) is None:
        context.stderr.write(
            f"lofi resume: nothing to resume, the point {point.point_id!r} has no "
            f"run directory under {store.root}\n"
        )

        return 1

    return _develop(store, point, namespace, context)


def _task(namespace: argparse.Namespace, context: _Context) -> int:
    """Execute the points one scheduler task or remote session covers.

    A task runs the points of one experiment, named by its document relative
    to the executing checkout, through the executor the experiment's
    configuration kind builds. The experiment declares the freeze, the root
    seed and the failure policy, so the task runs only when the run package
    executes from a checkout of the tree that freeze tags, and every point is
    seeded from the declared root. Its points are executed in the store the
    freeze addresses under the store root, so a point completed under another
    freeze is executed again in a directory of this freeze's, and the other
    freeze's directory is left as it was.

    Raises
    ------
    ExperimentError
        If the experiment document is refused.
    UnfrozenRevisionError
        If the executing checkout is not the tree the experiment's freeze tags.
    PointExecutionError
        If the points file lists points of another experiment or freeze.
    """
    checkout = package_checkout()
    experiment = read_experiment(
        checkout / namespace.experiment, overrides=tuple(namespace.overrides)
    )
    require_frozen_revision(checkout, experiment.freeze_id)
    store = ExperimentStore(_root(namespace, context), experiment)
    listed = read_points(namespace.points)

    if listed.identity != store.identity:
        msg = (
            f"the points file lists points of {listed.identity!r}, and the "
            f"task runs the experiment {store.identity!r}"
        )
        raise PointExecutionError(msg)

    points = task_points(listed.points, namespace.index, namespace.chunk_size)
    execute = experiment.config.executor(
        ExecutionContext(
            store, execution_environment(), experiment.failure, _training(context)
        ),
        points,
    )

    return _submit(store.runs, points, execute, context)


def _launching(context: _Context, command: str, /) -> ExperimentLaunching | None:
    """Return the launching collaborator, reporting its absence."""
    if context.launching is None:
        context.stderr.write(f"lofi {command}: the command was given no launching\n")

    return context.launching


def _launch(namespace: argparse.Namespace, context: _Context) -> int:
    """Launch one experiment, reporting a refused input or a failed launch.

    A launch whose input is refused, such as an experiment its freeze does not
    record or a launcher its configuration kind cannot run on, exits with
    status two; a launch that fails, such as one over its site's quota or
    without a price record, exits with status one.
    """
    try:
        return _launched(namespace, context)

    except ExperimentError:
        raise

    except RuntimeError as failed:
        context.stderr.write(f"lofi launch: {failed}\n")

        return 1

    except ValueError as refused:
        context.stderr.write(f"lofi launch: {refused}\n")

        return 2


def _launched(namespace: argparse.Namespace, context: _Context) -> int:
    """Launch one experiment from its freeze, or as a pilot, and write its record.

    A pilot runs every point in this process outside any freeze and submits
    nothing to a site, so it takes no dry run and no companion experiment.
    """
    launching = _launching(context, "launch")

    if launching is None:
        return 1

    overrides = tuple(namespace.overrides)

    if namespace.pilot:
        named = [
            option
            for option, given in (
                ("--dry-run", namespace.dry_run),
                ("--with", namespace.companions),
            )
            if given
        ]

        if named:
            context.stderr.write(
                f"lofi launch: a pilot runs in this process and submits nothing, "
                f"so it takes no {', '.join(named)}\n"
            )

            return 2

        return launching.pilot(
            namespace.experiment,
            root=namespace.root,
            overrides=overrides,
            training=_training(context),
            out=namespace.out,
            stdout=context.stdout,
        )

    return launching.launch(
        namespace.experiment,
        freeze=namespace.freeze,
        root=namespace.root,
        overrides=overrides,
        dry_run=namespace.dry_run,
        companions=tuple(namespace.companions),
        training=_training(context),
        environ=context.environ,
        out=namespace.out,
        stdout=context.stdout,
    )


def _status(namespace: argparse.Namespace, context: _Context) -> int:
    """Poll an experiment's store, rewriting its results directory each time."""
    launching = _launching(context, "status")

    if launching is None:
        return 1

    try:
        return launching.poll(
            namespace.experiment,
            root=namespace.root,
            overrides=tuple(namespace.overrides),
            interval=namespace.interval,
            polls=namespace.polls,
            environ=context.environ,
            stdout=context.stdout,
        )

    except RuntimeError as failed:
        context.stderr.write(f"lofi status: {failed}\n")

        return 1


def _freeze(namespace: argparse.Namespace, context: _Context) -> int:
    """Mint one freeze and echo the record it wrote."""
    if context.freeze is None:
        context.stderr.write("lofi freeze: the command was given no freeze minting\n")

        return 1

    try:
        record = context.freeze(
            _root(namespace, context),
            number=namespace.number,
            experiments=tuple(namespace.experiments),
            readiness=namespace.readiness,
        )

    except RuntimeError as refused:
        context.stderr.write(f"lofi freeze: {refused}\n")

        return 1

    context.stdout.write(f"{json.dumps(record, sort_keys=True)}\n")

    return 0


def _seed_budget(namespace: argparse.Namespace, context: _Context) -> int:
    """Price an experiment's seed count from a pilot's rows and echo the record."""
    if context.seed_budget is None:
        context.stderr.write(
            "lofi seed-budget: the command was given no seed budgeting\n"
        )

        return 1

    root = _root(namespace, context)

    try:
        written = context.seed_budget(
            read_experiment(root / namespace.experiment), root / namespace.pilot
        )

    except ExperimentError:
        raise

    except (OSError, ValueError) as refused:
        context.stderr.write(f"lofi seed-budget: {refused}\n")

        return 2

    record = json.loads(written.read_text(encoding="utf-8"))
    context.stdout.write(f"{json.dumps(record, indent=2)}\n")

    return 0


def _publication(namespace: argparse.Namespace, /) -> ReleasePublication | None:
    """Return the authorized publication a release names, or refuse a partial one.

    Raises
    ------
    ValueError
        If ``--publish`` is given without the evidence, the reviewer or the
        reviewed license.
    """
    if not namespace.publish:
        return None

    missing = [
        option
        for option, given in (
            ("--evidence", namespace.evidence),
            ("--reviewer", namespace.reviewer),
            ("--authorized-license", namespace.authorized_license),
        )
        if given is None
    ]

    if missing:
        msg = f"{', '.join(missing)} is required with --publish"
        raise ValueError(msg)

    return ReleasePublication(
        namespace.evidence,
        ReleaseAuthorization(
            namespace.reviewer,
            namespace.authorized_license,
            namespace.confirm_corpus_redistribution,
            namespace.confirm_artifact_redistribution,
        ),
        namespace.token_environment,
    )


def _release(namespace: argparse.Namespace, context: _Context) -> int:
    """Verify a model release locally, or publish it once it is authorized."""
    if context.release is None:
        context.stderr.write("lofi release: the command was given no release\n")

        return 1

    try:
        record = context.release(
            namespace.bundle,
            namespace.repo_id,
            namespace.run,
            publication=_publication(namespace),
            environ=context.environ,
        )

    except (OSError, RuntimeError, ValueError) as refused:
        context.stderr.write(f"lofi release: {refused}\n")

        return 2

    output = json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True)
    context.stdout.write(f"{output}\n")

    return 0


def _seconds(text: str) -> float:
    """Return a nonnegative number of seconds named on the command line."""
    value = float(text)

    if not value >= 0:
        msg = "a number of seconds must be nonnegative"
        raise ValueError(msg)

    return value


def _configs(namespace: argparse.Namespace, context: _Context) -> int:
    """Print each packaged configuration group with the options it holds."""
    del namespace

    for slot in packaged_groups():
        context.stdout.write(f"{slot}: {' '.join(packaged_options(slot))}\n")

    return 0


def _show_config(namespace: argparse.Namespace, context: _Context) -> int:
    """Print the resolved document annotated with each leaf's source."""
    with run_context(RunContext(context.cwd, namespace.seed)):
        composition = trace(_selection(namespace, context.cwd))
    context.stdout.write(
        render_annotated(resolved_document(composition.spec), composition.provenance)
    )

    return 0


def _write_price(
    experiment: Experiment, price: ExperimentPrice, directory: Path, context: _Context
) -> None:
    """Report each stack's timed step, the experiment's hours and where they went."""
    for stack in price.stacks:
        peak = stack.peak_allocated_bytes
        held = "unmeasured" if peak is None else f"{peak} bytes"
        context.stdout.write(
            f"stack {stack.stack} points={len(stack.points)} "
            f"length={stack.length.count} {stack.length.unit} "
            f"steps={stack.steps} "
            f"mean={stack.mean_seconds:.6f} s/step "
            f"peak {held}\n"
        )

    match experiment.compute:
        case ComputeEstimate(p50_hours=p50, p90_hours=p90, cost_model=model):
            declared = f"p50 {p50} p90 {p90} ({model})"

        case Unpriced():
            declared = "unpriced"

    hardware = price.hardware
    context.stdout.write(
        f"device: {price.device} ({hardware.device_name}, torch {hardware.torch}, "
        f"host {hardware.host_cpu})\n"
        f"points: {price.points}\n"
        f"cost model: {price.cost_model}\n"
        f"resamples: {price.resamples} (seed {price.bootstrap_seed})\n"
        f"p50 hours: {price.p50_hours:.{HOUR_DECIMALS}f}\n"
        f"p90 hours: {price.p90_hours:.{HOUR_DECIMALS}f}\n"
        f"declared: {declared}\n"
        f"written: {directory / PRICE_NAME}\n"
    )


def _price(namespace: argparse.Namespace, context: _Context) -> int:
    """Price an experiment from timed steps of its own stacks and write the price.

    The experiment document is read relative to the repository root, its sweep
    is expanded under that root and the price is written beneath it, so the
    record lies where a freeze of that root expands the same sweep. A stack
    whose peak lies above the activation budget its device policy declares is
    priced and named, and an array launch leaves its points out.
    """
    if context.timer is None:
        context.stderr.write("lofi price: the command was given no step timer\n")

        return 1

    root = _root(namespace, context)

    try:
        experiment = read_experiment(root / namespace.experiment)
        price = price_experiment(
            experiment,
            root,
            context.timer,
            device=namespace.device,
            warmup=namespace.warmup,
            repeats=namespace.repeats,
        )

    except RuntimeError as checked:
        context.stderr.write(f"lofi price: {checked}\n")

        return 1

    except ValueError as malformed:
        context.stderr.write(f"lofi price: {malformed}\n")

        return 2

    directory = root / SWEEPS_NAME / price.sweep_id
    write_price(directory, price)
    _write_price(experiment, price, directory, context)

    for stack in price.stacks:
        if stack.over_budget:
            context.stderr.write(
                f"lofi price: the stack {stack.stack} held "
                f"{stack.peak_allocated_bytes} bytes of the "
                f"{stack.budget.max_bytes} its device policy allows, so its "
                f"{len(stack.points)} points are not submitted to an array\n"
            )

    return 0


def _contract(namespace: argparse.Namespace, context: _Context) -> int:
    """Apply one declared contraction rung to an experiment and record it.

    The experiment document is read relative to the repository root, the rung
    is applied to the sweep in force and the amendment is recorded in the
    experiment's ledger, which every later load applies again.
    """
    root = _root(namespace, context)
    experiment = read_experiment(root / namespace.experiment)
    amended = amend_experiment(
        experiment,
        namespace.rung,
        root,
        on=datetime.now(tz=UTC).date(),
    )
    amendment = amended.amendments[-1]
    context.stdout.write(
        f"rung: {amendment.rung}\n"
        f"dropped: {json.dumps(list(amendment.dropped))}\n"
        f"sweep: {amendment.previous_sweep_id} -> {amendment.sweep_id}\n"
        f"points: {amendment.points}\n"
        f"ledger: {amendment_ledger(experiment.document)}\n"
    )

    return 0


def _ingest(namespace: argparse.Namespace, context: _Context) -> int:
    """Ingest an experiment's results against its freeze and write the accepted tables.

    The experiment document, the freeze record and the results directory are
    read relative to the repository root, the rows are accepted only once they
    conform to the freeze, and the accepted rows and their audit are written
    beside the results.
    """
    if context.ingest is None:
        context.stderr.write("lofi ingest: the command was given no ingestion\n")

        return 1

    root = _root(namespace, context)

    try:
        ingested = context.ingest(
            root / namespace.experiment, root / namespace.freeze, root
        )

    except ExperimentError:
        raise

    except ValueError as refused:
        context.stderr.write(f"lofi ingest: {refused}\n")

        return 1

    context.stdout.write(
        f"rows: {ingested.rows}\ningested: {ingested.table}\naudit: {ingested.audit}\n"
    )

    return 0


def _results(namespace: argparse.Namespace, context: _Context) -> int:
    """Render tables declared by an experiment from its stored rows."""
    if context.reports is None:
        context.stderr.write("lofi results: the command was given no report renderer\n")
        return 1
    root = _root(namespace, context)
    try:
        written = context.reports(
            root / namespace.experiment, root, pilot=namespace.pilot
        )
    except ValueError as refused:
        context.stderr.write(f"lofi results: {refused}\n")
        return 1
    for path in written:
        context.stdout.write(f"{path}\n")
    return 0


def _export(namespace: argparse.Namespace, context: _Context) -> int:
    """Write the offline bundle of a trained point and report where it lies."""
    if context.export is None:
        context.stderr.write("lofi export: the command was given no exporter\n")

        return 1

    try:
        context.export(namespace.run, namespace.release, namespace.out)

    except (OSError, ValueError) as refused:
        context.stderr.write(f"lofi export: {refused}\n")

        return 2

    context.stdout.write(f"{namespace.out}\n")

    return 0


def _bead_predict(namespace: argparse.Namespace, context: _Context) -> int:
    """Score bead inference items with the trained point a run holds."""
    if context.bead_predict is None:
        context.stderr.write(
            "lofi bead-predict: the command was given no bead prediction operation\n"
        )
        return 1

    try:
        written = context.bead_predict(
            namespace.run,
            namespace.serving,
            namespace.items,
            namespace.out,
        )
    except (OSError, ValueError) as refused:
        context.stderr.write(f"lofi bead-predict: {refused}\n")
        return 2

    context.stdout.write(f"{written} {namespace.out}\n")
    return 0


def _probe(namespace: argparse.Namespace, context: _Context) -> int:
    """Probe the trained point a run directory holds and report where the rows lie.

    The probes and sinks are those of the run's recorded configuration with the
    command line's overlays and overrides composed over it.
    """
    if context.probe is None:
        context.stderr.write("lofi probe: the command was given no probe operation\n")

        return 1

    run: Path = namespace.run
    selection = Selection(
        config=run / RESOLVED_NAME,
        profile=None,
        overrides=tuple(namespace.overrides),
        overlays=tuple(namespace.overlays),
        search_path=config_search_path(context.cwd),
    )

    try:
        written = context.probe(run, selection, namespace.out)

    except (OSError, ValueError) as refused:
        context.stderr.write(f"lofi probe: {refused}\n")

        return 2

    context.stdout.write(f"{written} {namespace.out}\n")

    return 0


def _publish(namespace: argparse.Namespace, context: _Context) -> int:
    """Publish a trained point's labeled derivations and report each role's file."""
    if context.publish is None:
        context.stderr.write("lofi publish: the command was given no publication\n")

        return 1

    try:
        published = context.publish(
            namespace.run, namespace.tokenizations, namespace.selection, namespace.out
        )

    except (OSError, ValueError) as refused:
        context.stderr.write(f"lofi publish: {refused}\n")

        return 2

    for role in published:
        context.stdout.write(
            f"{role.role}: {role.records} records, {role.unlabeled} unlabeled, "
            f"{role.path}\n"
        )

    return 0


def _active_learn(namespace: argparse.Namespace, context: _Context) -> int:
    """Run one active learning document and report where its run stands.

    A run that completes reports its rounds and directory. A run whose
    collector awaits the responses to a batch reports the round and the batch
    file; running the command again once the responses are ingested resumes it.
    """
    if context.active_learn is None:
        context.stderr.write(
            "lofi active-learn: the command was given no active-learning operation\n"
        )
        return 1

    out: Path = namespace.out

    try:
        manifest = context.active_learn(namespace.document, namespace.seed, out)

    except (OSError, TypeError, ValueError) as refused:
        context.stderr.write(f"lofi active-learn: {refused}\n")
        return 2

    batch = awaited_batch(out, manifest)

    if batch is None:
        context.stdout.write(f"completed {manifest.rounds} {out}\n")
    else:
        context.stdout.write(f"awaiting-responses {manifest.awaiting} {batch}\n")

    return 0


def _overlay_options(parser: argparse.ArgumentParser) -> None:
    """Add the overlay and override options a composing command takes."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an override applied last; repeatable",
    )
    parser.add_argument(
        "--extra",
        dest="overlays",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="an extra overlay document applied before the overrides; repeatable",
    )


def _composition_options(parser: argparse.ArgumentParser) -> None:
    """Add the options every composing command takes."""
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="the configuration document to compose",
    )
    parser.add_argument(
        "--profile", metavar="NAME", help="the named profile to apply over it"
    )
    _overlay_options(parser)


def _experiment_overrides(parser: argparse.ArgumentParser) -> None:
    """Add the overrides an experiment document is read with."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an override of the experiment document, such as "
        "launch.kind=array; repeatable",
    )


def _root_option(parser: argparse.ArgumentParser) -> None:
    """Add the store root option."""
    parser.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the directory run directories live under; the working directory "
        "when omitted",
    )


def _run_seed_option(parser: argparse.ArgumentParser) -> None:
    """Add the seed from which a composed run derives named streams."""
    parser.add_argument("--seed", type=int, default=0, metavar="N", help="the run seed")


def _experiment_root(text: str) -> ExperimentSeed:
    """Return the experiment root seed a command line names.

    Parameters
    ----------
    text : str
        The seed as typed.

    Raises
    ------
    ValueError
        If it is not a nonnegative integer.
    """
    return ExperimentSeed(int(text))


def _positive_count(text: str) -> int:
    """Return a positive count named on the command line."""
    count = int(text)

    if count <= 0:
        msg = "a count must be positive"
        raise ValueError(msg)

    return count


def _failure_options(parser: argparse.ArgumentParser) -> None:
    """Add one option per field of the failure policy a development point runs under."""
    fields = FailurePolicySpec.__field_specs__

    for field in ("numerical", "out_of_memory"):
        choices = get_args(fields[field].annotation)
        parser.add_argument(
            f"--{field.replace('_', '-')}",
            dest=field,
            choices=choices,
            default=None,
            help=f"the failure policy's {field.replace('_', ' ')} handling "
            f"(default: {fields[field].default})",
        )

    parser.add_argument(
        "--infrastructure-attempts",
        dest="infrastructure_attempts",
        type=_positive_count,
        default=None,
        metavar="N",
        help="the attempts the point receives across launches before an "
        "infrastructure failure settles it (default: "
        f"{fields['infrastructure_attempts'].default})",
    )


def _point_options(parser: argparse.ArgumentParser) -> None:
    """Add the options a single development point takes."""
    _composition_options(parser)
    _run_seed_option(parser)
    parser.add_argument(
        "--experiment-seed",
        type=_experiment_root,
        required=True,
        metavar="N",
        help="the root seed the point's run seed is derived from",
    )
    _failure_options(parser)
    _root_option(parser)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the command line."""
    parser = argparse.ArgumentParser(prog="lofi")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="train one composed point")
    _point_options(run)
    run.set_defaults(handler=_run)

    resume = commands.add_parser(
        "resume", help="train one composed point again from its run directory"
    )
    _point_options(resume)
    resume.set_defaults(handler=_resume)

    task = commands.add_parser(
        "task", help="train the listed points one scheduler task covers"
    )
    task.add_argument("points", type=Path, help="the points file a submission wrote")
    task.add_argument("index", type=int, help="the task index, from zero")
    task.add_argument("chunk_size", type=int, help="the points one task runs")
    task.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document the points belong to, relative to the "
        "executing checkout; it declares the freeze, the root seed and the "
        "failure policy",
    )
    _experiment_overrides(task)
    _root_option(task)
    task.set_defaults(handler=_task)

    launch = commands.add_parser(
        "launch", help="launch an experiment's points from its freeze, or as a pilot"
    )
    launch.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document to launch",
    )
    mode = launch.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--freeze",
        type=Path,
        metavar="PATH",
        help="the FREEZE.json of the freeze the experiment is launched from",
    )
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="run the experiment in this process outside any freeze, under "
        "pilot/ in the checkout, with its results marked unfrozen",
    )
    _experiment_overrides(launch)
    launch.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the store root an in-process launch runs under; the working "
        "directory when omitted, and pilot/ in the checkout for a pilot",
    )
    launch.add_argument(
        "--dry-run",
        action="store_true",
        help="write the launcher's submission and submit nothing",
    )
    launch.add_argument(
        "--with",
        dest="companions",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="an experiment of the same freeze admitted together with this one; "
        "repeatable",
    )
    launch.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="where to write the launch record",
    )
    launch.set_defaults(handler=_launch)

    status = commands.add_parser(
        "status", help="poll an experiment's store and rewrite its results"
    )
    status.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document to poll",
    )
    _experiment_overrides(status)
    _root_option(status)
    status.add_argument(
        "--interval",
        type=_seconds,
        required=True,
        metavar="SECONDS",
        help="seconds waited between polls",
    )
    status.add_argument(
        "--polls",
        type=_positive_count,
        required=True,
        metavar="N",
        help="the most polls made before the loop ends",
    )
    status.set_defaults(handler=_status)

    freeze = commands.add_parser(
        "freeze", help="tag a clean HEAD as a freeze and record what it froze"
    )
    freeze.add_argument(
        "--number", type=int, required=True, metavar="N", help="the freeze number"
    )
    freeze.add_argument(
        "--experiment",
        dest="experiments",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="an experiment document run against the freeze; repeatable",
    )
    freeze.add_argument(
        "--readiness",
        type=Path,
        required=True,
        metavar="PATH",
        help="a file with one line for each of FG-1 to FG-9",
    )
    freeze.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the repository root; the working directory when omitted",
    )
    freeze.set_defaults(handler=_freeze)

    seed_budget = commands.add_parser(
        "seed-budget",
        help="price an experiment's seed count from a pilot's rows before freezing",
    )
    seed_budget.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document whose seed count is priced",
    )
    seed_budget.add_argument(
        "--pilot",
        type=Path,
        required=True,
        metavar="DIR",
        help="the results directory whose rows score the budgeted metric",
    )
    seed_budget.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the repository root the experiment and pilot are read under; the "
        "working directory when omitted",
    )
    seed_budget.set_defaults(handler=_seed_budget)

    release = commands.add_parser(
        "release",
        help="verify a model release locally, or publish it once authorized",
    )
    release.add_argument(
        "--bundle",
        type=Path,
        required=True,
        metavar="DIR",
        help="the offline bundle lofi export wrote",
    )
    release.add_argument(
        "--repo-id",
        required=True,
        metavar="OWNER/NAME",
        help="the Hugging Face repository the release is published to",
    )
    release.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="DIR",
        help="the run directory whose declared release checkpoint the bundle "
        "reproduces",
    )
    release.add_argument(
        "--publish",
        action="store_true",
        help="publish the release; without it the command only verifies it",
    )
    release.add_argument(
        "--evidence",
        type=Path,
        metavar="PATH",
        help="where a publication records its evidence",
    )
    release.add_argument(
        "--reviewer", metavar="NAME", help="the reviewer who authorized publication"
    )
    release.add_argument(
        "--authorized-license",
        metavar="SPDX",
        help="the license expression the reviewer approved",
    )
    release.add_argument(
        "--confirm-corpus-redistribution",
        action="store_true",
        help="confirm the corpus's redistribution terms were reviewed",
    )
    release.add_argument(
        "--confirm-artifact-redistribution",
        action="store_true",
        help="confirm the artifact's redistribution terms were reviewed",
    )
    release.add_argument(
        "--token-environment",
        default="HF_TOKEN",
        metavar="NAME",
        help="the environment variable the Hub token is read from "
        "(default: %(default)s)",
    )
    release.set_defaults(handler=_release)

    configs = commands.add_parser(
        "configs", help="print the packaged configuration groups and their options"
    )
    configs.set_defaults(handler=_configs)

    show = commands.add_parser(
        "show-config", help="print the resolved document with each leaf's source"
    )
    _composition_options(show)
    _run_seed_option(show)
    show.set_defaults(handler=_show_config)

    price = commands.add_parser(
        "price", help="price an experiment from timed training steps of its own stacks"
    )
    price.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document to price",
    )
    price.add_argument(
        "--device",
        required=True,
        metavar="DEVICE",
        help=(
            "the device every stack is timed on, such as cpu, mps or cuda:0, "
            "of the kind each point's device policy declares"
        ),
    )
    price.add_argument(
        "--warmup",
        type=int,
        required=True,
        metavar="N",
        help="untimed steps each stack runs before its timed ones",
    )
    price.add_argument(
        "--repeats",
        type=int,
        required=True,
        metavar="N",
        help="timed steps each stack runs, at least two",
    )
    price.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the repository root the experiment is read and expanded under and the "
        "price is written beneath; the working directory when omitted",
    )
    price.set_defaults(handler=_price)

    contract = commands.add_parser(
        "contract",
        help="apply one declared contraction rung to an experiment and record it",
    )
    contract.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document to contract",
    )
    contract.add_argument(
        "--rung",
        type=str,
        required=True,
        metavar="RUNG",
        help="the declared contraction rung to apply",
    )
    contract.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the repository root the experiment is read and expanded under; "
        "the working directory when omitted",
    )
    contract.set_defaults(handler=_contract)

    ingest = commands.add_parser(
        "ingest",
        help="accept an experiment's results against its freeze and write them",
    )
    ingest.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document whose results are ingested",
    )
    ingest.add_argument(
        "--freeze",
        type=Path,
        required=True,
        metavar="PATH",
        help="the FREEZE.json of the freeze the experiment ran under",
    )
    ingest.add_argument(
        "--root",
        type=Path,
        metavar="DIR",
        help="the repository root the experiment, freeze and results are read "
        "under; the working directory when omitted",
    )
    ingest.set_defaults(handler=_ingest)

    results = commands.add_parser(
        "results", help="render the tables an experiment declares from rows.parquet"
    )
    results.add_argument(
        "--experiment",
        type=Path,
        required=True,
        metavar="PATH",
        help="the experiment document whose reports are rendered",
    )
    results.add_argument(
        "--pilot",
        action="store_true",
        help="render the experiment's pilot results, run outside any freeze, "
        "from pilot/results/<experiment>/ under the root",
    )
    _root_option(results)
    results.set_defaults(handler=_results)

    export = commands.add_parser(
        "export",
        help="bundle a trained point for the Hugging Face Hub, without a network",
    )
    export.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="DIR",
        help="the run directory of the completed point",
    )
    export.add_argument(
        "--release",
        type=Path,
        required=True,
        metavar="PATH",
        help="the release declaration naming the model card's metadata",
    )
    export.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="the bundle directory to create, which must not exist",
    )
    export.set_defaults(handler=_export)

    bead_predict = commands.add_parser(
        "bead-predict",
        help="score bead inference items with the trained point a run holds",
    )
    bead_predict.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="DIR",
        help="the run directory of the completed point",
    )
    bead_predict.add_argument(
        "--serving",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "the serving document declaring the response encoding and the "
            "element each input role of the point reads"
        ),
    )
    bead_predict.add_argument(
        "--items",
        type=Path,
        required=True,
        metavar="PATH",
        help="the bead items JSONL document",
    )
    bead_predict.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="the prediction rows JSONL file to create, which must not exist",
    )
    bead_predict.set_defaults(handler=_bead_predict)

    probe = commands.add_parser(
        "probe",
        help="execute declared probes against the trained point a run directory holds",
    )
    probe.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="DIR",
        help="the run directory of the completed point",
    )
    probe.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="the probe rows file to create, which must not exist",
    )
    _overlay_options(probe)
    probe.set_defaults(handler=_probe)

    publish = commands.add_parser(
        "publish",
        help="publish a trained point's labeled derivations as Layers records",
    )
    publish.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="DIR",
        help="the run directory of the completed point",
    )
    publish.add_argument(
        "--tokenizations",
        type=Path,
        required=True,
        metavar="PATH",
        help="the document joining each item id to its Layers tokenization",
    )
    publish.add_argument(
        "--selection",
        type=Path,
        required=True,
        metavar="PATH",
        help="the document naming the evaluator roles, the split and the formalism",
    )
    publish.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="the records directory to create, which must not exist",
    )
    publish.set_defaults(handler=_publish)

    active_learn = commands.add_parser(
        "active-learn",
        help="run the rounds of one active-learning document",
    )
    active_learn.add_argument(
        "document", type=Path, metavar="DOCUMENT", help="the active-learning document"
    )
    active_learn.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="the directory the run keeps its rounds, manifest and rows under; "
        "the same directory resumes a run awaiting responses",
    )
    _run_seed_option(active_learn)
    active_learn.set_defaults(handler=_active_learn)

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    training: TrainingOperations | None = None,
    launching: ExperimentLaunching | None = None,
    freeze: FreezeMinting | None = None,
    seed_budget: SeedBudgeting | None = None,
    release: ModelReleasing | None = None,
    timer: StepTimer | None = None,
    export: PointExport | None = None,
    bead_predict: BeadPrediction | None = None,
    probe: PointProbe | None = None,
    publish: PointPublication | None = None,
    ingest: ResultIngestion | None = None,
    reports: ExperimentReporting | None = None,
    active_learn: ActiveLearningRun | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command line and return its exit status.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments after the program name; the process's own when None.
    training : TrainingOperations | None
        What prepares, fits and scores one planned run, which every training
        command executes its points through, or None for a command that does
        not train.
    launching : ExperimentLaunching | None
        What launches, pilots and polls experiments against the site, or None
        for a command that launches nothing.
    freeze : FreezeMinting | None
        What mints a freeze, or None for a command that mints none.
    seed_budget : SeedBudgeting | None
        What prices an experiment's seed count, or None for a command that
        prices none.
    release : ModelReleasing | None
        What verifies and publishes a model release, or None for a command
        that releases nothing.
    timer : StepTimer | None
        What builds and times the stacks an experiment is priced from, or None
        for a command that prices nothing.
    export : PointExport | None
        What writes a trained point's offline bundle, or None for a command
        that exports nothing.
    bead_predict : BeadPrediction | None
        What scores bead inference items with a trained point, or None for a
        command that predicts nothing.
    probe : PointProbe | None
        What probes a trained point a run directory holds, or None for a command
        that probes nothing.
    publish : PointPublication | None
        What publishes a trained point's predictions as external records, or
        None for a command that publishes nothing.
    ingest : ResultIngestion | None
        What accepts an experiment's results against its freeze, or None for
        a command that ingests nothing.
    reports : ExperimentReporting | None
        What renders an experiment's declared reports, or None for a command
        that renders no reports.
    active_learn : ActiveLearningRun | None
        What runs an active learning document, or None for a command that runs
        none.
    environ : Mapping[str, str] | None
        The environment a launch or poll resolves the site profile from; the
        process's own when None.
    stdout : TextIO | None
        Where results are written; the process's own when None.
    stderr : TextIO | None
        Where refusals and failures are written; the process's own when None.

    Returns
    -------
    int
        Zero on success, one when a point failed or a report was refused,
        and two when the input was refused.
    """
    namespace = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    context = _Context(
        training=training,
        launching=launching,
        freeze=freeze,
        seed_budget=seed_budget,
        release=release,
        timer=timer,
        export=export,
        bead_predict=bead_predict,
        probe=probe,
        publish=publish,
        ingest=ingest,
        reports=reports,
        active_learn=active_learn,
        environ=os.environ if environ is None else environ,
        stdout=sys.stdout if stdout is None else stdout,
        stderr=sys.stderr if stderr is None else stderr,
        cwd=Path.cwd(),
    )

    try:
        return namespace.handler(namespace, context)

    except _REFUSALS as refused:
        context.stderr.write(f"lofi {namespace.command}: {refused}\n")

        return 2

    except (
        PointExecutionError,
        SourceRevisionError,
        SubmissionError,
        UnfrozenRevisionError,
    ) as failed:
        context.stderr.write(f"lofi {namespace.command}: {failed}\n")

        return 1
