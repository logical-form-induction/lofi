"""Mint one freeze: tag the gate commit and record what it froze.

A freeze is the annotated tag ``freeze/<n>`` on a clean ``HEAD``, the
file ``freezes/<n>/FREEZE.json`` and one row of ``freezes/ACTIVE.tsv``.
The file gathers identities that existing producers already compute: the
lockfile digest and the installed didactic version, the run-schema
fingerprint, each experiment's sweep identifier and preregistration digest, the
checkpoint-manifest schema, the benchmark baseline directory digest and the
pinned fixture. File digests are taken from the blobs ``HEAD`` tracks, and every
experiment, configuration document and preregistration must be tracked. An
experiment's configuration is a document of any installed kind, and its sweep
identifier is taken through the same expansion whatever the kind. Every
experiment must declare its compute estimate. The declared P50, P90 and
cost-model version of an experiment whose configuration kind trains a timed
step must be what the tracked ``sweeps/<sweep_id>/price.json``
that ``lofi price --root .`` wrote for its sweep recomputes to, at the decimals
the price is reported in; that record must name the experiment's sweep, the
experiment and its freeze, and the experiment's ``compute`` entry names the
record as its source. No producer prices a kind that trains no timed step, such
as a judgment fit, so such an experiment's estimate is supplied by the PI: it
is recorded verbatim in the experiment's ``compute`` entry with the source
``supplied``, and it must name a cost model other than the measured one. The
file records under ``identities`` what the experiment's configuration kind
reports for each point, so a run point's grammar, codec, corpus and per-role
component identities equal what its manifest records when it trains. The
readiness notes are external facts the PI supplies, recorded verbatim and
listed under ``supplied``. A freeze number is minted once; a change is the next
number. The file and row are written before the tag is created, and removed if
the tag cannot be, so a freeze is never tagged without its record. A pilot runs
outside any freeze, so a ``HEAD`` that tracks any file under ``pilot/`` is
refused with ``PilotResultsError`` before anything is written: a pilot's
results never ride inside a freeze's tag. The written file and row are left for
the PI to commit. ``lofi freeze`` mints one::

    uv run lofi freeze --number 1 \
        --experiment experiments/headline.yaml \
        --experiment experiments/analysis.yaml \
        --readiness ~/freeze-1-readiness.txt
"""

import json
import shutil
import subprocess
from hashlib import blake2b
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NamedTuple

from lofi.eval import freeze_number
from lofi.run import (
    COST_MODEL,
    GRID_COST_MODEL,
    HOUR_DECIMALS,
    PRICE_NAME,
    SWEEPS_NAME,
    ComputeEstimate,
    ExperimentError,
    Unpriced,
    expand_experiment,
    read_experiment,
    read_price,
    sweep_id,
)
from lofi.run._benchmark_fixture import PinnedFingerprint
from lofi.run._boundary.experiment import model_schema_uri
from lofi.run.results import PILOT_NAME, PilotResultsError
from lofi.spec import FingerprintIso, fingerprint_run
from lofi.train import CheckpointManifest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.eval import Preregistration
    from lofi.run import Experiment, SweepPoint

LOCKFILE: Final = Path("uv.lock")
BASELINES: Final = Path("tests/benchmarks/baselines")
FIXTURE_FINGERPRINT: Final = Path("tests/benchmarks/fixtures/chart_200.fingerprint")
FREEZES: Final = Path("freezes")
ACTIVE: Final = FREEZES / "ACTIVE.tsv"
FREEZE_NAME: Final = "FREEZE.json"
ACTIVE_HEADER: Final = ("freeze_id", "number", "commit", "tag_object")
READINESS_CONDITIONS: Final = tuple(f"FG-{condition}" for condition in range(1, 10))
SUPPLIED: Final = ("readiness_notes",)


class FreezeError(RuntimeError):
    """A freeze that cannot be minted from the repository as it stands."""


class Loaded(NamedTuple):
    """An experiment, its path, its sweep, its estimate and its points' identities."""

    path: Path
    experiment: Experiment
    sweep_id: str
    compute: dict[str, JsonValue]
    identities: dict[str, JsonValue]


class Supplied(NamedTuple):
    """What the PI supplies because no producer in the repository computes it."""

    readiness_notes: tuple[str, ...]


class Minted(NamedTuple):
    """Where a minted freeze was recorded and what it recorded."""

    tag: str
    path: Path
    record: dict[str, JsonValue]


def _git(
    root: Path, *arguments: str, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Return the result of one git command run against a repository."""
    executable = shutil.which("git")

    if executable is None:
        msg = "git is not installed"
        raise FreezeError(msg)

    return subprocess.run(
        (executable, "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        input=stdin,
    )


def _git_bytes(root: Path, *arguments: str, stdin: bytes | None = None) -> bytes:
    """Return a git command's output, refusing a failed command."""
    result = _git(root, *arguments, stdin=stdin)

    if result.returncode:
        stderr = result.stderr.decode(errors="replace").strip()
        msg = f"git {' '.join(arguments)} failed: {stderr}"
        raise FreezeError(msg)

    return result.stdout


def _git_output(root: Path, *arguments: str) -> str:
    """Return a git command's decoded output, refusing a failed command."""
    return _git_bytes(root, *arguments).decode().strip()


def blob_blake2b(root: Path, path: Path, /) -> str:
    """Return the BLAKE2b digest of the bytes ``HEAD`` tracks at a path.

    Parameters
    ----------
    root : Path
        The repository root.
    path : Path
        The file, relative to the root.

    Raises
    ------
    FreezeError
        If ``HEAD`` tracks no file at the path.
    """
    return blake2b(
        _git_bytes(root, "cat-file", "blob", f"HEAD:{path.as_posix()}")
    ).hexdigest()


def tree_blake2b(root: Path, directory: Path, /) -> str:
    """Return the BLAKE2b digest of every file ``HEAD`` tracks under a directory.

    The digest covers one line per tracked file, in path order, holding the
    file's POSIX path relative to the directory, a NUL and the BLAKE2b digest of
    its blob, so a renamed, added, removed or edited file changes it.

    Parameters
    ----------
    root : Path
        The repository root.
    directory : Path
        The directory, relative to the root.

    Raises
    ------
    FreezeError
        If ``HEAD`` tracks no file under the directory.
    """
    listing = _git_bytes(
        root, "ls-tree", "-r", "-z", "HEAD", "--", directory.as_posix()
    )
    lines: list[str] = []

    for entry in listing.split(b"\0"):
        header, _, name = entry.decode().partition("\t")

        if not name or header.split(" ")[1] != "blob":
            continue

        relative = PurePosixPath(name).relative_to(directory.as_posix())
        blob = _git_bytes(root, "cat-file", "blob", header.split(" ")[2])
        lines.append(f"{relative}\0{blake2b(blob).hexdigest()}\n")

    if not lines:
        msg = f"HEAD tracks no file under {directory}"
        raise FreezeError(msg)

    return blake2b("".join(sorted(lines)).encode()).hexdigest()


def preregistration_digest(preregistration: Preregistration, /) -> str:
    """Return the BLAKE2b digest of a preregistration's canonical JSON.

    Parameters
    ----------
    preregistration : Preregistration
        The plan whose grid, hypothesis count, families and margins are hashed.
    """
    canonical = json.dumps(
        preregistration.to_spec(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return blake2b(canonical.encode()).hexdigest()


def read_readiness(path: Path, /) -> tuple[str, ...]:
    """Return the nine readiness lines, one per fork condition, verbatim.

    Parameters
    ----------
    path : Path
        A text file holding one line for each of FG-1 to FG-9, in order.

    Raises
    ------
    FreezeError
        If the file is unreadable or does not hold exactly those nine lines.
    """
    try:
        lines = tuple(
            line for line in path.read_text(encoding="utf-8").splitlines() if line
        )

    except OSError as refused:
        msg = f"the readiness notes cannot be read: {refused}"
        raise FreezeError(msg) from refused

    if len(lines) != len(READINESS_CONDITIONS):
        msg = f"the readiness notes hold {len(lines)} lines, not one per FG-1 to FG-9"
        raise FreezeError(msg)

    for condition, line in zip(READINESS_CONDITIONS, lines, strict=True):
        label, _, note = line.partition(" ")

        if label.rstrip(":") != condition or not note.strip():
            msg = f"the readiness line for {condition} reads {line!r}"
            raise FreezeError(msg)

    return lines


def _require_new(root: Path, number: int, tag: str) -> None:
    """Refuse a freeze number already tagged, written or listed as active."""
    found = _git(root, "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}")

    if not found.returncode:
        msg = f"{tag} is already tagged, and a freeze is never re-tagged"
        raise FreezeError(msg)

    if (root / FREEZES / str(number)).exists():
        msg = f"{FREEZES / str(number)} already exists"
        raise FreezeError(msg)

    active = root / ACTIVE

    if active.is_file() and any(
        line.split("\t", 1)[0] == tag
        for line in active.read_text(encoding="utf-8").splitlines()
    ):
        msg = f"{tag} is already listed in {ACTIVE}"
        raise FreezeError(msg)


def _require_no_pilot(root: Path, /) -> None:
    """Refuse a ``HEAD`` that tracks a pilot's runs or results.

    Raises
    ------
    PilotResultsError
        If ``HEAD`` tracks any file under the pilot root.
    """
    tracked = _git_output(
        root, "ls-tree", "-r", "--name-only", "HEAD", "--", PILOT_NAME
    )

    if tracked:
        listed = ", ".join(tracked.splitlines())
        msg = (
            f"HEAD tracks a pilot's files, run outside any freeze, which a freeze "
            f"would tag: {listed}"
        )
        raise PilotResultsError(msg)


def _require_priced(
    root: Path, path: Path, experiment: Experiment, sweep: str
) -> dict[str, JsonValue]:
    """Return an experiment's admitted estimate, refusing one no record supports.

    Parameters
    ----------
    root : Path
        The repository root.
    path : Path
        The experiment document as named on the command line.
    experiment : Experiment
        The loaded experiment.
    sweep : str
        The identifier of the experiment's expanded sweep.

    Returns
    -------
    dict[str, JsonValue]
        The declared estimate and its source: the price record the estimate of
        an experiment whose configuration kind trains a timed step was
        recomputed from, or ``supplied`` for any other experiment, which no
        producer prices.

    Raises
    ------
    FreezeError
        If the experiment is unpriced, is a judgment fit experiment whose supplied
        estimate names the measured cost model, or is a run experiment whose
        price record is not tracked or not readable, names another sweep,
        experiment or freeze, or recomputes another figure or cost-model version
        than the experiment declares.
    """
    match experiment.compute:
        case Unpriced():
            msg = (
                f"the experiment {path} is unpriced: price it with lofi price "
                f"and declare the estimate it reports"
            )
            raise FreezeError(msg)

        case ComputeEstimate() as declared:
            pass

    estimate: dict[str, JsonValue] = {
        "p50_hours": declared.p50_hours,
        "p90_hours": declared.p90_hours,
        "cost_model": declared.cost_model,
    }

    if not experiment.config.timed:
        if declared.cost_model in {COST_MODEL, GRID_COST_MODEL}:
            msg = (
                f"the experiment {path} supplies an estimate under the measured cost "
                f"model {declared.cost_model!r}, which prices only timed run stacks; "
                f"name the model the supplied figures came from"
            )
            raise FreezeError(msg)

        return {**estimate, "source": "supplied"}

    record = Path(SWEEPS_NAME) / sweep / PRICE_NAME
    tracked = _git(root, "ls-files", "--error-unmatch", "--", str(record))

    if tracked.returncode:
        msg = f"the price record {record} of the experiment {path} is not tracked"
        raise FreezeError(msg)

    try:
        price = read_price((root / record).parent)

    except (TypeError, ValueError) as refused:
        msg = (
            f"the price record {record} of the experiment {path} is refused: {refused}"
        )
        raise FreezeError(msg) from refused

    if price is None:
        msg = f"the price record {record} of the experiment {path} is absent"
        raise FreezeError(msg)

    for field, recorded, expected in (
        ("sweep_id", price.sweep_id, sweep),
        ("experiment", price.experiment, experiment.name),
        ("freeze_id", price.freeze_id, experiment.freeze_id),
    ):
        if recorded != expected:
            msg = (
                f"the price record {record} of the experiment {path} records "
                f"{field} {recorded!r}, not the experiment's {expected!r}"
            )
            raise FreezeError(msg)

    recomputed = ComputeEstimate(
        round(price.p50_hours, HOUR_DECIMALS),
        round(price.p90_hours, HOUR_DECIMALS),
        price.cost_model,
    )

    for field, stated, computed in zip(
        ComputeEstimate._fields, declared, recomputed, strict=True
    ):
        if stated != computed:
            msg = (
                f"the experiment {path} declares {field} {stated!r}, but its price "
                f"record {record} recomputes {computed!r}"
            )
            raise FreezeError(msg)

    return {**estimate, "source": record.as_posix()}


def _identities(
    path: Path, experiment: Experiment, points: Sequence[SweepPoint]
) -> dict[str, JsonValue]:
    """Return the identities the experiment's configuration kind records per point.

    Parameters
    ----------
    path : Path
        The experiment document, named in a refusal.
    experiment : Experiment
        The experiment the points expand from.
    points : Sequence[SweepPoint]
        Its expanded points.

    Returns
    -------
    dict[str, JsonValue]
        By point identifier, the identities the point's manifest records when
        it executes, such as a run point's grammar, codec, corpus and component
        fingerprints; a kind that plans no blueprint records none.

    Raises
    ------
    FreezeError
        If a point cannot be planned for what it executes.
    """
    try:
        return dict(experiment.config.identities(points))

    except ValueError as refused:
        msg = f"the experiment {path} is refused: {refused}"
        raise FreezeError(msg) from refused


def _experiments(root: Path, paths: Sequence[Path], tag: str) -> dict[str, Loaded]:
    """Return each admitted experiment by name with its path, sweep and estimate."""
    if not paths:
        msg = "a freeze names at least one experiment"
        raise FreezeError(msg)

    experiments: dict[str, Loaded] = {}

    for path in paths:
        located = (root / path).resolve()

        if not located.is_relative_to(root):
            msg = f"the experiment {path} lies outside the repository"
            raise FreezeError(msg)

        try:
            experiment = read_experiment(located)

        except ExperimentError as refused:
            raise FreezeError(str(refused)) from refused

        if experiment.freeze_id != tag:
            msg = (
                f"the experiment {path} runs against {experiment.freeze_id}, "
                f"not the {tag} being minted"
            )
            raise FreezeError(msg)

        if experiment.name in experiments:
            msg = f"two experiments are named {experiment.name!r}"
            raise FreezeError(msg)

        documents = (
            located,
            experiment.config.path,
            experiment.preregistration_document,
        )
        tracked = _git(root, "ls-files", "--error-unmatch", "--", *map(str, documents))

        if tracked.returncode:
            stderr = tracked.stderr.decode(errors="replace").strip()
            msg = (
                f"the experiment {path}, its configuration document and its "
                f"preregistration are not all tracked: {stderr}"
            )
            raise FreezeError(msg)

        try:
            points = expand_experiment(experiment, root)

        except ExperimentError as refused:
            raise FreezeError(str(refused)) from refused

        expanded = sweep_id(points)
        estimate = _require_priced(root, path, experiment, expanded)
        experiments[experiment.name] = Loaded(
            located.relative_to(root),
            experiment,
            expanded,
            estimate,
            _identities(path, experiment, points),
        )

    return experiments


def mint(
    root: Path,
    *,
    number: int,
    experiments: Sequence[Path],
    supplied: Supplied,
) -> Minted:
    """Tag a clean ``HEAD`` as a freeze and record what it froze.

    Every identity is computed and every refusal is checked before anything
    is written. The record and the active row are then written and the tag is
    created last; if writing or tagging fails, what was written is removed, so
    a refused freeze leaves the repository untouched.

    Parameters
    ----------
    root : Path
        The repository root.
    number : int
        The freeze number, positive and never minted before.
    experiments : Sequence[Path]
        The experiment documents, relative to the root, run against the freeze.
    supplied : Supplied
        The identities the PI supplies.

    Returns
    -------
    Minted
        The tag, the written file and its record.

    Raises
    ------
    PilotResultsError
        If ``HEAD`` tracks any file under the pilot root.
    FreezeError
        If the number does not form a freeze identifier, the tree is
        dirty, the number was minted before, an experiment is refused, untracked,
        unpriced, declares an estimate its tracked price record does not
        recompute, runs against another freeze or has a point training cannot
        plan, ``HEAD`` tracks no
        lockfile, baseline or pinned fixture, or the tag cannot be created.
    OSError
        If the pinned fixture record cannot be read, or the record or the
        active row cannot be written.
    """
    root = root.resolve()
    tag = f"freeze/{number}"

    try:
        freeze_number(tag)

    except (TypeError, ValueError) as refused:
        raise FreezeError(str(refused)) from refused

    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")

    if status:
        msg = f"the working tree is not clean:\n{status}"
        raise FreezeError(msg)

    _require_new(root, number, tag)
    _require_no_pilot(root)
    loaded = _experiments(root, experiments, tag)
    commit = _git_output(root, "rev-parse", "HEAD")
    pinned = PinnedFingerprint.read(root / FIXTURE_FINGERPRINT)
    checkpoint_fields: list[JsonValue] = [*sorted(CheckpointManifest.__field_specs__)]
    readiness: list[JsonValue] = list(supplied.readiness_notes)
    listed: list[JsonValue] = list(SUPPLIED)
    record: dict[str, JsonValue] = {
        "freeze_id": tag,
        "number": number,
        "experiments": {
            name: {"path": entry.path.as_posix(), "role": entry.experiment.role}
            for name, entry in loaded.items()
        },
        "lockfile_digest": {
            "blake2b": blob_blake2b(root, LOCKFILE),
            "didactic": version("didactic"),
        },
        "schema_fingerprint": json.loads(FingerprintIso().forward(fingerprint_run({}))),
        "sweep_id": {name: entry.sweep_id for name, entry in loaded.items()},
        "compute": {name: entry.compute for name, entry in loaded.items()},
        "preregistration_digest": {
            name: preregistration_digest(entry.experiment.preregistration)
            for name, entry in loaded.items()
        },
        "checkpoint_schema": {
            "fields": checkpoint_fields,
            "schema": model_schema_uri(CheckpointManifest),
        },
        "budget_baselines": {"baselines": tree_blake2b(root, BASELINES)},
        "fixture_fingerprint": {
            "blake2b": blob_blake2b(root, FIXTURE_FINGERPRINT),
            "fingerprint": pinned.fingerprint,
        },
        "identities": {name: entry.identities for name, entry in loaded.items()},
        "readiness_notes": readiness,
        "supplied": listed,
    }

    tagger = _git_output(root, "var", "GIT_COMMITTER_IDENT")
    annotation = f"object {commit}\ntype commit\ntag {tag}\ntagger {tagger}\n\n{tag}\n"
    tag_object = _git_bytes(root, "mktag", stdin=annotation.encode()).decode().strip()
    record["git_tag"] = {"name": tag, "commit": commit, "object": tag_object}
    directory = root / FREEZES / str(number)
    created = [
        folder for folder in (directory, directory.parent) if not folder.exists()
    ]
    path = directory / FREEZE_NAME
    active = root / ACTIVE
    prior = active.read_bytes() if active.is_file() else None
    row = (tag, str(number), commit, tag_object)

    try:
        directory.mkdir(parents=True)

        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{json.dumps(record, indent=2, sort_keys=True)}\n")

        with active.open("a", encoding="utf-8", newline="\n") as handle:
            if handle.tell() == 0:
                handle.write("\t".join(ACTIVE_HEADER) + "\n")

            handle.write("\t".join(row) + "\n")

        _git_bytes(root, "update-ref", f"refs/tags/{tag}", tag_object, "")

    except BaseException:
        path.unlink(missing_ok=True)

        if prior is not None:
            active.write_bytes(prior)

        elif active.is_file():
            active.unlink()

        for folder in created:
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

        raise

    return Minted(tag, path, record)


def mint_freeze(
    root: Path, /, *, number: int, experiments: Sequence[Path], readiness: Path
) -> dict[str, JsonValue]:
    """Mint one freeze from the PI's readiness notes and return what it recorded.

    Parameters
    ----------
    root : Path
        The repository root.
    number : int
        The freeze number, positive and never minted before.
    experiments : Sequence[Path]
        The experiment documents, relative to the root, run against the freeze.
    readiness : Path
        A text file with one line for each of FG-1 to FG-9.

    Raises
    ------
    FreezeError
        If the readiness notes are malformed or the freeze is refused.
    PilotResultsError
        If ``HEAD`` tracks any file under the pilot root.
    """
    supplied = Supplied(readiness_notes=read_readiness(readiness))

    return mint(root, number=number, experiments=experiments, supplied=supplied).record
