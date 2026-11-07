import io
import json
import shutil
import subprocess
from hashlib import blake2b, sha256
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
from tests.unit.run.test__experiment import (
    PREREGISTRATION,
    a_judgment_experiment_document,
    an_experiment_document,
    write_experiment,
)

from lofi.data import BatchShape
from lofi.eval import read_preregistration
from lofi.judgment import JudgmentFitSpec
from lofi.run import (
    COST_MODEL,
    HOUR_DECIMALS,
    SWEEPS_NAME,
    Axis,
    ExperimentPrice,
    GridSweepExpander,
    HardwareDescriptor,
    MemoryBudget,
    Selection,
    StackPrice,
    StackTiming,
    Sweep,
    build_parser,
    composer,
    expand_experiment,
    plan_training,
    point_id,
    read_experiment,
    resolve_point,
    sweep_id,
    write_price,
)
from lofi.run.freezing import (
    ACTIVE_HEADER,
    SUPPLIED,
    FreezeError,
    mint_freeze,
    read_readiness,
)
from lofi.run.results import PilotResultsError
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    FingerprintIso,
    RunSpecCodec,
    SpecLoader,
    config_search_path,
    fingerprint_run,
)
from lofi.train import CheckpointManifest
from lofi.training_plugins import TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.run import SweepPoint

TREE: Final = Path(__file__).resolve().parents[3]
"""The checkout whose pinned fixture record the temporary repository copies."""

GIT: Final = shutil.which("git") or "git"

SYNTHETIC: Final = TREE / "lofi/configs/run/synthetic-parser.yaml"
"""The packaged run document every run experiment of the repository trains."""

PACKAGED_CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))

READINESS: Final = "\n".join(
    f"FG-{condition}: confirmed by the PI on bluehive, 2026-10-12"
    for condition in range(1, 10)
)


def main(argv: Sequence[str], *, stdout: io.StringIO | None = None) -> int:
    """Mint a freeze as ``lofi freeze`` does, raising what the command reports."""
    namespace = build_parser().parse_args(["freeze", *argv])
    record = mint_freeze(
        Path.cwd() if namespace.root is None else namespace.root,
        number=namespace.number,
        experiments=tuple(namespace.experiments),
        readiness=namespace.readiness,
    )
    (io.StringIO() if stdout is None else stdout).write(
        f"{json.dumps(record, sort_keys=True)}\n"
    )

    return 0


def git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        (GIT, "-C", str(root), *arguments), check=True, capture_output=True
    )

    return completed.stdout


def git(root: Path, *arguments: str) -> str:
    return git_bytes(root, *arguments).decode().strip()


ANALYSIS_PREREGISTRATION: Final = {
    **PREREGISTRATION,
    "grid": {"trainer.epochs": [1, 3]},
}


def record_price(root: Path, name: str) -> Path:
    """Record a price for one experiment's sweep and declare its recomputed figures."""
    path = root / "experiments" / f"{name}.yaml"
    experiment = read_experiment(path)
    points = expand_experiment(experiment, root)
    price = ExperimentPrice(
        experiment=experiment.name,
        freeze_id=experiment.freeze_id,
        sweep_id=sweep_id(points),
        device="cpu",
        hardware=HardwareDescriptor(
            device_name="cpu", driver=None, cuda=None, torch="2.8.0", host_cpu="arm"
        ),
        warmup=1,
        repeats=3,
        resamples=1_000,
        bootstrap_seed=experiment.seed_policy.root,
        stacks=[
            StackPrice(
                stack="stack",
                points=[point.point_id for point in points],
                length=TrainingLength(2, TrainingUnit.EPOCHS),
                timing=StackTiming(
                    steps_per_unit=3,
                    positions=[2, 0, 2],
                    seconds=[0.5, 0.7, 1.3],
                    peak_allocated_bytes=None,
                    shape=BatchShape(9.0, 4.0),
                ),
                budget=MemoryBudget(8 * 1000**3),
            )
        ],
        grid=None,
    )
    directory = root / SWEEPS_NAME / price.sweep_id
    write_price(directory, price)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["compute"] = {
        "kind": "estimate",
        "p50_hours": round(price.p50_hours, HOUR_DECIMALS),
        "p90_hours": round(price.p90_hours, HOUR_DECIMALS),
        "cost_model": price.cost_model,
    }
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return directory / "price.json"


def declare(root: Path, name: str, field: str, value: JsonValue) -> None:
    path = root / "experiments" / f"{name}.yaml"
    document = json.loads(path.read_text(encoding="utf-8"))
    compute = document["compute"]
    compute[field] = value
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")


def commit_all(root: Path, message: str) -> None:
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", message)


def train_the_synthetic_parser(root: Path) -> None:
    """Make the packaged synthetic parser the run document every experiment reads."""
    run = json.loads(PACKAGED_CODEC(SYNTHETIC).model_dump_json())
    run["evaluators"] = {
        "held-out": {
            "kind": "launcher-unit-evaluator",
            "interval": {"replicates": 50, "level": 0.95},
        }
    }
    (root / "experiments" / "run.json").write_text(
        json.dumps(run), encoding="utf-8", newline="\n"
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "--quiet", "--initial-branch", "main")
    git(root, "config", "user.name", "Freeze Test")
    git(root, "config", "user.email", "freeze@example.org")
    git(root, "config", "commit.gpgSign", "false")
    git(root, "config", "tag.gpgSign", "false")
    (root / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "lofi"\n', encoding="utf-8", newline="\n"
    )
    baselines = root / "tests" / "benchmarks" / "baselines" / "Linux-CPython-3.14"
    baselines.mkdir(parents=True)
    (baselines / "0001_eval.json").write_text(
        '{"benchmarks": []}\n', encoding="utf-8", newline="\n"
    )
    (baselines / "0002_probe.json").write_text(
        '{"benchmarks": [1]}\n', encoding="utf-8", newline="\n"
    )
    (baselines / "0003_scratch.local").write_text(
        "ignored, never frozen\n", encoding="utf-8", newline="\n"
    )
    (root / ".gitignore").write_text("*.local\n", encoding="utf-8", newline="\n")
    fixtures = root / "tests" / "benchmarks" / "fixtures"
    fixtures.mkdir(parents=True)
    shutil.copyfile(
        TREE / "tests" / "benchmarks" / "fixtures" / "chart_200.fingerprint",
        fixtures / "chart_200.fingerprint",
    )
    write_experiment(root, an_experiment_document(), name="headline")
    analysis = an_experiment_document()
    analysis["role"] = "analysis"
    analysis["preregistration"] = "analysis-preregistration.json"
    analysis["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1, 3]}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    write_experiment(root, analysis, name="analysis")
    train_the_synthetic_parser(root)
    (root / "experiments" / "analysis-preregistration.json").write_text(
        json.dumps(ANALYSIS_PREREGISTRATION), encoding="utf-8", newline="\n"
    )
    record_price(root, "headline")
    record_price(root, "analysis")
    commit_all(root, "Gate commit")

    return root


def arguments(
    root: Path, readiness: Path, *, number: int = 1, experiment: str = "headline"
) -> list[str]:
    return [
        "--root",
        str(root),
        "--number",
        str(number),
        "--experiment",
        f"experiments/{experiment}.yaml",
        "--experiment",
        "experiments/analysis.yaml",
        "--readiness",
        str(readiness),
    ]


@pytest.fixture
def readiness(tmp_path: Path) -> Path:
    path = tmp_path / "readiness.txt"
    path.write_text(f"{READINESS}\n", encoding="utf-8", newline="\n")

    return path


def tags(root: Path) -> list[str]:
    return git(root, "tag", "--list").splitlines()


def expected_sweep_id(root: Path, epochs: tuple[int, ...]) -> str:
    selection = Selection(
        config=root / "experiments" / "run.json",
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(root),
    )
    sweep = Sweep(grid=(Axis(path="trainer.epochs", values=epochs),), seeds=(0, 1, 2))
    points = GridSweepExpander(composer(selection)).expand(sweep)
    listing = "\n".join(point.point_id for point in points)

    return sha256(listing.encode()).hexdigest()[:16]


def head_blob_digest(root: Path, path: str) -> str:
    return blake2b(git_bytes(root, "cat-file", "blob", f"HEAD:{path}")).hexdigest()


def head_tree_digest(root: Path, directory: str) -> str:
    lines = []

    for entry in git(root, "ls-tree", "-r", "HEAD", "--", directory).splitlines():
        header, name = entry.split("\t")
        relative = name.removeprefix(f"{directory}/")
        blob = git_bytes(root, "cat-file", "blob", header.split()[2])
        lines.append(f"{relative}\0{blake2b(blob).hexdigest()}\n")

    return blake2b("".join(sorted(lines)).encode()).hexdigest()


def preregistration_digest(path: Path) -> str:
    spec = read_preregistration(path.read_text(encoding="utf-8")).to_spec()
    text = json.dumps(spec, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    return blake2b(text.encode()).hexdigest()


def test_a_freeze_records_digests_recomputable_from_the_same_bytes(
    repository: Path, readiness: Path
) -> None:
    head = git(repository, "rev-parse", "HEAD")
    echoed = io.StringIO()

    assert main(arguments(repository, readiness), stdout=echoed) == 0

    written = repository / "freezes" / "1" / "FREEZE.json"
    record = json.loads(written.read_text(encoding="utf-8"))
    experiments = repository / "experiments"
    fixture = "tests/benchmarks/fixtures/chart_200.fingerprint"

    assert json.loads(echoed.getvalue()) == record
    assert record["freeze_id"] == "freeze/1"
    assert record["git_tag"] == {
        "name": "freeze/1",
        "commit": head,
        "object": git(repository, "rev-parse", "refs/tags/freeze/1"),
    }
    assert record["lockfile_digest"] == {
        "blake2b": head_blob_digest(repository, "uv.lock"),
        "didactic": version("didactic"),
    }
    assert record["schema_fingerprint"] == json.loads(
        FingerprintIso().forward(fingerprint_run({}))
    )
    assert record["sweep_id"] == {
        "headline": expected_sweep_id(repository, (1, 2)),
        "analysis": expected_sweep_id(repository, (1, 3)),
    }
    assert record["preregistration_digest"] == {
        "headline": preregistration_digest(experiments / "preregistration.json"),
        "analysis": preregistration_digest(
            experiments / "analysis-preregistration.json"
        ),
    }
    assert (
        record["preregistration_digest"]["headline"]
        != record["preregistration_digest"]["analysis"]
    )
    assert record["budget_baselines"] == {
        "baselines": head_tree_digest(repository, "tests/benchmarks/baselines")
    }
    assert record["fixture_fingerprint"] == {
        "blake2b": head_blob_digest(repository, fixture),
        "fingerprint": json.loads((repository / fixture).read_text(encoding="utf-8"))[
            "fingerprint"
        ],
    }
    assert record["checkpoint_schema"] == {
        "fields": sorted(CheckpointManifest.__field_specs__),
        "schema": dx.schema_uri(CheckpointManifest),
    }
    assert record["experiments"] == {
        "headline": {"path": "experiments/headline.yaml", "role": "headline"},
        "analysis": {"path": "experiments/analysis.yaml", "role": "analysis"},
    }
    assert record["readiness_notes"] == READINESS.splitlines()
    assert record["supplied"] == list(SUPPLIED) == ["readiness_notes"]


def test_a_freeze_records_what_training_plans_for_every_point(
    repository: Path, readiness: Path
) -> None:
    main(arguments(repository, readiness), stdout=io.StringIO())
    record = json.loads(
        (repository / "freezes" / "1" / "FREEZE.json").read_text("utf-8")
    )
    codec = RunSpecCodec()

    for name in ("headline", "analysis"):
        experiment = read_experiment(repository / "experiments" / f"{name}.yaml")
        expected: dict[str, JsonValue] = {}

        for point in expand_experiment(experiment, repository):
            blueprint = plan_training(resolve_point(point, codec))
            expected[point.point_id] = {
                "grammar_fingerprint": blueprint.grammar_fingerprint,
                "codec_fingerprint": blueprint.codec_fingerprint,
                "corpus_revision": blueprint.corpus_revision,
                "component_fingerprints": dict(blueprint.component_fingerprints),
            }

        assert record["identities"][name] == expected
        assert len(expected) == 6


def test_the_hand_supplied_identity_options_no_longer_exist(
    repository: Path, readiness: Path
) -> None:
    for option, value in (
        ("--corpus-revision", "ccgbank"),
        ("--codec-shape", "prefix"),
        ("--grammar-fingerprint", "ccg=0f3a"),
        ("--component-fingerprint", "parser=9c1d"),
    ):
        with pytest.raises(SystemExit):
            main([*arguments(repository, readiness), option, value])

    assert tags(repository) == []


def test_a_freeze_creates_one_annotated_tag_and_one_active_row(
    repository: Path, readiness: Path
) -> None:
    head = git(repository, "rev-parse", "HEAD")

    main(arguments(repository, readiness), stdout=io.StringIO())

    tag_object = git(repository, "rev-parse", "refs/tags/freeze/1")
    active = (repository / "freezes" / "ACTIVE.tsv").read_text(encoding="utf-8")

    assert tags(repository) == ["freeze/1"]
    assert git(repository, "cat-file", "-t", "freeze/1") == "tag"
    assert git(repository, "rev-parse", "freeze/1^{commit}") == head
    assert active.splitlines() == [
        "\t".join(ACTIVE_HEADER),
        f"freeze/1\t1\t{head}\t{tag_object}",
    ]


def test_a_second_freeze_appends_one_row_under_the_header(
    repository: Path, readiness: Path
) -> None:
    main(arguments(repository, readiness), stdout=io.StringIO())
    commit_all(repository, "Record freeze/1")

    for path in (repository / "experiments").glob("*.yaml"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["freeze_id"] = "freeze/2"
        path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    record_price(repository, "headline")
    record_price(repository, "analysis")
    commit_all(repository, "Move the experiments to freeze/2")
    main(arguments(repository, readiness, number=2), stdout=io.StringIO())

    rows = (
        (repository / "freezes" / "ACTIVE.tsv").read_text(encoding="utf-8").splitlines()
    )

    assert sorted(tags(repository)) == ["freeze/1", "freeze/2"]
    assert rows[0] == "\t".join(ACTIVE_HEADER)
    assert [row.split("\t")[0] for row in rows[1:]] == ["freeze/1", "freeze/2"]


def test_a_dirty_tree_is_refused_without_tagging(
    repository: Path, readiness: Path
) -> None:
    (repository / "notes.txt").write_text(
        "uncommitted\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(FreezeError, match="not clean"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_a_tracked_pilot_result_is_refused_by_name_without_tagging(
    repository: Path, readiness: Path
) -> None:
    pilot = repository / "pilot" / "results" / "headline"
    pilot.mkdir(parents=True)
    (pilot / "STATUS.json").write_text(
        '{"freeze_id": "unfrozen"}\n', encoding="utf-8", newline="\n"
    )
    commit_all(repository, "Track a pilot's results")

    with pytest.raises(PilotResultsError, match=r"pilot/results/headline/STATUS\.json"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_an_existing_freeze_number_is_refused(
    repository: Path, readiness: Path
) -> None:
    main(arguments(repository, readiness), stdout=io.StringIO())
    commit_all(repository, "Record freeze/1")
    before = git(repository, "rev-parse", "refs/tags/freeze/1")

    with pytest.raises(FreezeError, match="already tagged"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == ["freeze/1"]
    assert git(repository, "rev-parse", "refs/tags/freeze/1") == before


def test_a_written_freeze_number_is_refused_even_untagged(
    repository: Path, readiness: Path
) -> None:
    record = repository / "freezes" / "1" / "FREEZE.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}\n", encoding="utf-8", newline="\n")
    commit_all(repository, "A stray freeze directory")

    with pytest.raises(FreezeError, match="already exists"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []


def test_an_experiment_for_another_freeze_is_refused(
    repository: Path, readiness: Path
) -> None:
    write_experiment(
        repository, an_experiment_document(freeze_id="freeze/2"), name="revision"
    )
    commit_all(repository, "An experiment against the next freeze")

    with pytest.raises(FreezeError, match="runs against freeze/2"):
        main(
            arguments(repository, readiness, experiment="revision"),
            stdout=io.StringIO(),
        )

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_a_refused_experiment_is_refused_as_a_freeze(
    repository: Path, readiness: Path
) -> None:
    document = an_experiment_document()
    del document["contraction"]
    write_experiment(repository, document, name="broken")
    commit_all(repository, "An experiment without a contraction ladder")

    with pytest.raises(FreezeError, match="contraction"):
        main(
            arguments(repository, readiness, experiment="broken"), stdout=io.StringIO()
        )

    assert tags(repository) == []


def test_readiness_notes_name_each_condition_once_in_order(tmp_path: Path) -> None:
    path = tmp_path / "readiness.txt"
    lines = READINESS.splitlines()

    path.write_text("\n".join(lines[:8]), encoding="utf-8", newline="\n")

    with pytest.raises(FreezeError, match="8 lines"):
        read_readiness(path)

    path.write_text(
        "\n".join([lines[1], lines[0], *lines[2:]]), encoding="utf-8", newline="\n"
    )

    with pytest.raises(FreezeError, match="FG-1"):
        read_readiness(path)

    path.write_text("\n".join(["FG-1:", *lines[1:]]), encoding="utf-8", newline="\n")

    with pytest.raises(FreezeError, match="FG-1"):
        read_readiness(path)

    path.write_text("\n\n".join(lines), encoding="utf-8", newline="\n")

    assert read_readiness(path) == tuple(lines)


def test_a_freeze_listed_as_active_is_refused_even_untagged_and_unwritten(
    repository: Path, readiness: Path
) -> None:
    active = repository / "freezes" / "ACTIVE.tsv"
    active.parent.mkdir()
    active.write_text(
        "\t".join(ACTIVE_HEADER) + "\nfreeze/1\t1\tabc\tdef\n",
        encoding="utf-8",
        newline="\n",
    )
    commit_all(repository, "A stray active row")

    with pytest.raises(FreezeError, match="already listed"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes" / "1").exists()


def test_an_experiment_outside_the_repository_is_refused(
    repository: Path, readiness: Path, tmp_path: Path
) -> None:
    write_experiment(tmp_path, an_experiment_document(), name="outside")
    argv = arguments(repository, readiness)
    argv[argv.index("experiments/headline.yaml")] = "../experiments/outside.yaml"

    with pytest.raises(FreezeError, match="outside the repository"):
        main(argv, stdout=io.StringIO())

    assert tags(repository) == []


def test_two_experiments_of_one_name_are_refused(
    repository: Path, readiness: Path
) -> None:
    with pytest.raises(FreezeError, match="two experiments are named 'analysis'"):
        main(
            arguments(repository, readiness, experiment="analysis"),
            stdout=io.StringIO(),
        )

    assert tags(repository) == []


@pytest.mark.parametrize("number", ["0", "-1"])
def test_a_number_that_forms_no_freeze_identifier_is_refused(
    repository: Path, readiness: Path, number: str
) -> None:
    argv = arguments(repository, readiness)
    argv[argv.index("--number") + 1] = number

    with pytest.raises(FreezeError, match="freeze/<n>"):
        main(argv, stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


@pytest.mark.parametrize(
    "ignored", ["headline.yaml", "run.json", "preregistration.json"]
)
def test_an_experiment_reading_an_untracked_document_is_refused(
    repository: Path, readiness: Path, ignored: str
) -> None:
    git(repository, "rm", "--quiet", "--cached", f"experiments/{ignored}")
    (repository / ".gitignore").write_text(
        f"*.local\nexperiments/{ignored}\n", encoding="utf-8", newline="\n"
    )
    commit_all(repository, f"Stop tracking {ignored}")

    with pytest.raises(FreezeError, match="not all tracked"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_a_tag_that_cannot_be_created_leaves_nothing_written(
    repository: Path, readiness: Path
) -> None:
    active = repository / "freezes" / "ACTIVE.tsv"
    active.parent.mkdir()
    before = "\t".join(ACTIVE_HEADER) + "\nfreeze/7\t7\tabc\tdef\n"
    active.write_text(before, encoding="utf-8", newline="\n")
    commit_all(repository, "Another active freeze")
    locks = repository / ".git" / "refs" / "tags" / "freeze"
    locks.mkdir(parents=True)
    (locks / "1.lock").write_text("", encoding="utf-8", newline="\n")

    with pytest.raises(FreezeError, match="update-ref"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert active.read_text(encoding="utf-8") == before
    assert not (repository / "freezes" / "1").exists()
    assert git(repository, "status", "--porcelain", "--untracked-files=all") == ""


def test_a_record_that_cannot_be_written_is_never_tagged(
    repository: Path, readiness: Path
) -> None:
    blocker = repository / "freezes" / "ACTIVE.tsv" / "blocker"
    blocker.parent.mkdir(parents=True)
    blocker.write_text(
        "ACTIVE.tsv is a directory here\n", encoding="utf-8", newline="\n"
    )
    commit_all(repository, "Block the active list")

    with pytest.raises((IsADirectoryError, PermissionError)):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes" / "1").exists()


def commit_fit_experiment(root: Path, document: dict[str, JsonValue]) -> None:
    write_experiment(root, document, name="fits")
    train_the_synthetic_parser(root)
    commit_all(root, "Add the judgment-fit experiment")


def expected_fit_sweep_id(root: Path, warmups: tuple[int, ...]) -> str:
    loader = SpecLoader(search_path=config_search_path(root))
    fit = root / "experiments" / "fit.json"
    listing = "\n".join(
        point_id(
            json.loads(
                loader.compose_traced(
                    fit,
                    schema=JudgmentFitSpec,
                    overrides=(f"inference.warmup={warmup}",),
                ).spec.model_dump_json()
            ),
            seed=seed,
        )
        for warmup in warmups
        for seed in (0, 1, 2)
    )

    return sha256(listing.encode()).hexdigest()[:16]


def test_a_judgment_fit_experiment_freezes_through_the_same_route(
    repository: Path, readiness: Path
) -> None:
    commit_fit_experiment(repository, a_judgment_experiment_document())

    assert (
        main(arguments(repository, readiness, experiment="fits"), stdout=io.StringIO())
        == 0
    )

    written = repository / "freezes" / "1" / "FREEZE.json"
    record = json.loads(written.read_text(encoding="utf-8"))

    assert record["sweep_id"] == {
        "fits": expected_fit_sweep_id(repository, (100, 200)),
        "analysis": expected_sweep_id(repository, (1, 3)),
    }
    assert record["experiments"]["fits"] == {
        "path": "experiments/fits.yaml",
        "role": "headline",
    }
    assert tags(repository) == ["freeze/1"]
    assert record["compute"]["fits"] == {
        "p50_hours": 12.5,
        "p90_hours": 20,
        "cost_model": "price-v3",
        "source": "supplied",
    }
    assert record["compute"]["analysis"]["source"] == (
        f"sweeps/{expected_sweep_id(repository, (1, 3))}/price.json"
    )


def test_an_unpriced_judgment_fit_experiment_is_refused(
    repository: Path, readiness: Path
) -> None:
    document = a_judgment_experiment_document()
    document["compute"] = {"kind": "unpriced"}
    commit_fit_experiment(repository, document)

    with pytest.raises(FreezeError, match=r"fits\.yaml is unpriced"):
        main(arguments(repository, readiness, experiment="fits"), stdout=io.StringIO())

    assert tags(repository) == []


def test_a_judgment_fit_experiment_reading_an_untracked_fit_is_refused(
    repository: Path, readiness: Path
) -> None:
    commit_fit_experiment(repository, a_judgment_experiment_document())
    git(repository, "rm", "--quiet", "--cached", "experiments/fit.json")
    (repository / ".gitignore").write_text(
        "*.local\nexperiments/fit.json\n", encoding="utf-8", newline="\n"
    )
    commit_all(repository, "Stop tracking the fit")

    with pytest.raises(FreezeError, match="not all tracked"):
        main(arguments(repository, readiness, experiment="fits"), stdout=io.StringIO())

    assert tags(repository) == []


def test_an_experiment_whose_configuration_is_not_its_declared_kind_is_refused(
    repository: Path, readiness: Path
) -> None:
    document = a_judgment_experiment_document()
    document["config"] = {"kind": "judgment-fit", "path": "run.json"}
    commit_fit_experiment(repository, document)

    with pytest.raises(FreezeError, match="not a judgment-fit document"):
        main(arguments(repository, readiness, experiment="fits"), stdout=io.StringIO())


def test_an_unpriced_experiment_is_refused(repository: Path, readiness: Path) -> None:
    path = repository / "experiments" / "headline.yaml"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["compute"] = {"kind": "unpriced"}
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    commit_all(repository, "Withdraw the headline estimate")

    with pytest.raises(FreezeError, match=r"headline\.yaml is unpriced"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_an_experiment_whose_price_record_is_untracked_is_refused(
    repository: Path, readiness: Path
) -> None:
    record = record_price(repository, "headline").relative_to(repository)
    git(repository, "rm", "--quiet", "--cached", record.as_posix())
    (repository / ".gitignore").write_text(
        f"*.local\n{record.as_posix()}\n", encoding="utf-8", newline="\n"
    )
    commit_all(repository, "Stop tracking the headline price")

    assert (repository / record).is_file()

    with pytest.raises(FreezeError, match=r"price record .* is not tracked"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []


def test_an_experiment_without_a_price_record_is_refused(
    repository: Path, readiness: Path
) -> None:
    record = record_price(repository, "headline").relative_to(repository)
    git(repository, "rm", "--quiet", record.as_posix())
    commit_all(repository, "Delete the headline price")

    with pytest.raises(FreezeError, match=r"price record .* is not tracked"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []


@pytest.mark.parametrize(
    ("field", "shift"),
    [("p50_hours", 1e-6), ("p90_hours", 1e-6), ("cost_model", "")],
    ids=("p50", "p90", "cost-model"),
)
def test_a_declared_estimate_the_price_record_does_not_recompute_is_refused(
    repository: Path, readiness: Path, field: str, shift: float | str
) -> None:
    path = repository / "experiments" / "headline.yaml"
    declared = json.loads(path.read_text(encoding="utf-8"))["compute"][field]
    changed: JsonValue = (
        f"{declared}-edited" if isinstance(shift, str) else declared + shift
    )
    declare(repository, "headline", field, changed)
    commit_all(repository, f"Edit the headline {field}")

    with pytest.raises(FreezeError, match=f"{field}.*recomputes"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


def test_a_price_record_copied_from_another_experiment_is_refused(
    repository: Path, readiness: Path
) -> None:
    headline = expected_sweep_id(repository, (1, 2))
    analysis = expected_sweep_id(repository, (1, 3))
    shutil.copyfile(
        repository / SWEEPS_NAME / analysis / "price.json",
        repository / SWEEPS_NAME / headline / "price.json",
    )
    path = repository / "experiments" / "headline.yaml"
    document = json.loads(path.read_text(encoding="utf-8"))
    analysis_path = repository / "experiments" / "analysis.yaml"
    document["compute"] = json.loads(analysis_path.read_text(encoding="utf-8"))[
        "compute"
    ]
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    commit_all(repository, "Copy the analysis price over the headline price")

    with pytest.raises(FreezeError, match=f"sweep_id '{analysis}'.*'{headline}'"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []
    assert not (repository / "freezes").exists()


@pytest.mark.parametrize(
    ("field", "recorded"),
    [
        ("sweep_id", "fedcba9876543210"),
        ("experiment", "analysis"),
        ("freeze_id", "freeze/2"),
    ],
    ids=("sweep", "experiment", "freeze"),
)
def test_a_price_record_naming_another_identity_is_refused(
    repository: Path, readiness: Path, field: str, recorded: str
) -> None:
    record = repository / SWEEPS_NAME / expected_sweep_id(repository, (1, 2))
    price = json.loads((record / "price.json").read_text(encoding="utf-8"))
    price[field] = recorded
    (record / "price.json").write_text(
        json.dumps(price), encoding="utf-8", newline="\n"
    )
    commit_all(repository, f"Record another {field} in the headline price")

    with pytest.raises(FreezeError, match=f"{field} '{recorded}'"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []


def test_a_supplied_estimate_claiming_the_measured_cost_model_is_refused(
    repository: Path, readiness: Path
) -> None:
    document = a_judgment_experiment_document()
    document["compute"] = {
        "kind": "estimate",
        "p50_hours": 1.0,
        "p90_hours": 2.0,
        "cost_model": COST_MODEL,
    }
    commit_fit_experiment(repository, document)

    with pytest.raises(FreezeError, match="measured cost model"):
        main(arguments(repository, readiness, experiment="fits"), stdout=io.StringIO())

    assert tags(repository) == []


def test_a_run_experiment_whose_points_training_cannot_plan_is_refused(
    repository: Path, readiness: Path
) -> None:
    write_experiment(repository, an_experiment_document(), name="unplanned")
    record_price(repository, "headline")
    record_price(repository, "analysis")
    commit_all(repository, "A run document no trainer plans")

    with pytest.raises(FreezeError, match="cannot be planned for training"):
        main(arguments(repository, readiness), stdout=io.StringIO())

    assert tags(repository) == []


def test_a_freeze_records_the_identities_the_configuration_kind_reports(
    repository: Path, readiness: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reported: dict[str, JsonValue] = {"0123456789abcdef": {"declared": "by the kind"}}

    def identities(
        configuration: object, points: Sequence[SweepPoint], /
    ) -> dict[str, JsonValue]:
        del configuration, points
        return reported

    configuration = read_experiment(repository / "experiments" / "headline.yaml").config
    monkeypatch.setattr(type(configuration), "identities", identities)

    assert main(arguments(repository, readiness), stdout=io.StringIO()) == 0

    record = json.loads(
        (repository / "freezes" / "1" / "FREEZE.json").read_text(encoding="utf-8")
    )

    assert record["identities"] == {"headline": reported, "analysis": reported}
