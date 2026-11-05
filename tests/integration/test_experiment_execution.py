import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
from tests.unit.run.test__experiment import PREREGISTRATION, an_experiment_document
from tests.unit.run.test_freezing import READINESS, record_price

import lofi.run._cli as cli_module
import lofi.run._judgment_fit_configuration as judgment_fit_configuration
from lofi.judgment import (
    LINEAR,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    DesignProvenance,
    FixedDiscrimination,
    InferenceMethod,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    Predictor,
    PriorBlock,
    RandomIntercept,
    ReadDesign,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    decode_fit_manifest,
)
from lofi.judgment.compiler import PARTICIPANT
from lofi.run import (
    POINTS_NAME,
    ExperimentStore,
    LaunchReport,
    ManifestStatus,
    RunStatus,
    UnfrozenRevisionError,
    decode_manifest,
    decode_seed_derivation,
    encode_seed_derivation,
    expand_experiment,
    read_experiment,
    write_points,
)
from lofi.run import main as cli_main
from lofi.run._entrypoint import main as lofi_main
from lofi.run.freezing import mint_freeze
from lofi.run.launching import ExperimentLaunching
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.run.results import (
    SEED_BUDGET_SUFFIX,
    STATUS_NAME,
    ingest_results,
    write_seed_budget,
)
from lofi.train import NumericalFailureError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run import (
        HeldOutScoring,
        MetricSink,
        PointRecorder,
        TrainingBlueprint,
        TrainingOperations,
    )
    from lofi.run.results import IngestedResults
    from lofi.train import RestorableRandomStreams

TREE: Final = Path(__file__).resolve().parents[2]
GIT: Final = shutil.which("git") or "git"
CONVENTIONS: Final = (
    "{punctuation: keep, unary_chains: keep, trivial_spans: keep, "
    "minimum_length: 0, aggregation: sentence, coverage: zero, "
    "self_arcs: allow, gold_derivation: canonical}"
)
EVALUATORS: Final = f"""evaluators:
  brackets:
    kind: bracketing
    interval: {{replicates: 50, level: 0.95}}
    punctuation: {{source: none, forms: []}}
    decoder: minimum-risk
    conventions: {CONVENTIONS}
  lexicon:
    kind: lexical-types
    interval: {{replicates: 50, level: 0.95}}
    readout: category-terms
    alignment:
      kind: identity
      fit_on: dev
      score_on: test
      random_floor: {{replicates: 50, level: 0.95}}
      stability: {{replicates: 50, level: 0.95}}
    conventions: {CONVENTIONS}
"""
EVALUATED_METRICS: Final = (
    ("brackets", "unlabeled-bracketing-f1"),
    ("lexicon", "lexical-type-f1"),
    ("lexicon", "primitive-identity-accuracy"),
    ("lexicon", "primitive-random-relabeling-floor"),
    ("lexicon", "constructor-identity-accuracy"),
    ("lexicon", "constructor-random-relabeling-floor"),
)
FREEZE_ID: Final = "freeze/1"
LEVELS: Final = 7
SCORE: Final = "score"
DATASET: Final = "tiny-ratings"


def evaluated_run(
    *,
    size: int,
    grammaticality_only: bool = False,
    second_decoder: bool = False,
) -> str:
    """Return a small overlay on the current packaged synthetic parser run."""
    loss = (
        "loss:\n  terms:\n    - objective: {kind: grammaticality, readout: sentences}\n"
        if grammaticality_only
        else ""
    )
    decoder = (
        "model:\n"
        "  decoders:\n"
        "    secondary: {kind: tree-rnn, width: 8}\n"
        "  spaces:\n"
        "    categories:\n"
        "      combinators:\n"
        "        decoders:\n"
        '          "application:\\\\": secondary\n'
        if second_decoder
        else ""
    )
    return (
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "data:\n"
        "  datasets:\n"
        "    train:\n"
        "      dataset:\n"
        "        retained: {dev: 0.25, test: 0.25}\n"
        f"        size: {size}\n"
        "trainer:\n"
        "  accumulation_steps: 1\n"
        f"{loss}"
        f"{decoder}"
        f"{EVALUATORS}"
    )


def run_experiment_document() -> dict[str, JsonValue]:
    """Return an experiment reporting the two evaluators in ``EVALUATORS``."""
    document = an_experiment_document()
    brackets: dict[str, JsonValue] = {
        "evaluator": "brackets",
        "metric": "unlabeled-bracketing-f1",
    }
    lexicon: dict[str, JsonValue] = {
        "evaluator": "lexicon",
        "metric": "lexical-type-f1",
    }
    document["rows"] = [brackets, lexicon]
    document["primary"] = brackets
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": brackets,
            "statistic": "mean",
            "direction": "above",
            "threshold": 0.5,
        },
        "fallback": "The parser does not recover held-out bracketing structure.",
    }
    evaluation = cast("dict[str, JsonValue]", document["evaluation"])
    seed_budget = cast("dict[str, JsonValue]", evaluation["seed_budget"])
    seed_budget["metric"] = "unlabeled-bracketing-f1"

    return document


def fit_experiment_document() -> dict[str, JsonValue]:
    """Return an experiment reporting the declared fit diagnostics."""
    document = an_experiment_document()
    rhat: dict[str, JsonValue] = {"evaluator": "fit", "metric": "max-rhat"}
    ess: dict[str, JsonValue] = {"evaluator": "fit", "metric": "min-ess"}
    document["rows"] = [rhat, ess]
    document["primary"] = rhat
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": rhat,
            "statistic": "max",
            "direction": "below",
            "threshold": 1.01,
        },
        "fallback": "The judgment model does not converge under this fit.",
    }
    evaluation = cast("dict[str, JsonValue]", document["evaluation"])
    seed_budget = cast("dict[str, JsonValue]", evaluation["seed_budget"])
    seed_budget["metric"] = "max-rhat"

    return document


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        (GIT, "-C", str(root), *arguments), check=True, capture_output=True
    )

    return completed.stdout.decode().strip()


def commit_all(root: Path, message: str) -> None:
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", message)


class Frozen(NamedTuple):
    root: Path
    experiment: Path
    out: Path

    @property
    def freeze(self) -> Path:
        return self.root / "freezes" / "1" / "FREEZE.json"

    @property
    def results(self) -> Path:
        return self.root / "results" / "1" / self.experiment.stem

    def execute(
        self, training: TrainingOperations = TRAINING_OPERATIONS
    ) -> LaunchReport:
        stdout = io.StringIO()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(judgment_fit_configuration, "RegisteredDesigns", TinyDesigns)
            status = ExperimentLaunching(checkout=self.root).launch(
                self.experiment,
                freeze=self.freeze,
                root=self.root,
                overrides=(),
                dry_run=False,
                companions=(),
                training=training,
                environ={},
                out=self.out,
                stdout=stdout,
            )

        assert status == 0

        return LaunchReport.model_validate_json(stdout.getvalue())

    def ingest(self) -> IngestedResults:
        return ingest_results(
            self.results,
            experiment=read_experiment(self.experiment),
            freeze=self.freeze,
            root=self.root,
        )

    def status(self) -> dict[str, JsonValue]:
        return json.loads((self.results / STATUS_NAME).read_text(encoding="utf-8"))

    def points(self) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", self.status()["points"])


class TinyDesigns:
    """Resolves one small crossed rating design."""

    def read(self, dataset: str, /) -> ReadDesign:
        if dataset != DATASET:
            msg = f"no tiny design is named {dataset!r}"
            raise LookupError(msg)

        codec = DesignCodec(LEVELS, factors=(), readouts=(SCORE,))
        design = codec.encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                (participant + 2 * item) % LEVELS,
                RowColumns(factors={}, readouts={SCORE: -1.5 + item}),
            )
            for participant in range(2)
            for item in range(4)
        )
        provenance = DesignProvenance(
            tag=dataset,
            checkpoint=None,
            checkpoint_dtype=None,
            unigram_digest=None,
            corpus_digests={},
            seed=None,
            participants=None,
            items=None,
            levels=None,
            population_digest=None,
            producer_run=None,
            producer_point=None,
        )

        return ReadDesign(design, provenance)


def a_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "--quiet", "--initial-branch", "main")
    git(root, "config", "user.name", "Experiment Test")
    git(root, "config", "user.email", "experiment@example.org")
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
    fixtures = root / "tests" / "benchmarks" / "fixtures"
    fixtures.mkdir(parents=True)
    shutil.copyfile(
        TREE / "tests" / "benchmarks" / "fixtures" / "chart_200.fingerprint",
        fixtures / "chart_200.fingerprint",
    )
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "preregistration.json").write_text(
        json.dumps(PREREGISTRATION), encoding="utf-8", newline="\n"
    )

    return root


def preregister(root: Path, factor: str, levels: list[JsonValue], /) -> None:
    """Write the exact factor grid one execution fixture sweeps."""
    document = {**PREREGISTRATION, "grid": {factor: levels}}
    (root / "experiments" / "preregistration.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )


def freeze(root: Path, experiment: Path, tmp_path: Path) -> Frozen:
    commit_all(root, "Gate commit")
    readiness = tmp_path / "readiness.txt"
    readiness.write_text(f"{READINESS}\n", encoding="utf-8", newline="\n")
    mint_freeze(
        root,
        number=1,
        experiments=(experiment.relative_to(root),),
        readiness=readiness,
    )

    return Frozen(root, experiment, tmp_path / "launch.json")


@pytest.fixture
def frozen(tmp_path: Path) -> Frozen:
    root = a_repository(tmp_path)
    folder = root / "experiments"
    (folder / "run.yaml").write_text(
        evaluated_run(size=16), encoding="utf-8", newline="\n"
    )
    document = run_experiment_document()
    document["config"] = {"kind": "run", "path": "run.yaml"}
    document["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1]}],
        "seeds": [0, 1, 2],
        "expected_points": 3,
    }
    preregister(root, "trainer.epochs", [1])
    experiment = folder / "headline.yaml"
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    record_price(root, "headline")

    return freeze(root, experiment, tmp_path)


@pytest.fixture
def frozen_learning_rates(tmp_path: Path) -> Frozen:
    root = a_repository(tmp_path)
    folder = root / "experiments"
    (folder / "run.yaml").write_text(
        evaluated_run(size=4, grammaticality_only=True), encoding="utf-8", newline="\n"
    )
    document = run_experiment_document()
    document["config"] = {"kind": "run", "path": "run.yaml"}
    document["sweep"] = {
        "grid": [{"path": "optimizer.learning_rate", "values": ["1e+35", 1e-3]}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    preregister(root, "optimizer.learning_rate", ["1e+35", 1e-3])
    experiment = folder / "rates.yaml"
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    record_price(root, "rates")

    return freeze(root, experiment, tmp_path)


@pytest.fixture
def frozen_two_decoders(tmp_path: Path) -> Frozen:
    root = a_repository(tmp_path)
    folder = root / "experiments"
    (folder / "run.yaml").write_text(
        evaluated_run(size=4, second_decoder=True), encoding="utf-8", newline="\n"
    )
    experiment_document = run_experiment_document()
    experiment_document["config"] = {"kind": "run", "path": "run.yaml"}
    experiment_document["sweep"] = {
        "grid": [{"path": "trainer.epochs", "values": [1]}],
        "seeds": [0, 1, 2],
        "expected_points": 3,
    }
    preregister(root, "trainer.epochs", [1])
    experiment = folder / "decoders.yaml"
    experiment.write_text(
        json.dumps(experiment_document), encoding="utf-8", newline="\n"
    )
    record_price(root, "decoders")

    return freeze(root, experiment, tmp_path)


def a_fit_response(*grouped: Predictor) -> JsonValue:
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),
            *grouped,
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    return {"ResponseDeclaration_responsespec": json.loads(response.model_dump_json())}


def freeze_fits(
    tmp_path: Path, fit: dict[str, JsonValue], *, axis: str = "inference.warmup"
) -> Frozen:
    root = a_repository(tmp_path)
    folder = root / "experiments"
    (folder / "fit.json").write_text(json.dumps(fit), encoding="utf-8", newline="\n")
    document = fit_experiment_document()
    document["config"] = {"kind": "judgment-fit", "path": "fit.json"}
    document["sweep"] = {
        "grid": [{"path": axis, "values": [5]}],
        "seeds": [0, 1, 2],
        "expected_points": 3,
    }
    preregister(root, axis, [5])
    document["compute"] = {
        "kind": "estimate",
        "p50_hours": 0.01,
        "p90_hours": 0.02,
        "cost_model": "pi-estimate-v1",
    }
    experiment = folder / "fits.yaml"
    experiment.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return freeze(root, experiment, tmp_path)


@pytest.fixture
def frozen_fits(tmp_path: Path) -> Frozen:
    fit: dict[str, JsonValue] = {
        "dataset": DATASET,
        "response": a_fit_response(),
        "inference": {
            "inference": "nuts",
            "warmup": 5,
            "retained": 8,
            "chains": 2,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }

    return freeze_fits(tmp_path, fit)


@pytest.fixture
def frozen_calibration(tmp_path: Path) -> Frozen:
    fit: dict[str, JsonValue] = {
        "dataset": DATASET,
        "response": a_fit_response(RandomIntercept(group=PARTICIPANT)),
        "inference": {
            "inference": "nuts",
            "warmup": 5,
            "retained": 8,
            "chains": 2,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": {
            "replicates": 2,
            "guide_iterations": 20,
            "guide_learning_rate": 0.05,
        },
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }

    return freeze_fits(tmp_path, fit)


@pytest.fixture
def frozen_variational_fits(tmp_path: Path) -> Frozen:
    fit: dict[str, JsonValue] = {
        "dataset": DATASET,
        "response": a_fit_response(RandomIntercept(group=PARTICIPANT)),
        "inference": {
            "inference": "svi",
            "guide": "structured",
            "iterations": 30,
            "draws": 8,
            "learning_rate": 0.05,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }

    return freeze_fits(tmp_path, fit, axis="inference.draws")


class FailingFirstPoint:
    """Trains through production operations, failing the first point numerically."""

    def __init__(self) -> None:
        self.trained = 0

    def execute(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        self.trained += 1

        if self.trained == 1:
            quantity = "the parser loss in epoch 1"
            raise NumericalFailureError(quantity, None)

        TRAINING_OPERATIONS.execute(
            blueprint,
            directory,
            streams,
            policy,
            scoring=scoring,
            sinks=sinks,
            recorder=recorder,
        )


class UnreachedOperations:
    def execute(
        self,
        blueprint: TrainingBlueprint,
        directory: Path,
        streams: RestorableRandomStreams,
        policy: DevicePolicy,
        /,
        *,
        scoring: HeldOutScoring | None,
        sinks: Sequence[MetricSink],
        recorder: PointRecorder,
    ) -> None:
        del blueprint, directory, streams, policy, scoring, sinks, recorder
        msg = "a completed or failed point was trained again"
        raise AssertionError(msg)


@pytest.mark.integration
def test_a_frozen_experiment_executes_and_its_results_are_ingested(
    frozen: Frozen,
) -> None:
    report = frozen.execute()
    accepted = frozen.ingest()
    points = expand_experiment(read_experiment(frozen.experiment), frozen.root)
    status = frozen.status()

    assert (report.offered, report.launched, report.cache_hits) == (3, 3, 0)
    assert accepted.freeze_id == FREEZE_ID
    assert {row["freeze_id"] for row in accepted.rows} == {FREEZE_ID}
    assert sorted(
        (row["point_id"], row["task"], row["metric"]) for row in accepted.rows
    ) == sorted(
        (point.point_id, task, metric)
        for point in points
        for task, metric in EVALUATED_METRICS
    )
    assert {row["metric"] for row in accepted.rows} == {
        metric for _, metric in EVALUATED_METRICS
    }
    assert all(
        (row["ci_low"] is None) == (row["ci_high"] is None) for row in accepted.rows
    )
    assert all(
        row["ci_low"] is not None
        for row in accepted.rows
        if row["metric"] in {"unlabeled-bracketing-f1", "lexical-type-f1"}
    )
    assert all(
        (row["ci_level"] is None) == (row["ci_low"] is None) for row in accepted.rows
    )
    assert {
        row["ci_level"] for row in accepted.rows if row["ci_level"] is not None
    } == {0.95}
    assert all(
        cast("float", row["ci_low"])
        <= cast("float", row["value"])
        <= cast("float", row["ci_high"])
        for row in accepted.rows
        if row["ci_low"] is not None
    )
    assert accepted.completed_fraction == 1.0
    assert accepted.residuals.rows == len(points) * len(EVALUATED_METRICS)
    assert accepted.residuals.measured == 0
    assert status["points"] == {
        point.point_id: {"status": "completed"} for point in points
    }
    assert sorted(path.stem for path in (frozen.results / "manifests").iterdir()) == (
        sorted(point.point_id for point in points)
    )

    for point in points:
        manifest = decode_manifest(
            (frozen.results / "manifests" / f"{point.point_id}.json").read_bytes()
        )

        assert manifest.status is ManifestStatus.COMPLETED
        assert manifest.git_revision == git(
            frozen.root, "rev-parse", f"{FREEZE_ID}^{{commit}}"
        )
        assert manifest.seed_derivation == encode_seed_derivation(
            read_experiment(frozen.experiment).seed_policy.point(
                point.point_id, point.seed
            )
        )


@pytest.mark.integration
def test_re_executing_a_completed_experiment_is_a_cache_hit(frozen: Frozen) -> None:
    frozen.execute()
    first = frozen.ingest()
    again = frozen.execute(UnreachedOperations())

    assert (again.offered, again.launched, again.cache_hits) == (3, 0, 3)
    assert frozen.ingest().rows == first.rows


@pytest.mark.integration
def test_a_task_trains_its_points_under_the_root_its_experiment_declares(
    frozen: Frozen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = read_experiment(frozen.experiment)
    point = expand_experiment(experiment, frozen.root)[0]
    store = ExperimentStore(tmp_path / "scratch", experiment)
    listing = tmp_path / POINTS_NAME
    write_points(listing, (point,), identity=store.identity)
    monkeypatch.setattr(cli_module, "package_checkout", lambda: frozen.root)

    status = lofi_main(
        [
            "task",
            str(listing),
            "0",
            "1",
            "--experiment",
            str(frozen.experiment.relative_to(frozen.root)),
            "--root",
            str(store.root),
        ]
    )
    recorded = store.manifest(point)

    assert status == 0
    assert recorded is not None

    manifest = decode_manifest(recorded)

    assert manifest.status is ManifestStatus.COMPLETED
    assert manifest.seed_derivation == encode_seed_derivation(
        experiment.seed_policy.point(point.point_id, point.seed)
    )


@pytest.mark.integration
def test_an_injected_numerical_failure_is_a_recorded_outcome(frozen: Frozen) -> None:
    failing = FailingFirstPoint()
    points = expand_experiment(read_experiment(frozen.experiment), frozen.root)
    failed = points[0].point_id

    report = frozen.execute(failing)
    accepted = frozen.ingest()
    recorded = frozen.points()

    assert failing.trained == len(points)
    assert recorded[failed] == {
        "status": "numerical-failure",
        "diagnostic": "the parser loss in epoch 1 is not finite",
        "attempts": 1,
    }
    assert {row["point_id"] for row in accepted.rows} == {
        point.point_id for point in points
    } - {failed}
    assert (frozen.results / "manifests" / f"{failed}.json").is_file()
    assert [outcome.status.value for outcome in report.outcomes].count("failed") == 1

    frozen.execute(UnreachedOperations())

    assert frozen.points()[failed] == recorded[failed]


@pytest.mark.integration
def test_a_diverging_point_is_recorded_numerical_and_the_next_point_completes(
    frozen_learning_rates: Frozen,
) -> None:
    points = expand_experiment(
        read_experiment(frozen_learning_rates.experiment), frozen_learning_rates.root
    )
    diverging, converging = list(points[:3]), list(points[3:])
    rates = {
        point.point_id: cast("dict[str, JsonValue]", point.config["optimizer"])[
            "learning_rate"
        ]
        for point in points
    }

    report = frozen_learning_rates.execute()
    recorded = frozen_learning_rates.points()
    outcomes = {outcome.point_id: outcome.status for outcome in report.outcomes}

    assert [rates[point.point_id] for point in points] == [1e35] * 3 + [1e-3] * 3
    assert {outcomes[point.point_id] for point in diverging} == {RunStatus.FAILED}
    assert {outcomes[point.point_id] for point in converging} == {RunStatus.COMPLETED}
    assert all(
        recorded[point.point_id] == {"status": "completed"} for point in converging
    )

    for point in diverging:
        failure = cast("dict[str, JsonValue]", recorded[point.point_id])

        assert (failure["status"], failure["attempts"]) == ("numerical-failure", 1)
        assert "is not finite" in str(failure["diagnostic"])

    assert frozen_learning_rates.status()["counts"] == {
        "numerical-failure": 3,
        "completed": 3,
    }


@pytest.mark.integration
def test_a_freeze_records_the_identities_its_trained_manifests_record(
    frozen_two_decoders: Frozen,
) -> None:
    frozen_two_decoders.execute()
    record = json.loads(frozen_two_decoders.freeze.read_text(encoding="utf-8"))
    identities = record["identities"]["decoders"]
    manifests = frozen_two_decoders.results / "manifests"

    assert sorted(identities) == sorted(path.stem for path in manifests.iterdir())

    for point, frozen in identities.items():
        manifest = decode_manifest((manifests / f"{point}.json").read_bytes())

        assert manifest.status is ManifestStatus.COMPLETED
        assert frozen["component_fingerprints"] == manifest.component_fingerprints
        assert frozen["grammar_fingerprint"] == manifest.grammar_fingerprint
        assert frozen["corpus_revision"] == manifest.corpus_revision
        assert {
            "model.decoders.application",
            "model.decoders.secondary",
        } <= set(manifest.component_fingerprints)


@pytest.mark.integration
def test_executing_from_a_head_other_than_the_freeze_tag_is_refused(
    frozen: Frozen,
) -> None:
    (frozen.root / "NOTES.md").write_text(
        "a later edit\n", encoding="utf-8", newline="\n"
    )
    commit_all(frozen.root, "Move past the freeze")

    with pytest.raises(UnfrozenRevisionError, match=FREEZE_ID):
        frozen.execute()

    assert not (frozen.root / FREEZE_ID).exists()
    assert not frozen.results.exists()
    assert not frozen.out.exists()


@pytest.mark.integration
def test_executing_over_a_modified_tracked_file_is_refused(frozen: Frozen) -> None:
    run = frozen.root / "experiments" / "run.yaml"
    run.write_text(
        run.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(UnfrozenRevisionError, match=r"run\.yaml"):
        frozen.execute()

    assert not (frozen.root / FREEZE_ID).exists()


@pytest.mark.integration
def test_a_judgment_fit_experiment_executes_freezes_and_ingests(
    frozen_fits: Frozen,
) -> None:
    experiment = read_experiment(frozen_fits.experiment)
    points = expand_experiment(experiment, frozen_fits.root)

    report = frozen_fits.execute()
    accepted = frozen_fits.ingest()
    again = frozen_fits.execute()

    assert (report.offered, report.launched, report.cache_hits) == (3, 3, 0)
    assert (again.launched, again.cache_hits) == (0, 3)
    assert {row["point_id"] for row in accepted.rows} == {
        point.point_id for point in points
    }
    assert {row["metric"] for row in accepted.rows} == {
        "divergences",
        "max-rhat",
        "min-ess",
        "response-prediction",
        "split-half-ceiling",
    }
    assert frozen_fits.status()["points"] == {
        point.point_id: {"status": "completed"} for point in points
    }

    for point in points:
        manifest = decode_fit_manifest(
            (frozen_fits.results / "manifests" / f"{point.point_id}.json").read_bytes()
        )

        assert decode_seed_derivation(
            manifest.seed_derivation
        ) == experiment.seed_policy.point(point.point_id, point.seed)
        assert json.loads(manifest.fit.model_dump_json()) == point.config
        assert manifest.git_revision == git(frozen_fits.root, "rev-parse", "HEAD")


@pytest.mark.integration
def test_a_fit_document_reports_its_declared_interannotator_ceiling(
    tmp_path: Path,
) -> None:
    fit: dict[str, JsonValue] = {
        "dataset": DATASET,
        "response": a_fit_response(),
        "inference": {
            "inference": "nuts",
            "warmup": 5,
            "retained": 8,
            "chains": 2,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": {
            "coefficients": ["spearman"],
            "replicates": 2000,
            "level": 0.95,
        },
    }
    frozen_ceiling = freeze_fits(tmp_path, fit)

    frozen_ceiling.execute()
    accepted = frozen_ceiling.ingest()
    ceilings = {
        row["metric"]
        for row in accepted.rows
        if str(row["metric"]).startswith("interannotator-")
    }

    assert ceilings == {"interannotator-spearman"}


@pytest.mark.integration
def test_an_executed_pilot_prices_the_seed_count_of_an_experiment(
    frozen: Frozen,
) -> None:
    frozen.execute()
    document = json.loads(frozen.experiment.read_text(encoding="utf-8"))
    document["evaluation"]["seed_budget"]["metric"] = "lexical-type-f1"
    document["evaluation"]["seed_budget"]["precision"] = 0.5
    priced = frozen.experiment.with_name("priced.yaml")
    priced.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    stdout = io.StringIO()

    status = cli_main(
        ["seed-budget", "--experiment", str(priced), "--pilot", str(frozen.results)],
        seed_budget=write_seed_budget,
        stdout=stdout,
    )
    record = json.loads(
        priced.with_name(f"priced{SEED_BUDGET_SUFFIX}").read_text(encoding="utf-8")
    )

    assert status == 0
    assert json.loads(stdout.getvalue()) == record
    assert record["metric"] == "lexical-type-f1"
    assert record["seed_floor"] == 3
    assert record["seeds_required"] >= document["evaluation"]["minimum_seeds"]
    assert record["seeds_required"] == max(record["cells"].values())


@pytest.mark.integration
def test_a_structured_variational_fit_experiment_executes_and_ingests(
    frozen_variational_fits: Frozen,
) -> None:
    experiment = read_experiment(frozen_variational_fits.experiment)
    points = expand_experiment(experiment, frozen_variational_fits.root)

    report = frozen_variational_fits.execute()
    accepted = frozen_variational_fits.ingest()

    assert (report.offered, report.launched, report.cache_hits) == (3, 3, 0)
    assert {row["point_id"] for row in accepted.rows} == {
        point.point_id for point in points
    }
    assert {(row["metric"], row["n"]) for row in accepted.rows} == {
        ("evidence-bound", 5),
        ("response-prediction", 1),
        ("split-half-ceiling", 8),
    }

    for point in points:
        manifest = decode_fit_manifest(
            (
                frozen_variational_fits.results / "manifests" / f"{point.point_id}.json"
            ).read_bytes()
        )

        assert manifest.fit.inference.method is InferenceMethod.STRUCTURED


@pytest.mark.integration
def test_a_calibrated_judgment_fit_experiment_executes_and_its_rows_ingest(
    frozen_calibration: Frozen,
) -> None:
    experiment = read_experiment(frozen_calibration.experiment)
    points = expand_experiment(experiment, frozen_calibration.root)

    report = frozen_calibration.execute()
    accepted = frozen_calibration.ingest()
    again = frozen_calibration.execute()
    statuses = frozen_calibration.points()
    failed = {
        outcome.point_id
        for outcome in report.outcomes
        if outcome.status is RunStatus.FAILED
    }
    assert len(failed) == 1
    failed_point = next(iter(failed))
    completed = {point.point_id for point in points} - failed
    failure = cast("dict[str, JsonValue]", statuses[failed_point])
    calibrated = [row for row in accepted.rows if row["summary"] is not None]

    assert (report.launched, report.cache_hits) == (3, 0)
    assert (again.launched, again.cache_hits) == (0, 3)
    assert {
        outcome.point_id: outcome.status
        for outcome in again.outcomes
        if outcome.status is not RunStatus.COMPLETED
    } == {failed_point: RunStatus.FAILED}
    assert [statuses[point_id] for point_id in completed] == [
        {"status": "completed"}
    ] * 2
    assert (failure["status"], failure["attempts"]) == ("numerical-failure", 1)
    assert "NUTS cutpoint" in cast("str", failure["diagnostic"])
    assert "does not clear 0.80" in cast("str", failure["diagnostic"])
    assert frozen_calibration.status()["counts"] == {
        "completed": 2,
        "numerical-failure": 1,
    }
    assert {row["point_id"] for row in accepted.rows} == completed
    assert {row["point_id"] for row in calibrated} == completed

    assert {row["metric"] for row in accepted.rows if row["summary"] is None} == {
        "divergences",
        "max-rhat",
        "min-ess",
        "response-prediction",
        "split-half-ceiling",
    }
    assert {row["summary"] for row in calibrated} == {
        "sbc",
        "sbc_rank_bin",
        "coverage",
        "cutpoint_recovery",
    }
    assert {row["method"] for row in calibrated if row["summary"] == "coverage"} == {
        "nuts",
        "structured",
        "mean_field",
    }
    assert {row["replicates"] for row in calibrated} == {2}
    assert {row["budget_kind"] for row in accepted.rows} == {"none"}
    assert accepted.residuals.not_applicable == accepted.residuals.rows
