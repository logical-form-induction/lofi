import json
import shutil
from datetime import date
from hashlib import sha256
from statistics import fmean
from typing import TYPE_CHECKING, Final, NamedTuple

import pyarrow.parquet as pq
import pytest
from scripts.judgment.agreement_ceiling import record as ceiling_record
from scripts.judgment.morcela_pilot import (
    RULE,
    CeilingRecord,
    Correlation,
    Outcome,
    decide,
    main,
    read_report,
)
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    write_experiment,
)

from lofi.judgment import (
    LINEAR,
    ComparisonSpec,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    DesignProvenance,
    FixedDiscrimination,
    InferenceMethod,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    MapInference,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    PriorBlock,
    RandomIntercept,
    ReadDesign,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    VariationalInference,
    linear_steps,
    rank_correlation,
    split_half_ceiling,
)
from lofi.judgment.morcela import FAMILY, FREQUENCY, LENGTH, morcela_steps
from lofi.run import (
    ExecutionEnvironment,
    ExperimentSeed,
    ExperimentStore,
    ManifestStatus,
    RunStore,
    SweepPoint,
    installed_packages,
    read_experiment,
)
from lofi.run.fits import HELD_OUT_PREDICTIVE_METRIC, FitExecutor
from lofi.run.results import MANIFESTS_NAME, ROWS_NAME, write_results
from lofi.spec import HeldOutParticipantPolicy
from lofi.spec.optimizers.adam import AdamOptimizerSpec

if TYPE_CHECKING:
    from pathlib import Path

    import didactic.api as dx

    from lofi._types import JsonValue
    from lofi.judgment import Predictor


def write_ceiling(responses: Path, destination: Path, /) -> None:
    """Write the ceiling record of the responses, grouped by item."""
    groups: dict[str, list[float]] = {}
    for line in responses.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        groups.setdefault(fields[3], []).append(float(fields[5]))
    ceiling = split_half_ceiling(groups.values(), splits=10, level=0.95)
    written = ceiling_record(
        ceiling,
        "pilot",
        ("verb",),
        {responses.name: sha256(responses.read_bytes()).hexdigest()},
        splits=10,
        level=0.95,
    )
    destination.write_text(json.dumps(written), encoding="utf-8", newline="\n")


LM_LOG_PROBABILITY: Final = "lm_log_probability"
UNIGRAM_LOG_PROBABILITY: Final = "unigram_log_probability"
SENTENCE_LENGTH: Final = "length"


LEVELS: Final = 7
ITEMS: Final = 12
DATASET: Final = "morcela-pilot"
FREEZE: Final = "freeze/1"
REVISION: Final = "0123abcd"
CANDIDATES: Final = ("morcela", "linear")
RESPONSES: Final = "releases/responses.tsv"
OFFSETS: Final = {"a": 0, "b": -1, "shared": 1}


def participants_of(item: int) -> tuple[str, ...]:
    return (f"a-{item % 2}", f"b-{item % 2}", "shared")


def level_of(participant: str, item: int) -> int:
    base = round((LEVELS - 1) * (ITEMS - 1 - item) / (ITEMS - 1))
    jitter = (item * (len(participant) + 1)) % 3 - 1
    shifted = base + OFFSETS[participant.split("-", maxsplit=1)[0]] + jitter

    return max(0, min(LEVELS - 1, shifted))


def responses_tsv() -> bytes:
    lines = [
        (
            "participant\tlist\tpresentationorder\tverb\tframe\tresponse\t"
            "nativeenglish\tsentence"
        ),
        *(
            f"{participant}\tl1\t{item}\tverb{item}\tNP_V\t"
            f"{level_of(participant, item) + 1}\tTrue\tSomeone verb{item}."
            for item in range(ITEMS)
            for participant in participants_of(item)
        ),
    ]

    return ("\n".join(lines) + "\n").encode("utf-8")


class PilotDesigns:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(
            LEVELS,
            factors=("list",),
            readouts=(LM_LOG_PROBABILITY, UNIGRAM_LOG_PROBABILITY, SENTENCE_LENGTH),
        )
        design = codec.encode(
            JudgmentRow(
                participant,
                f"item-{item}",
                level_of(participant, item),
                RowColumns(
                    factors={"list": f"list-{item % 2}"},
                    readouts={
                        LM_LOG_PROBABILITY: -10.0 - 3.0 * item,
                        UNIGRAM_LOG_PROBABILITY: -20.0 - 2.0 * item,
                        SENTENCE_LENGTH: 3.0 + item % 4,
                    },
                ),
            )
            for item in range(ITEMS)
            for participant in participants_of(item)
        )
        provenance = DesignProvenance(
            tag=dataset,
            checkpoint="fixture/causal-lm@" + "0" * 40,
            checkpoint_dtype="float32",
            unigram_digest=None,
            corpus_digests={RESPONSES: self._digest},
            seed=None,
            participants=None,
            items=None,
            levels=None,
            population_digest=None,
            producer_run=None,
            producer_point=None,
        )

        return ReadDesign(design, provenance)


def dumped(model: dx.Model) -> JsonValue:
    return json.loads(model.model_dump_json())


def a_fit(
    candidate: str, predictor: tuple[Predictor, ...], *, posterior: bool = False
) -> dict[str, JsonValue]:
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=predictor,
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    comparison = ComparisonSpec(
        held_out_unit="list",
        readout_units={
            LM_LOG_PROBABILITY: "item",
            UNIGRAM_LOG_PROBABILITY: "item",
            SENTENCE_LENGTH: "item",
        },
        splitter="leave-one-group-out",
        folds=None,
        seed=23,
        coverage_threshold=0.0,
        unseen_participant=HeldOutParticipantPolicy.PLUG_IN,
        models=ModelComparisonSpec(
            candidates=CANDIDATES,
            candidate=candidate,
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )

    return {
        "dataset": DATASET,
        "response": {"ResponseDeclaration_responsespec": dumped(response)},
        "inference": dumped(
            VariationalInference(
                guide=InferenceMethod.MEAN_FIELD,
                iterations=20,
                draws=100,
                learning_rate=0.05,
            )
            if posterior
            else MapInference(
                optimizer=AdamOptimizerSpec(learning_rate=0.05),
                steps=20,
                initial_scale=0.1,
            )
        ),
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": dumped(comparison),
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }


MORCELA: Final = LinkTerm(
    readout=LM_LOG_PROBABILITY,
    linking=LinkingSpec(
        family=FAMILY,
        columns={LENGTH: SENTENCE_LENGTH, FREQUENCY: UNIGRAM_LOG_PROBABILITY},
    ),
)


LINEAR_TERMS: Final = tuple(
    LinkTerm(readout=readout, linking=LinkingSpec(family=LINEAR))
    for readout in (LM_LOG_PROBABILITY, UNIGRAM_LOG_PROBABILITY, SENTENCE_LENGTH)
)


class Pilot(NamedTuple):
    root: Path
    results: Path
    ceiling: Path
    responses: Path


def executed(root: Path, *, posterior: bool) -> Pilot:
    responses = root / "responses.tsv"
    responses.write_bytes(responses_tsv())
    ceiling = root / "ceiling.json"
    write_ceiling(responses, ceiling)

    document = a_judgment_experiment_document()
    document["seed_policy"] = {"root": 7}
    document["freeze_id"] = FREEZE
    experiment = read_experiment(
        write_experiment(root / "repository", document, name="pilot")
    )
    store = RunStore(root / "store", freeze_id=FREEZE, seed=ExperimentSeed(7))
    executor = FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=PilotDesigns(sha256(responses.read_bytes()).hexdigest()),
        compiler=ResponseCompiler({FAMILY: morcela_steps, LINEAR: linear_steps}),
        environment=ExecutionEnvironment(installed_packages(), REVISION, "cpu"),
        infrastructure_attempts=3,
    )
    points = (
        SweepPoint(
            a_fit(
                "morcela",
                (RandomIntercept(group="participant"), MORCELA),
                posterior=posterior,
            ),
            seed=0,
        ),
        SweepPoint(
            a_fit(
                "linear",
                (RandomIntercept(group="participant"), *LINEAR_TERMS),
                posterior=posterior,
            ),
            seed=0,
        ),
    )

    for point in points:
        assert executor(point).status is ManifestStatus.COMPLETED

    results = root / "results"
    write_results(results, points=points, store=executor.store, today=date(2026, 11, 1))

    return Pilot(root, results, ceiling, responses)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory: pytest.TempPathFactory) -> Pilot:
    return executed(tmp_path_factory.mktemp("pilot"), posterior=False)


@pytest.fixture(scope="module")
def posterior_pilot(tmp_path_factory: pytest.TempPathFactory) -> Pilot:
    return executed(tmp_path_factory.mktemp("posterior"), posterior=True)


def run(pilot: Pilot, out: Path, *extra: str) -> int:
    return main(
        [
            str(pilot.results),
            "--table",
            ROWS_NAME,
            "--candidate",
            "morcela",
            "--ceiling",
            str(pilot.ceiling),
            "--seed",
            "5",
            "--replicates",
            "199",
            "--out",
            str(out),
            *extra,
        ]
    )


def rows_of(pilot: Pilot) -> list[dict[str, JsonValue]]:
    return pq.read_table(pilot.results / ROWS_NAME).to_pylist()


def test_the_report_carries_the_correlation_computed_from_the_recorded_predictions(
    pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"

    assert run(pilot, out) == 0

    report = read_report(out)
    rows = rows_of(pilot)
    predicted: dict[str, list[tuple[float, float]]] = {}

    for row in rows:
        if row["metric"] == "response-prediction" and row["point_id"] == (
            report.point_id
        ):
            distribution = row["distribution"]
            response = row["response"]
            item = row["item"]
            assert isinstance(distribution, list)
            assert isinstance(response, int)
            assert isinstance(item, str)
            expected = sum(
                level * float(str(probability))
                for level, probability in enumerate(distribution)
            )
            predicted.setdefault(item, []).append((expected, float(response)))

    means = [
        (fmean(p for p, _ in pairs), fmean(o for _, o in pairs))
        for pairs in predicted.values()
    ]

    assert report.correlation.items == ITEMS == len(means)
    assert report.correlation.value == pytest.approx(
        rank_correlation([p for p, _ in means], [o for _, o in means])
    )
    assert report.correlation.lower <= report.correlation.upper
    assert (report.correlation.replicates, report.correlation.seed) == (199, 5)
    assert report.candidate == "morcela"
    assert report.git_revision == REVISION
    assert report.dataset == DATASET


def test_the_predictive_density_is_the_candidates_held_out_density_by_regime(
    pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"

    assert run(pilot, out) == 0

    report = read_report(out)
    totals: dict[str, tuple[float, int]] = {}

    for row in rows_of(pilot):
        if (
            row["metric"] == HELD_OUT_PREDICTIVE_METRIC
            and row["candidate"] == "morcela"
        ):
            regime = row["participant_regime"]
            value = row["value"]
            n = row["n"]
            assert isinstance(regime, str)
            assert isinstance(value, float)
            assert isinstance(n, int)
            total, count = totals.get(regime, (0.0, 0))
            totals[regime] = (total + value * n, count + n)

    assert totals
    assert {
        density.regime: (density.total, density.observations)
        for density in report.densities
    } == pytest.approx(totals)
    assert {density.metric for density in report.densities} == {
        HELD_OUT_PREDICTIVE_METRIC
    }


def test_a_posterior_comparison_reports_the_candidates_leave_one_out_density(
    posterior_pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"

    assert run(posterior_pilot, out) == 0

    report = read_report(out)
    [summary] = [
        row
        for row in rows_of(posterior_pilot)
        if row["metric"] == "psis-loo-elpd" and row["model"] == "morcela"
    ]

    assert [
        (density.metric, density.regime, density.total, density.observations)
        for density in report.densities
    ] == [("psis-loo-elpd", None, summary["elpd"], summary["observations"])]
    assert report.correlation.items == ITEMS
    assert report.outcome is decide(report.correlation, report.ceiling)


def test_the_report_carries_the_ceiling_its_script_computed_and_the_rule_applied(
    pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"

    assert run(pilot, out) == 0

    report = read_report(out)
    recorded = json.loads(pilot.ceiling.read_text(encoding="utf-8"))

    assert report.ceiling.value == pytest.approx(recorded["value"])
    assert report.ceiling.source_digests == recorded["source_digests"]
    assert report.rule == RULE
    assert report.outcome is decide(report.correlation, report.ceiling)


def test_a_hand_edited_outcome_that_contradicts_its_numbers_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"
    assert run(pilot, out) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    other = (
        Outcome.PREDICTION
        if written["outcome"] == Outcome.COMPARISON.value
        else Outcome.COMPARISON
    )
    written["outcome"] = other.value
    out.write_text(json.dumps(written), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="gives"):
        read_report(out)


def test_a_hand_edited_ceiling_that_moves_the_outcome_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    out = tmp_path / "report.json"
    assert run(pilot, out) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    correlation = written["correlation"]
    written["ceiling"]["value"] = (
        correlation["upper"] + 1.0
        if written["outcome"] == Outcome.COMPARISON.value
        else (correlation["lower"] + correlation["upper"]) / 2.0
    )
    out.write_text(json.dumps(written), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="gives"):
        read_report(out)


def test_a_report_stating_another_rule_is_refused(pilot: Pilot, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    assert run(pilot, out) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    written["rule"] = "The paper is a prediction paper."
    out.write_text(json.dumps(written), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="another rule"):
        read_report(out)


def a_ceiling(value: float) -> CeilingRecord:
    return CeilingRecord(
        value=value,
        lower=value - 0.1,
        upper=value + 0.1,
        level=0.95,
        splits=10,
        items=40,
        responses=400,
        coefficient="split-half",
        dataset="pilot",
        group_by=("verb",),
        source_digests={"responses.tsv": "0" * 64},
    )


def a_correlation(lower: float, upper: float) -> Correlation:
    return Correlation(
        coefficient="spearman",
        value=(lower + upper) / 2.0,
        lower=lower,
        upper=upper,
        level=0.95,
        unit="item",
        items=40,
        replicates=999,
        seed=0,
    )


@pytest.mark.parametrize(
    ("lower", "upper", "ceiling", "outcome"),
    [
        (0.6, 0.8, 0.7, Outcome.COMPARISON),
        (0.6, 0.7, 0.7, Outcome.COMPARISON),
        (0.4, 0.6, 0.7, Outcome.PREDICTION),
        (0.75, 0.9, 0.7, Outcome.PREDICTION),
    ],
)
def test_the_rule_reads_comparison_exactly_when_the_interval_covers_the_ceiling(
    lower: float, upper: float, ceiling: float, outcome: Outcome
) -> None:
    assert decide(a_correlation(lower, upper), a_ceiling(ceiling)) is outcome


def test_a_ceiling_computed_from_another_release_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    other = tmp_path / "other.tsv"
    other.write_bytes(
        responses_tsv() + b"extra\tl1\t0\tverb0\tNP_V\t4\tTrue\tSomeone verb0.\n"
    )
    ceiling = tmp_path / "other.json"
    write_ceiling(other, ceiling)

    with pytest.raises(ValueError, match="did not read"):
        run(pilot, tmp_path / "report.json", "--ceiling", str(ceiling))


def test_a_candidate_reading_no_morcela_score_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="reads no score"):
        run(pilot, tmp_path / "report.json", "--candidate", "linear")


def test_a_candidate_no_comparison_declares_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="comparison sets"):
        run(pilot, tmp_path / "report.json", "--candidate", "absent")


def test_a_comparison_member_fitting_another_head_is_refused(
    pilot: Pilot, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    shutil.copytree(pilot.results, results)

    for path in (results / MANIFESTS_NAME).glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        fit = manifest["fit"]

        if fit["comparison"]["models"]["candidate"] == "linear":
            response = fit["response"]["ResponseDeclaration_responsespec"]
            response["identification"]["value"] = 2.0
            path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="another ordinal head"):
        run(pilot._replace(results=results), tmp_path / "report.json")


def test_a_run_writes_only_the_report_it_is_asked_for(
    pilot: Pilot, tmp_path: Path
) -> None:

    out = tmp_path / "pilot" / "report.json"

    assert run(pilot, out) == 0
    assert sorted(path.name for path in tmp_path.rglob("*")) == ["pilot", "report.json"]
