"""Tests for active-learning runs driven from their declared documents."""

import json
from typing import TYPE_CHECKING, Final

import pytest
from tests.unit.run.test__active_inputs import (
    LABELS,
    annotation_lines,
    encoding_document,
    fixture_generator,
    item_document,
)
from tests.unit.run.test__experiment import PREREGISTRATION, an_experiment_document

from lofi.judgment import (
    LINEAR,
    AcquisitionBatch,
    ActiveLearningResult,
    ActiveRound,
    ConditionalLikelihood,
    Cumulative,
    Elicitation,
    ElicitationMethod,
    FixedDiscrimination,
    JudgmentFitSpec,
    LinkingSpec,
    LinkTerm,
    MapInference,
    NutsInference,
    PriorBlock,
    ResponseSpec,
    ThresholdPriors,
)
from lofi.judgment._spec import (
    ComparisonSpec,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
)
from lofi.registry import UnknownTagError
from lofi.run import (
    ExecutionEnvironment,
    ExperimentStore,
    ManifestStatus,
    PointStatus,
    decode_active_learning_manifest,
    expand_experiment,
    read_experiment,
    read_rows,
)
from lofi.run._active_inputs import read_pool
from lofi.run._boundary.active import (
    RESPONSES_NAME,
    compose_active_learning,
    round_directory,
)
from lofi.run.active import (
    ROWS_NAME,
    ActiveLearning,
    ActiveLearningExecutor,
    ActiveLearningInputs,
    active_learn,
    round_rows,
)
from lofi.spec import HeldOutParticipantPolicy
from lofi.spec.acquisition import PointReadoutSettings, TokenizerSettings
from lofi.spec.optimizers.adam import AdamOptimizerSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from bead.items.item import Item

    from lofi._types import JsonValue

ENVIRONMENT: Final = ExecutionEnvironment({"lofi": "0.1.0"}, "0" * 40, "cpu")
ANNOTATORS: Final = ("annotator-a", "annotator-b", "annotator-c")
CROSS_VALIDATION: Final[dict[str, JsonValue]] = {
    "held_out_unit": "item",
    "splitter": "kfold",
    "folds": 3,
    "coverage_threshold": 0.0,
    "unseen_participant": "plug_in",
}
COMPARISON: Final = ComparisonSpec(
    held_out_unit="item",
    readout_units={"p_gram": "item"},
    splitter="kfold",
    folds=3,
    seed=5,
    coverage_threshold=0.0,
    unseen_participant=HeldOutParticipantPolicy.PLUG_IN,
    models=ModelComparisonSpec(
        candidates=("root-mass", "intercept"),
        candidate="root-mass",
        criterion=ModelComparisonCriterion.PSIS_LOO,
        weighting=ModelWeighting.STACKING,
        pareto_fraction=0.1,
        pareto_threshold=0.7,
    ),
)


WHITESPACE_TOKENIZER: dict[str, JsonValue] = {
    "backend": "whitespace",
    "language": "en",
    "model_name": None,
}


def fit_document(**changed: JsonValue) -> dict[str, JsonValue]:
    """Return a MAP judgment fit reading the root log mass as its predictor."""
    fit = JudgmentFitSpec(
        dataset="collected-acceptability",
        response=ResponseSpec(
            family=Cumulative(),
            levels=len(LABELS),
            objective=ConditionalLikelihood(),
            predictor=(LinkTerm(readout="p_gram", linking=LinkingSpec(family=LINEAR)),),
            identification=FixedDiscrimination(value=1.0),
            priors=PriorBlock(thresholds=ThresholdPriors()),
        ),
        inference=MapInference(
            optimizer=AdamOptimizerSpec(learning_rate=0.1),
            steps=300,
            initial_scale=0.1,
        ),
        elicitation=Elicitation(method=ElicitationMethod.PARTICIPANTS, version="1"),
        calibration=None,
        comparison=None,
        dynamic_range=None,
        ablation=(),
        interannotator=None,
    )
    document = json.loads(fit.model_dump_json())
    document.update(changed)
    return document


def active_document(**changed: JsonValue) -> dict[str, JsonValue]:
    """Return an active-learning document over the fixture files."""
    document: dict[str, JsonValue] = {
        "acquisition": {
            "kind": "rc2.1",
            "batch_size": 2,
            "allocate_by": "item['structure']",
            "constraints": [
                {"constraint_type": "uniqueness", "property_expression": "item['verb']"}
            ],
            "share_tolerance": 0.5,
        },
        "stopping": {"kind": "asymptote", "window": 3, "maximum_gain": 0.0},
        "max_rounds": 2,
        "bootstrap_samples": 20,
        "pool": "items.json",
        "factors": {"structure": "item['structure']", "verb": "item['verb']"},
        "initial": {"kind": "bead", "responses": "initial.jsonl"},
        "encoding": "encoding.json",
        "fit": "fit.json",
        "predictors": {
            "run": "trained",
            "inputs": {"words": "sentence"},
            "tokenizer": WHITESPACE_TOKENIZER,
            "readouts": {"p_gram": {"readout": "sentences", "measure": "log-mass"}},
        },
        "collector": {"kind": "bead"},
        "cross_validation": CROSS_VALIDATION,
    }
    document.update(changed)
    return document


def _order(item: Item, /) -> tuple[str, str]:
    """Return an order interleaving the two structures verb by verb."""
    return (str(item.item_metadata["verb"]), str(item.item_metadata["structure"]))


def spread_predictors(
    run: Path,
    items: Sequence[Item],
    inputs: Mapping[str, str],
    tokenizer: TokenizerSettings,
    readouts: Mapping[str, PointReadoutSettings],
    /,
) -> dict[str, Mapping[str, float]]:
    """Return predictors spread one unit apart in an order interleaving strata."""
    assert run.name == "trained"
    assert dict(inputs) == {"words": "sentence"}
    assert tokenizer.backend == "whitespace"
    order = sorted(items, key=_order)
    return {
        str(item.id): {column: -3.0 + 0.5 * order.index(item) for column in readouts}
        for item in items
    }


INPUTS: Final = ActiveLearningInputs(
    generator_for=fixture_generator, predictors=spread_predictors
)


def _ranked(folder: Path, /) -> tuple[tuple[str, str], ...]:
    """Return every pool item and its structure in ascending order of its predictor."""
    pool = read_pool(folder / "items.json", generator_for=fixture_generator).items
    columns = spread_predictors(
        folder / "trained",
        pool,
        {"words": "sentence"},
        TokenizerSettings("whitespace", "en", None),
        {"p_gram": PointReadoutSettings("sentences", "log-mass")},
    )
    ordered = sorted(pool, key=lambda item: columns[str(item.id)]["p_gram"])
    return tuple(
        (str(item.id), str(item.item_metadata["structure"])) for item in ordered
    )


def responses_of(folder: Path, /) -> dict[str, int]:
    """Return every pool item's response, the tercile of its predictor."""
    ranked = _ranked(folder)
    return {
        item: position * len(LABELS) // len(ranked)
        for position, (item, _) in enumerate(ranked)
    }


def initially_judged(folder: Path, /) -> tuple[str, ...]:
    """Return the two lowest and two highest predictors of each structure."""
    ranked = _ranked(folder)
    judged: list[str] = []
    for structure in ("transitive", "intransitive"):
        members = tuple(item for item, held in ranked if held == structure)
        judged.extend((*members[:2], *members[-2:]))
    return tuple(judged)


def write_inputs(
    folder: Path, /, *, fit: dict[str, JsonValue] | None = None
) -> dict[str, int]:
    """Write every document an active-learning run reads, returning responses."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "items.json").write_text(
        json.dumps(item_document()), encoding="utf-8", newline="\n"
    )
    (folder / "encoding.json").write_text(
        json.dumps(encoding_document()), encoding="utf-8", newline="\n"
    )
    (folder / "fit.json").write_text(
        json.dumps(fit_document() if fit is None else fit),
        encoding="utf-8",
        newline="\n",
    )
    (folder / "trained").mkdir()
    responses = responses_of(folder)
    (folder / "initial.jsonl").write_text(
        annotation_lines(
            {item: responses[item] for item in initially_judged(folder)}, ANNOTATORS
        ),
        encoding="utf-8",
        newline="\n",
    )
    (folder / "active.json").write_text(
        json.dumps(active_document()), encoding="utf-8", newline="\n"
    )
    return responses


def answer(directory: Path, round_index: int, responses: dict[str, int], /) -> None:
    """Ingest every annotator's response to one written batch."""
    folder = round_directory(directory, round_index)
    batch = tuple(
        json.loads(line)["id"]
        for line in (folder / "batch.jsonl").read_text(encoding="utf-8").splitlines()
    )
    (folder / RESPONSES_NAME).write_text(
        annotation_lines({item: responses[item] for item in batch}, ANNOTATORS),
        encoding="utf-8",
        newline="\n",
    )


@pytest.mark.parametrize(
    ("fit", "reason"),
    [
        (
            fit_document(
                inference=json.loads(
                    NutsInference(warmup=5, retained=8, chains=2).model_dump_json()
                )
            ),
            "by MAP",
        ),
        (
            fit_document(comparison=json.loads(COMPARISON.model_dump_json())),
            "comparison",
        ),
        (
            fit_document(
                dynamic_range={
                    "readout": "p_gram",
                    "criterion": {
                        "registered_at": "2026-10-01T00:00:00+00:00",
                        "minimum_interquartile_range": 0.5,
                        "ceiling": 3.0,
                        "maximum_ceiling_fraction": 0.2,
                    },
                }
            ),
            "dynamic_range",
        ),
    ],
)
def test_a_fit_an_active_run_cannot_refit_as_declared_is_refused(
    tmp_path: Path, fit: dict[str, JsonValue], reason: str
) -> None:
    write_inputs(tmp_path, fit=fit)

    with pytest.raises(ValueError, match=reason):
        ActiveLearning(
            compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
        )


def test_a_cross_validation_naming_no_installed_splitter_is_refused(
    tmp_path: Path,
) -> None:
    write_inputs(tmp_path)
    document = active_document(
        cross_validation={**CROSS_VALIDATION, "splitter": "bootstrap"}
    )
    (tmp_path / "active.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )

    with pytest.raises(UnknownTagError, match="bootstrap"):
        ActiveLearning(
            compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
        )


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"splitter": "leave-one-group-out"}, "null"),
        (
            {"held_out_unit": "participant", "folds": 2, "unseen_participant": None},
            "no policy for predicting an unseen participant",
        ),
    ],
)
def test_the_declared_cross_validation_is_the_one_each_round_runs(
    tmp_path: Path, changed: dict[str, JsonValue], reason: str
) -> None:
    folder = tmp_path / "documents"
    write_inputs(folder)
    document = active_document(cross_validation={**CROSS_VALIDATION, **changed})
    (folder / "active.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match=reason):
        active_learn(
            folder / "active.json", 3, tmp_path / "out", ENVIRONMENT, inputs=INPUTS
        )


def test_each_stratum_share_row_records_its_performance_and_floor(
    tmp_path: Path,
) -> None:
    write_inputs(tmp_path)
    learning = ActiveLearning(
        compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
    )
    batch = AcquisitionBatch(
        (),
        {"transitive": 1.0, "intransitive": 0.0},
        0.25,
        performance={"transitive": 0.5, "intransitive": -0.25},
        floored=("intransitive",),
    )
    result = ActiveLearningResult(
        (ActiveRound(1, batch, 0.5, (0.4, 0.6), 0.5, (0.4, 0.6), "max-rounds", 6),),
        (),
        awaiting=None,
        stop_reason="max-rounds",
    )

    rows = round_rows(learning, result, identity={}, seed=0)

    shares = [row for row in rows if row["metric"] == "stratum-share"]
    assert {row["stratum"]: row["floored"] for row in shares} == {
        "transitive": False,
        "intransitive": True,
    }
    assert {row["stratum"]: row["performance"] for row in shares} == {
        "transitive": 0.5,
        "intransitive": -0.25,
    }
    assert all("frame" not in row for row in rows)


def test_predictors_the_fit_does_not_read_are_refused(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    document = active_document(
        predictors={
            "run": "trained",
            "inputs": {"words": "sentence"},
            "tokenizer": WHITESPACE_TOKENIZER,
            "readouts": {
                "p_gram": {"readout": "sentences", "measure": "log-mass"},
                "p_gram_complement": {
                    "readout": "sentences",
                    "measure": "log-complement",
                },
            },
        }
    )
    (tmp_path / "active.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="trained point fills"):
        ActiveLearning(
            compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
        )


def test_an_absent_declared_file_is_refused(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "items.json").unlink()

    with pytest.raises(ValueError, match="item-construction document"):
        ActiveLearning(
            compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
        )


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        (
            {
                "acquisition": {
                    "kind": "rc2.1",
                    "batch_size": 2,
                    "allocate_by": "item['frame']",
                    "constraints": [],
                    "share_tolerance": 0.5,
                }
            },
            r"allocation stratum \"item\['frame'\]\"",
        ),
        (
            {
                "acquisition": {
                    "kind": "rc2.1",
                    "batch_size": 2,
                    "allocate_by": "item['structure']",
                    "constraints": [
                        {
                            "constraint_type": "uniqueness",
                            "property_expression": "item['predicate']",
                        }
                    ],
                    "share_tolerance": 0.5,
                }
            },
            "a uniqueness constraint",
        ),
        (
            {
                "initial": {
                    "kind": "collected",
                    "stratify_by": "item['language']",
                    "items_per_stratum": 1,
                },
                "collector": {
                    "kind": "registered-design",
                    "design": "toy",
                    "join": None,
                },
                "encoding": None,
            },
            "initial stratification",
        ),
        (
            {"factors": {"structure": "item['structure']", "frame": "item['frame']"}},
            "factor 'frame'",
        ),
    ],
)
def test_an_attribute_the_pool_does_not_carry_is_refused_at_composition(
    tmp_path: Path, changed: dict[str, JsonValue], reason: str
) -> None:
    write_inputs(tmp_path)
    (tmp_path / "active.json").write_text(
        json.dumps(active_document(**changed)), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match=reason):
        ActiveLearning(
            compose_active_learning(tmp_path / "active.json"), tmp_path, inputs=INPUTS
        )


def test_a_command_line_run_awaits_each_batch_and_resumes_once_answered(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "documents"
    responses = write_inputs(folder)
    out = tmp_path / "out"

    first = active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS)

    assert first.status == "awaiting-responses"
    assert first.awaiting == 1
    assert first.rounds == 0
    assert first.initial_judgments == 8 * len(ANNOTATORS)

    answer(out, 1, responses)
    second = active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS)

    assert second.status == "awaiting-responses"
    assert second.awaiting == 2
    assert second.rounds == 1

    answer(out, 2, responses)
    third = active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS)
    rows = json.loads((out / ROWS_NAME).read_text(encoding="utf-8"))["rows"]
    recorded = decode_active_learning_manifest(
        (out / "active-learning.json").read_bytes()
    )

    assert third.status == "completed"
    assert third.rounds == 2
    assert third.stop_reason == "pool-exhausted"
    assert recorded == third
    assert {row["round"] for row in rows} == {1, 2}
    assert {row["metric"] for row in rows} == {
        "agreement",
        "rank-correlation",
        "batch-variance",
        "stratum-share",
    }
    agreement = [row for row in rows if row["metric"] == "agreement"]
    assert all(row["value"] == pytest.approx(1.0) for row in agreement)
    assert [row["n"] for row in agreement] == [30, 36]
    assert all(row["task"] == "active-learning" for row in rows)
    assert all(row["dataset"] == "collected-acceptability" for row in rows)
    assert [
        (row["round"], row["stopped"], row["stop_reason"]) for row in agreement
    ] == [
        (1, False, None),
        (2, True, "pool-exhausted"),
    ]
    assert all("candidate" not in row for row in rows)
    assert {
        row["stratum"]: row["value"] for row in rows if row["metric"] == "stratum-share"
    } == {"transitive": 0.5, "intransitive": 0.5}
    assert not any(row["floored"] for row in rows if row["metric"] == "stratum-share")
    assert set(third.source_digests) == {
        "encoding.json",
        "fit.json",
        "initial.jsonl",
        "items.json",
    }


def test_a_collected_start_fields_its_initial_draw_as_round_zero(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "documents"
    responses = write_inputs(folder)
    (folder / "initial.jsonl").unlink()
    document = active_document(
        initial={
            "kind": "collected",
            "stratify_by": "item['structure']",
            "items_per_stratum": 3,
        }
    )
    (folder / "active.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )
    out = tmp_path / "out"

    first = active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS)
    drawn = (round_directory(out, 0) / "batch.jsonl").read_text(encoding="utf-8")

    assert first.status == "awaiting-responses"
    assert first.awaiting == 0
    assert first.rounds == 0
    assert first.initial_judgments == 0
    assert drawn.count("\n") == 6
    assert active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS) == (
        first
    )

    answer(out, 0, responses)
    second = active_learn(folder / "active.json", 3, out, ENVIRONMENT, inputs=INPUTS)

    assert second.awaiting == 1
    assert second.initial_judgments == 6 * len(ANNOTATORS)
    assert (round_directory(out, 0) / "batch.jsonl").read_text(
        encoding="utf-8"
    ) == drawn
    assert set(second.source_digests) == {"encoding.json", "fit.json", "items.json"}


def test_a_simulated_run_from_a_collected_start_reads_no_bead_file(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "documents"
    write_inputs(folder)
    (folder / "initial.jsonl").unlink()
    (folder / "encoding.json").unlink()
    document = active_document(
        initial={
            "kind": "collected",
            "stratify_by": "item['structure']",
            "items_per_stratum": 3,
        },
        collector={
            "kind": "predictor-truth",
            "column": "p_gram",
            "participants": 8,
            "scale": "standardized",
            "latent_range": {"lower": -2.0, "upper": 2.0},
        },
        stopping={"kind": "asymptote", "window": 5, "maximum_gain": 0.0},
        max_rounds=5,
        encoding=None,
    )
    (folder / "active.json").write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )
    out = tmp_path / "out"

    manifest = active_learn(folder / "active.json", 0, out, ENVIRONMENT, inputs=INPUTS)
    rows = json.loads((out / ROWS_NAME).read_text(encoding="utf-8"))["rows"]

    assert manifest.status == "completed"
    last = max(row["round"] for row in rows)

    assert manifest.initial_judgments == 6 * 8
    assert manifest.rounds == last
    assert manifest.stop_reason == "pool-exhausted"
    assert set(manifest.source_digests) == {"fit.json", "items.json"}
    assert not (out / "rounds").exists()
    assert {row["stop_reason"] for row in rows if row["round"] < last} == {None}
    assert {row["stop_reason"] for row in rows if row["round"] == last} == {
        "pool-exhausted"
    }


@pytest.mark.parametrize(
    "changed",
    [
        {},
        {"collector": {"kind": "registered-design", "design": "toy", "join": None}},
        {
            "initial": {
                "kind": "collected",
                "stratify_by": "item['structure']",
                "items_per_stratum": 3,
            }
        },
    ],
)
def test_the_encoding_is_checked_against_the_fit_levels_before_any_round(
    tmp_path: Path, changed: dict[str, JsonValue]
) -> None:
    folder = tmp_path / "documents"
    write_inputs(folder)
    encoding = {**encoding_document(), "n_levels": 2, "labels": ["low", "high"]}
    (folder / "encoding.json").write_text(
        json.dumps(encoding), encoding="utf-8", newline="\n"
    )
    (folder / "active.json").write_text(
        json.dumps(active_document(**changed)), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="2 levels and the fit's response has 3"):
        active_learn(
            folder / "active.json", 3, tmp_path / "out", ENVIRONMENT, inputs=INPUTS
        )

    assert not (tmp_path / "out").exists()


def test_a_declared_encoding_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "documents"
    write_inputs(folder)
    (folder / "encoding.json").unlink()

    with pytest.raises(ValueError, match=r"response encoding 'encoding\.json'"):
        active_learn(
            folder / "active.json", 3, tmp_path / "out", ENVIRONMENT, inputs=INPUTS
        )


def test_a_run_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    folder = tmp_path / "documents"
    write_inputs(folder)

    active_learn(
        folder / "active.json", 3, tmp_path / "first", ENVIRONMENT, inputs=INPUTS
    )
    active_learn(
        folder / "active.json", 3, tmp_path / "second", ENVIRONMENT, inputs=INPUTS
    )

    assert (round_directory(tmp_path / "first", 1) / "batch.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 2
    first = [
        json.loads(line)["id"]
        for line in (round_directory(tmp_path / "first", 1) / "batch.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    second = [
        json.loads(line)["id"]
        for line in (round_directory(tmp_path / "second", 1) / "batch.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert first == second


def an_active_learning_experiment(folder: Path, /) -> Path:
    """Write an experiment sweeping the active-learning document's round bound."""
    document = an_experiment_document()
    document["config"] = {"kind": "active-learning", "path": "active.json"}
    document["rows"] = [
        {"evaluator": "active-learning", "metric": "agreement"},
        {"evaluator": "active-learning", "metric": "rank-correlation"},
    ]
    agreement: dict[str, JsonValue] = {
        "evaluator": "active-learning",
        "metric": "agreement",
    }
    document["primary"] = agreement
    document["decision_point"] = {
        "date": "2026-12-01",
        "signal": {
            "kind": "metric",
            "row": agreement,
            "statistic": "mean",
            "direction": "above",
            "threshold": 0.5,
        },
        "fallback": "Collected judgments agree across annotators.",
    }
    document["sweep"] = {
        "grid": [{"path": "max_rounds", "values": [2]}],
        "seeds": [0],
        "expected_points": 1,
    }
    document["contraction"] = {
        "rungs": [{"kind": "fallback"}],
        "factorial": "one round bound",
    }
    document["evaluation"] = {
        "alpha": 0.05,
        "correction": {"kind": "benjamini-hochberg"},
        "dependence": "prds",
        "minimum_seeds": 1,
        "seed_budget": {
            "metric": "agreement",
            "target": "mean",
            "precision": 0.05,
            "level": 0.95,
        },
    }
    (folder / "preregistration.json").write_text(
        json.dumps({**PREREGISTRATION, "grid": {"max_rounds": [2]}}),
        encoding="utf-8",
        newline="\n",
    )
    document["preregistration"] = "preregistration.json"
    path = folder / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    return path


def test_an_experiment_point_records_awaiting_and_then_completed_rounds(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "experiments"
    responses = write_inputs(folder)
    experiment = read_experiment(an_active_learning_experiment(folder))
    (point,) = expand_experiment(experiment, folder)
    store = ExperimentStore(tmp_path / "store", experiment, pilot=True)
    executor = ActiveLearningExecutor(
        store, environment=ENVIRONMENT, infrastructure_attempts=1, inputs=INPUTS
    )

    executor.check(point)
    first = executor(point)

    assert first.status is ManifestStatus.KILLED
    assert store.entry(point) == {"status": PointStatus.AWAITING_RESPONSES.value}
    assert not store.is_final(point)

    directory = store.directory(point)
    answer(directory, 1, responses)
    executor(point)
    answer(directory, 2, responses)
    last = executor(point)
    recorded = read_rows(directory)

    assert last.status is ManifestStatus.COMPLETED
    assert store.is_completed(point)
    assert recorded is not None
    assert {row["point_id"] for row in recorded.rows} == {point.point_id}
    assert {row["round"] for row in recorded.rows} == {1, 2}
    assert executor(point).cached
