from typing import TYPE_CHECKING, cast

import pytest
from tests.support.installed import installed

from lofi.eval import (
    DeclaredHypothesis,
    PointMetadata,
    Preregistration,
    ReductionContext,
    SignatureAssignments,
    reduce_experiment_rows,
    sample_signature_twins,
)
from lofi.eval._comparison import (
    SourceComparisonObservation,
    reduce_source_comparisons,
)
from lofi.eval._reducers import (
    SOURCE_COMPARISON_PLUGIN,
    ExperimentReducer,
)
from lofi.registry import ComponentPlugin
from lofi.spec.reducers import SourceComparisonReducerSpec
from lofi.stats import (
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Bonferroni,
    Dependence,
    DependenceError,
    Holm,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from lofi._types import JsonValue

type ReducerPlugin = ComponentPlugin[object, ReductionContext, ExperimentReducer]


SOURCE = SignatureAssignments(
    ("S", "NP", "N"),
    ("forward", "backward"),
    ("application", "composition"),
)
RIVAL = SignatureAssignments(
    ("S", "NP"),
    ("forward", "backward"),
    ("application",),
)


def _observation(
    candidate: str,
    seed: int,
    score: float,
    assignments: SignatureAssignments,
    *,
    arm: str | None = None,
    role: str = "candidate",
    data_size: int = 64,
    description_length: float = 0.0,
    direction: str = "bidirectional",
) -> SourceComparisonObservation:
    return SourceComparisonObservation(
        source="source",
        candidate=candidate,
        arm=candidate if arm is None else arm,
        role=role,
        seed=seed,
        data_size=data_size,
        score=score,
        description_length=description_length,
        signature_digest="signature-1",
        parameter_count=128,
        direction=direction,
        assignments=assignments,
    )


def _campaign(*, permuted: bool = False) -> tuple[SourceComparisonObservation, ...]:
    source_twins = sample_signature_twins(SOURCE, count=3, seed=7)
    rival_twins = sample_signature_twins(RIVAL, count=3, seed=9)
    rows: list[SourceComparisonObservation] = []

    for seed in range(6):
        source_score = 7.0 + seed / 10 if not permuted else 2.0 + seed / 10
        rival_score = 3.0 + seed / 10 if not permuted else 6.0 + seed / 10
        rows.extend(
            (
                _observation(
                    "source", seed, source_score + 2.0, SOURCE, description_length=2.0
                ),
                _observation(
                    "rival", seed, rival_score + 1.0, RIVAL, description_length=1.0
                ),
                _observation(
                    "source",
                    seed,
                    0.5 + seed / 10,
                    SOURCE,
                    arm="source-reversed",
                    role="reversed-ground-truth",
                    description_length=2.0,
                ),
                _observation(
                    "source",
                    seed,
                    0.25 + seed / 10,
                    SOURCE,
                    arm="source-lexicon-scrambled",
                    role="lexicon-scrambled",
                    description_length=2.0,
                ),
            )
        )
        rows.extend(
            _observation(
                "source",
                seed,
                source_score - 2.0 - position / 10,
                assignments,
                arm=f"source-twin-{position}",
                role="twin",
                description_length=2.0,
            )
            for position, assignments in enumerate(source_twins)
        )
        rows.extend(
            _observation(
                "rival",
                seed,
                rival_score - 1.0 - position / 10,
                assignments,
                arm=f"rival-twin-{position}",
                role="twin",
                description_length=1.0,
            )
            for position, assignments in enumerate(rival_twins)
        )

    return tuple(rows)


def _plan() -> Preregistration:
    return Preregistration(
        {"candidate": ["source", "rival"]},
        [
            DeclaredHypothesis(
                "source-against-rival",
                "source",
                "rival",
                family="primary",
                margin=0.25,
            )
        ],
    )


def _rows(
    report: Iterable[Mapping[str, JsonValue]], kind: str
) -> list[dict[str, JsonValue]]:
    return [dict(row) for row in report if row["reducer"] == kind]


def test_the_generating_source_wins_a_planted_two_candidate_comparison() -> None:
    report = reduce_source_comparisons(
        _campaign(),
        _plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )

    verdict = _rows(report.rows, "source-comparison-verdict")
    tests = _rows(report.rows, "source-comparison-test")
    deltas = _rows(report.rows, "source-comparison-delta")
    twins = _rows(report.rows, "source-comparison-twin")

    assert verdict == [
        {
            **verdict[0],
            "source": "source",
            "winner": "source",
            "source_rank": 1,
            "positive": True,
            "beats_candidates": True,
            "beats_twins": True,
            "validity_controls_pass": True,
        }
    ]
    assert len(tests) == 3
    assert all(
        cast("float", row["raw_p"]) <= cast("float", row["corrected_p"])
        for row in tests
    )
    assert (
        next(row for row in tests if row["candidate"] == "rival")["tost_verdict"]
        == "separated"
    )
    assert len(deltas) == 6
    assert len(twins) == 12
    assert report.comparisons[0].left == "source"
    assert report.comparisons[0].right == "rival"


def _bonferroni(raw: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(min(1.0, len(raw) * value) for value in raw)


def _holm(raw: tuple[float, ...]) -> tuple[float, ...]:
    adjusted = [0.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(raw)), key=lambda at: raw[at])):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return tuple(adjusted)


@pytest.mark.parametrize(
    ("correction", "adjust"),
    [(Bonferroni(), _bonferroni), (Holm(), _holm)],
)
def test_the_declared_correction_adjusts_each_size_family(
    correction: Bonferroni | Holm,
    adjust: Callable[[tuple[float, ...]], tuple[float, ...]],
) -> None:
    report = reduce_source_comparisons(
        _campaign(),
        _plan(),
        alpha=0.05,
        correction=correction,
        dependence=Dependence.INDEPENDENT,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )
    tests = _rows(report.rows, "source-comparison-test")
    raw = tuple(cast("float", row["raw_p"]) for row in tests)

    assert {row["correction"] for row in tests} == {correction.name}
    assert {row["dependence"] for row in tests} == {"independent"}
    assert tuple(cast("float", row["corrected_p"]) for row in tests) == pytest.approx(
        adjust(raw)
    )


def test_the_declared_dependence_reaches_the_correction() -> None:
    with pytest.raises(DependenceError):
        reduce_source_comparisons(
            _campaign(),
            _plan(),
            alpha=0.05,
            correction=BenjaminiHochberg(),
            dependence=Dependence.ARBITRARY,
            replicates=999,
            seed=13,
            falsifier="source fails",
            smoke_test=False,
        )

    report = reduce_source_comparisons(
        _campaign(),
        _plan(),
        alpha=0.05,
        correction=BenjaminiYekutieli(),
        dependence=Dependence.ARBITRARY,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )
    tests = _rows(report.rows, "source-comparison-test")

    assert {row["correction"] for row in tests} == {"benjamini-yekutieli"}
    assert {row["dependence"] for row in tests} == {"arbitrary"}


def test_permuting_the_sources_labels_prevents_a_positive_verdict() -> None:
    report = reduce_source_comparisons(
        _campaign(permuted=True),
        _plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )
    [verdict] = _rows(report.rows, "source-comparison-verdict")

    assert verdict["winner"] == "rival"
    assert verdict["wrong_winner_class"] == "simpler"
    assert verdict["positive"] is False


def test_declared_source_comparison_derives_observations_from_rows_and_points(
    tmp_path: Path,
) -> None:
    points: dict[str, tuple[PointMetadata, ...]] = {}
    rows: list[dict[str, JsonValue]] = []
    for index, observation in enumerate(_campaign()):
        config_hash = f"config-{index}"
        result_seed = 10_000 + index
        twin = "observed" if observation.role != "twin" else observation.arm
        condition = (
            "primary" if observation.role in {"candidate", "twin"} else observation.role
        )
        config: dict[str, JsonValue] = {
            "comparison": {
                "source": observation.source,
                "condition": condition,
                "direction": observation.direction,
                "twin": twin,
            },
            "candidate": {"name": observation.candidate},
            "data": {"size": observation.data_size},
        }
        points[config_hash] = (
            PointMetadata(
                config,
                point_id=f"point-{index}",
                seed=observation.seed,
                result_seed=result_seed,
                parameter_count=observation.parameter_count,
                signature_fingerprint=observation.signature_digest,
                assignments=observation.assignments,
            ),
        )
        common: dict[str, JsonValue] = {
            "method": observation.arm,
            "seed": result_seed,
            "config_hash": config_hash,
            "n": 10,
        }
        rows.extend(
            (
                {**common, "metric": "held-out-fit", "value": observation.score},
                {
                    **common,
                    "metric": "description-length-nats",
                    "value": observation.description_length * 10,
                },
            )
        )

    declaration = SourceComparisonReducerSpec.model_validate(
        {
            "kind": "source-comparison",
            "metric": "held-out-fit",
            "source_path": "comparison.source",
            "candidate_path": "candidate.name",
            "data_size_path": "data.size",
            "condition_path": "comparison.condition",
            "direction_path": "comparison.direction",
            "twin_path": "comparison.twin",
            "description_length_metric": "description-length-nats",
            "primary_condition": "primary",
            "reversed_ground_truth_condition": "reversed-ground-truth",
            "lexicon_scrambled_condition": "lexicon-scrambled",
            "observed_twin": "observed",
            "permutation": {"replicates": 999, "seed": 13},
            "equivalence": {"replicates": 999, "seed": 17},
            "falsifier": "source fails",
            "smoke_test": False,
        }
    )
    context = ReductionContext(
        tmp_path,
        seed=1,
        points=points,
        preregistration=_plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
    )
    registry = installed(
        "reducer",
        {"source-comparison": cast("ReducerPlugin", SOURCE_COMPARISON_PLUGIN)},
    )
    report = reduce_experiment_rows(rows, (declaration,), context, registry=registry)

    [verdict] = _rows(report.rows, "source-comparison-verdict")
    [first_delta, *_] = _rows(report.rows, "source-comparison-delta")
    assert verdict["positive"] is True
    assert first_delta["predictive_delta"] == pytest.approx(5.0)
    assert first_delta["description_length_delta"] == pytest.approx(1.0)
    assert first_delta["value"] == pytest.approx(4.0)
    assert [
        (comparison.left, comparison.right) for comparison in report.comparisons
    ] == [("source", "rival")]

    first_hash = next(iter(points))
    [first] = points[first_hash]
    points[first_hash] = (
        PointMetadata(
            first.config,
            point_id=first.point_id,
            seed=first.seed,
            result_seed=first.result_seed,
            parameter_count=first.parameter_count,
            signature_fingerprint=first.signature_fingerprint,
        ),
    )
    missing = ReductionContext(
        tmp_path,
        seed=1,
        points=points,
        preregistration=_plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
    )
    with pytest.raises(ValueError, match="declares no signature_assignment"):
        reduce_experiment_rows(rows, (declaration,), missing, registry=registry)


def test_the_source_by_winner_matrix_and_first_winning_size_are_emitted() -> None:
    observations = (
        *(
            SourceComparisonObservation(
                source=row.source,
                candidate=row.candidate,
                arm=row.arm,
                role=row.role,
                seed=row.seed,
                data_size=16,
                score=(
                    row.score - 6.0
                    if row.role == "candidate" and row.candidate == "source"
                    else row.score
                ),
                description_length=row.description_length,
                signature_digest=row.signature_digest,
                parameter_count=row.parameter_count,
                direction=row.direction,
                assignments=row.assignments,
            )
            for row in _campaign()
        ),
        *_campaign(),
    )
    report = reduce_source_comparisons(
        observations,
        _plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )
    matrix = _rows(report.rows, "source-by-winner")
    [threshold] = _rows(report.rows, "source-win-threshold")

    assert [(row["data_size"], row["winner"]) for row in matrix] == [
        (16, "rival"),
        (64, "source"),
    ]
    assert threshold["data_size"] == 64


def test_comparisons_require_exact_measured_matched_capacity() -> None:
    rows = list(_campaign())
    changed = rows[-1]
    values = {
        "source": changed.source,
        "candidate": changed.candidate,
        "arm": changed.arm,
        "role": changed.role,
        "seed": changed.seed,
        "data_size": changed.data_size,
        "score": changed.score,
        "description_length": changed.description_length,
        "signature_digest": changed.signature_digest,
        "parameter_count": changed.parameter_count,
        "direction": changed.direction,
        "assignments": changed.assignments,
    }
    values["parameter_count"] = 127
    rows[-1] = SourceComparisonObservation(**values)

    with pytest.raises(ValueError, match="matched capacity"):
        reduce_source_comparisons(
            rows,
            _plan(),
            alpha=0.05,
            correction=BenjaminiHochberg(),
            dependence=Dependence.PRDS,
            replicates=999,
            seed=13,
            falsifier="source fails",
            smoke_test=False,
        )


def test_comparisons_license_permuted_codecs_by_inventory_not_ordered_digest() -> None:
    rows = list(_campaign())
    changed = rows[-1]
    rows[-1] = SourceComparisonObservation(
        source=changed.source,
        candidate=changed.candidate,
        arm=changed.arm,
        role=changed.role,
        seed=changed.seed,
        data_size=changed.data_size,
        score=changed.score,
        description_length=changed.description_length,
        signature_digest="permuted-codec-signature",
        parameter_count=changed.parameter_count,
        direction=changed.direction,
        assignments=changed.assignments,
    )

    report = reduce_source_comparisons(
        rows,
        _plan(),
        alpha=0.05,
        correction=BenjaminiHochberg(),
        dependence=Dependence.PRDS,
        replicates=999,
        seed=13,
        falsifier="source fails",
        smoke_test=False,
    )

    assert _rows(report.rows, "source-comparison-verdict")


def test_comparisons_refuse_different_signature_inventories() -> None:
    rows = list(_campaign())
    changed = rows[-1]
    rows[-1] = SourceComparisonObservation(
        source=changed.source,
        candidate=changed.candidate,
        arm=changed.arm,
        role=changed.role,
        seed=changed.seed,
        data_size=changed.data_size,
        score=changed.score,
        description_length=changed.description_length,
        signature_digest=changed.signature_digest,
        parameter_count=changed.parameter_count,
        direction=changed.direction,
        assignments=SignatureAssignments(("S", "VP"), (), ()),
    )

    with pytest.raises(ValueError, match="signature-family inventories"):
        reduce_source_comparisons(
            rows,
            _plan(),
            alpha=0.05,
            correction=BenjaminiHochberg(),
            dependence=Dependence.PRDS,
            replicates=999,
            seed=13,
            falsifier="source fails",
            smoke_test=False,
        )
