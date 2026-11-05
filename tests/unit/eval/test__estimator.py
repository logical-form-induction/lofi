import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi._types import ItemId
from lofi.eval._estimator import (
    GROUNDING_FIELD,
    PRIMARY_LEFT,
    PRIMARY_RIGHT,
    READOUT_FIELD,
    REGIME_FIELDS,
    SIGNAL_FIELD,
    ArmCost,
    ArmSpecification,
    BudgetSetting,
    EstimatorComparison,
    EstimatorProtocol,
    EstimatorVerdict,
    GroundedRun,
    Outcome,
    RegimeRunDirectory,
    RunKind,
    SpecificationMismatchError,
    UnmatchedBudgetError,
    estimator_arms,
    estimator_report,
    reduce_grounding_runs,
)
from lofi.eval._prereg import PreregistrationError
from lofi.stats import Verdict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue

ITEMS: Final = tuple(ItemId(f"pair-{at:02d}") for at in range(10))
REGISTERED: Final = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
STARTED: Final = REGISTERED + timedelta(days=1)
SHARED: Final[dict[str, JsonValue]] = {
    "data.datasets.train.dataset": "quantifier-free",
    "trainer": "parser",
    "seeds": [0, 1, 2],
}
BEAM: Final[dict[str, JsonValue]] = {
    "strategy": "beam",
    "width": 4,
    "clamp": None,
}
REWARD_LOSS: Final[JsonValue] = [
    {
        "objective": {
            "kind": "inference",
            "signal": "reward",
            "response": {"kind": "label"},
            "auxiliary_weight": 0.0,
        }
    }
]
LIKELIHOOD_LOSS: Final[JsonValue] = [
    {
        "objective": {
            "kind": "inference",
            "signal": "likelihood",
            "response": {"kind": "label"},
            "auxiliary_weight": 0.0,
        }
    }
]
REGIMES: Final[dict[str, Mapping[str, JsonValue]]] = {
    "prover-reward": {
        GROUNDING_FIELD: "prover-reward",
        READOUT_FIELD: {
            "strategy": "sample",
            "restriction": {"kind": "full"},
            "draws": 2,
            "clamp": None,
        },
        SIGNAL_FIELD: REWARD_LOSS,
    },
    "prover-enumerated": {
        GROUNDING_FIELD: "prover-enumerated",
        READOUT_FIELD: BEAM,
        SIGNAL_FIELD: LIKELIHOOD_LOSS,
    },
    "expected-embedding": {
        GROUNDING_FIELD: "expected-embedding",
        READOUT_FIELD: BEAM,
        SIGNAL_FIELD: LIKELIHOOD_LOSS,
    },
    "nli-baseline": {
        GROUNDING_FIELD: "nli-baseline",
        READOUT_FIELD: BEAM,
        SIGNAL_FIELD: LIKELIHOOD_LOSS,
    },
}
REPLICATES: Final = 199
SEED: Final = 7
MARGIN: Final = 0.1
FALSIFIER: Final = "prover reward is no better than expected embedding"


def scores(pattern: str) -> tuple[float, ...]:
    return tuple(1.0 if mark == "1" else 0.0 for mark in pattern)


MOSTLY_RIGHT: Final = scores("1111111110")
MOSTLY_WRONG: Final = scores("1100000000")
SPLIT: Final = scores("1111100000")


def cost(
    *,
    optimizer_steps: int = 40,
    prover_cpu_seconds: float = 1.25,
    wall_seconds: float = 2.5,
) -> ArmCost:
    return ArmCost(
        optimizer_steps=optimizer_steps,
        prover_cpu_seconds=prover_cpu_seconds,
        wall_seconds=wall_seconds,
    )


def run(
    arm: str,
    setting: BudgetSetting,
    preserving: Sequence[float],
    *,
    likelihood: Sequence[float] | None = None,
    accuracy: Sequence[float] | None = None,
    started_at: datetime = STARTED,
    arm_cost: ArmCost | None = None,
    items: Sequence[ItemId] = ITEMS,
) -> GroundedRun:
    outcomes: dict[Outcome, Mapping[ItemId, float]] = {
        Outcome.SYMBOLIC_RATE: dict(zip(items, preserving, strict=True)),
        Outcome.LABEL_LIKELIHOOD: dict(
            zip(items, preserving if likelihood is None else likelihood, strict=True)
        ),
        Outcome.LABEL_ACCURACY: dict(
            zip(items, preserving if accuracy is None else accuracy, strict=True)
        ),
    }
    return GroundedRun(
        arm,
        setting,
        kind=RunKind.SYNTHETIC,
        started_at=started_at,
        cost=cost() if arm_cost is None else arm_cost,
        outcomes=outcomes,
    )


def protocol(
    *,
    left: str = PRIMARY_LEFT,
    right: str = PRIMARY_RIGHT,
    outcome: Outcome = Outcome.SYMBOLIC_RATE,
    margin: float = MARGIN,
    registered_at: datetime = REGISTERED,
) -> EstimatorProtocol:
    return EstimatorProtocol(
        left,
        right,
        outcome=outcome,
        margin=margin,
        falsifier=FALSIFIER,
        family="grounding-regime comparison",
        registered_at=registered_at,
        alpha=0.05,
        replicates=REPLICATES,
        seed=SEED,
        level=0.95,
    )


def runs(
    reward: Sequence[float], embedding: Sequence[float]
) -> dict[tuple[str, BudgetSetting], GroundedRun]:
    scores_by_arm = {
        "prover-reward": reward,
        "prover-enumerated": MOSTLY_RIGHT,
        "expected-embedding": embedding,
        "nli-baseline": SPLIT,
    }
    return {
        (arm, setting): run(arm, setting, held)
        for arm, held in scores_by_arm.items()
        for setting in BudgetSetting
    }


def comparison(
    reward: Sequence[float] = MOSTLY_RIGHT,
    embedding: Sequence[float] = MOSTLY_WRONG,
) -> EstimatorComparison:
    return EstimatorComparison(
        protocol(), estimator_arms(SHARED, REGIMES), runs(reward, embedding)
    )


def test_standard_arms_declare_the_yoked_regime_axis() -> None:
    arms = estimator_arms(SHARED, REGIMES)

    assert [arm.name for arm in arms] == list(REGIMES)
    assert set(arms[0].regime) == set(REGIME_FIELDS)
    assert arms[0].regime == dict(REGIMES["prover-reward"])
    assert all(
        {name: value for name, value in arm.fields.items() if name not in REGIME_FIELDS}
        == SHARED
        for arm in arms
    )


def test_an_additional_declared_scorer_is_an_ordinary_arm() -> None:
    regimes: dict[str, Mapping[str, JsonValue]] = {
        **REGIMES,
        "model-checking-sampled": {
            GROUNDING_FIELD: "model-checking-sampled",
            READOUT_FIELD: {"strategy": "beam", "width": 4, "clamp": None},
            SIGNAL_FIELD: LIKELIHOOD_LOSS,
        },
    }

    assert estimator_arms(SHARED, regimes)[-1].name == "model-checking-sampled"


def test_shared_fields_may_not_fix_a_regime_field() -> None:
    with pytest.raises(ValueError, match=GROUNDING_FIELD):
        estimator_arms({**SHARED, GROUNDING_FIELD: "prover-reward"}, REGIMES)


def test_each_arm_declares_the_whole_yoked_axis() -> None:
    incomplete = {
        "prover-reward": {
            GROUNDING_FIELD: "prover-reward",
            SIGNAL_FIELD: REWARD_LOSS,
        }
    }

    with pytest.raises(ValueError, match=READOUT_FIELD):
        estimator_arms(SHARED, incomplete)


def test_two_arm_names_may_not_declare_one_regime_tuple() -> None:
    doubled = {**REGIMES, "same": REGIMES["prover-reward"]}

    with pytest.raises(SpecificationMismatchError, match="distinct regime tuples"):
        estimator_arms(SHARED, doubled)


def test_a_nonregime_field_may_not_change_between_arms() -> None:
    arms = list(estimator_arms(SHARED, REGIMES))
    arms[1] = ArmSpecification(arms[1].name, {**arms[1].fields, "trainer": "other"})

    with pytest.raises(SpecificationMismatchError, match="'trainer'"):
        EstimatorComparison(protocol(), tuple(arms), runs(MOSTLY_RIGHT, MOSTLY_WRONG))


def test_an_arm_emits_its_fragment_name_and_complete_regime() -> None:
    spec = estimator_arms(SHARED, REGIMES)[0].to_spec()

    assert spec["arm"] == "prover-reward"
    assert spec["regime"] == dict(REGIMES["prover-reward"])
    assert spec["fields"] == {**SHARED, **REGIMES["prover-reward"]}


def test_a_cost_reports_the_matched_and_measured_budgets() -> None:
    assert cost().to_spec() == {
        "optimizer_steps": 40,
        "prover_cpu_seconds": 1.25,
        "wall_seconds": 2.5,
    }

    with pytest.raises(ValueError, match="optimizer steps"):
        cost(optimizer_steps=0)
    with pytest.raises(ValueError, match="CPU seconds"):
        cost(prover_cpu_seconds=-0.1)
    with pytest.raises(ValueError, match="wall seconds"):
        cost(wall_seconds=-0.1)


def test_a_synthetic_run_reports_all_primary_outcomes() -> None:
    held = run(
        "prover-reward",
        BudgetSetting.FULL,
        SPLIT,
        likelihood=tuple(-value for value in SPLIT),
        accuracy=MOSTLY_RIGHT,
    )

    assert held.value(Outcome.SYMBOLIC_RATE) == pytest.approx(0.5)
    assert held.value(Outcome.LABEL_LIKELIHOOD) == pytest.approx(-0.5)
    assert held.value(Outcome.LABEL_ACCURACY) == pytest.approx(0.9)
    assert held.scores(Outcome.SYMBOLIC_RATE) == SPLIT


def test_a_judgment_run_reports_likelihood_alone() -> None:
    held = GroundedRun(
        "nli-baseline",
        BudgetSetting.FULL,
        kind=RunKind.JUDGMENT,
        started_at=STARTED,
        cost=cost(),
        outcomes={Outcome.JUDGMENT_LIKELIHOOD: dict(zip(ITEMS, SPLIT, strict=True))},
    )

    assert held.outcomes == (Outcome.JUDGMENT_LIKELIHOOD,)
    assert held.value(Outcome.JUDGMENT_LIKELIHOOD) == pytest.approx(0.5)


def test_a_run_refuses_outcomes_from_the_other_task() -> None:
    with pytest.raises(ValueError, match="synthetic run reports"):
        GroundedRun(
            "prover-reward",
            BudgetSetting.FULL,
            kind=RunKind.SYNTHETIC,
            started_at=STARTED,
            cost=cost(),
            outcomes={Outcome.JUDGMENT_LIKELIHOOD: {ITEMS[0]: 0.0}},
        )


def test_a_protocol_registers_the_fragment_named_primary_contrast() -> None:
    assert protocol().to_spec() == {
        "left": "prover-reward",
        "right": "expected-embedding",
        "primary_outcome": "entailment-preserving-rate-at-1",
        "margin": MARGIN,
        "falsifier": FALSIFIER,
        "family": "grounding-regime comparison",
        "registered_at": REGISTERED.isoformat(),
        "alpha": 0.05,
        "level": 0.95,
        "replicates": REPLICATES,
        "seed": SEED,
    }


def test_registration_precedes_every_run() -> None:
    early = runs(MOSTLY_RIGHT, MOSTLY_WRONG)
    early[PRIMARY_LEFT, BudgetSetting.FULL] = run(
        PRIMARY_LEFT,
        BudgetSetting.FULL,
        MOSTLY_RIGHT,
        started_at=REGISTERED - timedelta(seconds=1),
    )

    with pytest.raises(PreregistrationError, match="does not precede"):
        EstimatorComparison(protocol(), estimator_arms(SHARED, REGIMES), early)


def test_arms_are_matched_on_optimizer_steps() -> None:
    unmatched = runs(MOSTLY_RIGHT, MOSTLY_WRONG)
    unmatched["nli-baseline", BudgetSetting.HALF] = run(
        "nli-baseline",
        BudgetSetting.HALF,
        SPLIT,
        arm_cost=cost(optimizer_steps=41),
    )

    with pytest.raises(UnmatchedBudgetError, match="optimizer steps"):
        EstimatorComparison(protocol(), estimator_arms(SHARED, REGIMES), unmatched)


def test_every_arm_scores_the_same_units() -> None:
    apart = runs(MOSTLY_RIGHT, MOSTLY_WRONG)
    apart["nli-baseline", BudgetSetting.HALF] = run(
        "nli-baseline",
        BudgetSetting.HALF,
        SPLIT,
        items=tuple(ItemId(f"other-{at}") for at in range(10)),
    )

    with pytest.raises(ValueError, match="different held-out units"):
        EstimatorComparison(protocol(), estimator_arms(SHARED, REGIMES), apart)


def test_four_arms_form_six_contrasts_at_each_budget() -> None:
    built = comparison()

    assert len(built.rows) == 12
    assert len(built.primary) == 2
    assert {(row.left.arm, row.right.arm) for row in built.primary} == {
        (PRIMARY_LEFT, PRIMARY_RIGHT)
    }
    assert {row.setting for row in built.primary} == set(BudgetSetting)


def test_the_primary_contrast_carries_paired_and_tost_results() -> None:
    row = comparison().primary[0]

    assert row.separation.statistic == pytest.approx(0.7)
    assert row.equivalence.observed == pytest.approx(0.7)
    assert row.equivalence.verdict is Verdict.SEPARATED
    assert row.verdict is EstimatorVerdict.LEFT_SEPARATES


def test_equal_known_forms_are_equivalent_within_the_margin() -> None:
    built = comparison(reward=SPLIT, embedding=SPLIT)

    assert (
        built.verdict(BudgetSetting.FULL) is EstimatorVerdict.EQUIVALENT_WITHIN_MARGIN
    )


def test_the_report_emits_outcomes_costs_and_statistics() -> None:
    spec = estimator_report(comparison()).to_spec()
    rows = cast("list[JsonValue]", spec["rows"])
    primary = next(
        row
        for row in rows
        if isinstance(row, dict) and row["primary"] is True and row["setting"] == "full"
    )
    left = cast("dict[str, JsonValue]", primary["left"])

    assert left["arm"] == PRIMARY_LEFT
    assert left["entailment-preserving-rate-at-1"] == pytest.approx(0.9)
    assert left["held-out-label-log-likelihood"] == pytest.approx(0.9)
    assert left["entailment-label-accuracy"] == pytest.approx(0.9)
    assert left["cost"] == {
        "optimizer_steps": 40,
        "prover_cpu_seconds": 1.25,
        "wall_seconds": 2.5,
    }
    assert primary["equivalence"] == Verdict.SEPARATED.value
    assert json.loads(json.dumps(spec)) == spec


def _write_run(
    directory: Path,
    *,
    seed: int,
    task: RunKind,
    values: Mapping[Outcome, float],
    steps: int = 4,
    cpu_seconds: float = 1.5,
    wall_seconds: float = 2.0,
) -> None:
    directory.mkdir(parents=True)
    (directory / "point.json").write_text(
        json.dumps({"freeze_id": "fixture", "seed_derivation": {}}),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8", newline="\n"
    )
    step_metrics = {
        "grounded/cpu_seconds": cpu_seconds,
        "grounded/wall_seconds": wall_seconds,
    }
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "stage": "closed",
                "point": {"seed": seed},
                "fitting": {
                    "optimizer_steps": steps,
                    "epochs": [{"steps": [{"metrics": step_metrics}]}],
                    "steps": [],
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "rows.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "metric": outcome.value,
                        "value": value,
                        "seed": seed,
                        "dataset": task.value,
                    }
                    for outcome, value in values.items()
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_the_reducer_reads_completed_synthetic_run_directories(tmp_path: Path) -> None:
    directories: list[RegimeRunDirectory] = []
    rates = {
        "prover-reward": (1.0, 1.0, 0.0),
        "prover-enumerated": (1.0, 0.0, 1.0),
        "expected-embedding": (0.0, 0.0, 1.0),
        "nli-baseline": (0.0, 1.0, 0.0),
    }

    for arm, observed in rates.items():
        for seed, value in enumerate(observed):
            directory = tmp_path / arm / str(seed)
            _write_run(
                directory,
                seed=seed,
                task=RunKind.SYNTHETIC,
                values={
                    Outcome.SYMBOLIC_RATE: value,
                    Outcome.LABEL_LIKELIHOOD: value - 1.0,
                    Outcome.LABEL_ACCURACY: value,
                },
            )
            directories.append(RegimeRunDirectory(arm, BudgetSetting.FULL, directory))

    report = reduce_grounding_runs(
        estimator_arms(SHARED, REGIMES), directories, protocol=protocol()
    )
    runs_by_arm = {held.arm: held for held in report.runs}

    assert runs_by_arm[PRIMARY_LEFT].value(Outcome.LABEL_ACCURACY) == pytest.approx(
        2.0 / 3.0
    )
    assert runs_by_arm[PRIMARY_LEFT].cost == ArmCost(
        optimizer_steps=12, prover_cpu_seconds=4.5, wall_seconds=6.0
    )
    assert len(report.rows) == 6


def test_the_same_reducer_reports_judgment_likelihood_beside_nli(
    tmp_path: Path,
) -> None:
    regimes = {
        PRIMARY_LEFT: REGIMES[PRIMARY_LEFT],
        "nli-baseline": REGIMES["nli-baseline"],
    }
    directories = []

    for arm, values in {
        PRIMARY_LEFT: (-0.2, -0.3, -0.25),
        "nli-baseline": (-0.7, -0.6, -0.65),
    }.items():
        for seed, value in enumerate(values):
            directory = tmp_path / arm / str(seed)
            _write_run(
                directory,
                seed=seed,
                task=RunKind.JUDGMENT,
                values={Outcome.JUDGMENT_LIKELIHOOD: value},
                cpu_seconds=0.0,
            )
            directories.append(RegimeRunDirectory(arm, BudgetSetting.FULL, directory))

    report = reduce_grounding_runs(
        estimator_arms(SHARED, regimes),
        directories,
        protocol=protocol(right="nli-baseline", outcome=Outcome.JUDGMENT_LIKELIHOOD),
    )

    assert [held.arm for held in report.runs] == [PRIMARY_LEFT, "nli-baseline"]
    assert report.primary[0].left.value(Outcome.JUDGMENT_LIKELIHOOD) == pytest.approx(
        -0.25
    )
    assert report.primary[0].right.value(Outcome.JUDGMENT_LIKELIHOOD) == pytest.approx(
        -0.65
    )


def test_the_reducer_refuses_an_incomplete_run_directory(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "point.json").write_text(
        json.dumps({"freeze_id": "fixture", "seed_derivation": {}}),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="not completed"):
        reduce_grounding_runs(
            estimator_arms(SHARED, REGIMES),
            (RegimeRunDirectory(PRIMARY_LEFT, BudgetSetting.FULL, directory),),
            protocol=protocol(),
        )


def test_the_reducer_accepts_only_the_current_exact_metric_names(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"
    _write_run(
        directory,
        seed=0,
        task=RunKind.SYNTHETIC,
        values={
            Outcome.SYMBOLIC_RATE: 1.0,
            Outcome.LABEL_LIKELIHOOD: -0.1,
            Outcome.LABEL_ACCURACY: 1.0,
        },
    )
    rows_path = directory / "rows.json"
    rows = cast(
        "dict[str, JsonValue]", json.loads(rows_path.read_text(encoding="utf-8"))
    )
    declared = cast("list[JsonValue]", rows["rows"])
    symbolic = cast("dict[str, JsonValue]", declared[0])
    symbolic["metric"] = "entailment-preserving-rate-at-2"
    rows_path.write_text(json.dumps(rows), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match=Outcome.SYMBOLIC_RATE.value):
        reduce_grounding_runs(
            estimator_arms(SHARED, REGIMES),
            (RegimeRunDirectory(PRIMARY_LEFT, BudgetSetting.FULL, directory),),
            protocol=protocol(),
        )
