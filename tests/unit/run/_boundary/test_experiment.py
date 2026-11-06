from datetime import date
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run import Amendment, ArrayLaunchSpec, ExperimentConfigSpec
from lofi.run._boundary.experiment import (
    ActiveLearningExperimentConfig,
    Direction,
    EvaluationSpec,
    ExperimentCompute,
    ExperimentContraction,
    ExperimentDecision,
    FactorLevel,
    FactorSpan,
    FailurePolicySpec,
    GeneratorSpec,
    LanguageRung,
    MetricPredicate,
    PointCountPredicate,
    ResolutionIVRung,
    RowReference,
    SeedBudgetSpec,
    SeedRung,
    SignatureTwinSettings,
    SignatureTwinsSpec,
    Statistic,
    decode_amendments,
    encode_amendments,
)
from lofi.stats import Dependence

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_an_estimate_keeps_its_two_percentiles_and_cost_model() -> None:
    estimate = ExperimentCompute(p50_hours=2.0, p90_hours=3.5, cost_model="v1")

    assert (estimate.p50_hours, estimate.p90_hours, estimate.cost_model) == (
        2.0,
        3.5,
        "v1",
    )


def test_signature_twins_lower_to_local_deterministic_settings() -> None:
    declaration = SignatureTwinsSpec(
        count=3,
        seed=17,
        primary_condition="primary",
        observed_twin="observed",
    )

    assert declaration.settings() == SignatureTwinSettings(3, 17, "primary", "observed")

    with pytest.raises(dx.ValidationError, match="fewer than three"):
        declaration.with_(count=2)

    with pytest.raises(dx.ValidationError, match="negative"):
        declaration.with_(seed=-1)

    with pytest.raises(dx.ValidationError, match="sampled-twin identity"):
        declaration.with_(observed_twin="twin-001")


@pytest.mark.parametrize(
    ("p50", "p90"),
    [
        pytest.param(3.0, 2.0, id="p90-below-p50"),
        pytest.param(-1.0, 2.0, id="negative"),
        pytest.param(float("nan"), 2.0, id="not-finite"),
    ],
)
def test_an_estimate_refuses_hours_no_cost_model_could_produce(
    p50: float, p90: float
) -> None:
    with pytest.raises(dx.ValidationError):
        ExperimentCompute(p50_hours=p50, p90_hours=p90, cost_model="v1")


def test_a_ladder_declared_in_ladder_order_is_accepted() -> None:
    ladder = ExperimentContraction.model_validate(
        {
            "rungs": [
                {"kind": "languages", "axis": "data.language", "order": ["de", "en"]},
                {"kind": "seeds", "floor": 3},
            ],
            "factorial": "2x2",
        }
    )

    assert tuple(rung.kind for rung in ladder.rungs) == ("languages", "seeds")
    assert isinstance(ladder.rungs[0], LanguageRung)
    assert ladder.rungs[0].order == ("de", "en")


@pytest.mark.parametrize(
    "rungs",
    [
        pytest.param([], id="empty"),
        pytest.param(
            [{"kind": "seeds", "floor": 3}, {"kind": "seeds", "floor": 4}],
            id="repeated",
        ),
    ],
)
def test_a_ladder_refuses_rungs_out_of_ladder_order(rungs: list[JsonValue]) -> None:
    with pytest.raises(dx.ValidationError):
        ExperimentContraction.model_validate({"rungs": rungs, "factorial": "2x2"})


def test_a_seed_rung_refuses_a_floor_that_reports_no_seed() -> None:
    with pytest.raises(dx.ValidationError, match="seed floor"):
        SeedRung(floor=0)


def an_evaluation(**changed: JsonValue) -> dict[str, JsonValue]:
    return {
        "alpha": 0.05,
        "correction": {"kind": "holm"},
        "dependence": "arbitrary",
        "minimum_seeds": 3,
        "seed_budget": {
            "metric": "recovery",
            "target": "upper-decile",
            "precision": 0.1,
            "level": 0.9,
        },
        **changed,
    }


def test_an_evaluation_section_declares_every_statistical_setting() -> None:
    declared = EvaluationSpec.model_validate(an_evaluation())

    assert declared.alpha == 0.05
    assert declared.correction.kind == "holm"
    assert declared.correction.procedure().name == "holm"
    assert declared.dependence is Dependence.ARBITRARY
    assert declared.minimum_seeds == 3
    assert declared.seed_budget == SeedBudgetSpec(
        metric="recovery", target="upper-decile", precision=0.1, level=0.9
    )
    assert EvaluationSpec.model_validate_json(declared.model_dump_json()) == declared


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"alpha": 0.0}, "not a proportion"),
        ({"alpha": 1.0}, "not a proportion"),
        ({"correction": {"kind": "benjamini-hochberg"}}, "not valid under arbitrary"),
        ({"minimum_seeds": 0}, "at least one seed"),
        (
            {
                "seed_budget": {
                    "metric": "recovery",
                    "target": "median",
                    "precision": 0.1,
                    "level": 0.9,
                }
            },
            "target",
        ),
        (
            {
                "seed_budget": {
                    "metric": "recovery",
                    "target": "mean",
                    "precision": 0.0,
                    "level": 0.9,
                }
            },
            "not positive",
        ),
        (
            {
                "seed_budget": {
                    "metric": "",
                    "target": "mean",
                    "precision": 0.1,
                    "level": 0.9,
                }
            },
            "budgeted metric",
        ),
        (
            {
                "seed_budget": {
                    "metric": "recovery",
                    "target": "mean",
                    "precision": 0.1,
                    "level": 1.0,
                }
            },
            "not a proportion",
        ),
    ],
)
def test_a_malformed_evaluation_section_is_refused(
    changed: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        EvaluationSpec.model_validate(an_evaluation(**changed))


def test_a_language_rung_declares_a_drop_order() -> None:
    with pytest.raises(dx.ValidationError, match="language drop order"):
        LanguageRung(axis="data.language", order=())


def test_resolution_iv_generators_span_no_word_shorter_than_four_letters() -> None:
    accepted = ResolutionIVRung(
        generators=(
            GeneratorSpec(factor="e", word=("a", "b", "c")),
            GeneratorSpec(factor="f", word=("b", "c", "d")),
        )
    )

    assert len(accepted.generators) == 2

    with pytest.raises(dx.ValidationError, match=r"\['d', 'e', 'f'\]"):
        ResolutionIVRung(
            generators=(
                GeneratorSpec(factor="e", word=("a", "b", "c")),
                GeneratorSpec(factor="f", word=("b", "c", "d", "a")),
            )
        )

    with pytest.raises(dx.ValidationError, match="shorter than the 4 letters"):
        ResolutionIVRung(generators=(GeneratorSpec(factor="c", word=("a", "b")),))

    with pytest.raises(dx.ValidationError, match="also appear in a word"):
        ResolutionIVRung(
            generators=(
                GeneratorSpec(factor="e", word=("a", "b", "c")),
                GeneratorSpec(factor="f", word=("e", "b", "d")),
            )
        )


def test_a_decision_point_refuses_blank_fallback_text() -> None:
    signal = PointCountPredicate(statuses=("completed",), at_least=1)

    with pytest.raises(dx.ValidationError):
        ExperimentDecision(date="2026-10-14", signal=signal, fallback=" ")


def test_a_metric_predicate_refuses_a_threshold_that_is_not_finite() -> None:
    row = RowReference(evaluator="held-out", metric="recovery")

    with pytest.raises(dx.ValidationError):
        MetricPredicate(
            row=row,
            statistic=Statistic.MEAN,
            direction=Direction.ABOVE,
            threshold=float("nan"),
        )


def test_a_factor_span_refuses_a_level_mapped_twice() -> None:
    with pytest.raises(dx.ValidationError, match="maps one level twice"):
        FactorSpan(
            axis="trainer.epochs",
            levels=(
                FactorLevel(level="w1", value=1),
                FactorLevel(level="w1", value=2),
            ),
        )


def test_an_amendment_ledger_round_trips_and_refuses_a_broken_chain() -> None:
    first = Amendment(
        "unpreregistered-cells",
        (3,),
        "a" * 16,
        "b" * 16,
        12,
        date(2026, 11, 2),
    )
    second = Amendment("seeds", (4,), "b" * 16, "c" * 16, 9, date(2026, 11, 3))

    ledger = decode_amendments(encode_amendments((first, second)))

    assert tuple(record.rung for record in ledger.amendments) == (
        "unpreregistered-cells",
        "seeds",
    )
    assert ledger.amendments[1].dropped == (4,)

    with pytest.raises(ValueError, match="starts from the sweep"):
        encode_amendments((first, second._replace(previous_sweep_id="d" * 16)))


def test_a_configuration_names_its_kind_and_document() -> None:
    config = ExperimentConfigSpec.model_validate(
        {"kind": "judgment-fit", "path": "fit.yaml"}
    )

    assert config.kind == "judgment-fit"
    assert config.path == "fit.yaml"

    with pytest.raises(dx.ValidationError):
        ExperimentConfigSpec.model_validate({"kind": "run", "path": ""})


def test_an_active_learning_document_is_a_configuration_kind() -> None:
    config = ExperimentConfigSpec.model_validate(
        {"kind": "active-learning", "path": "active.yaml"}
    )

    assert isinstance(config, ActiveLearningExperimentConfig)
    assert config.path == "active.yaml"


def test_launch_and_failure_policies_are_declared_at_the_experiment_boundary() -> None:
    launch = ArrayLaunchSpec(chunk_size=4, throttle=2)
    failure = FailurePolicySpec(numerical="reseed", out_of_memory="halve-batch")

    assert (launch.kind, launch.chunk_size, launch.throttle) == ("array", 4, 2)
    assert (failure.numerical, failure.out_of_memory) == (
        "reseed",
        "halve-batch",
    )


def test_a_failure_policy_declares_the_infrastructure_attempts_of_each_point() -> None:
    declared = FailurePolicySpec(
        numerical="final", out_of_memory="retry", infrastructure_attempts=5
    )
    defaulted = FailurePolicySpec(numerical="final", out_of_memory="retry")

    assert declared.infrastructure_attempts == 5
    assert defaulted.infrastructure_attempts == 3

    with pytest.raises(dx.ValidationError, match="infrastructure attempt"):
        FailurePolicySpec(
            numerical="final", out_of_memory="retry", infrastructure_attempts=0
        )
