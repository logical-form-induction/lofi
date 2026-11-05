import json

import didactic.api as dx
import pytest
from tests.unit.spec.components.conftest import PinningLoss, declaration, minimal

from lofi.spec.components.loss import WeightedLossSpec, WeightScheduleSpec
from lofi.spec.components.run import RunSpec


def test_a_term_weight_defaults_to_one() -> None:
    loss = minimal().loss
    assert loss is not None
    (term,) = loss.terms

    assert term.weight == 1.0


def test_each_term_weight_is_independently_configurable() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "loss": {
                "terms": [
                    {"objective": {"kind": "coherence-loss"}, "weight": 0.1},
                    {"objective": {"kind": "pinning-loss"}, "weight": 0.2},
                ]
            }
        }
    )

    assert run.loss is not None
    assert [term.weight for term in run.loss.terms] == [0.1, 0.2]


def test_the_terms_keep_the_order_they_were_declared_in() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "loss": {
                "terms": [
                    {"objective": {"kind": "pinning-loss"}},
                    {"objective": {"kind": "coherence-loss"}},
                ]
            }
        }
    )

    assert run.loss is not None
    assert [term.tag for term in run.loss.terms] == ["pinning-loss", "coherence-loss"]


def test_a_term_reports_the_tag_its_objective_was_declared_under() -> None:
    term = WeightedLossSpec(objective=PinningLoss(margin=0.75), weight=2.0)

    assert term.tag == "pinning-loss"
    assert "tag" not in json.loads(term.model_dump_json())


@pytest.mark.parametrize(
    "weight",
    [-0.1, float("inf"), float("-inf"), float("nan")],
)
def test_a_term_weight_is_finite_and_nonnegative(weight: float) -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(
            declaration()
            | {
                "loss": {
                    "terms": [
                        {"objective": {"kind": "coherence-loss"}, "weight": weight}
                    ]
                }
            }
        )


def test_a_term_weight_of_zero_is_accepted() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {"loss": {"terms": [{"objective": {"kind": "coherence-loss"}, "weight": 0}]}}
    )

    assert run.loss is not None
    assert run.loss.terms[0].weight == 0.0


def test_a_loss_section_with_no_term_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="at least one term"):
        RunSpec.model_validate(declaration() | {"loss": {"terms": []}})


def test_two_terms_naming_one_objective_are_refused() -> None:
    with pytest.raises(dx.ValidationError, match="coherence-loss"):
        RunSpec.model_validate(
            declaration()
            | {
                "loss": {
                    "terms": [
                        {"objective": {"kind": "coherence-loss"}, "weight": 0.5},
                        {"objective": {"kind": "coherence-loss"}, "weight": 0.25},
                    ]
                }
            }
        )


def test_two_terms_of_one_objective_coexist_under_their_own_names() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "loss": {
                "terms": [
                    {"objective": {"kind": "coherence-loss"}, "weight": 0.5},
                    {
                        "objective": {"kind": "coherence-loss"},
                        "weight": 0.25,
                        "name": "coherence-secondary",
                    },
                ]
            }
        }
    )

    assert run.loss is not None
    assert [term.tag for term in run.loss.terms] == ["coherence-loss"] * 2
    assert [term.key for term in run.loss.terms] == [
        "coherence-loss",
        "coherence-secondary",
    ]
    assert RunSpec.model_validate_json(run.model_dump_json()) == run


def test_a_declared_name_that_repeats_another_term_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="'shared'"):
        RunSpec.model_validate(
            declaration()
            | {
                "loss": {
                    "terms": [
                        {"objective": {"kind": "coherence-loss"}, "name": "shared"},
                        {"objective": {"kind": "pinning-loss"}, "name": "shared"},
                    ]
                }
            }
        )


def test_a_declared_name_that_repeats_another_terms_tag_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="'coherence-loss'"):
        RunSpec.model_validate(
            declaration()
            | {
                "loss": {
                    "terms": [
                        {"objective": {"kind": "coherence-loss"}},
                        {
                            "objective": {"kind": "pinning-loss"},
                            "name": "coherence-loss",
                        },
                    ]
                }
            }
        )


def test_a_term_without_a_name_is_keyed_by_its_tag() -> None:
    term = WeightedLossSpec(objective=PinningLoss(margin=0.75))
    named = WeightedLossSpec(objective=PinningLoss(margin=0.75), name="strict")

    assert term.name is None
    assert term.key == "pinning-loss"
    assert named.key == "strict"
    assert named.tag == "pinning-loss"


@pytest.mark.parametrize(
    ("name", "detail"), [("", "empty"), ("loss/strict", "contains no slash")]
)
def test_a_term_name_is_one_nonempty_key_segment(name: str, detail: str) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        WeightedLossSpec(objective=PinningLoss(margin=0.75), name=name)


def test_a_loss_section_no_longer_takes_a_single_objective() -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(
            declaration() | {"loss": {"objective": {"kind": "coherence-loss"}}}
        )


def test_a_term_is_unscheduled_unless_it_declares_a_schedule() -> None:
    loss = minimal().loss
    assert loss is not None
    (term,) = loss.terms

    assert term.schedule is None


def test_a_declared_schedule_round_trips_through_the_run_document() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "loss": {
                "terms": [
                    {
                        "objective": {"kind": "coherence-loss"},
                        "weight": 0.5,
                        "schedule": {"start": 0.0, "steps": 10},
                    }
                ]
            }
        }
    )

    assert run.loss is not None
    assert run.loss.terms[0].schedule == WeightScheduleSpec(start=0.0, steps=10)
    assert RunSpec.model_validate_json(run.model_dump_json()) == run


@pytest.mark.parametrize(
    ("schedule", "detail"),
    [
        ({"start": -0.5, "steps": 10}, "start"),
        ({"start": float("inf"), "steps": 10}, "start"),
        ({"start": 0.0, "steps": 0}, "steps"),
        ({"start": 0.0}, "steps"),
        ({"start": 0.0, "steps": 2, "shape": "cosine"}, "shape"),
    ],
)
def test_an_invalid_schedule_is_refused(
    schedule: dict[str, float | str], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        WeightedLossSpec.model_validate(
            {"objective": {"kind": "pinning-loss"}, "schedule": schedule}
        )
