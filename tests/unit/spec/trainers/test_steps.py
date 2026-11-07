"""Tests for declared composed-training steps."""

from types import MappingProxyType
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.mixing import ProportionalMixingSpec, RoundRobinMixingSpec
from lofi.spec import RunRole, TrainingBatchKind
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec
from lofi.spec.trainers.generation import TermGenerationSpec
from lofi.spec.trainers.steps import (
    AlternatingStepSpec,
    ControllerStepSpec,
    ParserStepSpec,
    StepSpec,
    TermsStepSpec,
    TokenWarmupStepSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


@pytest.mark.parametrize(
    "document",
    [
        {"kind": "parser", "name": "parser", "dataset": "sentences"},
        {
            "kind": "supervised-warm-start",
            "name": "supervised",
            "dataset": "sentences",
        },
        {"kind": "judgment", "name": "judgment", "dataset": "pairs"},
        {"kind": "grounded", "name": "grounded", "dataset": "pairs"},
        {
            "kind": "token-warm-up",
            "name": "tokens",
            "dataset": "sentences",
            "space": "categories",
        },
        {
            "kind": "terms",
            "name": "terms",
            "dataset": "sentences",
            "space": "categories",
            "combinators": True,
            "generation": {"max_depth": 3},
        },
        {
            "kind": "controller",
            "name": "controller",
            "dataset": "sentences",
            "space": "categories",
            "max_depth": 3,
        },
        {
            "kind": "alternating",
            "name": "nested",
            "dataset": "sentences",
            "tasks": [
                {"kind": "parser", "name": "one", "dataset": "sentences"},
                {"kind": "grounded", "name": "two", "dataset": "pairs"},
            ],
            "mixing": {"kind": "round-robin"},
        },
    ],
)
def test_every_step_kind_is_reachable_through_the_family_root(
    document: dict[str, JsonValue],
) -> None:
    declared = StepSpec.model_validate(document)

    assert declared.kind == document["kind"]


def test_nested_alternation_unifies_dataset_capabilities_and_streams() -> None:
    declared = AlternatingStepSpec(
        name="joint",
        dataset="sentences",
        tasks=(
            ParserStepSpec(name="parser", dataset="sentences"),
            StepSpec.model_validate(
                {"kind": "grounded", "name": "grounded", "dataset": "pairs"}
            ),
        ),
        mixing=ProportionalMixingSpec(),
    )

    assert declared.dataset_requirements() == MappingProxyType(
        {
            "sentences": frozenset({TrainingBatchKind.SENTENCES}),
            "pairs": frozenset({TrainingBatchKind.INFERENCE_PAIRS}),
        }
    )
    assert declared.featurized_datasets() == {"sentences", "pairs"}
    assert declared.sampler_purposes() == {
        (role, purpose)
        for role in ("sentences", "pairs")
        for purpose in (
            "validation-split",
            "training-batches",
            "validation-batches",
        )
    }


def test_an_alternating_step_requires_two_distinct_tasks() -> None:
    with pytest.raises(dx.ValidationError, match="at least two"):
        AlternatingStepSpec(
            name="joint",
            dataset="sentences",
            tasks=(ParserStepSpec(name="parser", dataset="sentences"),),
            mixing=RoundRobinMixingSpec(),
        )


@pytest.mark.parametrize(
    ("declared", "purpose"),
    [
        (
            TermsStepSpec(
                name="terms",
                dataset="sentences",
                space="categories",
                combinators=True,
                generation=TermGenerationSpec(max_depth=2),
                stop=AccuracyStoppingSpec(
                    metric="loss/categories/combinators/exact_match",
                    target=0.95,
                    max_steps=10,
                ),
            ),
            "terms-held-out-batches",
        ),
        (
            ControllerStepSpec(
                name="controller",
                dataset="sentences",
                space="categories",
                max_depth=2,
                stop=AccuracyStoppingSpec(
                    metric="loss/controller/accuracy",
                    target=0.95,
                    max_steps=10,
                ),
            ),
            "controller-held-out-batches",
        ),
    ],
)
def test_accuracy_targeted_generated_steps_declare_a_held_out_stream(
    declared: TermsStepSpec | ControllerStepSpec,
    purpose: str,
) -> None:
    target = declared.accuracy_target()

    assert (declared.dataset, purpose) in declared.sampler_purposes()
    assert target is not None
    assert target.metric in declared.evaluated_metrics()


def test_a_step_refuses_an_objective_the_loss_does_not_declare() -> None:
    nested = AlternatingStepSpec(
        name="joint",
        dataset="sentences",
        tasks=(
            ParserStepSpec(name="one", dataset="sentences", objectives=("pinning",)),
            ParserStepSpec(name="two", dataset="sentences", objectives=("simplicity",)),
        ),
        mixing=ProportionalMixingSpec(),
    )

    nested.refuse_objectives(frozenset({"pinning", "simplicity"}))

    with pytest.raises(
        ValueError, match=r"task 'two' selects absent objectives \['simplicity'\]"
    ):
        nested.refuse_objectives(frozenset({"pinning"}))


@pytest.mark.parametrize(
    "document",
    [
        {"kind": "token-warm-up", "space": "categories"},
        {
            "kind": "terms",
            "space": "categories",
            "combinators": False,
            "generation": {"max_depth": 3},
        },
        {"kind": "controller", "space": "categories", "max_depth": 3},
        {
            "kind": "alternating",
            "tasks": [
                {"kind": "parser", "name": "one", "dataset": "sentences"},
                {"kind": "grounded", "name": "two", "dataset": "pairs"},
            ],
            "mixing": {"kind": "round-robin"},
        },
    ],
)
def test_a_step_that_trains_no_loss_objective_refuses_to_select_one(
    document: dict[str, JsonValue],
) -> None:
    declared = StepSpec.model_validate(
        {
            **document,
            "name": "unscored",
            "dataset": "sentences",
            "objectives": ["pinning"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"task 'unscored' .* trains no loss objective, so it cannot select "
        r"\['pinning'\]",
    ):
        declared.refuse_objectives(frozenset({"pinning"}))


@pytest.mark.parametrize(
    "kind", ["parser", "supervised-warm-start", "judgment", "grounded"]
)
def test_a_step_that_trains_loss_objectives_selects_declared_ones(kind: str) -> None:
    declared = StepSpec.model_validate(
        {
            "kind": kind,
            "name": "scored",
            "dataset": "sentences",
            "objectives": ["pinning"],
        }
    )

    declared.refuse_objectives(frozenset({"pinning"}))


def test_a_terms_step_evaluates_combinators_only_when_it_trains_them() -> None:
    autoencoding = TermsStepSpec(
        name="inventory",
        dataset="sentences",
        space="categories",
        combinators=False,
        generation=TermGenerationSpec(max_depth=2),
    )

    assert autoencoding.accuracy_target() is None
    assert autoencoding.evaluated_metrics() == {
        "loss/categories/autoencoding/exact_match"
    }
    assert autoencoding.sampler_purposes() == {("sentences", "terms-batches")}


def test_constraint_consumption_is_declared_by_the_step_and_its_nested_tasks() -> None:
    warmup = TokenWarmupStepSpec(name="tokens", dataset="sentences", space="categories")
    parser = ParserStepSpec(name="parser", dataset="sentences")
    nested = AlternatingStepSpec(
        name="joint",
        dataset="sentences",
        tasks=(parser, warmup),
        mixing=RoundRobinMixingSpec(),
    )

    assert warmup.consumes_constraints()
    assert not parser.consumes_constraints()
    assert nested.consumes_constraints()


def test_generated_steps_name_the_samplers_their_instances_come_from() -> None:
    controller = ControllerStepSpec(
        name="controller", dataset="sentences", space="categories", max_depth=2
    )
    terms = TermsStepSpec(
        name="inventory",
        dataset="sentences",
        space="categories",
        combinators=False,
        generation=TermGenerationSpec(max_depth=2),
    )
    pairs = TermsStepSpec(
        name="combinators",
        dataset="sentences",
        space="categories",
        combinators=True,
        generation=TermGenerationSpec(max_depth=2),
    )
    joint = AlternatingStepSpec(
        name="joint",
        dataset="sentences",
        tasks=(ParserStepSpec(name="parser", dataset="sentences"), terms),
        mixing=ProportionalMixingSpec(),
    )

    assert controller.sampler_roles() == {RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER}
    assert terms.sampler_roles() == {RunRole.TYPE_SAMPLER}
    assert pairs.sampler_roles() == {RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER}
    assert ParserStepSpec(name="parser", dataset="sentences").sampler_roles() == set()
    assert joint.sampler_roles() == {RunRole.TYPE_SAMPLER}
