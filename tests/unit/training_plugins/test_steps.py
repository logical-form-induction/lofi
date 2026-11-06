"""Tests for installed training-step plugins."""

from typing import cast

import pytest

from lofi.data import Relational, SupervisionArm, Unlabeled
from lofi.mixing import ProportionalMixingSpec
from lofi.spec.trainers.generation import TermGenerationSpec
from lofi.spec.trainers.steps import (
    AlternatingStepSpec,
    ControllerStepSpec,
    GroundedStepSpec,
    JudgmentStepSpec,
    ParserStepSpec,
    StepSpec,
    SupervisedWarmStartStepSpec,
    TermsStepSpec,
    TokenWarmupStepSpec,
)
from lofi.training_plugins.steps import (
    merge_presentations,
    step_plugin,
    steps_presentation,
)


def declarations() -> tuple[StepSpec, ...]:
    """Return one complete declaration of every installed step kind."""
    return (
        ParserStepSpec(name="parser", dataset="sentences"),
        SupervisedWarmStartStepSpec(name="forms", dataset="sentences"),
        JudgmentStepSpec(name="judgments", dataset="pairs"),
        GroundedStepSpec(name="grounded", dataset="pairs"),
        TokenWarmupStepSpec(name="tokens", dataset="sentences", space="categories"),
        TermsStepSpec(
            name="terms",
            dataset="sentences",
            space="categories",
            combinators=True,
            generation=TermGenerationSpec(max_depth=2),
        ),
        ControllerStepSpec(
            name="controller",
            dataset="sentences",
            space="categories",
            max_depth=2,
        ),
        AlternatingStepSpec(
            name="nested",
            dataset="sentences",
            tasks=(
                ParserStepSpec(name="first", dataset="sentences"),
                ParserStepSpec(name="second", dataset="sentences"),
            ),
            mixing=ProportionalMixingSpec(),
        ),
    )


@pytest.mark.parametrize("declaration", declarations(), ids=lambda held: held.kind)
def test_every_step_declaration_reaches_its_exact_installed_builder(
    declaration: StepSpec,
) -> None:
    plugin = step_plugin(declaration)

    assert plugin.spec is type(declaration)


PRESENTED: dict[str, dict[str, frozenset[SupervisionArm]]] = {
    "parser": {"sentences": frozenset({Unlabeled})},
    "supervised-warm-start": {"sentences": frozenset({Unlabeled})},
    "judgment": {"pairs": frozenset({Relational})},
    "grounded": {"pairs": frozenset({Relational})},
    "token-warm-up": {},
    "terms": {},
    "controller": {},
    "alternating": {"sentences": frozenset({Unlabeled})},
}


@pytest.mark.parametrize("declaration", declarations(), ids=lambda held: held.kind)
def test_every_step_declares_the_arm_it_presents_to_its_featurizer(
    declaration: StepSpec,
) -> None:
    presented = step_plugin(declaration).presented(declaration)

    assert dict(presented) == PRESENTED[cast("str", declaration.kind)]


def test_a_sequence_of_steps_presents_every_arm_by_dataset() -> None:
    assert steps_presentation(declarations()) == {
        "sentences": frozenset({Unlabeled}),
        "pairs": frozenset({Relational}),
    }
    assert merge_presentations(
        (
            {"train": frozenset({Unlabeled})},
            {"train": frozenset({Relational})},
        )
    ) == {"train": frozenset({Unlabeled, Relational})}
