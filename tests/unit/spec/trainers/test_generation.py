"""Tests for the generated-term distribution declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.trainers import TermGenerationSpec
from lofi.spec.trainers.generation import (
    autoencoding_objective,
    combinator_objective,
    terms_metrics,
)
from lofi.spec.type_samplers.distribution import (
    ArityBranchingSpec,
    UniformChildDepthSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_generation_declares_only_its_depth_and_keeps_the_sampler_defaults() -> None:
    generation = TermGenerationSpec.model_validate({"max_depth": 3})

    assert generation.max_depth == 3
    assert generation.child_depth == UniformChildDepthSpec()
    assert generation.branching == ArityBranchingSpec()
    assert generation.constructor_weights == {}
    assert generation.teacher_forcing_saturation is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_depth", 0),
        ("child_depth", {"kind": "power", "exponent": float("inf")}),
        ("branching", {"by_arity": [{"arity": 2, "deepest": [1.0]}]}),
        ("constructor_weights", {"application": -0.1}),
        ("teacher_forcing_saturation", 0.0),
        ("teacher_forcing_saturation", 1.1),
    ],
)
def test_an_invalid_generation_is_refused(field: str, value: JsonValue) -> None:
    with pytest.raises(dx.ValidationError):
        TermGenerationSpec.model_validate({"max_depth": 2, field: value})


def test_term_objectives_are_named_by_their_space() -> None:
    assert autoencoding_objective("forms") == "forms/autoencoding"
    assert combinator_objective("forms") == "forms/combinators"


def test_the_evaluated_metrics_cover_exactly_the_trained_objectives() -> None:
    assert terms_metrics((("types", False), ("forms", True))) == {
        "loss/types/autoencoding/exact_match",
        "loss/forms/autoencoding/exact_match",
        "loss/forms/combinators/exact_match",
    }
