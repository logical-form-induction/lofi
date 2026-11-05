import math
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.coherence import (
    CoherenceDivergenceSpec,
    CoherenceEstimator,
    CoherenceLossSpec,
    CrossEntropyDivergenceSpec,
    FenchelYoungDivergenceSpec,
    KLDivergenceSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec.components.model import ModelSection


def test_the_tag_reaches_the_coherence_variant_through_the_family_root() -> None:
    declared = LossSpec.model_validate(
        {
            "kind": "coherence",
            "readout": "syntactic-types",
            "space": "syntactic-combinators",
        }
    )

    assert type(declared) is CoherenceLossSpec
    assert declared == CoherenceLossSpec(
        readout="syntactic-types",
        space="syntactic-combinators",
    )
    assert LossSpec.__variants__["coherence"] is CoherenceLossSpec


def test_the_defaults_name_exact_kl_and_a_thousand_step_warmup() -> None:
    declared = CoherenceLossSpec(readout="types", space="combinators")

    assert declared.estimator is CoherenceEstimator.EXACT
    assert declared.divergence == KLDivergenceSpec()
    assert declared.warmup == 1000
    assert declared.dependencies() == (Dependency("types", "coherence-readout"),)
    assert declared.declared_output is None


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"kind": "kl"}, KLDivergenceSpec()),
        ({"kind": "cross-entropy"}, CrossEntropyDivergenceSpec()),
        (
            {"kind": "fenchel-young", "alpha": 1.75},
            FenchelYoungDivergenceSpec(alpha=1.75),
        ),
    ],
)
def test_each_closed_divergence_variant_reaches_its_declaration(
    document: dict[str, JsonValue],
    expected: CoherenceDivergenceSpec,
) -> None:
    assert CoherenceDivergenceSpec.model_validate(document) == expected


@pytest.mark.parametrize("estimator", list(CoherenceEstimator))
def test_each_action_estimator_round_trips_by_its_declared_name(
    estimator: CoherenceEstimator,
) -> None:
    declared = CoherenceLossSpec(
        readout="types",
        space="combinators",
        estimator=estimator,
    )
    restored = LossSpec.model_validate_json(declared.model_dump_json())

    assert type(restored) is CoherenceLossSpec
    assert restored.estimator is estimator


def test_truncated_coherence_requests_its_optional_residual_override() -> None:
    inherited = CoherenceLossSpec(
        readout="types",
        space="combinators",
        estimator=CoherenceEstimator.TRUNCATED,
    )
    overridden = inherited.with_(residual_mass=0.05)
    inherited_request = inherited.approximation_request()
    overridden_request = overridden.approximation_request()

    assert inherited_request is not None
    assert overridden_request is not None
    assert inherited_request.residual_mass is None
    assert overridden_request.residual_mass == pytest.approx(0.05)
    assert overridden.settings().residual_mass == pytest.approx(0.05)


def test_exact_coherence_refuses_a_residual_override() -> None:
    with pytest.raises(dx.ValidationError, match=r"exact.*residual_mass"):
        CoherenceLossSpec(
            readout="types",
            space="combinators",
            residual_mass=0.1,
        )


@pytest.mark.parametrize("warmup", [None, 0, 1, 37])
def test_nullable_nonnegative_warmup_values_round_trip_exactly(
    warmup: int | None,
) -> None:
    declared = CoherenceLossSpec(
        readout="types",
        space="combinators",
        warmup=warmup,
    )

    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared
    assert declared.warmup == warmup


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            {"kind": "coherence", "space": "combinators"},
            "readout",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
            },
            "space",
        ),
        (
            {
                "kind": "coherence",
                "readout": "",
                "space": "combinators",
            },
            "component role",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "",
            },
            "component role",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "combinators",
                "estimator": "straight-through",
            },
            "straight-through",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "combinators",
                "divergence": {"kind": "jensen-shannon"},
            },
            "jensen-shannon",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "combinators",
                "warmup": -1,
            },
            "warmup",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "combinators",
                "warmup": True,
            },
            "warmup",
        ),
        (
            {
                "kind": "coherence",
                "readout": "types",
                "space": "combinators",
                "temperature": 0.5,
            },
            "temperature",
        ),
    ],
)
def test_invalid_or_extra_coherence_fields_are_refused_by_name(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


@pytest.mark.parametrize("alpha", [0.99, math.inf, -math.inf, math.nan, True])
def test_fenchel_young_alone_accepts_a_finite_alpha_of_at_least_one(
    alpha: float,
) -> None:
    with pytest.raises(dx.ValidationError, match="alpha"):
        CoherenceDivergenceSpec.model_validate(
            {"kind": "fenchel-young", "alpha": alpha}
        )


@pytest.mark.parametrize("kind", ["kl", "cross-entropy"])
def test_alpha_is_not_a_field_of_the_other_divergences(kind: str) -> None:
    with pytest.raises(dx.ValidationError, match="alpha"):
        CoherenceDivergenceSpec.model_validate({"kind": kind, "alpha": 1.5})


def test_a_complete_declaration_round_trips_through_json() -> None:
    declared = CoherenceLossSpec(
        readout="logical-forms",
        space="logical-combinators",
        estimator=CoherenceEstimator.TRUNCATED,
        divergence=FenchelYoungDivergenceSpec(alpha=2.0),
        warmup=None,
    )

    restored = LossSpec.model_validate_json(declared.model_dump_json())

    assert restored == declared
    assert restored.model_dump_json() == declared.model_dump_json()


def test_coherence_refuses_a_space_the_model_does_not_declare(
    packaged_model: ModelSection,
) -> None:
    CoherenceLossSpec(readout="category-terms", space="categories").refuse_model(
        packaged_model, "loss.terms.coherence"
    )

    with pytest.raises(
        ValueError,
        match=r"loss\.terms\.coherence\.space: the objective names space 'types', "
        r"which is not declared among \['categories'\]",
    ):
        CoherenceLossSpec(readout="category-terms", space="types").refuse_model(
            packaged_model, "loss.terms.coherence"
        )
