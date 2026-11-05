"""Tests for the held-out likelihood evaluator declaration."""

import didactic.api as dx
import pytest

from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.held_out_likelihood import (
    HeldOutLikelihoodSettings,
    HeldOutLikelihoodSpec,
)

CONVENTIONS = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def declaration() -> HeldOutLikelihoodSpec:
    """Return one valid held-out likelihood declaration."""
    return HeldOutLikelihoodSpec.model_validate(
        {
            "kind": "held-out-likelihood",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "types",
            "output": "sentences",
            "measurements": ["string", "derivation", "description-length"],
            "conventions": CONVENTIONS,
        }
    )


def test_the_variant_declares_both_readout_capabilities_and_seven_rows() -> None:
    declared = EvaluatorSpec.model_validate(declaration().model_dump())

    assert type(declared) is HeldOutLikelihoodSpec
    assert tuple(
        (dependency.role, dependency.port) for dependency in declared.dependencies()
    ) == (("types", "span-readout"), ("types", "terms-readout"))
    assert declared.metrics() == (
        "string-log-likelihood-per-sentence",
        "string-log-likelihood-per-token",
        "derivational-log-likelihood-per-sentence",
        "derivational-log-likelihood-per-token",
        "derivational-log-likelihood-per-sentence",
        "derivational-log-likelihood-per-token",
        "description-length-nats",
    )


def test_the_variant_returns_immutable_build_settings() -> None:
    declared = declaration()

    assert declared.settings() == HeldOutLikelihoodSettings(
        "types",
        "sentences",
        declared.symbol_cost,
        declared.conventions.settings(),
        ("string", "derivation", "description-length"),
    )


@pytest.mark.parametrize("field", ["readout", "output"])
def test_a_readout_role_and_output_must_be_named(field: str) -> None:
    payload = declaration().model_dump()
    payload[field] = ""

    with pytest.raises(dx.ValidationError, match="held-out likelihood"):
        HeldOutLikelihoodSpec.model_validate(payload)


def test_selected_measurements_determine_dependencies_and_rows() -> None:
    payload = declaration().model_dump()
    payload["measurements"] = ("description-length", "string")

    declared = HeldOutLikelihoodSpec.model_validate(payload)

    assert declared.measurements == ("string", "description-length")
    assert tuple(
        (dependency.role, dependency.port) for dependency in declared.dependencies()
    ) == (("types", "span-readout"), ("types", "terms-readout"))
    assert declared.metrics() == (
        "string-log-likelihood-per-sentence",
        "string-log-likelihood-per-token",
        "description-length-nats",
    )


@pytest.mark.parametrize("measurements", [(), ("string", "string")])
def test_measurements_must_be_nonempty_and_distinct(
    measurements: tuple[str, ...],
) -> None:
    payload = declaration().model_dump()
    payload["measurements"] = measurements

    with pytest.raises(dx.ValidationError, match="measurement"):
        HeldOutLikelihoodSpec.model_validate(payload)
