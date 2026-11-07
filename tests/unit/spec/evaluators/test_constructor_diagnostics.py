from pathlib import Path
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import PACKAGED_CONFIG_ROOT, Dependency, RunSpecCodec
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.constructor_diagnostics import (
    ConstructorDiagnosticSettings,
    ConstructorDiagnosticSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT))


def document() -> dict[str, JsonValue]:
    """Return an A-times constructor-diagnostic declaration."""
    return {
        "kind": "constructor-diagnostics",
        "interval": {"replicates": 50, "level": 0.95},
        "readout": "category-terms",
        "view": {
            "representative": "NP",
            "roles": {
                "arrow": ["result", "argument"],
                "*": ["left-factor", "right-factor"],
            },
        },
        "normalization": "nodes",
        "widths": [1, 4],
        "depths": [1, 2, 4],
        "collapse_margin": 0.5,
        "conventions": {
            "punctuation": "keep",
            "unary_chains": "keep",
            "trivial_spans": "keep",
            "minimum_length": 0,
            "aggregation": "sentence",
            "coverage": "zero",
            "self_arcs": "allow",
            "gold_derivation": "canonical",
        },
    }


def test_an_arrow_product_document_declares_every_constructor_diagnostic_row() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert type(declared) is ConstructorDiagnosticSpec
    assert declared.dependencies() == (Dependency("category-terms", "terms-readout"),)
    assert declared.metrics() == (
        "beam-accuracy@1",
        "beam-accuracy@4",
        "typeless-accuracy@1",
        "typeless-accuracy@4",
        "truncated-accuracy@1",
        "truncated-accuracy@2",
        "truncated-accuracy@4",
        "mean-reciprocal-rank",
        "primitive-induced-count",
        "primitive-gold-count",
        "primitive-inventory-exact",
        "primitive-f1",
        "constructor-induced-count",
        "constructor-gold-count",
        "constructor-inventory-exact",
        "constructor-role-information",
        "lexical-type-f1",
        "lexical-token-accuracy",
        "depth-gap",
        "depth-collapse",
    )
    settings = declared.settings()
    assert settings.collapse_margin == 0.5
    assert isinstance(settings, ConstructorDiagnosticSettings)
    assert settings.view.roles == (
        ("arrow", ("result", "argument")),
        ("*", ("left-factor", "right-factor")),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readout", ""),
        ("widths", []),
        ("widths", [2, 2]),
        ("depths", [0]),
        ("normalization", "slash"),
        ("collapse_margin", -0.1),
        ("collapse_margin", float("inf")),
        ("view", {"representative": "NP", "roles": {}}),
    ],
)
def test_empty_nonascending_or_slash_specific_routes_are_refused(
    field: str, value: JsonValue
) -> None:
    payload = document()
    payload[field] = value

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_declaration_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared


def test_the_development_run_declares_all_four_beam_normalizations() -> None:
    path = (
        Path(__file__).parents[4]
        / "lofi/configs/run/synthetic-constructor-diagnostics.yaml"
    )
    run = CODEC(path)

    assert {
        declaration.normalization
        for declaration in run.evaluators.values()
        if isinstance(declaration, ConstructorDiagnosticSpec)
    } == {"none", "depth", "nodes", "terminals"}
