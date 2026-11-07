from pathlib import Path
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import PACKAGED_CONFIG_ROOT, Dependency, RunSpecCodec
from lofi.spec.components.run import EvaluatorSpec, ParseSpec, ViterbiSemiringSpec
from lofi.spec.evaluators.coherence_diagnostics import (
    CoherenceDiagnosticsSettings,
    CoherenceDiagnosticsSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT))


def document() -> dict[str, JsonValue]:
    """Return the complete held-out coherence diagnostic declaration."""
    return {
        "kind": "coherence-diagnostics",
        "interval": {"replicates": 50, "level": 0.95},
        "objective": "coherence",
        "readout": "category-terms",
        "root_mass": "sentences",
        "collapse_threshold": 0.99,
        "astar": {},
        "decode": {"kind": "viterbi"},
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


def test_the_document_declares_every_coherence_diagnostic_row() -> None:
    declared = EvaluatorSpec.model_validate(document())

    assert type(declared) is CoherenceDiagnosticsSpec
    assert declared.dependencies() == (Dependency("category-terms", "terms-readout"),)
    assert declared.metrics() == (
        "coherence-loss-per-token",
        "coherence-loss-by-action",
        "coherence-loss-by-span-length",
        "controller-entropy-per-span",
        "decoder-entropy-per-span",
        "type-distribution-collapse-rate",
        "coherence-accuracy-correlation",
        "combinator-use-jsd",
        "root-mass-agreement",
    )
    assert isinstance(declared.settings(), CoherenceDiagnosticsSettings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective", ""),
        ("readout", ""),
        ("root_mass", ""),
        ("collapse_threshold", 0.0),
        ("collapse_threshold", 1.0),
    ],
)
def test_unnamed_roles_and_non_strict_thresholds_are_refused(
    field: str, value: JsonValue
) -> None:
    payload = document()
    payload[field] = value

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(payload)


def test_the_development_document_reaches_the_registered_evaluator() -> None:
    path = (
        Path(__file__).parents[4]
        / "lofi/configs/run/synthetic-coherence-diagnostics.yaml"
    )
    run = CODEC(path)

    assert isinstance(run.evaluators["coherence-diagnostics"], CoherenceDiagnosticsSpec)


def test_symbolic_root_mass_needs_marginals_under_any_decode() -> None:
    declared = CoherenceDiagnosticsSpec.model_validate(document())

    assert (
        declared.parse_refusal(ParseSpec(retain="forest", reachability="all")) is None
    )
    assert "does not supply" in str(
        declared.parse_refusal(
            ParseSpec(
                retain="forest", reachability="all", semiring=ViterbiSemiringSpec()
            )
        )
    )
