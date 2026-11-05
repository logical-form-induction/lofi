import didactic.api as dx
import pytest

from lofi.spec._ports import Dependency
from lofi.spec.components.run import ParseSpec, ViterbiSemiringSpec
from lofi.spec.evaluators.structural_diagnostics import StructuralDiagnosticsSpec


def test_structural_diagnostics_declare_every_requested_report_family() -> None:
    spec = StructuralDiagnosticsSpec.model_validate(
        {
            "kind": "structural-diagnostics",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "category-terms",
            "astar": {"proposer": {"kind": "supertag-beam"}},
            "decode": {"kind": "viterbi"},
            "derivation_counts": True,
            "dependency_interpretation": "dependency",
            "labels": ["S", "NP"],
            "lengths": [2, 3],
            "depths": [0, 1],
            "attachment_band": "all",
            "attachment_limit": None,
            "attachment_direction": "directed",
            "dependency_mode": "labeled-directed",
            "dependency_measure": "f-score",
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
    )

    proposer = spec.settings().astar.proposer
    assert spec.dependencies() == (Dependency("category-terms", "terms-readout"),)
    assert proposer is not None
    assert proposer.name == "supertag-beam"
    assert spec.settings().metrics == spec.metrics()
    assert "bracketing-baseline:left" in spec.metrics()
    assert "bracketing-baseline:right" in spec.metrics()
    assert "bracketing-baseline:balanced" in spec.metrics()
    assert "normal-form-retention" in spec.metrics()
    assert any(name.startswith("attachment:") for name in spec.metrics())
    assert any(name.startswith("dependency:") for name in spec.metrics())


def test_structural_diagnostics_can_omit_the_exact_derivation_census() -> None:
    spec = StructuralDiagnosticsSpec.model_validate(
        {
            "kind": "structural-diagnostics",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "category-terms",
            "astar": {"proposer": {"kind": "supertag-beam"}},
            "decode": {"kind": "viterbi"},
            "derivation_counts": False,
            "dependency_interpretation": "dependency",
            "labels": [],
            "lengths": [],
            "depths": [],
            "attachment_band": "all",
            "attachment_limit": None,
            "attachment_direction": "directed",
            "dependency_mode": "labeled-directed",
            "dependency_measure": "f-score",
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
    )

    assert spec.settings().derivation_counts is False
    assert not any("derivation-count" in name for name in spec.metrics())
    assert "normal-form-retention" not in spec.metrics()


def test_structural_diagnostics_require_a_checked_rule_proposer() -> None:
    with pytest.raises(dx.ValidationError, match="proposer"):
        StructuralDiagnosticsSpec.model_validate(
            {
                "kind": "structural-diagnostics",
                "interval": {"replicates": 20, "level": 0.95},
                "readout": "category-terms",
                "astar": {},
                "decode": {"kind": "viterbi"},
                "derivation_counts": False,
                "dependency_interpretation": "dependency",
                "labels": [],
                "lengths": [],
                "depths": [],
                "attachment_band": "all",
                "attachment_limit": None,
                "attachment_direction": "directed",
                "dependency_mode": "labeled-directed",
                "dependency_measure": "f-score",
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
        )


def test_an_mbr_structural_decode_is_refused_a_viterbi_run_parse() -> None:
    spec = StructuralDiagnosticsSpec.model_validate(
        {
            "kind": "structural-diagnostics",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "category-terms",
            "astar": {"proposer": {"kind": "supertag-beam"}},
            "decode": {"kind": "mbr", "candidates": 4},
            "derivation_counts": False,
            "dependency_interpretation": "dependency",
            "labels": [],
            "lengths": [],
            "depths": [],
            "attachment_band": "all",
            "attachment_limit": None,
            "attachment_direction": "directed",
            "dependency_mode": "labeled-directed",
            "dependency_measure": "f-score",
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
    )
    viterbi = ParseSpec(
        retain="forest", reachability="all", semiring=ViterbiSemiringSpec()
    )

    assert spec.parse_refusal(ParseSpec(retain="forest", reachability="all")) is None
    assert "does not supply" in str(spec.parse_refusal(viterbi))
