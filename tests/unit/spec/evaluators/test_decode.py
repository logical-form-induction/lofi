from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.interpret import ConstituentUtility
from lofi.spec import Dependency
from lofi.spec._errors import SemiringCapabilityError
from lofi.spec.components.run import EvaluatorSpec, ParseSpec
from lofi.spec.evaluators.conventions import ConventionSpec
from lofi.spec.evaluators.decode import (
    DecodeEvaluatorSettings,
    DecodeEvaluatorSpec,
    MbrDecision,
    ViterbiDecision,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


CONVENTIONS: dict[str, JsonValue] = {
    "punctuation": "keep",
    "unary_chains": "keep",
    "trivial_spans": "keep",
    "minimum_length": 0,
    "aggregation": "sentence",
    "coverage": "zero",
    "self_arcs": "allow",
    "gold_derivation": "canonical",
}


def _document() -> dict[str, JsonValue]:
    return {
        "kind": "decode",
        "interval": {"replicates": 50, "level": 0.95},
        "readout": "category-terms",
        "output": "lexical-types",
        "astar": {
            "strategy": {"kind": "binary"},
            "root_mode": {"kind": "free"},
            "heuristic": {"kind": "zero"},
            "proposer": {"kind": "supertag-beam"},
        },
        "decode": {
            "kind": "mbr",
            "utility": {"kind": "constituent"},
            "candidates": 16,
        },
        "conventions": CONVENTIONS,
    }


def test_the_decode_variant_resolves_every_open_search_choice() -> None:
    declared = EvaluatorSpec.model_validate(_document())

    assert type(declared) is DecodeEvaluatorSpec
    assert declared.dependencies() == (Dependency("category-terms", "terms-readout"),)
    assert declared.metrics() == (
        "decode-candidates",
        "decode-parsed",
        "decode-refused",
        "decode-attempts",
        "decode-widenings",
    )
    settings = declared.settings()

    assert isinstance(settings, DecodeEvaluatorSettings)
    assert settings.astar is not None
    assert settings.astar.strategy.name == "binary"
    assert settings.astar.root_mode.name == "free"
    assert settings.astar.heuristic.name == "zero"
    assert settings.astar.proposer is not None
    assert settings.astar.proposer.name == "supertag-beam"
    assert settings.decode is not None
    assert settings.decode == MbrDecision(ConstituentUtility(), 16)
    assert dict(settings.decode.conventions) == {
        "decode_method": "mbr",
        "decode_utility": "constituent",
        "decode_candidates": 16,
    }
    assert settings.conventions == ConventionSpec.model_validate(CONVENTIONS).settings()


def test_the_default_astar_setup_is_declared_in_the_document_shape() -> None:
    document = _document()
    document["astar"] = {}
    document["decode"] = {"kind": "viterbi"}
    declared = DecodeEvaluatorSpec.model_validate(document)
    settings = declared.settings()

    assert settings.astar is not None
    assert settings.astar.strategy.name == "linear"
    assert settings.astar.root_mode.name == "goal"
    assert settings.astar.heuristic.name == "supertag-outside"
    assert settings.astar.proposer is None
    assert settings.decode is not None
    assert settings.decode == ViterbiDecision()
    assert settings.decode.conventions["decode_candidates"] == 1


def _parse(**fields: JsonValue) -> ParseSpec:
    return ParseSpec.model_validate(
        {"retain": "forest", "reachability": "all"} | fields
    )


def test_a_viterbi_decode_runs_under_a_declared_viterbi_parse() -> None:
    document = _document()
    document["decode"] = {"kind": "viterbi"}
    declared = DecodeEvaluatorSpec.model_validate(document)

    assert declared.parse_refusal(_parse(semiring={"kind": "viterbi"})) is None
    assert declared.parse_refusal(_parse()) is None


@pytest.mark.parametrize(
    ("parse", "reason"),
    [
        (
            {"semiring": {"kind": "viterbi"}},
            "parse.semiring 'viterbi' does not supply",
        ),
        ({"retain": "inside"}, "declare parse.retain: forest"),
        (
            {"reachability": "roots"},
            "root mode 'free' reads spanning items",
        ),
    ],
)
def test_a_run_parse_the_astar_decode_cannot_use_is_refused(
    parse: dict[str, JsonValue],
    reason: str,
) -> None:
    declared = DecodeEvaluatorSpec.model_validate(_document())
    refusal = declared.parse_refusal(_parse(**parse))

    assert refusal is not None
    assert reason in str(refusal)


@pytest.mark.parametrize("kind", ["counting", "boolean", "forest"])
def test_astar_refuses_a_semiring_without_log_probabilities_by_name(
    kind: str,
) -> None:
    document = _document()
    document["decode"] = {"kind": "viterbi"}
    declared = DecodeEvaluatorSpec.model_validate(document)
    refusal = declared.parse_refusal(_parse(semiring={"kind": kind}))

    assert isinstance(refusal, SemiringCapabilityError)
    assert refusal.capability == "log-probability"
    assert str(refusal) == (
        "the A* decode needs float log-probability weights, which "
        f"parse.semiring '{kind}' does not supply"
    )


def test_an_mbr_decode_refuses_a_semiring_without_marginals_by_name() -> None:
    refusal = DecodeEvaluatorSpec.model_validate(_document()).parse_refusal(
        _parse(semiring={"kind": "viterbi"})
    )

    assert isinstance(refusal, SemiringCapabilityError)
    assert refusal.capability == "marginals"


def test_goal_roots_survive_a_root_reachability_parse() -> None:
    document = _document()
    document["astar"] = {"root_mode": {"kind": "goal"}}
    declared = DecodeEvaluatorSpec.model_validate(document)

    assert declared.parse_refusal(_parse(reachability="roots")) is None


def test_a_plain_beam_decode_reads_no_symbolic_parse() -> None:
    declared = DecodeEvaluatorSpec.model_validate(
        {
            "kind": "decode",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "output": "logical-forms",
            "conventions": CONVENTIONS,
        }
    )

    assert declared.parse_refusal(_parse(retain="inside")) is None


def test_a_plain_logical_form_beam_needs_no_symbolic_decode() -> None:
    declared = DecodeEvaluatorSpec.model_validate(
        {
            "kind": "decode",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "decoded-forms",
            "output": "logical-forms",
            "conventions": CONVENTIONS,
        }
    )

    assert declared.metrics() == ("decode-candidates",)
    assert declared.settings().astar is None
    assert declared.settings().decode is None


@pytest.mark.parametrize(
    "change",
    [
        {"readout": ""},
        {"output": "logical-forms"},
        {"decode": None},
        {"astar": None},
        {
            "decode": {
                "kind": "mbr",
                "utility": {"kind": "constituent"},
                "candidates": 0,
            }
        },
        {"astar": {"semiring": {"kind": "viterbi"}}},
        {"astar": {"heuristic": {"kind": "unknown"}}},
        {"astar": {"proposer": {"kind": "unknown"}}},
        {"beam": 4},
    ],
)
def test_incoherent_unknown_or_superseded_routes_are_refused(
    change: dict[str, JsonValue],
) -> None:
    document = _document()
    document.update(change)

    with pytest.raises(dx.ValidationError):
        EvaluatorSpec.model_validate(document)


def test_the_variant_round_trips_through_its_json_shape() -> None:
    declared = EvaluatorSpec.model_validate(_document())

    assert EvaluatorSpec.model_validate_json(declared.model_dump_json()) == declared
