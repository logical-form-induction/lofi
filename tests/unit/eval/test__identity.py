import json
import subprocess
import sys
from copy import copy, deepcopy
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._identity import UNFROZEN, ReportIdentity, config_digest, freeze_number
from lofi.eval._values import (
    Aggregation,
    ConventionVector,
    CoveragePenalty,
    GoldDerivation,
    PunctuationPolicy,
    SelfArcPolicy,
    TrivialSpanPolicy,
    UnaryChainPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi._types import JsonValue


class _Config:
    def __init__(self, values: Mapping[str, JsonValue]) -> None:
        self._values = values

    def model_dump(self) -> Mapping[str, JsonValue]:
        return self._values


class _Grammar:
    def __init__(self, fingerprint: str = "grammar:é") -> None:
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


def _conventions(
    *,
    punctuation: PunctuationPolicy = "strip",
    unary_chains: UnaryChainPolicy = "collapse",
    trivial_spans: TrivialSpanPolicy = "drop",
    minimum_length: int = 3,
    aggregation: Aggregation = "sentence",
    coverage: CoveragePenalty = "zero",
    self_arcs: SelfArcPolicy = "forbid",
    gold_derivation: GoldDerivation = "canonical",
) -> ConventionVector:
    return ConventionVector(
        punctuation=punctuation,
        unary_chains=unary_chains,
        trivial_spans=trivial_spans,
        minimum_length=minimum_length,
        aggregation=aggregation,
        coverage=coverage,
        self_arcs=self_arcs,
        gold_derivation=gold_derivation,
    )


def _identity(
    *,
    config: Mapping[str, JsonValue] | None = None,
    grammar: str = "grammar:é",
    conventions: ConventionVector | None = None,
    freeze_id: str = "freeze/1",
) -> ReportIdentity:
    return ReportIdentity(
        _Config({"model": {"kind": "parser"}, "seed": 7} if config is None else config),
        _Grammar(grammar),
        _conventions() if conventions is None else conventions,
        freeze_id=freeze_id,
    )


def _canonical(value: JsonValue) -> str:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return sha256(text.encode("utf-8")).hexdigest()


def test_identity_uses_canonical_json_and_copies_the_grammar_fingerprint() -> None:
    config: dict[str, JsonValue] = {"z": [3, 2], "a": {"beta": False}}
    conventions = _conventions()
    identity = _identity(config=config, conventions=conventions)

    assert identity.config_hash == _canonical(config)
    assert identity.grammar_fingerprint == "grammar:é"
    assert identity.convention_hash == _canonical(dict(conventions.factors))
    assert identity.freeze_id == "freeze/1"
    assert identity.to_spec() == {
        "freeze_id": "freeze/1",
        "config_hash": identity.config_hash,
        "grammar_fingerprint": "grammar:é",
        "convention_hash": identity.convention_hash,
    }
    assert len(identity.config_hash) == 64
    assert identity.convention_hash is not None
    assert len(identity.convention_hash) == 64
    assert set(identity.config_hash + identity.convention_hash) <= set(
        "0123456789abcdef"
    )


@pytest.mark.parametrize(
    "conventions",
    [
        _conventions(punctuation="keep"),
        _conventions(unary_chains="keep"),
        _conventions(trivial_spans="keep"),
        _conventions(minimum_length=0),
        _conventions(aggregation="corpus"),
        _conventions(coverage="ignore"),
        _conventions(self_arcs="allow"),
        _conventions(gold_derivation="oracle"),
    ],
)
def test_each_convention_factor_changes_the_digest(
    conventions: ConventionVector,
) -> None:
    assert (
        _identity().convention_hash
        != _identity(conventions=conventions).convention_hash
    )


def test_a_different_freeze_is_a_different_identity() -> None:
    baseline = _identity()
    refrozen = _identity(freeze_id="freeze/2")

    assert baseline != refrozen
    assert baseline.config_hash == refrozen.config_hash
    assert refrozen.to_spec()["freeze_id"] == "freeze/2"
    assert "freeze_id='freeze/2'" in repr(refrozen)


def test_an_identity_without_parsing_conventions_records_no_convention_digest() -> None:
    identity = ReportIdentity(_Config({}), _Grammar(), None, freeze_id="freeze/1")

    assert identity.convention_hash is None
    assert identity.to_spec()["convention_hash"] is None
    assert identity != _identity()


def test_a_row_scored_outside_any_freeze_names_the_unfrozen_identity() -> None:
    unfrozen = _identity(freeze_id=UNFROZEN)

    assert unfrozen.to_spec()["freeze_id"] == "unfrozen"
    assert unfrozen != _identity()
    assert unfrozen.config_hash == _identity().config_hash

    with pytest.raises(ValueError, match="freeze identifier"):
        freeze_number(UNFROZEN)


def test_an_identity_cannot_be_built_without_a_freeze() -> None:
    constructor = cast("Callable[..., object]", ReportIdentity)

    with pytest.raises(TypeError, match="freeze_id"):
        constructor(_Config({}), _Grammar(), _conventions())


@pytest.mark.parametrize(
    "freeze_id",
    [
        "",
        "1",
        "freeze/paper-a",
        "freeze/A/1",
        "freeze//1",
        "freeze/0",
        "freeze/00",
        "freeze/01",
        "freeze/-1",
        "freeze/\uff11",
        "freeze/one",
        "freeze/1 ",
        "freeze/\t1",
        "freeze/1/extra",
    ],
)
def test_a_malformed_freeze_identifier_is_refused(freeze_id: str) -> None:
    with pytest.raises(ValueError, match="freeze identifier"):
        _identity(freeze_id=freeze_id)


def test_a_freeze_identifier_names_its_number() -> None:
    assert freeze_number("freeze/1") == 1
    assert freeze_number("freeze/10") == 10

    with pytest.raises(TypeError, match="not a string"):
        freeze_number(cast("str", 1))


def test_the_configuration_digest_is_the_identity_column() -> None:
    config: dict[str, JsonValue] = {"z": [3, 2], "a": {"beta": False}}

    assert config_digest(config) == _identity(config=config).config_hash
    assert config_digest({"a": {"beta": False}, "z": [3, 2]}) == config_digest(config)


def test_configuration_and_grammar_changes_reach_only_their_columns() -> None:
    baseline = _identity()
    config_changed = _identity(config={"model": {"kind": "other"}, "seed": 7})
    grammar_changed = _identity(grammar="grammar:ö")

    assert baseline.config_hash != config_changed.config_hash
    assert baseline.convention_hash == config_changed.convention_hash
    assert baseline.grammar_fingerprint != grammar_changed.grammar_fingerprint
    assert baseline.config_hash == grammar_changed.config_hash


def test_identity_accepts_a_conforming_non_dictionary_mapping_dump() -> None:
    config: Mapping[str, JsonValue] = MappingProxyType(
        {"model": {"kind": "parser"}, "seed": 7}
    )

    assert _identity(config=config).config_hash == _canonical(dict(config))


def test_identity_is_stable_in_a_fresh_process() -> None:
    script = r"""
import json
from lofi.eval import ConventionVector, ReportIdentity

class Config:
    def model_dump(self):
        return {"z": [3, 2], "a": {"beta": False}}

class Grammar:
    fingerprint = "grammar:é"

conventions = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)
print(json.dumps(
    ReportIdentity(
        Config(), Grammar(), conventions, freeze_id="freeze/1"
    ).to_spec(),
    sort_keys=True,
))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert (
        json.loads(completed.stdout)
        == _identity(config={"z": [3, 2], "a": {"beta": False}}).to_spec()
    )


def test_identity_is_frozen_and_copies_by_identity() -> None:
    identity = _identity()

    assert copy(identity) is identity
    assert deepcopy(identity) is identity
    assert identity == _identity()
    assert hash(identity) == hash(_identity())
    assert identity != identity.to_spec()
    assert "config_hash=" in repr(identity)

    with pytest.raises(AttributeError, match="immutable"):
        identity.__setattr__("_config_hash", "0" * 64)


def test_identity_rejects_raw_digests_mappings_and_invalid_sources() -> None:
    with pytest.raises(TypeError, match="run specification is not one"):
        ReportIdentity(
            cast("_Config", {}),
            _Grammar(),
            _conventions(),
            freeze_id="freeze/1",
        )

    with pytest.raises(TypeError, match="run specification is not one"):
        ReportIdentity(
            cast("_Config", "0" * 64),
            _Grammar(),
            _conventions(),
            freeze_id="freeze/1",
        )

    with pytest.raises(TypeError, match="grammar is not one"):
        ReportIdentity(
            _Config({}),
            cast("_Grammar", "0" * 64),
            _conventions(),
            freeze_id="freeze/1",
        )

    with pytest.raises(TypeError, match="conventions are not a vector"):
        ReportIdentity(
            _Config({}),
            _Grammar(),
            cast("ConventionVector", {}),
            freeze_id="freeze/1",
        )

    with pytest.raises(ValueError, match="fingerprint is empty"):
        _identity(grammar="")


def test_identity_rejects_noncanonical_numbers_and_non_mapping_dumps() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        _identity(config={"loss": float("nan")})

    class BadConfig:
        def model_dump(self) -> list[JsonValue]:
            return []

    with pytest.raises(TypeError, match="model dump is not a mapping"):
        ReportIdentity(
            cast("_Config", BadConfig()),
            _Grammar(),
            _conventions(),
            freeze_id="freeze/1",
        )


def test_identity_rejects_a_non_json_value_nested_in_a_mapping_dump() -> None:
    invalid = cast("Mapping[str, JsonValue]", {"options": {"fast"}})

    with pytest.raises(TypeError, match="not JSON-shaped"):
        ReportIdentity(
            _Config(invalid), _Grammar(), _conventions(), freeze_id="freeze/1"
        )
