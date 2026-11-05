import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.data import ClassRole, QuantilePopulation
from lofi.spec.constraint_policy import (
    AmbiguityMode,
    CategoryClassSelectorSpec,
    CategoryClassSettings,
    CategoryRoleSelectorSpec,
    CategoryRoleSettings,
    ConstraintPolicySection,
    ConstraintPolicySpec,
    CountBandSettings,
    CountSelectorSpec,
    FrequencyBandSettings,
    FrequencySelectorSpec,
    PolicyDomain,
    QuantileSelectorSpec,
    QuantileSettings,
    RankBandSettings,
    RankSelectorSpec,
    SelectionSettings,
    SelectorSettings,
    SelectorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_neutral_policy_selects_every_form_and_emits_none() -> None:
    policy = ConstraintPolicySpec()

    assert policy.ambiguity is AmbiguityMode.UNION
    assert policy.selector == RankSelectorSpec(first=0.0, last=1.0)
    assert policy.coverage == 0.0
    assert policy.seed == 0
    assert policy.settings() == SelectionSettings(
        AmbiguityMode.UNION, RankBandSettings(0.0, 1.0), 0.0, 0
    )


def test_the_section_declares_independent_neutral_policies() -> None:
    section = ConstraintPolicySection()

    assert section.syntactic == ConstraintPolicySpec()
    assert section.semantic == ConstraintPolicySpec()


def test_the_section_hands_out_the_policy_a_domain_names() -> None:
    syntactic = ConstraintPolicySpec(coverage=1.0)
    semantic = ConstraintPolicySpec(
        selector=QuantileSelectorSpec(share=0.25), coverage=0.5
    )
    section = ConstraintPolicySection(syntactic=syntactic, semantic=semantic)

    assert section.serving(PolicyDomain.SYNTACTIC) == syntactic
    assert section.serving(PolicyDomain.SEMANTIC) == semantic


@pytest.mark.parametrize("ambiguity", list(AmbiguityMode))
def test_each_ambiguity_mode_round_trips(ambiguity: AmbiguityMode) -> None:
    policy = ConstraintPolicySpec(ambiguity=ambiguity)

    assert ConstraintPolicySpec.model_validate_json(policy.model_dump_json()) == policy
    assert policy.settings().ambiguity is ambiguity


def test_a_policy_declares_one_selector_and_no_band() -> None:
    with pytest.raises(dx.ValidationError, match="band"):
        ConstraintPolicySpec.model_validate({"band": {"at_least": 2}})


@pytest.mark.parametrize(
    ("document", "expected", "settings"),
    [
        ({"kind": "rank"}, RankSelectorSpec(), RankBandSettings(0.0, 1.0)),
        (
            {"kind": "rank", "first": 0.1, "last": 0.4},
            RankSelectorSpec(first=0.1, last=0.4),
            RankBandSettings(0.1, 0.4),
        ),
        (
            {"kind": "frequency", "at_least": 0.01},
            FrequencySelectorSpec(at_least=0.01),
            FrequencyBandSettings(0.01, 1.0),
        ),
        (
            {"kind": "count", "at_least": 2, "at_most": 9},
            CountSelectorSpec(at_least=2, at_most=9),
            CountBandSettings(2, 9),
        ),
        (
            {"kind": "quantile", "share": 0.25},
            QuantileSelectorSpec(share=0.25, over=QuantilePopulation.TYPES),
            QuantileSettings(0.25, QuantilePopulation.TYPES),
        ),
        (
            {"kind": "quantile", "share": 0.8, "over": "tokens"},
            QuantileSelectorSpec(share=0.8, over=QuantilePopulation.TOKENS),
            QuantileSettings(0.8, QuantilePopulation.TOKENS),
        ),
        (
            {"kind": "category-class", "categories": ["NP", "N"]},
            CategoryClassSelectorSpec(categories=frozenset({"N", "NP"})),
            CategoryClassSettings(frozenset({"N", "NP"})),
        ),
        (
            {"kind": "category-role", "role": "closed", "share": 0.2},
            CategoryRoleSelectorSpec(role=ClassRole.CLOSED, share=0.2),
            CategoryRoleSettings(ClassRole.CLOSED, 0.2),
        ),
    ],
)
def test_each_selector_is_reached_through_the_closed_union(
    document: dict[str, JsonValue], expected: SelectorSpec, settings: SelectorSettings
) -> None:
    declared = SelectorSpec.model_validate(document)

    assert declared == expected
    assert declared.settings() == settings
    assert (
        ConstraintPolicySpec.model_validate({"selector": document}).settings().selector
        == settings
    )


def test_the_selector_union_root_builds_no_selector() -> None:
    with pytest.raises(TypeError, match="not a selector"):
        SelectorSpec.settings(RankSelectorSpec())


def test_an_undeclared_selector_tag_is_refused_by_name() -> None:
    with pytest.raises(dx.ValidationError) as refused:
        SelectorSpec.model_validate({"kind": "frequency-count", "at_most": 10})

    assert "frequency-count" in str(refused.value)


def test_an_undeclared_ambiguity_value_is_refused_by_name() -> None:
    with pytest.raises(dx.ValidationError) as refused:
        ConstraintPolicySpec.model_validate({"ambiguity": "intersection"})

    assert "intersection" in str(refused.value)


@pytest.mark.parametrize(
    ("document", "detail"),
    [
        ({"kind": "rank", "first": -0.1}, "first share"),
        ({"kind": "rank", "last": 1.5}, "last share"),
        ({"kind": "rank", "first": 0.6, "last": 0.2}, "admits none"),
        ({"kind": "rank", "at_least": 1}, "at_least"),
        ({"kind": "frequency", "at_least": 2.0}, "floor"),
        ({"kind": "frequency", "at_least": 0.4, "at_most": 0.1}, "admits none"),
        ({"kind": "count", "at_least": 0}, "floor"),
        ({"kind": "count", "at_least": 3, "at_most": 2}, "runs from 3 to 2"),
        ({"kind": "quantile", "share": -0.1}, "-0.1"),
        ({"kind": "quantile", "share": 1.1}, "1.1"),
        ({"kind": "category-class", "categories": []}, "at least one category"),
        ({"kind": "category-class", "categories": [""]}, "empty category"),
        ({"kind": "category-role", "role": "closed", "share": 1.2}, "1.2"),
        ({"kind": "category-role", "role": "function", "share": 0.2}, "function"),
    ],
)
def test_a_malformed_selector_is_refused(
    document: dict[str, JsonValue],
    detail: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        SelectorSpec.model_validate(document)


@pytest.mark.parametrize("coverage", [-0.1, 1.1, float("inf"), float("nan")])
def test_coverage_is_a_finite_fraction(coverage: float) -> None:
    with pytest.raises(dx.ValidationError):
        ConstraintPolicySpec(coverage=coverage)


def test_a_selection_seed_is_nonnegative() -> None:
    with pytest.raises(dx.ValidationError):
        ConstraintPolicySpec(seed=-1)


def test_unknown_policy_settings_are_refused() -> None:
    with pytest.raises(dx.ValidationError, match="sead"):
        ConstraintPolicySpec.model_validate({"coverage": 0.5, "sead": 7})


def test_a_policy_round_trips_through_its_wire_form() -> None:
    policy = ConstraintPolicySpec(
        ambiguity=AmbiguityMode.MAJORITY,
        selector=CategoryRoleSelectorSpec(role=ClassRole.OPEN, share=0.5),
        coverage=0.75,
        seed=3,
    )

    assert (
        ConstraintPolicySpec.model_validate(json.loads(policy.model_dump_json()))
        == policy
    )
