import pytest

from lofi.data import ClassRole, QuantilePopulation
from lofi.data._selection import (
    CategoryClass,
    CategoryRole,
    FormSelector,
    FrequencyBand,
    FrequencyCount,
    FrequencyQuantile,
    MajorityRule,
    PerOccurrenceRule,
    RankBand,
    SelectionPolicy,
    UnionRule,
    WeightedRule,
)
from lofi.data._selection_factory import build_selection_policy, build_selector
from lofi.spec.constraint_policy import (
    AmbiguityMode,
    CategoryClassSettings,
    CategoryRoleSettings,
    CountBandSettings,
    FrequencyBandSettings,
    QuantileSettings,
    RankBandSettings,
    SelectionSettings,
    SelectorSettings,
)


@pytest.mark.parametrize(
    ("mode", "rule"),
    [
        (AmbiguityMode.UNION, UnionRule()),
        (AmbiguityMode.MAJORITY, MajorityRule()),
        (AmbiguityMode.WEIGHTED, WeightedRule()),
        (AmbiguityMode.PER_OCCURRENCE, PerOccurrenceRule()),
    ],
)
def test_every_ambiguity_setting_builds_its_runtime_rule(
    mode: AmbiguityMode,
    rule: UnionRule[str]
    | MajorityRule[str]
    | WeightedRule[str]
    | PerOccurrenceRule[str],
) -> None:
    policy = build_selection_policy(
        SelectionSettings(mode, RankBandSettings(0.0, 1.0), 0.5, 7)
    )

    assert policy == SelectionPolicy(rule, coverage=0.5, seed=7)


@pytest.mark.parametrize(
    ("settings", "selector"),
    [
        (RankBandSettings(0.1, 0.5), RankBand(0.1, 0.5)),
        (FrequencyBandSettings(0.01, 0.2), FrequencyBand(0.01, 0.2)),
        (CountBandSettings(2, None), FrequencyCount(2, None)),
        (CountBandSettings(3, 9), FrequencyCount(3, 9)),
        (
            QuantileSettings(0.4, QuantilePopulation.TYPES),
            FrequencyQuantile(0.4, over=QuantilePopulation.TYPES),
        ),
        (
            QuantileSettings(0.6, QuantilePopulation.TOKENS),
            FrequencyQuantile(0.6, over=QuantilePopulation.TOKENS),
        ),
        (
            CategoryClassSettings(frozenset({"N", "NP"})),
            CategoryClass({"N", "NP"}),
        ),
        (
            CategoryRoleSettings(ClassRole.CLOSED, 0.2),
            CategoryRole(ClassRole.CLOSED, 0.2),
        ),
        (
            CategoryRoleSettings(ClassRole.OPEN, 0.5),
            CategoryRole(ClassRole.OPEN, 0.5),
        ),
    ],
)
def test_every_selector_setting_builds_its_runtime_value(
    settings: SelectorSettings, selector: FormSelector[str]
) -> None:
    assert build_selector(settings) == selector
    assert build_selection_policy(
        SelectionSettings(AmbiguityMode.UNION, settings, 0.75, 11)
    ) == SelectionPolicy(UnionRule(), selector=selector, coverage=0.75, seed=11)
