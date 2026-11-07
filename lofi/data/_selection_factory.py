"""Construction of selection policies from validated settings."""

from typing import TYPE_CHECKING, assert_never

from lofi.data._selection import (
    CategoryClass,
    CategoryRole,
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
from lofi.data._supervision import Weighted
from lofi.spec.constraint_policy import (
    AmbiguityMode,
    CategoryClassSettings,
    CategoryRoleSettings,
    CountBandSettings,
    FrequencyBandSettings,
    QuantileSettings,
    RankBandSettings,
)

if TYPE_CHECKING:
    from lofi.data._ports import AmbiguityRule
    from lofi.data._selection import FormSelector
    from lofi.spec.constraint_policy import SelectionSettings, SelectorSettings

__all__ = ["build_selection_policy", "build_selector"]

type _SelectionLabel = str | Weighted[str]


def _ambiguity(mode: AmbiguityMode, /) -> AmbiguityRule[str, _SelectionLabel]:
    """Return the ambiguity rule a validated mode names.

    Parameters
    ----------
    mode : AmbiguityMode
        The declared resolution of a form observed under several labels.
    """
    match mode:
        case AmbiguityMode.UNION:
            return UnionRule()
        case AmbiguityMode.MAJORITY:
            return MajorityRule()
        case AmbiguityMode.WEIGHTED:
            return WeightedRule()
        case AmbiguityMode.PER_OCCURRENCE:
            return PerOccurrenceRule()
        case unreachable:
            assert_never(unreachable)


def build_selector(settings: SelectorSettings, /) -> FormSelector[str]:
    """Return the form selector validated selector settings describe.

    Parameters
    ----------
    settings : SelectorSettings
        The settings of one declared selector.
    """
    match settings:
        case RankBandSettings(first=first, last=last):
            return RankBand(first, last)
        case FrequencyBandSettings(at_least=at_least, at_most=at_most):
            return FrequencyBand(at_least, at_most)
        case CountBandSettings(at_least=at_least, at_most=ceiling):
            return FrequencyCount(at_least, ceiling)
        case QuantileSettings(share=share, over=over):
            return FrequencyQuantile(share, over=over)
        case CategoryClassSettings(categories=categories):
            return CategoryClass(categories)
        case CategoryRoleSettings(role=role, share=share):
            return CategoryRole(role, share)
        case unreachable:
            assert_never(unreachable)


def build_selection_policy(
    settings: SelectionSettings, /
) -> SelectionPolicy[str, _SelectionLabel]:
    """Build a selection policy from validated settings.

    Parameters
    ----------
    settings : SelectionSettings
        The settings of one declared constraint policy.
    """
    return SelectionPolicy(
        _ambiguity(settings.ambiguity),
        selector=build_selector(settings.selector),
        coverage=settings.coverage,
        seed=settings.seed,
    )
