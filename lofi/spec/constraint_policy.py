"""Constraint-selection policy declarations.

A policy names where in a corpus the forms it constrains sit through one
selector. Every selector but the count band is read off the corpus it is
applied to, as a share of its forms, of its token mass or of its category
inventory, so one declaration selects the same part of the distribution
whatever size of corpus a grammar yields. The count band admits absolute
frequencies and is the one selector whose selection moves with corpus size.
"""

import enum
from typing import Literal, NamedTuple, Self, assert_never, override

import didactic.api as dx

from lofi._checks import ensure_fraction, ensure_non_negative, ensure_positive
from lofi._types import ClassRole, QuantilePopulation

__all__ = [
    "AmbiguityMode",
    "CategoryClassSelectorSpec",
    "CategoryClassSettings",
    "CategoryRoleSelectorSpec",
    "CategoryRoleSettings",
    "ConstraintPolicySection",
    "ConstraintPolicySpec",
    "CountBandSettings",
    "CountSelectorSpec",
    "FrequencyBandSettings",
    "FrequencySelectorSpec",
    "PolicyDomain",
    "QuantileSelectorSpec",
    "QuantileSettings",
    "RankBandSettings",
    "RankSelectorSpec",
    "SelectionSettings",
    "SelectorSettings",
    "SelectorSpec",
]


class AmbiguityMode(enum.StrEnum):
    """Enumerates resolutions of a form observed under several labels."""

    UNION = "union"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    PER_OCCURRENCE = "per-occurrence"


class PolicyDomain(enum.StrEnum):
    """Enumerates the label domains a constraint policy selects over.

    Syntactic policies select constraints labeled with categories, semantic
    policies constraints labeled with logical forms.
    """

    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"


class RankBandSettings(NamedTuple):
    """The settings of a rank band.

    Attributes
    ----------
    first : float
        The share of the ranked forms above the band.
    last : float
        The share of the ranked forms at or above the band's lowest rank.
    """

    first: float
    last: float


class FrequencyBandSettings(NamedTuple):
    """The settings of a relative-frequency band.

    Attributes
    ----------
    at_least : float
        The lowest admitted share of the token mass.
    at_most : float
        The highest admitted share of the token mass.
    """

    at_least: float
    at_most: float


class CountBandSettings(NamedTuple):
    """The settings of an absolute-frequency band.

    Attributes
    ----------
    at_least : int
        The lowest admitted count.
    at_most : int | None
        The highest admitted count, or None for no ceiling.
    """

    at_least: int
    at_most: int | None


class QuantileSettings(NamedTuple):
    """The settings of a frequency quantile.

    Attributes
    ----------
    share : float
        The share of the population covered.
    over : QuantilePopulation
        The population the share is measured on.
    """

    share: float
    over: QuantilePopulation


class CategoryClassSettings(NamedTuple):
    """The settings of a named category class.

    Attributes
    ----------
    categories : frozenset[str]
        The categories naming the class.
    """

    categories: frozenset[str]


class CategoryRoleSettings(NamedTuple):
    """The settings of a category class addressed by role.

    Attributes
    ----------
    role : ClassRole
        The end of the class-size order the class is taken from.
    share : float
        The share of the observed categories the class spans.
    """

    role: ClassRole
    share: float


type SelectorSettings = (
    RankBandSettings
    | FrequencyBandSettings
    | CountBandSettings
    | QuantileSettings
    | CategoryClassSettings
    | CategoryRoleSettings
)
"""The settings of any declared selector."""


class SelectorSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares where selected forms sit in a corpus."""

    def settings(self) -> SelectorSettings:
        """Return the settings the selector is built from.

        Raises
        ------
        TypeError
            If the declaration is the bare union root rather than a selector.
        """
        msg = f"{type(self).__name__} is the selector union, not a selector"
        raise TypeError(msg)


class RankSelectorSpec(SelectorSpec):
    """Declares a band of frequency ranks as shares of the ranked forms.

    The band runs from the form below the first share of the ranking down to
    the last form within the last share, so ``first: 0.0, last: 0.1`` names the
    most frequent tenth of the forms and the defaults name every form.
    """

    kind: Literal["rank"] = "rank"
    first: float = 0.0
    last: float = 1.0

    @dx.validates("first")
    @classmethod
    def _first_is_fractional(cls, value: float) -> float:
        """Return a first share in the closed unit interval."""
        return ensure_fraction("a rank band's first share", value)

    @dx.validates("last")
    @classmethod
    def _last_is_fractional(cls, value: float) -> float:
        """Return a last share in the closed unit interval."""
        return ensure_fraction("a rank band's last share", value)

    @dx.model_validator()
    def _last_follows_first(self) -> Self:
        """Return a rank band whose last share is not above its first."""
        if self.last < self.first:
            msg = (
                f"a rank band runs from share {self.first} to share {self.last}, "
                "which admits none"
            )
            raise ValueError(msg)

        return self

    @override
    def settings(self) -> RankBandSettings:
        """Return the settings the rank band is built from."""
        return RankBandSettings(self.first, self.last)


class FrequencySelectorSpec(SelectorSpec):
    """Declares a band of relative frequencies, as shares of the token mass."""

    kind: Literal["frequency"] = "frequency"
    at_least: float = 0.0
    at_most: float = 1.0

    @dx.validates("at_least")
    @classmethod
    def _floor_is_fractional(cls, value: float) -> float:
        """Return a relative-frequency floor in the closed unit interval."""
        return ensure_fraction("a frequency band's floor", value)

    @dx.validates("at_most")
    @classmethod
    def _ceiling_is_fractional(cls, value: float) -> float:
        """Return a relative-frequency ceiling in the closed unit interval."""
        return ensure_fraction("a frequency band's ceiling", value)

    @dx.model_validator()
    def _ceiling_follows_floor(self) -> Self:
        """Return a nonempty relative-frequency band."""
        if self.at_most < self.at_least:
            msg = (
                f"a frequency band runs from {self.at_least} to {self.at_most}, "
                "which admits none"
            )
            raise ValueError(msg)

        return self

    @override
    def settings(self) -> FrequencyBandSettings:
        """Return the settings the frequency band is built from."""
        return FrequencyBandSettings(self.at_least, self.at_most)


class CountSelectorSpec(SelectorSpec):
    """Declares an inclusive band of absolute corpus frequencies.

    The band admits the same counts whatever the corpus holds, so the share of
    the corpus it selects changes with corpus size.
    """

    kind: Literal["count"] = "count"
    at_least: int = 1
    at_most: int | None = None

    @dx.validates("at_least")
    @classmethod
    def _floor_is_positive(cls, value: int) -> int:
        """Return a positive count floor."""
        return ensure_positive("a count band's floor", value)

    @dx.validates("at_most")
    @classmethod
    def _ceiling_is_positive_or_absent(cls, value: int | None) -> int | None:
        """Return a positive count ceiling, or no ceiling."""
        if value is None:
            return None

        return ensure_positive("a count band's ceiling", value)

    @dx.model_validator()
    def _ceiling_follows_floor(self) -> Self:
        """Return a nonempty count band."""
        if self.at_most is not None and self.at_most < self.at_least:
            msg = (
                f"a count band runs from {self.at_least} to {self.at_most}, "
                "which admits none"
            )
            raise ValueError(msg)

        return self

    @override
    def settings(self) -> CountBandSettings:
        """Return the settings the count band is built from."""
        return CountBandSettings(self.at_least, self.at_most)


class QuantileSelectorSpec(SelectorSpec):
    """Declares the most frequent forms covering a population share."""

    kind: Literal["quantile"] = "quantile"
    share: float
    over: QuantilePopulation = QuantilePopulation.TYPES

    @dx.validates("share")
    @classmethod
    def _share_is_fractional(cls, value: float) -> float:
        """Return a share in the closed unit interval."""
        return ensure_fraction("a quantile selector's share", value)

    @override
    def settings(self) -> QuantileSettings:
        """Return the settings the quantile is built from."""
        return QuantileSettings(self.share, self.over)


class CategoryClassSelectorSpec(SelectorSpec):
    """Declares forms observed under at least one named category."""

    kind: Literal["category-class"] = "category-class"
    categories: frozenset[str]

    @dx.validates("categories")
    @classmethod
    def _categories_are_present(cls, values: frozenset[str]) -> frozenset[str]:
        """Return a nonempty set of nonempty category names."""
        if not values:
            msg = "a category-class selector names at least one category"
            raise ValueError(msg)

        if "" in values:
            msg = "a category-class selector names no empty category"
            raise ValueError(msg)

        return values

    @override
    def settings(self) -> CategoryClassSettings:
        """Return the settings the category class is built from."""
        return CategoryClassSettings(self.categories)


class CategoryRoleSelectorSpec(SelectorSpec):
    """Declares forms observed under the categories at one end of the class sizes.

    The observed categories are ordered by how many forms bear them, smallest
    first for the closed role and largest first for the open role, and the
    class is the declared share of that order, taken to the nearest category
    and including every category tied with the last one taken.
    """

    kind: Literal["category-role"] = "category-role"
    role: ClassRole
    share: float

    @dx.validates("share")
    @classmethod
    def _share_is_fractional(cls, value: float) -> float:
        """Return a share in the closed unit interval."""
        return ensure_fraction("a category-role selector's share", value)

    @override
    def settings(self) -> CategoryRoleSettings:
        """Return the settings the role class is built from."""
        return CategoryRoleSettings(self.role, self.share)


class SelectionSettings(NamedTuple):
    """The settings a selection policy is built from.

    Attributes
    ----------
    ambiguity : AmbiguityMode
        How a form observed under several labels resolves.
    selector : SelectorSettings
        Where in the corpus a selected form sits.
    coverage : float
        The share of the selected forms or occurrences emitted.
    seed : int
        The seed the covered share is drawn under.
    """

    ambiguity: AmbiguityMode
    selector: SelectorSettings
    coverage: float
    seed: int


class ConstraintPolicySpec(dx.Model, extra="forbid"):
    """Declares how a constraint source selects and resolves evidence."""

    ambiguity: AmbiguityMode = AmbiguityMode.UNION
    selector: SelectorSpec = dx.field(default_factory=RankSelectorSpec)
    coverage: float = 0.0
    seed: int = 0

    @dx.validates("coverage")
    @classmethod
    def _coverage_is_fractional(cls, value: float) -> float:
        """Return a coverage in the closed unit interval."""
        return ensure_fraction("a constraint policy's coverage", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int) -> int:
        """Return a nonnegative selection seed."""
        return ensure_non_negative("a constraint policy's seed", value)

    def settings(self) -> SelectionSettings:
        """Return the settings the selection policy is built from."""
        return SelectionSettings(
            self.ambiguity, self.selector.settings(), self.coverage, self.seed
        )


class ConstraintPolicySection(dx.Model, extra="forbid"):
    """Declares independent syntactic and semantic constraint policies."""

    syntactic: ConstraintPolicySpec = dx.field(default_factory=ConstraintPolicySpec)
    semantic: ConstraintPolicySpec = dx.field(default_factory=ConstraintPolicySpec)

    def serving(self, domain: PolicyDomain, /) -> ConstraintPolicySpec:
        """Return the policy declared for one label domain.

        Parameters
        ----------
        domain : PolicyDomain
            The domain a constraint source declares it serves.
        """
        match domain:
            case PolicyDomain.SYNTACTIC:
                return self.syntactic
            case PolicyDomain.SEMANTIC:
                return self.semantic
            case unreachable:
                assert_never(unreachable)
