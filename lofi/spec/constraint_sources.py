"""Synthetic gold-constraint source declarations.

Each source declares the label domain it serves, and the run selects under the
constraint policy it declares for that domain: a syntactic source selects
categories, a semantic source logical forms.
"""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, cast

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_fraction, ensure_name
from lofi._frozen import FrozenValue
from lofi.lexicon import (
    ClampMode,
    ClampPolicy,
    ClampScope,
    DropClampPolicy,
    ErrorPolicy,
    SkipItemPolicy,
)
from lofi.registry import resolve_plugin
from lofi.spec._matching import IdentityMatchingSpec, MatchingSpec
from lofi.spec.components.data import ConstraintSourceSpec
from lofi.spec.constraint_policy import PolicyDomain

if TYPE_CHECKING:
    from collections import abc
    from typing import Protocol

    from lofi.lexicon import UnsatisfiablePolicy
    from lofi.terms import Matching, Term

    class _MatchingDeclaration(Protocol):
        """A validated matching variant with its runtime builder."""

        def build(self) -> Matching[Term]:
            """Build the declared relation."""

    class _PinDeclaration(Protocol):
        """A validated pin variant with its runtime lowering."""

        def settings(self) -> ConstraintPinSettings:
            """Lower the declaration into one runtime pattern."""


__all__ = [
    "ClampSpec",
    "ConstraintPinSettings",
    "ConstraintPinSpec",
    "DeclaredConstraintSourceSettings",
    "DeclaredConstraintSourceSpec",
    "DropClampPolicySpec",
    "ErrorPolicySpec",
    "GoldConstraintSourceSettings",
    "GoldDerivationConstraintSourceSpec",
    "GoldLexiconConstraintSourceSpec",
    "GoldLogicalFormConstraintSourceSpec",
    "ItemFeaturePinSpec",
    "ScoreQuantileSettings",
    "ScoreQuantileSpec",
    "SentenceTopPinSpec",
    "SkipItemPolicySpec",
    "SpanPatternPinSpec",
    "UnsatisfiablePolicyPlugin",
    "UnsatisfiablePolicySpec",
    "unsatisfiable_policy_plugin",
]


class UnsatisfiablePolicySpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares the response to an unsatisfiable clamp."""

    def build(self) -> UnsatisfiablePolicy:
        """Build the selected installed policy."""
        return unsatisfiable_policy_plugin(self).construct(self)


class UnsatisfiablePolicyPlugin[Declaration](FrozenValue):
    """Pair an open policy declaration with its runtime builder."""

    __slots__ = ("_build", "_spec")

    _noun = "unsatisfiable-policy plugin"

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: abc.Callable[[Declaration], UnsatisfiablePolicy],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> UnsatisfiablePolicy:
        """Build one policy from a validated declaration."""
        return self._build(declaration)


def unsatisfiable_policy_plugin(
    declaration: UnsatisfiablePolicySpec, /
) -> UnsatisfiablePolicyPlugin[UnsatisfiablePolicySpec]:
    """Return the installed plugin selected by a policy declaration."""
    plugin = resolve_plugin(
        "unsatisfiable_policy",
        cast("str", declaration.kind),
        UnsatisfiablePolicyPlugin,
        type(declaration),
    )

    return cast("UnsatisfiablePolicyPlugin[UnsatisfiablePolicySpec]", plugin)


class ErrorPolicySpec(UnsatisfiablePolicySpec, extra="forbid"):
    """Abort on an unsatisfiable clamp."""

    kind: Literal["error"] = "error"


class DropClampPolicySpec(UnsatisfiablePolicySpec, extra="forbid"):
    """Drop each unsatisfiable clamp from its item."""

    kind: Literal["drop_clamp"] = "drop_clamp"


class SkipItemPolicySpec(UnsatisfiablePolicySpec, extra="forbid"):
    """Skip an item holding an unsatisfiable clamp."""

    kind: Literal["skip_item"] = "skip_item"


def _build_error(declaration: ErrorPolicySpec, /) -> ErrorPolicy:
    """Build the aborting policy."""
    del declaration
    return ErrorPolicy()


def _build_drop_clamp(declaration: DropClampPolicySpec, /) -> DropClampPolicy:
    """Build the clamp-dropping policy."""
    del declaration
    return DropClampPolicy()


def _build_skip_item(declaration: SkipItemPolicySpec, /) -> SkipItemPolicy:
    """Build the item-skipping policy."""
    del declaration
    return SkipItemPolicy()


ERROR_POLICY_PLUGIN = UnsatisfiablePolicyPlugin(
    spec=ErrorPolicySpec,
    build=_build_error,
)
DROP_CLAMP_POLICY_PLUGIN = UnsatisfiablePolicyPlugin(
    spec=DropClampPolicySpec,
    build=_build_drop_clamp,
)
SKIP_ITEM_POLICY_PLUGIN = UnsatisfiablePolicyPlugin(
    spec=SkipItemPolicySpec,
    build=_build_skip_item,
)


class ClampSpec(dx.Model, extra="forbid"):
    """Declares how a constraint source applies each emitted clamp."""

    mode: ClampMode = ClampMode.HARD
    penalty: float | None = None
    scope: ClampScope = ClampScope.CONSTITUENT
    matching: MatchingSpec | None = None
    meaning_matching: MatchingSpec = dx.field(default_factory=IdentityMatchingSpec)
    require_constituent: bool = True
    negate: bool = False
    on_unsatisfiable: UnsatisfiablePolicySpec = dx.field(
        default_factory=ErrorPolicySpec
    )

    @dx.model_validator()
    def _coherent(self) -> Self:
        """Return a mode whose penalty and constituency settings are coherent."""
        if self.mode is ClampMode.HARD:
            if self.penalty is not None:
                msg = "a hard clamp has no penalty"
                raise ValueError(msg)
        elif self.penalty is None:
            msg = "a soft clamp declares a positive finite penalty"
            raise ValueError(msg)
        elif ensure_finite("a soft clamp penalty", self.penalty) <= 0.0:
            msg = f"a soft clamp penalty is not positive: {self.penalty!r}"
            raise ValueError(msg)

        if not self.require_constituent and self.scope is ClampScope.OCCURRENCE:
            msg = "a token-scoped clamp cannot exclude a constituent"
            raise ValueError(msg)

        if not self.require_constituent and self.mode is ClampMode.SOFT:
            msg = "a negative-bracket clamp is hard"
            raise ValueError(msg)

        if not self.require_constituent and self.negate:
            msg = (
                "a negative-bracket clamp already excludes its span and cannot "
                "also be negated"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> ClampPolicy[Term]:
        """Lower the boundary declaration into runtime clamp settings."""
        matching = self.matching

        return ClampPolicy(
            mode=self.mode,
            penalty=self.penalty,
            scope=self.scope,
            matching=(
                None
                if matching is None
                else cast("_MatchingDeclaration", matching).build()
            ),
            meaning_matching=cast(
                "_MatchingDeclaration", self.meaning_matching
            ).build(),
            require_constituent=self.require_constituent,
            negate=self.negate,
            on_unsatisfiable=self.on_unsatisfiable.build(),
        )

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation consumed by this clamp's objective."""
        return "target" if self.require_constituent else "negative-bracket"


class GoldConstraintSourceSettings(NamedTuple):
    """The settings a gold constraint source is built from.

    Attributes
    ----------
    policy : PolicyDomain
        The label domain the source serves, which names the run policy it
        selects under.
    """

    policy: PolicyDomain
    clamp: ClampPolicy[Term]


_DEFAULT_CLAMP: Final = dx.field(default_factory=ClampSpec)
"""The single default clamp declaration shared by both gold sources."""


class GoldDerivationConstraintSourceSpec(ConstraintSourceSpec):
    """Select constituent constraints from gold derivations.

    A gold derivation labels its constituents with categories, so the source
    serves the syntactic domain and refuses a declaration of any other.
    """

    kind: Literal["gold-derivation"] = "gold-derivation"
    clamp: ClampSpec = _DEFAULT_CLAMP

    @dx.model_validator()
    def _serves_categories(self) -> Self:
        """Return a declaration whose domain gold derivations label.

        Raises
        ------
        ValueError
            If the source declares a domain other than the syntactic one.
        """
        if self.policy is not PolicyDomain.SYNTACTIC:
            msg = (
                "gold derivations label constituents with syntactic categories, "
                f"so a gold-derivation source cannot select under the "
                f"{self.policy} policy"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> GoldConstraintSourceSettings:
        """Return the settings the plugin builds this source from."""
        return GoldConstraintSourceSettings(
            policy=self.policy,
            clamp=self.clamp.settings(),
        )

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation consumed by this source's objective."""
        return self.clamp.constraint_route()


class GoldLexiconConstraintSourceSpec(ConstraintSourceSpec):
    """Select syntactic token constraints from the gold lexicon."""

    kind: Literal["gold-lexicon"] = "gold-lexicon"
    clamp: ClampSpec = _DEFAULT_CLAMP

    @dx.model_validator()
    def _serves_categories(self) -> Self:
        """Return a declaration selecting syntactic categories alone."""
        if self.policy is not PolicyDomain.SYNTACTIC:
            msg = (
                "a gold lexicon labels forms with syntactic categories, so a "
                f"gold-lexicon source cannot select under the {self.policy} policy"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> GoldConstraintSourceSettings:
        """Return the settings the plugin builds this source from."""
        return GoldConstraintSourceSettings(
            policy=self.policy,
            clamp=self.clamp.settings(),
        )

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation consumed by this source's objective."""
        return self.clamp.constraint_route()


class GoldLogicalFormConstraintSourceSpec(ConstraintSourceSpec):
    """Select semantic token constraints from the interpreted gold lexicon."""

    kind: Literal["gold-logical-form"] = "gold-logical-form"
    clamp: ClampSpec = _DEFAULT_CLAMP

    @dx.model_validator()
    def _serves_meanings(self) -> Self:
        """Return a declaration selecting logical forms alone."""
        if self.policy is not PolicyDomain.SEMANTIC:
            msg = (
                "gold logical forms label forms with meanings, so a "
                f"gold-logical-form source cannot select under the {self.policy} policy"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> GoldConstraintSourceSettings:
        """Return the settings the plugin builds this source from."""
        return GoldConstraintSourceSettings(
            policy=self.policy,
            clamp=self.clamp.settings(),
        )

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation consumed by this source's objective."""
        return self.clamp.constraint_route()


class ConstraintPinSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one way of locating spans and the labels placed there."""


class ConstraintPinSettings(NamedTuple):
    """Frozen runtime-neutral settings for one declared span pin."""

    pattern: tuple[str | None, ...] | None
    permitted: frozenset[str]


class ItemFeaturePinSpec(ConstraintPinSpec, extra="forbid"):
    """Pin every occurrence of one designated lexical item to one feature."""

    kind: Literal["item-feature"] = "item-feature"
    item: str
    feature: str

    @dx.validates("item")
    @classmethod
    def _item_is_named(cls, value: str) -> str:
        """Return a nonempty lexical item."""
        return ensure_name("a pinned lexical item", value)

    @dx.validates("feature")
    @classmethod
    def _feature_is_named(cls, value: str) -> str:
        """Return a nonempty latent-feature name."""
        return ensure_name("a pinned latent feature", value)

    def settings(self) -> ConstraintPinSettings:
        """Return the exact one-word pattern and its designated feature."""
        return ConstraintPinSettings((self.item,), frozenset({self.feature}))


class SpanPatternPinSpec(ConstraintPinSpec, extra="forbid"):
    """Pin every span matching a declared word-and-wildcard pattern."""

    kind: Literal["span-pattern"] = "span-pattern"
    pattern: tuple[str | None, ...]
    permitted: frozenset[str]

    @dx.validates("pattern")
    @classmethod
    def _pattern_is_nonempty(
        cls, value: tuple[str | None, ...]
    ) -> tuple[str | None, ...]:
        """Return a nonempty pattern of named words and wildcards."""
        if not value:
            msg = "a span-pattern pin declares at least one matcher"
            raise ValueError(msg)

        return tuple(
            None if member is None else ensure_name("a span pattern word", member)
            for member in value
        )

    @dx.validates("permitted")
    @classmethod
    def _labels_are_nonempty(cls, value: frozenset[str]) -> frozenset[str]:
        """Return a nonempty set of named labels."""
        if not value:
            msg = "a span-pattern pin permits at least one label"
            raise ValueError(msg)

        return frozenset(ensure_name("a permitted label", label) for label in value)

    def settings(self) -> ConstraintPinSettings:
        """Return the declared wildcard surface pattern."""
        return ConstraintPinSettings(self.pattern, self.permitted)


class SentenceTopPinSpec(ConstraintPinSpec, extra="forbid"):
    """Pin the span covering the complete sentence."""

    kind: Literal["sentence-top"] = "sentence-top"
    permitted: frozenset[str]

    @dx.validates("permitted")
    @classmethod
    def _labels_are_nonempty(cls, value: frozenset[str]) -> frozenset[str]:
        """Return a nonempty set of named labels."""
        if not value:
            msg = "a sentence-top pin permits at least one label"
            raise ValueError(msg)

        return frozenset(ensure_name("a permitted label", label) for label in value)

    def settings(self) -> ConstraintPinSettings:
        """Return the whole-sentence pattern."""
        return ConstraintPinSettings(None, self.permitted)


class ScoreQuantileSettings(NamedTuple):
    """Frozen runtime-neutral settings for one score upper tail."""

    quantile: float
    scores: tuple[tuple[str, float], ...]


class ScoreQuantileSpec(dx.Model, extra="forbid"):
    """Declare the upper score tail that receives constraints."""

    quantile: float
    scores: dict[str, float]

    @dx.validates("quantile")
    @classmethod
    def _quantile_is_fractional(cls, value: float) -> float:
        """Return a quantile in the closed unit interval."""
        return ensure_fraction("an score quantile", value)

    @dx.validates("scores")
    @classmethod
    def _scores_are_finite_and_named(cls, value: dict[str, float]) -> dict[str, float]:
        """Return at least one finite item score in item-name order."""
        if not value:
            msg = "an score quantile declares at least one item score"
            raise ValueError(msg)

        return {
            ensure_name("a scored item id", item_id): ensure_finite(
                f"the score of item {item_id!r}", score
            )
            for item_id, score in sorted(value.items())
        }

    def settings(self) -> ScoreQuantileSettings:
        """Lower the declaration into an upper-tail selector."""
        return ScoreQuantileSettings(self.quantile, tuple(self.scores.items()))


class DeclaredConstraintSourceSettings(NamedTuple):
    """Runtime settings for one declared-pattern constraint source."""

    policy: PolicyDomain
    clamp: ClampPolicy[Term]
    patterns: tuple[ConstraintPinSettings, ...]
    score_quantile: ScoreQuantileSettings | None


class DeclaredConstraintSourceSpec(ConstraintSourceSpec):
    """Declare occurrence constraints by item, span pattern or sentence top."""

    kind: Literal["declared"] = "declared"
    pins: tuple[ConstraintPinSpec, ...]
    score_quantile: ScoreQuantileSpec | None = None
    clamp: ClampSpec = _DEFAULT_CLAMP

    @dx.validates("pins")
    @classmethod
    def _pins_are_nonempty(
        cls, value: tuple[ConstraintPinSpec, ...]
    ) -> tuple[ConstraintPinSpec, ...]:
        """Return at least one declared constraint pin."""
        if not value:
            msg = "a declared constraint source names at least one pin"
            raise ValueError(msg)

        return value

    def settings(self) -> DeclaredConstraintSourceSettings:
        """Return the settings the source plugin constructs."""
        score_quantile = self.score_quantile

        return DeclaredConstraintSourceSettings(
            policy=self.policy,
            clamp=self.clamp.settings(),
            patterns=tuple(
                cast("_PinDeclaration", pin).settings() for pin in self.pins
            ),
            score_quantile=(
                None if score_quantile is None else score_quantile.settings()
            ),
        )

    def constraint_route(self) -> Literal["target", "negative-bracket"]:
        """Return the representation consumed by this source's objective."""
        return self.clamp.constraint_route()
