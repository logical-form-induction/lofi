"""Development-fitted label alignment with declared test summaries."""

from math import ceil, factorial
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.eval._alignment import Alignment, Contingency, apply_alignment
from lofi.eval._floor import Floor, random_relabeling_floor
from lofi.eval._policies import ManyToOne, OneToOne
from lofi.eval._stability import AlignmentStability, alignment_stability
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.eval._ports import TieBreak
    from lofi.eval._values import MetricResult
    from lofi.spec.evaluators.alignment import AlignmentKind

__all__ = [
    "DeclaredAlignmentScore",
    "FittedDeclaredAlignment",
    "RelabelingBasePrediction",
    "fit_declared_alignment",
    "predict_relabeling_base",
    "score_declared_alignment",
    "stable_declared_alignments",
]


class AlignmentConfiguration(Protocol):
    """Interior view of a validated alignment declaration."""

    @property
    def kind(self) -> AlignmentKind:
        """Return the alignment family."""

    @property
    def tie_break(self) -> TieBreak | None:
        """Return the assignment tie-break, when the family needs one."""

    @property
    def fit_on(self) -> str:
        """Return the fitting split."""

    @property
    def score_on(self) -> str:
        """Return the scored split."""

    @property
    def floor_replicates(self) -> int:
        """Return the random-floor bootstrap budget."""

    @property
    def floor_level(self) -> float:
        """Return the random-floor interval coverage."""

    @property
    def stability_replicates(self) -> int:
        """Return the stability bootstrap budget."""

    @property
    def stability_level(self) -> float:
        """Return the stability interval coverage."""


@final
class FittedDeclaredAlignment(FrozenValue):
    """A declared policy and the correspondence it froze on development."""

    _noun = "declared alignment"
    __slots__ = ("_alignment", "_settings")

    def __init__(
        self,
        settings: AlignmentConfiguration,
        alignment: Alignment,
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_alignment", alignment)

    @property
    def settings(self) -> AlignmentConfiguration:
        """Return the validated declaration that selected the policy."""
        return self._settings

    @property
    def alignment(self) -> Alignment:
        """Return the correspondence frozen on the fitting split."""
        return self._alignment

    @override
    def __repr__(self) -> str:
        """Return the policy and fitting split."""
        return (
            f"FittedDeclaredAlignment(kind={self._settings.kind!r}, "
            f"split={self._alignment.split!r})"
        )


@final
class DeclaredAlignmentScore(FrozenValue):
    """A held-out alignment score beside its random-relabeling floor."""

    _noun = "declared alignment score"
    __slots__ = ("_floor", "_kind", "_observed", "_rows")

    def __init__(
        self,
        kind: str,
        observed: MetricResult,
        floor: Floor,
    ) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_observed", observed)
        object.__setattr__(self, "_floor", floor)
        object.__setattr__(
            self,
            "_rows",
            MappingProxyType(
                {
                    f"{kind}-accuracy": observed,
                    "random-relabeling-floor": floor.result,
                }
            ),
        )

    @property
    def kind(self) -> str:
        """Return the alignment policy whose accuracy was measured."""
        return self._kind

    @property
    def observed(self) -> MetricResult:
        """Return the held-out aligned accuracy."""
        return self._observed

    @property
    def floor(self) -> Floor:
        """Return the chance floor computed over the held-out gold marginals."""
        return self._floor

    @property
    def rows(self) -> Mapping[str, MetricResult]:
        """Return the observed and floor rows in report order."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the alignment kind and its two values."""
        return (
            f"DeclaredAlignmentScore(kind={self._kind!r}, "
            f"observed={self._observed.value!r}, floor={self._floor.value!r})"
        )


@final
class RelabelingBasePrediction(FrozenValue):
    """Combinatorial anchoring prediction for a declared signature.

    A clamp may distinguish at most ``ambiguity_budget`` category alternatives.
    The natural action of a symmetric group on an inventory of size ``n`` has
    base size ``n - 1``. A type clamp can anchor primitive and constructor
    positions together, so the product action needs the larger of the two bases,
    divided across the alternatives one clamp can distinguish.
    """

    _noun = "relabeling-base prediction"
    __slots__ = (
        "_ambiguity_budget",
        "_base_size",
        "_constructor_count",
        "_group_size",
        "_primitive_count",
    )

    def __init__(
        self,
        primitive_count: int,
        constructor_count: int,
        ambiguity_budget: int,
    ) -> None:
        primitives = ensure_non_negative("a primitive count", primitive_count)
        constructors = ensure_non_negative("a constructor count", constructor_count)
        ambiguity = ensure_positive("an ambiguity budget", ambiguity_budget)
        required = max(primitives - 1, constructors - 1, 0)
        object.__setattr__(self, "_primitive_count", primitives)
        object.__setattr__(self, "_constructor_count", constructors)
        object.__setattr__(self, "_ambiguity_budget", ambiguity)
        object.__setattr__(
            self,
            "_group_size",
            factorial(primitives) * factorial(constructors),
        )
        object.__setattr__(self, "_base_size", ceil(required / ambiguity))

    @property
    def group_size(self) -> int:
        """Return the order of the primitive-constructor relabeling group."""
        return self._group_size

    @property
    def base_size(self) -> int:
        """Return the predicted minimum clamp count."""
        return self._base_size

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the inputs and predictions as report columns."""
        return {
            "primitive_count": self._primitive_count,
            "constructor_count": self._constructor_count,
            "ambiguity_budget": self._ambiguity_budget,
            "relabeling_group_size": self._group_size,
            "predicted_base_size": self._base_size,
        }

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"RelabelingBasePrediction({self._primitive_count!r}, "
            f"{self._constructor_count!r}, {self._ambiguity_budget!r})"
        )


def _identity(contingency: Contingency) -> Alignment:
    """Return literal identity over labels shared by both inventories."""
    gold = set(contingency.gold_labels)
    return Alignment(
        {label: label for label in contingency.induced_labels if label in gold},
        contingency,
    )


def _policy(settings: AlignmentConfiguration, contingency: Contingency) -> Alignment:
    """Fit the one policy named by validated settings."""
    tie_break = settings.tie_break

    if settings.kind == "identity":
        return _identity(contingency)

    if tie_break is None:
        msg = f"the {settings.kind!r} alignment declares no tie-break"
        raise ValueError(msg)

    if settings.kind == "many-to-one":
        return ManyToOne(tie_break=tie_break).fit(contingency)

    return OneToOne(tie_break=tie_break).fit(contingency)


def fit_declared_alignment(
    development: Contingency,
    settings: AlignmentConfiguration,
    /,
) -> FittedDeclaredAlignment:
    """Fit the declared label correspondence on its development split."""
    if development.split != settings.fit_on:
        msg = (
            f"an alignment declares fit_on={settings.fit_on!r}, not "
            f"{development.split!r}"
        )
        raise ValueError(msg)

    return FittedDeclaredAlignment(settings, _policy(settings, development))


def score_declared_alignment(
    fitted: FittedDeclaredAlignment,
    test: Contingency,
    /,
    *,
    seed: int,
) -> DeclaredAlignmentScore:
    """Apply one frozen development alignment and compute its chance floor."""
    settings = fitted.settings

    if test.split != settings.score_on:
        msg = (
            f"an alignment declares score_on={settings.score_on!r}, not {test.split!r}"
        )
        raise ValueError(msg)

    observed = apply_alignment(fitted.alignment, test, unit="token")
    floor = random_relabeling_floor(
        test,
        inventory=max(len(test.induced_labels), 1),
        bootstrap=Bootstrap(
            unit="token",
            replicates=settings.floor_replicates,
            seed=seed,
            level=settings.floor_level,
        ),
    )

    return DeclaredAlignmentScore(settings.kind, observed, floor)


def stable_declared_alignments(
    fitted: Sequence[FittedDeclaredAlignment],
    /,
    *,
    seed: int,
) -> AlignmentStability:
    """Measure seed stability under the one shared declaration."""
    held = tuple(fitted)

    if not held:
        msg = "alignment stability has no fitted alignments"
        raise ValueError(msg)

    settings = held[0].settings

    if any(entry.settings != settings for entry in held[1:]):
        msg = "alignment stability received more than one declaration"
        raise ValueError(msg)

    return alignment_stability(
        tuple(entry.alignment for entry in held),
        bootstrap=Bootstrap(
            unit="seed",
            replicates=settings.stability_replicates,
            seed=seed,
            level=settings.stability_level,
        ),
    )


def predict_relabeling_base(
    *,
    primitive_count: int,
    constructor_count: int,
    ambiguity_budget: int,
) -> RelabelingBasePrediction:
    """Predict the relabeling group order and its clamp base size."""
    return RelabelingBasePrediction(
        primitive_count,
        constructor_count,
        ambiguity_budget,
    )
