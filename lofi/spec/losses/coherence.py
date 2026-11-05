"""Coherence-loss declaration."""

import enum
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.spec._ports import Dependency
from lofi.spec.approximation import ApproximationRequest, residual_mass_bound
from lofi.spec.components.loss import LossSpec

if TYPE_CHECKING:
    from lofi.spec.components.model import ModelSection

__all__ = [
    "CoherenceDivergenceSettings",
    "CoherenceDivergenceSpec",
    "CoherenceEstimator",
    "CoherenceLossSettings",
    "CoherenceLossSpec",
    "CrossEntropyDivergenceSettings",
    "CrossEntropyDivergenceSpec",
    "FenchelYoungDivergenceSettings",
    "FenchelYoungDivergenceSpec",
    "KLDivergenceSettings",
    "KLDivergenceSpec",
]


class CoherenceEstimator(enum.StrEnum):
    """Names the completeness required of paired frontier enumeration."""

    EXACT = "exact"
    TRUNCATED = "truncated"


class KLDivergenceSettings(NamedTuple):
    """The settings of forward Kullback-Leibler divergence.

    Attributes
    ----------
    kind : Literal["kl"]
        The divergence's tag.
    """

    kind: Literal["kl"]


class CrossEntropyDivergenceSettings(NamedTuple):
    """The settings of cross-entropy under the parent distribution.

    Attributes
    ----------
    kind : Literal["cross-entropy"]
        The divergence's tag.
    """

    kind: Literal["cross-entropy"]


class FenchelYoungDivergenceSettings(NamedTuple):
    """The settings of a Fenchel-Young loss.

    Attributes
    ----------
    kind : Literal["fenchel-young"]
        The divergence's tag.
    alpha : float
        The entropy-family index, at least one.
    """

    kind: Literal["fenchel-young"]
    alpha: float


type CoherenceDivergenceSettings = (
    KLDivergenceSettings
    | CrossEntropyDivergenceSettings
    | FenchelYoungDivergenceSettings
)
"""The resolved settings of whichever divergence a coherence term declares."""


class _DivergenceDeclaration(Protocol):
    """A divergence declaration, of whichever variant."""

    def settings(self) -> CoherenceDivergenceSettings:
        """Return the declared divergence's settings."""
        ...


class CoherenceDivergenceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares the comparison between parent and composed distributions."""


class KLDivergenceSpec(CoherenceDivergenceSpec):
    """Declares forward Kullback-Leibler divergence."""

    kind: Literal["kl"] = "kl"

    def settings(self) -> KLDivergenceSettings:
        """Return the divergence's settings."""
        return KLDivergenceSettings(kind=self.kind)


class CrossEntropyDivergenceSpec(CoherenceDivergenceSpec):
    """Declares cross-entropy under the parent distribution."""

    kind: Literal["cross-entropy"] = "cross-entropy"

    def settings(self) -> CrossEntropyDivergenceSettings:
        """Return the divergence's settings."""
        return CrossEntropyDivergenceSettings(kind=self.kind)


class FenchelYoungDivergenceSpec(CoherenceDivergenceSpec):
    """Declares a Fenchel-Young loss from the alpha-indexed entropy family.

    Attributes
    ----------
    kind : Literal["fenchel-young"]
        Divergence tag.
    alpha : float
        Finite entropy-family index, at least one.
    """

    kind: Literal["fenchel-young"] = "fenchel-young"
    alpha: float = 1.5

    @dx.validates("alpha")
    @classmethod
    def _alpha_is_in_the_entropy_family(cls, value: float) -> float:
        """Validate a finite entropy-family index of at least one."""
        alpha = ensure_finite("alpha", value)

        if alpha < 1.0:
            msg = f"alpha indexes the entropy family from one upward, not {value!r}"
            raise ValueError(msg)

        return alpha

    def settings(self) -> FenchelYoungDivergenceSettings:
        """Return the divergence's settings."""
        return FenchelYoungDivergenceSettings(kind=self.kind, alpha=self.alpha)


class CoherenceLossSettings(NamedTuple):
    """The settings a coherence objective is built from.

    Attributes
    ----------
    readout : str
        The term-readout role supplied by the model graph.
    space : str
        The represented-space role supplying the combinator family.
    estimator : CoherenceEstimator
        Whether paired frontier enumeration must be complete or may truncate.
    residual_mass : float | None
        Explicit residual-mass override for a truncated estimator.
    divergence : CoherenceDivergenceSettings
        The comparison between the parent and composed distributions.
    warmup : int | None
        Steps over which marginal weighting is introduced. Zero and None disable
        the schedule.
    """

    readout: str
    space: str
    estimator: CoherenceEstimator
    residual_mass: float | None
    divergence: CoherenceDivergenceSettings
    warmup: int | None


class CoherenceLossSpec(LossSpec):
    """Declares coherence under one term readout and represented space.

    Attributes
    ----------
    kind : Literal["coherence"]
        Registry tag.
    readout : str
        Term-readout role supplied by the model graph.
    space : str
        Represented-space role supplying the combinator family.
    estimator : CoherenceEstimator
        Whether paired frontier enumeration must be complete or may truncate.
    residual_mass : float | None
        Explicit residual-mass override for a truncated estimator.
    divergence : CoherenceDivergenceSpec
        Comparison between the parent and composed distributions.
    warmup : int | None
        Steps over which marginal weighting is introduced. Zero and None disable
        the schedule.
    """

    kind: Literal["coherence"] = "coherence"
    readout: str
    space: str
    estimator: CoherenceEstimator = CoherenceEstimator.EXACT
    residual_mass: float | None = None
    divergence: CoherenceDivergenceSpec = dx.field(default_factory=KLDivergenceSpec)
    warmup: int | None = 1000

    @dx.validates("readout", "space")
    @classmethod
    def _roles_are_named(cls, value: str) -> str:
        """Validate a nonempty component role."""
        return ensure_name("a coherence component role", value)

    @dx.validates("warmup")
    @classmethod
    def _warmup_is_nonnegative(cls, value: int | None) -> int | None:
        """Validate a nullable nonnegative warmup duration."""
        if value is None:
            return None

        return ensure_non_negative("a coherence warmup", value)

    @dx.validates("residual_mass")
    @classmethod
    def _residual_mass_is_a_bound(cls, value: float | None) -> float | None:
        """Validate an optional component-specific residual budget."""
        return None if value is None else residual_mass_bound(value)

    @dx.model_validator()
    def _override_is_for_truncation(self) -> CoherenceLossSpec:
        """Refuse a residual override on an exact estimator."""
        if (
            self.estimator is CoherenceEstimator.EXACT
            and self.residual_mass is not None
        ):
            msg = "an exact coherence estimator takes no residual_mass override"
            raise ValueError(msg)

        return self

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the term-readout dependency."""
        return (Dependency(self.readout, "coherence-readout"),)

    def approximation_request(self) -> ApproximationRequest | None:
        """Return the residual budget requested by truncated coherence."""
        if self.estimator is CoherenceEstimator.EXACT:
            return None

        return ApproximationRequest(self.residual_mass)

    @override
    def refuse_model(self, model: ModelSection, path: str, /) -> None:
        """Refuse a represented space the model does not declare.

        Raises
        ------
        ValueError
            If ``space`` is not one of the model's represented spaces.
        """
        if self.space not in model.spaces:
            msg = (
                f"{path}.space: the objective names space {self.space!r}, which is "
                f"not declared among {sorted(model.spaces)}"
            )
            raise ValueError(msg)

    def settings(self) -> CoherenceLossSettings:
        """Return the settings the plugin builds this objective from."""
        return CoherenceLossSettings(
            readout=self.readout,
            space=self.space,
            estimator=self.estimator,
            residual_mass=self.residual_mass,
            divergence=cast("_DivergenceDeclaration", self.divergence).settings(),
            warmup=self.warmup,
        )
