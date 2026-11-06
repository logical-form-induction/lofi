"""Declarations of the collectors that answer one active learning batch.

A run's collector either fields each batch through bead or answers it from a
simulated rating design over the whole candidate pool, drawn once from the
run's collector seed. Every participant of a simulated design rates every
candidate, so a simulated batch is answered by each participant once per item.
A drawn known-effects design places its population cutpoints evenly over a
declared latent range: the range is cut into as many equal intervals as the
fit has response levels, and a cutpoint lies at each interior boundary.
"""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.components.base import ComponentSpec

__all__ = [
    "ActiveCollectorSpec",
    "BeadCollectorSettings",
    "BeadCollectorSpec",
    "CollectorSettings",
    "DrawnEffectsCollectorSettings",
    "DrawnEffectsCollectorSpec",
    "EffectScale",
    "LatentRangeSettings",
    "LatentRangeSpec",
    "PredictorTruthCollectorSettings",
    "PredictorTruthCollectorSpec",
    "RegisteredDesignCollectorSettings",
    "RegisteredDesignCollectorSpec",
]

type EffectScale = Literal["standardized", "raw"]
"""How a predictor column becomes the true effect of each candidate."""


class BeadCollectorSettings(NamedTuple):
    """Resolved collection of every batch through bead."""

    kind: Literal["bead"]


class LatentRangeSettings(NamedTuple):
    """Resolved latent interval a drawn design's cutpoints divide evenly."""

    lower: float
    upper: float


class PredictorTruthCollectorSettings(NamedTuple):
    """Resolved simulation taking one predictor column as the true item effect."""

    kind: Literal["predictor-truth"]
    column: str
    participants: int
    scale: EffectScale
    latent_range: LatentRangeSettings


class RegisteredDesignCollectorSettings(NamedTuple):
    """Resolved simulation answering from a registered rating design."""

    kind: Literal["registered-design"]
    design: str
    join: str | None


class DrawnEffectsCollectorSettings(NamedTuple):
    """Resolved simulation over the pool with its own drawn item effects."""

    kind: Literal["drawn-effects"]
    participants: int
    latent_range: LatentRangeSettings


type CollectorSettings = (
    BeadCollectorSettings
    | PredictorTruthCollectorSettings
    | RegisteredDesignCollectorSettings
    | DrawnEffectsCollectorSettings
)
"""The resolved settings of an installed collector."""


class LatentRangeSpec(dx.Model, extra="forbid"):
    r"""Declare the latent interval a drawn design's cutpoints divide evenly.

    For $m$ response levels the cutpoints lie at
    $\kappa_k=a+k(b-a)/m$, for $k\in\{1,\ldots,m-1\}$,
    where $a$ and $b$ are `lower` and `upper`.

    Parameters
    ----------
    lower : float
        Lower end of the interval.
    upper : float
        Upper end of the interval, above ``lower``.
    """

    lower: float
    upper: float

    @dx.validates("lower", "upper")
    @classmethod
    def _end_is_finite(cls, value: float) -> float:
        return ensure_finite("a latent range's end", value)

    @dx.model_validator()
    def _range_is_ordered(self) -> LatentRangeSpec:
        """Validate that the interval has positive width."""
        if self.upper <= self.lower:
            msg = (
                f"a latent range's upper end {self.upper!r} is not above its "
                f"lower end {self.lower!r}"
            )
            raise ValueError(msg)
        return self

    def settings(self) -> LatentRangeSettings:
        """Discard the boundary declaration into immutable settings."""
        return LatentRangeSettings(self.lower, self.upper)


class ActiveCollectorSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declare how the responses to each acquired batch are collected."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the collectors."""
        return "active_collector"

    def reads_bead_responses(self) -> bool:
        """Return whether the collector reads bead responses.

        A collector that does reads them through the response encoding the
        active learning document declares.
        """
        return False


class BeadCollectorSpec(ActiveCollectorSpec):
    """Collect each batch through bead.

    The collector writes a round's batch as bead items and stops the run until
    the responses to that batch are placed beside it, then reads them through
    the bead judgment source under the response encoding the active learning
    document declares.
    """

    kind: Literal["bead"] = "bead"

    @override
    def reads_bead_responses(self) -> bool:
        """Return that the collector reads bead responses."""
        return True

    def settings(self) -> BeadCollectorSettings:
        """Discard the boundary declaration into immutable settings."""
        return BeadCollectorSettings(self.kind)


class PredictorTruthCollectorSpec(ActiveCollectorSpec):
    """Answer each batch from a predictor column taken as the true item effect.

    The candidates' values in the column are their item scores in the known
    effects design over the pool: each declared participant draws an
    intercept and cutpoint offsets and rates every candidate on the fit's
    response levels, whose population cutpoints divide ``latent_range``
    evenly.

    Parameters
    ----------
    column : str
        The predictor column, one the trained point fills, holding the true
        effect of every candidate.
    participants : int
        Number of simulated participants.
    scale : EffectScale
        ``standardized`` centers the column over the pool and divides it by
        its standard deviation; ``raw`` takes its values as they are.
    latent_range : LatentRangeSpec
        Latent interval the population cutpoints divide evenly.
    """

    kind: Literal["predictor-truth"] = "predictor-truth"
    column: str
    participants: int
    scale: EffectScale
    latent_range: LatentRangeSpec

    @dx.validates("column")
    @classmethod
    def _column_is_named(cls, value: str) -> str:
        return ensure_name("a predictor column", value)

    @dx.validates("participants")
    @classmethod
    def _participants_are_positive(cls, value: int) -> int:
        return ensure_positive("a count of simulated participants", value)

    def settings(self) -> PredictorTruthCollectorSettings:
        """Discard the boundary declaration into immutable settings."""
        return PredictorTruthCollectorSettings(
            self.kind,
            self.column,
            self.participants,
            self.scale,
            self.latent_range.settings(),
        )


class RegisteredDesignCollectorSpec(ActiveCollectorSpec):
    """Answer each batch from a registered rating design.

    Each candidate receives every response the design records for the design
    item it is joined to.

    Parameters
    ----------
    design : str
        Tag of a design in the ``lofi.judgment_design`` entry point group.
    join : str | None
        A bead expression over each candidate's attributes naming the design
        item it is joined to, or None to join it to the design item sharing
        its identity.
    """

    kind: Literal["registered-design"] = "registered-design"
    design: str
    join: str | None

    @dx.validates("design")
    @classmethod
    def _design_is_named(cls, value: str) -> str:
        return ensure_name("a registered design", value)

    @dx.validates("join")
    @classmethod
    def _join_is_named(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_name("a registered design's join expression", value)

    def settings(self) -> RegisteredDesignCollectorSettings:
        """Discard the boundary declaration into immutable settings."""
        return RegisteredDesignCollectorSettings(self.kind, self.design, self.join)


class DrawnEffectsCollectorSpec(ActiveCollectorSpec):
    """Answer each batch from a known-effects design with drawn item scores.

    The design has one item per candidate, the fit's response levels and the
    declared participants; its item scores and participant effects are drawn
    from the run's collector seed, and its population cutpoints divide
    ``latent_range`` evenly.

    Parameters
    ----------
    participants : int
        Number of simulated participants.
    latent_range : LatentRangeSpec
        Latent interval the population cutpoints divide evenly.
    """

    kind: Literal["drawn-effects"] = "drawn-effects"
    participants: int
    latent_range: LatentRangeSpec

    @dx.validates("participants")
    @classmethod
    def _participants_are_positive(cls, value: int) -> int:
        return ensure_positive("a count of simulated participants", value)

    def settings(self) -> DrawnEffectsCollectorSettings:
        """Discard the boundary declaration into immutable settings."""
        return DrawnEffectsCollectorSettings(
            self.kind, self.participants, self.latent_range.settings()
        )
