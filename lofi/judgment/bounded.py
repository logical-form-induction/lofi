"""Program steps and density for bounded truncated-normal responses."""

import math
from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, final, override

import torch
from torch.distributions import Distribution, constraints

from lofi.judgment._attributes import NO_FACTORS
from lofi.judgment._design import (
    BoundedDesign,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    ClosedScale,
    DesignError,
    RowColumns,
)
from lofi.judgment._program import (
    Apply,
    Bind,
    Draw,
    Function,
    Literal,
    Name,
    ObserveBounded,
    Prior,
    ensure_identifier,
)
from lofi.judgment._sources import (
    AnonymousParticipantPolicy as _AnonymousParticipantPolicy,
)
from lofi.judgment._sources import JudgmentSkip
from lofi.judgment.relations import related_item_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor
    from torch.distributions.constraints import Constraint

    from lofi.judgment._attributes import ItemFactors
    from lofi.judgment._ports import BoundedRatedItem
    from lofi.judgment._program import Step
    from lofi.judgment._sources import AnonymousParticipantPolicy

__all__ = [
    "BoundedSource",
    "ClosedUnitTruncatedNormal",
    "TruncatedNormalFamily",
    "closed_unit_log_prob",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)
"""Natural logarithm of two pi."""

_LOG_HALF = math.log(0.5)
"""Switch point for a stable logarithmic complement."""


@final
class BoundedSource:
    """Continuous finite-scale judgments read as bounded design rows.

    Missing responses are counted as skips. Unattributed judgments are either
    skipped or assigned distinct singleton participants under the declared
    policy.

    Parameters
    ----------
    ratings : Iterable[BoundedRatedItem]
        Scalar or relational-scalar responses, as a corpus reader emits them.
    factors : ItemFactors
        The declared factors, read from the item record a row names: the rated
        item, or the related-item key of the rated item and the items its
        response relates it to.
    participant_policy : AnonymousParticipantPolicy
        Treatment of responses naming no participant.

    Raises
    ------
    DesignError
        If a response is not on a finite continuous scale, or a declared factor
        has no item record under a row's key.
    TypeError
        If the participant policy has the wrong type.
    """

    __slots__ = (
        "_factors",
        "_participant_policy",
        "_rows",
        "_singleton_participants",
        "_skips",
    )

    def __init__(
        self,
        ratings: Iterable[BoundedRatedItem],
        *,
        factors: ItemFactors = NO_FACTORS,
        participant_policy: AnonymousParticipantPolicy = (
            _AnonymousParticipantPolicy.SKIP
        ),
    ) -> None:
        if not isinstance(participant_policy, _AnonymousParticipantPolicy):
            msg = (
                "a bounded source's participant policy is not an "
                f"AnonymousParticipantPolicy: {participant_policy!r}"
            )

            raise TypeError(msg)

        held = tuple(ratings)
        occupied = {
            participant
            for rated in held
            if (participant := rated.supervision.annotator) is not None
        }
        rows: list[BoundedJudgmentRow] = []
        skips: Counter[str] = Counter()
        singleton_participants = 0

        for rated in held:
            supervision = rated.supervision
            scale = supervision.scale
            related = supervision.related_items

            item = (
                related_item_key((rated.item.item_id, *related))
                if related
                else rated.item.item_id
            )

            if scale.kind != "continuous" or scale.maximum is None:
                msg = (
                    f"the item {item!r} was rated on a noncontinuous or unbounded scale"
                )

                raise DesignError(msg)

            value = supervision.value

            if value is None:
                skips[JudgmentSkip.MISSING_RESPONSE] += 1

                continue

            participant = supervision.annotator

            if participant is None:
                if participant_policy is _AnonymousParticipantPolicy.SKIP:
                    skips[JudgmentSkip.MISSING_PARTICIPANT] += 1

                    continue

                base = f"anonymous-{singleton_participants}"
                participant = base
                collision = 0

                while participant in occupied:
                    collision += 1
                    participant = f"{base}-{collision}"

                occupied.add(participant)
                singleton_participants += 1

            rows.append(
                BoundedJudgmentRow(
                    participant,
                    item,
                    value,
                    ClosedScale(scale.minimum, scale.maximum),
                    RowColumns(factors.levels(item), {}),
                )
            )

        self._factors = factors.names
        self._participant_policy = participant_policy
        self._rows = tuple(rows)
        self._singleton_participants = singleton_participants
        self._skips: Mapping[str, int] = MappingProxyType(dict(skips))

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factors every row has, in column order."""
        return self._factors

    @property
    def participant_policy(self) -> AnonymousParticipantPolicy:
        """Return the treatment of unattributed judgments."""
        return self._participant_policy

    @property
    def singleton_participants(self) -> int:
        """Return how many singleton participants the policy created."""
        return self._singleton_participants

    @property
    def skips(self) -> Mapping[str, int]:
        """Return skipped judgment counts by reason."""
        return self._skips

    def rows(self) -> tuple[BoundedJudgmentRow, ...]:
        """Return one bounded row per admitted judgment."""
        return self._rows

    def design(self) -> BoundedDesign:
        """Return the rows encoded as a validated bounded design."""
        return BoundedDesignCodec(factors=self._factors).encode(self._rows)

    @override
    def __repr__(self) -> str:
        """Return counts, factors and the anonymous-participant policy."""
        return (
            f"BoundedSource(rows={len(self._rows)}, factors={self._factors!r}, "
            f"participant_policy={self._participant_policy.value!r}, "
            f"singleton_participants={self._singleton_participants}, "
            f"skipped={sum(self._skips.values())})"
        )


def _log1mexp(value: Tensor) -> Tensor:
    """Return ``log(1 - exp(value))`` for a nonpositive argument.

    Parameters
    ----------
    value : Tensor
        Nonpositive logarithm.
    """
    return torch.where(
        value < _LOG_HALF,
        torch.log1p(-torch.exp(value)),
        torch.log(-torch.expm1(value)),
    )


def _log_standard_normal_interval(lower: Tensor, upper: Tensor) -> Tensor:
    """Return ``log(Phi(upper) - Phi(lower))`` without tail cancellation."""
    lower_log_cdf = torch.special.log_ndtr(lower)
    upper_log_cdf = torch.special.log_ndtr(upper)
    below_zero = upper <= 0.0
    above_zero = lower >= 0.0
    lower_tail = upper_log_cdf + _log1mexp(lower_log_cdf - upper_log_cdf)
    upper_log_survival = torch.special.log_ndtr(-upper)
    lower_log_survival = torch.special.log_ndtr(-lower)
    upper_tail = lower_log_survival + _log1mexp(upper_log_survival - lower_log_survival)
    crossing = ~(below_zero | above_zero)
    central_mass = 0.5 * (
        torch.erf(upper / math.sqrt(2.0)) - torch.erf(lower / math.sqrt(2.0))
    )
    central = torch.log(
        torch.where(crossing, central_mass, torch.ones_like(central_mass))
    )

    return torch.where(
        below_zero, lower_tail, torch.where(above_zero, upper_tail, central)
    )


def closed_unit_log_prob(response: Tensor, location: Tensor, scale: Tensor) -> Tensor:
    """Return truncated-normal log density on the inclusive unit interval.

    Parameters
    ----------
    response : Tensor
        Values to score.
    location : Tensor
        Normal locations.
    scale : Tensor
        Positive normal standard deviations.
    """
    standardized = (response - location) / scale
    base = -0.5 * standardized.square() - torch.log(scale) - 0.5 * _LOG_TWO_PI
    lower = -location / scale
    upper = (1.0 - location) / scale
    log_normalizer = _log_standard_normal_interval(lower, upper)
    inside = (response >= 0.0) & (response <= 1.0)
    density = base - log_normalizer

    return torch.where(inside, density, density.new_full((), -math.inf))


@final
class ClosedUnitTruncatedNormal(Distribution):
    """A normal distribution restricted to the closed unit interval."""

    arg_constraints: ClassVar[dict[str, Constraint]] = {
        "location": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = False

    def __init__(
        self,
        location: Tensor,
        scale: Tensor,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.location, self.scale = torch.broadcast_tensors(location, scale)
        super().__init__(
            batch_shape=self.location.shape,
            validate_args=validate_args,
        )

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Return the inclusive truncated-normal log density.

        Parameters
        ----------
        value : Tensor
            Values to score.
        """
        if self._validate_args:
            self._validate_sample(value)

        return closed_unit_log_prob(value, self.location, self.scale)


@final
class TruncatedNormalFamily:
    """A bounded response family with location ``sigmoid(eta)``.

    Parameters
    ----------
    scale : Prior | float
        Prior of one positive global standard deviation or a fixed positive
        value on the normalized unit interval.

    Raises
    ------
    TypeError
        If a fixed scale is a boolean.
    ValueError
        If a fixed scale is nonfinite or nonpositive.
    """

    __slots__ = ("_scale",)

    def __init__(self, scale: Prior | float) -> None:
        if isinstance(scale, bool):
            msg = f"a truncated-normal scale is not a number: {scale!r}"

            raise TypeError(msg)

        if isinstance(scale, float) and not (math.isfinite(scale) and scale > 0.0):
            msg = f"a truncated-normal scale is not a finite positive number: {scale!r}"

            raise ValueError(msg)

        self._scale = scale

    @property
    def scale(self) -> Prior | float:
        """Return the scale prior or fixed value."""
        return self._scale

    @property
    def scalar_object(self) -> str:
        """Return the name of the set indexing the global scale."""
        return "Unit"

    @property
    def location_site(self) -> str:
        """Return the site holding each row's unit-interval location."""
        return "bounded_location"

    @property
    def scale_site(self) -> str:
        """Return the site holding the positive global scale."""
        return "bounded_scale"

    @property
    def bound_names(self) -> tuple[str, ...]:
        """Return variables drawn or bound by the family in program order."""
        return (self.location_site, self.scale_site)

    def steps(self, *, predictor: str, response: str, plate: str) -> tuple[Step, ...]:
        """Return program steps observing normalized bounded responses.

        Parameters
        ----------
        predictor : str
            Previously bound latent-predictor variable.
        response : str
            Host column holding normalized floating responses.
        plate : str
            Finite set indexing response rows.

        Raises
        ------
        ValueError
            If a name is invalid or collides with a family-owned name.
        """
        ensure_identifier(predictor, "a predictor")
        ensure_identifier(response, "a response")
        ensure_identifier(plate, "a response plate")
        own = frozenset(self.bound_names)

        for name, role in ((predictor, "a predictor"), (response, "a response")):
            if name in own:
                msg = f"{role} must not be a name this family binds: {name!r}"

                raise ValueError(msg)

        if predictor == response:
            msg = f"a predictor and a response must differ: {predictor!r}"

            raise ValueError(msg)

        if plate == self.scalar_object:
            msg = f"a response plate must not be the scalar object: {plate!r}"

            raise ValueError(msg)

        scale: Step

        if isinstance(self._scale, Prior):
            scale = Draw(self.scale_site, (self.scalar_object,), self._scale)
        else:
            scale = Bind(self.scale_site, Literal(self._scale))

        return (
            Bind(self.location_site, Apply(Function.SIGMOID, Name(predictor))),
            scale,
            ObserveBounded(
                response,
                plate,
                location=self.location_site,
                scale=self.scale_site,
            ),
        )
