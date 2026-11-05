"""Restricted symbol sampling from logits."""

from typing import TYPE_CHECKING, Final, Protocol, final, override, runtime_checkable

import torch

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.nn._values import LogDistribution
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.spec.decoding import SamplingSettings

__all__ = [
    "FULL",
    "KEEP_OFFERED",
    "NUCLEUS",
    "TOP_K",
    "LogitRestriction",
    "SamplingRestrictionFactory",
    "SamplingStrategy",
    "restrict",
    "sample_from_logits",
    "sampling_strategy",
]


class LogitRestriction(Protocol):
    """Choose which symbols on offer keep mass in one draw."""

    @property
    def identity(self) -> tuple[str, float | None]:
        """Return the restriction's tag and its one declared setting."""
        ...

    def kept(self, scaled: Tensor, offered: Tensor, dim: int, /) -> Tensor:
        """Return which positions keep mass.

        Parameters
        ----------
        scaled : Tensor
            The scaled scores, negative infinity where nothing is on offer.
        offered : Tensor
            Boolean, of the same shape, true where a symbol is on offer.
        dim : int
            The axis the draw runs along.

        Returns
        -------
        Tensor
            Boolean, of the same shape, true at the kept positions.
        """
        ...


@runtime_checkable
class SamplingRestrictionFactory(Protocol):
    """Build one installed logit restriction from its resolved settings."""

    def build(self, settings: SamplingSettings, /) -> LogitRestriction:
        """Return the restriction the settings declare."""
        ...


def _checked_temperature(value: float) -> float:
    """Return a sampling temperature.

    Parameters
    ----------
    value : float
        The temperature the logits are divided by.

    Returns
    -------
    float
        The temperature.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is not above zero, where the division is undefined rather than merely
        sharp.
    """
    temperature = ensure_finite("a sampling temperature", value)

    if temperature <= 0.0:
        msg = f"a sampling temperature is above zero, not {temperature}"
        raise ValueError(msg)

    return temperature


@final
class SamplingStrategy:
    """Configures a logit restriction.

    Parameters
    ----------
    restriction : LogitRestriction
        Which symbols on offer keep mass.
    temperature : float
        What the logits are divided by before anything else, above zero.

    Raises
    ------
    ValueError
        If the temperature is not above zero.
    """

    __slots__ = ("_restriction", "_temperature")

    def __init__(
        self, restriction: LogitRestriction, *, temperature: float = 1.0
    ) -> None:
        self._restriction = restriction
        self._temperature = _checked_temperature(temperature)

    @property
    def restriction(self) -> LogitRestriction:
        """Return the logit restriction."""
        return self._restriction

    @property
    def temperature(self) -> float:
        """Return the logit scaling temperature."""
        return self._temperature

    def _identity(self) -> tuple[tuple[str, float | None], float]:
        """Return the fields two strategies are equal on."""
        return (self._restriction.identity, self._temperature)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value describes the same restriction.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not SamplingStrategy:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the restriction and the temperature."""
        return hash((SamplingStrategy, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the restriction and the temperature in keyword form."""
        name, setting = self._restriction.identity
        declared = "" if setting is None else f", setting={setting!r}"

        return (
            f"SamplingStrategy({name!r}{declared}, temperature={self._temperature!r})"
        )


@final
class _Full(FrozenValue):
    """Keep every symbol the mask offers."""

    _noun = "full logit restriction"
    __slots__ = ()

    @property
    def identity(self) -> tuple[str, float | None]:
        return ("full", None)

    def kept(self, scaled: Tensor, offered: Tensor, dim: int, /) -> Tensor:
        del scaled, dim

        return offered


@final
class _TopK(FrozenValue):
    """Keep the ``count`` highest scoring symbols on offer of each row."""

    _noun = "top-k logit restriction"
    __slots__ = ("_count",)

    def __init__(self, count: int, /) -> None:
        object.__setattr__(self, "_count", count)

    @property
    def identity(self) -> tuple[str, float | None]:
        return ("top-k", float(self._count))

    def kept(self, scaled: Tensor, offered: Tensor, dim: int, /) -> Tensor:
        wanted = min(self._count, scaled.shape[dim])
        chosen = scaled.topk(wanted, dim=dim).indices
        kept = torch.zeros_like(scaled, dtype=torch.bool)

        return kept.scatter(dim, chosen, value=True) & offered


@final
class _Nucleus(FrozenValue):
    """Keep the shortest descending prefix whose mass reaches a threshold."""

    _noun = "nucleus logit restriction"
    __slots__ = ("_mass",)

    def __init__(self, mass: float, /) -> None:
        object.__setattr__(self, "_mass", mass)

    @property
    def identity(self) -> tuple[str, float | None]:
        return ("nucleus", self._mass)

    def kept(self, scaled: Tensor, offered: Tensor, dim: int, /) -> Tensor:
        ordered, positions = scaled.sort(dim=dim, descending=True)
        ranked = ordered.softmax(dim=dim)
        ahead = ranked.cumsum(dim=dim) - ranked
        kept = torch.zeros_like(scaled, dtype=torch.bool)

        return kept.scatter(dim, positions, ahead < self._mass) & offered


KEEP_OFFERED: Final[LogitRestriction] = _Full()
"""The restriction under which every symbol on offer keeps its mass."""


@final
class _FullFactory(FrozenValue):
    _noun = "full restriction factory"
    __slots__ = ()

    def build(self, settings: SamplingSettings, /) -> LogitRestriction:
        del settings

        return KEEP_OFFERED


@final
class _TopKFactory(FrozenValue):
    _noun = "top-k restriction factory"
    __slots__ = ()

    def build(self, settings: SamplingSettings, /) -> LogitRestriction:
        if settings.k is None:
            msg = "a top-k restriction resolved without a count"
            raise ValueError(msg)

        return _TopK(settings.k)


@final
class _NucleusFactory(FrozenValue):
    _noun = "nucleus restriction factory"
    __slots__ = ()

    def build(self, settings: SamplingSettings, /) -> LogitRestriction:
        if settings.p is None:
            msg = "a nucleus restriction resolved without a mass"
            raise ValueError(msg)

        return _Nucleus(settings.p)


FULL: Final[SamplingRestrictionFactory] = _FullFactory()
"""The installed restriction that keeps every symbol on offer."""

TOP_K: Final[SamplingRestrictionFactory] = _TopKFactory()
"""The installed restriction that keeps the highest scoring symbols."""

NUCLEUS: Final[SamplingRestrictionFactory] = _NucleusFactory()
"""The installed restriction that keeps a descending prefix of declared mass."""


def sampling_strategy(settings: SamplingSettings, /) -> SamplingStrategy:
    """Return the strategy one resolved sampling declaration describes.

    Parameters
    ----------
    settings : SamplingSettings
        The validated restriction and temperature.

    Returns
    -------
    SamplingStrategy
        The installed restriction the declaration names, under its temperature.

    Raises
    ------
    RegistryError
        If no installed ``lofi.sampling_restriction`` entry is named by the
        declaration, or the entry builds no restriction.
    """
    member = Registry[object]("sampling_restriction").get(settings.kind)
    if not isinstance(member, SamplingRestrictionFactory):
        msg = (
            f"the sampling_restriction entry {settings.kind!r} names {member!r}, "
            "not a SamplingRestrictionFactory"
        )
        raise RegistryError(msg, family="sampling_restriction")

    return SamplingStrategy(member.build(settings), temperature=settings.temperature)


def _offered(logits: LogDistribution) -> Tensor:
    """Return which positions of a batch of logits hold a symbol on offer.

    Parameters
    ----------
    logits : LogDistribution
        The scores and the mask that came with them.

    Returns
    -------
    Tensor
        Boolean, of the logits' own shape, true where a symbol is on offer.
    """
    scores = logits.tensor
    mask = logits.mask

    if mask is None:
        return torch.ones_like(scores, dtype=torch.bool)

    spread = mask.reshape(tuple(mask.shape) + (1,) * (scores.ndim - mask.ndim))

    return spread.expand(scores.shape)


def _scaled(logits: LogDistribution, offered: Tensor, temperature: float) -> Tensor:
    """Return the logits divided by a temperature, with what is not on offer removed.

    Parameters
    ----------
    logits : LogDistribution
        The scores and the mask that came with them.
    offered : Tensor
        Boolean, of the logits' own shape, true where a symbol is on offer.
    temperature : float
        What to divide by.

    Returns
    -------
    Tensor
        Of the logits' own shape and floating dtype: the scaled score at every position
        on offer and negative infinity everywhere else.

    Raises
    ------
    ValueError
        If a position on offer holds a value that is not finite.
    """
    scores = logits.tensor

    if not bool((scores.isfinite() | ~offered).all()):
        loose = int((~scores.isfinite() & offered).sum())
        msg = (
            f"{loose} of the {int(offered.sum())} positions on offer hold a "
            "value that is not finite; a symbol left out is left out by the "
            "mask rather than by a number written into the scores"
        )
        raise ValueError(msg)

    return torch.where(offered, scores / temperature, -torch.inf)


def restrict(
    logits: LogDistribution, /, *, strategy: SamplingStrategy, over: str
) -> LogDistribution:
    """Return the distribution a draw is taken from, restricted and renormalized.

    Parameters
    ----------
    logits : LogDistribution
        The scores, over any axes, with the mask saying which symbols are on offer.
    strategy : SamplingStrategy
        The restriction to apply and the settings its kind declares.
    over : str
        The axis the draw runs along.

    Returns
    -------
    LogDistribution
        Normalized log probabilities on the input axes, with negative infinity outside
        the returned mask.

    Raises
    ------
    TypeError
        If the strategy is not a ``SamplingStrategy``.
    ValueError
        If the logits declare no axis of that name, or if a position on offer holds a
        value that is not finite.
    """
    if not isinstance(strategy, SamplingStrategy):
        msg = f"a draw is restricted by a SamplingStrategy, not {strategy!r}"
        raise TypeError(msg)

    dim = logits.axis(over)
    offered = _offered(logits)
    scaled = _scaled(logits, offered, strategy.temperature)

    kept = strategy.restriction.kept(scaled, offered, dim)

    restricted = torch.where(kept, scaled, -torch.inf)
    total = restricted.logsumexp(dim=dim, keepdim=True)

    return LogDistribution(
        torch.where(kept, restricted - total, -torch.inf),
        logits.axes,
        kept,
        normalized_over=over,
    )


def sample_from_logits(
    logits: LogDistribution,
    /,
    *,
    strategy: SamplingStrategy,
    over: str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return one symbol drawn per position, under one declared strategy.

    Parameters
    ----------
    logits : LogDistribution
        The scores, over any axes, with the mask saying which symbols are on offer.
    strategy : SamplingStrategy
        The restriction to draw under and the settings its kind declares.
    over : str
        The axis the draw runs along.
    generator : torch.Generator | None
        The source the draw reads, or None to read the ambient one.

    Returns
    -------
    Tensor
        Integral, of rank one less than the logits, over the logits' remaining axes in
        their own order: the position along ``over`` of the symbol drawn for each row.

    Raises
    ------
    TypeError
        If the strategy is not a ``SamplingStrategy``.
    ValueError
        If the logits declare no axis of that name, if a position on offer holds a value
        that is not finite, or if a row offers no symbol at all, which is a row nothing
        can be drawn for.
    """
    distribution = restrict(logits, strategy=strategy, over=over)
    dim = distribution.axis(over)
    weights = distribution.tensor.exp().movedim(dim, -1)
    rows = weights.reshape(-1, weights.shape[-1])
    empty = int((rows.sum(dim=1) <= 0.0).sum())

    if empty:
        msg = (
            f"{empty} of {rows.shape[0]} rows offer no symbol, so no draw is "
            "defined there"
        )
        raise ValueError(msg)

    drawn = torch.multinomial(rows, 1, generator=generator)

    return drawn.reshape(weights.shape[:-1])
