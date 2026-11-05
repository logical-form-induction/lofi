"""Random intercepts of one grouping factor over the cumulative-link head."""

from typing import TYPE_CHECKING, final, override

import torch
from torch import nn

from lofi._checks import ensure_name
from lofi.judgment.ordinal import RowEffects

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

__all__ = ["RandomEffectStore", "VarianceComponent"]


@final
class VarianceComponent:
    """Stores the variance of one random effect.

    Parameters
    ----------
    group : str
        Grouping factor the effect varies over.
    parameter : str
        Head parameter the effect offsets: the latent score, or one
        unconstrained cutpoint span.
    variance : float
        Variance of the offset to that parameter under the model.
    realized : float | None
        Sample variance of the fitted offsets across the groups, or None
        where fewer than two groups exist to compute one.
    groups : int
        Number of groups the effect is fitted for.
    """

    __slots__ = ("_group", "_groups", "_parameter", "_realized", "_variance")

    def __init__(
        self,
        group: str,
        parameter: str,
        variance: float,
        realized: float | None,
        *,
        groups: int,
    ) -> None:
        self._group = group
        self._parameter = parameter
        self._variance = variance
        self._realized = realized
        self._groups = groups

    @property
    def group(self) -> str:
        """Return the grouping factor."""
        return self._group

    @property
    def parameter(self) -> str:
        """Return the head parameter the effect offsets."""
        return self._parameter

    @property
    def variance(self) -> float:
        """Return the variance of the offset to the parameter under the model."""
        return self._variance

    @property
    def realized(self) -> float | None:
        """Return the sample variance of the fitted offsets, if defined."""
        return self._realized

    @property
    def groups(self) -> int:
        """Return the number of groups the effect is fitted for."""
        return self._groups

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this component."""
        return (
            f"VarianceComponent({self._group!r}, {self._parameter!r}, "
            f"{self._variance!r}, {self._realized!r}, {self._groups!r})"
        )


@final
class RandomEffectStore(nn.Module):
    """Random intercepts of one grouping factor, gathered by group index.

    Intercepts use a noncentered parameterization: standardized group values
    times one scale per head parameter. An extra zero row represents the
    population mean for unseen groups.

    Parameters
    ----------
    group : str
        Name of the grouping factor.
    levels : Iterable[str]
        Identifier of each group, in index order.
    cutpoints : int
        Number of cutpoint spans that receive an intercept, or zero where the
        store moves the latent score alone.
    dtype : torch.dtype
        Floating-point kind of the intercepts.

    Raises
    ------
    TypeError
        If the cutpoint count is not an integer, or the dtype is not a
        floating-point kind.
    ValueError
        If the cutpoint count is negative, no level is given, or a level is
        repeated.
    """

    _standardized: nn.Parameter
    _log_scale: nn.Parameter

    def __init__(
        self,
        group: str,
        levels: Iterable[str],
        *,
        cutpoints: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        if isinstance(cutpoints, bool) or not isinstance(cutpoints, int):
            msg = f"a store's cutpoint count is not an integer: {cutpoints!r}"

            raise TypeError(msg)

        if cutpoints < 0:
            msg = f"a store's cutpoint count is negative: {cutpoints!r}"

            raise ValueError(msg)

        if not dtype.is_floating_point:
            msg = f"a store's intercepts must be floating point, not {dtype}"

            raise TypeError(msg)

        table = tuple(ensure_name(f"a level of {group!r}", level) for level in levels)

        if not table:
            msg = f"a store over {group!r} was given no levels"

            raise ValueError(msg)

        lookup = {level: index for index, level in enumerate(table)}

        if len(lookup) != len(table):
            msg = f"a store over {group!r} repeats a level: {table!r}"

            raise ValueError(msg)

        width = 1 + cutpoints
        self._group = ensure_name("a grouping factor", group)
        self._levels = table
        self._lookup = lookup
        self._cutpoints = cutpoints
        self._standardized = nn.Parameter(torch.zeros(len(table), width, dtype=dtype))
        self._log_scale = nn.Parameter(torch.zeros(width, dtype=dtype))

    @property
    def group(self) -> str:
        """Return the name of the grouping factor."""
        return self._group

    @property
    def levels(self) -> tuple[str, ...]:
        """Return the identifier of each group, in index order."""
        return self._levels

    @property
    def cutpoints(self) -> int:
        """Return the number of cutpoint spans that receive an intercept."""
        return self._cutpoints

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the name of each head parameter the store offsets.

        The latent score comes first, then one span per cutpoint, numbered
        from the lowest.
        """
        return (
            "latent",
            *(f"span_{position}" for position in range(1, self._cutpoints + 1)),
        )

    @property
    def fallback(self) -> int:
        """Return the index scored at the population mean."""
        return len(self._levels)

    @property
    def scale(self) -> Tensor:
        """Return the standard deviation of the intercepts over each parameter."""
        return torch.exp(self._log_scale)

    def align(self, levels: Iterable[str], /) -> Tensor:
        """Return the store index of each given level.

        A level the store has never seen maps to the fallback index. Indexing
        the result by a design's dense group indices gives the store indices
        of the design's rows, which is the argument of ``effects``. The
        alignment is computed once per design and reused across its batches.

        Parameters
        ----------
        levels : Iterable[str]
            Group identifiers in a design's index order.
        """
        fallback = self.fallback

        return torch.tensor(
            [self._lookup.get(level, fallback) for level in levels],
            dtype=torch.int64,
            device=self._standardized.device,
        )

    def intercepts(self) -> Tensor:
        """Return the intercepts of every group and of the population mean.

        The result has one row per group followed by the fallback row, and one
        column per parameter, in the order of ``parameter_names``.
        """
        standardized = self._standardized
        population = standardized.new_zeros((1, standardized.shape[1]))

        return torch.cat((standardized, population), dim=0) * self.scale

    def effects(self, index: Tensor, /) -> RowEffects:
        """Return the intercepts of each row, gathered by store index.

        Parameters
        ----------
        index : Tensor
            Store index of each row, at most the fallback index.
        """
        rows = self.intercepts().index_select(0, index)

        if self._cutpoints == 0:
            return RowEffects(rows[:, 0])

        return RowEffects(rows[:, 0], rows[:, 1:])

    def variance_components(self) -> tuple[VarianceComponent, ...]:
        """Return the variance of the intercepts over each parameter.

        Model variance is the squared scale; realized variance is the sample
        variance across fitted groups and is absent for one group. Cutpoint
        span values remain on the unconstrained scale.
        """
        scale = self.scale.detach()
        fitted = self._standardized.detach() * scale
        groups = len(self._levels)
        realized: list[float | None] = (
            torch.var(fitted, dim=0, correction=1).tolist()
            if groups > 1
            else [None] * fitted.shape[1]
        )

        return tuple(
            VarianceComponent(
                self._group,
                parameter,
                float(variance),
                sample,
                groups=groups,
            )
            for parameter, variance, sample in zip(
                self.parameter_names, scale**2, realized, strict=True
            )
        )
