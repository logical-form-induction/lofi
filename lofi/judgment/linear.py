"""A scalar readout entering the latent scale through a fitted slope."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi.judgment._program import Binary, LinkingSteps, Literal, Name, Operator, Prior
from lofi.judgment._values import LinkingDomain, SpecialCase
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._ports import ReadoutColumns
    from lofi.judgment._program import Expression
    from lofi.judgment._spec import LinkTerm

__all__ = ["LINEAR", "LinearLinking", "linear_steps"]

#: The linking family name for a scalar multiplied by a slope.
LINEAR: Final = "linear"

_SLOPE: Final = "slope"
"""The one parameter of a linear linking."""

_VALUE_INTERVAL: Final = Interval(-1000.0, 1000.0)
"""The declared finite range of a scalar readout."""

_SLOPE_INTERVAL: Final = Interval(0.0, 10.0)
"""The declared range used by monotone-link conformance laws."""

_SLOPE_PRIOR: Final = Prior("HalfNormal", 1.0)
"""The prior on a slope under an otherwise free scale."""

_SIGNED_SLOPE_PRIOR: Final = Prior("Normal", 0.0, 1.0)
"""The prior on a slope under fixed cutpoint variance."""


@final
class LinearLinking:
    """A scalar readout multiplied by a fitted or pinned slope.

    Parameters
    ----------
    term : LinkTerm
        Term declaring the scalar readout and optional pinned slope.
    unit_slope : bool
        Whether scale identification holds the slope at one.

    Raises
    ------
    ValueError
        If the term declares another family, further columns, an unknown
        parameter, or a nonunit pinned slope under unit-slope identification.
    """

    __slots__ = ("_argument", "_domain", "_pinned")

    def __init__(self, term: LinkTerm, *, unit_slope: bool = False) -> None:
        linking = term.linking

        if linking.family != LINEAR:
            msg = f"a linear linking was declared under the family {linking.family!r}"
            raise ValueError(msg)

        if linking.columns:
            msg = (
                "a linear linking reads its argument alone, and this one "
                f"names the roles {sorted(linking.columns)!r}"
            )
            raise ValueError(msg)

        unknown = sorted(frozenset(linking.fixed) - {_SLOPE})

        if unknown:
            msg = (
                f"a linear linking has the parameter {_SLOPE!r}, and these "
                f"were pinned: {unknown!r}"
            )
            raise ValueError(msg)

        pinned = dict(linking.fixed)

        if unit_slope:
            if pinned.get(_SLOPE, 1.0) != 1.0:
                msg = (
                    "the identification holds the slope at one, and the term "
                    f"pins it at {pinned[_SLOPE]!r}"
                )
                raise ValueError(msg)

            pinned[_SLOPE] = 1.0

        self._argument = term.readout
        self._pinned: Mapping[str, float] = MappingProxyType(pinned)
        self._domain = LinkingDomain(
            self._argument,
            {self._argument: _VALUE_INTERVAL},
            {} if _SLOPE in pinned else {_SLOPE: _SLOPE_INTERVAL},
        )

    @property
    def domain(self) -> LinkingDomain:
        """Return the scalar column and any fitted slope."""
        return self._domain

    @property
    def pinned(self) -> Mapping[str, float]:
        """Return the declared fixed slope, if any."""
        return self._pinned

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        """Return the identity map when a unit slope remains reachable."""
        pinned = self._pinned.get(_SLOPE)

        if pinned is None:
            return (SpecialCase("identity", (1.0,)),)

        if pinned == 1.0:
            return (SpecialCase("identity", ()),)

        return ()

    def _slope(self, parameters: Tensor) -> Tensor | float:
        """Return the fitted or pinned slope."""
        pinned = self._pinned.get(_SLOPE)

        return parameters[0] if pinned is None else pinned

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        """Return the slope-scaled readout for every row."""
        return self._slope(parameters) * columns.readouts[self._argument]

    def steps(self, prefix: str, *, signed_slope: bool = False) -> LinkingSteps:
        """Return program steps for the fitted or pinned linear contribution."""
        pinned = self._pinned.get(_SLOPE)
        site = f"{prefix}_{_SLOPE}"
        prior = _SIGNED_SLOPE_PRIOR if signed_slope else _SLOPE_PRIOR
        draws = {} if pinned is not None else {site: prior}
        slope: Expression = Name(site) if pinned is None else Literal(pinned)

        return LinkingSteps(
            draws,
            Binary(Operator.MUL, slope, Name(self._argument)),
        )

    @override
    def __repr__(self) -> str:
        """Return the argument and pinned slope."""
        return (
            f"LinearLinking(argument={self._argument!r}, pinned={dict(self._pinned)!r})"
        )


def linear_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    """Return program steps for a scalar readout entering through a slope."""
    return LinearLinking(term, unit_slope=unit_slope).steps(
        prefix, signed_slope=signed_slope
    )
