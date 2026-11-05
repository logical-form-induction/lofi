"""Links over a published mass and its complement.

The logit link computes ``log m - log(1 - m)`` from the two stored log
probabilities, preserving precision near unit mass. The probability link
returns ``exp(log m)``.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi.judgment._program import (
    Apply,
    Binary,
    Function,
    LinkingSteps,
    Literal,
    Name,
    Operator,
    Prior,
)
from lofi.judgment._values import LinkingDomain, SpecialCase
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._ports import ReadoutColumns
    from lofi.judgment._program import Expression
    from lofi.judgment._spec import LinkTerm

__all__ = [
    "COMPLEMENT",
    "LOGIT",
    "PROBABILITY",
    "SLOPE",
    "LogitLinking",
    "ProbabilityLinking",
    "log_odds",
    "log_odds_spread",
    "logit_steps",
    "probability_steps",
]

#: The linking family name a link term declares to enter through the log-odds.
LOGIT: Final = "logit"

#: The linking family name a link term declares to enter through the mass.
PROBABILITY: Final = "probability"

#: The column role of the log mass everywhere but the target set.
COMPLEMENT: Final = "complement"

#: The one parameter of the logit link: the slope on the log-odds.
SLOPE: Final = "slope"

_LOG_PROBABILITY: Final = Interval(-1000.0, 0.0)
"""The interval a published log probability is declared over."""

_SLOPE_INTERVAL: Final = Interval(0.0, 10.0)
"""The interval the slope on the log-odds is declared over."""

_SLOPE_PRIOR: Final = Prior("HalfNormal", 1.0)
"""The prior a fitted slope is drawn from in a program."""

_SIGNED_SLOPE_PRIOR: Final = Prior("Normal", 0.0, 1.0)
"""The prior on a slope identified by a fixed cutpoint variance."""


def log_odds(log_mass: Tensor, log_complement: Tensor) -> Tensor:
    """Return the log-odds of a mass from its published log pair.

    Parameters
    ----------
    log_mass : Tensor
        Log of the mass, one value per row.
    log_complement : Tensor
        Log of one minus the mass, one value per row.
    """
    return log_mass - log_complement


def log_odds_spread(log_mass: Tensor, log_complement: Tensor) -> Tensor:
    """Measure the range of a published mass pair on the log-odds scale.

    Parameters
    ----------
    log_mass : Tensor
        Log of the mass, one value per row.
    log_complement : Tensor
        Log of one minus the mass, one value per row.
    """
    odds = log_odds(log_mass, log_complement)

    return torch.max(odds) - torch.min(odds)


def _ensure_family(term: LinkTerm, family: str) -> None:
    """Check that a link term declares the expected linking family.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the linking.
    family : str
        Family the linking implements.

    Raises
    ------
    ValueError
        If the term's linking is another family.
    """
    if term.linking.family != family:
        msg = (
            f"a {family} linking was declared under the family {term.linking.family!r}"
        )

        raise ValueError(msg)


@final
class LogitLinking:
    """The log-odds of a published mass pair, times a slope.

    Each row contributes ``slope * (log m - log(1 - m))``. The slope is fitted
    unless pinned by the term or held at one by scale identification.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument, the complement column by role,
        and the pinned slope, if any.
    unit_slope : bool
        Whether the slope is held at one by the scale identification.

    Raises
    ------
    ValueError
        If the term's linking is another family, names a role other than
        the complement, pins a parameter this linking does not have, pins
        the slope away from one while the identification holds it there,
        or reads one column as both the argument and the complement.
    """

    __slots__ = ("_argument", "_complement", "_domain", "_pinned")

    def __init__(self, term: LinkTerm, *, unit_slope: bool = False) -> None:
        _ensure_family(term, LOGIT)
        linking = term.linking
        roles = frozenset(linking.columns)

        if roles != {COMPLEMENT}:
            msg = (
                f"a logit linking reads the role {COMPLEMENT!r}, and this one "
                f"names {sorted(roles)!r}"
            )

            raise ValueError(msg)

        unknown = sorted(frozenset(linking.fixed) - {SLOPE})

        if unknown:
            msg = (
                f"a logit linking has the parameter {SLOPE!r}, and these were "
                f"pinned: {unknown!r}"
            )

            raise ValueError(msg)

        argument = term.readout
        complement = linking.columns[COMPLEMENT]

        if argument == complement:
            msg = (
                f"a logit linking reads two columns, and this one reads "
                f"{argument!r} as both the argument and the complement"
            )

            raise ValueError(msg)

        pinned = dict(linking.fixed)

        if unit_slope:
            if pinned.get(SLOPE, 1.0) != 1.0:
                msg = (
                    f"the identification holds the slope at one, and the term "
                    f"pins it at {pinned[SLOPE]!r}"
                )

                raise ValueError(msg)

            pinned[SLOPE] = 1.0

        self._argument = argument
        self._complement = complement
        self._pinned: Mapping[str, float] = MappingProxyType(pinned)
        self._domain = LinkingDomain(
            argument,
            {argument: _LOG_PROBABILITY, complement: _LOG_PROBABILITY},
            {} if SLOPE in self._pinned else {SLOPE: _SLOPE_INTERVAL},
        )

    @property
    def domain(self) -> LinkingDomain:
        """Return the columns read, the argument, and the fitted parameters."""
        return self._domain

    @property
    def pinned(self) -> Mapping[str, float]:
        """Return the parameters held at declared values, by name."""
        return self._pinned

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        """Return the log-odds as the case at a slope of one.

        The case is declared unless the slope is pinned elsewhere, and a
        pinned unit slope declares it with no fitted coordinate.
        """
        pinned = self._pinned.get(SLOPE)

        if pinned is None:
            return (SpecialCase("log_odds", (1.0,)),)

        if pinned == 1.0:
            return (SpecialCase("log_odds", ()),)

        return ()

    def _slope(self, parameters: Tensor) -> Tensor | float:
        """Return the slope, pinned or fitted.

        Parameters
        ----------
        parameters : Tensor
            Fitted parameter vector, in the order the domain declares.
        """
        pinned = self._pinned.get(SLOPE)

        return parameters[0] if pinned is None else pinned

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        """Return one latent-scale contribution per row.

        Parameters
        ----------
        columns : ReadoutColumns
            Readout columns holding the log mass and its log complement.
        parameters : Tensor
            Fitted parameter vector, in the order the domain declares.
        """
        readouts = columns.readouts

        return self._slope(parameters) * log_odds(
            readouts[self._argument], readouts[self._complement]
        )

    def steps(self, prefix: str, *, signed_slope: bool = False) -> LinkingSteps:
        """Return the program form of this linking.

        A fitted slope is drawn at the site named by the prefix and the
        parameter, and a pinned slope is written as a literal.

        Parameters
        ----------
        prefix : str
            Site prefix, distinct for every link term of a program.
        signed_slope : bool
            Whether a fitted slope may cross zero.
        """
        pinned = self._pinned.get(SLOPE)
        site = f"{prefix}_{SLOPE}"
        prior = _SIGNED_SLOPE_PRIOR if signed_slope else _SLOPE_PRIOR
        draws = {} if pinned is not None else {site: prior}
        slope: Expression = Name(site) if pinned is None else Literal(pinned)
        odds = Binary(Operator.SUB, Name(self._argument), Name(self._complement))

        return LinkingSteps(draws, Binary(Operator.MUL, slope, odds))

    @override
    def __repr__(self) -> str:
        """Return the columns and the pinned slope."""
        return (
            f"LogitLinking(argument={self._argument!r}, "
            f"complement={self._complement!r}, pinned={dict(self._pinned)!r})"
        )


@final
class ProbabilityLinking:
    """The mass of a published mass pair, read from its log.

    The contribution of a row is ``exp(log m)``, with ``log m`` the argument
    column. Nothing is fitted, and the linking has one case, itself.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument.

    Raises
    ------
    ValueError
        If the term's linking is another family, names a further column,
        or pins a parameter.
    """

    __slots__ = ("_argument", "_domain")

    def __init__(self, term: LinkTerm) -> None:
        _ensure_family(term, PROBABILITY)
        linking = term.linking

        if linking.columns:
            msg = (
                f"a probability linking reads its argument alone, and this one "
                f"names the roles {sorted(linking.columns)!r}"
            )

            raise ValueError(msg)

        if linking.fixed:
            msg = (
                f"a probability linking has no parameter, and these were pinned: "
                f"{sorted(linking.fixed)!r}"
            )

            raise ValueError(msg)

        self._argument = term.readout
        self._domain = LinkingDomain(self._argument, {self._argument: _LOG_PROBABILITY})

    @property
    def domain(self) -> LinkingDomain:
        """Return the argument column, which is all the linking reads."""
        return self._domain

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        """Return the mass as the one case, at the empty parameter vector."""
        return (SpecialCase("mass", ()),)

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        """Return one latent-scale contribution per row.

        Parameters
        ----------
        columns : ReadoutColumns
            Readout columns holding the log mass.
        parameters : Tensor
            Fitted parameter vector, which is empty.
        """
        del parameters

        return torch.exp(columns.readouts[self._argument])

    def steps(self, prefix: str) -> LinkingSteps:
        """Return the program form of this linking, which draws nothing.

        Parameters
        ----------
        prefix : str
            Site prefix, distinct for every link term of a program.
        """
        del prefix

        return LinkingSteps({}, Apply(Function.EXP, Name(self._argument)))

    @override
    def __repr__(self) -> str:
        """Return the argument column."""
        return f"ProbabilityLinking(argument={self._argument!r})"


def logit_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    """Return the program form of a link term entering through the log-odds.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument, the complement column by role,
        and the pinned slope, if any.
    prefix : str
        Site prefix, distinct for every link term of a program.
    unit_slope : bool
        Whether the slope is held at one by the scale identification.
    signed_slope : bool
        Whether a fitted slope may cross zero.

    Raises
    ------
    ValueError
        If the term does not declare a logit linking, or pins the slope
        away from one while the identification holds it there.
    """
    return LogitLinking(term, unit_slope=unit_slope).steps(
        prefix, signed_slope=signed_slope
    )


def probability_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    """Return the program form of a link term entering through the mass.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument.
    prefix : str
        Site prefix, distinct for every link term of a program.
    unit_slope : bool
        Whether the identification holds slopes at one, which this linking
        has none of.
    signed_slope : bool
        Whether a fitted slope may cross zero, which this linking has none of.

    Raises
    ------
    ValueError
        If the term does not declare a probability linking.
    """
    del signed_slope, unit_slope

    return ProbabilityLinking(term).steps(prefix)
