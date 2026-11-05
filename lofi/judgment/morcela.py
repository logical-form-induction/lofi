"""MORCELA linking from language-model scores to a latent scale.

The score is ``slope * (p - beta * u + gamma) / l ** delta``. Here ``p`` and
``u`` are sentence and unigram log probabilities, and ``l`` is token count.
SLOR, raw log probability, and either correction alone are nested parameter
settings.
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
from lofi.judgment._spec import LinkingSpec
from lofi.judgment._values import LinkingDomain, SpecialCase
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._ports import ReadoutColumns
    from lofi.judgment._program import Expression
    from lofi.judgment._spec import LinkTerm

__all__ = [
    "FAMILY",
    "FREQUENCY",
    "LENGTH",
    "PARAMETERS",
    "MorcelaLinking",
    "morcela_rungs",
    "morcela_steps",
]

#: The linking family name a link term declares to enter through MORCELA.
FAMILY: Final = "morcela"

#: The column role of the sentence length, in tokens.
LENGTH: Final = "length"

#: The column role of the sentence's unigram log-probability.
FREQUENCY: Final = "frequency"

#: The parameters in vector order: the slope, the frequency-correction
#: strength, the length-correction strength, and the length exponent.
PARAMETERS: Final = ("slope", "frequency", "length", "normalization")

_SLOPE: Final = PARAMETERS[0]
"""The parameter multiplying the corrected score on the latent scale."""

_INTERVALS: Final[Mapping[str, Interval]] = MappingProxyType(
    {
        "slope": Interval(0.0, 10.0),
        "frequency": Interval(-5.0, 5.0),
        "length": Interval(-100.0, 100.0),
        "normalization": Interval(0.0, 1.0),
    }
)
"""The interval each parameter is declared over."""

_PRIORS: Final[Mapping[str, Prior]] = MappingProxyType(
    {
        "slope": Prior("HalfNormal", 1.0),
        "frequency": Prior("Normal", 1.0, 1.0),
        "length": Prior("Normal", 0.0, 10.0),
        "normalization": Prior("Uniform", 0.0, 1.0),
    }
)
"""The prior each fitted parameter is drawn from in a program."""

_LOG_PROBABILITY: Final = Interval(-1000.0, 0.0)
"""The interval a sentence's log-probability under a model is declared over."""

_TOKENS: Final = Interval(1.0, 500.0)
"""The interval a sentence's length in tokens is declared over."""

_CASES: Final[Mapping[str, tuple[float, ...]]] = MappingProxyType(
    {
        "slor": (1.0, 1.0, 0.0, 1.0),
        "log_probability": (1.0, 0.0, 0.0, 0.0),
    }
)
"""The full parameter vector of each nested case."""

_RUNGS: Final[Mapping[str, tuple[float, ...]]] = MappingProxyType(
    {
        **_CASES,
        "length_only": (1.0, 0.0, 0.0, 1.0),
        "unigram_frequency_only": (1.0, 1.0, 0.0, 0.0),
    }
)
"""The full parameter vector of each surface-statistic rung."""


def morcela_rungs(*, length: str, frequency: str) -> Mapping[str, LinkingSpec]:
    """Return the four nested surface-statistic parameterizations.

    The returned linking specifications are, in order, SLOR, raw sentence log
    probability, sentence log probability with only length normalization, and
    sentence log probability with only unigram-frequency correction. Every
    parameter is fixed, so each specification is a closed-form control rather
    than a fitted MORCELA linking.

    Parameters
    ----------
    length : str
        Readout column containing each sentence's token count.
    frequency : str
        Readout column containing each sentence's unigram log probability.
    """
    columns = {LENGTH: length, FREQUENCY: frequency}

    return MappingProxyType(
        {
            name: LinkingSpec(
                family=FAMILY,
                columns=columns,
                fixed=dict(zip(PARAMETERS, vector, strict=True)),
            )
            for name, vector in _RUNGS.items()
        }
    )


@final
class MorcelaLinking:
    """MORCELA over a link term's readout and its length and frequency columns.

    Each row contributes
    ``slope * (p - frequency * u + length) / l ** normalization``. Pinned
    parameters remain fixed; the rest are fitted in ``PARAMETERS`` order.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument, the two further columns by role,
        and the pinned parameters.
    unit_slope : bool
        Whether the slope is held at one by the scale identification.

    Raises
    ------
    ValueError
        If the term's linking is another family, names roles other than the
        length and the frequency, names a parameter this linking does not
        have, pins the slope away from one while the identification holds
        it there, or reads one column under two of its three roles.
    """

    __slots__ = ("_argument", "_domain", "_frequency", "_length", "_pinned")

    def __init__(self, term: LinkTerm, *, unit_slope: bool = False) -> None:
        linking = term.linking

        if linking.family != FAMILY:
            msg = f"a MORCELA linking was declared under the family {linking.family!r}"

            raise ValueError(msg)

        roles = frozenset(linking.columns)

        if roles != {LENGTH, FREQUENCY}:
            msg = (
                f"a MORCELA linking reads the roles {LENGTH!r} and {FREQUENCY!r}, "
                f"and this one names {sorted(roles)!r}"
            )

            raise ValueError(msg)

        unknown = sorted(frozenset(linking.fixed) - frozenset(PARAMETERS))

        if unknown:
            msg = (
                f"a MORCELA linking has the parameters {PARAMETERS!r}, and these "
                f"were pinned: {unknown!r}"
            )

            raise ValueError(msg)

        argument = term.readout
        length = linking.columns[LENGTH]
        frequency = linking.columns[FREQUENCY]

        if len({argument, length, frequency}) != 3:
            msg = (
                f"a MORCELA linking reads three columns, and this one reads "
                f"{argument!r} as the argument, {length!r} as the length and "
                f"{frequency!r} as the frequency"
            )

            raise ValueError(msg)

        pinned = dict(linking.fixed)

        if unit_slope:
            if pinned.get(_SLOPE, 1.0) != 1.0:
                msg = (
                    f"the identification holds the slope at one, and the term "
                    f"pins it at {pinned[_SLOPE]!r}"
                )

                raise ValueError(msg)

            pinned[_SLOPE] = 1.0

        self._argument = argument
        self._length = length
        self._frequency = frequency
        self._pinned: Mapping[str, float] = MappingProxyType(pinned)
        intervals = {LENGTH: _TOKENS, FREQUENCY: _LOG_PROBABILITY}
        self._domain = LinkingDomain(
            argument,
            {
                argument: _LOG_PROBABILITY,
                **{name: intervals[role] for role, name in linking.columns.items()},
            },
            {name: _INTERVALS[name] for name in PARAMETERS if name not in self._pinned},
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
        """Return the nested cases the pinned parameters leave reachable.

        SLOR fixes slope and frequency to one, length correction to zero, and the
        normalization exponent to one. Raw log probability fixes slope to one and
        every correction to zero. Cases inconsistent with pinned parameters are
        omitted, and returned vectors contain only fitted coordinates.
        """
        cases = []

        for name, vector in _CASES.items():
            full = dict(zip(PARAMETERS, vector, strict=True))

            if all(full[pin] == value for pin, value in self._pinned.items()):
                cases.append(
                    SpecialCase(
                        name,
                        tuple(full[fitted] for fitted in self._domain.parameter_names),
                    )
                )

        return tuple(cases)

    def _values(self, parameters: Tensor) -> tuple[Tensor | float, ...]:
        """Return every parameter in vector order, pinned or fitted.

        Parameters
        ----------
        parameters : Tensor
            Fitted parameter vector, in the order the domain declares.
        """
        fitted = iter(parameters)

        return tuple(
            self._pinned[name] if name in self._pinned else next(fitted)
            for name in PARAMETERS
        )

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        """Return one latent-scale contribution per row.

        Parameters
        ----------
        columns : ReadoutColumns
            Readout columns holding the argument, the length and the
            frequency.
        parameters : Tensor
            Fitted parameter vector, in the order the domain declares.
        """
        slope, frequency, length, normalization = self._values(parameters)
        readouts = columns.readouts
        corrected = (
            readouts[self._argument] - frequency * readouts[self._frequency] + length
        )

        return slope * corrected / torch.pow(readouts[self._length], normalization)

    def steps(self, prefix: str) -> LinkingSteps:
        """Return the program form of this linking.

        Each fitted parameter is drawn at the site named by the prefix and
        the parameter, and each pinned parameter is written as a literal.

        Parameters
        ----------
        prefix : str
            Site prefix, distinct for every link term of a program.
        """
        draws = {
            f"{prefix}_{name}": _PRIORS[name]
            for name in PARAMETERS
            if name not in self._pinned
        }
        written: dict[str, Expression] = {
            name: Literal(self._pinned[name])
            if name in self._pinned
            else Name(f"{prefix}_{name}")
            for name in PARAMETERS
        }
        corrected = Binary(
            Operator.ADD,
            Binary(
                Operator.SUB,
                Name(self._argument),
                Binary(Operator.MUL, written["frequency"], Name(self._frequency)),
            ),
            written["length"],
        )
        divisor = Apply(
            Function.EXP,
            Binary(
                Operator.MUL,
                written["normalization"],
                Apply(Function.LOG, Name(self._length)),
            ),
        )
        expression = Binary(
            Operator.DIV, Binary(Operator.MUL, written["slope"], corrected), divisor
        )

        return LinkingSteps(draws, expression)

    @override
    def __repr__(self) -> str:
        """Return the columns and the pinned parameters."""
        return (
            f"MorcelaLinking(argument={self._argument!r}, length={self._length!r}, "
            f"frequency={self._frequency!r}, pinned={dict(self._pinned)!r})"
        )


def morcela_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    """Return the program form of a link term entering through MORCELA.

    Parameters
    ----------
    term : LinkTerm
        Link term declaring the argument, the two further columns by role,
        and the pinned parameters.
    prefix : str
        Site prefix, distinct for every link term of a program.
    unit_slope : bool
        Whether the slope is held at one by the scale identification.
    signed_slope : bool
        Whether a fitted slope may cross zero. MORCELA keeps its declared
        monotone domain, so this option has no effect.

    Raises
    ------
    ValueError
        If the term does not declare a MORCELA linking, or pins the slope
        away from one while the identification holds it there.
    """
    del signed_slope

    return MorcelaLinking(term, unit_slope=unit_slope).steps(prefix)
