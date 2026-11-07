"""Interior judgment values.

Attributes
----------
AgreementCeiling
    Agreement estimate, interval, and sample counts.
LinkingDomain
    The readout columns and fitted parameters a linking is declared over.
SentenceScore
    The per-token log-probabilities and unigram log-probabilities of one
    sentence.
SpecialCase
    A nested special case of a linking at declared parameter values.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor

__all__ = [
    "AgreementCeiling",
    "LinkingDomain",
    "SentenceScore",
    "SpecialCase",
]


@final
class AgreementCeiling:
    """An agreement estimate and interval.

    Parameters
    ----------
    value : float
        Agreement estimate.
    lower : float
        Lower interval bound.
    upper : float
        Upper interval bound.
    items : int
        Number of evaluated items.
    responses : int
        Total number of responses.

    Raises
    ------
    TypeError
        If a numeric value or count has the wrong type.
    ValueError
        If a value is not finite, a count is negative, or the interval does
        not contain the estimate.
    """

    __slots__ = (
        "_items",
        "_lower",
        "_responses",
        "_upper",
        "_value",
    )

    def __init__(
        self,
        value: float,
        lower: float,
        upper: float,
        *,
        items: int,
        responses: int,
    ) -> None:
        self._value = ensure_finite("an agreement ceiling's value", value)
        self._lower = ensure_finite("an agreement ceiling's lower bound", lower)
        self._upper = ensure_finite("an agreement ceiling's upper bound", upper)

        if not self._lower <= self._value <= self._upper:
            msg = (
                f"an agreement ceiling's interval must bracket its value: "
                f"{self._lower!r}, {self._value!r}, {self._upper!r}"
            )

            raise ValueError(msg)

        self._items = ensure_non_negative("an agreement ceiling's item count", items)
        self._responses = ensure_non_negative(
            "an agreement ceiling's response count", responses
        )

    @property
    def value(self) -> float:
        """Return the agreement estimate."""
        return self._value

    @property
    def interval(self) -> tuple[float, float]:
        """Return the confidence interval."""
        return self._lower, self._upper

    @property
    def items(self) -> int:
        """Return the item count."""
        return self._items

    @property
    def responses(self) -> int:
        """Return the response count."""
        return self._responses


def _named_intervals(role: str, entries: Mapping[str, Interval]) -> dict[str, Interval]:
    """Return a copy of named intervals with every name checked.

    Parameters
    ----------
    role : str
        What the names denote, used in error messages.
    entries : Mapping[str, Interval]
        Intervals by name.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty.
    """
    checked: dict[str, Interval] = {}

    for name, interval in entries.items():
        if not isinstance(interval, Interval) or interval.level is not None:
            msg = f"the interval of {name!r} is not a domain interval: {interval!r}"
            raise TypeError(msg)

        if interval.high <= interval.low:
            msg = f"the interval of {name!r} has no interior: {interval!r}"
            raise ValueError(msg)

        checked[ensure_name(role, name)] = interval

    return checked


@final
class LinkingDomain:
    """The readout columns and fitted parameters a linking is declared over.

    The argument is the column the linking is monotone in. Every column
    named here, the argument included, is read by the linking and nothing
    else is. The parameter intervals fix the order of the fitted parameter
    vector.

    Parameters
    ----------
    argument : str
        Readout column the linking is non-decreasing in.
    columns : Mapping[str, Interval]
        Interval of every readout column the linking reads, by name.
    parameters : Mapping[str, Interval]
        Interval of every fitted parameter, by name, in vector order.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty or the argument is not among the columns.
    """

    __slots__ = ("_argument", "_columns", "_parameters")

    def __init__(
        self,
        argument: str,
        columns: Mapping[str, Interval],
        parameters: Mapping[str, Interval] = MappingProxyType({}),
    ) -> None:
        checked_columns = _named_intervals("a readout column", columns)
        checked_argument = ensure_name("a linking's argument", argument)

        if checked_argument not in checked_columns:
            msg = (
                f"a linking's argument {checked_argument!r} is not among its "
                f"columns {tuple(checked_columns)!r}"
            )

            raise ValueError(msg)

        self._argument = checked_argument
        self._columns: Mapping[str, Interval] = MappingProxyType(checked_columns)
        self._parameters: Mapping[str, Interval] = MappingProxyType(
            _named_intervals("a linking parameter", parameters)
        )

    @property
    def argument(self) -> str:
        """Return the column the linking is non-decreasing in."""
        return self._argument

    @property
    def columns(self) -> Mapping[str, Interval]:
        """Return the interval of each readout column the linking reads."""
        return self._columns

    @property
    def parameters(self) -> Mapping[str, Interval]:
        """Return the interval of each fitted parameter, in vector order."""
        return self._parameters

    @property
    def readouts(self) -> tuple[str, ...]:
        """Return the names of the readout columns the linking reads."""
        return tuple(self._columns)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the fitted parameter names in vector order."""
        return tuple(self._parameters)

    def _identity(
        self,
    ) -> tuple[str, tuple[tuple[str, Interval], ...], tuple[tuple[str, Interval], ...]]:
        """Return the fields in comparison order."""
        return (
            self._argument,
            tuple(self._columns.items()),
            tuple(self._parameters.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another domain declares the same names and intervals."""
        if type(other) is not LinkingDomain:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the names and intervals."""
        return hash((LinkingDomain, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this domain."""
        return (
            f"LinkingDomain({self._argument!r}, {dict(self._columns)!r}, "
            f"{dict(self._parameters)!r})"
        )


@final
class SpecialCase:
    """A nested special case of a linking at declared parameter values.

    Parameters
    ----------
    name : str
        Name of the case.
    parameters : Iterable[float]
        Parameter vector at which the linking reduces to the case.

    Raises
    ------
    TypeError
        If the name is not a string or a parameter is not a number.
    ValueError
        If the name is empty or a parameter is not finite.
    """

    __slots__ = ("_name", "_parameters")

    def __init__(self, name: str, parameters: Iterable[float]) -> None:
        self._name = ensure_name("a special case's name", name)
        self._parameters = tuple(
            ensure_finite(f"a parameter of the special case {self._name!r}", value)
            for value in parameters
        )

    @property
    def name(self) -> str:
        """Return the name of the case."""
        return self._name

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return the parameter vector at which the linking reduces to the case."""
        return self._parameters

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another case has the same name and parameters."""
        if type(other) is not SpecialCase:
            return NotImplemented

        return (self._name, self._parameters) == (other._name, other._parameters)

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and parameters."""
        return hash((SpecialCase, self._name, self._parameters))

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this case."""
        return f"SpecialCase({self._name!r}, {self._parameters!r})"


def _log_probabilities(field: str, values: Tensor, tokens: int) -> Tensor:
    """Return one nonpositive, finite log probability per token.

    Parameters
    ----------
    field : str
        What the values are, used in error messages.
    values : Tensor
        Candidate log-probabilities.
    tokens : int
        Number of tokens the values must cover.

    Raises
    ------
    TypeError
        If the values are not floating point.
    ValueError
        If the values are not one per token, or a value is not finite or is
        positive.
    """
    if not values.is_floating_point():
        msg = f"{field} hold {values.dtype}, which is not a floating-point kind"

        raise TypeError(msg)

    if values.ndim != 1 or values.numel() != tokens:
        msg = f"{field} have shape {tuple(values.shape)} over {tokens} tokens"

        raise ValueError(msg)

    if not bool(values.isfinite().all()):
        msg = f"{field} contain a value that is not finite"

        raise ValueError(msg)

    if bool((values > 0.0).any()):
        msg = f"{field} contain a positive value, which no probability has"

        raise ValueError(msg)

    return values


@final
class SentenceScore:
    """The per-token log-probabilities and unigram log-probabilities of one sentence.

    Both vectors run over the same tokens in order. A token's log probability
    is the language model's own, conditioned on the tokens before it, and its
    unigram log probability is read from a frequency table.

    Parameters
    ----------
    tokens : tuple[str, ...]
        The sentence's tokens, at least one.
    log_probabilities : Tensor
        Log probability of each token under the language model.
    unigram_log_probabilities : Tensor
        Log relative frequency of each token in the unigram table.

    Raises
    ------
    TypeError
        If a token is not a string, or a vector is not floating point.
    ValueError
        If no token is given, a token is empty, a vector is not one value per
        token, or a value is not finite or is positive.
    """

    __slots__ = ("_log_probabilities", "_tokens", "_unigram_log_probabilities")

    def __init__(
        self,
        tokens: tuple[str, ...],
        log_probabilities: Tensor,
        unigram_log_probabilities: Tensor,
    ) -> None:
        if not tokens:
            msg = "a sentence score covers no tokens"

            raise ValueError(msg)

        self._tokens = tuple(ensure_name("a scored token", token) for token in tokens)
        self._log_probabilities = _log_probabilities(
            "a sentence's log-probabilities", log_probabilities, len(self._tokens)
        )
        self._unigram_log_probabilities = _log_probabilities(
            "a sentence's unigram log-probabilities",
            unigram_log_probabilities,
            len(self._tokens),
        )

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the scored tokens in order."""
        return self._tokens

    @property
    def length(self) -> int:
        """Return the number of tokens."""
        return len(self._tokens)

    @property
    def log_probabilities(self) -> Tensor:
        """Return each token's log probability under the language model."""
        return self._log_probabilities

    @property
    def unigram_log_probabilities(self) -> Tensor:
        """Return each token's log relative frequency in the unigram table."""
        return self._unigram_log_probabilities

    @property
    def log_probability(self) -> Tensor:
        """Return the sentence's log probability, summed over its tokens."""
        return self._log_probabilities.sum()

    @property
    def unigram_log_probability(self) -> Tensor:
        """Return the sentence's unigram log probability, summed over its tokens."""
        return self._unigram_log_probabilities.sum()

    @override
    def __repr__(self) -> str:
        """Return the tokens and the two totals."""
        return (
            f"SentenceScore(tokens={self._tokens!r}, "
            f"log_probability={float(self.log_probability)!r}, "
            f"unigram_log_probability={float(self.unigram_log_probability)!r})"
        )
