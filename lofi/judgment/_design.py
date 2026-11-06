"""Dense tensor encodings of judgment rows."""

from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Never, Protocol, cast, final, override

from lofi._checks import ensure_finite, ensure_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, dtype

    from lofi.judgment._ports import BoundedScale, JudgmentSource

__all__ = [
    "POSITION",
    "BoundedDesign",
    "BoundedDesignCodec",
    "BoundedJudgmentRow",
    "ClosedScale",
    "DesignCodec",
    "DesignError",
    "JudgmentDesign",
    "JudgmentRow",
    "ResponseDesign",
    "RowColumns",
    "design_of",
]

#: The factor whose level is an alternative's place in presentation order.
POSITION = "position"

_MINIMUM_LEVELS = 2
"""The minimum response-level count."""

_MINIMUM_ALTERNATIVES = 2
"""The minimum forced-choice alternative count."""


class DesignError(ValueError):
    """Reports an invalid judgment design."""


class _Torch(Protocol):
    """Torch operations used by the codec."""

    int64: dtype
    float64: dtype

    def tensor(
        self,
        data: Sequence[int | float],
        *,
        dtype: dtype,
    ) -> Tensor:
        """Construct a tensor."""
        ...

    def unique(self, values: Tensor) -> Tensor:
        """Return sorted unique values."""
        ...

    def equal(self, left: Tensor, right: Tensor) -> bool:
        """Return whether two tensors agree."""
        ...

    def arange(self, end: int, *, dtype: dtype) -> Tensor:
        """Return consecutive integers below an endpoint."""
        ...

    def isfinite(self, values: Tensor) -> Tensor:
        """Return a finite-value mask."""
        ...


def _torch() -> _Torch:
    """Load torch when design encoding is requested."""
    return cast("_Torch", import_module("torch"))


@final
class RowColumns:
    """Stores factor and readout values for one row.

    A row judged on its own gives each readout one value. A row on which
    several alternatives were shown together gives each readout one value
    per alternative instead, in the order the alternatives were presented.

    Parameters
    ----------
    factors : dict[str, str]
        Level of each declared factor.
    readouts : dict[str, float]
        Value of each declared readout for a row judged on its own.
    alternative_readouts : dict[str, tuple[float, ...]] | None
        Value of each declared readout for every alternative of a row, in
        presentation order.
    """

    __slots__ = ("_alternative_readouts", "_factors", "_readouts")

    def __init__(
        self,
        factors: dict[str, str],
        readouts: dict[str, float],
        alternative_readouts: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self._factors: Mapping[str, str] = MappingProxyType(dict(factors))
        self._readouts: Mapping[str, float] = MappingProxyType(dict(readouts))
        self._alternative_readouts: Mapping[str, tuple[float, ...]] = MappingProxyType(
            {} if alternative_readouts is None else dict(alternative_readouts)
        )

    @property
    def factors(self) -> Mapping[str, str]:
        """Return factor levels."""
        return self._factors

    @property
    def readouts(self) -> Mapping[str, float]:
        """Return readout values."""
        return self._readouts

    @property
    def alternative_readouts(self) -> Mapping[str, tuple[float, ...]]:
        """Return readout values per alternative, in presentation order."""
        return self._alternative_readouts


@final
class ClosedScale:
    """A finite closed response interval.

    Parameters
    ----------
    lower : float
        Inclusive lower endpoint.
    upper : float
        Inclusive upper endpoint, greater than the lower endpoint.

    Raises
    ------
    TypeError
        If either endpoint is not numeric.
    ValueError
        If either endpoint is nonfinite or the interval is empty or reversed.
    """

    __slots__ = ("_lower", "_upper")

    def __init__(self, lower: float, upper: float) -> None:
        checked_lower = ensure_finite("a closed scale's lower endpoint", lower)
        checked_upper = ensure_finite("a closed scale's upper endpoint", upper)

        if checked_lower >= checked_upper:
            msg = (
                "a closed scale's lower endpoint is less than its upper endpoint: "
                f"{checked_lower!r}, {checked_upper!r}"
            )

            raise DesignError(msg)

        self._lower = checked_lower
        self._upper = checked_upper

    @property
    def lower(self) -> float:
        """Return the inclusive lower endpoint."""
        return self._lower

    @property
    def upper(self) -> float:
        """Return the inclusive upper endpoint."""
        return self._upper

    def normalize(self, value: float, /) -> float:
        """Return a response rescaled to the closed unit interval.

        Parameters
        ----------
        value : float
            Response recorded on this scale.

        Raises
        ------
        DesignError
            If the response falls outside the closed interval.
        """
        checked = ensure_finite("a bounded response", value)

        if checked < self._lower or checked > self._upper:
            msg = (
                f"a bounded response {checked!r} falls outside the closed scale "
                f"[{self._lower!r}, {self._upper!r}]"
            )

            raise DesignError(msg)

        return (checked - self._lower) / (self._upper - self._lower)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another scale has the same endpoints."""
        if type(other) is not ClosedScale:
            return NotImplemented

        return (self._lower, self._upper) == (other._lower, other._upper)

    @override
    def __hash__(self) -> int:
        """Return a hash over both endpoints."""
        return hash((ClosedScale, self._lower, self._upper))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"ClosedScale({self._lower!r}, {self._upper!r})"


@final
class BoundedJudgmentRow:
    """One floating response recorded on a finite closed scale.

    Parameters
    ----------
    participant : str
        Identifier of the participant who responded.
    item : str
        Identifier of the item judged.
    response : float
        Floating response in the scale's closed interval.
    scale : BoundedScale
        Original response scale.
    columns : RowColumns
        Factor levels and readout values of the row.

    Raises
    ------
    TypeError
        If the response is not a floating-point value.
    DesignError
        If the response is outside its scale or alternative readouts are given.
    """

    __slots__ = (
        "_factors",
        "_item",
        "_participant",
        "_readouts",
        "_response",
        "_scale",
    )

    def __init__(
        self,
        participant: str,
        item: str,
        response: float,
        scale: BoundedScale,
        columns: RowColumns,
    ) -> None:
        if not isinstance(response, float):
            msg = f"a bounded response is not a floating-point value: {response!r}"

            raise TypeError(msg)

        if columns.alternative_readouts:
            msg = "a bounded response is recorded for one item, not alternatives"

            raise DesignError(msg)

        checked_factors = {
            ensure_name("a factor name", name): ensure_name(
                f"factor {name!r}'s level", value
            )
            for name, value in columns.factors.items()
        }
        checked_readouts = {
            ensure_name("a readout name", name): ensure_finite(
                f"readout {name!r}", value
            )
            for name, value in columns.readouts.items()
        }
        checked_scale = ClosedScale(scale.lower, scale.upper)
        self._participant = ensure_name("a participant", participant)
        self._item = ensure_name("an item", item)
        self._response = ensure_finite("a bounded response", response)
        checked_scale.normalize(self._response)
        self._scale = checked_scale
        self._factors: Mapping[str, str] = MappingProxyType(checked_factors)
        self._readouts: Mapping[str, float] = MappingProxyType(checked_readouts)

    @property
    def participant(self) -> str:
        """Return the participant identifier."""
        return self._participant

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item

    @property
    def response(self) -> float:
        """Return the response on its original scale."""
        return self._response

    @property
    def scale(self) -> ClosedScale:
        """Return the original closed response scale."""
        return self._scale

    @property
    def factors(self) -> Mapping[str, str]:
        """Return declared factor levels."""
        return self._factors

    @property
    def readouts(self) -> Mapping[str, float]:
        """Return declared readout values."""
        return self._readouts

    def _identity(
        self,
    ) -> tuple[
        str,
        str,
        float,
        ClosedScale,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, float], ...],
    ]:
        """Return the row fields in comparison order."""
        return (
            self._participant,
            self._item,
            self._response,
            self._scale,
            tuple(sorted(self._factors.items())),
            tuple(sorted(self._readouts.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same fields."""
        if type(other) is not BoundedJudgmentRow:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the row fields."""
        return hash((BoundedJudgmentRow, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this row."""
        return (
            f"BoundedJudgmentRow({self._participant!r}, {self._item!r}, "
            f"{self._response!r}, {self._scale!r}, "
            f"RowColumns({dict(self._factors)!r}, {dict(self._readouts)!r}))"
        )


@final
class JudgmentRow:
    """One response-model input row.

    Parameters
    ----------
    participant : str
        Identifier of the participant who responded.
    item : str
        Identifier of the item judged.
    response : int
        Zero-based level recorded.
    columns : RowColumns
        Factor levels and readout values of the row.
    alternatives : tuple[str, ...] | None
        Identifiers of the alternatives shown together, in presentation
        order, or None for a row judged on its own.

    Raises
    ------
    TypeError
        If the response is not an integer.
    DesignError
        If fewer than two alternatives are shown, an alternative repeats,
        or a readout is given per alternative without an alternative for
        every value.
    """

    __slots__ = (
        "_alternative_readouts",
        "_alternatives",
        "_factors",
        "_item",
        "_participant",
        "_readouts",
        "_response",
    )

    def __init__(
        self,
        participant: str,
        item: str,
        response: int,
        columns: RowColumns,
        alternatives: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(response, bool) or not isinstance(response, int):
            msg = f"a judgment response is not an integer: {response!r}"

            raise TypeError(msg)

        checked_factors = {
            ensure_name("a factor name", name): ensure_name(
                f"factor {name!r}'s level", value
            )
            for name, value in columns.factors.items()
        }
        checked_readouts = {
            ensure_name("a readout name", name): ensure_finite(
                f"readout {name!r}", value
            )
            for name, value in columns.readouts.items()
        }

        if alternatives is not None:
            if len(alternatives) < _MINIMUM_ALTERNATIVES:
                msg = (
                    f"a forced-choice row has {len(alternatives)} alternatives and "
                    f"needs {_MINIMUM_ALTERNATIVES}"
                )

                raise DesignError(msg)

            checked_alternatives = tuple(
                ensure_name("an alternative", alternative)
                for alternative in alternatives
            )

            if len(frozenset(checked_alternatives)) != len(checked_alternatives):
                msg = f"a forced-choice row repeats an alternative: {alternatives!r}"

                raise DesignError(msg)
        else:
            checked_alternatives = None

        shown = 0 if checked_alternatives is None else len(checked_alternatives)
        checked_per_alternative = {
            ensure_name("a readout name", name): tuple(
                ensure_finite(f"readout {name!r} of alternative {at}", value)
                for at, value in enumerate(values)
            )
            for name, values in columns.alternative_readouts.items()
        }

        for name, values in checked_per_alternative.items():
            if len(values) != shown:
                msg = (
                    f"readout {name!r} has {len(values)} values and the row shows "
                    f"{shown} alternatives"
                )

                raise DesignError(msg)

        self._participant = ensure_name("a participant", participant)
        self._item = ensure_name("an item", item)
        self._response = response
        self._factors: Mapping[str, str] = MappingProxyType(checked_factors)
        self._readouts: Mapping[str, float] = MappingProxyType(checked_readouts)
        self._alternatives = checked_alternatives
        self._alternative_readouts: Mapping[str, tuple[float, ...]] = MappingProxyType(
            checked_per_alternative
        )

    @property
    def participant(self) -> str:
        """Return the participant identifier."""
        return self._participant

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item

    @property
    def response(self) -> int:
        """Return the zero-based response index."""
        return self._response

    @property
    def factors(self) -> Mapping[str, str]:
        """Return declared factor levels."""
        return self._factors

    @property
    def readouts(self) -> Mapping[str, float]:
        """Return declared readout values."""
        return self._readouts

    @property
    def alternatives(self) -> tuple[str, ...] | None:
        """Return forced-choice alternatives, if present."""
        return self._alternatives

    @property
    def alternative_readouts(self) -> Mapping[str, tuple[float, ...]]:
        """Return declared readout values per alternative, in presentation order."""
        return self._alternative_readouts

    def _identity(
        self,
    ) -> tuple[
        str,
        str,
        int,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, float], ...],
        tuple[str, ...] | None,
        tuple[tuple[str, tuple[float, ...]], ...],
    ]:
        """Return the row fields in comparison order."""
        return (
            self._participant,
            self._item,
            self._response,
            tuple(sorted(self._factors.items())),
            tuple(sorted(self._readouts.items())),
            self._alternatives,
            tuple(sorted(self._alternative_readouts.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same fields."""
        if type(other) is not JudgmentRow:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the row fields."""
        return hash((JudgmentRow, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call for this row."""
        per_alternative = (
            ""
            if not self._alternative_readouts
            else f", {dict(self._alternative_readouts)!r}"
        )

        return (
            f"JudgmentRow({self._participant!r}, {self._item!r}, "
            f"{self._response!r}, RowColumns({dict(self._factors)!r}, "
            f"{dict(self._readouts)!r}{per_alternative}), {self._alternatives!r})"
        )


class _DesignData(NamedTuple):
    """Stores validated design fields for construction."""

    participants: Tensor
    participant_levels: tuple[str, ...]
    items: Tensor
    item_levels: tuple[str, ...]
    factors: Mapping[str, Tensor]
    factor_levels: Mapping[str, tuple[str, ...]]
    responses: Tensor
    response_levels: int
    readouts: Mapping[str, Tensor]
    alternatives: tuple[tuple[str, ...] | None, ...]
    alternatives_per_row: int | None


@final
class JudgmentDesign:
    """Stores compact response-model columns.

    A design whose rows were judged on their own gives each readout one
    value per row. A design whose rows showed alternatives together gives
    each readout one value per alternative, the alternatives of a row
    adjacent and in presentation order.
    """

    __slots__ = (
        "_alternatives",
        "_alternatives_per_row",
        "_factor_levels",
        "_factors",
        "_item_levels",
        "_items",
        "_participant_levels",
        "_participants",
        "_readouts",
        "_response_levels",
        "_responses",
    )

    def __new__(cls) -> Never:
        """Prevent construction outside DesignCodec."""
        msg = "a judgment design is constructed by DesignCodec"

        raise TypeError(msg)

    def __init__(self, data: _DesignData) -> None:
        object.__setattr__(self, "_participants", data.participants)
        object.__setattr__(self, "_participant_levels", data.participant_levels)
        object.__setattr__(self, "_items", data.items)
        object.__setattr__(self, "_item_levels", data.item_levels)
        object.__setattr__(self, "_factors", data.factors)
        object.__setattr__(self, "_factor_levels", data.factor_levels)
        object.__setattr__(self, "_responses", data.responses)
        object.__setattr__(self, "_response_levels", data.response_levels)
        object.__setattr__(self, "_readouts", data.readouts)
        object.__setattr__(self, "_alternatives", data.alternatives)
        object.__setattr__(self, "_alternatives_per_row", data.alternatives_per_row)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a judgment design is immutable"

        raise AttributeError(msg)

    @property
    def participants(self) -> Tensor:
        """Return participant indices by row."""
        return self._participants

    @property
    def participant_levels(self) -> tuple[str, ...]:
        """Return participant identifiers by index."""
        return self._participant_levels

    @property
    def items(self) -> Tensor:
        """Return item indices by row."""
        return self._items

    @property
    def item_levels(self) -> tuple[str, ...]:
        """Return item identifiers by index."""
        return self._item_levels

    @property
    def factors(self) -> Mapping[str, Tensor]:
        """Return factor index columns."""
        return self._factors

    @property
    def factor_levels(self) -> Mapping[str, tuple[str, ...]]:
        """Return factor levels by column and index."""
        return self._factor_levels

    @property
    def responses(self) -> Tensor:
        """Return response indices by row."""
        return self._responses

    @property
    def unit_responses(self) -> None:
        """Return no continuous unit interval response column."""
        return None

    @property
    def response_levels(self) -> int:
        """Return the declared response level count."""
        return self._response_levels

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return readout columns."""
        return self._readouts

    @property
    def alternatives(self) -> tuple[tuple[str, ...] | None, ...]:
        """Return forced-choice alternatives by row."""
        return self._alternatives

    @property
    def alternatives_per_row(self) -> int | None:
        """Return how many alternatives every row shows, or None if rows show none."""
        return self._alternatives_per_row

    @property
    def n_rows(self) -> int:
        """Return the number of judgment rows."""
        return self._responses.numel()

    @property
    def n_readout_rows(self) -> int:
        """Return the length of every readout column."""
        shown = self._alternatives_per_row

        return self.n_rows if shown is None else self.n_rows * shown

    def select_rows(self, positions: Sequence[int], /) -> JudgmentDesign:
        """Return selected rows while retaining every global level table.

        Parameters
        ----------
        positions : Sequence[int]
            Distinct row positions in the order the result should retain them.

        Raises
        ------
        TypeError
            If a position is not an integer.
        IndexError
            If a position lies outside this design.
        ValueError
            If no position is given or one repeats.
        """
        held = tuple(positions)

        if not held:
            msg = "a selected judgment design needs at least one row"

            raise ValueError(msg)

        for position in held:
            if isinstance(position, bool) or not isinstance(position, int):
                msg = f"a judgment row position is not an integer: {position!r}"

                raise TypeError(msg)

            if not 0 <= position < self.n_rows:
                msg = f"there is no row {position} in a design of {self.n_rows} rows"

                raise IndexError(msg)

        if len(frozenset(held)) != len(held):
            msg = f"selected judgment row positions repeat: {held!r}"

            raise ValueError(msg)

        indices = self._responses.new_tensor(held, dtype=self._responses.dtype)
        shown = self._alternatives_per_row

        if shown is None:
            readout_indices = indices
        else:
            readout_indices = self._responses.new_tensor(
                [
                    position * shown + offset
                    for position in held
                    for offset in range(shown)
                ],
                dtype=self._responses.dtype,
            )

        data = _DesignData(
            participants=self._participants.index_select(0, indices),
            participant_levels=self._participant_levels,
            items=self._items.index_select(0, indices),
            item_levels=self._item_levels,
            factors=MappingProxyType(
                {
                    name: values.index_select(0, indices)
                    for name, values in self._factors.items()
                }
            ),
            factor_levels=self._factor_levels,
            responses=self._responses.index_select(0, indices),
            response_levels=self._response_levels,
            readouts=MappingProxyType(
                {
                    name: values.index_select(0, readout_indices)
                    for name, values in self._readouts.items()
                }
            ),
            alternatives=tuple(self._alternatives[position] for position in held),
            alternatives_per_row=shown,
        )
        selected = object.__new__(JudgmentDesign)
        selected.__init__(data)

        return selected

    @override
    def __repr__(self) -> str:
        """Return the design dimensions."""
        return (
            f"JudgmentDesign(rows={self.n_rows}, "
            f"participants={len(self._participant_levels)}, "
            f"items={len(self._item_levels)}, levels={self._response_levels})"
        )


class _BoundedDesignData(NamedTuple):
    """Stores validated bounded-design fields for construction."""

    participants: Tensor
    participant_levels: tuple[str, ...]
    items: Tensor
    item_levels: tuple[str, ...]
    factors: Mapping[str, Tensor]
    factor_levels: Mapping[str, tuple[str, ...]]
    responses: Tensor
    unit_responses: Tensor
    scales: tuple[ClosedScale, ...]
    readouts: Mapping[str, Tensor]


@final
class BoundedDesign:
    """Compact columns of floating responses on declared closed scales."""

    __slots__ = (
        "_factor_levels",
        "_factors",
        "_item_levels",
        "_items",
        "_participant_levels",
        "_participants",
        "_readouts",
        "_responses",
        "_scales",
        "_unit_responses",
    )

    def __new__(cls) -> Never:
        """Prevent construction outside BoundedDesignCodec."""
        msg = "a bounded design is constructed by BoundedDesignCodec"

        raise TypeError(msg)

    def __init__(self, data: _BoundedDesignData) -> None:
        object.__setattr__(self, "_participants", data.participants)
        object.__setattr__(self, "_participant_levels", data.participant_levels)
        object.__setattr__(self, "_items", data.items)
        object.__setattr__(self, "_item_levels", data.item_levels)
        object.__setattr__(self, "_factors", data.factors)
        object.__setattr__(self, "_factor_levels", data.factor_levels)
        object.__setattr__(self, "_responses", data.responses)
        object.__setattr__(self, "_unit_responses", data.unit_responses)
        object.__setattr__(self, "_scales", data.scales)
        object.__setattr__(self, "_readouts", data.readouts)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a bounded design is immutable"

        raise AttributeError(msg)

    @property
    def participants(self) -> Tensor:
        """Return participant indices by row."""
        return self._participants

    @property
    def participant_levels(self) -> tuple[str, ...]:
        """Return participant identifiers by index."""
        return self._participant_levels

    @property
    def items(self) -> Tensor:
        """Return item indices by row."""
        return self._items

    @property
    def item_levels(self) -> tuple[str, ...]:
        """Return item identifiers by index."""
        return self._item_levels

    @property
    def factors(self) -> Mapping[str, Tensor]:
        """Return factor index columns."""
        return self._factors

    @property
    def factor_levels(self) -> Mapping[str, tuple[str, ...]]:
        """Return factor levels by column and index."""
        return self._factor_levels

    @property
    def responses(self) -> Tensor:
        """Return floating responses on their original scales."""
        return self._responses

    @property
    def unit_responses(self) -> Tensor:
        """Return responses rescaled to the closed unit interval."""
        return self._unit_responses

    @property
    def scales(self) -> tuple[ClosedScale, ...]:
        """Return each response's original closed scale."""
        return self._scales

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return readout columns."""
        return self._readouts

    @property
    def response_levels(self) -> None:
        """Return no ordinal level count for continuous responses."""
        return None

    @property
    def alternatives(self) -> tuple[None, ...]:
        """Return that no row presents alternatives."""
        return (None,) * self.n_rows

    @property
    def alternatives_per_row(self) -> None:
        """Return that rows present no alternatives."""
        return None

    @property
    def n_rows(self) -> int:
        """Return the number of bounded response rows."""
        return self._responses.numel()

    @property
    def n_readout_rows(self) -> int:
        """Return the length of every readout column."""
        return self.n_rows

    @override
    def __repr__(self) -> str:
        """Return the design dimensions."""
        return (
            f"BoundedDesign(rows={self.n_rows}, "
            f"participants={len(self._participant_levels)}, "
            f"items={len(self._item_levels)})"
        )


type ResponseDesign = JudgmentDesign | BoundedDesign
"""A dense ordinal or bounded-continuous response design."""


def _compact(values: Iterable[str]) -> tuple[Tensor, tuple[str, ...]]:
    """Return dense indices under a canonical lexical level order."""
    held = tuple(values)
    levels = tuple(sorted(frozenset(held)))
    indices = {value: index for index, value in enumerate(levels)}
    encoded = [indices[value] for value in held]

    runtime = _torch()

    return runtime.tensor(encoded, dtype=runtime.int64), levels


def _ensure_index_column(
    name: str,
    values: Tensor,
    levels: int,
    rows: int,
) -> None:
    """Validate one dense index column."""
    if values.ndim != 1 or values.numel() != rows:
        msg = f"I-J1: {name} has shape {tuple(values.shape)}, expected ({rows},)"

        raise DesignError(msg)

    runtime = _torch()

    if values.dtype is not runtime.int64:
        msg = f"I-J1: {name} has dtype {values.dtype}, expected torch.int64"

        raise DesignError(msg)

    observed = runtime.unique(values)

    if observed.numel() != levels or not runtime.equal(
        observed, runtime.arange(levels, dtype=runtime.int64)
    ):
        msg = f"I-J1: {name} indices are not dense over 0..{levels - 1}"

        raise DesignError(msg)


def _ensure_responses(values: Tensor, levels: int, rows: int) -> None:
    """Validate the response column."""
    if values.ndim != 1 or values.numel() != rows:
        msg = f"I-J2: responses have shape {tuple(values.shape)}, expected ({rows},)"

        raise DesignError(msg)

    runtime = _torch()

    if values.dtype is not runtime.int64:
        msg = f"I-J2: responses have dtype {values.dtype}, expected torch.int64"

        raise DesignError(msg)

    if rows and (int(values.min()) < 0 or int(values.max()) >= levels):
        msg = f"I-J2: responses must fall in 0..{levels - 1}: {values.tolist()!r}"

        raise DesignError(msg)


def _ensure_readout(name: str, values: Tensor, rows: int) -> None:
    """Validate one readout column."""
    if values.ndim != 1 or values.numel() != rows:
        msg = f"readout {name!r} has shape {tuple(values.shape)}, expected ({rows},)"

        raise DesignError(msg)

    if not bool(_torch().isfinite(values).all()):
        msg = f"readout {name!r} contains a nonfinite value"

        raise DesignError(msg)


def _validate(data: _DesignData, expected_rows: int) -> None:
    """Validate design invariants at the codec boundary."""
    rows = data.responses.numel()

    if rows != expected_rows or len(data.alternatives) != expected_rows:
        msg = (
            f"I-J3: {expected_rows} judgments became {rows} response rows and "
            f"{len(data.alternatives)} alternative rows"
        )

        raise DesignError(msg)

    shown = data.alternatives_per_row

    for position, alternatives in enumerate(data.alternatives):
        held = 0 if alternatives is None else len(alternatives)

        if shown is not None and held != shown:
            msg = (
                f"row {position} shows {held} alternatives and the design shows {shown}"
            )

            raise DesignError(msg)

    _ensure_index_column(
        "participants", data.participants, len(data.participant_levels), rows
    )
    _ensure_index_column("items", data.items, len(data.item_levels), rows)
    _ensure_responses(data.responses, data.response_levels, rows)

    if data.factors.keys() != data.factor_levels.keys():
        msg = "I-J1: factor columns and factor level tables disagree"

        raise DesignError(msg)

    for name, values in data.factors.items():
        _ensure_index_column(
            f"factor {name!r}", values, len(data.factor_levels[name]), rows
        )

    readout_rows = rows if shown is None else rows * shown

    for name, values in data.readouts.items():
        _ensure_readout(name, values, readout_rows)


@final
class DesignCodec:
    """Converts judgment rows to dense tensor columns.

    The codec is the only code holding both the record form of a judgment
    and its interior form. Declaring an alternative count makes it the
    codec of a forced-choice design: every row then shows that many
    alternatives and gives each readout one value per alternative.

    Parameters
    ----------
    response_levels : int
        Number of ordered levels every response falls in.
    factors : Sequence[str]
        Declared factors, in column order.
    readouts : Sequence[str]
        Declared readouts, in column order.
    alternatives : int | None
        Number of alternatives every row shows, or None for rows judged on
        their own.

    Raises
    ------
    DesignError
        If fewer than two response levels or fewer than two alternatives
        are declared, or a factor or readout repeats.
    """

    __slots__ = ("_alternatives", "_factors", "_readouts", "_response_levels")

    def __init__(
        self,
        response_levels: int,
        factors: Sequence[str] = (),
        readouts: Sequence[str] = (),
        alternatives: int | None = None,
    ) -> None:
        if alternatives is not None and (
            isinstance(alternatives, bool)
            or not isinstance(alternatives, int)
            or alternatives < _MINIMUM_ALTERNATIVES
        ):
            msg = (
                f"a forced-choice design codec needs at least "
                f"{_MINIMUM_ALTERNATIVES} alternatives: {alternatives!r}"
            )

            raise DesignError(msg)

        if (
            isinstance(response_levels, bool)
            or not isinstance(response_levels, int)
            or response_levels < _MINIMUM_LEVELS
        ):
            msg = (
                f"a design codec needs at least {_MINIMUM_LEVELS} response levels: "
                f"{response_levels!r}"
            )

            raise DesignError(msg)

        checked_factors = tuple(ensure_name("a factor name", name) for name in factors)
        checked_readouts = tuple(
            ensure_name("a readout name", name) for name in readouts
        )

        if len(frozenset(checked_factors)) != len(checked_factors):
            msg = f"a design codec repeats a factor: {checked_factors!r}"

            raise DesignError(msg)

        if len(frozenset(checked_readouts)) != len(checked_readouts):
            msg = f"a design codec repeats a readout: {checked_readouts!r}"

            raise DesignError(msg)

        self._response_levels = response_levels
        self._factors = checked_factors
        self._readouts = checked_readouts
        self._alternatives = alternatives

    @property
    def alternatives(self) -> int | None:
        """Return how many alternatives every row shows, or None if rows show none."""
        return self._alternatives

    def _readout_values(self, position: int, row: JudgmentRow) -> tuple[float, ...]:
        """Return one row's readout values in column order, then in presentation order.

        Parameters
        ----------
        position : int
            The row's place among the rows encoded, named in error messages.
        row : JudgmentRow
            Row to read.

        Raises
        ------
        DesignError
            If the row shows other alternatives than the codec declares, or
            declares readouts on the side the codec does not read.
        """
        shown = 0 if row.alternatives is None else len(row.alternatives)

        if self._alternatives is not None and shown != self._alternatives:
            msg = (
                f"row {position} shows {shown} alternatives and the design shows "
                f"{self._alternatives}"
            )

            raise DesignError(msg)

        held = row.alternative_readouts if self._alternatives else row.readouts
        aside = row.readouts if self._alternatives else row.alternative_readouts

        if frozenset(held) != frozenset(self._readouts) or aside:
            shown = "per alternative" if self._alternatives else "per row"
            msg = (
                f"row {position} has the readouts {sorted(held)} {shown} and "
                f"{sorted(aside)} on the other side, expected "
                f"{sorted(self._readouts)} {shown} alone"
            )

            raise DesignError(msg)

        if self._alternatives is None:
            return tuple(row.readouts[name] for name in self._readouts)

        return tuple(
            value for name in self._readouts for value in row.alternative_readouts[name]
        )

    def encode(self, rows: Iterable[JudgmentRow], /) -> JudgmentDesign:
        """Encode boundary rows as a validated design.

        Parameters
        ----------
        rows : Iterable[JudgmentRow]
            Rows to encode, in row order.

        Raises
        ------
        DesignError
            If a row declares other factors or readouts than the codec, or
            the encoded columns break a design invariant.
        """
        records = tuple(rows)
        expected_factors = frozenset(self._factors)
        values = tuple(
            self._readout_values(position, row) for position, row in enumerate(records)
        )

        for position, row in enumerate(records):
            if frozenset(row.factors) != expected_factors:
                msg = (
                    f"row {position} has factors {sorted(row.factors)}, expected "
                    f"{sorted(expected_factors)}"
                )

                raise DesignError(msg)

        participants, participant_levels = _compact(row.participant for row in records)
        items, item_levels = _compact(row.item for row in records)
        factors: dict[str, Tensor] = {}
        factor_levels: dict[str, tuple[str, ...]] = {}

        for name in self._factors:
            column, levels = _compact(row.factors[name] for row in records)
            factors[name] = column
            factor_levels[name] = levels

        runtime = _torch()
        width = 1 if self._alternatives is None else self._alternatives
        data = _DesignData(
            participants=participants,
            participant_levels=participant_levels,
            items=items,
            item_levels=item_levels,
            factors=MappingProxyType(factors),
            factor_levels=MappingProxyType(factor_levels),
            responses=runtime.tensor(
                [row.response for row in records], dtype=runtime.int64
            ),
            response_levels=self._response_levels,
            readouts=MappingProxyType(
                {
                    name: runtime.tensor(
                        [
                            row[at * width + offset]
                            for row in values
                            for offset in range(width)
                        ],
                        dtype=runtime.float64,
                    )
                    for at, name in enumerate(self._readouts)
                }
            ),
            alternatives=tuple(row.alternatives for row in records),
            alternatives_per_row=self._alternatives,
        )
        _validate(data, len(records))

        design = object.__new__(JudgmentDesign)
        design.__init__(data)

        return design

    def decode(self, design: JudgmentDesign, /) -> tuple[JudgmentRow, ...]:
        """Decode a validated design as boundary rows.

        Parameters
        ----------
        design : JudgmentDesign
            Design to decode.

        Raises
        ------
        DesignError
            If the design was encoded by a codec declaring other levels,
            factors, readouts or alternatives than this one.
        """
        if design.alternatives_per_row != self._alternatives:
            msg = (
                f"a design shows {design.alternatives_per_row!r} alternatives per "
                f"row and its codec shows {self._alternatives!r}"
            )

            raise DesignError(msg)

        if design.response_levels != self._response_levels:
            msg = (
                f"a design has {design.response_levels} response levels and its "
                f"codec has {self._response_levels}"
            )

            raise DesignError(msg)

        if tuple(design.factors) != self._factors:
            msg = (
                f"a design has factors {tuple(design.factors)!r} and its codec "
                f"has {self._factors!r}"
            )

            raise DesignError(msg)

        if tuple(design.readouts) != self._readouts:
            msg = (
                f"a design has readouts {tuple(design.readouts)!r} and its codec "
                f"has {self._readouts!r}"
            )

            raise DesignError(msg)

        data = _DesignData(
            design.participants,
            design.participant_levels,
            design.items,
            design.item_levels,
            design.factors,
            design.factor_levels,
            design.responses,
            design.response_levels,
            design.readouts,
            design.alternatives,
            design.alternatives_per_row,
        )
        _validate(data, design.n_rows)
        participants = design.participants.tolist()
        items = design.items.tolist()
        responses = design.responses.tolist()
        factor_columns = {
            name: values.tolist() for name, values in design.factors.items()
        }
        readout_columns = {
            name: values.tolist() for name, values in design.readouts.items()
        }

        width = 1 if self._alternatives is None else self._alternatives

        return tuple(
            JudgmentRow(
                participant=design.participant_levels[participants[position]],
                item=design.item_levels[items[position]],
                response=responses[position],
                columns=RowColumns(
                    factors={
                        name: design.factor_levels[name][column[position]]
                        for name, column in factor_columns.items()
                    },
                    readouts={}
                    if self._alternatives
                    else {
                        name: column[position]
                        for name, column in readout_columns.items()
                    },
                    alternative_readouts=None
                    if self._alternatives is None
                    else {
                        name: tuple(
                            column[position * width + offset] for offset in range(width)
                        )
                        for name, column in readout_columns.items()
                    },
                ),
                alternatives=design.alternatives[position],
            )
            for position in range(design.n_rows)
        )


def _validate_bounded(data: _BoundedDesignData, expected_rows: int) -> None:
    """Validate bounded-design invariants at the codec boundary."""
    rows = data.responses.numel()

    if rows != expected_rows or data.unit_responses.numel() != expected_rows:
        msg = (
            f"{expected_rows} bounded judgments became {rows} response rows and "
            f"{data.unit_responses.numel()} normalized rows"
        )

        raise DesignError(msg)

    if len(data.scales) != expected_rows:
        msg = (
            f"{expected_rows} bounded judgments became {len(data.scales)} "
            "response scales"
        )

        raise DesignError(msg)

    _ensure_index_column(
        "participants", data.participants, len(data.participant_levels), rows
    )
    _ensure_index_column("items", data.items, len(data.item_levels), rows)

    runtime = _torch()

    for name, values in (
        ("responses", data.responses),
        ("normalized responses", data.unit_responses),
    ):
        if values.ndim != 1 or values.numel() != rows:
            msg = f"{name} have shape {tuple(values.shape)}, expected ({rows},)"

            raise DesignError(msg)

        if values.dtype is not runtime.float64:
            msg = f"{name} have dtype {values.dtype}, expected torch.float64"

            raise DesignError(msg)

        if not bool(runtime.isfinite(values).all()):
            msg = f"{name} contain a nonfinite value"

            raise DesignError(msg)

    if rows and bool(((data.unit_responses < 0.0) | (data.unit_responses > 1.0)).any()):
        msg = "normalized bounded responses fall outside the closed unit interval"

        raise DesignError(msg)

    if data.factors.keys() != data.factor_levels.keys():
        msg = "factor columns and factor level tables disagree"

        raise DesignError(msg)

    for name, values in data.factors.items():
        _ensure_index_column(
            f"factor {name!r}", values, len(data.factor_levels[name]), rows
        )

    for name, values in data.readouts.items():
        _ensure_readout(name, values, rows)


@final
class BoundedDesignCodec:
    """Converts floating closed-scale rows to dense tensor columns.

    Parameters
    ----------
    factors : Sequence[str]
        Declared factors, in column order.
    readouts : Sequence[str]
        Declared readouts, in column order.

    Raises
    ------
    DesignError
        If a factor or readout repeats.
    """

    __slots__ = ("_factors", "_readouts")

    def __init__(
        self,
        factors: Sequence[str] = (),
        readouts: Sequence[str] = (),
    ) -> None:
        checked_factors = tuple(ensure_name("a factor name", name) for name in factors)
        checked_readouts = tuple(
            ensure_name("a readout name", name) for name in readouts
        )

        if len(frozenset(checked_factors)) != len(checked_factors):
            msg = f"a bounded design codec repeats a factor: {checked_factors!r}"

            raise DesignError(msg)

        if len(frozenset(checked_readouts)) != len(checked_readouts):
            msg = f"a bounded design codec repeats a readout: {checked_readouts!r}"

            raise DesignError(msg)

        self._factors = checked_factors
        self._readouts = checked_readouts

    def encode(self, rows: Iterable[BoundedJudgmentRow], /) -> BoundedDesign:
        """Encode boundary rows as a validated bounded design.

        Parameters
        ----------
        rows : Iterable[BoundedJudgmentRow]
            Rows to encode, in row order.

        Raises
        ------
        DesignError
            If a row declares other factors or readouts than the codec.
        """
        records = tuple(rows)
        expected_factors = frozenset(self._factors)
        expected_readouts = frozenset(self._readouts)

        for position, row in enumerate(records):
            if frozenset(row.factors) != expected_factors:
                msg = (
                    f"row {position} has factors {sorted(row.factors)}, expected "
                    f"{sorted(expected_factors)}"
                )

                raise DesignError(msg)

            if frozenset(row.readouts) != expected_readouts:
                msg = (
                    f"row {position} has readouts {sorted(row.readouts)}, expected "
                    f"{sorted(expected_readouts)}"
                )

                raise DesignError(msg)

        participants, participant_levels = _compact(row.participant for row in records)
        items, item_levels = _compact(row.item for row in records)
        factors: dict[str, Tensor] = {}
        factor_levels: dict[str, tuple[str, ...]] = {}

        for name in self._factors:
            column, levels = _compact(row.factors[name] for row in records)
            factors[name] = column
            factor_levels[name] = levels

        runtime = _torch()
        data = _BoundedDesignData(
            participants=participants,
            participant_levels=participant_levels,
            items=items,
            item_levels=item_levels,
            factors=MappingProxyType(factors),
            factor_levels=MappingProxyType(factor_levels),
            responses=runtime.tensor(
                [row.response for row in records], dtype=runtime.float64
            ),
            unit_responses=runtime.tensor(
                [row.scale.normalize(row.response) for row in records],
                dtype=runtime.float64,
            ),
            scales=tuple(row.scale for row in records),
            readouts=MappingProxyType(
                {
                    name: runtime.tensor(
                        [row.readouts[name] for row in records],
                        dtype=runtime.float64,
                    )
                    for name in self._readouts
                }
            ),
        )
        _validate_bounded(data, len(records))
        design = object.__new__(BoundedDesign)
        design.__init__(data)

        return design

    def decode(self, design: BoundedDesign, /) -> tuple[BoundedJudgmentRow, ...]:
        """Decode a validated bounded design as boundary rows.

        Parameters
        ----------
        design : BoundedDesign
            Design to decode.

        Raises
        ------
        DesignError
            If the design declares other factors or readouts than this codec.
        """
        if tuple(design.factors) != self._factors:
            msg = (
                f"a bounded design has factors {tuple(design.factors)!r} and its "
                f"codec has {self._factors!r}"
            )

            raise DesignError(msg)

        if tuple(design.readouts) != self._readouts:
            msg = (
                f"a bounded design has readouts {tuple(design.readouts)!r} and its "
                f"codec has {self._readouts!r}"
            )

            raise DesignError(msg)

        data = _BoundedDesignData(
            design.participants,
            design.participant_levels,
            design.items,
            design.item_levels,
            design.factors,
            design.factor_levels,
            design.responses,
            design.unit_responses,
            design.scales,
            design.readouts,
        )
        _validate_bounded(data, design.n_rows)
        participants = design.participants.tolist()
        items = design.items.tolist()
        responses = design.responses.tolist()
        factor_columns = {
            name: values.tolist() for name, values in design.factors.items()
        }
        readout_columns = {
            name: values.tolist() for name, values in design.readouts.items()
        }

        return tuple(
            BoundedJudgmentRow(
                participant=design.participant_levels[participants[position]],
                item=design.item_levels[items[position]],
                response=responses[position],
                scale=design.scales[position],
                columns=RowColumns(
                    factors={
                        name: design.factor_levels[name][column[position]]
                        for name, column in factor_columns.items()
                    },
                    readouts={
                        name: column[position]
                        for name, column in readout_columns.items()
                    },
                ),
            )
            for position in range(design.n_rows)
        )


def design_of(source: JudgmentSource, /) -> JudgmentDesign:
    """Return the validated design of a judgment source.

    The codec takes its response levels and its factors from the source,
    and the source's rows have no readout columns.

    Parameters
    ----------
    source : JudgmentSource
        Judgments to encode.

    Raises
    ------
    DesignError
        If the source's rows do not form a valid design.
    """
    return DesignCodec(source.response_levels, source.factors).encode(source.rows())
