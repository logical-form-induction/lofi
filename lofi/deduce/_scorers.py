"""Split posteriors and local-attention edge scoring."""

from math import exp, fsum, isclose, isfinite, isnan
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._native import SpanDependent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.deduce._outside import Outside
    from lofi.deduce._ports import Chart, EdgeScorer
    from lofi.deduce._spans import Cell
    from lofi.judge import Rule
    from lofi.lexicon import Entry

__all__ = [
    "LocalAttendScorer",
    "MarginalScorer",
    "SplitLogWeights",
    "SplitProbabilities",
]

_BINARY_ARITY = 2
"""Binary edge arity and minimum split-bearing span width."""

type _SplitRow = tuple[tuple[int, float], ...]
"""One split row in canonical order."""

type _SplitFields = tuple[tuple[Cell, _SplitRow], ...]
"""Split rows in reconstruction order."""

type _SplitReduction[V] = tuple[
    Callable[[_SplitFields, float], V], tuple[_SplitFields, float]
]
"""Split-value pickle reduction."""

type _LocalFields[T] = tuple[EdgeScorer[T, float], SplitLogWeights]
"""Local-attention scorer fields."""

type _LocalReduction[T] = tuple[
    Callable[[_LocalFields[T]], LocalAttendScorer[T]],
    tuple[_LocalFields[T]],
]
"""Local-attention scorer pickle reduction."""


def _checked_span(span: Cell) -> Cell:
    """Return a checked binary span."""
    start, end = span
    checked_start = ensure_non_negative("a split row's start", start)
    checked_end = ensure_non_negative("a split row's end", end)

    if checked_end - checked_start < _BINARY_ARITY:
        msg = f"a split row's span is not binary: {span!r}"
        raise ValueError(msg)

    return (checked_start, checked_end)


def _checked_rows(
    rows: Mapping[Cell, Mapping[int, float]],
    *,
    log_space: bool,
    tolerance: float,
) -> _SplitFields:
    """Return checked split rows in canonical order."""
    checked: list[tuple[Cell, _SplitRow]] = []

    for supplied_span, supplied_row in rows.items():
        span = _checked_span(supplied_span)
        start, end = span
        row: list[tuple[int, float]] = []

        for supplied_split, supplied_weight in supplied_row.items():
            split = ensure_non_negative("a split point", supplied_split)

            if not start < split < end:
                msg = f"split {split} is not internal to span {span!r}"
                raise ValueError(msg)

            if log_space:
                if isinstance(supplied_weight, bool) or not isinstance(
                    supplied_weight, int | float
                ):
                    msg = f"a split log weight is not a number: {supplied_weight!r}"
                    raise TypeError(msg)

                weight = float(supplied_weight)

                if weight > 0.0 or isnan(weight):
                    msg = f"a split log weight is invalid: {supplied_weight!r}"
                    raise ValueError(msg)

            else:
                weight = ensure_finite("a split probability", supplied_weight)

                if not 0.0 <= weight <= 1.0:
                    msg = f"a split probability is outside [0, 1]: {weight!r}"
                    raise ValueError(msg)

            row.append((split, weight))

        if not row:
            msg = f"split row {span!r} is empty"
            raise ValueError(msg)

        row.sort(key=lambda pair: pair[0])
        total = fsum(
            exp(weight)
            if log_space and isfinite(weight)
            else 0.0
            if log_space
            else weight
            for _, weight in row
        )

        if not isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            scale = "log weights" if log_space else "probabilities"
            msg = f"split {scale} for span {span!r} sum to {total!r}, not 1.0"
            raise ValueError(msg)

        checked.append((span, tuple(row)))

    checked.sort(key=lambda pair: pair[0])

    return tuple(checked)


def _tolerance(tolerance: float) -> float:
    """Return a checked absolute normalization tolerance."""
    checked = ensure_finite("a split normalization tolerance", tolerance)

    if checked < 0.0:
        msg = f"a split normalization tolerance is negative: {tolerance!r}"
        raise ValueError(msg)

    return checked


def _mapping(fields: _SplitFields) -> Mapping[Cell, Mapping[int, float]]:
    """Build nested read-only mappings from canonical rows."""
    return MappingProxyType({span: MappingProxyType(dict(row)) for span, row in fields})


@final
class SplitProbabilities:
    """Immutable normalized split probabilities.

    Parameters
    ----------
    rows : Mapping[Cell, Mapping[int, float]]
        Split probabilities by binary span and internal split point.
    tolerance : float
        Absolute distance from one that each row's total may take.
    """

    __slots__ = ("_fields", "_rows", "_tolerance")

    def __init__(
        self,
        rows: Mapping[Cell, Mapping[int, float]],
        *,
        tolerance: float = 1e-9,
    ) -> None:
        held = _tolerance(tolerance)
        fields = _checked_rows(rows, log_space=False, tolerance=held)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_rows", _mapping(fields))
        object.__setattr__(self, "_tolerance", held)

    @property
    def tolerance(self) -> float:
        """Return the absolute tolerance each row's total was checked against."""
        return self._tolerance

    @property
    def rows(self) -> Mapping[Cell, Mapping[int, float]]:
        """Return the normalized rows by span."""
        return self._rows

    @property
    def spans(self) -> tuple[Cell, ...]:
        """Return declared spans in canonical order."""
        return tuple(span for span, _ in self._fields)

    def at(self, span: Cell, /) -> Mapping[int, float]:
        """Return one span's split probabilities."""
        return self._rows[span]

    def probability(self, span: Cell, split: int, /) -> float:
        """Return one split probability."""
        return self._rows[span][split]

    def __copy__(self) -> SplitProbabilities:
        """Reconstruct the probabilities from frozen rows."""
        return _restore_probabilities(self._fields, self._tolerance)

    def __deepcopy__(self, memo: dict[int, SplitProbabilities]) -> SplitProbabilities:
        """Reconstruct the probabilities from frozen rows."""
        del memo

        return _restore_probabilities(self._fields, self._tolerance)

    @override
    def __reduce__(self) -> _SplitReduction[SplitProbabilities]:
        """Return the probability reconstruction call."""
        return (_restore_probabilities, (self._fields, self._tolerance))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "split probabilities are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "split probabilities are immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare canonical split rows."""
        if type(other) is not SplitProbabilities:
            return NotImplemented

        return self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Hash canonical split rows."""
        return hash((SplitProbabilities, self._fields))

    @override
    def __repr__(self) -> str:
        """Return a rendering with the declared row count."""
        return f"SplitProbabilities(rows={len(self._fields)})"


@final
class SplitLogWeights:
    """Immutable normalized split weights in log space.

    Parameters
    ----------
    rows : Mapping[Cell, Mapping[int, float]]
        Split exponentiated log weights by binary span and internal split point.
    tolerance : float
        Absolute distance from one that each row's total may take.
    """

    __slots__ = ("_fields", "_rows", "_tolerance")

    def __init__(
        self,
        rows: Mapping[Cell, Mapping[int, float]],
        *,
        tolerance: float = 1e-9,
    ) -> None:
        held = _tolerance(tolerance)
        fields = _checked_rows(rows, log_space=True, tolerance=held)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_rows", _mapping(fields))
        object.__setattr__(self, "_tolerance", held)

    @property
    def tolerance(self) -> float:
        """Return the absolute tolerance each row's total was checked against."""
        return self._tolerance

    @property
    def rows(self) -> Mapping[Cell, Mapping[int, float]]:
        """Return the normalized log-weight rows by span."""
        return self._rows

    @property
    def spans(self) -> tuple[Cell, ...]:
        """Return declared spans in canonical order."""
        return tuple(span for span, _ in self._fields)

    def at(self, span: Cell, /) -> Mapping[int, float]:
        """Return one span's split log weights."""
        return self._rows[span]

    def weight(self, span: Cell, split: int, /) -> float:
        """Return one split log weight."""
        return self._rows[span][split]

    def __copy__(self) -> SplitLogWeights:
        """Reconstruct the log weights from frozen rows."""
        return _restore_log_weights(self._fields, self._tolerance)

    def __deepcopy__(self, memo: dict[int, SplitLogWeights]) -> SplitLogWeights:
        """Reconstruct the log weights from frozen rows."""
        del memo

        return _restore_log_weights(self._fields, self._tolerance)

    @override
    def __reduce__(self) -> _SplitReduction[SplitLogWeights]:
        """Return the log-weight reconstruction call."""
        return (_restore_log_weights, (self._fields, self._tolerance))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "split log weights are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "split log weights are immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare canonical split rows."""
        if type(other) is not SplitLogWeights:
            return NotImplemented

        return self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Hash canonical split rows."""
        return hash((SplitLogWeights, self._fields))

    @override
    def __repr__(self) -> str:
        """Return a rendering with the declared row count."""
        return f"SplitLogWeights(rows={len(self._fields)})"


@final
class MarginalScorer[T]:
    """Binary split posteriors from a completed forest."""

    __slots__ = ("_probabilities",)

    def __init__(self, chart: Chart[T, float], outside: Outside[T]) -> None:
        if not outside.is_for(chart):
            msg = "the outside result was computed from a different chart"
            raise ValueError(msg)

        rows: dict[Cell, dict[int, float]] = {}

        for span in chart.index.spans_by_length(min_length=2):
            by_split: dict[int, list[float]] = {}

            for parent in chart.items_at(span):
                for edge, marginal in outside.rule_marginals(parent):
                    if edge.arity != _BINARY_ARITY:
                        continue

                    split = edge.split

                    if split is None or not span[0] < split < span[1]:
                        msg = (
                            f"binary edge {edge.rule_name!r} has invalid split "
                            f"{split!r}"
                        )
                        raise ValueError(msg)

                    by_split.setdefault(split, []).append(marginal)

            masses = {split: fsum(marginals) for split, marginals in by_split.items()}
            total = fsum(masses.values())

            if total > 0.0:
                rows[span] = {split: mass / total for split, mass in masses.items()}

        object.__setattr__(self, "_probabilities", SplitProbabilities(rows))

    @property
    def probabilities(self) -> SplitProbabilities:
        """Return binary split posteriors by chart span."""
        return self._probabilities

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a marginal scorer is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a marginal scorer is immutable"

        raise AttributeError(msg)


@final
class LocalAttendScorer[T]:
    """Edge scorer with supplied binary split log weights."""

    __slots__ = ("_base", "_split_weights")

    def __init__(
        self,
        base: EdgeScorer[T, float],
        split_weights: SplitLogWeights,
    ) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(
            self,
            "_split_weights",
            SplitLogWeights(split_weights.rows, tolerance=split_weights.tolerance),
        )

    @property
    def base(self) -> EdgeScorer[T, float]:
        """Return the wrapped edge scorer."""
        return self._base

    @property
    def split_weights(self) -> SplitLogWeights:
        """Return the local split log weights."""
        return self._split_weights

    @property
    def lowering(self) -> SpanDependent:
        """Declare that binary weights depend on their span and split."""
        return SpanDependent()

    def axiom_weight(self, position: int, entry: Entry[T], /) -> float:
        """Return the wrapped lexical-axiom weight."""
        return self._base.axiom_weight(position, entry)

    def edge_weight(
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the wrapped edge weight with binary local attention."""
        if len(inputs) != _BINARY_ARITY:
            return self._base.edge_weight(rule, inputs, output, span, split)

        if split is None or not span[0] < split < span[1]:
            msg = f"binary edge split {split!r} is not internal to span {span!r}"
            raise ValueError(msg)

        try:
            local = self._split_weights.weight(span, split)

        except KeyError:
            msg = f"no local-attention weight exists for split {split} of span {span!r}"
            raise ValueError(msg) from None

        return self._base.edge_weight(rule, inputs, output, span, split) + local

    def __copy__(self) -> LocalAttendScorer[T]:
        """Reconstruct the scorer with its shared base."""
        return _restore_local((self._base, self._split_weights))

    def __deepcopy__(
        self, memo: dict[int, LocalAttendScorer[T]]
    ) -> LocalAttendScorer[T]:
        """Reconstruct the scorer with its shared base."""
        del memo

        return _restore_local((self._base, self._split_weights))

    @override
    def __reduce__(self) -> _LocalReduction[T]:
        """Return the local-attention reconstruction call."""
        return (_restore_local, ((self._base, self._split_weights),))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a local-attention scorer is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a local-attention scorer is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the wrapped scorer and row count."""
        return (
            f"LocalAttendScorer(base={self._base!r}, "
            f"rows={len(self._split_weights.spans)})"
        )


def _restore_probabilities(
    fields: _SplitFields, tolerance: float
) -> SplitProbabilities:
    """Reconstruct split probabilities from canonical rows."""
    return SplitProbabilities(
        {span: dict(row) for span, row in fields}, tolerance=tolerance
    )


def _restore_log_weights(fields: _SplitFields, tolerance: float) -> SplitLogWeights:
    """Reconstruct split log weights from canonical rows."""
    return SplitLogWeights(
        {span: dict(row) for span, row in fields}, tolerance=tolerance
    )


def _restore_local[T](fields: _LocalFields[T]) -> LocalAttendScorer[T]:
    """Reconstruct a local-attention scorer from frozen fields."""
    base, split_weights = fields

    return LocalAttendScorer(base, split_weights)
