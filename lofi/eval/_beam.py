"""Accuracy of ranked type beams by width and depth.

Each measure ranks candidates by a selected length normalization of their log
weights, allowing several normalizations to share one decoded beam.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, Never, final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.terms import truncate, typeless

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.terms import Constructors, Term

__all__ = [
    "AccuracyCurve",
    "BeamNormalization",
    "ScoredBeam",
    "SweepPoint",
    "accuracy_at_beam_size",
    "beam_size_sweep",
    "depth_truncated_accuracy",
    "mean_reciprocal_rank",
    "term_depth",
    "typeless_accuracy",
]


type ScoredBeam = Sequence[tuple[Term, float]]
"""One item's ranked candidate types, each beside its log weight."""


def term_depth(term: Term) -> int:
    """Return the number of nodes on a term's longest root-to-leaf path.

    Parameters
    ----------
    term : Term
        The term to measure.
    """
    return 1 + max((term_depth(child) for child in term.args), default=0)


def _terminals(term: Term) -> int:
    """Return how many leaves a term has.

    Parameters
    ----------
    term : Term
        The term to measure.
    """
    if not term.args:
        return 1

    return sum(_terminals(child) for child in term.args)


class BeamNormalization(Enum):
    """Selects what a candidate's log weight is divided by before ranking."""

    #: The log weight as the decoder gave it.
    NONE = auto()

    #: The number of nodes on the candidate's longest root-to-leaf path.
    DEPTH = auto()

    #: The candidate's node count.
    NODES = auto()

    #: The candidate's leaf count.
    TERMINALS = auto()

    def divisor(self, term: Term, /) -> int:
        """Return what a candidate's log weight is divided by.

        Parameters
        ----------
        term : Term
            The candidate.
        """
        match self:
            case BeamNormalization.NONE:
                return 1
            case BeamNormalization.DEPTH:
                return term_depth(term)
            case BeamNormalization.NODES:
                return term.size
            case BeamNormalization.TERMINALS:
                return _terminals(term)

    def ranked(self, beam: ScoredBeam, /) -> tuple[Term, ...]:
        """Return a beam's candidates from the greatest normalized log weight down.

        Candidates at one normalized log weight keep the order the beam gave them.

        Parameters
        ----------
        beam : ScoredBeam
            The candidates, each beside its log weight.

        Raises
        ------
        TypeError
            If a log weight is not a number.
        ValueError
            If a log weight is infinite or not a number.
        """
        normalized = [
            ensure_finite("a candidate's log weight", weight) / self.divisor(term)
            for term, weight in beam
        ]
        order = sorted(range(len(normalized)), key=lambda at: -normalized[at])

        return tuple(beam[at][0] for at in order)


@final
class AccuracyCurve:
    """Accuracies at each level of a one-based index, over a number of items.

    Parameters
    ----------
    hits : Sequence[int]
        How many items count as correct at each level, from level one up.
    items : int
        How many items were scored.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or a level counts more hits than there are items.
    """

    __slots__ = ("_hits", "_items")

    def __init__(self, hits: Sequence[int], items: int) -> None:
        scored = ensure_non_negative("the items a curve is over", items)
        counted = tuple(ensure_non_negative("a hit count", hit) for hit in hits)

        for level, hit in enumerate(counted, start=1):
            if hit > scored:
                msg = f"level {level} counts {hit} hits among {scored} items"
                raise ValueError(msg)

        object.__setattr__(self, "_hits", counted)
        object.__setattr__(self, "_items", scored)

    @property
    def hits(self) -> tuple[int, ...]:
        """Return how many items count as correct at each level, from level one up."""
        return self._hits

    @property
    def items(self) -> int:
        """Return how many items were scored."""
        return self._items

    @property
    def rates(self) -> tuple[float, ...]:
        """Return the accuracy at each level, or nothing where no item was scored."""
        if self._items == 0:
            return ()

        return tuple(hit / self._items for hit in self._hits)

    def at(self, level: int) -> float | None:
        """Return the accuracy at one level, or None where no item was scored.

        Parameters
        ----------
        level : int
            The level, counted from one.

        Raises
        ------
        IndexError
            If the curve has no such level.
        """
        if not 1 <= level <= len(self._hits):
            msg = f"the curve runs over levels 1 to {len(self._hits)}, not {level}"
            raise IndexError(msg)

        if self._items == 0:
            return None

        return self._hits[level - 1] / self._items

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an accuracy curve is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an accuracy curve is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a curve with the same hits and items."""
        if type(other) is not AccuracyCurve:
            return NotImplemented

        return self._hits == other._hits and self._items == other._items

    @override
    def __hash__(self) -> int:
        """Return a hash over the hits and the items."""
        return hash((AccuracyCurve, self._hits, self._items))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this curve."""
        return f"AccuracyCurve({list(self._hits)!r}, {self._items!r})"


class SweepPoint(NamedTuple):
    """How often the gold type was in the beam decoded at one width.

    Attributes
    ----------
    width : int
        The width the beams were decoded at.
    hits : int
        How many items had their gold type somewhere in the beam.
    items : int
        How many items were scored.
    """

    width: int
    hits: int
    items: int

    @property
    def rate(self) -> float | None:
        """Return the accuracy at this width, or None where no item was scored."""
        if self.items == 0:
            return None

        return self.hits / self.items


def _paired(
    beams: Sequence[ScoredBeam],
    gold: Sequence[Term],
    normalization: BeamNormalization,
) -> tuple[tuple[tuple[Term, ...], Term], ...]:
    """Return each beam, ranked under the normalization, beside its gold type.

    Parameters
    ----------
    beams : Sequence[ScoredBeam]
        One beam per item.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    normalization : BeamNormalization
        What each candidate's log weight is divided by before ranking.

    Raises
    ------
    ValueError
        If there is not one beam per gold type.
    """
    if len(beams) != len(gold):
        msg = f"there are {len(beams)} beams and {len(gold)} gold types"
        raise ValueError(msg)

    return tuple(
        (normalization.ranked(beam), target)
        for beam, target in zip(beams, gold, strict=True)
    )


def _positive_width(max_width: int) -> None:
    """Reject a beam width that is not positive.

    Parameters
    ----------
    max_width : int
        The beam width to check.
    """
    if max_width <= 0:
        msg = f"a beam width is positive, not {max_width}"
        raise ValueError(msg)


def _hits_by_width(
    pairs: Sequence[tuple[Sequence[Term], Term]], max_width: int
) -> list[int]:
    """Return how many items have their gold type among the first entries, per width.

    Parameters
    ----------
    pairs : Sequence[tuple[Sequence[Term], Term]]
        Each ranked beam beside the type it is scored against.
    max_width : int
        The widest prefix to score.
    """
    hits = [0] * max_width

    for beam, target in pairs:
        wanted = target.key()

        for rank, candidate in enumerate(beam[:max_width]):
            if candidate.key() == wanted:
                for width in range(rank, max_width):
                    hits[width] += 1

                break

    return hits


def accuracy_at_beam_size(
    beams: Sequence[ScoredBeam],
    gold: Sequence[Term],
    *,
    normalization: BeamNormalization,
    max_width: int,
) -> AccuracyCurve:
    """Return how often the gold type sits within the first entries of a beam.

    The curve is nondecreasing: an item first counts at the smallest width whose
    prefix contains the gold structural key. Empty beams never count.

    Parameters
    ----------
    beams : Sequence[ScoredBeam]
        One beam per item.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    normalization : BeamNormalization
        What each candidate's log weight is divided by before ranking.
    max_width : int
        The widest beam prefix to score, which is the last level of the curve.

    Returns
    -------
    AccuracyCurve
        One level per width from one to ``max_width``.

    Raises
    ------
    ValueError
        If ``max_width`` is not positive, if there is not one beam per gold type, or
        if a log weight is not finite.
    """
    _positive_width(max_width)
    pairs = _paired(beams, gold, normalization)

    return AccuracyCurve(_hits_by_width(pairs, max_width), len(pairs))


def typeless_accuracy(
    beams: Sequence[ScoredBeam],
    gold: Sequence[Term],
    primitive: Term,
    constructors: Constructors[Term],
    *,
    normalization: BeamNormalization,
    max_width: int,
) -> AccuracyCurve:
    """Return how often a beam has the gold type's shape within its first entries.

    Candidates and targets are projected to constructor structure with one shared
    primitive label. The resulting curve is at least the exact-type curve; their
    difference counts label-only errors.

    Parameters
    ----------
    beams : Sequence[ScoredBeam]
        One beam per item.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    primitive : Term
        The nonconstructor leaf every primitive is replaced by.
    constructors : Constructors[Term]
        The declared constructor algebra the types are built over.
    normalization : BeamNormalization
        What each candidate's log weight is divided by before ranking.
    max_width : int
        The widest beam prefix to score, which is the last level of the curve.

    Returns
    -------
    AccuracyCurve
        One level per width from one to ``max_width``.

    Raises
    ------
    NeighborhoodError
        If a type violates the constructor algebra, or if the primitive is not a
        nonconstructor leaf.
    ValueError
        If ``max_width`` is not positive, if there is not one beam per gold type, or
        if a log weight is not finite.
    """
    _positive_width(max_width)
    pairs = _paired(beams, gold, normalization)
    shapes = tuple(
        (
            tuple(
                typeless(candidate, primitive, constructors)
                for candidate in beam[:max_width]
            ),
            typeless(target, primitive, constructors),
        )
        for beam, target in pairs
    )

    return AccuracyCurve(_hits_by_width(shapes, max_width), len(pairs))


def depth_truncated_accuracy(
    beams: Sequence[ScoredBeam],
    gold: Sequence[Term],
    primitive: Term,
    constructors: Constructors[Term],
    *,
    normalization: BeamNormalization,
    max_depth: int,
) -> AccuracyCurve:
    """Return how often the best entry of a beam agrees with the gold type to a depth.

    At each depth, the best candidate and target are truncated by replacing
        deeper constructors with the primitive. The curve is nonincreasing with depth;
    an empty beam agrees at none.

    Parameters
    ----------
    beams : Sequence[ScoredBeam]
        One beam per item.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    primitive : Term
        The nonconstructor leaf a truncated constructor is replaced by.
    constructors : Constructors[Term]
        The declared constructor algebra the types are built over.
    normalization : BeamNormalization
        What each candidate's log weight is divided by before ranking.
    max_depth : int
        The deepest truncation to score, which is the last level of the curve.

    Returns
    -------
    AccuracyCurve
        One level per depth from one to ``max_depth``.

    Raises
    ------
    NeighborhoodError
        If a type violates the constructor algebra, if a constructor is incompatible,
        or if the primitive is not a nonconstructor leaf.
    ValueError
        If ``max_depth`` is not positive, if there is not one beam per gold type, or
        if a log weight is not finite.
    """
    if max_depth <= 0:
        msg = f"a truncation depth is positive, not {max_depth}"
        raise ValueError(msg)

    pairs = _paired(beams, gold, normalization)
    hits = [0] * max_depth

    for beam, target in pairs:
        if not beam:
            continue

        best = beam[0]

        for depth in range(1, max_depth + 1):
            cut = truncate(best, primitive, constructors, depth=depth)
            wanted = truncate(target, primitive, constructors, depth=depth)

            if cut.key() == wanted.key():
                hits[depth - 1] += 1

    return AccuracyCurve(hits, len(pairs))


def mean_reciprocal_rank(
    beams: Sequence[ScoredBeam],
    gold: Sequence[Term],
    *,
    normalization: BeamNormalization,
) -> float | None:
    """Return the mean over items of one over the rank the gold type has in its beam.

    An item whose gold type is absent from its beam contributes zero. Types are
    compared by structural key.

    Parameters
    ----------
    beams : Sequence[ScoredBeam]
        One beam per item.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    normalization : BeamNormalization
        What each candidate's log weight is divided by before ranking.

    Returns
    -------
    float | None
        The mean, or None where no item was scored.

    Raises
    ------
    ValueError
        If there is not one beam per gold type, or if a log weight is not finite.
    """
    pairs = _paired(beams, gold, normalization)

    if not pairs:
        return None

    total = 0.0

    for beam, target in pairs:
        wanted = target.key()

        for rank, candidate in enumerate(beam, start=1):
            if candidate.key() == wanted:
                total += 1 / rank

                break

    return total / len(pairs)


def beam_size_sweep(
    decode: Callable[[int], Sequence[ScoredBeam]],
    gold: Sequence[Term],
    *,
    widths: Sequence[int],
) -> tuple[SweepPoint, ...]:
    """Return how often the gold type is in the beam, re-decoding at each width.

    Each width is decoded independently because wider search can prune
    differently. A flat curve indicates that added width does not recover more
    gold structural keys.

    Parameters
    ----------
    decode : Callable[[int], Sequence[ScoredBeam]]
        Return one beam per item, decoded at the width it is given.
    gold : Sequence[Term]
        The gold type of each item, in item order.
    widths : Sequence[int]
        The widths to decode at, each positive and greater than the one before.

    Returns
    -------
    tuple[SweepPoint, ...]
        One point per width, in the order given.

    Raises
    ------
    ValueError
        If a width is not positive or not greater than the one before it, if a
        decode does not return one beam per gold type, or if a beam holds more
        candidates than the width it was decoded at.
    """
    points: list[SweepPoint] = []
    previous = 0

    for width in widths:
        if width <= previous:
            msg = f"beam widths ascend from one, so {width} cannot follow {previous}"
            raise ValueError(msg)

        pairs = _paired(decode(width), gold, BeamNormalization.NONE)

        for beam, _ in pairs:
            if len(beam) > width:
                msg = f"a beam decoded at width {width} holds {len(beam)} candidates"
                raise ValueError(msg)

        hits = sum(
            any(candidate.key() == target.key() for candidate in beam)
            for beam, target in pairs
        )
        points.append(SweepPoint(width, hits, len(pairs)))
        previous = width

    return tuple(points)
