"""Unlabeled bracketing metrics and conventions."""

from collections import Counter
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi.eval._values import ConventionVector, MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import ItemId

__all__ = [
    "Bracketing",
    "BracketingF1",
    "Constituent",
    "GoldBracketing",
    "balanced",
    "left_branching",
    "right_branching",
]


class Constituent:
    """A labeled gold span and its depth.

    Parameters
    ----------
    start : int
        First covered token index.
    end : int
        Exclusive end index.
    label : str
        Gold category.
    depth : int
        Zero-based tree depth.

    Raises
    ------
    TypeError
        If an index or depth is not an integer, or the label is not a string.
    ValueError
        If an index or depth is negative, the label is empty, or the span is empty.
    """

    __slots__ = ("_depth", "_end", "_label", "_start")

    def __init__(self, start: int, end: int, *, label: str, depth: int) -> None:
        self._start = ensure_non_negative("a constituent's start", start)
        self._end = ensure_non_negative("a constituent's end", end)
        self._label = ensure_name("a constituent's label", label)
        self._depth = ensure_non_negative("a constituent's depth", depth)

        if self._end <= self._start:
            msg = f"a constituent covers no token: [{start}, {end})"
            raise ValueError(msg)

    @property
    def start(self) -> int:
        """Return the index of the first token the span covers."""
        return self._start

    @property
    def end(self) -> int:
        """Return the index one past the last token the span covers."""
        return self._end

    @property
    def label(self) -> str:
        """Return the gold category."""
        return self._label

    @property
    def depth(self) -> int:
        """Return the span depth."""
        return self._depth

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open span, as a start and an end."""
        return (self._start, self._end)

    @property
    def width(self) -> int:
        """Return how many tokens the span covers."""
        return self._end - self._start

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a constituent with the same four fields."""
        if type(other) is not Constituent:
            return NotImplemented

        return (
            self._start == other._start
            and self._end == other._end
            and self._label == other._label
            and self._depth == other._depth
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the span, the label and the depth."""
        return hash((Constituent, self._start, self._end, self._label, self._depth))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this constituent."""
        return (
            f"Constituent({self._start!r}, {self._end!r}, "
            f"label={self._label!r}, depth={self._depth!r})"
        )


class Bracketing:
    """An unlabeled bracketing of a token sequence.

    Parameters
    ----------
    length : int
        Token count, including punctuation.
    spans : Iterable[tuple[int, int]]
        Half-open constituent spans.

    Raises
    ------
    TypeError
        If the length is not an integer or a span is not an integer pair.
    ValueError
        If the length or a span boundary is invalid.
    """

    __slots__ = ("_length", "_spans")

    def __init__(self, length: int, spans: Iterable[tuple[int, int]]) -> None:
        self._length = ensure_non_negative("a bracketing's length", length)
        self._spans = tuple(sorted({self._checked(span) for span in spans}))

    def _checked(self, span: tuple[int, int]) -> tuple[int, int]:
        """Return one span.

        Parameters
        ----------
        span : tuple[int, int]
            The half-open pair to check.

        Returns
        -------
        tuple[int, int]
            Validated span.

        Raises
        ------
        TypeError
            If the pair is not a start and an end.
        ValueError
            If a boundary is negative or the span is empty or out of range.
        """
        try:
            start, end = span

        except (TypeError, ValueError) as error:
            msg = f"a span is not a start and an end: {span!r}"
            raise TypeError(msg) from error

        low = ensure_non_negative("a span's start", start)
        high = ensure_non_negative("a span's end", end)

        if high <= low:
            msg = f"a span covers no token: [{start}, {end})"
            raise ValueError(msg)

        if high > self._length:
            msg = (
                f"a span runs past a sentence of {self._length} tokens: "
                f"[{start}, {end})"
            )
            raise ValueError(msg)

        return (low, high)

    @property
    def length(self) -> int:
        """Return how many tokens the sentence has."""
        return self._length

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        """Return the spans, in ascending order and without repeats."""
        return self._spans

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value brackets the same sentence the same way."""
        if type(other) is not Bracketing:
            return NotImplemented

        return self._length == other._length and self._spans == other._spans

    @override
    def __hash__(self) -> int:
        """Return a hash over the length and the spans."""
        return hash((Bracketing, self._length, self._spans))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bracketing."""
        return f"Bracketing({self._length!r}, {list(self._spans)!r})"


class GoldBracketing:
    """A gold bracketing and its punctuation.

    Parameters
    ----------
    length : int
        Token count, including punctuation.
    constituents : Iterable[Constituent]
        Gold constituents.
    punctuation : Iterable[int]
        Punctuation token indices.

    Raises
    ------
    TypeError
        If the length or an index is not an integer, or an entry is not a
        constituent.
    ValueError
        If the length or an index is negative or an entry is out of range.
    """

    __slots__ = ("_constituents", "_length", "_punctuation")

    def __init__(
        self,
        length: int,
        constituents: Iterable[Constituent],
        *,
        punctuation: Iterable[int],
    ) -> None:
        self._length = ensure_non_negative("a gold bracketing's length", length)
        self._constituents = tuple(
            sorted(
                (self._checked(entry) for entry in constituents),
                key=lambda entry: (entry.start, entry.end, entry.depth, entry.label),
            )
        )
        self._punctuation = frozenset(self._marked(index) for index in punctuation)

    def _checked(self, entry: Constituent) -> Constituent:
        """Return one constituent.

        Parameters
        ----------
        entry : Constituent
            The constituent to check.

        Returns
        -------
        Constituent
            Validated constituent.

        Raises
        ------
        TypeError
            If it is not a constituent.
        ValueError
            If it runs past the end of the sentence.
        """
        if not isinstance(entry, Constituent):
            msg = f"a gold constituent is not one: {entry!r}"
            raise TypeError(msg)

        if entry.end > self._length:
            msg = (
                f"a constituent runs past a sentence of {self._length} tokens: "
                f"{entry!r}"
            )
            raise ValueError(msg)

        return entry

    def _marked(self, index: int) -> int:
        """Return one punctuation index.

        Parameters
        ----------
        index : int
            The token index to check.

        Returns
        -------
        int
            Validated index.

        Raises
        ------
        TypeError
            If it is not an integer.
        ValueError
            If it is negative or is not a token of this sentence.
        """
        position = ensure_non_negative("a punctuation index", index)

        if position >= self._length:
            msg = (
                f"a punctuation index falls outside a sentence of "
                f"{self._length} tokens: {index!r}"
            )
            raise ValueError(msg)

        return position

    @property
    def length(self) -> int:
        """Return how many tokens the sentence has."""
        return self._length

    @property
    def constituents(self) -> tuple[Constituent, ...]:
        """Return the gold constituents, in span, depth and label order."""
        return self._constituents

    @property
    def punctuation(self) -> frozenset[int]:
        """Return the indices of the punctuation tokens."""
        return self._punctuation

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same gold analysis."""
        if type(other) is not GoldBracketing:
            return NotImplemented

        return (
            self._length == other._length
            and self._constituents == other._constituents
            and self._punctuation == other._punctuation
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the length, the constituents and the punctuation."""
        return hash(
            (GoldBracketing, self._length, self._constituents, self._punctuation)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this gold analysis."""
        return (
            f"GoldBracketing({self._length!r}, {list(self._constituents)!r}, "
            f"punctuation={sorted(self._punctuation)!r})"
        )


class _GoldHit(NamedTuple):
    """A normalized gold span and its recovery status.

    Parameters
    ----------
    label : str
        Gold category.
    width : int
        Width after punctuation removal.
    depth : int
        Gold tree depth.
    recalled : bool
        Whether the predicted spans contain this occurrence.
    """

    label: str
    width: int
    depth: int
    recalled: bool


class _Scored(NamedTuple):
    """Stores a scored sentence's contributions.

    Parameters
    ----------
    matched : int
        Matched span count.
    predicted : int
        Predicted span count after filtering.
    f1 : float
        Sentence-level F1.
    gold : tuple[_GoldHit, ...]
        Surviving gold spans.
    """

    matched: int
    predicted: int
    f1: float
    gold: tuple[_GoldHit, ...]


class _Item(NamedTuple):
    """Stores one update's metric state.

    Parameters
    ----------
    pred : Bracketing | None
        Predicted bracketing, if available.
    gold : GoldBracketing
        Gold bracketing.
    scored : _Scored | None
        Score contribution, if included by the conventions.
    """

    pred: Bracketing | None
    gold: GoldBracketing
    scored: _Scored | None


def _boundaries(length: int, dropped: frozenset[int]) -> tuple[int, ...]:
    """Return normalized token boundaries after filtering.

    Parameters
    ----------
    length : int
        Token count.
    dropped : frozenset[int]
        Removed token indices.

    Returns
    -------
    tuple[int, ...]
        Kept-token count at each boundary.
    """
    counts = [0]

    for index in range(length):
        counts.append(counts[-1] + (0 if index in dropped else 1))

    return tuple(counts)


def _reindexed(
    boundaries: tuple[int, ...], start: int, end: int
) -> tuple[int, int] | None:
    """Return a span in the coordinates the kept tokens have.

    Parameters
    ----------
    boundaries : tuple[int, ...]
        Kept-token count at each boundary.
    start : int
        Original start index.
    end : int
        Original end index.

    Returns
    -------
    tuple[int, int] | None
        Reindexed span, or ``None`` if all covered tokens were removed.
    """
    low, high = boundaries[start], boundaries[end]

    return (low, high) if low < high else None


def _is_trivial(span: tuple[int, int], length: int) -> bool:
    """Return whether any bracketing contains the span.

    Parameters
    ----------
    span : tuple[int, int]
        The half-open span.
    length : int
        Filtered token count.

    Returns
    -------
    bool
        Whether the span is a single token or the whole sentence.
    """
    low, high = span

    return high - low == 1 or (low == 0 and high == length)


def _harmonic(precision: float, recall: float) -> float:
    """Return the harmonic mean of a precision and a recall.

    Parameters
    ----------
    precision : float
        The proportion of predicted spans that are gold.
    recall : float
        The proportion of gold spans that were predicted.

    Returns
    -------
    float
        Harmonic mean, or zero when both inputs are zero.
    """
    total = precision + recall

    return 0.0 if total == 0.0 else 2.0 * precision * recall / total


def surviving_gold(
    gold: GoldBracketing,
    boundaries: tuple[int, ...],
    length: int,
    conventions: ConventionVector,
) -> tuple[tuple[tuple[int, int], str, int], ...]:
    """Return gold spans after applying the conventions.

    Parameters
    ----------
    gold : GoldBracketing
        The gold analysis.
    boundaries : tuple[int, ...]
        Kept-token count at each boundary.
    length : int
        Filtered token count.
    conventions : ConventionVector
        Bracketing conventions.

    Returns
    -------
    tuple[tuple[tuple[int, int], str, int], ...]
        Span, label, and depth triples in deterministic order.
    """
    reindexed: list[tuple[tuple[int, int], str, int]] = []

    for entry in gold.constituents:
        span = _reindexed(boundaries, entry.start, entry.end)

        if span is None:
            continue

        if conventions.trivial_spans == "drop" and _is_trivial(span, length):
            continue

        reindexed.append((span, entry.label, entry.depth))

    if conventions.unary_chains == "collapse":
        outermost: dict[tuple[int, int], tuple[str, int]] = {}

        for span, label, depth in reindexed:
            standing = outermost.get(span)

            if standing is None or depth < standing[1]:
                outermost[span] = (label, depth)

        reindexed = [(span, label, depth) for span, (label, depth) in outermost.items()]

    return tuple(sorted(reindexed, key=lambda entry: (entry[0], entry[2], entry[1])))


def surviving_spans(
    spans: Iterable[tuple[int, int]],
    boundaries: tuple[int, ...],
    length: int,
    conventions: ConventionVector,
) -> frozenset[tuple[int, int]]:
    """Return predicted spans after applying the conventions.

    Parameters
    ----------
    spans : Iterable[tuple[int, int]]
        Predicted spans.
    boundaries : tuple[int, ...]
        Kept-token count at each boundary.
    length : int
        Filtered token count.
    conventions : ConventionVector
        Bracketing conventions.

    Returns
    -------
    frozenset[tuple[int, int]]
        The surviving spans.
    """
    reindexed: set[tuple[int, int]] = set()

    for start, end in spans:
        span = _reindexed(boundaries, start, end)

        if span is None:
            continue

        if conventions.trivial_spans == "drop" and _is_trivial(span, length):
            continue

        reindexed.add(span)

    return frozenset(reindexed)


def right_branching(length: int) -> Bracketing:
    """Return the right-branching bracketing of a sentence of a given length.

    Parameters
    ----------
    length : int
        Token count.

    Returns
    -------
    Bracketing
        Right-branching spans of width two or greater.

    Raises
    ------
    TypeError
        If the length is not an integer.
    ValueError
        If the length is negative.
    """
    size = ensure_non_negative("a sentence length", length)
    nodes = size - 1 if size > 0 else 0

    return Bracketing(size, ((start, size) for start in range(nodes)))


def left_branching(length: int) -> Bracketing:
    """Return the left-branching bracketing of a sentence.

    Parameters
    ----------
    length : int
        Token count.

    Returns
    -------
    Bracketing
        Left-branching spans of width two or greater.
    """
    size = ensure_non_negative("a sentence length", length)

    return Bracketing(size, ((0, end) for end in range(2, size + 1)))


def balanced(length: int) -> Bracketing:
    """Return the recursively balanced bracketing of a sentence.

    Every nontrivial range splits at its lower midpoint. Thus odd lengths have
    their extra token on the right, making the baseline deterministic without
    consulting the gold tree.

    Parameters
    ----------
    length : int
        Token count.

    Returns
    -------
    Bracketing
        Balanced spans of width two or greater.
    """
    size = ensure_non_negative("a sentence length", length)
    spans: list[tuple[int, int]] = []

    def visit(start: int, end: int) -> None:
        if end - start < 2:
            return

        spans.append((start, end))
        middle = start + (end - start) // 2
        visit(start, middle)
        visit(middle, end)

    visit(0, size)

    return Bracketing(size, spans)


class BracketingF1:
    """Scores unlabeled spans under explicit bracketing conventions.

    Parameters
    ----------
    conventions : ConventionVector
        Required scoring conventions.
    seed : int | None
        Run seed recorded in metric results.

    Raises
    ------
    TypeError
        If the conventions are not a vector, or the seed is not an integer.
    ValueError
        If the seed is negative.
    """

    __slots__ = ("_constituents", "_conventions", "_items", "_sentences")

    def __init__(
        self, conventions: ConventionVector, *, seed: int | None = None
    ) -> None:
        if not isinstance(conventions, ConventionVector):
            msg = f"a bracketing metric's conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        self._conventions = conventions
        self._sentences = Resampling("sentence", seed)
        self._constituents = Resampling("constituent", seed)
        self._items: dict[ItemId, _Item] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "unlabeled-bracketing-f1"

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every sentence scored so far."""
        self._items = {}

    def update(
        self, pred: Bracketing | None, gold: GoldBracketing, *, key: ItemId
    ) -> None:
        """Score one predicted bracketing against its gold analysis.

        Parameters
        ----------
        pred : Bracketing | None
            Predicted bracketing, or ``None`` for a failed parse.
        gold : GoldBracketing
            The gold analysis.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If the prediction is neither a bracketing nor absent, or the gold
            analysis is not one.
        ValueError
            If token counts differ or the key was updated with different values.
        """
        if not isinstance(gold, GoldBracketing):
            msg = f"a gold bracketing is not one: {gold!r}"
            raise TypeError(msg)

        if pred is not None:
            if not isinstance(pred, Bracketing):
                msg = f"a predicted bracketing is not one: {pred!r}"
                raise TypeError(msg)

            if pred.length != gold.length:
                msg = (
                    f"a prediction of {pred.length} tokens was scored against a "
                    f"gold analysis of {gold.length}"
                )
                raise ValueError(msg)

        standing = self._items.get(key)

        if standing is not None:
            if (standing.pred, standing.gold) != (pred, gold):
                msg = (
                    f"the item {key!r} was already scored against a different analysis"
                )
                raise ValueError(msg)

            return

        self._items[key] = self._scored(pred, gold)

    def compute(self) -> MetricResult:
        """Return the aggregate over every sentence scored so far.

        Returns
        -------
        MetricResult
            Configured aggregate and supporting counts, or an absent result when
            no sentence is scored.
        """
        scored = self._surviving()
        uncovered = sum(1 for item in self._items.values() if item.pred is None)

        if not scored:
            return MetricResult(None, 0, self._sentences)

        matched = sum(entry.matched for entry in scored)
        predicted = sum(entry.predicted for entry in scored)
        gold = sum(len(entry.gold) for entry in scored)
        precision = matched / predicted if predicted else 0.0
        recall = matched / gold if gold else 0.0
        corpus = _harmonic(precision, recall)
        sentence = fmean([entry.f1 for entry in scored])

        return MetricResult(
            sentence if self._conventions.aggregation == "sentence" else corpus,
            len(scored),
            self._sentences,
            components={
                "sentence_f1": sentence,
                "corpus_f1": corpus,
                "precision": precision,
                "recall": recall,
                "matched": float(matched),
                "predicted": float(predicted),
                "gold": float(gold),
                "scored": float(len(scored)),
                "dropped": float(len(self._items) - len(scored)),
                "uncovered": float(uncovered),
            },
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each sentence's own F1, keyed by the key it was updated with.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            Sentence results keyed by item identifier.
        """
        results: dict[ItemId, MetricResult] = {}

        for key, item in self._items.items():
            if item.scored is None:
                results[key] = MetricResult(None, 0, self._sentences)

            else:
                results[key] = MetricResult(
                    item.scored.f1,
                    1,
                    self._sentences,
                    components={
                        "matched": float(item.scored.matched),
                        "predicted": float(item.scored.predicted),
                        "gold": float(len(item.scored.gold)),
                    },
                )

        return MappingProxyType(results)

    def per_label_recall(self) -> Mapping[str, MetricResult]:
        """Return recall over the gold spans of each gold category.

        Returns
        -------
        Mapping[str, MetricResult]
            Constituent recall by gold label.
        """
        tally: dict[str, tuple[int, int]] = {}

        for hit in self._hits():
            tally[hit.label] = _tallied(tally.get(hit.label), recalled=hit.recalled)

        return MappingProxyType(
            {label: self._recall(*tally[label]) for label in sorted(tally)}
        )

    def recall_by_length(self) -> Mapping[int, MetricResult]:
        """Return recall over the gold spans of each width.

        Returns
        -------
        Mapping[int, MetricResult]
            Constituent recall by filtered span width.
        """
        tally: dict[int, tuple[int, int]] = {}

        for hit in self._hits():
            tally[hit.width] = _tallied(tally.get(hit.width), recalled=hit.recalled)

        return MappingProxyType(
            {width: self._recall(*tally[width]) for width in sorted(tally)}
        )

    def recall_by_depth(self) -> Mapping[int, MetricResult]:
        """Return recall over the gold spans at each depth.

        Returns
        -------
        Mapping[int, MetricResult]
            Constituent recall by surviving gold depth.
        """
        tally: dict[int, tuple[int, int]] = {}

        for hit in self._hits():
            tally[hit.depth] = _tallied(tally.get(hit.depth), recalled=hit.recalled)

        return MappingProxyType(
            {depth: self._recall(*tally[depth]) for depth in sorted(tally)}
        )

    def _recall(self, total: int, recalled: int) -> MetricResult:
        """Return one bucket's recall.

        Parameters
        ----------
        total : int
            Gold span count.
        recalled : int
            Recalled span count.

        Returns
        -------
        MetricResult
            Recall counted in constituents.
        """
        return MetricResult(
            recalled / total,
            total,
            self._constituents,
            components={"recalled": float(recalled), "gold": float(total)},
        )

    def _hits(self) -> tuple[_GoldHit, ...]:
        """Return every surviving gold span, in key and then span order."""
        return tuple(entry for scored in self._surviving() for entry in scored.gold)

    def _surviving(self) -> tuple[_Scored, ...]:
        """Return sentence contributions in ascending key order."""
        scored: list[_Scored] = []

        for _, item in sorted(self._items.items()):
            if item.scored is not None:
                scored.append(item.scored)

        return tuple(scored)

    def _scored(self, pred: Bracketing | None, gold: GoldBracketing) -> _Item:
        """Return a sentence's contributions under the metric conventions.

        Parameters
        ----------
        pred : Bracketing | None
            Predicted bracketing, if available.
        gold : GoldBracketing
            The gold analysis.

        Returns
        -------
        _Item
            Stored inputs and score contribution.
        """
        conventions = self._conventions
        removed = (
            gold.punctuation if conventions.punctuation == "strip" else frozenset()
        )
        boundaries = _boundaries(gold.length, removed)
        length = boundaries[gold.length]

        if length < conventions.minimum_length:
            return _Item(pred=pred, gold=gold, scored=None)

        surviving = surviving_gold(gold, boundaries, length, conventions)

        if not surviving:
            return _Item(pred=pred, gold=gold, scored=None)

        if pred is None and conventions.coverage == "ignore":
            return _Item(pred=pred, gold=gold, scored=None)

        proposed = (
            frozenset()
            if pred is None
            else surviving_spans(pred.spans, boundaries, length, conventions)
        )
        available = Counter(span for span, _, _ in surviving)
        hits: list[_GoldHit] = []

        for span, label, depth in surviving:
            recalled = span in proposed and available[span] > 0

            if recalled:
                available[span] = 0

            hits.append(
                _GoldHit(
                    label=label,
                    width=span[1] - span[0],
                    depth=depth,
                    recalled=recalled,
                )
            )

        matched = sum(1 for entry in hits if entry.recalled)
        precision = matched / len(proposed) if proposed else 0.0
        recall = matched / len(surviving)

        return _Item(
            pred=pred,
            gold=gold,
            scored=_Scored(
                matched=matched,
                predicted=len(proposed),
                f1=_harmonic(precision, recall),
                gold=tuple(hits),
            ),
        )


def _tallied(standing: tuple[int, int] | None, *, recalled: bool) -> tuple[int, int]:
    """Add a hit to a bucket's total and recovered count.

    Parameters
    ----------
    standing : tuple[int, int] | None
        Existing total and recalled counts, if any.
    recalled : bool
        Whether the prediction proposed this gold span.

    Returns
    -------
    tuple[int, int]
        Updated total and recalled counts.
    """
    total, hits = standing if standing is not None else (0, 0)

    return (total + 1, hits + int(recalled))
