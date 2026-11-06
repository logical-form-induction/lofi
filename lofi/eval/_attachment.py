"""Attachment scoring over CoNLL-style dependency trees.

Token positions are one-based and the artificial root is zero. Directed
accuracy matches heads; undirected accuracy also accepts a reversed arc.
Reports group both measures by sentence-length band.
"""

from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.data import AttachmentTree as _AttachmentTree
from lofi.eval._values import ConventionVector, MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue

__all__ = [
    "ATTACHMENT_DIRECTIONS",
    "WSJ_LENGTH_BANDS",
    "AttachmentDirection",
    "AttachmentScoreRow",
    "AttachmentScorer",
    "LengthBand",
]

type AttachmentDirection = Literal["directed", "undirected"]
"""An attachment matching relation."""

ATTACHMENT_DIRECTIONS: Final[tuple[AttachmentDirection, ...]] = (
    "directed",
    "undirected",
)
"""Attachment matching relations in report order."""

ROOT: Final = 0
"""The head position of the token the artificial root governs."""


@final
class LengthBand(FrozenValue):
    """A sentence-length band a score is reported within.

    Parameters
    ----------
    name : str
        Band name used in report rows.
    limit : int | None
        Largest counted length admitted, or None for every length.

    Raises
    ------
    TypeError
        If the name is not a string or the limit is not an integer.
    ValueError
        If the name is empty or the limit is negative.
    """

    _noun = "length band"

    __slots__ = ("_limit", "_name")

    def __init__(self, name: str, limit: int | None) -> None:
        object.__setattr__(self, "_name", ensure_name("a length band name", name))
        object.__setattr__(
            self,
            "_limit",
            None if limit is None else ensure_non_negative("a length limit", limit),
        )

    @property
    def name(self) -> str:
        """Return the band name."""
        return self._name

    @property
    def limit(self) -> int | None:
        """Return the largest admitted length, if the band has one."""
        return self._limit

    def admits(self, length: int) -> bool:
        """Return whether a counted sentence length falls within the band.

        Parameters
        ----------
        length : int
            Token count after the punctuation policy is applied.
        """
        return self._limit is None or length <= self._limit

    def _fields(self) -> tuple[str, int | None]:
        """Return fields in declaration order."""
        return self._name, self._limit

    @override
    def __reduce__(self) -> tuple[type[LengthBand], tuple[str, int | None]]:
        """Return the reconstruction call."""
        return LengthBand, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another band has the same name and limit."""
        if type(other) is not LengthBand:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and limit."""
        return hash((LengthBand, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"LengthBand{self._fields()!r}"


WSJ_LENGTH_BANDS: Final[tuple[LengthBand, ...]] = (
    LengthBand("wsj10", 10),
    LengthBand("wsj20", 20),
    LengthBand("wsj-infinity", None),
)
"""The Wall Street Journal length bands, from ten words upward to no limit."""


@final
class AttachmentScoreRow(FrozenValue):
    """One attachment score within a length band under declared conventions.

    Parameters
    ----------
    band : LengthBand
        Length band the score is computed within.
    direction : AttachmentDirection
        Attachment matching relation.
    conventions : ConventionVector
        Scoring conventions the score was computed under.
    result : MetricResult
        The score, its token and sentence counts, and its coverage.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If the direction is unknown.
    """

    _noun = "attachment score row"

    __slots__ = ("_band", "_conventions", "_direction", "_result")

    def __init__(
        self,
        band: LengthBand,
        direction: AttachmentDirection,
        conventions: ConventionVector,
        result: MetricResult,
    ) -> None:
        if not isinstance(band, LengthBand):
            msg = f"an attachment band is not one: {band!r}"
            raise TypeError(msg)

        if direction not in ATTACHMENT_DIRECTIONS:
            msg = f"an unknown attachment direction was requested: {direction!r}"
            raise ValueError(msg)

        if not isinstance(conventions, ConventionVector):
            msg = f"attachment conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        if not isinstance(result, MetricResult):
            msg = f"an attachment score result is not one: {result!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_band", band)
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_conventions", conventions)
        object.__setattr__(self, "_result", result)

    @property
    def name(self) -> str:
        """Return the row name, unique within a scorer's rows."""
        return f"attachment:{self._band.name}:{self._direction}"

    @property
    def band(self) -> LengthBand:
        """Return the length band."""
        return self._band

    @property
    def direction(self) -> AttachmentDirection:
        """Return the attachment matching relation."""
        return self._direction

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def result(self) -> MetricResult:
        """Return the metric result."""
        return self._result

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as serializable report columns."""
        return {
            "name": self.name,
            "band": self._band.name,
            "length_limit": self._band.limit,
            "direction": self._direction,
            "value": self._result.value,
            "sentences": self._result.n,
            "aggregation": self._conventions.aggregation,
            "coverage": self._conventions.coverage,
            "punctuation": self._conventions.punctuation,
            "minimum_length": self._conventions.minimum_length,
            **dict(self._result.components),
        }

    def _fields(
        self,
    ) -> tuple[LengthBand, AttachmentDirection, ConventionVector, MetricResult]:
        """Return fields in declaration order."""
        return self._band, self._direction, self._conventions, self._result

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same fields."""
        if type(other) is not AttachmentScoreRow:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((AttachmentScoreRow, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"AttachmentScoreRow{self._fields()!r}"


class _Counts(NamedTuple):
    """One sentence's scored tokens under one direction."""

    matched: int
    tokens: int


class _Item(NamedTuple):
    """One stored sentence and its counted length."""

    prediction: _AttachmentTree | None
    gold: _AttachmentTree
    length: int


def _matches(
    prediction: _AttachmentTree, gold: _AttachmentTree, position: int
) -> tuple[bool, bool]:
    """Return whether a token's gold arc is recovered directed and undirected.

    Parameters
    ----------
    prediction : _AttachmentTree
        Predicted tree.
    gold : _AttachmentTree
        Gold tree.
    position : int
        One-based token position.
    """
    gold_head = gold.head_of(position)
    directed = prediction.head_of(position) == gold_head
    reversed_arc = gold_head != ROOT and prediction.head_of(gold_head) == position

    return directed, directed or reversed_arc


@final
class AttachmentScorer:
    """An accumulator of attachment accuracy within length bands.

    Punctuation, minimum-length, aggregation, and coverage conventions control
    which tokens and sentences enter each score. The aggregate is directed
    accuracy over all admitted sentences.

    Parameters
    ----------
    conventions : ConventionVector
        Scoring conventions.
    bands : Sequence[LengthBand]
        Length bands to report within, in report order.
    seed : int | None
        Run seed recorded in metric results.

    Raises
    ------
    TypeError
        If the conventions are not a vector or a band is not one.
    ValueError
        If no band is given, two bands share a name, or the seed is negative.
    """

    __slots__ = ("_bands", "_conventions", "_items", "_sentences")

    def __init__(
        self,
        conventions: ConventionVector,
        bands: Sequence[LengthBand] = WSJ_LENGTH_BANDS,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(conventions, ConventionVector):
            msg = f"attachment conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        held = tuple(bands)

        for position, band in enumerate(held):
            if not isinstance(band, LengthBand):
                msg = f"length band {position} is not one: {band!r}"
                raise TypeError(msg)

        if not held:
            msg = "an attachment scorer reports within at least one length band"
            raise ValueError(msg)

        if len({band.name for band in held}) != len(held):
            msg = f"length bands share a name: {[band.name for band in held]!r}"
            raise ValueError(msg)

        self._conventions = conventions
        self._bands = held
        self._sentences = Resampling("sentence", seed)
        self._items: dict[ItemId, _Item] = {}

    @property
    def name(self) -> str:
        """Return the scorer name."""
        return "attachment"

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def bands(self) -> tuple[LengthBand, ...]:
        """Return the length bands in report order."""
        return self._bands

    @property
    def order_invariant(self) -> bool:
        """Return whether rows are invariant to update order."""
        return True

    def reset(self) -> None:
        """Forget every sentence scored so far."""
        self._items = {}

    def _scored_positions(self, gold: _AttachmentTree) -> tuple[int, ...]:
        """Return the one-based positions the conventions score.

        Parameters
        ----------
        gold : _AttachmentTree
            Gold tree whose punctuation the policy reads.
        """
        removed = (
            gold.punctuation
            if self._conventions.punctuation == "strip"
            else frozenset[int]()
        )

        return tuple(
            position
            for position in range(1, gold.length + 1)
            if position not in removed
        )

    def update(
        self,
        pred: _AttachmentTree | None,
        gold: _AttachmentTree,
        *,
        key: ItemId,
    ) -> None:
        """Store one predicted tree against its gold tree.

        Parameters
        ----------
        pred : _AttachmentTree | None
            Predicted tree, or None for a failed parse.
        gold : _AttachmentTree
            Gold tree.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If the prediction is neither a tree nor absent, or the gold tree
            is not one.
        ValueError
            If the trees differ in length or forms, or the key was already
            scored against other trees.
        """
        if not isinstance(gold, _AttachmentTree):
            msg = f"a gold attachment tree is not one: {gold!r}"
            raise TypeError(msg)

        if pred is not None:
            if not isinstance(pred, _AttachmentTree):
                msg = f"a predicted attachment tree is not one: {pred!r}"
                raise TypeError(msg)

            if pred.forms != gold.forms:
                msg = (
                    f"a predicted tree over {pred.forms!r} is scored against "
                    f"a gold tree over {gold.forms!r}"
                )
                raise ValueError(msg)

        identifier = ItemId(ensure_name("an attachment item id", key))
        item = _Item(pred, gold, len(self._scored_positions(gold)))
        standing = self._items.get(identifier)

        if standing is not None:
            if standing != item:
                msg = f"item {identifier!r} was already scored against other trees"
                raise ValueError(msg)

            return

        self._items[identifier] = item

    def _counts(self, item: _Item, direction: AttachmentDirection) -> _Counts:
        """Count a sentence's recovered arcs under one direction.

        Parameters
        ----------
        item : _Item
            Stored sentence.
        direction : AttachmentDirection
            Attachment matching relation.
        """
        positions = self._scored_positions(item.gold)

        if item.prediction is None:
            return _Counts(0, len(positions))

        matched = 0

        for position in positions:
            directed, undirected = _matches(item.prediction, item.gold, position)
            matched += directed if direction == "directed" else undirected

        return _Counts(matched, len(positions))

    def _admits(self, item: _Item) -> bool:
        """Return whether the conventions score a stored sentence.

        Parameters
        ----------
        item : _Item
            Stored sentence.
        """
        conventions = self._conventions

        return (
            item.length >= conventions.minimum_length
            and item.length > 0
            and (conventions.coverage == "zero" or item.prediction is not None)
        )

    def _eligible(self, band: LengthBand | None) -> tuple[_Item, ...]:
        """Return the stored sentences a band scores, in key order.

        Parameters
        ----------
        band : LengthBand | None
            Length band, or None for every length.
        """
        return tuple(
            item
            for _, item in sorted(self._items.items())
            if self._admits(item) and (band is None or band.admits(item.length))
        )

    def _result(
        self, eligible: Sequence[_Item], direction: AttachmentDirection
    ) -> MetricResult:
        """Aggregate the eligible sentences under one direction.

        Parameters
        ----------
        eligible : Sequence[_Item]
            Sentences the conventions admit, in key order.
        direction : AttachmentDirection
            Attachment matching relation.
        """
        counts = tuple(self._counts(item, direction) for item in eligible)
        matched = sum(entry.matched for entry in counts)
        tokens = sum(entry.tokens for entry in counts)
        parsed = sum(item.prediction is not None for item in eligible)

        if not counts:
            value = None
        elif self._conventions.aggregation == "corpus":
            value = matched / tokens
        else:
            value = fmean(entry.matched / entry.tokens for entry in counts)

        return MetricResult(
            value,
            len(counts),
            self._sentences,
            components={
                "matched": float(matched),
                "tokens": float(tokens),
                "parsed": float(parsed),
            },
        )

    def _row(
        self, band: LengthBand, direction: AttachmentDirection
    ) -> AttachmentScoreRow:
        """Build one band's row under one direction.

        Parameters
        ----------
        band : LengthBand
            Length band.
        direction : AttachmentDirection
            Attachment matching relation.
        """
        return AttachmentScoreRow(
            band,
            direction,
            self._conventions,
            self._result(self._eligible(band), direction),
        )

    def rows(self) -> tuple[AttachmentScoreRow, ...]:
        """Return every band's directed and undirected rows in report order."""
        return tuple(
            self._row(band, direction)
            for band in self._bands
            for direction in ATTACHMENT_DIRECTIONS
        )

    def compute(self) -> MetricResult:
        """Return directed attachment accuracy over every admitted sentence."""
        return self._result(self._eligible(None), "directed")

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each sentence's directed accuracy, keyed by its item id.

        A sentence the conventions do not admit has no value.
        """
        results: dict[ItemId, MetricResult] = {}

        for key, item in self._items.items():
            if self._admits(item):
                results[key] = self._result((item,), "directed")

            else:
                results[key] = MetricResult(None, 0, self._sentences)

        return MappingProxyType(results)
