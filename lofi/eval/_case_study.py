"""Span-group reports for induced type distributions."""

from math import fsum, isclose
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.data import ByteSpan, Item, TokenSpan
from lofi.eval._supertagging import TokenSupertags

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.terms import TermKey

__all__ = [
    "COMPLEMENT_CONTRAST",
    "FINITENESS_CONTRAST",
    "MissingByteSpanError",
    "MissingPosteriorError",
    "MissingTextError",
    "SpanContrast",
    "SpanGroup",
    "SpanGroupReport",
    "SpanHit",
    "TokenSurfaceMismatchError",
    "induced_type_report",
    "locate_span_groups",
]

_PROBABILITY_TOLERANCE = 1e-9
"""Tolerance for aggregated probability mass."""

type _SpanGroupFields = tuple[str, tuple[str, ...]]
"""Fields needed to reconstruct a span group."""

type _SpanGroupReduction = tuple[
    Callable[[_SpanGroupFields], SpanGroup],
    tuple[_SpanGroupFields],
]
"""Span-group pickle reduction."""

type _SpanContrastFields = tuple[str, tuple[SpanGroup, ...]]
"""Fields needed to reconstruct a span contrast."""

type _SpanContrastReduction = tuple[
    Callable[[_SpanContrastFields], SpanContrast],
    tuple[_SpanContrastFields],
]
"""Span-contrast pickle reduction."""

type _SpanHitFields = tuple[SpanGroup, ItemId, str, TokenSpan, ByteSpan]
"""Fields needed to reconstruct a span hit."""

type _SpanHitReduction = tuple[
    Callable[[_SpanHitFields], SpanHit],
    tuple[_SpanHitFields],
]
"""Span-hit pickle reduction."""

type _StructuralProbability = tuple[TermKey, float]
"""Probability mass assigned to one structural term key."""

type _SpanGroupReportFields = tuple[
    SpanGroup,
    tuple[SpanHit, ...],
    tuple[_StructuralProbability, ...],
]
"""Fields needed to reconstruct a span-group report."""

type _SpanGroupReportReduction = tuple[
    Callable[[_SpanGroupReportFields], SpanGroupReport],
    tuple[_SpanGroupReportFields],
]
"""Span-group-report pickle reduction."""


@final
class SpanGroup(FrozenValue):
    """A named group of exact surface forms."""

    _noun = "case-study value"

    __slots__ = ("_name", "_substrings")

    def __init__(self, name: str, substrings: Sequence[str]) -> None:
        held = tuple(
            ensure_name(f"span group substring {position}", substring)
            for position, substring in enumerate(substrings)
        )

        if not held:
            msg = "a span group declares no substring"
            raise ValueError(msg)

        if len(set(held)) != len(held):
            msg = "a span group repeats a substring"
            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("a span group name", name))
        object.__setattr__(self, "_substrings", held)

    @property
    def name(self) -> str:
        """Return the group name."""
        return self._name

    @property
    def substrings(self) -> tuple[str, ...]:
        """Return exact surface forms in declaration order."""
        return self._substrings

    def _fields(self) -> _SpanGroupFields:
        """Return fields in reconstruction order."""
        return self._name, self._substrings

    @override
    def __copy__(self) -> SpanGroup:
        """Reconstruct the span group."""
        return _restore_span_group(self._fields())

    @override
    def __deepcopy__(self, memo: dict[int, SpanGroup]) -> SpanGroup:
        """Reconstruct the span group."""
        del memo

        return _restore_span_group(self._fields())

    @override
    def __reduce__(self) -> _SpanGroupReduction:
        """Return the span-group reconstruction call."""
        return _restore_span_group, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another group has the same declaration."""
        if type(other) is not SpanGroup:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the declaration."""
        return hash((SpanGroup, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"SpanGroup({self._name!r}, {self._substrings!r})"


def _restore_span_group(fields: _SpanGroupFields) -> SpanGroup:
    """Reconstruct a span group from frozen fields."""
    return SpanGroup(*fields)


@final
class SpanContrast(FrozenValue):
    """A named contrast between two span groups."""

    _noun = "case-study value"

    __slots__ = ("_groups", "_name")

    def __init__(self, name: str, groups: Sequence[SpanGroup]) -> None:
        held = tuple(groups)

        try:
            first, second = held
        except ValueError as error:
            msg = f"a span contrast requires two groups, not {len(held)}"

            raise ValueError(msg) from error

        for position, group in enumerate(held):
            if not isinstance(group, SpanGroup):
                msg = f"span contrast group {position} is not a SpanGroup: {group!r}"
                raise TypeError(msg)

        if first.name == second.name:
            msg = f"a span contrast repeats group name {first.name!r}"
            raise ValueError(msg)

        declared = tuple(substring for group in held for substring in group.substrings)

        if len(set(declared)) != len(declared):
            msg = "a span contrast declares one substring in multiple groups"
            raise ValueError(msg)

        object.__setattr__(self, "_name", ensure_name("a span contrast name", name))
        object.__setattr__(self, "_groups", held)

    @property
    def name(self) -> str:
        """Return the contrast name."""
        return self._name

    @property
    def groups(self) -> tuple[SpanGroup, SpanGroup]:
        """Return the two groups in declaration order."""
        first, second = self._groups

        return first, second

    def _fields(self) -> _SpanContrastFields:
        """Return fields in reconstruction order."""
        return self._name, self._groups

    @override
    def __copy__(self) -> SpanContrast:
        """Reconstruct the span contrast."""
        return _restore_span_contrast(self._fields())

    @override
    def __deepcopy__(self, memo: dict[int, SpanContrast]) -> SpanContrast:
        """Reconstruct the span contrast."""
        del memo

        return _restore_span_contrast(self._fields())

    @override
    def __reduce__(self) -> _SpanContrastReduction:
        """Return the span-contrast reconstruction call."""
        return _restore_span_contrast, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another contrast has the same declaration."""
        if type(other) is not SpanContrast:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the declaration."""
        return hash((SpanContrast, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"SpanContrast({self._name!r}, {self._groups!r})"


def _restore_span_contrast(fields: _SpanContrastFields) -> SpanContrast:
    """Reconstruct a span contrast from frozen fields."""
    return SpanContrast(*fields)


@final
class SpanHit(FrozenValue):
    """One exact surface-form occurrence at a token span."""

    _noun = "case-study value"

    __slots__ = ("_byte_span", "_group", "_item_id", "_substring", "_token_span")

    def __init__(
        self,
        group: SpanGroup,
        item_id: ItemId,
        substring: str,
        token_span: TokenSpan,
        byte_span: ByteSpan,
    ) -> None:
        if not isinstance(group, SpanGroup):
            msg = f"a span hit group is not a SpanGroup: {group!r}"
            raise TypeError(msg)

        if not item_id:
            msg = "a span hit has no item id"
            raise ValueError(msg)

        held_substring = ensure_name("a span hit substring", substring)

        if not any(held_substring == declared for declared in group.substrings):
            msg = (
                f"span group {group.name!r} does not declare substring "
                f"{held_substring!r}"
            )
            raise ValueError(msg)

        if token_span.length == 0:
            msg = "a span hit covers no token"
            raise ValueError(msg)

        if byte_span.length == 0:
            msg = "a span hit covers no byte"
            raise ValueError(msg)

        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_item_id", item_id)
        object.__setattr__(self, "_substring", held_substring)
        object.__setattr__(self, "_token_span", token_span)
        object.__setattr__(self, "_byte_span", byte_span)

    @property
    def group(self) -> SpanGroup:
        """Return the matched group."""
        return self._group

    @property
    def item_id(self) -> ItemId:
        """Return the source item identifier."""
        return self._item_id

    @property
    def substring(self) -> str:
        """Return the matched declared surface form."""
        return self._substring

    @property
    def token_span(self) -> TokenSpan:
        """Return the matched token interval."""
        return self._token_span

    @property
    def byte_span(self) -> ByteSpan:
        """Return the matched UTF-8 byte interval."""
        return self._byte_span

    def _fields(self) -> _SpanHitFields:
        """Return fields in reconstruction order."""
        return (
            self._group,
            self._item_id,
            self._substring,
            self._token_span,
            self._byte_span,
        )

    @override
    def __copy__(self) -> SpanHit:
        """Reconstruct the span hit."""
        return _restore_span_hit(self._fields())

    @override
    def __deepcopy__(self, memo: dict[int, SpanHit]) -> SpanHit:
        """Reconstruct the span hit."""
        del memo

        return _restore_span_hit(self._fields())

    @override
    def __reduce__(self) -> _SpanHitReduction:
        """Return the span-hit reconstruction call."""
        return _restore_span_hit, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another hit has the same fields."""
        if type(other) is not SpanHit:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the hit fields."""
        return hash((SpanHit, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"SpanHit({self._group!r}, {self._item_id!r}, {self._substring!r}, "
            f"{self._token_span!r}, {self._byte_span!r})"
        )


def _restore_span_hit(fields: _SpanHitFields) -> SpanHit:
    """Reconstruct a span hit from frozen fields."""
    return SpanHit(*fields)


@final
class SpanGroupReport(FrozenValue):
    """An induced structural type distribution for one span group."""

    _noun = "case-study value"

    __slots__ = ("_distribution", "_group", "_hits", "_sentence_count")

    def __init__(
        self,
        group: SpanGroup,
        hits: Sequence[SpanHit],
        distribution: Sequence[_StructuralProbability],
    ) -> None:
        if not isinstance(group, SpanGroup):
            msg = f"a span-group report group is not a SpanGroup: {group!r}"
            raise TypeError(msg)

        held_hits = tuple(hits)

        for position, hit in enumerate(held_hits):
            if not isinstance(hit, SpanHit):
                msg = f"span-group report hit {position} is not a SpanHit: {hit!r}"
                raise TypeError(msg)

            if hit.group != group:
                msg = f"span-group report hit {position} belongs to {hit.group.name!r}"
                raise ValueError(msg)

        held_distribution = tuple(
            (key, ensure_finite(f"structural probability {position}", probability))
            for position, (key, probability) in enumerate(distribution)
        )

        if len({key for key, _ in held_distribution}) != len(held_distribution):
            msg = "a span-group report repeats a structural category"
            raise ValueError(msg)

        if any(not 0.0 <= probability <= 1.0 for _, probability in held_distribution):
            msg = "a span-group report has probability outside [0, 1]"
            raise ValueError(msg)

        if held_hits and not isclose(
            fsum(probability for _, probability in held_distribution),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            msg = "a nonempty span-group report distribution does not sum to one"
            raise ValueError(msg)

        if not held_hits and held_distribution:
            msg = "an empty span group has a nonempty distribution"
            raise ValueError(msg)

        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_hits", held_hits)
        object.__setattr__(self, "_distribution", held_distribution)
        object.__setattr__(
            self,
            "_sentence_count",
            len(dict.fromkeys(hit.item_id for hit in held_hits)),
        )

    @property
    def group(self) -> SpanGroup:
        """Return the reported span group."""
        return self._group

    @property
    def hits(self) -> tuple[SpanHit, ...]:
        """Return occurrences in source and token order."""
        return self._hits

    @property
    def distribution(self) -> tuple[_StructuralProbability, ...]:
        """Return structural category probabilities in first-seen order."""
        return self._distribution

    @property
    def sentence_count(self) -> int:
        """Return the number of distinct source sentences with a hit."""
        return self._sentence_count

    @property
    def occurrence_count(self) -> int:
        """Return the number of distinct span hits."""
        return len(self._hits)

    def _fields(self) -> _SpanGroupReportFields:
        """Return fields in reconstruction order."""
        return self._group, self._hits, self._distribution

    @override
    def __copy__(self) -> SpanGroupReport:
        """Reconstruct the span-group report."""
        return _restore_span_group_report(self._fields())

    @override
    def __deepcopy__(self, memo: dict[int, SpanGroupReport]) -> SpanGroupReport:
        """Reconstruct the span-group report."""
        del memo

        return _restore_span_group_report(self._fields())

    @override
    def __reduce__(self) -> _SpanGroupReportReduction:
        """Return the span-group-report reconstruction call."""
        return _restore_span_group_report, (self._fields(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same fields."""
        if type(other) is not SpanGroupReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the report fields."""
        return hash((SpanGroupReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"SpanGroupReport({self._group!r}, {self._hits!r}, {self._distribution!r})"
        )


def _restore_span_group_report(fields: _SpanGroupReportFields) -> SpanGroupReport:
    """Reconstruct a span-group report from frozen fields."""
    return SpanGroupReport(*fields)


@final
class MissingTextError(ValueError):
    """A source item without the text its byte spans address."""

    def __init__(self, item_id: ItemId) -> None:
        self.item_id = item_id
        super().__init__(f"item {item_id!r} has no source text")


@final
class MissingByteSpanError(ValueError):
    """A source token without a UTF-8 byte span."""

    def __init__(self, item_id: ItemId, position: int) -> None:
        self.item_id = item_id
        self.position = position
        super().__init__(f"item {item_id!r} token {position} has no byte span")


@final
class MissingPosteriorError(KeyError):
    """A located span without a caller-supplied type posterior."""

    def __init__(self, item_id: ItemId, span: TokenSpan) -> None:
        self.item_id = item_id
        self.span = span
        super().__init__(f"item {item_id!r} span {span!r} has no type posterior")


@final
class TokenSurfaceMismatchError(ValueError):
    """A token whose surface disagrees with its declared byte span."""

    def __init__(self, item_id: ItemId, position: int) -> None:
        self.item_id = item_id
        self.position = position
        super().__init__(
            f"item {item_id!r} token {position} disagrees with its byte span"
        )


COMPLEMENT_CONTRAST = SpanContrast(
    "declarative-versus-interrogative-complements",
    (
        SpanGroup("declarative", ("that something happened",)),
        SpanGroup(
            "interrogative",
            ("whether something happened", "which thing happened"),
        ),
    ),
)
"""The declarative-versus-interrogative complement contrast."""

FINITENESS_CONTRAST = SpanContrast(
    "finite-versus-infinitival-complements",
    (
        SpanGroup(
            "finite",
            (
                "that something happened",
                "whether something happened",
                "which thing happened",
            ),
        ),
        SpanGroup(
            "infinitival",
            (
                "to do something",
                "whether to do something",
                "which thing to do",
                "for something to happen",
            ),
        ),
    ),
)
"""The finite-versus-infinitival complement contrast."""


def _token_spans(item: Item) -> tuple[ByteSpan, ...]:
    """Validate one item's token surfaces and return their byte spans."""
    text = item.text

    if text is None:
        raise MissingTextError(item.item_id)

    spans: list[ByteSpan] = []

    for token in item.tokens:
        span = token.span

        if span is None:
            raise MissingByteSpanError(item.item_id, token.index)

        try:
            surface = span.substring(text)
        except UnicodeDecodeError as error:
            raise TokenSurfaceMismatchError(item.item_id, token.index) from error

        if surface != token.surface:
            raise TokenSurfaceMismatchError(item.item_id, token.index)

        if spans and span.start < spans[-1].end:
            msg = f"item {item.item_id!r} token byte spans overlap"
            raise ValueError(msg)

        spans.append(span)

    return tuple(spans)


def _declared_substring(group: SpanGroup, surface: str) -> str | None:
    """Return the declared substring exactly equal to a token window."""
    for substring in group.substrings:
        if surface == substring:
            return substring

    return None


def locate_span_groups(
    items: Sequence[Item],
    contrast: SpanContrast,
    /,
) -> tuple[SpanHit, ...]:
    """Locate exact declared forms at token-boundary byte spans.

    Raises
    ------
    ValueError
        If item identifiers repeat or token byte spans overlap.
    MissingTextError
        If an item has no text.
    TokenSurfaceMismatchError
        If a token's declared surface differs from its byte span.
    """
    held_items = tuple(items)

    if len({item.item_id for item in held_items}) != len(held_items):
        msg = "span-group source items repeat an item id"
        raise ValueError(msg)

    validated = tuple((item, _token_spans(item)) for item in held_items)
    hits: list[SpanHit] = []

    for group in contrast.groups:
        for item, spans in validated:
            text = item.text

            if text is None:
                raise MissingTextError(item.item_id)

            for start, first in enumerate(spans):
                for end in range(start + 1, len(spans) + 1):
                    byte_span = ByteSpan(first.start, spans[end - 1].end)
                    substring = _declared_substring(group, byte_span.substring(text))

                    if substring is not None:
                        hits.append(
                            SpanHit(
                                group,
                                item.item_id,
                                substring,
                                TokenSpan(start, end),
                                byte_span,
                            )
                        )

    return tuple(hits)


def _group_distribution(
    hits: tuple[SpanHit, ...],
    posteriors: Mapping[tuple[ItemId, TokenSpan], TokenSupertags],
) -> tuple[_StructuralProbability, ...]:
    """Average structural posterior mass across a group's occurrences."""
    if not hits:
        return ()

    masses: dict[TermKey, list[float]] = {}

    for hit in hits:
        key = hit.item_id, hit.token_span
        posterior = posteriors.get(key)

        if posterior is None:
            raise MissingPosteriorError(*key)

        if not isinstance(posterior, TokenSupertags):
            msg = (
                f"posterior for item {hit.item_id!r} span {hit.token_span!r} is invalid"
            )
            raise TypeError(msg)

        for candidate in posterior.candidates:
            masses.setdefault(candidate.category.key(), []).append(
                candidate.probability
            )

    denominator = len(hits)

    return tuple((key, fsum(values) / denominator) for key, values in masses.items())


def induced_type_report(
    items: Sequence[Item],
    contrast: SpanContrast,
    posteriors: Mapping[tuple[ItemId, TokenSpan], TokenSupertags],
    /,
) -> tuple[SpanGroupReport, ...]:
    """Report mean induced type mass for each declared span group."""
    hits = locate_span_groups(items, contrast)

    return tuple(
        SpanGroupReport(
            group,
            group_hits := tuple(hit for hit in hits if hit.group == group),
            _group_distribution(group_hits, posteriors),
        )
        for group in contrast.groups
    )
