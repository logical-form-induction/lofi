from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.token_encoder import (
    TOKEN_ENCODER_LAWS,
    TokenEncoderConformance,
    TokenEncoderSuite,
)

from lofi._checks import LawViolationError
from lofi.data import Alignment
from lofi.nn._values import Encoded, TokenEncoding
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

BUDGET: Final = 4

ITEMS: Final = (("ab", "c"), ("d",))

OVERLONG: Final = (("abc", "de", "f", "gh", "ij"),)

ROWS: Final = Site(ComponentId("piece-encoder"), Role.TOKEN_EMBEDDING)

GATES: Final = Site(ComponentId("piece-encoder"), Role.GATE)

VECTORS: Final = SiteSpec(
    ("item", "row", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)


class Pieces:
    def __init__(self, rows: int = BUDGET) -> None:
        self._rows = rows

    @property
    def output_width(self) -> int:
        return 1

    def _budget(self, items: Sequence[Sequence[str]]) -> int:
        del items

        return self._rows

    def _rows_for(self, word: str) -> int:
        return len(word)

    def _claimed(self, word: str) -> int:
        return self._rows_for(word)

    def _entries(
        self, spans: tuple[tuple[int, int] | None, ...], widest: int
    ) -> tuple[tuple[int, int] | None, ...]:
        del widest

        return spans

    def _masked(self, mask: torch.Tensor) -> torch.Tensor:
        return mask

    def _recorded(
        self, items: Sequence[Sequence[str]], widest: int
    ) -> tuple[tuple[str, ...], ...]:
        del widest

        return tuple(tuple(words) for words in items)

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        budget = self._budget(items)
        widest = max((len(words) for words in items), default=0)
        vectors = torch.zeros(len(items), budget, self.output_width)
        mask = torch.zeros(len(items), budget, dtype=torch.bool)
        alignments: list[Alignment] = []

        for item, words in enumerate(items):
            spans: list[tuple[int, int] | None] = []
            at = 0
            claimed = 0

            for position, word in enumerate(words):
                end = at + self._rows_for(word)

                if end > budget:
                    spans.extend([None] * (len(words) - position))
                    break

                vectors[item, at:end, 0] = float(ord(word[0]))
                mask[item, at:end] = True
                spans.append((claimed, claimed + self._claimed(word)))
                claimed += self._claimed(word)
                at = end

            alignments.append(Alignment(self._entries(tuple(spans), widest)))

        rows = Encoded(vectors, ("item", "row", "width"), self._masked(mask))

        return TokenEncoding(
            rows,
            tuple(alignments),
            words=self._recorded(items, widest),
        )

    def site_map(self) -> SiteMap:
        return SiteMap(ROWS.component, ComponentKind.TOKEN_ENCODER, {ROWS: VECTORS})


class _UncheckedEncoding:
    __slots__ = ("_alignments", "_encoding")

    def __init__(
        self,
        encoding: TokenEncoding,
        alignments: tuple[Alignment, ...],
    ) -> None:
        self._encoding = encoding
        self._alignments = alignments

    @property
    def vectors(self) -> Encoded:
        return self._encoding.vectors

    @property
    def lengths(self) -> torch.Tensor:
        return self._encoding.lengths

    @property
    def alignments(self) -> tuple[Alignment, ...]:
        return self._alignments

    @property
    def truncated(self) -> tuple[tuple[int, ...], ...]:
        return tuple(alignment.dropped for alignment in self._alignments)


def _unchecked(
    encoding: TokenEncoding,
    alignments: tuple[Alignment, ...],
) -> TokenEncoding:
    return cast("TokenEncoding", _UncheckedEncoding(encoding, alignments))


class Padding(Pieces):
    @override
    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        encoding = super().encode_words(items)
        widest = max((len(words) for words in items), default=0)
        alignments = tuple(
            Alignment(alignment.rows + (None,) * (widest - alignment.word_count))
            for alignment in encoding.alignments
        )

        return _unchecked(encoding, alignments)


class Counted(Pieces):
    @override
    def _rows_for(self, word: str) -> int:
        del word

        return 1

    @override
    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        encoding = super().encode_words(items)
        alignments: list[Alignment] = []

        for words, original in zip(items, encoding.alignments, strict=True):
            cursor = 0
            spans: list[tuple[int, int] | None] = []

            for word, source in zip(words, original.rows, strict=True):
                if source is None:
                    spans.append(None)
                    continue

                spans.append((cursor, cursor + len(word)))
                cursor += len(word)

            alignments.append(Alignment(tuple(spans)))

        return _unchecked(encoding, tuple(alignments))


class Overrun(Pieces):
    @override
    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        encoding = super().encode_words(items)
        alignments = tuple(
            Alignment(
                tuple(
                    None if span is None else (span[0] + BUDGET, span[1] + BUDGET)
                    for span in alignment.rows
                )
            )
            for alignment in encoding.alignments
        )

        return _unchecked(encoding, alignments)


class Filled(Pieces):
    @override
    def _masked(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(mask)


class Roomy(Pieces):
    @override
    def _budget(self, items: Sequence[Sequence[str]]) -> int:
        needed = max(
            (sum(self._rows_for(word) for word in words) for words in items),
            default=0,
        )

        return max(self._rows, needed)


class Boasting(Pieces):
    @override
    def site_map(self) -> SiteMap:
        return SiteMap(
            ROWS.component,
            ComponentKind.TOKEN_ENCODER,
            {ROWS: VECTORS, GATES: VECTORS},
        )


def _against(build: type[Pieces]) -> TokenEncoderConformance[Pieces]:
    """Return the conformance the laws run one kind of encoder through.

    Parameters
    ----------
    build
        The encoder to build.

    Returns
    -------
    TokenEncoderConformance
        The fixture, over the ragged batch and the overlong batch above.
    """
    return TokenEncoderConformance(
        build=build,
        items=ITEMS,
        overlong=OVERLONG,
        budget=BUDGET,
        sites=SiteObservation(
            build=build,
            observe=lambda subject: {ROWS: subject.encode_words(ITEMS).vectors},
            read=lambda subject: (subject.encode_words(ITEMS).vectors.tensor,),
        ),
    )


def _red(conformance: TokenEncoderConformance[Pieces]) -> frozenset[str]:
    """Return the name of every law that fails against a subject.

    Parameters
    ----------
    conformance
        The subject and what the laws need to run against it.

    Returns
    -------
    frozenset of str
        The names of the failing laws.
    """
    suite = TokenEncoderSuite()
    red: list[str] = []

    for law in TOKEN_ENCODER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, type[Pieces]], ...]] = (
    ("test_the_alignment_covers_every_input_word", Padding),
    ("test_the_row_axis_is_not_the_word_axis", Counted),
    ("test_the_row_axis_is_not_the_word_axis", Overrun),
    ("test_padding_is_marked_rather_than_left_live", Filled),
    ("test_an_item_past_the_window_is_reported_rather_than_shortened", Roomy),
    ("test_the_site_map_matches_what_instrumentation_yields", Boasting),
)


class TestATokenEncoderThatReportsWhatItDropped(TokenEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TokenEncoderConformance[Pieces]:
        return _against(Pieces)


def test_every_law_holds_of_an_encoder_that_reports_what_it_dropped() -> None:
    assert _red(_against(Pieces)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[f"{law.removeprefix('test_')}-{build.__name__}" for law, build in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: type[Pieces]) -> None:
    assert _red(_against(build)) == frozenset({law})


def test_the_overlong_batch_loses_a_word_rather_than_a_row() -> None:
    encoding = Pieces().encode_words(OVERLONG)

    assert encoding.vectors.size("row") == BUDGET
    assert encoding.truncated == ((1, 2, 3, 4),)
    assert encoding.alignments[0].word_count == len(OVERLONG[0])
    assert encoding.alignments[0].rows_for(0) == (0, 3)
    assert encoding.alignments[0].rows_for(2) is None
    assert encoding.truncation is not None
    assert encoding.truncation[0].words == ("de", "f", "gh", "ij")


def test_the_published_point_describes_the_rows_the_encoder_returns() -> None:
    subject = Pieces()
    spec = subject.site_map()[ROWS]

    rows = subject.encode_words(ITEMS).vectors

    assert spec.axes == rows.axes
    assert rows.tensor.shape[spec.position("row")] == BUDGET
    assert Access.PATCH not in spec.access
