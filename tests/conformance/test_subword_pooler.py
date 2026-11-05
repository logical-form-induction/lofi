from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.subword_pooler import (
    SUBWORD_POOLER_LAWS,
    SubwordPoolerConformance,
    SubwordPoolerSuite,
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
    from collections.abc import Mapping

    from torch import Tensor

ROW_AXES: Final = ("item", "row", "width")

POOLED_AXES: Final = ("item", "word", "width")

WORDS: Final = 3

VECTORS: Final = torch.tensor(
    [
        [[1.0], [2.0], [3.0], [0.0], [0.0]],
        [[7.0], [0.0], [0.0], [0.0], [0.0]],
    ]
)

LIVE: Final = torch.tensor(
    [
        [True, True, True, False, False],
        [True, False, False, False, False],
    ]
)

ALIGNMENTS: Final = (
    Alignment(((0, 2), (2, 3), None)),
    Alignment(((0, 1),)),
)

ENCODING: Final = TokenEncoding(
    Encoded(VECTORS, ROW_AXES, LIVE),
    ALIGNMENTS,
    words=(("multi", "piece", "dropped"), ("kept",)),
)

POOLER: Final = ComponentId("mean-pooler")

WORD_SPEC: Final = SiteSpec(
    POOLED_AXES,
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)


class MeanPooler:
    """A pooler averaging the rows its alignment names for each word."""

    @property
    def output_width(self) -> int:
        return 1

    def _extent(self, encoding: TokenEncoding) -> int:
        return max(
            (alignment.word_count for alignment in encoding.alignments), default=0
        )

    def _span(self, alignment: Alignment, word: int) -> tuple[int, int] | None:
        return alignment.rows_for(word)

    def _keeps(self, alignment: Alignment, word: int) -> bool:
        return word < alignment.word_count and alignment.rows_for(word) is not None

    def _gather(self, rows: Tensor, live: Tensor, span: tuple[int, int]) -> Tensor:
        del live
        start, end = span

        return rows[start:end].mean(dim=0)

    def pool(self, encoding: TokenEncoding, /) -> Encoded:
        vectors = encoding.vectors
        rows = vectors.tensor
        live = vectors.mask
        extent = self._extent(encoding)
        pooled = torch.zeros(
            len(encoding.alignments),
            extent,
            vectors.size("width"),
            dtype=rows.dtype,
        )
        kept = torch.zeros(len(encoding.alignments), extent, dtype=torch.bool)

        for item, alignment in enumerate(encoding.alignments):
            for word in range(extent):
                if not self._keeps(alignment, word):
                    continue

                kept[item, word] = True
                span = self._span(alignment, word)

                if span is None or live is None:
                    continue

                pooled[item, word] = self._gather(rows[item], live[item], span)

        return Encoded(pooled, POOLED_AXES, kept)

    def site_map(self) -> SiteMap:
        return SiteMap(
            POOLER,
            ComponentKind.SUBWORD_POOLER,
            {
                Site(POOLER, Role.WORD_VECTOR): WORD_SPEC,
            },
        )

    def instrument(self, encoding: TokenEncoding) -> Mapping[Site, Encoded]:
        return {Site(POOLER, Role.WORD_VECTOR): self.pool(encoding)}


class Padded(MeanPooler):
    @override
    def _extent(self, encoding: TokenEncoding) -> int:
        return super()._extent(encoding) + 1


class Filling(MeanPooler):
    @override
    def _keeps(self, alignment: Alignment, word: int) -> bool:
        return word < alignment.word_count


class Leaky(MeanPooler):
    @override
    def _gather(self, rows: Tensor, live: Tensor, span: tuple[int, int]) -> Tensor:
        return super()._gather(rows, live, span) + rows[~live].sum(dim=0)


class Positional(MeanPooler):
    @override
    def _span(self, alignment: Alignment, word: int) -> tuple[int, int] | None:
        if alignment.rows_for(word) is None:
            return None

        return word, word + 1


class Narrow(MeanPooler):
    @property
    @override
    def output_width(self) -> int:
        return 2


class Silent(MeanPooler):
    @override
    def instrument(self, encoding: TokenEncoding) -> Mapping[Site, Encoded]:
        del encoding

        return {}


def _against(build: type[MeanPooler]) -> SubwordPoolerConformance[MeanPooler]:
    """Return the conformance the laws run one kind of pooler through.

    Parameters
    ----------
    build
        The pooler to build.

    Returns
    -------
    SubwordPoolerConformance
        The fixture, over the ragged batch above.
    """
    return SubwordPoolerConformance(
        build=build,
        encoding=ENCODING,
        sites=SiteObservation(
            build=build,
            observe=lambda subject: subject.instrument(ENCODING),
            read=lambda subject: (subject.pool(ENCODING).tensor,),
        ),
    )


def _red(conformance: SubwordPoolerConformance[MeanPooler]) -> frozenset[str]:
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
    suite = SubwordPoolerSuite()
    red: list[str] = []

    for law in SUBWORD_POOLER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, type[MeanPooler]], ...]] = (
    ("test_the_vectors_run_over_an_item_a_word_and_a_width", Padded),
    ("test_a_word_no_row_encoded_is_masked_off_rather_than_filled", Filling),
    ("test_a_row_the_encoder_masked_off_reaches_no_word", Leaky),
    ("test_a_word_is_pooled_from_the_rows_its_alignment_names", Positional),
    ("test_the_declared_width_is_the_trailing_axis", Narrow),
    ("test_the_site_map_matches_what_instrumentation_yields", Silent),
)


class TestASubwordPoolerThatAveragesTheRowsItWasGiven(SubwordPoolerSuite):
    @pytest.fixture
    def conformance(self) -> SubwordPoolerConformance[MeanPooler]:
        return _against(MeanPooler)


def test_every_law_holds_of_a_pooler_that_averages_the_rows_it_was_given() -> None:
    assert _red(_against(MeanPooler)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: type[MeanPooler]) -> None:
    assert _red(_against(build)) == frozenset({law})


def test_the_six_laws_are_stated_once_each() -> None:
    assert len(SUBWORD_POOLER_LAWS) == len(_BROKEN)
    assert len({law.__name__ for law in SUBWORD_POOLER_LAWS}) == len(
        SUBWORD_POOLER_LAWS
    )
    assert {law.__name__ for law in SUBWORD_POOLER_LAWS} == {law for law, _ in _BROKEN}


def test_a_word_is_the_average_of_the_rows_its_alignment_names() -> None:
    pooled = MeanPooler().pool(ENCODING)

    assert pooled.size("word") == WORDS
    assert float(pooled.tensor[0, 0, 0]) == pytest.approx(1.5)
    assert float(pooled.tensor[0, 1, 0]) == pytest.approx(3.0)
    assert float(pooled.tensor[1, 0, 0]) == pytest.approx(7.0)


def test_the_words_no_row_encoded_are_the_ones_the_mask_drops() -> None:
    mask = MeanPooler().pool(ENCODING).mask

    assert mask is not None
    assert mask.tolist() == [[True, True, False], [True, False, False]]
    assert ENCODING.truncated == ((2,), ())
