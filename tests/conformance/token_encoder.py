from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteMapSuite, SiteObservation

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.nn._values import TokenEncoding
    from lofi.nn.ports import TokenEncoder

__all__ = [
    "TOKEN_ENCODER_LAWS",
    "TokenEncoderConformance",
    "TokenEncoderSuite",
    "ragged_items",
]

_ROW_AXIS: Final = "row"

type _Build[T: TokenEncoder] = Callable[[], T]


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def ragged_items(
    fewest: int = 1, most: int = 5
) -> st.SearchStrategy[tuple[tuple[str, ...], ...]]:
    """Return a strategy over batches of items whose words are of mixed length.

    Parameters
    ----------
    fewest : int
        The fewest items a generated batch holds.
    most : int
        The most it holds.

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        The strategy.
    """
    words = st.lists(
        st.text(alphabet="abcde", min_size=1, max_size=4), min_size=1, max_size=5
    ).map(tuple)

    return st.lists(words, min_size=fewest, max_size=most).map(tuple)


class TokenEncoderConformance[T: TokenEncoder](NamedTuple):
    """Inputs for the token-encoder conformance laws.

    Parameters
    ----------
    build : Callable[[], T]
        Return the encoder under test.
    items : Sequence[Sequence[str]]
        A ragged batch: items of different total length, so that some row of the
        batch is padding.
    overlong : Sequence[Sequence[str]]
        A batch holding an item the encoder's window cannot fit whole.
    budget : int
        How many rows the encoder was built to produce for one item.
    sites : SiteObservation[T]
        How the site map law reaches this encoder's interior.
    """

    build: Callable[[], T]
    items: Sequence[Sequence[str]]
    overlong: Sequence[Sequence[str]]
    budget: int
    sites: SiteObservation[T]


@settings(max_examples=50)
@given(items=ragged_items())
def _over_ragged_batches[T: TokenEncoder](
    build: _Build[T], items: tuple[tuple[str, ...], ...]
) -> None:
    """Check one generated ragged batch for its mask and its alignment entries.

    Parameters
    ----------
    build : _Build[T]
        Return the encoder under test.
    items : tuple[tuple[str, ...], ...]
        The generated batch.

    Raises
    ------
    LawViolationError
        If the batch comes back without a mask, if its lengths disagree with that mask,
        if an alignment disagrees with the input words or names a masked row, or if an
        omitted position is not the exact reported suffix.
    """
    encoding = build().encode_words(items)
    law = "any_ragged_batch_keeps_its_words_and_its_mask"
    mask = encoding.vectors.mask

    if mask is None:
        _fail(
            f"{items} came back without a mask",
            law=law,
        )

    if len(encoding.alignments) != len(items):
        _fail(
            f"{len(items)} items came back with {len(encoding.alignments)} alignments",
            law=law,
        )

    mask_lengths = tuple(int(length) for length in mask.sum(dim=1))
    encoded_lengths = tuple(int(length) for length in encoding.lengths)

    if encoded_lengths != mask_lengths:
        _fail(
            f"the encoded lengths are {encoded_lengths} while the mask sums to "
            f"{mask_lengths}",
            law=law,
        )

    for item, words in enumerate(items):
        alignment = encoding.alignments[item]

        if alignment.word_count != len(words):
            _fail(
                f"item {item} of {items} holds {len(words)} words and its "
                f"alignment holds {alignment.word_count} entries",
                law=law,
            )

        dropped = alignment.dropped

        if dropped:
            suffix = tuple(range(dropped[0], len(words)))

            if dropped != suffix:
                _fail(
                    f"item {item} omits {dropped} rather than the suffix {suffix}",
                    law=law,
                )

            reports = encoding.truncation

            if reports is None:
                _fail(
                    f"item {item} omits {dropped} without a truncation report",
                    law=law,
                )

            report = reports[item]
            expected_words = tuple(words[position] for position in suffix)

            if report.positions != suffix or report.words != expected_words:
                _fail(
                    f"item {item} reports {report.positions} and {report.words} "
                    f"rather than {suffix} and {expected_words}",
                    law=law,
                )

        for word, span in enumerate(alignment.rows):
            if span is None:
                continue

            start, end = span

            if end > mask.shape[1] or not bool(mask[item, start:end].all()):
                _fail(
                    f"word {word} of item {item} names rows [{start}, {end}) but "
                    "at least one is absent from the mask",
                    law=law,
                )


def _live(encoding: TokenEncoding, item: int) -> int:
    """Return how many rows of one item hold data.

    Parameters
    ----------
    encoding : TokenEncoding
        The batch to read.
    item : int
        Which item of it.

    Returns
    -------
    int
        The item's live row count, read off the mask the encoding derives it from.
    """
    return int(encoding.lengths[item])


class TokenEncoderSuite(SiteMapSuite):
    pytestmark = pytest.mark.conformance

    def test_the_alignment_covers_every_input_word[T: TokenEncoder](
        self,
        conformance: TokenEncoderConformance[T],
    ) -> None:
        """Check that every word of every item has an entry in the alignment.

        Parameters
        ----------
        conformance : TokenEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If an item's alignment holds a number of entries other than that item's word
            count, or if a report of what was dropped disagrees with the alignment it
            was read from.
        """
        encoding = conformance.build().encode_words(conformance.items)

        for item, words in enumerate(conformance.items):
            alignment = encoding.alignments[item]

            if alignment.word_count != len(words):
                _fail(
                    f"item {item} holds {len(words)} words and its alignment "
                    f"holds {alignment.word_count} entries",
                    law="the_alignment_covers_every_input_word",
                )

            if encoding.truncated[item] != alignment.dropped:
                _fail(
                    f"item {item} reports {encoding.truncated[item]} dropped "
                    f"where its alignment dropped {alignment.dropped}",
                    law="the_alignment_covers_every_input_word",
                )

    def test_the_row_axis_is_not_the_word_axis[T: TokenEncoder](
        self,
        conformance: TokenEncoderConformance[T],
    ) -> None:
        """Check that the rows the alignment gives a word are rows the item holds.

        A consumer reads a word's rows from the alignment rather than from the word's
        position, so every row an alignment names must be one the item encoded.

        Parameters
        ----------
        conformance : TokenEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the rows have no mask or an alignment refers beyond the live row range.
        """
        encoding = conformance.build().encode_words(conformance.items)
        mask = encoding.vectors.mask

        if mask is None:
            _fail(
                "the rows came back without a mask, so no row is known to be live",
                law="the_row_axis_is_not_the_word_axis",
            )

        rows = encoding.vectors.size(_ROW_AXIS)

        for item, _ in enumerate(conformance.items):
            alignment = encoding.alignments[item]
            live = _live(encoding, item)

            for word, span in enumerate(alignment.rows):
                if span is None:
                    continue

                start, end = span

                if end > rows or not bool(mask[item, start:end].all()):
                    _fail(
                        f"word {word} of item {item} is aligned to rows "
                        f"[{start}, {end}) and the item holds {live} live rows "
                        f"of the {rows} on the row axis",
                        law="the_row_axis_is_not_the_word_axis",
                    )

    def test_padding_is_marked_rather_than_left_live[T: TokenEncoder](
        self,
        conformance: TokenEncoderConformance[T],
    ) -> None:
        """Check that the mask separates a padded row from an encoded one.

        Parameters
        ----------
        conformance : TokenEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the rows have no mask or a ragged batch has no masked padding.
        """
        encoding = conformance.build().encode_words(conformance.items)
        mask = encoding.vectors.mask

        if mask is None:
            _fail(
                "the rows of a ragged batch came back without a mask",
                law="padding_is_marked_rather_than_left_live",
            )

        rows = encoding.vectors.size(_ROW_AXIS)
        live = {_live(encoding, item) for item, _ in enumerate(conformance.items)}

        if live == {rows}:
            _fail(
                f"every item of a ragged batch holds all {rows} rows, so no row "
                "is marked as padding",
                law="padding_is_marked_rather_than_left_live",
            )

    def test_an_item_past_the_window_is_reported_rather_than_shortened[T: TokenEncoder](
        self,
        conformance: TokenEncoderConformance[T],
    ) -> None:
        """Check the golden case of a batch the encoder's window cannot fit.

        Parameters
        ----------
        conformance : TokenEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the rows outrun the declared budget, which is a window that silently
            grew, or if nothing at all is reported as dropped, which is a sentence that
            was silently cut.
        """
        encoding = conformance.build().encode_words(conformance.overlong)
        rows = encoding.vectors.size(_ROW_AXIS)

        if rows > conformance.budget:
            _fail(
                f"a window of {conformance.budget} rows returned {rows} of them",
                law="an_item_past_the_window_is_reported_rather_than_shortened",
            )

        if not any(encoding.truncated):
            _fail(
                f"{conformance.overlong} does not fit {conformance.budget} rows "
                "and no word is reported dropped",
                law="an_item_past_the_window_is_reported_rather_than_shortened",
            )

    def test_any_ragged_batch_keeps_its_words_and_its_mask[T: TokenEncoder](
        self,
        conformance: TokenEncoderConformance[T],
    ) -> None:
        """Check alignment, masks, lengths, and suffix reports over generated batches.

        Parameters
        ----------
        conformance : TokenEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a generated batch violates an alignment, mask, length, or truncation
            report law.
        """
        _over_ragged_batches(conformance.build)


TOKEN_ENCODER_LAWS: Final = (
    TokenEncoderSuite.test_the_alignment_covers_every_input_word,
    TokenEncoderSuite.test_the_row_axis_is_not_the_word_axis,
    TokenEncoderSuite.test_padding_is_marked_rather_than_left_live,
    TokenEncoderSuite.test_an_item_past_the_window_is_reported_rather_than_shortened,
    TokenEncoderSuite.test_the_site_map_matches_what_instrumentation_yields,
)
