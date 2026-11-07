from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
import torch
from tests.conformance._sites import SiteMapSuite, SiteObservation

from lofi._checks import LawViolationError
from lofi.nn._values import Encoded, TokenEncoding

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.ports import SubwordPooler

__all__ = [
    "SUBWORD_POOLER_LAWS",
    "SubwordPoolerConformance",
    "SubwordPoolerSuite",
]

_ITEM_AXIS: Final = "item"

_ROW_AXIS: Final = "row"

_WORD_AXIS: Final = "word"

_WIDTH_AXIS: Final = "width"

_POOLED_AXES: Final = (_ITEM_AXIS, _WORD_AXIS, _WIDTH_AXIS)

_MASK_AXES: Final = (_ITEM_AXIS, _WORD_AXIS)

_SHIFT: Final = 5.0


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


class SubwordPoolerConformance[T: SubwordPooler](NamedTuple):
    """Inputs for the subword-pooler conformance laws.

    Parameters
    ----------
    build : Callable[[], T]
        Return a fresh pooler.
    encoding : TokenEncoding
        The rows to pool and the alignment saying which rows hold which word.
    sites : SiteObservation[T]
        How the site map law reaches this pooler's interior.
    """

    build: Callable[[], T]
    encoding: TokenEncoding
    sites: SiteObservation[T]


def _at(pooled: Encoded, item: int, word: int) -> Tensor:
    """Return the vector one word of one item was gathered into.

    Parameters
    ----------
    pooled : Encoded
        What a pass returned.
    item : int
        Which item of the batch.
    word : int
        Which word of that item.

    Returns
    -------
    torch.Tensor
        The vector at that position, addressed by axis name.
    """
    index: list[int | slice] = [slice(None)] * pooled.tensor.ndim
    index[pooled.axis(_ITEM_AXIS)] = item
    index[pooled.axis(_WORD_AXIS)] = word

    return pooled.tensor[tuple(index)]


def _keeps(pooled: Encoded, item: int, word: int) -> bool:
    """Return whether the mask marks one word of one item as pooled.

    Parameters
    ----------
    pooled : Encoded
        What a pass returned.
    item : int
        Which item of the batch.
    word : int
        Which word of that item.

    Returns
    -------
    bool
        Whether the position holds a vector.
    """
    mask = pooled.mask

    if mask is None or pooled.axis(_WORD_AXIS) >= mask.ndim:
        return True

    index: list[int | slice] = [slice(None)] * mask.ndim
    index[pooled.axis(_ITEM_AXIS)] = item
    index[pooled.axis(_WORD_AXIS)] = word

    return bool(mask[tuple(index)])


def _live_values(pooled: Encoded) -> Tensor:
    """Return the pooled vectors at the positions the mask keeps.

    Parameters
    ----------
    pooled : Encoded
        What a pass returned.

    Returns
    -------
    torch.Tensor
        Every kept vector, in order.
    """
    if pooled.mask is None:
        return pooled.tensor

    return pooled.tensor[pooled.mask]


def _shifted(encoding: TokenEncoding, where: Tensor) -> TokenEncoding:
    """Return the batch again, with the rows a mask selects moved.

    Parameters
    ----------
    encoding : TokenEncoding
        The batch to move rows in.
    where : Tensor
        True at the values to move, over the outermost axes of the rows or over all of
        them.

    Returns
    -------
    TokenEncoding
        The batch, with the same mask and the same alignments, holding the moved rows.
    """
    vectors = encoding.vectors
    moved = vectors.tensor.clone()
    moved[where] += _SHIFT

    return TokenEncoding(
        Encoded(moved, vectors.axes, vectors.mask),
        encoding.alignments,
        words=encoding.words,
    )


def _a_word_among_others(encoding: TokenEncoding) -> tuple[int, int] | None:
    """Return an item and one of its encoded words that has an encoded neighbor.

    Parameters
    ----------
    encoding : TokenEncoding
        The batch to look through.

    Returns
    -------
    tuple of int or None
        The item and the word, or None where no item holds two words a row encoded, in
        which case nothing distinguishes an alignment from a count.
    """
    for item, alignment in enumerate(encoding.alignments):
        encoded = [word for word, span in enumerate(alignment.rows) if span is not None]

        if len(encoded) > 1:
            return item, encoded[0]

    return None


class SubwordPoolerSuite(SiteMapSuite):
    pytestmark = pytest.mark.conformance

    def test_the_vectors_run_over_an_item_a_word_and_a_width[T: SubwordPooler](
        self,
        conformance: SubwordPoolerConformance[T],
    ) -> None:
        """Check that the word axis holds the words of the batch and no more.

        Parameters
        ----------
        conformance : SubwordPoolerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the vectors name axes other than the declared ones, hold a number of
            items other than the batch's, or run over a number of words other than the
            widest item's word count.
        """
        pooled = conformance.build().pool(conformance.encoding)
        alignments = conformance.encoding.alignments

        if pooled.axes != _POOLED_AXES:
            _fail(
                f"pooling returned {pooled.axes} where {_POOLED_AXES} is declared",
                law="the_vectors_run_over_an_item_a_word_and_a_width",
            )

        items = conformance.encoding.vectors.size(_ITEM_AXIS)

        if pooled.size(_ITEM_AXIS) != items:
            _fail(
                f"pooling returned {pooled.size(_ITEM_AXIS)} items for a batch "
                f"of {items}",
                law="the_vectors_run_over_an_item_a_word_and_a_width",
            )

        widest = max((alignment.word_count for alignment in alignments), default=0)

        if pooled.size(_WORD_AXIS) != widest:
            _fail(
                f"pooling returned {pooled.size(_WORD_AXIS)} words where the "
                f"longest item of the batch holds {widest}",
                law="the_vectors_run_over_an_item_a_word_and_a_width",
            )

    def test_a_word_no_row_encoded_is_masked_off_rather_than_filled[T: SubwordPooler](
        self,
        conformance: SubwordPoolerConformance[T],
    ) -> None:
        """Check that the mask marks exactly the words a row encoded.

        Parameters
        ----------
        conformance : SubwordPoolerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture's batch holds no dropped word, so that nothing separates a
            masked word from a filled one, if the vectors have no mask over the item
            and word axes, or if a position is kept where its alignment names no row, or
            dropped where it names one.
        """
        alignments = conformance.encoding.alignments
        dropped = sum(len(alignment.dropped) for alignment in alignments)

        if not dropped:
            _fail(
                "no word of the fixture's batch was left unencoded, so nothing "
                "separates a masked word from a filled one",
                law="a_word_no_row_encoded_is_masked_off_rather_than_filled",
            )

        pooled = conformance.build().pool(conformance.encoding)

        if pooled.mask is None:
            _fail(
                "the pooled vectors came back without a mask, so nothing says "
                "which words a row encoded",
                law="a_word_no_row_encoded_is_masked_off_rather_than_filled",
            )

        if pooled.mask_axes != _MASK_AXES:
            _fail(
                f"the mask covers {pooled.mask_axes} where {_MASK_AXES} is declared",
                law="a_word_no_row_encoded_is_masked_off_rather_than_filled",
            )

        for item, alignment in enumerate(alignments):
            for word in range(pooled.size(_WORD_AXIS)):
                encoded = (
                    word < alignment.word_count and alignment.rows_for(word) is not None
                )

                kept = _keeps(pooled, item, word)

                if kept != encoded:
                    named = "one row" if encoded else "no row"
                    _fail(
                        f"{named} encodes word {word} of item {item} and the "
                        f"mask reads {kept}",
                        law="a_word_no_row_encoded_is_masked_off_rather_than_filled",
                    )

    def test_a_row_the_encoder_masked_off_reaches_no_word[T: SubwordPooler](
        self,
        conformance: SubwordPoolerConformance[T],
    ) -> None:
        """Check that moving the padded rows moves nothing a word was pooled into.

        Parameters
        ----------
        conformance : SubwordPoolerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture's batch holds no padded row, so that the law has nothing to
            move, if moving the padded rows changes which words were pooled, or if it
            changes a vector a word was pooled into.
        """
        mask = conformance.encoding.vectors.mask

        if mask is None:
            _fail(
                "the fixture's batch came without a mask, so no row of it reads "
                "as padding",
                law="a_row_the_encoder_masked_off_reaches_no_word",
            )

        if not bool((~mask).any()):
            _fail(
                "every row of the fixture's batch holds data, so the law has no "
                "padding to move",
                law="a_row_the_encoder_masked_off_reaches_no_word",
            )

        subject = conformance.build()
        before = subject.pool(conformance.encoding)
        after = subject.pool(_shifted(conformance.encoding, ~mask))

        if not _same_mask(before, after):
            _fail(
                "moving the padded rows changed which words were pooled",
                law="a_row_the_encoder_masked_off_reaches_no_word",
            )

        if not torch.equal(_live_values(before), _live_values(after)):
            _fail(
                "moving the padded rows moved a vector a word was pooled into, "
                "so a word reads rows the encoder marked as padding",
                law="a_row_the_encoder_masked_off_reaches_no_word",
            )

    def test_a_word_is_pooled_from_the_rows_its_alignment_names[T: SubwordPooler](
        self,
        conformance: SubwordPoolerConformance[T],
    ) -> None:
        """Check that moving one word's rows moves that word and no other.

        Parameters
        ----------
        conformance : SubwordPoolerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture's batch holds no item with two encoded words, if moving a
            word's rows changes which words were pooled, if it leaves that word's own
            vector alone, or if it moves another word's.
        """
        chosen = _a_word_among_others(conformance.encoding)

        if chosen is None:
            _fail(
                "no item of the fixture's batch holds two words a row encoded, "
                "so nothing separates an alignment from a running count",
                law="a_word_is_pooled_from_the_rows_its_alignment_names",
            )

        item, word = chosen
        span = conformance.encoding.alignments[item].rows_for(word)

        if span is None:
            _fail(
                f"word {word} of item {item} was chosen as one a row encoded and "
                "its alignment names none",
                law="a_word_is_pooled_from_the_rows_its_alignment_names",
            )

        subject = conformance.build()
        before = subject.pool(conformance.encoding)
        after = subject.pool(
            _shifted(conformance.encoding, _rows_of(conformance.encoding, item, span))
        )

        if not _same_mask(before, after):
            _fail(
                f"moving the rows of word {word} of item {item} changed which "
                "words were pooled",
                law="a_word_is_pooled_from_the_rows_its_alignment_names",
            )

        if torch.equal(_at(before, item, word), _at(after, item, word)):
            _fail(
                f"moving the rows {span} that encode word {word} of item {item} "
                "left that word's vector alone",
                law="a_word_is_pooled_from_the_rows_its_alignment_names",
            )

        for other_item in range(before.size(_ITEM_AXIS)):
            for other_word in range(before.size(_WORD_AXIS)):
                if (other_item, other_word) == (item, word):
                    continue

                if not _keeps(before, other_item, other_word):
                    continue

                if not torch.equal(
                    _at(before, other_item, other_word),
                    _at(after, other_item, other_word),
                ):
                    _fail(
                        f"moving the rows of word {word} of item {item} moved "
                        f"word {other_word} of item {other_item}",
                        law="a_word_is_pooled_from_the_rows_its_alignment_names",
                    )

    def test_the_declared_width_is_the_trailing_axis[T: SubwordPooler](
        self,
        conformance: SubwordPoolerConformance[T],
    ) -> None:
        """Check that the vectors are as wide as the pooler says they are.

        Parameters
        ----------
        conformance : SubwordPoolerConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the pooled vectors are of a width other than the declared one, which a
            consumer that allocated against the declaration has already sized itself to.
        """
        subject = conformance.build()
        pooled = subject.pool(conformance.encoding)

        if pooled.size(_WIDTH_AXIS) != subject.output_width:
            _fail(
                f"pooling returned a width of {pooled.size(_WIDTH_AXIS)} where "
                f"{subject.output_width} is declared",
                law="the_declared_width_is_the_trailing_axis",
            )


def _same_mask(before: Encoded, after: Encoded) -> bool:
    """Return whether two passes agree about which words were pooled.

    Parameters
    ----------
    before : Encoded
        What one pass returned.
    after : Encoded
        What the other returned.

    Returns
    -------
    bool
        Whether both have an agreeing mask, or neither has one.
    """
    if before.mask is None or after.mask is None:
        return before.mask is None and after.mask is None

    return torch.equal(before.mask, after.mask)


def _rows_of(encoding: TokenEncoding, item: int, span: tuple[int, int]) -> Tensor:
    """Return a mask selecting the rows of one item, over the item and row axes.

    Parameters
    ----------
    encoding : TokenEncoding
        The batch the rows belong to.
    item : int
        Which item of it.
    span : tuple[int, int]
        The half-open interval of rows to select.

    Returns
    -------
    torch.Tensor
        A boolean tensor over the item and row axes, true only on that span.
    """
    vectors = encoding.vectors
    chosen = torch.zeros_like(vectors.tensor, dtype=torch.bool)
    start, end = span
    index: list[int | slice] = [slice(None)] * chosen.ndim
    index[vectors.axis(_ITEM_AXIS)] = item
    index[vectors.axis(_ROW_AXIS)] = slice(start, end)
    chosen[tuple(index)] = True

    return chosen


SUBWORD_POOLER_LAWS: Final = (
    SubwordPoolerSuite.test_the_vectors_run_over_an_item_a_word_and_a_width,
    SubwordPoolerSuite.test_a_word_no_row_encoded_is_masked_off_rather_than_filled,
    SubwordPoolerSuite.test_a_row_the_encoder_masked_off_reaches_no_word,
    SubwordPoolerSuite.test_a_word_is_pooled_from_the_rows_its_alignment_names,
    SubwordPoolerSuite.test_the_declared_width_is_the_trailing_axis,
    SubwordPoolerSuite.test_the_site_map_matches_what_instrumentation_yields,
)
