from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, cast

import pytest
import torch
from tests.conformance._sites import SiteMapSuite, SiteObservation

from lofi._checks import LawViolationError
from lofi.nn._values import Encoded, LogDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.nn.ports import SoftTermEncoder, TermEncoder
    from lofi.terms import BatchEncoding

__all__ = [
    "SOFT_TERM_ENCODER_LAWS",
    "TERM_ENCODER_LAWS",
    "SoftTermEncoderSuite",
    "TermEncoderConformance",
    "TermEncoderSuite",
]

_TERM_AXES: Final = ("term", "width")

_NODE_AXES: Final = ("term", "slot", "width")

_SOFT_ATOL: Final = 1e-6

_PROBE: Final = "conformance_probe"


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


def _stay[T](subject: T, _placement: str) -> T:
    """Keep an unplaced subject at its current location.

    Parameters
    ----------
    subject : T
        The subject to place.
    _placement : str
        Where it was asked to go.

    Returns
    -------
    The subject, unchanged.
    """
    return subject


def _no_submodules[T](_subject: T) -> tuple[str, ...]:
    """Report that a subject has no submodules.

    Parameters
    ----------
    _subject : T
        The encoder to inspect.

    Returns
    -------
    tuple of str
        The empty tuple.
    """
    return ()


class TermEncoderConformance[T: TermEncoder](NamedTuple):
    """Inputs for the term-encoder conformance laws.

    Attributes
    ----------
    build : Callable[[], T]
        Return a fresh encoder.
    terms : BatchEncoding
        The linearized batch the laws encode.
    choices : Callable[[BatchEncoding], tuple[tuple[int, ...], ...]]
        Return the payload-aware choice code at every term slot.
    choice_width : int
        The number of choices available to the one-hot law.
    sites : SiteObservation[T]
        The observations used to check the site map.
    submodules : Callable[[T], tuple[str, ...]]
        The names of the modules the encoder owns, in a stable order.
    placements : tuple[str, ...]
        Where the encoder runs, the first being where it is built.
    place : Callable[[T, str], T]
        The operation that moves an encoder to a placement.
    """

    build: Callable[[], T]
    terms: BatchEncoding
    choices: Callable[[BatchEncoding], tuple[tuple[int, ...], ...]]
    choice_width: int
    sites: SiteObservation[T]
    submodules: Callable[[T], tuple[str, ...]] = _no_submodules
    placements: tuple[str, ...] = ("cpu",)
    place: Callable[[T, str], T] = _stay


def one_hot(codes: tuple[tuple[int, ...], ...], width: int, /) -> LogDistribution:
    """Return the distribution concentrated on one batch of terms.

    Parameters
    ----------
    codes : tuple[tuple[int, ...], ...]
        The payload-aware choice code at each term slot.
    width : int
        How many choices the codes are drawn from.

    Returns
    -------
    LogDistribution
        Rank 3 over the axes ``("term", "slot", "choice")`` at double precision,
        normalized over the choice axis, holding zero at each term's own symbol and
        negative infinity elsewhere.
    """
    held = torch.tensor(codes, dtype=torch.int64)
    mass = torch.nn.functional.one_hot(held, width).double()

    return LogDistribution(
        mass.log(), ("term", "slot", "choice"), normalized_over="choice"
    )


class TermEncoderSuite(SiteMapSuite):
    pytestmark = pytest.mark.conformance

    def test_rank_and_axes_are_as_declared[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that term and node encodings have their declared axes.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If either encoding names axes other than the declared ones, or sizes the
            term or slot axis differently from the batch it was handed.
        """
        subject = conformance.build()
        terms = subject.encode(conformance.terms)
        nodes = subject.encode_nodes(conformance.terms)

        for what, encoded, axes in (
            ("encode", terms, _TERM_AXES),
            ("encode_nodes", nodes, _NODE_AXES),
        ):
            if encoded.axes != axes:
                _fail(
                    f"{what} returned {encoded.axes} where {axes} is declared",
                    law="rank_and_axes_are_as_declared",
                )

            if encoded.size("term") != len(conformance.terms.rows):
                _fail(
                    f"{what} returned {encoded.size('term')} rows for a batch of "
                    f"{len(conformance.terms.rows)} terms",
                    law="rank_and_axes_are_as_declared",
                )

        if nodes.size("slot") != conformance.terms.width:
            _fail(
                f"encode_nodes returned {nodes.size('slot')} slots for a "
                f"linearization {conformance.terms.width} slots wide",
                law="rank_and_axes_are_as_declared",
            )

    def test_the_declared_width_is_the_trailing_axis[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that each encoding uses the declared output width.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the term or node encoding is of a width other than the declared one.
        """
        subject = conformance.build()
        for what, encoded in (
            ("encode", subject.encode(conformance.terms)),
            ("encode_nodes", subject.encode_nodes(conformance.terms)),
        ):
            if encoded.size("width") != subject.output_width:
                _fail(
                    f"{what} returned a width of {encoded.size('width')} where "
                    f"{subject.output_width} is declared",
                    law="the_declared_width_is_the_trailing_axis",
                )

    def test_the_signature_accepts_no_extra_keyword[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that encoding methods reject undeclared keywords.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If any of the three encoding methods accepts an undeclared keyword.
        """
        subject = conformance.build()
        offered: Mapping[str, int] = {_PROBE: 0}
        calls: tuple[
            tuple[
                str,
                Callable[..., Encoded],
                BatchEncoding,
                Mapping[str, BatchEncoding],
            ],
            ...,
        ] = (
            (
                "encode",
                cast("Callable[..., Encoded]", subject.encode),
                conformance.terms,
                {},
            ),
            (
                "encode_nodes",
                cast("Callable[..., Encoded]", subject.encode_nodes),
                conformance.terms,
                {},
            ),
        )

        for what, method, argument, declared in calls:
            try:
                method(argument, **declared, **offered)

            except TypeError:
                continue

            _fail(
                f"{what} accepted a keyword named {_PROBE}",
                law="the_signature_accepts_no_extra_keyword",
            )

    def test_encoding_is_pure[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that repeated encoding preserves results and module state.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If two calls disagree or if the encoder gained a module between the calls.
        """
        subject = conformance.build()
        before = conformance.submodules(subject)
        first = subject.encode(conformance.terms)
        second = subject.encode(conformance.terms)

        if not torch.equal(first.tensor, second.tensor):
            _fail(
                "two calls over one batch returned different rows",
                law="encoding_is_pure",
            )

        after = conformance.submodules(subject)

        if after != before:
            _fail(
                f"the encoder held {before} before the calls and {after} after",
                law="encoding_is_pure",
            )

    def test_placement_does_not_change_the_encoding[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that placement preserves encoded values.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture names no placement, or if the rows read at one placement
            differ from those read at the first.
        """
        if not conformance.placements:
            _fail(
                "the fixture names no placement to run on",
                law="placement_does_not_change_the_encoding",
            )

        home = conformance.placements[0]
        subject = conformance.place(conformance.build(), home)
        baseline = subject.encode(conformance.terms).tensor.cpu()

        for placement in (*conformance.placements[1:], home):
            subject = conformance.place(subject, placement)
            reading = subject.encode(conformance.terms).tensor.cpu()

            if not torch.equal(reading, baseline):
                _fail(
                    f"the rows read on {home} and the rows read on {placement} "
                    "are not the same numbers",
                    law="placement_does_not_change_the_encoding",
                )

    def test_a_node_no_term_occupies_is_masked[T: TermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check that the node mask equals term occupancy.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the node encoding has no mask, or one that disagrees with the
            occupancy of the batch it was handed.
        """
        subject = conformance.build()
        nodes = subject.encode_nodes(conformance.terms)
        occupied = torch.tensor(conformance.terms.mask)

        if nodes.mask is None:
            _fail(
                "the node encoding of a ragged batch came back without a mask",
                law="a_node_no_term_occupies_is_masked",
            )

        if not torch.equal(nodes.mask, occupied):
            _fail(
                f"the node mask reads {nodes.mask.tolist()} where the "
                f"linearization is occupied at {occupied.tolist()}",
                law="a_node_no_term_occupies_is_masked",
            )


class SoftTermEncoderSuite:
    """The agreement law for encoders supporting distributions over choices."""

    pytestmark = pytest.mark.conformance

    def test_a_one_hot_distribution_encodes_as_the_term[T: SoftTermEncoder](
        self,
        conformance: TermEncoderConformance[T],
    ) -> None:
        """Check the soft encoding against the hard encoding it relaxes.

        Parameters
        ----------
        conformance : TermEncoderConformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If encoding the distribution concentrated on a batch of terms differs from
            encoding those terms by more than 1e-6.
        """
        subject = conformance.build()
        hard = subject.encode(conformance.terms)
        soft = subject.encode_soft(
            one_hot(conformance.choices(conformance.terms), conformance.choice_width),
            structure=conformance.terms,
        )

        if soft.axes != hard.axes:
            _fail(
                f"the soft encoding is over {soft.axes} and the hard encoding "
                f"over {hard.axes}",
                law="a_one_hot_distribution_encodes_as_the_term",
            )

        if (soft.mask is None) != (hard.mask is None) or (
            soft.mask is not None
            and hard.mask is not None
            and not torch.equal(soft.mask, hard.mask)
        ):
            _fail(
                "the soft and hard encodings disagree on which terms are present",
                law="a_one_hot_distribution_encodes_as_the_term",
            )

        if not torch.allclose(soft.tensor, hard.tensor, atol=_SOFT_ATOL):
            worst = float(soft.tensor.sub(hard.tensor).abs().max().detach())

            _fail(
                f"the soft and the hard encoding differ by {worst}, which is "
                f"more than {_SOFT_ATOL}",
                law="a_one_hot_distribution_encodes_as_the_term",
            )


TERM_ENCODER_LAWS: Final = (
    TermEncoderSuite.test_rank_and_axes_are_as_declared,
    TermEncoderSuite.test_the_declared_width_is_the_trailing_axis,
    TermEncoderSuite.test_the_signature_accepts_no_extra_keyword,
    TermEncoderSuite.test_encoding_is_pure,
    TermEncoderSuite.test_placement_does_not_change_the_encoding,
    TermEncoderSuite.test_a_node_no_term_occupies_is_masked,
    TermEncoderSuite.test_the_site_map_matches_what_instrumentation_yields,
)

SOFT_TERM_ENCODER_LAWS: Final = (
    SoftTermEncoderSuite.test_a_one_hot_distribution_encodes_as_the_term,
)
