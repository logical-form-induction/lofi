"""Conformance laws for the independent term-decoder protocols."""

from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, Protocol, cast

import pytest
import torch

from lofi._checks import LawViolationError
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.ports import (
    FrontierDecoder,
    MarginalDecoder,
    TeacherForcedDecoder,
)
from lofi.nn.strategies import score

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.terms import BatchEncoding, ChoiceIndex, HoleContext, Sort

__all__ = [
    "DECODER_AGREEMENT_LAWS",
    "FRONTIER_DECODER_LAWS",
    "MARGINAL_DECODER_LAWS",
    "TEACHER_FORCED_DECODER_LAWS",
    "DecoderAgreementSuite",
    "DecoderConformance",
    "FrontierDecoderSuite",
    "MarginalDecoderSuite",
    "TeacherForcedDecoderSuite",
]

_ROW_AXES: Final = ("term", "slot", "choice")
_FRONTIER_AXES: Final = ("choice",)
_OLD_SURFACES: Final = ("distribution", "draw", "score")
_AGREEMENT_RTOL: Final = 1e-6
_AGREEMENT_ATOL: Final = 1e-7


class _CompleteDecoder[S](
    FrontierDecoder[S], TeacherForcedDecoder, MarginalDecoder, Protocol
):
    """Combine the narrow protocols only for cross-protocol laws."""


class DecoderConformance[T](NamedTuple):
    """Inputs shared by the decoder protocol laws.

    Attributes
    ----------
    build : Callable[[], T]
        Return a fresh decoder.
    inputs : Encoded
        One vector per teacher term over ``("term", "width")``.
    terms : BatchEncoding
        One teacher term per input vector.
    index : ChoiceIndex
        The payload-aware choice inventory the decoder reports rows over.
    root : HoleContext
        The root context for the first input's frontier.
    root_sort : Sort
        The represented sort marginal decoding starts from.
    slots : int
        The fixed number of slots reported by marginal decoding.
    """

    build: Callable[[], T]
    inputs: Encoded
    terms: BatchEncoding
    index: ChoiceIndex
    root: HoleContext
    root_sort: Sort
    slots: int


def _fail(detail: str, *, law: str) -> NoReturn:
    message = f"{law}: {detail}"
    raise LawViolationError(message)


def _check_rows(
    rows: LogDistribution,
    conformance: DecoderConformance[object],
    /,
    *,
    what: str,
) -> None:
    expected = (
        len(conformance.terms.rows),
        conformance.slots,
        conformance.index.width,
    )
    if rows.axes != _ROW_AXES:
        _fail(
            f"{what} returned axes {rows.axes} where {_ROW_AXES} are declared",
            law="choice_rows_have_the_declared_shape",
        )
    if rows.normalized_over != "choice":
        _fail(
            f"{what} normalized over {rows.normalized_over!r}, not 'choice'",
            law="choice_rows_have_the_declared_shape",
        )
    if tuple(rows.tensor.shape) != expected:
        _fail(
            f"{what} returned shape {tuple(rows.tensor.shape)} instead of {expected}",
            law="choice_rows_have_the_declared_shape",
        )

    mass = rows.tensor.exp().sum(dim=2)
    live = mass.ne(0.0)

    if bool(live.any()) and not torch.allclose(
        mass[live],
        torch.ones_like(mass[live]),
        rtol=_AGREEMENT_RTOL,
        atol=_AGREEMENT_ATOL,
    ):
        _fail(
            f"{what} returned a live row whose choice mass does not sum to one",
            law="choice_rows_have_the_declared_shape",
        )


def _first(inputs: Encoded, /) -> Encoded:
    mask = None if inputs.mask is None else inputs.mask[:1]
    return Encoded(inputs.tensor[:1], inputs.axes, mask)


class TeacherForcedDecoderSuite:
    """Laws for reporting rows along supplied terms."""

    pytestmark = pytest.mark.conformance

    def test_choice_rows_have_the_declared_shape[T: TeacherForcedDecoder](
        self, conformance: DecoderConformance[T]
    ) -> None:
        rows = conformance.build().rows(conformance.inputs, terms=conformance.terms)
        _check_rows(rows, conformance, what="rows")

    def test_shared_scoring_selects_and_sums_the_teacher_rows[T: TeacherForcedDecoder](
        self, conformance: DecoderConformance[T]
    ) -> None:
        subject = conformance.build()
        rows = subject.rows(conformance.inputs, terms=conformance.terms)
        choices = torch.tensor(
            tuple(conformance.index.codes(row) for row in conformance.terms.rows),
            device=rows.tensor.device,
        )
        occupied = torch.tensor(conformance.terms.mask, device=rows.tensor.device)
        selected = rows.tensor.gather(2, choices.unsqueeze(-1)).squeeze(-1)
        expected = torch.where(occupied, selected, selected.new_zeros(())).sum(dim=1)
        actual = score(
            subject,
            conformance.inputs,
            conformance.terms,
            index=conformance.index,
        )

        if not torch.equal(actual.tensor, expected):
            _fail(
                "shared scoring disagrees with the occupied teacher choices",
                law="shared_scoring_selects_and_sums_the_teacher_rows",
            )

    def test_decoder_local_scoring_and_sampling_are_absent[T: TeacherForcedDecoder](
        self, conformance: DecoderConformance[T]
    ) -> None:
        subject = conformance.build()
        retained = tuple(name for name in _OLD_SURFACES if hasattr(subject, name))

        if retained:
            _fail(
                f"the decoder retains local operations {retained!r}",
                law="decoder_local_scoring_and_sampling_are_absent",
            )


class MarginalDecoderSuite:
    """Laws for rows that sum over preceding choices."""

    pytestmark = pytest.mark.conformance

    def test_marginal_choice_rows_have_the_declared_shape[T: MarginalDecoder](
        self, conformance: DecoderConformance[T]
    ) -> None:
        rows = conformance.build().marginal_rows(
            conformance.inputs, root_sort=conformance.root_sort
        )
        _check_rows(rows, conformance, what="marginal_rows")


class FrontierDecoderSuite:
    """Laws for weighing the next choice of a term frontier."""

    pytestmark = pytest.mark.conformance

    def test_a_frontier_row_is_one_choice_distribution[T](
        self, conformance: DecoderConformance[T]
    ) -> None:
        subject = cast("FrontierDecoder[object]", conformance.build())
        state = subject.start(_first(conformance.inputs))
        row = subject.weigh(state, context=conformance.root)

        if row.axes != _FRONTIER_AXES:
            _fail(
                f"weigh returned axes {row.axes} where {_FRONTIER_AXES} are declared",
                law="a_frontier_row_is_one_choice_distribution",
            )
        if row.normalized_over != "choice":
            _fail(
                f"weigh normalized over {row.normalized_over!r}, not 'choice'",
                law="a_frontier_row_is_one_choice_distribution",
            )
        if row.tensor.shape != (conformance.index.width,):
            _fail(
                f"weigh returned shape {tuple(row.tensor.shape)} beside "
                f"{conformance.index.width} choices",
                law="a_frontier_row_is_one_choice_distribution",
            )


class DecoderAgreementSuite:
    """Laws connecting independent protocols implemented by one decoder."""

    pytestmark = pytest.mark.conformance

    def test_all_protocols_share_the_root_choice_row[T](
        self, conformance: DecoderConformance[T]
    ) -> None:
        subject = cast("_CompleteDecoder[object]", conformance.build())
        teacher = subject.rows(conformance.inputs, terms=conformance.terms).tensor[0, 0]
        marginal = subject.marginal_rows(
            conformance.inputs, root_sort=conformance.root_sort
        ).tensor[0, 0]
        frontier = subject.weigh(
            subject.start(_first(conformance.inputs)), context=conformance.root
        ).tensor

        if not torch.allclose(
            teacher,
            marginal,
            rtol=_AGREEMENT_RTOL,
            atol=_AGREEMENT_ATOL,
        ):
            _fail(
                "teacher-forced and marginal decoding disagree at the root",
                law="all_protocols_share_the_root_choice_row",
            )
        if not torch.allclose(
            teacher,
            frontier,
            rtol=_AGREEMENT_RTOL,
            atol=_AGREEMENT_ATOL,
        ):
            _fail(
                "teacher-forced and frontier decoding disagree at the root",
                law="all_protocols_share_the_root_choice_row",
            )


TEACHER_FORCED_DECODER_LAWS: Final = (
    TeacherForcedDecoderSuite.test_choice_rows_have_the_declared_shape,
    TeacherForcedDecoderSuite.test_shared_scoring_selects_and_sums_the_teacher_rows,
    TeacherForcedDecoderSuite.test_decoder_local_scoring_and_sampling_are_absent,
)

MARGINAL_DECODER_LAWS: Final = (
    MarginalDecoderSuite.test_marginal_choice_rows_have_the_declared_shape,
)

FRONTIER_DECODER_LAWS: Final = (
    FrontierDecoderSuite.test_a_frontier_row_is_one_choice_distribution,
)

DECODER_AGREEMENT_LAWS: Final = (
    DecoderAgreementSuite.test_all_protocols_share_the_root_choice_row,
)
