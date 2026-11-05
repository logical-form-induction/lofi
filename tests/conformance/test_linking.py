from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.conformance.linking import (
    CLOSED_FORM_ATOL,
    LinkingSubject,
    LinkingSuite,
    SampledColumns,
    contribution_is_non_decreasing_in_the_argument,
    special_cases_recover_their_closed_forms,
    the_response_is_never_read,
)

from lofi._checks import LawViolationError
from lofi.judgment import (
    DesignCodec,
    JudgmentDesign,
    JudgmentRow,
    Linking,
    LinkingDomain,
    ReadoutColumns,
    RowColumns,
    SpecialCase,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

LEVELS: Final = 7
SCORE: Final = "score"
LENGTH: Final = "length"
ROWS: Final = 12


class AffineLink:
    """Multiplies the argument by a slope and adds an intercept."""

    def __init__(self) -> None:
        self._domain = LinkingDomain(
            SCORE,
            {SCORE: Interval(-10.0, 10.0)},
            {"slope": Interval(0.0, 4.0), "intercept": Interval(-1.0, 1.0)},
        )

    @property
    def domain(self) -> LinkingDomain:
        return self._domain

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (
            SpecialCase("identity", (1.0, 0.0)),
            SpecialCase("constant", (0.0, 0.5)),
        )

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return parameters[0] * columns.readouts[SCORE] + parameters[1]


class CorrectedLink:
    """Adds a fitted multiple of a second column to the argument."""

    def __init__(self) -> None:
        self._domain = LinkingDomain(
            SCORE,
            {SCORE: Interval(-200.0, 0.0), LENGTH: Interval(1.0, 40.0)},
            {"strength": Interval(0.0, 2.0)},
        )

    @property
    def domain(self) -> LinkingDomain:
        return self._domain

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (SpecialCase("raw", (0.0,)),)

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return columns.readouts[SCORE] + parameters[0] * columns.readouts[LENGTH]


class IdentityLink:
    """Returns the argument as it is, fitting nothing."""

    @property
    def domain(self) -> LinkingDomain:
        return LinkingDomain(SCORE, {SCORE: Interval(-10.0, 10.0)})

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (SpecialCase("identity", ()),)

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        del parameters

        return columns.readouts[SCORE]


class CountingLink(AffineLink):
    """Records the row count of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[int] = []

    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        self.rows.append(columns.n_rows)

        return super().contribution(columns, parameters)


class DecreasingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return -super().contribution(columns, parameters)


class SaturatingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        score = columns.readouts[SCORE]

        return torch.where(score > 5.0, score - 20.0, score)


class DippingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        score = columns.readouts[SCORE]

        return torch.where(score > 5.0, score - 0.5, score)


class DivergentLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters) / 0.0


class ScalarLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters).sum()


class DriftingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters) + 2.0 * CLOSED_FORM_ATOL


class MisdeclaredCaseLink(AffineLink):
    @property
    @override
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (SpecialCase("identity", (1.0,)), SpecialCase("constant", (0.0, 0.5)))


class RepeatedCaseLink(AffineLink):
    @property
    @override
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (
            SpecialCase("identity", (1.0, 0.0)),
            SpecialCase("identity", (0.0, 0.5)),
        )


class PeekingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        affine = super().contribution(columns, parameters)

        return affine + cast("JudgmentDesign", columns).responses.to(torch.float64)


class OverreachingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters) + columns.readouts[LENGTH]


class ReachingLink(AffineLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        levels = cast("JudgmentDesign", columns).participant_levels

        return super().contribution(columns, parameters) + len(levels)


def design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(SCORE, LENGTH))
    rows = [
        JudgmentRow(
            f"annotator-{index % 4}",
            f"item-{index // 4}",
            index % LEVELS,
            RowColumns(factors={}, readouts={SCORE: -1.5 * index, LENGTH: 1.0 + index}),
        )
        for index in range(ROWS)
    ]

    return codec.encode(rows)


def identity(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[SCORE]


def constant(readouts: Mapping[str, Tensor]) -> Tensor:
    return torch.full_like(readouts[SCORE], 0.5)


AFFINE_REFERENCES: Final[Mapping[str, Callable[[Mapping[str, Tensor]], Tensor]]] = {
    "identity": identity,
    "constant": constant,
}


def affine_subject(linking: Linking) -> LinkingSubject:
    return LinkingSubject(
        linking,
        design(),
        torch.tensor([2.0, 0.25], dtype=torch.float64),
        AFFINE_REFERENCES,
    )


def corrected_subject() -> LinkingSubject:
    return LinkingSubject(
        CorrectedLink(),
        design(),
        torch.tensor([0.5], dtype=torch.float64),
        {"raw": identity},
    )


class TestAffineLink(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return affine_subject(AffineLink())


class TestCorrectedLink(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return corrected_subject()


class TestIdentityLink(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            IdentityLink(),
            design(),
            torch.empty(0, dtype=torch.float64),
            {"identity": identity},
        )


def test_the_monotonicity_law_compares_ten_thousand_points_before_and_after() -> None:
    linking = CountingLink()

    contribution_is_non_decreasing_in_the_argument(affine_subject(linking))

    assert sum(linking.rows) == 2 * 10_000
    assert len(linking.rows) == 2 * 10


def test_the_witnesses_answer_the_declared_port() -> None:
    assert isinstance(AffineLink(), Linking)
    assert isinstance(CorrectedLink(), Linking)
    assert isinstance(IdentityLink(), Linking)
    assert isinstance(SampledColumns({SCORE: torch.zeros(3)}), ReadoutColumns)


def test_sampled_columns_report_their_rows() -> None:
    columns = SampledColumns({SCORE: torch.zeros(3), LENGTH: torch.ones(3)})

    assert columns.n_rows == 3
    assert tuple(columns.readouts) == (SCORE, LENGTH)


@pytest.mark.parametrize(
    ("linking", "detail"),
    [
        (DecreasingLink(), "lowered the contribution"),
        (SaturatingLink(), "lowered the contribution"),
        (DippingLink(), "lowered the contribution by 0\\.[0-4]"),
        (DivergentLink(), "is not finite"),
        (ScalarLink(), "has shape ()"),
    ],
)
def test_the_monotonicity_law_rejects_its_witnesses(
    linking: Linking, detail: str
) -> None:
    with pytest.raises(LawViolationError, match=detail):
        contribution_is_non_decreasing_in_the_argument(affine_subject(linking))


def test_the_monotonicity_law_names_the_argument_and_the_draw() -> None:
    with pytest.raises(
        LawViolationError, match=rf"raising '{SCORE}' from .* \(draw 0\)"
    ):
        contribution_is_non_decreasing_in_the_argument(affine_subject(DecreasingLink()))


@pytest.mark.parametrize(
    ("linking", "detail"),
    [
        (DriftingLink(), "departs from its closed form"),
        (MisdeclaredCaseLink(), "declares 1 parameters"),
        (RepeatedCaseLink(), "repeat a name"),
        (ScalarLink(), "has shape ()"),
    ],
)
def test_the_closed_form_law_rejects_its_witnesses(
    linking: Linking, detail: str
) -> None:
    with pytest.raises(LawViolationError, match=detail):
        special_cases_recover_their_closed_forms(affine_subject(linking))


def test_the_closed_form_law_requires_one_reference_per_declared_case() -> None:
    subject = affine_subject(AffineLink())
    fewer = subject._replace(references={"identity": identity})
    more = subject._replace(references={**AFFINE_REFERENCES, "extra": identity})

    with pytest.raises(LawViolationError, match="references cover"):
        special_cases_recover_their_closed_forms(fewer)

    with pytest.raises(LawViolationError, match="references cover"):
        special_cases_recover_their_closed_forms(more)


def test_the_closed_form_law_accepts_a_departure_within_tolerance() -> None:
    class NearLink(AffineLink):
        @override
        def contribution(
            self, columns: ReadoutColumns, parameters: Tensor, /
        ) -> Tensor:
            return super().contribution(columns, parameters) + CLOSED_FORM_ATOL / 2.0

    special_cases_recover_their_closed_forms(affine_subject(NearLink()))


@pytest.mark.parametrize(
    ("linking", "detail"),
    [
        (PeekingLink(), "read 'responses'"),
        (ReachingLink(), "read 'participant_levels'"),
        (OverreachingLink(), f"read '{LENGTH}'"),
        (DivergentLink(), "is not finite"),
        (ScalarLink(), "has shape ()"),
    ],
)
def test_the_no_peeking_law_rejects_its_witnesses(
    linking: Linking, detail: str
) -> None:
    with pytest.raises(LawViolationError, match=detail):
        the_response_is_never_read(affine_subject(linking))


def test_a_peeking_link_is_caught_only_through_the_recording_view() -> None:
    subject = affine_subject(PeekingLink())
    unguarded = subject.linking.contribution(subject.design, subject.parameters)

    assert unguarded.shape == (ROWS,)

    with pytest.raises(LawViolationError, match=r"it had read \['score'\]"):
        the_response_is_never_read(subject)


@pytest.mark.parametrize(
    "law",
    [
        contribution_is_non_decreasing_in_the_argument,
        special_cases_recover_their_closed_forms,
        the_response_is_never_read,
    ],
)
def test_every_law_checks_the_subject_against_the_declared_domain(
    law: Callable[[LinkingSubject], None],
) -> None:
    subject = affine_subject(AffineLink())
    short = subject._replace(parameters=torch.tensor([2.0], dtype=torch.float64))
    outside = subject._replace(
        parameters=torch.tensor([-1.0, 0.0], dtype=torch.float64)
    )
    lacking = subject._replace(
        design=DesignCodec(LEVELS, readouts=(LENGTH,)).encode([])
    )

    with pytest.raises(LawViolationError, match=r"parameters have shape \(1,\)"):
        law(short)

    with pytest.raises(LawViolationError, match=r"parameter 'slope' is -1\.0, outside"):
        law(outside)

    with pytest.raises(LawViolationError, match="lacks the declared readouts"):
        law(lacking)
