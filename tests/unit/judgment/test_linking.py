from typing import TYPE_CHECKING, Final, Never, cast, override

import pytest
import torch

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    JudgmentDesign,
    JudgmentRow,
    Linking,
    LinkingDomain,
    LinkingSpec,
    LinkTerm,
    PriorBlock,
    ReadoutColumns,
    RecordedColumns,
    ResponseSpec,
    RowColumns,
    SpecialCase,
    ThresholdPriors,
    UndeclaredColumnError,
    latent_contribution,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

LEVELS: Final = 5
ROWS: Final = 6
SCORE: Final = "score"
LENGTH: Final = "length"


class SlopeLink:
    @property
    def domain(self) -> LinkingDomain:
        return LinkingDomain(
            SCORE, {SCORE: Interval(-10.0, 10.0)}, {"slope": Interval(0.0, 4.0)}
        )

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (SpecialCase("identity", (1.0,)),)

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return parameters[0] * columns.readouts[SCORE]


class PeekingLink(SlopeLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        responses = cast("JudgmentDesign", columns).responses

        return super().contribution(columns, parameters) + responses


class OverreachingLink(SlopeLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters) + columns.readouts[LENGTH]


class ScalarLink(SlopeLink):
    @override
    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        return super().contribution(columns, parameters).sum()


class DeclaredLink:
    """Adds a fitted multiple of every further column a link term names."""

    def __init__(self, term: LinkTerm) -> None:
        self._roles = dict(term.linking.columns)
        self._domain = LinkingDomain(
            term.readout,
            {name: Interval(-10.0, 10.0) for name in term.readouts},
            {role: Interval(0.0, 4.0) for role in self._roles},
        )

    @property
    def domain(self) -> LinkingDomain:
        return self._domain

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return (SpecialCase("raw", (0.0,) * len(self._roles)),)

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        total = columns.readouts[self._domain.argument]

        for weight, role in zip(parameters, self._domain.parameter_names, strict=True):
            total = total + weight * columns.readouts[self._roles[role]]

        return total


def design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(SCORE, LENGTH))
    rows = [
        JudgmentRow(
            f"annotator-{index % 2}",
            f"item-{index // 2}",
            index % LEVELS,
            RowColumns(factors={}, readouts={SCORE: 0.5 * index, LENGTH: 2.0}),
        )
        for index in range(ROWS)
    ]

    return codec.encode(rows)


SLOPE = torch.tensor([2.0], dtype=torch.float64)


def test_a_recorded_view_reports_the_rows_of_its_source() -> None:
    view = RecordedColumns(design(), (SCORE,))

    assert view.n_rows == ROWS
    assert isinstance(view, ReadoutColumns)
    assert repr(view) == f"RecordedColumns(rows={ROWS}, admitted=['{SCORE}'])"


def test_a_recorded_view_hands_out_admitted_columns_and_records_the_reads() -> None:
    source = design()
    view = RecordedColumns(source, (SCORE,))

    assert view.touched == frozenset()
    assert torch.equal(view.readouts[SCORE], source.readouts[SCORE])
    assert view.touched == frozenset({SCORE})
    assert tuple(view.readouts) == (SCORE,)
    assert len(view.readouts) == 1
    assert SCORE in view.readouts
    assert LENGTH not in view.readouts


@pytest.mark.parametrize(
    "read",
    [
        lambda readouts: readouts.get(SCORE),
        lambda readouts: dict(readouts.items()),
        lambda readouts: list(readouts.values()),
    ],
)
def test_a_recorded_view_records_a_read_through_any_path(
    read: Callable[[Mapping[str, Tensor]], object],
) -> None:
    view = RecordedColumns(design(), (SCORE,))

    read(view.readouts)

    assert view.touched == frozenset({SCORE})


def test_a_recorded_view_records_no_read_for_membership_or_iteration() -> None:
    view = RecordedColumns(design(), (SCORE,))

    assert SCORE in view.readouts
    assert list(view.readouts) == [SCORE]
    assert len(view.readouts) == 1
    assert view.touched == frozenset()


def test_a_recorded_view_refuses_a_readout_outside_the_admitted_names() -> None:
    view = RecordedColumns(design(), (SCORE,))

    with pytest.raises(UndeclaredColumnError, match=f"column '{LENGTH}'") as raised:
        view.readouts[LENGTH]

    with pytest.raises(UndeclaredColumnError, match=f"column '{LENGTH}'"):
        view.readouts.get(LENGTH)

    assert raised.value.column == LENGTH
    assert raised.value.admitted == (SCORE,)
    assert view.touched == frozenset()


def test_a_recorded_view_refuses_an_admitted_name_the_source_lacks() -> None:
    view = RecordedColumns(design(), (SCORE, "absent"))

    with pytest.raises(KeyError):
        view.readouts["absent"]

    assert "absent" not in view.readouts
    assert list(view.readouts) == [SCORE]
    assert len(view.readouts) == 1


@pytest.mark.parametrize(
    ("column", "read"),
    [
        ("responses", lambda view: view.responses),
        ("participants", lambda view: view.participants),
        ("items", lambda view: view.items),
        ("factors", lambda view: view.factors),
        ("alternatives", lambda view: view.alternatives),
        ("participant_levels", lambda view: view.participant_levels),
        ("response_levels", lambda view: view.response_levels),
    ],
)
def test_a_recorded_view_refuses_every_column_beyond_the_readouts(
    column: str, read: Callable[[RecordedColumns], Never]
) -> None:
    view = RecordedColumns(design(), (SCORE,))

    with pytest.raises(UndeclaredColumnError, match=f"column '{column}'") as raised:
        read(view)

    assert raised.value.admitted == (SCORE,)


def test_a_recorded_view_hands_out_no_live_mapping() -> None:
    view = RecordedColumns(design(), (SCORE,))

    with pytest.raises(TypeError):
        cast("dict[str, Tensor]", view.readouts)[LENGTH] = torch.zeros(ROWS)


@pytest.mark.parametrize("name", ["", None])
def test_a_recorded_view_refuses_a_blank_admitted_name(name: str | None) -> None:
    with pytest.raises((ValueError, TypeError)):
        RecordedColumns(design(), (cast("str", name),))


def test_the_error_names_the_column_and_the_admitted_readouts_in_order() -> None:
    error = UndeclaredColumnError("responses", ("z", "a"))

    assert error.column == "responses"
    assert error.admitted == ("a", "z")
    assert str(error) == (
        "a linking read the column 'responses', and its declared readouts are "
        "('a', 'z')"
    )
    assert isinstance(error, LookupError)


def test_an_unchecked_contribution_is_the_linking_applied_to_the_columns() -> None:
    source = design()

    assert torch.equal(
        latent_contribution(SlopeLink(), source, SLOPE),
        2.0 * source.readouts[SCORE],
    )


def test_an_unchecked_contribution_hands_the_linking_the_columns_themselves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", False)
    source = design()

    assert torch.equal(
        latent_contribution(PeekingLink(), source, SLOPE),
        2.0 * source.readouts[SCORE] + source.responses,
    )


def test_a_checked_contribution_refuses_a_linking_that_reads_the_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)

    with pytest.raises(UndeclaredColumnError, match="column 'responses'"):
        latent_contribution(PeekingLink(), design(), SLOPE)


def test_a_checked_contribution_refuses_a_linking_reading_an_undeclared_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)

    with pytest.raises(UndeclaredColumnError, match=f"column '{LENGTH}'"):
        latent_contribution(OverreachingLink(), design(), SLOPE)


@pytest.mark.parametrize(
    "parameters",
    [torch.zeros(0), torch.zeros(2), torch.zeros((1, 1)), torch.zeros(())],
)
def test_a_checked_contribution_refuses_a_parameter_vector_of_another_shape(
    monkeypatch: pytest.MonkeyPatch, parameters: Tensor
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)

    with pytest.raises(ValueError, match="over the parameters \\('slope',\\)"):
        latent_contribution(SlopeLink(), design(), parameters)


def test_a_checked_contribution_refuses_a_result_that_is_not_one_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)

    with pytest.raises(ValueError, match=f"over {ROWS} rows returned"):
        latent_contribution(ScalarLink(), design(), SLOPE)


def test_a_checked_contribution_agrees_with_the_unchecked_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = design()
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", False)
    unchecked = latent_contribution(SlopeLink(), source, SLOPE)
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)
    checked = latent_contribution(SlopeLink(), source, SLOPE)

    assert torch.equal(checked, unchecked)


def test_a_link_term_reading_two_columns_is_evaluated_over_both_under_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.judgment.linking.CHECKS", True)
    term = LinkTerm(
        readout=SCORE, linking=LinkingSpec(family="f", columns={"length": LENGTH})
    )
    spec = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(term,),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    linking = DeclaredLink(term)
    codec = DesignCodec(LEVELS, readouts=spec.readouts)
    source = codec.encode(
        [
            JudgmentRow(
                "annotator",
                f"item-{index}",
                index % LEVELS,
                RowColumns(
                    factors={}, readouts={SCORE: 0.5 * index, LENGTH: 1.0 + index}
                ),
            )
            for index in range(ROWS)
        ]
    )
    strength = torch.tensor([0.5], dtype=torch.float64)
    view = RecordedColumns(source, linking.domain.readouts)

    assert spec.readouts == (SCORE, LENGTH)
    assert linking.domain.readouts == term.readouts
    assert isinstance(linking, Linking)
    assert torch.equal(
        latent_contribution(linking, source, strength),
        source.readouts[SCORE] + 0.5 * source.readouts[LENGTH],
    )
    assert torch.equal(
        linking.contribution(view, strength),
        latent_contribution(linking, view, strength),
    )
    assert view.touched == frozenset(spec.readouts)


def test_the_witness_answers_the_port() -> None:
    assert isinstance(SlopeLink(), Linking)
