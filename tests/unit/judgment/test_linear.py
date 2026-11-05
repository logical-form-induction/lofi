from typing import TYPE_CHECKING, Final

import pytest
import torch
from tests.conformance.linking import LinkingSubject, LinkingSuite

from lofi.judgment import (
    LINEAR,
    Binary,
    DesignCodec,
    JudgmentDesign,
    JudgmentRow,
    LinearLinking,
    Linking,
    LinkingSpec,
    LinkTerm,
    Literal,
    Name,
    Operator,
    Prior,
    RowColumns,
    linear_steps,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

READOUT: Final = "root_entropy"
LEVELS: Final = 7
VALUES: Final = (0.1, 0.7, 1.4, 2.2)


def term(**fixed: float) -> LinkTerm:
    return LinkTerm(
        readout=READOUT,
        linking=LinkingSpec(family=LINEAR, fixed=fixed),
    )


def design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(READOUT,))

    return codec.encode(
        JudgmentRow(
            "participant",
            f"item-{position}",
            position,
            RowColumns(factors={}, readouts={READOUT: value}),
        )
        for position, value in enumerate(VALUES)
    )


def identity(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[READOUT]


class TestLinearLinking(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LinearLinking(term()),
            design(),
            torch.tensor([1.5], dtype=torch.float64),
            {"identity": identity},
        )


class TestLinearLinkingWithAUnitSlope(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LinearLinking(term(slope=1.0)),
            design(),
            torch.empty(0, dtype=torch.float64),
            {"identity": identity},
        )


class TestLinearLinkingWithAnotherPinnedSlope(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LinearLinking(term(slope=2.0)),
            design(),
            torch.empty(0, dtype=torch.float64),
            {},
        )


def test_the_linear_linking_answers_the_port() -> None:
    assert isinstance(LinearLinking(term()), Linking)


def test_the_domain_is_the_scalar_and_one_slope() -> None:
    linking = LinearLinking(term())

    assert linking.domain.argument == READOUT
    assert linking.domain.readouts == (READOUT,)
    assert linking.domain.parameter_names == ("slope",)
    assert linking.pinned == {}


def test_the_contribution_is_the_slope_times_the_scalar() -> None:
    source = design()
    slope = torch.tensor([2.5], dtype=torch.float64)

    torch.testing.assert_close(
        LinearLinking(term()).contribution(source, slope),
        2.5 * source.readouts[READOUT],
    )


def test_a_free_slope_uses_the_identification_requested_prior() -> None:
    positive = linear_steps(term(), READOUT)
    signed = linear_steps(term(), READOUT, signed_slope=True)

    assert positive.draws == {"root_entropy_slope": Prior("HalfNormal", 1.0)}
    assert signed.draws == {"root_entropy_slope": Prior("Normal", 0.0, 1.0)}
    assert positive.expression == Binary(
        Operator.MUL,
        Name("root_entropy_slope"),
        Name(READOUT),
    )


def test_a_pinned_slope_is_written_directly_into_the_program() -> None:
    steps = linear_steps(term(slope=-0.75), READOUT, signed_slope=True)

    assert steps.draws == {}
    assert steps.expression == Binary(
        Operator.MUL,
        Literal(-0.75),
        Name(READOUT),
    )


def test_unit_slope_identification_draws_nothing() -> None:
    steps = linear_steps(term(), READOUT, unit_slope=True, signed_slope=True)

    assert steps.draws == {}
    assert steps.expression == Binary(
        Operator.MUL,
        Literal(1.0),
        Name(READOUT),
    )


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (
            LinkTerm(readout=READOUT, linking=LinkingSpec(family="other")),
            "family 'other'",
        ),
        (
            LinkTerm(
                readout=READOUT,
                linking=LinkingSpec(family=LINEAR, columns={"other": "column"}),
            ),
            "argument alone",
        ),
        (
            LinkTerm(
                readout=READOUT,
                linking=LinkingSpec(family=LINEAR, fixed={"other": 1.0}),
            ),
            "were pinned",
        ),
    ],
)
def test_a_linear_linking_refuses_an_incompatible_declaration(
    declared: LinkTerm, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LinearLinking(declared)


def test_unit_slope_identification_refuses_an_inconsistent_pin() -> None:
    with pytest.raises(ValueError, match="holds the slope at one"):
        LinearLinking(term(slope=2.0), unit_slope=True)


def test_the_representation_reconstructs_the_declared_linear_link() -> None:
    linking = LinearLinking(term(slope=0.5))

    assert (
        repr(linking) == "LinearLinking(argument='root_entropy', pinned={'slope': 0.5})"
    )
