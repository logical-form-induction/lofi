import math
from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._recursion import RecursionSpread
from lofi.probe._structure import NODE_AXES
from lofi.probe._values import Observation, Trace
from lofi.terms import BatchEncoding, Encoding

ENCODER: Final = ComponentId("hrr")
NODES: Final = Site(ENCODER, Role.NODE_ENCODING)
CELL: Final = Site(ComponentId("cky-parser"), Role.CHART_CELL, (0, 2))

NODE_SPEC: Final = SiteSpec(
    NODE_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
)
CELL_SPEC: Final = SiteSpec(
    ("item", "width"), DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
)

CONTEXT: Final = DiagnosticContext({NODES: NODE_SPEC})

# how many slots every term below is linearized into
SLOTS: Final = 4

ALONG: Final = (1.0, 0.0)
ACROSS: Final = (0.0, 1.0)
BACK: Final = (0.0, -1.0)


def a_chain(length: int) -> Encoding:
    filling = SLOTS - length

    return Encoding(
        (4,) * SLOTS,
        tuple(range(-1, length - 1)) + (Encoding.NO_PARENT,) * filling,
        tuple(range(length)) + (0,) * filling,
        (True,) * length + (False,) * filling,
    )


# terms of two, three and four nodes, so the node count varies across the batch
GROWING: Final = BatchEncoding((a_chain(2), a_chain(3), a_chain(4)))
PAIR: Final = BatchEncoding((a_chain(2),))
LONE: Final = BatchEncoding((a_chain(1),))


def a_trace(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> Trace:
    trace = Trace()
    trace.record(Observation(NODES, tensor, axes=NODE_AXES, mask=mask))
    trace.close()

    return trace


def rows(*vectors: tuple[float, float]) -> list[list[float]]:
    filling = [[0.0, 0.0]] * (SLOTS - len(vectors))

    return [list(vector) for vector in vectors] + filling


def collapsing() -> torch.Tensor:
    # mean likenesses of nought, a third and a half as the terms take on nodes
    return torch.tensor(
        [
            rows(ALONG, ACROSS),
            rows(ALONG, ALONG, ACROSS),
            rows(ALONG, ALONG, ALONG, ACROSS),
        ]
    )


def steady() -> torch.Tensor:
    # spreads of a half, a quarter and a half, which no line through the counts fits
    leaning = (0.25, math.sqrt(1.0 - 0.25**2))

    return torch.tensor(
        [
            rows(ALONG, ACROSS),
            rows(ALONG, ALONG, leaning),
            rows(ALONG, ACROSS, (-1.0, 0.0), ACROSS),
        ]
    )


def test_the_spread_of_a_term_is_the_mean_angle_between_its_own_nodes() -> None:
    result = RecursionSpread(PAIR).run(
        a_trace(torch.tensor([rows(ALONG, ACROSS)])), context=CONTEXT
    )

    assert result.tag == "recursion-spread"
    assert result.value(NODES, "terms") == 1.0
    assert result.value(NODES, "node-spread") == pytest.approx(0.5)


def test_a_term_whose_nodes_are_encoded_alike_spreads_not_at_all() -> None:
    result = RecursionSpread(PAIR).run(
        a_trace(torch.tensor([rows(ALONG, ALONG)])), context=CONTEXT
    )

    assert result.value(NODES, "node-spread") == pytest.approx(0.0)


def test_a_term_whose_nodes_are_opposed_spreads_wholly() -> None:
    result = RecursionSpread(PAIR).run(
        a_trace(torch.tensor([rows(ACROSS, BACK)])), context=CONTEXT
    )

    assert result.value(NODES, "node-spread") == pytest.approx(1.0)


def test_an_encoder_whose_nodes_run_together_as_a_term_grows_falls_with_the_count() -> (
    None
):
    result = RecursionSpread(GROWING).run(a_trace(collapsing()), context=CONTEXT)

    assert result.value(NODES, "terms") == 3.0
    assert result.value(NODES, "node-spread") == pytest.approx((0.5 + 1 / 3 + 0.25) / 3)
    assert result.value(NODES, "count-correlation") < -0.95
    assert result.value(NODES, "count-grading") > 0.95


def test_an_encoder_whose_spread_does_not_follow_the_count_grades_at_nothing() -> None:
    result = RecursionSpread(GROWING).run(a_trace(steady()), context=CONTEXT)

    assert result.value(NODES, "count-correlation") == pytest.approx(0.0, abs=1e-12)
    assert result.value(NODES, "count-grading") == pytest.approx(0.0, abs=1e-12)


def test_scaling_every_encoding_leaves_the_grading_where_it_was() -> None:
    scaled = RecursionSpread(GROWING).run(a_trace(collapsing() * 1e-6), context=CONTEXT)

    assert scaled.value(NODES, "count-correlation") < -0.95


def test_a_term_holding_one_live_slot_has_no_pair_and_is_left_out() -> None:
    result = RecursionSpread(LONE).run(
        a_trace(torch.tensor([rows(ALONG)])), context=CONTEXT
    )

    assert result.value(NODES, "terms") == 0.0

    with pytest.raises(KeyError, match="node-spread"):
        result.value(NODES, "node-spread")


def test_a_slot_the_reading_masks_off_enters_no_pair() -> None:
    covered = torch.tensor([[True, False, False, False]])
    result = RecursionSpread(PAIR).run(
        a_trace(torch.tensor([rows(ALONG, ACROSS)]), covered), context=CONTEXT
    )

    assert result.value(NODES, "terms") == 0.0


def test_a_diagnostic_built_without_terms_refuses_a_reading() -> None:
    with pytest.raises(ValueError, match="without the terms"):
        RecursionSpread().run(a_trace(collapsing()), context=CONTEXT)


def test_the_diagnostic_resolves_only_where_type_encodings_are_published() -> None:
    without = installed_diagnostics().matching(DiagnosticContext({CELL: CELL_SPEC}))

    assert "recursion-spread" not in [one.tag for one in without]
    assert "recursion-spread" in [
        one.tag for one in installed_diagnostics().matching(CONTEXT)
    ]


def test_the_diagnostic_says_what_it_holds_and_which_axes_it_reads() -> None:
    held = repr(RecursionSpread(GROWING))

    assert held == f"RecursionSpread(3 terms, axes={NODE_AXES!r})"
