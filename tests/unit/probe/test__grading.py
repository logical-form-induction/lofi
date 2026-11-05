import math
from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._grading import DepthGrading
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

# the chain of five nodes every reading here encodes
LENGTH: Final = 5

# the angle a graded encoder turns a node through per edge below the root
TURN: Final = math.pi / 8

CHAIN: Final = BatchEncoding(
    (
        Encoding(
            (7, 3, 3, 3, 2),
            tuple(range(-1, LENGTH - 1)),
            tuple(range(LENGTH)),
            (True,) * LENGTH,
        ),
    )
)


def a_trace(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> Trace:
    trace = Trace()
    trace.record(Observation(NODES, tensor, axes=NODE_AXES, mask=mask))
    trace.close()

    return trace


def graded() -> torch.Tensor:
    turned = torch.arange(LENGTH, dtype=torch.float32) * TURN

    return torch.stack((turned.cos(), turned.sin(), torch.zeros(LENGTH))).T.unsqueeze(0)


def blind() -> torch.Tensor:
    # alike to the root, then orthogonal to it, and so on down the chain
    return torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )


def rising() -> torch.Tensor:
    across = math.sqrt(3.0) / 2.0

    # likeness to the root of one, minus one, minus a half, a half and one
    return torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-0.5, across, 0.0],
                [0.5, across, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )


def test_a_planted_encoder_grades_its_nodes_strongly_by_depth() -> None:
    result = DepthGrading(CHAIN).run(a_trace(graded()), context=CONTEXT)

    assert result.tag == "depth-graded-encoding"
    assert result.value(NODES, "nodes") == float(LENGTH)
    assert result.value(NODES, "depth-correlation") < -0.95
    assert result.value(NODES, "depth-grading") > 0.95


def test_a_depth_blind_encoder_grades_its_nodes_not_at_all() -> None:
    result = DepthGrading(CHAIN).run(a_trace(blind()), context=CONTEXT)

    assert result.value(NODES, "nodes") == float(LENGTH)
    assert result.value(NODES, "depth-correlation") == pytest.approx(0.0, abs=1e-12)
    assert result.value(NODES, "depth-grading") == pytest.approx(0.0, abs=1e-12)


def test_the_grading_is_the_magnitude_however_the_coefficient_runs() -> None:
    upward = DepthGrading(CHAIN).run(a_trace(rising()), context=CONTEXT)
    downward = DepthGrading(CHAIN).run(a_trace(graded()), context=CONTEXT)

    assert upward.value(NODES, "depth-correlation") > 0.0
    assert downward.value(NODES, "depth-correlation") < 0.0

    for measured in (upward, downward):
        assert measured.value(NODES, "depth-grading") == pytest.approx(
            abs(measured.value(NODES, "depth-correlation"))
        )


def test_scaling_every_encoding_leaves_the_grading_where_it_was() -> None:
    measured = DepthGrading(CHAIN).run(a_trace(graded()), context=CONTEXT)
    scaled = DepthGrading(CHAIN).run(a_trace(graded() * 1e6), context=CONTEXT)

    assert scaled.value(NODES, "depth-grading") == pytest.approx(
        measured.value(NODES, "depth-grading")
    )


def test_a_reading_masked_off_everywhere_reports_no_node_and_no_coefficient() -> None:
    covered = torch.zeros(1, LENGTH, dtype=torch.bool)
    result = DepthGrading(CHAIN).run(a_trace(graded(), covered), context=CONTEXT)

    assert result.value(NODES, "nodes") == 0.0

    with pytest.raises(KeyError, match="depth-grading"):
        result.value(NODES, "depth-grading")


def test_a_term_whose_root_is_masked_off_leaves_its_nodes_unmeasured() -> None:
    covered = torch.tensor([[False, True, True, True, True]])
    result = DepthGrading(CHAIN).run(a_trace(graded(), covered), context=CONTEXT)

    assert result.value(NODES, "nodes") == 0.0


def test_a_diagnostic_built_without_terms_refuses_a_reading() -> None:
    with pytest.raises(ValueError, match="without the terms"):
        DepthGrading().run(a_trace(graded()), context=CONTEXT)


def test_a_component_publishing_no_type_encoding_is_measured_at_no_point() -> None:
    empty = Trace()
    empty.close()
    result = DepthGrading(CHAIN).run(
        empty, context=DiagnosticContext({CELL: CELL_SPEC})
    )

    assert result.measurements == ()


def test_the_diagnostic_resolves_only_where_type_encodings_are_published() -> None:
    without = installed_diagnostics().matching(DiagnosticContext({CELL: CELL_SPEC}))
    with_nodes = installed_diagnostics().matching(CONTEXT)

    assert "depth-graded-encoding" not in [one.tag for one in without]
    assert "depth-graded-encoding" in [one.tag for one in with_nodes]


def test_the_diagnostic_says_what_it_holds_and_which_axes_it_reads() -> None:
    assert repr(DepthGrading(CHAIN)) == f"DepthGrading(1 terms, axes={NODE_AXES!r})"
    assert repr(DepthGrading()).startswith("DepthGrading(0 terms")
