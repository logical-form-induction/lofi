from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._differentiation import AtomicDifferentiation
from lofi.probe._structure import NODE_AXES
from lofi.probe._values import Observation, Trace
from lofi.terms import BatchEncoding, Encoding

ENCODER: Final = ComponentId("tree-rnn")
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
SLOTS: Final = 3

# one constructor code throughout, so nothing can read the distinction off it
ONE_SYMBOL: Final = 4


def a_chain(length: int) -> Encoding:
    filling = SLOTS - length

    return Encoding(
        (ONE_SYMBOL,) * SLOTS,
        tuple(range(-1, length - 1)) + (Encoding.NO_PARENT,) * filling,
        tuple(range(length)) + (0,) * filling,
        (True,) * length + (False,) * filling,
    )


# a chain of three nodes beside a term of one, so two slots are complex and two atomic
MIXED: Final = BatchEncoding((a_chain(SLOTS), a_chain(1)))
LEAVES: Final = BatchEncoding((a_chain(1), a_chain(1)))


def a_trace(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> Trace:
    trace = Trace()
    trace.record(Observation(NODES, tensor, axes=NODE_AXES, mask=mask))
    trace.close()

    return trace


def apart() -> torch.Tensor:
    # the two complex slots one way and the two atomic slots the other
    return torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )


def crossed() -> torch.Tensor:
    # the two kinds at a right angle rather than opposed
    return torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )


def together() -> torch.Tensor:
    return torch.ones(2, SLOTS, 2)


def test_a_slot_is_complex_where_a_live_slot_hangs_off_it() -> None:
    result = AtomicDifferentiation(MIXED).run(a_trace(apart()), context=CONTEXT)

    assert result.tag == "atomic-complex-differentiation"
    assert result.value(NODES, "complex") == 2.0
    assert result.value(NODES, "atomic") == 2.0


def test_the_two_kinds_encoded_opposite_ways_are_wholly_differentiated() -> None:
    result = AtomicDifferentiation(MIXED).run(a_trace(apart()), context=CONTEXT)

    assert result.value(NODES, "differentiation") == pytest.approx(1.0)


def test_the_two_kinds_at_a_right_angle_are_differentiated_by_a_half() -> None:
    result = AtomicDifferentiation(MIXED).run(a_trace(crossed()), context=CONTEXT)

    assert result.value(NODES, "differentiation") == pytest.approx(0.5)


def test_an_encoder_holding_the_two_kinds_alike_differentiates_nothing() -> None:
    result = AtomicDifferentiation(MIXED).run(a_trace(together()), context=CONTEXT)

    assert result.value(NODES, "differentiation") == pytest.approx(0.0)


def test_scaling_every_encoding_leaves_the_differentiation_where_it_was() -> None:
    scaled = AtomicDifferentiation(MIXED).run(a_trace(apart() * 1e-6), context=CONTEXT)

    assert scaled.value(NODES, "differentiation") == pytest.approx(1.0)


def test_a_batch_of_leaves_alone_reports_its_counts_and_no_separation() -> None:
    result = AtomicDifferentiation(LEAVES).run(a_trace(apart()), context=CONTEXT)

    assert result.value(NODES, "atomic") == 2.0
    assert result.value(NODES, "complex") == 0.0

    with pytest.raises(KeyError, match="differentiation"):
        result.value(NODES, "differentiation")


def test_a_parent_whose_only_child_is_masked_off_counts_as_a_leaf() -> None:
    covered = torch.tensor([[True, True, False], [True, False, False]])
    result = AtomicDifferentiation(MIXED).run(
        a_trace(apart(), covered), context=CONTEXT
    )

    assert result.value(NODES, "complex") == 1.0
    assert result.value(NODES, "atomic") == 2.0


def test_a_diagnostic_built_without_terms_refuses_a_reading() -> None:
    with pytest.raises(ValueError, match="without the terms"):
        AtomicDifferentiation().run(a_trace(apart()), context=CONTEXT)


def test_the_diagnostic_resolves_only_where_type_encodings_are_published() -> None:
    without = installed_diagnostics().matching(DiagnosticContext({CELL: CELL_SPEC}))

    assert "atomic-complex-differentiation" not in [one.tag for one in without]
    assert "atomic-complex-differentiation" in [
        one.tag for one in installed_diagnostics().matching(CONTEXT)
    ]


def test_the_diagnostic_says_what_it_holds_and_which_axes_it_reads() -> None:
    held = repr(AtomicDifferentiation(MIXED))

    assert held == f"AtomicDifferentiation(2 terms, axes={NODE_AXES!r})"
