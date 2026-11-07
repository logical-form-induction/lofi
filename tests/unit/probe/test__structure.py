from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._structure import (
    NODE_AXES,
    NodeStructure,
    correlation,
    node_readings,
    pairwise_similarity,
    similarity,
    strictly_above,
)
from lofi.probe._values import Observation, Trace
from lofi.terms import BatchEncoding, Encoding

ENCODER: Final = ComponentId("hrr")
NODES: Final = Site(ENCODER, Role.NODE_ENCODING)
WHOLE: Final = Site(ENCODER, Role.TERM_ENCODING)
CELL: Final = Site(ComponentId("cky-parser"), Role.CHART_CELL, (0, 2))

NODE_SPEC: Final = SiteSpec(
    NODE_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
)
TERM_SPEC: Final = SiteSpec(
    (NODE_AXES.terms, NODE_AXES.features),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
CELL_SPEC: Final = SiteSpec(
    ("item", "width"), DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
)

CONTEXT: Final = DiagnosticContext({NODES: NODE_SPEC, WHOLE: TERM_SPEC})


def a_chain(length: int, slots: int, symbols: tuple[int, ...]) -> Encoding:
    filling = slots - length

    return Encoding(
        symbols + (0,) * filling,
        tuple(range(-1, length - 1)) + (Encoding.NO_PARENT,) * filling,
        tuple(range(length)) + (0,) * filling,
        (True,) * length + (False,) * filling,
    )


def a_trace(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def a_reading(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> Observation:
    return Observation(NODES, tensor, axes=NODE_AXES, mask=mask)


def test_two_vectors_pointing_the_same_way_are_wholly_alike() -> None:
    along = torch.tensor([1.0, 2.0, 2.0])

    assert float(similarity(along, along)) == pytest.approx(1.0)
    assert float(similarity(along, -along)) == pytest.approx(-1.0)
    aside = torch.tensor([2.0, -1.0, 0.0])

    assert float(similarity(along, aside)) == pytest.approx(0.0)


def test_a_vector_at_the_origin_is_alike_to_nothing_rather_than_undefined() -> None:
    assert float(similarity(torch.zeros(3), torch.ones(3))) == 0.0


def test_likeness_does_not_move_when_both_readings_are_scaled() -> None:
    left = torch.tensor([1.0, 3.0, -2.0])
    right = torch.tensor([2.0, 1.0, 0.5])
    apart = float(similarity(left, right))

    assert float(similarity(left * 1e6, right * 1e-6)) == pytest.approx(apart)


def test_every_pair_of_a_stack_is_scored_against_every_other() -> None:
    stack = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scored = pairwise_similarity(stack)

    assert tuple(scored.shape) == (3, 3)
    assert float(scored[0, 1]) == pytest.approx(0.0)
    assert float(scored[0, 2]) == pytest.approx(-1.0)
    assert float(scored[2, 2]) == pytest.approx(1.0)


def test_a_pair_is_counted_once_and_never_against_itself() -> None:
    once = strictly_above(4)

    assert int(once.sum()) == 6
    assert not bool(once.diagonal().any())
    assert bool(once[1, 3])
    assert not bool(once[3, 1])


def test_a_series_rising_with_another_correlates_perfectly() -> None:
    rising = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert correlation(rising, rising * 3.0) == pytest.approx(1.0)
    assert correlation(rising, -rising) == pytest.approx(-1.0)


def test_a_series_that_never_moves_correlates_with_nothing() -> None:
    assert correlation(torch.ones(5), torch.arange(5.0)) == 0.0
    assert correlation(torch.tensor([]), torch.tensor([])) == 0.0


def test_a_correlation_is_the_hand_computed_coefficient() -> None:
    # two deviations of -1 and 1 against 1 and -1 over a spread of two
    assert correlation(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 1.0, 1.0])
    ) == pytest.approx(-0.866025403784, rel=1e-9)


def test_a_type_encoding_published_without_the_node_axis_is_left_out() -> None:
    nodes = a_reading(torch.ones(1, 2, 3))
    whole = Observation(WHOLE, torch.ones(1, 3), axes=TERM_SPEC.axes)
    read = node_readings(CONTEXT, a_trace(nodes, whole))

    assert [site for site, _ in read] == [NODES]


def test_node_encodings_published_and_never_read_are_an_error() -> None:
    with pytest.raises(ValueError, match="read none of them"):
        node_readings(CONTEXT, a_trace())


def test_a_component_publishing_no_node_encoding_leaves_nothing_to_read() -> None:
    assert node_readings(DiagnosticContext({CELL: CELL_SPEC}), a_trace()) == ()


def test_a_diagnostic_built_without_terms_cannot_measure_a_reading() -> None:
    with pytest.raises(ValueError, match="without the terms"):
        NodeStructure(None, a_reading(torch.ones(1, 2, 3)))


def test_a_batch_holding_no_term_cannot_measure_a_reading() -> None:
    with pytest.raises(ValueError, match="no term"):
        NodeStructure(BatchEncoding(()), a_reading(torch.ones(1, 2, 3)))


def test_terms_and_a_reading_of_different_extents_name_both_shapes() -> None:
    terms = BatchEncoding((a_chain(2, 2, (5, 3)),))

    with pytest.raises(ValueError, match="1 terms of 2 slots") as mismatch:
        NodeStructure(terms, a_reading(torch.ones(1, 4, 3)))

    assert "4 slots" in str(mismatch.value)


def test_a_padded_slot_holds_no_node_and_is_not_live() -> None:
    terms = BatchEncoding((a_chain(2, 4, (5, 3)),))
    structure = NodeStructure(terms, a_reading(torch.ones(1, 4, 3)))

    assert structure.live.tolist() == [[True, True, False, False]]
    assert structure.counts().tolist() == [2]


def test_a_slot_the_reading_masks_off_is_not_live_either() -> None:
    terms = BatchEncoding((a_chain(3, 3, (5, 3, 3)),))
    covered = torch.tensor([[True, False, True]])
    structure = NodeStructure(terms, a_reading(torch.ones(1, 3, 2), covered))

    assert structure.live.tolist() == [[True, False, True]]


def test_a_slot_with_a_live_slot_hanging_off_it_is_complex() -> None:
    terms = BatchEncoding((a_chain(3, 4, (5, 4, 3)),))
    structure = NodeStructure(terms, a_reading(torch.ones(1, 4, 2)))

    assert structure.internal.tolist() == [[True, True, False, False]]
    assert structure.depths.tolist() == [[0, 1, 2, 0]]
    assert structure.symbols.tolist() == [[5, 4, 3, 0]]


def test_a_slot_whose_only_child_is_masked_off_is_atomic() -> None:
    terms = BatchEncoding((a_chain(2, 2, (5, 3)),))
    covered = torch.tensor([[True, False]])
    structure = NodeStructure(terms, a_reading(torch.ones(1, 2, 2), covered))

    assert structure.internal.tolist() == [[False, False]]


def test_the_root_is_the_live_slot_with_no_parent_above_it() -> None:
    terms = BatchEncoding((a_chain(2, 2, (5, 3)),))
    reading = a_reading(torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]))
    present, roots = NodeStructure(terms, reading).rooted()

    assert present.tolist() == [True]
    assert roots.tolist() == [[1.0, 0.0]]


def test_a_term_whose_root_is_masked_off_holds_none() -> None:
    terms = BatchEncoding((a_chain(2, 2, (5, 3)),))
    covered = torch.tensor([[False, True]])
    reading = a_reading(torch.ones(1, 2, 2), covered)
    present, _ = NodeStructure(terms, reading).rooted()

    assert present.tolist() == [False]


def test_a_structure_says_how_much_of_the_batch_holds_a_node() -> None:
    terms = BatchEncoding((a_chain(2, 3, (5, 3)), a_chain(3, 3, (5, 4, 3))))
    structure = NodeStructure(terms, a_reading(torch.ones(2, 3, 2)))

    assert repr(structure) == "NodeStructure(2 terms, 3 slots, 5 live)"
