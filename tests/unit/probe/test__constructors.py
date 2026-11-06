from typing import Final

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._constructors import ConstructorEmbedding
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._structure import NODE_AXES
from lofi.probe._values import Observation, Trace
from lofi.terms import BatchEncoding, Encoding

ENCODER: Final = ComponentId("transformer")
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


def a_chain(symbols: tuple[int, ...]) -> Encoding:
    length = len(symbols)
    filling = SLOTS - length

    return Encoding(
        symbols + (0,) * filling,
        tuple(range(-1, length - 1)) + (Encoding.NO_PARENT,) * filling,
        tuple(range(length)) + (0,) * filling,
        (True,) * length + (False,) * filling,
    )


# two slots under each of two constructors, so two pairs share and four differ
PAIRED: Final = BatchEncoding((a_chain((3, 3, 5, 5)),))
UNIFORM: Final = BatchEncoding((a_chain((3, 3, 3, 3)),))


def a_trace(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> Trace:
    trace = Trace()
    trace.record(Observation(NODES, tensor, axes=NODE_AXES, mask=mask))
    trace.close()

    return trace


def by_constructor() -> torch.Tensor:
    # the two slots of each constructor alike, and the constructors at right angles
    return torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])


def against_constructor() -> torch.Tensor:
    # the two slots of each constructor opposed, and the constructors at right angles
    return torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])


def indifferent() -> torch.Tensor:
    return torch.ones(1, SLOTS, 2)


def test_every_pair_of_live_slots_is_counted_once_and_split_by_constructor() -> None:
    result = ConstructorEmbedding(PAIRED).run(
        a_trace(by_constructor()), context=CONTEXT
    )

    assert result.tag == "constructor-embedding"
    assert result.value(NODES, "sharing-pairs") == 2.0
    assert result.value(NODES, "differing-pairs") == 4.0


def test_slots_sharing_a_constructor_encoded_alike_agree_wholly() -> None:
    result = ConstructorEmbedding(PAIRED).run(
        a_trace(by_constructor()), context=CONTEXT
    )

    assert result.value(NODES, "constructor-agreement") == pytest.approx(1.0)


def test_slots_encoded_against_their_constructor_agree_not_at_all() -> None:
    result = ConstructorEmbedding(PAIRED).run(
        a_trace(against_constructor()), context=CONTEXT
    )

    assert result.value(NODES, "constructor-agreement") == pytest.approx(0.0)


def test_an_encoder_blind_to_the_constructor_lands_on_the_coin_toss() -> None:
    result = ConstructorEmbedding(PAIRED).run(a_trace(indifferent()), context=CONTEXT)

    assert result.value(NODES, "constructor-agreement") == pytest.approx(0.5)


def test_scaling_every_encoding_leaves_the_agreement_where_it_was() -> None:
    scaled = ConstructorEmbedding(PAIRED).run(
        a_trace(by_constructor() * 1e6), context=CONTEXT
    )

    assert scaled.value(NODES, "constructor-agreement") == pytest.approx(1.0)


def test_a_reading_under_one_constructor_alone_reports_no_agreement() -> None:
    result = ConstructorEmbedding(UNIFORM).run(
        a_trace(by_constructor()), context=CONTEXT
    )

    assert result.value(NODES, "sharing-pairs") == 6.0
    assert result.value(NODES, "differing-pairs") == 0.0

    with pytest.raises(KeyError, match="constructor-agreement"):
        result.value(NODES, "constructor-agreement")


def test_a_slot_the_reading_masks_off_enters_no_pair() -> None:
    covered = torch.tensor([[True, True, True, False]])
    result = ConstructorEmbedding(PAIRED).run(
        a_trace(by_constructor(), covered), context=CONTEXT
    )

    assert result.value(NODES, "sharing-pairs") == 1.0
    assert result.value(NODES, "differing-pairs") == 2.0


def test_a_diagnostic_built_without_terms_refuses_a_reading() -> None:
    with pytest.raises(ValueError, match="without the terms"):
        ConstructorEmbedding().run(a_trace(by_constructor()), context=CONTEXT)


def test_the_diagnostic_resolves_only_where_type_encodings_are_published() -> None:
    without = installed_diagnostics().matching(DiagnosticContext({CELL: CELL_SPEC}))

    assert "constructor-embedding" not in [one.tag for one in without]
    assert "constructor-embedding" in [
        one.tag for one in installed_diagnostics().matching(CONTEXT)
    ]


def test_the_diagnostic_says_what_it_holds_and_which_axes_it_reads() -> None:
    held = repr(ConstructorEmbedding(PAIRED))

    assert held == f"ConstructorEmbedding(1 terms, axes={NODE_AXES!r})"
