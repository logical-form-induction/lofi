from types import MappingProxyType
from typing import Literal

import pytest
import torch

from lofi.nn.heads import TermHeads
from lofi.terms import (
    BindingSite,
    ChoiceIndex,
    HoleContext,
    Operator,
    Signature,
    Vocabulary,
)


def language_index() -> ChoiceIndex:
    """Return an index with payload-free, constant and variable choices."""
    signature = Signature(
        "heads",
        ("Ty", "Tm"),
        (
            Operator("e", "Ty", family="primitive"),
            Operator("t", "Ty", family="primitive"),
            Operator(
                "lam",
                "Tm",
                arg_sorts=("Tm",),
                binding=(BindingSite((0,), "Tm"),),
                family="binder",
            ),
            Operator("eval", "Tm", arg_sorts=("Tm", "Tm"), family="application"),
        ),
        vocabularies=(Vocabulary("Tm", ("dog", "bark")),),
        open_sorts=("Tm",),
    )
    return ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Tm": 2},
    )


def context(
    sort: str,
    *,
    remaining: int,
    in_scope: int = 0,
    free_used: int = 0,
) -> HoleContext:
    """Return one decoder hole context."""
    scope = {} if in_scope == 0 else {sort: in_scope}
    free = {} if free_used == 0 else {sort: free_used}
    return HoleContext(
        (),
        sort,
        MappingProxyType(scope),
        MappingProxyType(free),
        remaining,
    )


@pytest.mark.parametrize("kind", ["flat", "factored"])
def test_term_heads_normalize_only_over_choices_legal_at_each_hole(
    kind: Literal["flat", "factored"],
) -> None:
    index = language_index()
    heads = TermHeads(index=index, input_width=4, kind=kind)
    states = torch.zeros((2, 4))
    rows = heads(
        states,
        (
            context("Ty", remaining=0),
            context("Tm", remaining=2, in_scope=1),
        ),
    )

    assert rows.axes == ("term", "choice")
    assert rows.normalized_over == "choice"
    assert rows.mask is not None
    assert torch.allclose(rows.tensor.exp().sum(dim=1), torch.ones(2))
    assert not bool(rows.mask[0, index.index("eval")])
    assert bool(rows.mask[1, index.index("eval")])
    assert bool(rows.mask[1, index.index("#bvar[Tm]", 0)])
    assert not bool(rows.mask[1, index.index("#bvar[Tm]", 1)])
    assert bool(rows.mask[1, index.index("#fvar[Tm]", 0)])
    assert not bool(rows.mask[1, index.index("#fvar[Tm]", 1)])


def test_factored_heads_are_derived_from_the_choice_families() -> None:
    index = language_index()
    heads = TermHeads(index=index, input_width=3, kind="factored")

    assert heads.index is index
    assert heads.families == index.family_slices
    assert {
        tuple(parameter.shape)
        for name, parameter in heads.named_parameters()
        if "families" in name and name.endswith("weight")
    } == {(2, 3), (5, 3)}


def test_flat_logits_are_the_single_readout_masked_by_frontier_contexts() -> None:
    index = language_index()
    heads = TermHeads(index=index, input_width=3, kind="flat")
    states = torch.randn((2, 3))
    contexts = (
        context("Ty", remaining=0),
        context("Tm", remaining=2, in_scope=1),
    )
    rows = heads(states, contexts)
    logits = heads.logits(states)

    if rows.mask is None:
        pytest.fail("frontier rows did not publish their legal-choice mask")

    expected = logits.masked_fill(~rows.mask, -torch.inf).log_softmax(dim=-1)

    assert torch.allclose(rows.tensor, expected)


def test_a_head_refuses_a_context_whose_sort_the_language_does_not_declare() -> None:
    heads = TermHeads(index=language_index(), input_width=2, kind="flat")

    with pytest.raises(ValueError, match="undeclared sort 'Other'"):
        heads(torch.zeros((1, 2)), (context("Other", remaining=1),))
