import pytest
import torch
from torch.nn import functional

from lofi.nn._layers import activation
from lofi.nn.ports import StandardDeclaration
from lofi.nn.sites import Role
from lofi.nn.transformer import (
    BLOCK_INPUT,
    BLOCK_OUTPUT,
    BLOCK_POST_ATTENTION,
    AttentionBlock,
    transformer_structure,
)
from lofi.registry import RegistryError


@pytest.mark.parametrize(
    ("name", "standard"),
    [
        ("positions", StandardDeclaration.POSITIONS),
        ("positions-with-depth", StandardDeclaration.POSITIONS_WITH_DEPTH),
        ("ancestor-mask", StandardDeclaration.ANCESTOR_MASK),
        ("ancestor-and-sibling-bias", StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS),
        ("parent-recurrence", StandardDeclaration.PARENT_RECURRENCE),
    ],
)
def test_one_family_installs_every_standard_structure(
    name: str, standard: StandardDeclaration
) -> None:
    assert dict(transformer_structure(name)) == dict(standard.declaration)


def test_an_uninstalled_structure_is_refused_by_the_family() -> None:
    with pytest.raises(RegistryError, match="transformer_structure"):
        transformer_structure("unsupported")


def test_a_block_with_every_slot_live_needs_no_occupancy() -> None:
    torch.manual_seed(0)
    block = AttentionBlock(8, 2, 16, activation("gelu"))
    stream = torch.randn(3, 5, 8)
    attention = torch.zeros(1, 1, 5, 5)

    torch.testing.assert_close(
        block(stream, attention),
        block(stream, attention, occupied=torch.ones(3, 5, dtype=torch.bool)),
    )


def test_a_block_attends_as_scaled_dot_product_attention() -> None:
    torch.manual_seed(1)
    block = AttentionBlock(8, 2, 16, activation("gelu"))
    stream = torch.randn(2, 4, 8)
    attention = torch.zeros(1, 1, 4, 4)
    attention[..., 0, 1:] = -torch.inf
    normed = block.get_submodule("_attention_norm")(stream)
    queries, keys, values = (
        block.get_submodule("_queries_keys_values")(normed)
        .reshape(2, 4, 3, 2, 4)
        .permute(2, 0, 3, 1, 4)
    )
    mixed = functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=attention
    )
    attended = stream + block.get_submodule("_merge")(
        mixed.transpose(1, 2).reshape(2, 4, 8)
    )
    expected = attended + block.get_submodule("_contract")(
        functional.gelu(
            block.get_submodule("_expand")(
                block.get_submodule("_feedforward_norm")(attended)
            )
        )
    )

    torch.testing.assert_close(block(stream, attention), expected)


def test_an_unoccupied_slot_carries_nothing_and_every_stage_is_reported() -> None:
    block = AttentionBlock(4, 2, 8, activation("gelu"))
    occupied = torch.tensor([[True, True, False]])
    seen: list[tuple[Role, tuple[int | str, ...]]] = []

    def observe(
        role: Role,
        key: tuple[int | str, ...],
        value: torch.Tensor,
        live: torch.Tensor | None,
    ) -> None:
        del value
        assert live is occupied
        seen.append((role, key))

    output = block(
        torch.randn(1, 3, 4),
        torch.zeros(1, 1, 3, 3),
        occupied=occupied,
        layer=2,
        observe=observe,
    )

    assert bool((output[0, 2] == 0.0).all())
    assert seen == [
        (Role.RESIDUAL_STREAM, (2, BLOCK_INPUT)),
        (Role.ATTENTION_LOGITS, (2, 0)),
        (Role.ATTENTION_WEIGHTS, (2, 0)),
        (Role.ATTENTION_LOGITS, (2, 1)),
        (Role.ATTENTION_WEIGHTS, (2, 1)),
        (Role.RESIDUAL_STREAM, (2, BLOCK_POST_ATTENTION)),
        (Role.FFN_PREACTIVATION, (2,)),
        (Role.RESIDUAL_STREAM, (2, BLOCK_OUTPUT)),
    ]
