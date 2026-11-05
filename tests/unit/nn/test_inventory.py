from typing import Final

import pytest
import torch

from lofi.nn.inventory import EntangledInventory, InventoryTopology, LevelBatch
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

ATOM: Final = Operator("a", "T", family="primitive")
PAIR: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature("inventory", ("T",), (ATOM, PAIR))
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=2),
)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": CODEC.capacity})
STORE: Final = TermStore(SIGNATURE)
LEAF: Final = STORE.op(ATOM)
TERM: Final = STORE.op(PAIR, LEAF, LEAF)


def test_equal_addressed_subterms_are_evaluated_once_and_restored_per_row() -> None:
    terms = CODEC.encode_batch((TERM, TERM))
    topology = InventoryTopology(terms, CHOICES)
    calls: list[LevelBatch] = []

    def encode(batch: LevelBatch) -> torch.Tensor:
        calls.append(batch)
        child_total = (
            torch.zeros(len(batch.choices), dtype=torch.float32)
            if batch.children is None
            else batch.children.sum(dim=(1, 2))
        )

        return torch.stack((batch.choices.to(torch.float32), child_total), dim=1)

    embeddings = EntangledInventory(topology)(encode)

    assert topology.node_count == sum(terms.rows[0].occupied)
    assert len(calls) == len(topology.levels)
    assert torch.equal(embeddings.placed()[0], embeddings.placed()[1])
    assert torch.equal(embeddings.terms[0], embeddings.terms[1])


def test_an_empty_batch_builds_an_empty_inventory() -> None:
    topology = InventoryTopology(BatchEncoding(()), CHOICES)
    embeddings = EntangledInventory(topology)(
        lambda batch: torch.empty((len(batch.choices), 3))
    )

    assert topology.node_count == 0
    assert embeddings.nodes.shape == (0, 0)
    assert embeddings.terms.shape == (0, 0)
    assert embeddings.placed().shape == (0, 0, 0)


def test_a_level_encoder_must_return_one_floating_vector_per_node() -> None:
    inventory = EntangledInventory(
        InventoryTopology(CODEC.encode_batch((TERM,)), CHOICES)
    )

    with pytest.raises(TypeError, match="floating point"):
        inventory(lambda batch: torch.zeros(len(batch.choices), 2, dtype=torch.int64))

    with pytest.raises(ValueError, match="node by width"):
        inventory(lambda batch: torch.zeros(len(batch.choices)))
