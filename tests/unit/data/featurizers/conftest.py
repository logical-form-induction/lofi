from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch

from lofi.data import (
    CodeTensorizer,
    Derivation,
    DerivationNode,
    Inventory,
    Item,
    ItemHasher,
    ItemId,
    Token,
    TokenSpan,
)
from lofi.registry import plugin_for

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.data._join import ItemLookup
    from lofi.data._ports import FeaturizerComponent
    from lofi.spec import ConfiguredSpec, FeaturizerSpec

PAD: Final = "<pad>"


class TorchCodeTensorizer:
    """A tensorizer for padded integer rows and boolean presence masks."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max((len(row) for row in rows), default=0)
        vectors = torch.full((len(rows), width), padding, dtype=torch.long)
        present = torch.zeros((len(rows), width), dtype=torch.bool)

        for position, row in enumerate(rows):
            if row:
                vectors[position, : len(row)] = torch.tensor(row, dtype=torch.long)
                present[position, : len(row)] = True

        return vectors, present


class Resources(NamedTuple):
    """Dataset resources a featurizer is built over."""

    hasher: ItemHasher
    tensorizer: CodeTensorizer
    lookup: ItemLookup


def wired(spec: FeaturizerSpec, resources: Resources) -> FeaturizerComponent:
    """Build one declared featurizer through its plugin over the resources.

    Parameters
    ----------
    spec : FeaturizerSpec
        Validated component declaration.
    resources : Resources
        Dataset resources the featurizer is built over.

    Returns
    -------
    FeaturizerComponent
        The constructed featurizer.
    """
    declaration = cast("ConfiguredSpec[object]", spec)

    return cast(
        "FeaturizerComponent",
        plugin_for(declaration).construct(declaration, resources),
    )


@pytest.fixture
def hasher() -> ItemHasher:
    return ItemHasher(
        Inventory("words", ("the", "dog", "runs"), (PAD,)),
        Inventory("categories", ("S", "NP", "N", "NP/N", "S\\NP")),
    )


@pytest.fixture
def tensorize() -> CodeTensorizer:
    return TorchCodeTensorizer()


@pytest.fixture
def item_factory() -> Callable[..., Item]:
    def item(handle: str, *words: str) -> Item:
        return Item(
            ItemId(handle),
            tuple(Token(index, word) for index, word in enumerate(words)),
        )

    return item


@pytest.fixture
def derivation() -> Derivation[str]:
    return Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 3), "S", children=(1, 4)),
            1: DerivationNode(TokenSpan(0, 2), "NP", children=(2, 3)),
            2: DerivationNode(TokenSpan(0, 1), "NP/N"),
            3: DerivationNode(TokenSpan(1, 2), "N"),
            4: DerivationNode(TokenSpan(2, 3), "S\\NP"),
        },
    )
