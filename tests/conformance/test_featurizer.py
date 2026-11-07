from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.featurizer import (
    FEATURIZER_LAWS,
    FeaturizerConformance,
    FeaturizerSuite,
)

from lofi._checks import LawViolationError
from lofi.data import (
    Example,
    Featurizer,
    Inventory,
    Item,
    ItemHasher,
    ItemId,
    Scalar,
    Scale,
    Supervision,
    Token,
    Unlabeled,
)
from lofi.data.featurizers.unlabeled import UnlabeledFeaturizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data.featurizers import FeaturizedBatch

PAD: Final = "<pad>"
SCALE: Final = Scale("ordinal", 1.0, 7.0)
FIRST: Final = Item(ItemId("first"), (Token(0, "dog"),))
SECOND: Final = Item(ItemId("second"), (Token(0, "runs"),))


class Tensorizer:
    """Torch tensorizer for the conformance fixture."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad integer rows and return their presence mask.

        Parameters
        ----------
        rows : Sequence[tuple[int, ...]]
            Integer rows.
        padding : int
            Padding code.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Padded integer rows and a boolean presence mask.
        """
        width = max((len(row) for row in rows), default=0)
        values = torch.full((len(rows), width), padding, dtype=torch.long)
        present = torch.zeros((len(rows), width), dtype=torch.bool)

        for position, row in enumerate(rows):
            if row:
                values[position, : len(row)] = torch.tensor(row)
                present[position, : len(row)] = True

        return values, present


def _conformance() -> FeaturizerConformance:
    """Return a complete unlabeled-featurizer fixture."""
    hasher = ItemHasher(
        Inventory("words", ("dog", "runs"), (PAD,)),
        Inventory("categories", ()),
    )
    subject = cast(
        "Featurizer[Supervision[str], FeaturizedBatch[Supervision[str]]]",
        UnlabeledFeaturizer(hasher, PAD, Tensorizer()),
    )
    examples: tuple[Example[Supervision[str]], ...] = (
        Example(FIRST, Unlabeled()),
        Example(SECOND, Unlabeled()),
    )
    return FeaturizerConformance(
        subject,
        examples,
        tuple(example.supervision for example in examples),
        ((FIRST.item_id,), (SECOND.item_id,)),
        Example(FIRST, Scalar(4.0, SCALE)),
    )


class TestUnlabeledReference(FeaturizerSuite):
    @pytest.fixture
    def conformance(self) -> FeaturizerConformance:
        """Return the reference fixture."""
        return _conformance()


def test_every_law_holds_for_the_reference_featurizer() -> None:
    fixture = _conformance()

    for law in FEATURIZER_LAWS:
        law(fixture)


def test_the_member_law_rejects_a_wrong_expected_item() -> None:
    fixture = _conformance()
    broken = FeaturizerConformance(
        fixture.subject,
        fixture.examples,
        fixture.supervision,
        ((SECOND.item_id,), (FIRST.item_id,)),
        fixture.incompatible,
    )

    with pytest.raises(LawViolationError, match="member groups"):
        FEATURIZER_LAWS[1](broken)
