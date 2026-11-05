from enum import Enum, auto
from typing import Final, cast

import pytest
import torch

from lofi.lf import THREE_WAY, InferenceLabel
from lofi.nn._values import Encoded
from lofi.nn.entailment import EntailmentHead, PairedVectors

WIDTH: Final = 4
HIDDEN: Final = 6


class Polarity(Enum):
    YES = auto()
    NO = auto()


class Empty(Enum):
    pass


def vectors(*, pairs: int = 3, mask: torch.Tensor | None = None) -> PairedVectors:
    generator = torch.Generator().manual_seed(19)
    return PairedVectors(
        Encoded(
            torch.randn(pairs, WIDTH, generator=generator), ("pair", "width"), mask
        ),
        Encoded(
            torch.randn(pairs, WIDTH, generator=generator), ("pair", "width"), mask
        ),
    )


def test_head_predicts_one_normalized_distribution_per_pair() -> None:
    head = EntailmentHead(THREE_WAY.labels, width=WIDTH, hidden=HIDDEN)
    prediction = head.predict(vectors())

    assert prediction.axes == ("pair", "label")
    assert prediction.normalized_over == "label"
    assert prediction.tensor.exp().sum(1).tolist() == pytest.approx([1.0] * 3)
    assert head.labels == THREE_WAY.labels


def test_head_uses_the_declared_label_order() -> None:
    head = EntailmentHead(tuple(Polarity), width=WIDTH)

    assert head.column(Polarity.YES) == 0
    assert head.column(Polarity.NO) == 1

    with pytest.raises(ValueError, match="not among the labels"):
        head.column(cast("Polarity", InferenceLabel.NEUTRAL))


def test_masked_pairs_remain_masked() -> None:
    mask = torch.tensor([True, False, True])
    prediction = EntailmentHead(THREE_WAY.labels, width=WIDTH).predict(
        vectors(mask=mask)
    )

    assert prediction.mask is not None
    assert torch.equal(prediction.mask, mask)
    assert torch.isneginf(prediction.tensor[1]).all()


def test_empty_labels_and_invalid_widths_are_refused() -> None:
    with pytest.raises(ValueError, match="declares no label"):
        EntailmentHead(tuple(Empty), width=WIDTH)

    with pytest.raises(ValueError, match="zero"):
        EntailmentHead(THREE_WAY.labels, width=0)

    with pytest.raises(ValueError, match="zero"):
        EntailmentHead(THREE_WAY.labels, width=WIDTH, hidden=0)


def test_pair_vectors_validate_axes_and_shape() -> None:
    left = Encoded(torch.ones(2, WIDTH), ("item", "width"))
    right = Encoded(torch.ones(2, WIDTH), ("pair", "width"))

    with pytest.raises(ValueError, match="premise"):
        PairedVectors(left, right)

    with pytest.raises(ValueError, match="shape"):
        PairedVectors(
            Encoded(torch.ones(2, WIDTH), ("pair", "width")),
            Encoded(torch.ones(3, WIDTH), ("pair", "width")),
        )
