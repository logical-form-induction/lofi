"""Entailment prediction from paired structured or raw sentence representations.

Each head declares whether it reads expected logical form embeddings or the raw
token-encoder baseline.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Final, override

import torch
from torch import nn

from lofi.nn._guards import WIDTH_AXIS, positive
from lofi.nn._values import LogDistribution

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import Encoded

__all__ = [
    "EntailmentHead",
    "PairedVectors",
]

_PAIR_AXIS: Final[str] = "pair"
"""The axis the pairs of a batch run along."""

_LABEL_AXIS: Final[str] = "label"
"""The axis the entailment labels run along."""

_VECTOR_AXES: Final[tuple[str, ...]] = (_PAIR_AXIS, WIDTH_AXIS)
"""The axes of one vector per pair."""

_LABEL_AXES: Final[tuple[str, ...]] = (_PAIR_AXIS, _LABEL_AXIS)
"""The axes of one distribution over labels per pair."""


class PairedVectors:
    """One vector for the premise and one for the hypothesis of each pair.

    Parameters
    ----------
    premise : Encoded
        The premise of each pair, over the axes ``("pair", "width")``, with a mask
        over the pair axis or none.
    hypothesis : Encoded
        The hypothesis of each pair, in the order of the pair axis, of the premise's
        own shape and mask.

    Raises
    ------
    ValueError
        If either side is not over those axes, if a mask reaches into the feature
        axis, or if the two sides disagree about their shape or their mask.
    """

    __slots__ = ("_hypothesis", "_premise")

    def __init__(self, premise: Encoded, hypothesis: Encoded) -> None:
        for name, side in (("premise", premise), ("hypothesis", hypothesis)):
            if side.axes != _VECTOR_AXES:
                msg = (
                    f"the {name} of each pair runs over {_VECTOR_AXES}, not {side.axes}"
                )
                raise ValueError(msg)

            if side.mask is not None and side.mask.ndim != 1:
                msg = (
                    f"a pair is live or padded whole, so the {name} mask covers "
                    f"{_PAIR_AXIS!r} alone and not {side.mask_axes}"
                )
                raise ValueError(msg)

        if premise.tensor.shape != hypothesis.tensor.shape:
            msg = (
                f"the premises are of shape {tuple(premise.tensor.shape)} and the "
                f"hypotheses of shape {tuple(hypothesis.tensor.shape)}"
            )
            raise ValueError(msg)

        if not _same_mask(premise.mask, hypothesis.mask):
            msg = "the premises and the hypotheses are padded at different pairs"
            raise ValueError(msg)

        self._premise = premise
        self._hypothesis = hypothesis

    @property
    def premise(self) -> Encoded:
        """Return the premise of each pair."""
        return self._premise

    @property
    def hypothesis(self) -> Encoded:
        """Return the hypothesis of each pair, in the order of the pair axis."""
        return self._hypothesis

    @property
    def mask(self) -> Tensor | None:
        """Return which pairs hold data, or None where every pair does."""
        return self._premise.mask

    @property
    def pair_count(self) -> int:
        """Return how many pairs there are, padded ones included."""
        return self._premise.size(_PAIR_AXIS)

    @property
    def width(self) -> int:
        """Return the size of the feature axis."""
        return self._premise.size(WIDTH_AXIS)

    @override
    def __repr__(self) -> str:
        """Return the pair count, the width and whether any pair is padded."""
        return (
            f"PairedVectors(pairs={self.pair_count}, width={self.width}, "
            f"masked={self.mask is not None})"
        )


def _same_mask(first: Tensor | None, second: Tensor | None) -> bool:
    """Return whether two masks pad the same positions.

    Parameters
    ----------
    first : Tensor | None
        One mask, or None where nothing is padded.
    second : Tensor | None
        The other.
    """
    if first is None or second is None:
        return first is None and second is None

    return first.shape == second.shape and bool(torch.equal(first, second))


class EntailmentHead[LabelT: Hashable](nn.Module):
    """Predicts a distribution over entailment labels for each sentence pair.

    The head reads one representation of each pair, the one it declares, and
    scores the pair from the two vectors, their product and their absolute
    difference through one hidden layer. Its columns are the members of the label
    enumeration in definition order.

    Parameters
    ----------
    labels : tuple[LabelT, ...]
        Labels in the order the output columns take.
    width : int
        How wide the vectors of the declared representation are.
    hidden : int
        How wide the layer between the pair features and the label scores is.

    Raises
    ------
    TypeError
        If a width is not an integer.
    ValueError
        If the enumeration has no member, or a width is not positive.
    """

    def __init__(
        self,
        labels: tuple[LabelT, ...],
        *,
        width: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()

        members = tuple(labels)

        if not members:
            msg = "an entailment head declares no label, so there is nothing to predict"
            raise ValueError(msg)

        if len(set(members)) != len(members):
            msg = f"an entailment head repeats a label: {members!r}"
            raise ValueError(msg)

        vector = positive("a vector's width", width)
        inner = positive("the hidden width", hidden)

        # the two vectors, their product and their absolute difference
        self._read = nn.Linear(4 * vector, inner)
        self._score = nn.Linear(inner, len(members))
        self._labels = members
        self._width = vector

    @property
    def labels(self) -> tuple[LabelT, ...]:
        """Return the labels, in the order the columns take."""
        return self._labels

    @property
    def input_width(self) -> int:
        """Return the size of the feature axis of the vectors this head reads."""
        return self._width

    def column(self, label: LabelT) -> int:
        """Return the column a label's score sits in.

        Parameters
        ----------
        label : LabelT
            The label to locate.

        Returns
        -------
        int
            Its position along the label axis.

        Raises
        ------
        ValueError
            If the label is not one of this head's.
        """
        try:
            return self._labels.index(label)

        except ValueError as error:
            msg = f"{label!r} is not among the labels {self._labels}"
            raise ValueError(msg) from error

    def predict(self, pairs: PairedVectors, /) -> LogDistribution:
        """Return the log probability of each label for each pair.

        Parameters
        ----------
        pairs : PairedVectors
            One vector for each side of every pair.

        Returns
        -------
        LogDistribution
            Rank 2 over the axes ``("pair", "label")``, at the precision this head's
            own parameters hold, normalized over the label axis at every live pair,
            masked over the pair axis as the batch is and negative infinity
            throughout a padded pair.

        Raises
        ------
        ValueError
            If the pair vectors have another width than this head reads.
        """
        if pairs.width != self._width:
            msg = f"this head reads vectors {self._width} wide, not {pairs.width} wide"
            raise ValueError(msg)

        premise = pairs.premise.tensor
        hypothesis = pairs.hypothesis.tensor
        features = torch.cat(
            (premise, hypothesis, premise * hypothesis, (premise - hypothesis).abs()),
            dim=-1,
        )
        scores = self._score(torch.tanh(self._read(features)))
        weights = scores.log_softmax(dim=-1)
        mask = pairs.mask

        if mask is not None:
            weights = torch.where(mask.unsqueeze(-1), weights, -torch.inf)

        return LogDistribution(weights, _LABEL_AXES, mask, normalized_over=_LABEL_AXIS)
