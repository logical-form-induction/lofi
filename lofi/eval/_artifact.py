"""The hypothesis-only featurization and the label head an artifact control fits.

A model given only the hypothesis reaches roughly two thirds accuracy on SNLI
from annotation artifacts alone, so an inference accuracy with nothing beside
it may be reporting the annotation rather than the inference. The control
withholds the premise at the featurizer rather than downstream: the item its
featurizer takes carries a hypothesis and nothing else, so there is no premise
field to mask and no mask to leak through.
"""

from collections import Counter
from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._inference import UndeclaredLabelError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


__all__ = ["Hypothesis", "HypothesisFeaturizer", "LabelHead"]

_HYPOTHESIS_SIDE: Final = "hypothesis"
"""The side a hypothesis token is tagged with."""


class Hypothesis(NamedTuple):
    """One pair's hypothesis, as the control's featurizer receives it.

    The value has one field. The control cannot leak a premise it was never
    handed, which is why the withholding happens in the type the featurizer
    takes rather than in a mask over a premise the featurizer was given.

    Parameters
    ----------
    tokens : tuple[str, ...]
        The hypothesis's tokens.
    """

    tokens: tuple[str, ...]


def _sided(side: str, tokens: Sequence[str]) -> Counter[str]:
    """Return the tokens of one sentence, counted under the side they came from.

    Parameters
    ----------
    side : str
        Which sentence the tokens are from.
    tokens : Sequence[str]
        The sentence's tokens.

    Returns
    -------
    Counter[str]
        Counts keyed by the tagged token.
    """
    return Counter(
        f"{side}:{ensure_name(f'a {side} token', token)}" for token in tokens
    )


class HypothesisFeaturizer:
    """Counts the tokens of a hypothesis, tagged with the side they came from.

    Attributes
    ----------
    name : str
        The featurization's name.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the featurization's name."""
        return "hypothesis-tokens"

    def __call__(self, item: Hypothesis, /) -> Mapping[str, int]:
        """Return the counted tokens of the hypothesis.

        Parameters
        ----------
        item : Hypothesis
            The hypothesis to featurize.

        Returns
        -------
        Mapping[str, int]
            Counts keyed by tagged token, none of which names the premise.

        Raises
        ------
        TypeError
            If the value is not a hypothesis. A labeled pair does not reach
            this featurizer, which is what withholds the premise.
        ValueError
            If a token is empty.
        """
        if not isinstance(item, Hypothesis):
            msg = (
                f"the hypothesis-only featurizer was handed {item!r}, which is "
                f"not a hypothesis; it takes no premise, so a pair carrying one "
                f"does not reach it"
            )
            raise TypeError(msg)

        return MappingProxyType(dict(_sided(_HYPOTHESIS_SIDE, item.tokens)))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this featurizer."""
        return "HypothesisFeaturizer()"


class LabelHead(FrozenValue):
    """A bag-of-features classifier fitted over a declared label inventory.

    The head is what the comparison holds constant: the control and the paired
    arm differ in the featurization they are fitted over, not in how a label is
    chosen. Fitting is one pass of counts with add-one smoothing over the
    fitted vocabulary, so it is exact, takes no seed and no iteration count,
    and a second head over the same split is the same head. A feature the
    fitted split never carried is ignored at prediction rather than smoothed,
    since the head has no evidence about it either way. Where two labels score
    equally the most frequent fitted label wins, and where that is not among
    them the first in ascending order does.

    Attributes
    ----------
    inventory : frozenset[str]
        The declared label inventory.
    fitted : int
        How many items the head was fitted on.
    vocabulary : frozenset[str]
        The features the fitted split carried.
    majority : str
        The most frequent label of the fitted split.

    Parameters
    ----------
    features : Sequence[Mapping[str, int]]
        One item's features per fitted item, in split order.
    labels : Sequence[str]
        The gold label of each fitted item, in the same order.
    inventory : frozenset[str]
        The declared label inventory.

    Raises
    ------
    TypeError
        If a feature count is not an integer.
    ValueError
        If fewer than two labels are declared, the features and labels differ
        in length, nothing was fitted, a name is empty, or a declared label is
        absent from the fitted split.
    UndeclaredLabelError
        If a fitted label lies outside the inventory.
    """

    __slots__ = ("_counts", "_fitted", "_inventory", "_seen", "_totals", "_vocabulary")
    _noun = "fitted inference label head"

    def __init__(
        self,
        features: Sequence[Mapping[str, int]],
        labels: Sequence[str],
        *,
        inventory: frozenset[str],
    ) -> None:
        declared = frozenset(
            ensure_name("a declared label", label) for label in inventory
        )

        if len(declared) < 2:
            msg = (
                f"a head was declared over {len(declared)} labels and "
                f"classifies nothing"
            )
            raise ValueError(msg)

        if len(features) != len(labels):
            msg = (
                f"a head was fitted on {len(features)} featurized items and "
                f"{len(labels)} labels"
            )
            raise ValueError(msg)

        if not labels:
            msg = "a head fitted on no item classifies nothing"
            raise ValueError(msg)

        counts: dict[str, Counter[str]] = {label: Counter() for label in declared}
        seen: Counter[str] = Counter()

        for row, label in zip(features, labels, strict=True):
            if label not in declared:
                msg = (
                    f"the fitted label {label!r} is not in the declared "
                    f"inventory {sorted(declared)}"
                )
                raise UndeclaredLabelError(msg)

            seen[label] += 1

            for feature, count in row.items():
                counts[label][ensure_name("a feature", feature)] += ensure_non_negative(
                    f"the count of feature {feature!r}", count
                )

        absent = sorted(label for label in declared if seen[label] == 0)

        if absent:
            msg = (
                f"the fitted split carries no item labeled {absent}, so those "
                f"labels have no prior to fit"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_inventory", declared)
        object.__setattr__(self, "_fitted", len(labels))
        object.__setattr__(self, "_seen", MappingProxyType(dict(seen)))
        object.__setattr__(
            self,
            "_counts",
            MappingProxyType(
                {label: MappingProxyType(dict(row)) for label, row in counts.items()}
            ),
        )
        object.__setattr__(
            self,
            "_totals",
            MappingProxyType(
                {label: sum(row.values()) for label, row in counts.items()}
            ),
        )
        object.__setattr__(
            self,
            "_vocabulary",
            frozenset(feature for row in counts.values() for feature in row),
        )

    @property
    def inventory(self) -> frozenset[str]:
        """Return the declared label inventory."""
        return self._inventory

    @property
    def fitted(self) -> int:
        """Return how many items the head was fitted on."""
        return self._fitted

    @property
    def vocabulary(self) -> frozenset[str]:
        """Return the features the fitted split carried."""
        return self._vocabulary

    @property
    def majority(self) -> str:
        """Return the most frequent label of the fitted split."""
        return max(sorted(self._inventory), key=lambda label: self._seen[label])

    def scores(self, features: Mapping[str, int]) -> Mapping[str, float]:
        """Return each label's log score for one item's features.

        Parameters
        ----------
        features : Mapping[str, int]
            The item's feature counts.

        Returns
        -------
        Mapping[str, float]
            Log scores keyed by label.

        Raises
        ------
        TypeError
            If a count is not an integer.
        ValueError
            If a feature name is empty or a count is negative.
        """
        read = {
            ensure_name("a feature", feature): ensure_non_negative(
                f"the count of feature {feature!r}", count
            )
            for feature, count in features.items()
        }

        return MappingProxyType(
            {label: self._score(label, read) for label in sorted(self._inventory)}
        )

    def predict(self, features: Mapping[str, int]) -> str:
        """Return the label the head reads off one item's features.

        Parameters
        ----------
        features : Mapping[str, int]
            The item's feature counts.

        Returns
        -------
        str
            The chosen label.

        Raises
        ------
        TypeError
            If a count is not an integer.
        ValueError
            If a feature name is empty or a count is negative.
        """
        scored = self.scores(features)
        best = max(scored.values())
        tied = [label for label in sorted(scored) if scored[label] == best]
        majority = self.majority

        return majority if majority in tied else tied[0]

    def _score(self, label: str, features: Mapping[str, int]) -> float:
        """Return one label's log score for an item's features.

        Parameters
        ----------
        label : str
            The label to score.
        features : Mapping[str, int]
            The item's validated feature counts.
        """
        smoothed = self._totals[label] + len(self._vocabulary)
        row = self._counts[label]
        score = log(self._seen[label] / self._fitted)

        for feature, count in features.items():
            if feature in self._vocabulary:
                score += count * log((row.get(feature, 0) + 1) / smoothed)

        return score

    @override
    def __repr__(self) -> str:
        """Return the inventory, the fitted count and the vocabulary size."""
        return (
            f"LabelHead(inventory={sorted(self._inventory)!r}, "
            f"fitted={self._fitted!r}, vocabulary={len(self._vocabulary)!r})"
        )
