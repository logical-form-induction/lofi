"""Regression of tree edit distance onto distance between type encodings."""

from typing import TYPE_CHECKING, override

import torch

from lofi._checks import ensure_finite, ensure_name
from lofi.nn.loss._likelihood import live_rows
from lofi.nn.loss._ports import LossTerm
from lofi.terms import TreeEditDistance, class_sensitive_relabel

if TYPE_CHECKING:
    from lofi.nn.loss._ports import PublishedOutputs, TypePairs
    from lofi.nn.ports import TermEncoder
    from lofi.terms import RelabelCost

__all__ = ["TypeDistanceObjective"]


def inverse_distance_weights(distances: torch.Tensor, power: float, /) -> torch.Tensor:
    """Return normalized inverse distance weights for one nonempty pair batch.

    Parameters
    ----------
    distances : torch.Tensor
        Nonnegative edit distances, one per pair.
    power : float
        Finite nonnegative inverse distance exponent.

    Raises
    ------
    ValueError
        If the tensor is empty or contains a negative distance.
    """
    held_power = _nonnegative("inverse-distance power", power)

    if distances.numel() == 0:
        msg = "inverse-distance weighting requires at least one pair"
        raise ValueError(msg)

    if bool((distances < 0).any()):
        msg = "inverse-distance weighting received a negative distance"
        raise ValueError(msg)

    raw = (1.0 + distances).pow(-held_power)

    return raw / raw.sum()


class TypeDistanceObjective:
    """Scores how far the distance between two encodings sits from their edit distance.

    The objective regresses squared embedding distance onto squared tree edit
    distance and averages over complete pairs. Edit distances are cached only within
    the current step.

    Parameters
    ----------
    encoder : TermEncoder
        What encodes each type to a vector.
    inverse_distance_power : float
        Finite nonnegative exponent for normalized inverse distance weights.
    embedding_penalty : float
        Finite nonnegative coefficient on each pair's mean squared embedding norm.
    relabel : RelabelCost
        The cost of replacing one node label with another in the edit distance.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = (
        "_embedding_penalty",
        "_encoder",
        "_inverse_distance_power",
        "_name",
        "_relabel",
    )

    def __init__(
        self,
        encoder: TermEncoder,
        *,
        inverse_distance_power: float,
        embedding_penalty: float,
        relabel: RelabelCost = class_sensitive_relabel,
        name: str = "type-distance",
    ) -> None:
        self._encoder = encoder
        self._relabel = relabel
        self._inverse_distance_power = _nonnegative(
            "inverse_distance_power", inverse_distance_power
        )
        self._embedding_penalty = _nonnegative("embedding_penalty", embedding_penalty)
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, which is the batch."""
        return frozenset({"batch"})

    @property
    def relabel(self) -> RelabelCost:
        """Return the relabel cost of the edit distance the encodings regress onto."""
        return self._relabel

    @property
    def inverse_distance_power(self) -> float:
        """Return the exponent applied to inverse edit-distance weights."""
        return self._inverse_distance_power

    @property
    def embedding_penalty(self) -> float:
        """Return the coefficient on each pair's mean squared embedding norm."""
        return self._embedding_penalty

    def evaluate(self, outputs: PublishedOutputs[TypePairs], /) -> LossTerm:
        """Return the mean squared error between the two squared distances.

        Parameters
        ----------
        outputs : PublishedOutputs[TypePairs]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the pairs holding two types of the squared difference
            between the squared distance of their encodings and their squared edit
            distance, counted in those pairs, reporting the mean absolute difference
            between the unsquared distances as ``absolute_error``.

        Raises
        ------
        ValueError
            If the batch does not hold one pair of terms per row of each linearization.
        """
        batch = outputs.batch
        pairs = batch.pairs
        rows = (len(batch.left.rows), len(batch.right.rows))

        if rows != (len(pairs), len(pairs)):
            msg = (
                f"the batch holds {len(pairs)} pairs of types, {rows[0]} first "
                f"linearizations and {rows[1]} second ones"
            )
            raise ValueError(msg)

        left = self._encoder.encode(batch.left)
        right = self._encoder.encode(batch.right)
        squared = (left.tensor - right.tensor).pow(2).sum(dim=-1)

        if not pairs:
            return LossTerm.inapplicable(squared)

        distance = TreeEditDistance(self._relabel)
        edits = squared.new_tensor(
            [float(distance(first, second)) for first, second in pairs]
        )
        live = live_rows((left.mask, right.mask))
        errors = (squared - edits.pow(2)).pow(2)
        penalties = 0.5 * (
            left.tensor.pow(2).sum(dim=-1) + right.tensor.pow(2).sum(dim=-1)
        )
        contributions = errors + self._embedding_penalty * penalties
        weights = inverse_distance_weights(edits, self._inverse_distance_power)
        gaps = (squared.detach().clamp(min=0.0).sqrt() - edits).abs()

        if live is None:
            count = int(squared.shape[0])
        else:
            count = int(live.sum())
            contributions = torch.where(
                live, contributions, contributions.new_zeros(())
            )
            penalties = torch.where(live, penalties, penalties.new_zeros(()))
            weights = torch.where(live, weights, weights.new_zeros(()))
            gaps = torch.where(live, gaps, gaps.new_zeros(()))

        if count == 0:
            return LossTerm.inapplicable(squared)

        normalized = weights / weights.sum()

        return LossTerm(
            (normalized * contributions).sum(),
            count,
            diagnostics={
                "absolute_error": gaps.sum() / count,
                "embedding_penalty": (normalized * penalties).sum(),
            },
        )

    @override
    def __repr__(self) -> str:
        """Return the name and the encoder's type."""
        return (
            f"TypeDistanceObjective(name={self._name!r}, "
            f"encoder={type(self._encoder).__name__})"
        )


def _nonnegative(name: str, value: float, /) -> float:
    """Return a finite nonnegative coefficient."""
    checked = ensure_finite(name, value)

    if checked < 0.0:
        msg = f"{name} is negative: {value!r}"
        raise ValueError(msg)

    return checked
