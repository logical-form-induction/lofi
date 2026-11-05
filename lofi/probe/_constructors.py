"""How far a term encoder's node encodings follow the constructor at the slot."""

from typing import TYPE_CHECKING, ClassVar, Final, override

import torch

from lofi.nn.sites import Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._structure import (
    NODE_AXES,
    NodeAxes,
    NodeStructure,
    node_readings,
    pairwise_similarity,
    strictly_above,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace
    from lofi.terms import BatchEncoding

__all__ = ["ConstructorEmbedding"]

_AGREEMENT: Final[str] = "constructor-agreement"
"""The report name for how often sharing a constructor means encoding alike."""


def _precedence(sharing: Tensor, differing: Tensor) -> float:
    """Return how often the more alike of two drawn pairs is the sharing one.

    Parameters
    ----------
    sharing : Tensor
        The cosine between the encodings of each pair of slots carrying one
        constructor.
    differing : Tensor
        The cosine between the encodings of each pair of slots carrying two.

    Returns
    -------
    float
        The share of the pairings drawn one from each group whose sharing member
        is the more alike, a tie counting as half. A half says the constructor
        tells nothing about how alike two slots are encoded, and a one that every
        sharing pair is more alike than every differing pair.
    """
    ranked = differing.sort().values
    below = torch.searchsorted(ranked, sharing, right=False).to(torch.float64)
    through = torch.searchsorted(ranked, sharing, right=True).to(torch.float64)
    ahead = (below + through) / 2.0

    return float(ahead.sum() / (sharing.numel() * differing.numel()))


class ConstructorEmbedding:
    """Measures whether slots carrying one constructor are encoded alike.

    Every pair of live slots read at a point is scored by the cosine between the
    two encodings and split by whether the two slots carry the same constructor
    code. The two groups are then compared as distributions rather than as means,
    by how often a pair drawn from the sharing group is the more alike of the
    two, so a few pairs at an extreme cannot carry the number. A reading holding
    no pair of either kind reports its pair counts and nothing else.

    Parameters
    ----------
    terms : BatchEncoding | None
        The linearized terms the scope encoded, in batch order, or None for a
        diagnostic built to report which roles it needs rather than to measure.
    axes : NodeAxes
        The names a node encoding runs its terms, nodes and features over.

    Attributes
    ----------
    tag : ClassVar[str]
        The name this diagnostic registered under.
    requires : ClassVar[frozenset[Role]]
        The roles a model must publish for this diagnostic to apply.
    scores : ClassVar[frozenset[str]]
        The one number reported on the unit interval.
    """

    __slots__ = ("_axes", "_terms")

    tag: ClassVar[str] = "constructor-embedding"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.NODE_ENCODING})
    scores: ClassVar[frozenset[str]] = frozenset({_AGREEMENT})

    def __init__(
        self, terms: BatchEncoding | None = None, *, axes: NodeAxes = NODE_AXES
    ) -> None:
        self._terms = terms
        self._axes = axes

    @property
    def terms(self) -> BatchEncoding | None:
        """Return the linearized terms the encodings are measured against."""
        return self._terms

    @property
    def axes(self) -> NodeAxes:
        """Return the names a node encoding runs its axes under."""
        return self._axes

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return how far sharing a constructor goes with encoding alike.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Per node point read at, measurements named ``sharing-pairs`` and
            ``differing-pairs``, and where both groups hold a pair, one named
            ``constructor-agreement``.

        Raises
        ------
        KeyError
            If the trace holds a reading at a point the context does not publish,
            or a reading declares no axis under one of this diagnostic's names.
        ValueError
            If node encodings are published and the scope read none, if the
            diagnostic holds no terms, or if the terms and a reading disagree
            about how many terms or slots they hold.
        """
        measured: list[Measurement] = []

        for site, reading in node_readings(context, trace):
            structure = NodeStructure(self._terms, reading, axes=self._axes)
            vectors = structure.vectors[structure.live]
            symbols = structure.symbols[structure.live]
            once = strictly_above(vectors.shape[0])
            alike = symbols.reshape(-1, 1) == symbols.reshape(1, -1)
            cosines = pairwise_similarity(vectors)
            sharing = cosines[once & alike]
            differing = cosines[once & ~alike]

            measured.append(Measurement(site, "sharing-pairs", float(sharing.numel())))
            measured.append(
                Measurement(site, "differing-pairs", float(differing.numel()))
            )

            if sharing.numel() and differing.numel():
                measured.append(
                    Measurement(site, _AGREEMENT, _precedence(sharing, differing))
                )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return how many terms the diagnostic holds and the axes it reads."""
        held = 0 if self._terms is None else len(self._terms.rows)

        return f"ConstructorEmbedding({held} terms, axes={self._axes!r})"
