"""How a term encoder's node encodings spread as a term takes on more nodes."""

from typing import TYPE_CHECKING, ClassVar, Final, override

from lofi.nn.sites import Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._structure import (
    NODE_AXES,
    NodeAxes,
    NodeStructure,
    correlation,
    node_readings,
    pairwise_similarity,
    strictly_above,
)

if TYPE_CHECKING:
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace
    from lofi.terms import BatchEncoding

__all__ = ["RecursionSpread"]

_SPREAD: Final[str] = "node-spread"
"""The report name for how far apart a term holds its own nodes."""

_GRADING: Final[str] = "count-grading"
"""The report name for how strongly the spread moves with the node count."""


class RecursionSpread:
    """Measures how far a term holds its own nodes apart as it takes on more.

    A term's spread is the mean over its pairs of live slots of the angle between
    their encodings, at zero where every node of the term is encoded the same way
    and at one where every pair points the opposite way. Reported across the
    batch it says whether the encoder keeps the nodes of one term distinct at
    all. Correlated with the term's live node count it says what recursion costs:
    an encoder whose nodes run together as a term grows relates the two
    negatively, and one that keeps them apart relates them not at all. A term
    holding fewer than two live slots has no pair and is left out.

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
        The two numbers reported on the unit interval.
    """

    __slots__ = ("_axes", "_terms")

    tag: ClassVar[str] = "recursion-spread"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.NODE_ENCODING})
    scores: ClassVar[frozenset[str]] = frozenset({_GRADING, _SPREAD})

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
        """Return how far each term holds its nodes apart, against its node count.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Per node point read at, a measurement named ``terms`` holding how
            many terms held a pair of live slots, and where any did, one named
            ``node-spread``, one named ``count-correlation`` holding the signed
            coefficient against the live node count, and one named
            ``count-grading`` holding its magnitude.

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
            live = structure.live
            paired = (
                live.unsqueeze(2) & live.unsqueeze(1) & strictly_above(live.shape[1])
            )
            apart = (1.0 - pairwise_similarity(structure.vectors)) / 2.0
            pairs = paired.sum(dim=(1, 2))
            held = pairs > 0
            spreads = (apart * paired).sum(dim=(1, 2))[held] / pairs[held]

            measured.append(Measurement(site, "terms", float(spreads.numel())))

            if spreads.numel():
                coefficient = correlation(structure.counts()[held], spreads)

                measured.append(Measurement(site, _SPREAD, float(spreads.mean())))
                measured.append(Measurement(site, "count-correlation", coefficient))
                measured.append(Measurement(site, _GRADING, abs(coefficient)))

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return how many terms the diagnostic holds and the axes it reads."""
        held = 0 if self._terms is None else len(self._terms.rows)

        return f"RecursionSpread({held} terms, axes={self._axes!r})"
