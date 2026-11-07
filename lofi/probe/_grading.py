"""The depth grading of a term encoder's node encodings."""

from typing import TYPE_CHECKING, ClassVar, Final, override

from lofi.nn.sites import Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._structure import (
    NODE_AXES,
    NodeAxes,
    NodeStructure,
    correlation,
    node_readings,
    similarity,
)

if TYPE_CHECKING:
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace
    from lofi.terms import BatchEncoding

__all__ = ["DepthGrading"]

_GRADING: Final[str] = "depth-grading"
"""The report name for how strongly depth relates to a node's likeness to its root."""


class DepthGrading:
    """Measures how a node's likeness to the root above it moves with its depth.

    Each live slot is read against the encoding at the root of the term it
    belongs to, as the cosine of the angle between the two, and that likeness is
    correlated with the slot's depth. An encoder that grades its nodes by depth
    relates the two strongly, in whichever direction its construction happens to
    take. An encoder blind to depth relates them not at all, which is what makes
    the coefficient worth reading: a monotone artifact of the encoding's
    magnitude cannot produce it, since the cosine does not move with scale.

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

    tag: ClassVar[str] = "depth-graded-encoding"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.NODE_ENCODING})
    scores: ClassVar[frozenset[str]] = frozenset({_GRADING})

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
        """Return how strongly depth relates to a node's likeness to its root.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Per node point read at, a measurement named ``nodes``, and where any
            node was read, one named ``depth-correlation`` holding the signed
            coefficient and one named ``depth-grading`` holding its magnitude.

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
            present, roots = structure.rooted()
            likeness = similarity(structure.vectors, roots.unsqueeze(1))
            usable = structure.live & present.unsqueeze(1)
            depths = structure.depths[usable]

            measured.append(Measurement(site, "nodes", float(depths.numel())))

            if depths.numel():
                coefficient = correlation(depths, likeness[usable])

                measured.append(Measurement(site, "depth-correlation", coefficient))
                measured.append(Measurement(site, _GRADING, abs(coefficient)))

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return how many terms the diagnostic holds and the axes it reads."""
        held = 0 if self._terms is None else len(self._terms.rows)

        return f"DepthGrading({held} terms, axes={self._axes!r})"
