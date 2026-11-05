"""The separation a term encoder holds between atomic and complex nodes."""

from typing import TYPE_CHECKING, ClassVar, Final, override

from lofi.nn.sites import Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._structure import (
    NODE_AXES,
    NodeAxes,
    NodeStructure,
    node_readings,
    similarity,
)

if TYPE_CHECKING:
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace
    from lofi.terms import BatchEncoding

__all__ = ["AtomicDifferentiation"]

_DIFFERENTIATION: Final[str] = "differentiation"
"""The report name for how far the two kinds of node are encoded apart."""


class AtomicDifferentiation:
    """Measures how far apart a term encoder holds atomic and complex nodes.

    A slot is complex where a live slot hangs off it and atomic where none does,
    which reads the distinction off the linearization rather than off the
    alphabet. The two groups are summarized by their mean encodings, and the
    angle between those two means is reported as a number that is zero where the
    groups point the same way and one where they point opposite ways. A reading
    holding only one of the two kinds reports its counts and no separation, since
    there is nothing to be apart from.

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

    tag: ClassVar[str] = "atomic-complex-differentiation"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.NODE_ENCODING})
    scores: ClassVar[frozenset[str]] = frozenset({_DIFFERENTIATION})

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
        """Return how far apart the two kinds of node are encoded.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Per node point read at, measurements named ``atomic`` and ``complex``
            holding how many slots of each kind were read, and where both kinds
            were read, one named ``differentiation``.

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
            compound = structure.live & structure.internal
            atoms = structure.vectors[structure.live & ~structure.internal]
            compounds = structure.vectors[compound]

            measured.append(Measurement(site, "atomic", float(atoms.shape[0])))
            measured.append(Measurement(site, "complex", float(compounds.shape[0])))

            if atoms.numel() and compounds.numel():
                together = similarity(atoms.mean(dim=0), compounds.mean(dim=0))

                measured.append(
                    Measurement(site, _DIFFERENTIATION, (1.0 - float(together)) / 2.0)
                )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return how many terms the diagnostic holds and the axes it reads."""
        held = 0 if self._terms is None else len(self._terms.rows)

        return f"AtomicDifferentiation({held} terms, axes={self._axes!r})"
