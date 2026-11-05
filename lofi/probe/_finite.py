"""Diagnostics for non-finite probe readings."""

from typing import TYPE_CHECKING, ClassVar, Final, override

import torch

from lofi.nn.sites import DTypeClass
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._dtypes import LOG_ZERO

if TYPE_CHECKING:
    from lofi.nn.sites import Role
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace

__all__ = ["FiniteReadings"]

_SHARE: Final[str] = "nonfinite-share"
"""The report name for the non-finite share of live positions."""


class FiniteReadings:
    """Counts the live positions of each reading that hold no finite number.

    Attributes
    ----------
    tag : ClassVar[str]
        The name this diagnostic registered under.
    requires : ClassVar[frozenset[Role]]
        The roles a model must publish, which is none of them.
    scores : ClassVar[frozenset[str]]
        The one number reported on the unit interval.
    """

    __slots__ = ()

    tag: ClassVar[str] = "finite-readings"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset({_SHARE})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return how many live positions of each reading hold no number.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Measurements named ``live`` and ``nonfinite`` per point read at, and
            ``nonfinite-share`` where at least one position is live. A fourth named
            ``log-zero`` is present where the point's numbers are log probabilities.

        Raises
        ------
        KeyError
            If the trace holds a reading at a point the context does not publish.
        """
        taken = context.readings(trace)
        measured: list[Measurement] = []

        for site, observation in taken:
            values = observation.tensor
            covering = observation.covering()
            ruled_out = (
                (values == LOG_ZERO) & covering
                if context.spec_at(site).dtype_class is DTypeClass.LOG_PROBABILITY
                else None
            )
            absent = ~torch.isfinite(values) & covering

            if ruled_out is not None:
                absent = absent & ~ruled_out

            live = float(covering.sum())
            nonfinite = float(absent.sum())

            measured.append(Measurement(site, "live", live))
            measured.append(Measurement(site, "nonfinite", nonfinite))

            if ruled_out is not None:
                measured.append(Measurement(site, "log-zero", float(ruled_out.sum())))

            if live:
                measured.append(Measurement(site, _SHARE, nonfinite / live))

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return "FiniteReadings()"
