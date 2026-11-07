"""Controller-choice distribution diagnostics."""

from typing import TYPE_CHECKING, ClassVar, Final, override

from lofi._checks import ensure_name
from lofi.nn.sites import Role
from lofi.probe._diagnose import Aggregation, DiagnosticResult, Measurement
from lofi.probe._marginals import SpanAggregate
from lofi.probe._scores import (
    maximum_probability,
    normalized_entropy,
    top_two_margin,
)

if TYPE_CHECKING:
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace

__all__ = ["ControllerChoice"]

_CONFIDENCE: Final[str] = "confidence"
"""The largest action probability."""

_ENTROPY: Final[str] = "normalized-entropy"
"""Action entropy as a fraction of its maximum."""

_MARGIN: Final[str] = "top-two-probability-margin"
"""The gap between the two largest action probabilities."""


class ControllerChoice:
    """Measures three distributional properties of each controller choice.

    Each property is computed per live row and then averaged under the context's
    derivation weighting and under equal weights. The reported properties are the
    largest action probability, the entropy normalized by the number of live actions,
    and the gap between the two largest action probabilities.

    Parameters
    ----------
    over : str
        The name of the axis the actions run along.
    """

    __slots__ = ("_over",)

    tag: ClassVar[str] = "controller-choice"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.COMBINATOR_LOGITS})
    scores: ClassVar[frozenset[str]] = frozenset({_CONFIDENCE, _ENTROPY, _MARGIN})

    def __init__(self, over: str = "action") -> None:
        self._over = ensure_name("the axis controller choices run over", over)

    @property
    def over(self) -> str:
        """Return the name of the axis the actions run along."""
        return self._over

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return controller-choice scores at every decision point read.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Three score aggregations and a live-row count per decision point read.

        Raises
        ------
        KeyError
            If the trace holds a reading at an unpublished point, or a reading lacks
            the action axis.
        ValueError
            If declared decision points were not read, fewer than two action columns
            exist, no row is live, the numeric declaration cannot weigh outcomes, or
            the context cannot weight the score rows.
        """
        taken = dict(context.readings(trace))
        published = context.sites_for(Role.COMBINATOR_LOGITS)
        read = tuple(site for site in published if site in taken)

        if published and not read:
            msg = (
                f"the published controller-choice points {published!r} were not "
                "read, so there is no choice to measure"
            )
            raise ValueError(msg)

        measured: list[Measurement] = []
        aggregated: list[Aggregation] = []

        for site in read:
            observation = taken[site]
            declared = context.spec_at(site).dtype_class
            spread = normalized_entropy(observation, declared, over=self._over)
            scored = (
                (
                    _CONFIDENCE,
                    maximum_probability(observation, declared, over=self._over),
                ),
                (_ENTROPY, spread),
                (_MARGIN, top_two_margin(observation, declared, over=self._over)),
            )
            first = scored[0][1]
            weights = context.weights_over(first.axes, tuple(first.tensor.shape))
            rows = 0

            for name, score in scored:
                summary = SpanAggregate.over(score.tensor, weights, score.covering())
                aggregated.append(Aggregation(site, name, summary))
                rows = summary.count

            measured.append(Measurement(site, "rows", float(rows)))

        return DiagnosticResult(self.tag, measured, aggregated)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return f"ControllerChoice({self._over!r})"
