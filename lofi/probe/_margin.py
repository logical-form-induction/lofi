"""Action-margin diagnostics at decision sites."""

from typing import TYPE_CHECKING, ClassVar, override

import torch

from lofi._checks import ensure_name
from lofi.nn.sites import DTypeClass, Role
from lofi.probe._diagnose import Aggregation, DiagnosticResult, Measurement
from lofi.probe._dtypes import LOG_ZERO
from lofi.probe._marginals import SpanAggregate
from lofi.probe._scores import best_two

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Observation, Trace

__all__ = ["ActionMargin"]


def _live(observation: Observation, over: str) -> torch.Tensor | None:
    """Return which of a reading's rows hold data, with the named axis dropped.

    Parameters
    ----------
    observation : Observation
        The reading to read the mask of.
    over : str
        The axis the actions run along.

    Returns
    -------
    torch.Tensor | None
        Truth values over the reading's other axes, or None where the reading has no
        mask.

    Raises
    ------
    ValueError
        If the mask covers the named axis and differs along it.
    """
    masks = tuple(
        observation.select(**{over: position}).mask
        for position in range(observation.extent(over))
    )
    first = masks[0]

    for mask in masks[1:]:
        agrees = (
            torch.equal(first, mask)
            if first is not None and mask is not None
            else first is None and mask is None
        )

        if not agrees:
            msg = (
                "the mask differs from one action to the next, so the actions "
                "a gap is taken between are not the same at every column"
            )
            raise ValueError(msg)

    return first


def _covering(live: Tensor | None, gap: Tensor) -> Tensor:
    """Return which rows hold data, at the shape of the gaps.

    Parameters
    ----------
    live : Tensor | None
        Which rows hold data over the reading's outermost axes, or None where all do.
    gap : Tensor
        One gap per row.

    Returns
    -------
    Tensor
        Truth values of the gaps' shape, true where the row holds data.
    """
    if live is None:
        return torch.ones_like(gap, dtype=torch.bool)

    under = (1,) * (gap.ndim - live.ndim)

    return live.reshape(tuple(live.shape) + under).expand_as(gap)


class ActionMargin:
    """Measures the gap between the two highest action weights, averaged over rows.

    The gap is averaged twice, once under the context's derivation weighting and
    once uniformly, so the margin at the spans the model believes in stands beside
    the margin at every span alike.

    Parameters
    ----------
    over : str
        The name of the axis the actions run along, which the component's own
        specification declares.

    Attributes
    ----------
    tag : ClassVar[str]
        The name this diagnostic registered under.
    requires : ClassVar[frozenset[Role]]
        The roles a model must publish for this diagnostic to apply.
    scores : ClassVar[frozenset[str]]
        The numbers reported on the unit interval, which is none of them.

    Raises
    ------
    TypeError
        If the axis name is not a string.
    ValueError
        If it is empty.
    """

    __slots__ = ("_over",)

    tag: ClassVar[str] = "action-margin"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.COMBINATOR_LOGITS})
    scores: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, over: str = "action") -> None:
        self._over = ensure_name("the axis a margin is taken over", over)

    @property
    def over(self) -> str:
        """Return the name of the axis the actions run along."""
        return self._over

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return the mean gap between the two best actions at each point.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            What the components under study published.

        Returns
        -------
        DiagnosticResult
            Per combinator point read at, one aggregation named ``margin`` holding the
            gap's mean under the derivation weighting and under equal weights, a
            measurement named ``rows``, and a measurement named ``certain`` where the
            point's weights are log probabilities.

        Raises
        ------
        KeyError
            If the trace holds a reading at a point the context does not publish, or if
            a reading declares no axis under this diagnostic's name.
        ValueError
            If the trace lacks declared combinator points, a reading has fewer than two
            actions, no row has two valid actions, or the context holds no derivation
            weighting lined up with the rows.
        """
        taken = dict(context.readings(trace))
        published = context.sites_for(Role.COMBINATOR_LOGITS)
        read = tuple(site for site in published if site in taken)

        if published and not read:
            msg = (
                f"{len(published)} points weigh actions and the scope read none "
                "of them, so there is no decision to measure"
            )
            raise ValueError(msg)

        measured: list[Measurement] = []
        aggregated: list[Aggregation] = []

        for site in read:
            observation = taken[site]
            best = best_two(observation, over=self._over)
            gap = best.select(-1, 0) - best.select(-1, 1)
            rows = tuple(axis for axis in observation.axes if axis != self._over)
            covering = _covering(_live(observation, self._over), gap)

            if not bool(covering.any()):
                msg = f"every row read at {site!r} is masked off"
                raise ValueError(msg)

            certain: int | None = None

            if context.spec_at(site).dtype_class is DTypeClass.LOG_PROBABILITY:
                possible = best.select(-1, 1) != LOG_ZERO
                certain = int((~possible & covering).sum())
                covering = covering & possible

                if not bool(covering.any()):
                    msg = (
                        f"every row read at {site!r} weighs at most one possible "
                        "action, so no gap between two of them is a number"
                    )
                    raise ValueError(msg)

            weights = context.weights_over(rows, tuple(gap.shape))
            margin = SpanAggregate.over(gap, weights, covering)

            aggregated.append(Aggregation(site, "margin", margin))
            measured.append(Measurement(site, "rows", float(margin.count)))

            if certain is not None:
                measured.append(Measurement(site, "certain", float(certain)))

        return DiagnosticResult(self.tag, measured, aggregated)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return f"ActionMargin({self._over!r})"
