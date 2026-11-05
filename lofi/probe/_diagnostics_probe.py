"""The diagnostics probe: installed diagnostics over a set of published roles.

Each held-out pass is run once with every observable point of the selected
roles read. The diagnostics run over a pass are those of the ``lofi.diagnostic``
catalog whose required roles the pass reached, under a context holding the
pass's span addressing and, where a point plays ``CONSTITUENT_MARGINAL``, the
constituent marginals it read as the span weighting. A measurement is reported
as its mean over the passes that took it, and a quantity a diagnostic aggregates
over spans is reported twice, as the mean over those passes of its
marginal-weighted and of its uniform mean. A diagnostic that refuses a pass's
readings is counted, not dropped: its refusals are reported as a measurement of
their own.
"""

from typing import TYPE_CHECKING, Final, override

from lofi.nn.sites import Access, Role
from lofi.probe._boundary import DiagnosticsProbeSettings, DiagnosticsProbeSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._marginals import SpanWeighting
from lofi.probe._probing import (
    GradientMode,
    ProbeBuildContext,
    ProbeValue,
    UnpublishedSiteError,
    observable,
    observed_pass,
    published_sites,
    require_roles,
)
from lofi.registry import ComponentPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.sites import Site, SiteSpec
    from lofi.probe._catalog import DiagnosticCatalog
    from lofi.probe._probing import PassLayout, TrainedPoint
    from lofi.probe._values import Trace

__all__ = ["DiagnosticsProbe"]

_REFUSED: Final = "refused"
"""The measurement under which a diagnostic's refused passes are counted."""


def _weighting(trace: Trace, /) -> SpanWeighting | None:
    """Return the constituent marginals a pass read, as a span weighting.

    A marginal reading holds zero at every position outside its mask, so its
    values weigh the spans as they stand. Where the pass read no constituent
    marginal, or read more than one point of it, there is no single weighting
    and None is returned.
    """
    marginals = [
        observation
        for site, observation in trace.items()
        if site.role is Role.CONSTITUENT_MARGINAL
    ]

    if len(marginals) != 1:
        return None

    (reading,) = marginals

    return SpanWeighting(reading.tensor, axes=reading.axes)


def _context(
    published: Mapping[Site, SiteSpec], trace: Trace, layout: PassLayout, /
) -> DiagnosticContext:
    """Return what the diagnostics over one pass read the published points from."""
    return DiagnosticContext(
        {site: published[site] for site in trace.sites()},
        span_index=layout.spans,
        weighting=_weighting(trace),
    )


class DiagnosticsProbe:
    """Runs the cataloged diagnostics a set of published roles supports.

    Parameters
    ----------
    roles : frozenset[Role] | None
        The roles whose points are read, or None for every role the trained
        point publishes.
    catalog : DiagnosticCatalog
        The diagnostics to select from.
    """

    __slots__ = ("_catalog", "_roles")

    def __init__(
        self, roles: frozenset[Role] | None, catalog: DiagnosticCatalog
    ) -> None:
        self._roles = roles
        self._catalog = catalog

    @property
    def roles(self) -> frozenset[Role] | None:
        """Return the roles whose points are read, or None for every role."""
        return self._roles

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a point that publishes no observable point of a selected role.

        Parameters
        ----------
        published : Mapping[Site, SiteSpec]
            Every point the trained point publishes.

        Raises
        ------
        UnpublishedSiteError
            If a declared role is played by no observable published point, or,
            where every role is selected, if no published point is observable.
        """
        if self._roles is not None:
            require_roles(self._roles, published, Access.OBSERVE)

        elif not observable(published):
            msg = "the trained point publishes no point a diagnostic can observe"
            raise UnpublishedSiteError(msg)

    def _read(self, published: Mapping[Site, SiteSpec], /) -> frozenset[Site]:
        """Return the observable points of the selected roles."""
        return frozenset(
            site
            for site in observable(published)
            if self._roles is None or site.role in self._roles
        )

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return each diagnostic measurement's mean over the held-out passes.

        Parameters
        ----------
        point : TrainedPoint
            The trained point.

        Returns
        -------
        tuple[ProbeValue, ...]
            One value per diagnostic, measurement and point, named
            ``<tag>/<measurement>``, and two per aggregated quantity and point,
            named ``<tag>/<quantity>/weighted`` and ``<tag>/<quantity>/uniform``,
            in ascending order, then one
            ``<tag>/refused`` value per diagnostic that refused a pass, counting
            the passes it refused over the passes it was selected for.

        Raises
        ------
        ValueError
            If a declared role is reached by no held-out pass.
        """
        published = published_sites(point.components)
        read = self._read(published)
        taken: dict[tuple[str, Site], list[float]] = {}
        selected: dict[str, int] = {}
        refused: dict[str, int] = {}
        reached: set[Role] = set()

        for held_out in point.passes:
            trace, layout = observed_pass(
                point,
                held_out,
                observe=read,
                patch={},
                gradients=GradientMode.DISABLED,
            )
            context = _context(published, trace, layout)
            reached.update(context.roles)

            for diagnostic in self._catalog.matching(context):
                selected[diagnostic.tag] = selected.get(diagnostic.tag, 0) + 1

                try:
                    result = diagnostic.run(trace, context=context)

                except ValueError:
                    refused[diagnostic.tag] = refused.get(diagnostic.tag, 0) + 1
                    continue

                for measurement in result.measurements:
                    taken.setdefault(
                        (f"{result.tag}/{measurement.name}", measurement.site), []
                    ).append(measurement.value)

                for aggregation in result.aggregations:
                    name = f"{result.tag}/{aggregation.name}"
                    summary = aggregation.aggregate
                    taken.setdefault((f"{name}/weighted", aggregation.site), []).append(
                        summary.weighted
                    )
                    taken.setdefault((f"{name}/uniform", aggregation.site), []).append(
                        summary.uniform
                    )

        unreached = sorted(role.name for role in (self._roles or set()) - reached)

        if unreached:
            msg = f"no held-out pass of the trained point reached {unreached}"
            raise ValueError(msg)

        measured = tuple(
            ProbeValue(
                measure,
                site,
                sum(values) / len(values),
                len(values),
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for (measure, site), values in sorted(
                taken.items(), key=lambda entry: (entry[0][0], repr(entry[0][1]))
            )
        )
        refusals = tuple(
            ProbeValue(
                f"{tag}/{_REFUSED}",
                None,
                float(refused[tag]),
                selected[tag],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for tag in sorted(refused)
        )

        return measured + refusals

    @override
    def __repr__(self) -> str:
        """Return the roles the probe reads."""
        roles = None if self._roles is None else sorted(r.name for r in self._roles)

        return f"DiagnosticsProbe(roles={roles!r})"


def _build(
    settings: DiagnosticsProbeSettings, context: ProbeBuildContext, /
) -> DiagnosticsProbe:
    """Build a diagnostics probe over the installed catalog.

    Parameters
    ----------
    settings : DiagnosticsProbeSettings
        The roles the probe reads.
    context : ProbeBuildContext
        The declared role and point capabilities.
    """
    del context

    return DiagnosticsProbe(settings.roles, installed_diagnostics())


PLUGIN: Final = ComponentPlugin(spec=DiagnosticsProbeSpec, build=_build)
"""The plugin the ``diagnostics`` probe kind registers."""
