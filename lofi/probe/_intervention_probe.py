"""The intervention probe: one declared intervention per published site.

Each held-out pass runs once unpatched, reading every observable point of the
outcome role and every observable intervened point, and then once per declared
intervention with that intervention alone in place. An intervention drawn from
the unpatched pass, such as its mean at the site, is drawn from that pass's own
readings. The effect of an intervention on a pass is the mean absolute change
it caused over the live positions of every outcome reading, and the probe
reports each intervention's mean effect over the passes.
"""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi.nn.sites import Access
from lofi.probe._boundary import InterventionProbeSettings, InterventionProbeSpec
from lofi.probe._probing import (
    GradientMode,
    ProbeBuildContext,
    ProbeValue,
    UnpublishedSiteError,
    observable,
    observed_pass,
    published_sites,
    require_roles,
    site_label,
)
from lofi.registry import ComponentPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.sites import Role, Site, SiteSpec
    from lofi.probe._probing import InterventionSource, TrainedPoint
    from lofi.probe._values import Trace

__all__ = ["InterventionProbe"]


def _effect(clean: Trace, patched: Trace, outcomes: frozenset[Site], /) -> float:
    """Return the mean absolute change over the live positions of every outcome.

    Raises
    ------
    ValueError
        If neither pass reached an outcome point, or its readings hold no live
        position.
    """
    changes: list[torch.Tensor] = []

    for site in sorted(outcomes, key=repr):
        if site not in clean or site not in patched:
            continue

        before = clean[site]
        live = before.covering()
        changes.append(
            (patched[site].tensor - before.tensor).abs()[live].to(torch.float64)
        )

    if not changes:
        msg = "no outcome point was reached by the pass"
        raise ValueError(msg)

    joined = torch.cat(changes)

    if joined.numel() == 0:
        msg = "the outcome readings hold no live position"
        raise ValueError(msg)

    return float(joined.mean())


class InterventionProbe:
    """Scores one intervention per published site by its effect on an outcome role.

    Parameters
    ----------
    interventions : tuple[tuple[Site, InterventionSource], ...]
        Each intervened point and what draws its intervention, in order.
    outcome : Role
        The role of the points whose readings are compared.
    """

    __slots__ = ("_interventions", "_outcome")

    def __init__(
        self, interventions: tuple[tuple[Site, InterventionSource], ...], outcome: Role
    ) -> None:
        self._interventions = interventions
        self._outcome = outcome

    @property
    def outcome(self) -> Role:
        """Return the role of the points whose readings are compared."""
        return self._outcome

    @property
    def sites(self) -> tuple[Site, ...]:
        """Return the intervened points, in declared order."""
        return tuple(site for site, _ in self._interventions)

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a point that cannot be patched or whose outcome is not read.

        Parameters
        ----------
        published : Mapping[Site, SiteSpec]
            Every point the trained point publishes.

        Raises
        ------
        UnpublishedSiteError
            If an intervened point is not published or does not permit what its
            intervention needs, patching and, for an intervention drawn from the
            unpatched pass, observation; or if no observable point plays the
            outcome role.
        """
        for site, source in self._interventions:
            needed = source.access

            if site not in published or needed not in published[site].access:
                permitting = sorted(
                    site_label(candidate)
                    for candidate, spec in published.items()
                    if needed in spec.access
                )
                msg = (
                    f"no component publishes {site_label(site)} permitting "
                    f"{needed.name}; the trained point publishes {permitting} "
                    "permitting it"
                )
                raise UnpublishedSiteError(msg)

        require_roles((self._outcome,), published, Access.OBSERVE)

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return each intervention's mean effect on the outcome over the passes.

        Parameters
        ----------
        point : TrainedPoint
            The trained point.

        Returns
        -------
        tuple[ProbeValue, ...]
            One ``<tag>/effect`` value per intervention, in declared order, named
            by the tag of the intervention drawn.

        Raises
        ------
        ValueError
            If a pass reached no outcome point.
        """
        published = published_sites(point.components)
        readable = observable(published)
        outcomes = frozenset(site for site in readable if site.role is self._outcome)
        clean_reads = outcomes | (frozenset(self.sites) & readable)
        effects: dict[Site, list[float]] = {site: [] for site in self.sites}
        tags: dict[Site, str] = {}

        for held_out in point.passes:
            clean, _ = observed_pass(
                point,
                held_out,
                observe=clean_reads,
                patch={},
                gradients=GradientMode.DISABLED,
            )

            for site, source in self._interventions:
                intervention = source.drawn(clean)
                tags[site] = intervention.tag
                patched, _ = observed_pass(
                    point,
                    held_out,
                    observe=outcomes,
                    patch={site: intervention},
                    gradients=GradientMode.DISABLED,
                )
                effects[site].append(_effect(clean, patched, outcomes))

        return tuple(
            ProbeValue(
                f"{tags[site]}/effect",
                site,
                sum(effects[site]) / len(effects[site]),
                len(effects[site]),
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for site in self.sites
            if effects[site]
        )

    @override
    def __repr__(self) -> str:
        """Return the intervened points and the outcome role."""
        sites = [site_label(site) for site in self.sites]

        return f"InterventionProbe({sites!r}, outcome={self._outcome.name})"


def _build(
    settings: InterventionProbeSettings, context: ProbeBuildContext, /
) -> InterventionProbe:
    """Build an intervention probe from its settings.

    Parameters
    ----------
    settings : InterventionProbeSettings
        The interventions and the outcome role.
    context : ProbeBuildContext
        The declared role and point capabilities.
    """
    del context

    return InterventionProbe(settings.interventions, settings.outcome)


PLUGIN: Final = ComponentPlugin(spec=InterventionProbeSpec, build=_build)
"""The plugin the ``intervention`` probe kind registers."""
