"""Diagnostics over published model sites."""

from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_finite, ensure_name
from lofi.deduce import SpanIndex
from lofi.nn.sites import Role, Site, SiteSpec
from lofi.probe._marginals import SpanAggregate, SpanWeighting
from lofi.probe._observation_groups import ObservationGrouping

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.ports import Observable
    from lofi.probe._controller_relations import ControllerRelation
    from lofi.probe._values import Observation, Trace

__all__ = ["Aggregation", "DiagnosticContext", "DiagnosticResult", "Measurement"]


class _ControllerRelationValue:
    """Marker base for a validated controller-relation declaration."""

    __slots__ = ()


class DiagnosticContext:
    """Stores published sites and caller declarations for a diagnostic.

    Parameters
    ----------
    published : Mapping[Site, SiteSpec]
        Each point published, under the specification of what is found there.
    groupings : Mapping[Site, ObservationGrouping] | None
        Caller-declared row groups at published points, or None.
    relations : Mapping[Site, ControllerRelation] | None
        Caller-declared row relations at controller points, or None.
    span_index : SpanIndex | None
        The span addressing shared by chart readings, or None where the readings are
        not chart-shaped.
    weighting : SpanWeighting | None
        The constituent marginals the readings' spans are weighted by when a
        diagnostic aggregates over them, or None where none is known.

    Raises
    ------
    TypeError
        If a published, grouping, or relation key is not a site; a published value is
        not a site specification; a grouping or relation value is of the wrong kind;
        the span index is not one; or the weighting is not one.
    ValueError
        If a grouping names an unpublished point, or a relation names an unpublished
        or non-controller point.
    """

    __slots__ = (
        "_groupings",
        "_published",
        "_relations",
        "_roles",
        "_span_index",
        "_weighting",
    )

    def __init__(
        self,
        published: Mapping[Site, SiteSpec],
        *,
        groupings: Mapping[Site, ObservationGrouping] | None = None,
        relations: Mapping[Site, ControllerRelation] | None = None,
        span_index: SpanIndex | None = None,
        weighting: SpanWeighting | None = None,
    ) -> None:
        declared = dict(published)
        grouped = {} if groupings is None else dict(groupings)
        related = {} if relations is None else dict(relations)

        if weighting is not None and type(weighting) is not SpanWeighting:
            msg = f"a context's weighting is a SpanWeighting, not {weighting!r}"
            raise TypeError(msg)

        if span_index is not None and type(span_index) is not SpanIndex:
            msg = f"a context's span index is a SpanIndex, not {span_index!r}"
            raise TypeError(msg)

        for site, spec in declared.items():
            if not isinstance(site, Site):
                msg = f"a published point is not a Site: {site!r}"
                raise TypeError(msg)

            if not isinstance(spec, SiteSpec):
                msg = f"what is published at {site!r} is not a SiteSpec: {spec!r}"
                raise TypeError(msg)

        for site, grouping in grouped.items():
            if not isinstance(site, Site):
                msg = f"an observation grouping's point is not a Site: {site!r}"
                raise TypeError(msg)

            if not isinstance(grouping, ObservationGrouping):
                msg = (
                    f"the observation grouping at {site!r} is an "
                    f"ObservationGrouping, not {grouping!r}"
                )
                raise TypeError(msg)

            if site not in declared:
                msg = f"an observation grouping names unpublished point {site!r}"
                raise ValueError(msg)

        for site, relation in related.items():
            if not isinstance(site, Site):
                msg = f"a controller relation's point is not a Site: {site!r}"
                raise TypeError(msg)

            if not isinstance(relation, _ControllerRelationValue):
                msg = (
                    f"the controller relation at {site!r} is a ControllerRelation, "
                    f"not {relation!r}"
                )
                raise TypeError(msg)

            if site not in declared:
                msg = f"a controller relation names unpublished point {site!r}"
                raise ValueError(msg)

            if site.role is not Role.COMBINATOR_LOGITS:
                msg = (
                    f"a controller relation belongs at COMBINATOR_LOGITS, not "
                    f"{site.role.name} at {site!r}"
                )
                raise ValueError(msg)

        self._published = MappingProxyType(declared)
        self._groupings = MappingProxyType(grouped)
        self._relations = MappingProxyType(related)
        self._roles = frozenset(site.role for site in declared)
        self._span_index = span_index
        self._weighting = weighting

    @classmethod
    def observing(
        cls,
        *components: Observable,
        groupings: Mapping[Site, ObservationGrouping] | None = None,
        relations: Mapping[Site, ControllerRelation] | None = None,
        span_index: SpanIndex | None = None,
        weighting: SpanWeighting | None = None,
    ) -> DiagnosticContext:
        """Return the context the given components' published maps make up.

        Parameters
        ----------
        *components : Observable
            The components under study, each publishing its own map.
        groupings : Mapping[Site, ObservationGrouping] | None
            Caller-declared row groups at published points, or None.
        relations : Mapping[Site, ControllerRelation] | None
            Caller-declared row relations at controller points, or None.
        span_index : SpanIndex | None
            The span addressing shared by chart readings, or None.
        weighting : SpanWeighting | None
            The constituent marginals the readings' spans are weighted by, or None
            where none is known.

        Returns
        -------
        DiagnosticContext
            Their maps together.

        Raises
        ------
        ValueError
            If two components publish the same point under different specifications.
        """
        published: dict[Site, SiteSpec] = {}

        for component in components:
            for site, spec in component.site_map().items():
                declared = published.setdefault(site, spec)

                if declared != spec:
                    msg = (
                        f"{site!r} is published as {declared!r} and as {spec!r}, "
                        "so what is found there is not settled"
                    )
                    raise ValueError(msg)

        return cls(
            published,
            groupings=groupings,
            relations=relations,
            span_index=span_index,
            weighting=weighting,
        )

    @property
    def published(self) -> Mapping[Site, SiteSpec]:
        """Return each point published, as a read-only view."""
        return self._published

    @property
    def weighting(self) -> SpanWeighting | None:
        """Return the constituent marginals over the readings' spans, or None."""
        return self._weighting

    @property
    def span_index(self) -> SpanIndex | None:
        """Return the chart readings' shared span addressing, or None."""
        return self._span_index

    @property
    def groupings(self) -> Mapping[Site, ObservationGrouping]:
        """Return the caller-declared observation groups as a read-only view."""
        return self._groupings

    def grouping_at(self, site: Site) -> ObservationGrouping:
        """Return the observation grouping declared at one point.

        Parameters
        ----------
        site : Site
            The published point to look up.

        Returns
        -------
        ObservationGrouping
            The caller-declared row memberships.

        Raises
        ------
        KeyError
            If no grouping was supplied for the point.
        """
        try:
            return self._groupings[site]

        except KeyError as error:
            msg = f"no observation grouping was supplied for {site!r}"
            raise KeyError(msg) from error

    @property
    def relations(self) -> Mapping[Site, ControllerRelation]:
        """Return the caller-declared controller relations as a read-only view."""
        return self._relations

    def relation_at(self, site: Site) -> ControllerRelation:
        """Return the controller relation declared at one point.

        Parameters
        ----------
        site : Site
            The controller point to look up.

        Returns
        -------
        ControllerRelation
            The caller-declared row memberships.

        Raises
        ------
        KeyError
            If no relation was supplied for the point.
        """
        try:
            return self._relations[site]

        except KeyError as error:
            msg = f"no controller relation was supplied for {site!r}"
            raise KeyError(msg) from error

    def weights_over(self, axes: Sequence[str], shape: Sequence[int]) -> Tensor:
        """Return the constituent marginals broadcast over a reading's axes.

        Parameters
        ----------
        axes : Sequence[str]
            The reading's axis names, outermost first.
        shape : Sequence[int]
            The reading's extent along each of those axes.

        Returns
        -------
        Tensor
            Of the reading's shape, holding at each position the marginal of the span
            it sits under.

        Raises
        ------
        ValueError
            If the context holds no weighting, so an aggregate over spans cannot be
            weighted, or if the weighting does not line up with the reading's leading
            axes.
        """
        if self._weighting is None:
            msg = (
                "the context holds no derivation weighting, so an aggregate over "
                "spans cannot be reported under the marginals"
            )
            raise ValueError(msg)

        return self._weighting.aligned(axes, shape)

    @property
    def roles(self) -> frozenset[Role]:
        """Return the roles the published points cover, which resolution matches on."""
        return self._roles

    def sites_for(self, role: Role) -> tuple[Site, ...]:
        """Return the published points playing one role, in publication order.

        Parameters
        ----------
        role : Role
            The role to gather.

        Returns
        -------
        tuple[Site, ...]
            The points published under it.

        Raises
        ------
        TypeError
            If the value is not a role.
        """
        if not isinstance(role, Role):
            msg = f"a role is a Role, not {role!r}"
            raise TypeError(msg)

        return tuple(site for site in self._published if site.role is role)

    def spec_at(self, site: Site) -> SiteSpec:
        """Return what is published at one point.

        Parameters
        ----------
        site : Site
            The point to look up.

        Returns
        -------
        SiteSpec
            What is found there.

        Raises
        ------
        KeyError
            If the point is not published in this context.
        """
        try:
            return self._published[site]

        except KeyError as error:
            msg = f"{site!r} is not published to this diagnosis"
            raise KeyError(msg) from error

    def readings(self, trace: Trace) -> tuple[tuple[Site, Observation], ...]:
        """Return every reading a trace took, in the order it took them.

        Parameters
        ----------
        trace : Trace
            The readings to gather.

        Returns
        -------
        tuple[tuple[Site, Observation], ...]
            Each point read at beside what was read there.

        Raises
        ------
        KeyError
            If the trace holds a reading at a point this context does not publish.
        """
        for site in trace.sites():
            self.spec_at(site)

        return trace.items()

    @override
    def __repr__(self) -> str:
        """Return how many points the context holds and which roles they cover."""
        covered = sorted(role.name for role in self._roles)
        grouped = "" if not self._groupings else f", groupings={len(self._groupings)}"
        related = "" if not self._relations else f", relations={len(self._relations)}"
        indexed = (
            "" if self._span_index is None else f", span_index={self._span_index!r}"
        )
        weighed = "" if self._weighting is None else f", weighting={self._weighting!r}"

        return (
            f"DiagnosticContext({len(self._published)} sites, roles={covered}"
            f"{grouped}{related}{indexed}{weighed})"
        )


class Measurement:
    """Stores a diagnostic measurement and its source site.

    Parameters
    ----------
    site : Site
        The published point the number was computed from.
    name : str
        What the number is, in the diagnostic's own terms.
    value : float
        The number.

    Raises
    ------
    TypeError
        If the site is not a site, the name is not a string, or the value is not a
        number.
    ValueError
        If the name is empty, or the value is infinite or a NaN.
    """

    __slots__ = ("_name", "_site", "_value")

    def __init__(self, site: Site, name: str, value: float) -> None:
        if not isinstance(site, Site):
            msg = f"a measurement's site is not a Site: {site!r}"
            raise TypeError(msg)

        self._site = site
        self._name = ensure_name("a measurement's name", name)
        self._value = ensure_finite(f"the measurement named {name!r}", value)

    @property
    def site(self) -> Site:
        """Return the published point the number was computed from."""
        return self._site

    @property
    def name(self) -> str:
        """Return what the number is, in the diagnostic's own terms."""
        return self._name

    @property
    def value(self) -> float:
        """Return the number."""
        return self._value

    def _identity(self) -> tuple[Site, str, float]:
        """Return the three fields a measurement is equal on."""
        return (self._site, self._name, self._value)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same number at the same point.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Measurement:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the site, the name and the number."""
        return hash((Measurement, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this measurement."""
        return f"Measurement({self._site!r}, {self._name!r}, {self._value!r})"


class Aggregation:
    """Stores one quantity aggregated over the spans read at a site.

    Parameters
    ----------
    site : Site
        The published point the quantity was computed from.
    name : str
        What the quantity is, in the diagnostic's own terms.
    aggregate : SpanAggregate
        Its mean under the constituent marginals and under equal weights.

    Raises
    ------
    TypeError
        If the site is not a site, the name is not a string, or the aggregate is not
        one.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_aggregate", "_name", "_site")

    def __init__(self, site: Site, name: str, aggregate: SpanAggregate) -> None:
        if not isinstance(site, Site):
            msg = f"an aggregation's site is not a Site: {site!r}"
            raise TypeError(msg)

        if type(aggregate) is not SpanAggregate:
            msg = f"an aggregation holds a SpanAggregate, not {aggregate!r}"
            raise TypeError(msg)

        self._site = site
        self._name = ensure_name("an aggregation's name", name)
        self._aggregate = aggregate

    @property
    def site(self) -> Site:
        """Return the published point the quantity was computed from."""
        return self._site

    @property
    def name(self) -> str:
        """Return what the quantity is, in the diagnostic's own terms."""
        return self._name

    @property
    def aggregate(self) -> SpanAggregate:
        """Return the two means and the span count."""
        return self._aggregate

    def _identity(self) -> tuple[Site, str, SpanAggregate]:
        """Return the three fields an aggregation is equal on."""
        return (self._site, self._name, self._aggregate)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value holds the same aggregate at the same point."""
        if type(other) is not Aggregation:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the site, the name and the aggregate."""
        return hash((Aggregation, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this aggregation."""
        return f"Aggregation({self._site!r}, {self._name!r}, {self._aggregate!r})"


class DiagnosticResult:
    """Stores a diagnostic's ordered measurements and aggregations.

    Parameters
    ----------
    tag : str
        The tag of the diagnostic that made them.
    measurements : Iterable[Measurement]
        The single numbers it measured, in the order it measured them.
    aggregations : Iterable[Aggregation]
        The quantities it aggregated over spans, each under both weightings, in the
        order it aggregated them.

    Raises
    ------
    TypeError
        If the tag is not a string, an entry is not a measurement, or an entry is not
        an aggregation.
    ValueError
        If the tag is empty, or one point is measured or aggregated twice under one
        name.
    """

    __slots__ = ("_aggregations", "_measurements", "_tag")

    def __init__(
        self,
        tag: str,
        measurements: Iterable[Measurement],
        aggregations: Iterable[Aggregation] = (),
    ) -> None:
        self._tag = ensure_name("a result's tag", tag)
        recorded = tuple(measurements)
        aggregated = tuple(aggregations)
        seen: set[tuple[Site, str]] = set()

        for measurement in recorded:
            if not isinstance(measurement, Measurement):
                msg = f"a result holds measurements, not {measurement!r}"
                raise TypeError(msg)

            _unseen(seen, measurement.site, measurement.name)

        for aggregation in aggregated:
            if type(aggregation) is not Aggregation:
                msg = f"a result aggregates through aggregations, not {aggregation!r}"
                raise TypeError(msg)

            _unseen(seen, aggregation.site, aggregation.name)

        self._measurements = recorded
        self._aggregations = aggregated

    @property
    def tag(self) -> str:
        """Return the tag of the diagnostic that made these measurements."""
        return self._tag

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        """Return the single numbers it measured, in the order it measured them."""
        return self._measurements

    @property
    def aggregations(self) -> tuple[Aggregation, ...]:
        """Return what it aggregated over spans, in the order it aggregated them."""
        return self._aggregations

    @property
    def sites(self) -> tuple[Site, ...]:
        """Return the points measured or aggregated at, each once, measured first."""
        seen: dict[Site, None] = {}

        for measurement in self._measurements:
            seen.setdefault(measurement.site, None)

        for aggregation in self._aggregations:
            seen.setdefault(aggregation.site, None)

        return tuple(seen)

    def value(self, site: Site, name: str) -> float:
        """Return one measured number.

        Parameters
        ----------
        site : Site
            The point it was computed from.
        name : str
            What the number is.

        Returns
        -------
        float
            The number.

        Raises
        ------
        KeyError
            If nothing of that name was measured at that point.
        """
        for measurement in self._measurements:
            if measurement.site == site and measurement.name == name:
                return measurement.value

        msg = f"nothing named {name!r} was measured at {site!r}"
        raise KeyError(msg)

    def aggregate(self, site: Site, name: str) -> SpanAggregate:
        """Return one quantity aggregated over spans, under both weightings.

        Parameters
        ----------
        site : Site
            The point it was computed from.
        name : str
            What the quantity is.

        Returns
        -------
        SpanAggregate
            Its mean under the constituent marginals and under equal weights.

        Raises
        ------
        KeyError
            If nothing of that name was aggregated at that point.
        """
        for aggregation in self._aggregations:
            if aggregation.site == site and aggregation.name == name:
                return aggregation.aggregate

        msg = f"nothing named {name!r} was aggregated at {site!r}"
        raise KeyError(msg)

    def _identity(
        self,
    ) -> tuple[str, tuple[Measurement, ...], tuple[Aggregation, ...]]:
        """Return the three fields a result is equal on."""
        return (self._tag, self._measurements, self._aggregations)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result holds the same entries in order.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not DiagnosticResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the tag, the measurements and the aggregations."""
        return hash((DiagnosticResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the tag and how many entries of each kind the result holds."""
        return (
            f"DiagnosticResult({self._tag!r}, {len(self._measurements)} measurements, "
            f"{len(self._aggregations)} aggregations)"
        )


def _unseen(seen: set[tuple[Site, str]], site: Site, name: str) -> None:
    """Record one named entry at one point, refusing a repeat.

    Parameters
    ----------
    seen : set[tuple[Site, str]]
        The entries recorded so far, which the new one is added to.
    site : Site
        The point of the new entry.
    name : str
        Its name.

    Raises
    ------
    ValueError
        If the point already has an entry of that name.
    """
    key = (site, name)

    if key in seen:
        msg = f"{name!r} is measured twice at {site!r}, so one of the two is lost"
        raise ValueError(msg)

    seen.add(key)
