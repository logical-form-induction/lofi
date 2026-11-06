"""Probes a run document declares, and the trained point they read.

A run document declares its probes under ``probes``, each keyed by the role its
rows are reported under. A probe's kind names a plugin in the ``lofi.probe``
entry point group, which builds the probe from the declaration's settings. The
probe is admitted against the sites the trained point's components publish
before any fitting, so a probe that names something no component publishes is
refused while the run is composed. After training it is executed against the
trained point, which it observes only through published sites, over the
point's held-out passes, and it returns one value per measurement.

Executing a probe leaves the point as it found it. Every pass runs inside
instrumented regions that restore their components when they close, and each
probe explicitly selects whether its pass records an autograd graph.
"""

import json
from contextlib import ExitStack
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, runtime_checkable

import torch

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue, Unpicklable
from lofi.nn.sites import Access
from lofi.probe._instrument import instrument
from lofi.probe._values import Trace

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.deduce import SpanIndex
    from lofi.nn.ports import Reading
    from lofi.nn.sites import Role, Site, SiteSpec
    from lofi.probe._counterfactuals import CounterfactualGeneration
    from lofi.probe._graph import ComputationGraph
    from lofi.probe._instrument import Instrumented
    from lofi.probe._interventions import Intervention
    from lofi.probe._ports import Probed
    from lofi.stats import Dependence, NullProvenance

__all__ = [
    "PROBE_RECORDS_NAME",
    "CausalHeldOutPass",
    "DeclaredProbe",
    "GradientMode",
    "GraphHeldOutPass",
    "HeldOutPass",
    "InterventionSource",
    "MissingProbeCapabilityError",
    "PassLayout",
    "ProbeBuildContext",
    "ProbeRow",
    "ProbeValue",
    "RecordedProbe",
    "RecordingProbe",
    "TrainedPoint",
    "UnpublishedSiteError",
    "observable",
    "observed_pass",
    "published_sites",
    "reached_sites",
    "read_probe_records",
    "require_roles",
    "site_label",
    "write_probe_records",
]


PROBE_RECORDS_NAME: Final = "probe-records.json"
"""The file in a run directory holding the records its declared probes persist."""


class UnpublishedSiteError(ValueError):
    """A declared probe names a role or a site no component publishes."""


class MissingProbeCapabilityError(ValueError):
    """A declared probe requires a held-out capability the point does not publish."""


class GradientMode(Enum):
    """Whether a held-out pass records an autograd graph."""

    DISABLED = "disabled"
    ENABLED = "enabled"


class ProbeValue(NamedTuple):
    """One measurement a probe took over a trained point.

    Attributes
    ----------
    measure : str
        What was measured, as the probe names it.
    site : Site | None
        The published point the measurement belongs to, or None for one that
        belongs to no single point.
    value : float
        The measured value.
    count : int
        How many passes, spans or diagnostics the value was taken over.
    null : NullProvenance | None
        The null test that produced the inferential fields, or None for an
        ordinary scalar.
    raw_p : float | None
        The unadjusted tail probability, or None for an ordinary scalar.
    adjusted_p : float | None
        Its Benjamini-Hochberg adjustment within ``family``, or None for an
        ordinary scalar.
    rejected : bool | None
        The corresponding FDR decision, or None for an ordinary scalar.
    family : str | None
        The declared multiple comparison family, or None for an ordinary scalar.
    dependence : Dependence | None
        The dependence regime under which correction ran, or None for an
        ordinary scalar.
    """

    measure: str
    site: Site | None
    value: float
    count: int
    null: NullProvenance | None
    raw_p: float | None
    adjusted_p: float | None
    rejected: bool | None
    family: str | None
    dependence: Dependence | None


class ProbeRow(NamedTuple):
    """One measurement reported under the role a run document declared its probe.

    Attributes
    ----------
    probe : str
        The role the probe is declared under.
    kind : str
        The declared kind of the probe.
    measure : str
        What was measured.
    site : str | None
        The published point, as ``component/ROLE/key``, or None.
    value : float
        The measured value.
    count : int
        How many passes, spans or diagnostics the value was taken over.
    null : NullProvenance | None
        The null test behind an inferential row, or None for an ordinary row.
    raw_p : float | None
        The raw p-value, or None for an ordinary row.
    adjusted_p : float | None
        The Benjamini-Hochberg adjusted p-value, or None for an ordinary row.
    rejected : bool | None
        The FDR decision, or None for an ordinary row.
    family : str | None
        The declared multiple comparison family, or None for an ordinary row.
    dependence : Dependence | None
        The declared dependence regime, or None for an ordinary row.
    """

    probe: str
    kind: str
    measure: str
    site: str | None
    value: float
    count: int
    null: NullProvenance | None
    raw_p: float | None
    adjusted_p: float | None
    rejected: bool | None
    family: str | None
    dependence: Dependence | None


class PassLayout(NamedTuple):
    """What one held-out pass covered.

    Attributes
    ----------
    words : tuple[tuple[str, ...], ...]
        The words of each item of the batch, in batch order.
    spans : SpanIndex | None
        The span addressing the pass's chart readings follow, or None where the
        pass built no chart.
    gold : tuple[Mapping[tuple[int, int], str] | None, ...]
        Each item's gold constituents, the half-open span of each mapped to its
        label, or None for an item without a gold analysis.
    """

    words: tuple[tuple[str, ...], ...]
    spans: SpanIndex | None
    gold: tuple[Mapping[tuple[int, int], str] | None, ...]


class HeldOutPass(Protocol):
    """One held-out batch of a trained point, run on demand."""

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        """Run the trained forward pass over the batch once.

        Parameters
        ----------
        gradients : GradientMode
            Whether operations in the pass record an autograd graph.

        Returns
        -------
        PassLayout
            What the pass covered.
        """
        ...


@runtime_checkable
class GraphHeldOutPass(HeldOutPass, Protocol):
    """A held-out pass that publishes its computation graph as an output."""

    def computation_graph(self) -> ComputationGraph[object]:
        """Return the graph published by this pass."""
        ...


@runtime_checkable
class CausalHeldOutPass(HeldOutPass, Protocol):
    """A held-out pass that publishes a model-ready causal population."""

    def causal_population(self) -> CounterfactualGeneration:
        """Return the causal population published by this pass."""
        ...


class TrainedPoint(Protocol):
    """A trained model whose published components a declared probe reads."""

    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        """Return every component that publishes sites and admits a region."""
        ...

    @property
    def passes(self) -> tuple[HeldOutPass, ...]:
        """Return the held-out passes, in held-out order."""
        ...


class ProbeBuildContext(FrozenValue, Unpicklable):
    """The declared role and pass capabilities supplied to a probe builder."""

    __slots__ = ("_point", "_role")

    _noun = "probe build context"
    _because = "holds live trained-point pass capabilities"

    def __init__(self, role: str, point: TrainedPoint, /) -> None:
        object.__setattr__(self, "_role", ensure_name("a probe role", role))
        object.__setattr__(self, "_point", point)

    @property
    def role(self) -> str:
        """Return the role under which the probe was declared."""
        return self._role

    def graph_passes(self) -> tuple[GraphHeldOutPass, ...]:
        """Return passes publishing graphs, or refuse the first pass that does not."""
        held: list[GraphHeldOutPass] = []

        for position, candidate in enumerate(self._point.passes):
            if not isinstance(candidate, GraphHeldOutPass):
                msg = (
                    f"probe {self._role!r} requires a computation graph, but "
                    f"held-out pass {position} publishes none"
                )
                raise MissingProbeCapabilityError(msg)
            held.append(candidate)

        return tuple(held)

    def causal_passes(self) -> tuple[CausalHeldOutPass, ...]:
        """Return passes publishing causal populations, or name the first absence."""
        held: list[CausalHeldOutPass] = []

        for position, candidate in enumerate(self._point.passes):
            if not isinstance(candidate, CausalHeldOutPass):
                msg = (
                    f"probe {self._role!r} requires a causal population, but "
                    f"held-out pass {position} publishes none"
                )
                raise MissingProbeCapabilityError(msg)
            held.append(candidate)

        return tuple(held)


class DeclaredProbe(Protocol):
    """A probe built from a run document's declaration."""

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a trained point whose published sites this probe cannot read.

        Parameters
        ----------
        published : Mapping[Site, SiteSpec]
            The points the trained point's components publish, as
            :func:`reached_sites` leaves them when a run composes its probes.

        Raises
        ------
        UnpublishedSiteError
            If the probe names a role or a site no component publishes with the
            access the probe needs.
        """
        ...

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return what this probe measured over a trained point.

        Parameters
        ----------
        point : TrainedPoint
            The trained point, already admitted.

        Returns
        -------
        tuple[ProbeValue, ...]
            One value per measurement, in a stable order.
        """
        ...


class RecordedProbe(NamedTuple):
    """A probe's values and the records it persists beside them.

    Attributes
    ----------
    values : tuple[ProbeValue, ...]
        One value per measurement, in a stable order.
    records : Mapping[str, JsonValue]
        Named JSON records persisted beside the probe's rows, empty where the
        probe records nothing.
    """

    values: tuple[ProbeValue, ...]
    records: Mapping[str, JsonValue]


@runtime_checkable
class RecordingProbe(Protocol):
    """A probe whose measurement also yields records persisted beside its rows."""

    def recorded(self, point: TrainedPoint, /) -> RecordedProbe:
        """Return what the probe measured and the records it persists.

        Parameters
        ----------
        point : TrainedPoint
            The trained point, already admitted.
        """
        ...


class InterventionSource(Protocol):
    """Draws the intervention applied at a site from the clean pass's readings."""

    @property
    def access(self) -> Access:
        """Return what the intervened site must permit for the draw and its patch."""
        ...

    def drawn(self, clean: Trace, /) -> Intervention:
        """Return the intervention, given what the unpatched pass read.

        Parameters
        ----------
        clean : Trace
            The closed readings of the same pass without the intervention.

        Returns
        -------
        Intervention
            What replaces the value at the site.
        """
        ...


def site_label(site: Site, /) -> str:
    """Return a point as ``component/ROLE/key``, the key's parts joined by slashes.

    Parameters
    ----------
    site : Site
        The point to name.
    """
    return "/".join((str(site.component), site.role.name, *map(str, site.key)))


def published_sites(
    components: Mapping[str, Probed[Reading]], /
) -> Mapping[Site, SiteSpec]:
    """Return every point the components publish, each under its specification.

    Parameters
    ----------
    components : Mapping[str, Probed[Reading]]
        The components of a trained point.
    """
    return MappingProxyType(
        {
            site: spec
            for component in components.values()
            for site, spec in component.site_map().items()
        }
    )


def require_roles(
    roles: Iterable[Role],
    published: Mapping[Site, SiteSpec],
    access: Access,
    /,
) -> None:
    """Refuse roles no published point plays with the access a probe needs.

    Parameters
    ----------
    roles : Iterable[Role]
        The roles a probe declares.
    published : Mapping[Site, SiteSpec]
        Every point the trained point publishes.
    access : Access
        What the probe needs each role's points to permit.

    Raises
    ------
    UnpublishedSiteError
        If a role is played by no published point permitting the access; the
        message lists the roles that are.
    """
    offered = frozenset(
        site.role for site, spec in published.items() if access in spec.access
    )
    missing = sorted(role.name for role in frozenset(roles) - offered)

    if missing:
        available = sorted(role.name for role in offered)
        msg = (
            f"no component publishes {', '.join(missing)} permitting {access.name}; "
            f"the trained point publishes {available}"
        )
        raise UnpublishedSiteError(msg)


def _nested[Result](
    regions: tuple[Instrumented[Reading], ...], forward: Callable[[], Result], /
) -> Callable[[], Result]:
    """Return the pass run inside every region, the first region outermost."""
    run = forward

    for region in reversed(regions):
        run = _inside(region, run)

    return run


def _inside[Result](
    region: Instrumented[Reading], forward: Callable[[], Result], /
) -> Callable[[], Result]:
    """Return the pass run inside one region."""

    def run() -> Result:
        return region(forward)

    return run


def observed_pass(
    point: TrainedPoint,
    held_out: HeldOutPass,
    /,
    *,
    observe: frozenset[Site],
    patch: Mapping[Site, Intervention],
    gradients: GradientMode,
) -> tuple[Trace, PassLayout]:
    """Run one held-out pass with points observed and patched across components.

    One region is opened over each component that publishes an observed or a
    patched point, and the pass runs once inside all of them.

    Parameters
    ----------
    point : TrainedPoint
        The trained point.
    held_out : HeldOutPass
        The pass to run.
    observe : frozenset[Site]
        The points to read.
    patch : Mapping[Site, Intervention]
        What replaces the value at a point, keyed by the point.
    gradients : GradientMode
        Whether operations in the pass record an autograd graph.

    Returns
    -------
    tuple[Trace, PassLayout]
        The closed readings of every observed point the pass reached, and what
        the pass covered.

    Raises
    ------
    ValueError
        If a point is published by no component of the point, or does not permit
        what it was named for.
    """
    named = observe | frozenset(patch)
    owners = {
        site: component
        for component in point.components.values()
        for site in component.site_map()
        if site in named
    }
    unowned = sorted(map(site_label, named - frozenset(owners)))

    if unowned:
        msg = f"no component of the trained point publishes {unowned}"
        raise ValueError(msg)

    trace = Trace()

    with ExitStack() as regions:
        opened = tuple(
            regions.enter_context(
                instrument(
                    component,
                    observe=frozenset(
                        site for site in observe if owners[site] is component
                    ),
                    patch={
                        site: intervention
                        for site, intervention in patch.items()
                        if owners[site] is component
                    },
                )
            )
            for component in point.components.values()
            if any(owner is component for owner in owners.values())
        )
        layout = _nested(opened, lambda: held_out(gradients))()

        for region in opened:
            for site in region.trace.sites():
                trace.record(region.trace[site])

    trace.close()

    return trace, layout


def reached_sites(point: TrainedPoint, /) -> Mapping[Site, SiteSpec]:
    """Return the published points the held-out passes reach, each under its spec.

    Every held-out pass runs once with every observable published point read,
    under a forked random state, so the passes consume no draw the point's
    later use would see. A point that does not permit observation cannot be
    seen to be reached, and is kept.

    Parameters
    ----------
    point : TrainedPoint
        The point, trained or not; which points a pass reaches depends on the
        model's structure, not on its weights.

    Returns
    -------
    Mapping[Site, SiteSpec]
        Every published point some pass read, and every published point that
        cannot be read.
    """
    published = published_sites(point.components)
    readable = observable(published)
    seen: set[Site] = set()

    with torch.random.fork_rng():
        for held_out in point.passes:
            trace, _ = observed_pass(
                point,
                held_out,
                observe=readable,
                patch={},
                gradients=GradientMode.DISABLED,
            )
            seen.update(trace.sites())

    return MappingProxyType(
        {
            site: spec
            for site, spec in published.items()
            if site in seen or site not in readable
        }
    )


def observable(published: Mapping[Site, SiteSpec], /) -> frozenset[Site]:
    """Return the published points that permit observation.

    Parameters
    ----------
    published : Mapping[Site, SiteSpec]
        Every point the trained point publishes.
    """
    return frozenset(
        site for site, spec in published.items() if Access.OBSERVE in spec.access
    )


def write_probe_records(
    path: Path, records: Mapping[str, Mapping[str, JsonValue]], /
) -> None:
    """Write probe records as one JSON object keyed by probe role.

    Parameters
    ----------
    path : Path
        The records file.
    records : Mapping[str, Mapping[str, JsonValue]]
        Each recording probe's named records under its role.
    """
    path.write_text(
        json.dumps(
            {role: dict(named) for role, named in records.items()},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_probe_records(path: Path, /) -> dict[str, dict[str, JsonValue]]:
    """Return the probe records a records file holds, keyed by probe role.

    Parameters
    ----------
    path : Path
        The records file :func:`write_probe_records` wrote.

    Raises
    ------
    TypeError
        If the file does not hold one object of named records per role.
    """
    document = json.loads(path.read_text("utf-8"))

    if not isinstance(document, dict) or not all(
        isinstance(named, dict) for named in document.values()
    ):
        msg = f"a probe records file holds an object per role, not {document!r}"
        raise TypeError(msg)

    return cast("dict[str, dict[str, JsonValue]]", document)
