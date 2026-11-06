"""Diagnostics over recurrent states and gate activations."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple

import torch

from lofi.nn.sites import DTypeClass, Role
from lofi.probe._diagnose import DiagnosticResult, Measurement
from lofi.probe._structure import similarity
from lofi.terms import Encoding

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor

    from lofi.nn.sites import Site
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Observation, Trace

__all__ = [
    "GatePatterns",
    "HiddenStateEvolution",
    "MemoryCellDynamics",
    "RecurrentConnectionAnalysis",
]

_AXES: Final[tuple[str, str, str]] = ("term", "slot", "width")
_REAL: Final = frozenset({DTypeClass.REAL})
_GATE_CLASSES: Final = frozenset({DTypeClass.REAL, DTypeClass.PROBABILITY})
_DISPLACEMENT: Final[str] = "normalized-displacement"
_RETENTION: Final[str] = "cosine-retention"
_ALIGNMENT: Final[str] = "directional-alignment"


def _validated_parents(parents: Iterable[int] | None, /) -> tuple[int, ...] | None:
    """Return checked parent slots, or none for an unresolved diagnostic."""
    if parents is None:
        return None

    held = tuple(parents)

    for child, parent in enumerate(held):
        if isinstance(parent, bool) or not isinstance(parent, int):
            msg = f"the parent of slot {child} is an integer, not {parent!r}"
            raise TypeError(msg)

        if parent < Encoding.NO_PARENT or parent >= len(held):
            msg = (
                f"the parent of slot {child} is {parent}, outside "
                f"{Encoding.NO_PARENT} through {len(held) - 1}"
            )
            raise ValueError(msg)

        if parent == child:
            msg = f"slot {child} cannot be its own parent"
            raise ValueError(msg)

    return held


class _TreeParents:
    """Stores the caller-supplied parent slot of every recurrent state."""

    __slots__ = ("_parents",)

    def __init__(self, parents: Iterable[int] | None = None) -> None:
        self._parents = _validated_parents(parents)

    @property
    def parents(self) -> tuple[int, ...] | None:
        """Return one parent per slot, or none when no structure was supplied."""
        return self._parents

    def _for(self, site: Site, slots: int, /) -> tuple[int, ...]:
        """Return parents aligned with one reading's slot axis."""
        if self._parents is None:
            msg = f"the recurrent reading at {site!r} has no caller-supplied parents"
            raise ValueError(msg)

        if len(self._parents) != slots:
            msg = (
                f"the recurrent reading at {site!r} has {slots} slots, but "
                f"{len(self._parents)} parents were supplied"
            )
            raise ValueError(msg)

        return self._parents


def _role_readings(
    role: Role,
    context: DiagnosticContext,
    taken: Mapping[Site, Observation],
    declared: frozenset[DTypeClass],
    /,
) -> tuple[tuple[Site, Observation], ...]:
    """Return validated observations for one recurrent role in publication order."""
    published = context.sites_for(role)

    for site in published:
        spec = context.spec_at(site)

        if spec.axes != _AXES:
            msg = (
                f"{role.name} at {site!r} is published over {spec.axes!r}, "
                f"not {_AXES!r}"
            )
            raise ValueError(msg)

        if spec.dtype_class not in declared:
            allowed = tuple(sorted(one.value for one in declared))
            msg = (
                f"{role.name} at {site!r} is published as "
                f"{spec.dtype_class.value!r}, not one of {allowed!r}"
            )
            raise ValueError(msg)

    read = tuple((site, taken[site]) for site in published if site in taken)

    if published and not read:
        msg = f"the published {role.name} points {published!r} were not observed"
        raise ValueError(msg)

    for site, observation in read:
        if observation.axes != _AXES:
            msg = (
                f"the {role.name} observation at {site!r} runs over "
                f"{observation.axes!r}, not {_AXES!r}"
            )
            raise ValueError(msg)

        if not observation.tensor.is_floating_point():
            msg = (
                f"the {role.name} observation at {site!r} is floating point, "
                f"not {observation.tensor.dtype}"
            )
            raise TypeError(msg)

        if observation.tensor.shape[-1] == 0:
            msg = f"the {role.name} observation at {site!r} has no width"
            raise ValueError(msg)

    return read


class _Edges(NamedTuple):
    """Live child and parent vectors from one recurrent observation."""

    children: Tensor
    parents: Tensor
    count: int


def _edges(reading: Observation, parents: tuple[int, ...], /) -> _Edges:
    """Return child-parent vectors whose two endpoints are completely live."""
    values = reading.tensor.to(torch.float64)
    live = reading.covering().all(dim=-1)
    child_slots = tuple(
        child for child, parent in enumerate(parents) if parent != Encoding.NO_PARENT
    )

    if not child_slots:
        empty = values[:, :0, :].reshape(-1, values.shape[-1])

        return _Edges(empty, empty, 0)

    parent_slots = tuple(parents[child] for child in child_slots)
    child_index = torch.tensor(child_slots, dtype=torch.int64, device=values.device)
    parent_index = torch.tensor(parent_slots, dtype=torch.int64, device=values.device)
    children = values.index_select(1, child_index)
    ancestors = values.index_select(1, parent_index)
    both = live.index_select(1, child_index) & live.index_select(1, parent_index)
    child_vectors = children[both]
    parent_vectors = ancestors[both]

    return _Edges(child_vectors, parent_vectors, int(both.sum()))


def _displacements(edges: _Edges, /) -> Tensor:
    """Return defined normalized displacement values for child-parent edges."""
    spread = edges.children.norm(dim=-1) + edges.parents.norm(dim=-1)
    defined = spread > 0
    distances = (edges.children - edges.parents).norm(dim=-1)

    return (distances[defined] / spread[defined]).clamp(0.0, 1.0)


def _cosines(edges: _Edges, /) -> Tensor:
    """Return defined directional cosines for child-parent edges."""
    defined = (edges.children.norm(dim=-1) > 0) & (edges.parents.norm(dim=-1) > 0)

    return similarity(edges.children[defined], edges.parents[defined])


def _count(site: Site, name: str, values: Tensor, /) -> Measurement:
    """Return how many values contribute to one recurrent summary."""
    return Measurement(site, name, float(values.numel()))


class HiddenStateEvolution(_TreeParents):
    """Measures normalized hidden state displacement from each child to its parent."""

    __slots__ = ()

    tag: ClassVar[str] = "hidden-state-evolution"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.RECURRENT_HIDDEN})
    scores: ClassVar[frozenset[str]] = frozenset({_DISPLACEMENT})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return mean live child-to-parent hidden state displacement by site."""
        taken = dict(context.readings(trace))

        if not self.requires <= context.roles:
            return DiagnosticResult(self.tag, ())

        measured: list[Measurement] = []

        for site, reading in _role_readings(
            Role.RECURRENT_HIDDEN, context, taken, _REAL
        ):
            parents = self._for(site, reading.extent("slot"))
            edges = _edges(reading, parents)
            displacement = _displacements(edges)
            measured.append(Measurement(site, "edges", float(edges.count)))
            measured.append(_count(site, "displacements", displacement))

            if displacement.numel():
                measured.append(
                    Measurement(site, _DISPLACEMENT, float(displacement.mean()))
                )

        return DiagnosticResult(self.tag, measured)


class GatePatterns:
    """Measures the first two moments of every observed recurrent gate activation."""

    __slots__ = ()

    tag: ClassVar[str] = "gate-patterns"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.GATE})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return live-value counts, means and population variances by gate site."""
        taken = dict(context.readings(trace))

        if not self.requires <= context.roles:
            return DiagnosticResult(self.tag, ())

        measured: list[Measurement] = []

        for site, reading in _role_readings(Role.GATE, context, taken, _GATE_CLASSES):
            live = reading.covering()
            values = reading.tensor.to(torch.float64)[live]
            measured.append(_count(site, "values", values))

            if values.numel():
                measured.append(Measurement(site, "mean", float(values.mean())))
                measured.append(
                    Measurement(site, "variance", float(values.var(correction=0)))
                )

        return DiagnosticResult(self.tag, measured)


class MemoryCellDynamics(_TreeParents):
    """Measures displacement and directional retention across memory cell edges."""

    __slots__ = ()

    tag: ClassVar[str] = "memory-cell-dynamics"
    requires: ClassVar[frozenset[Role]] = frozenset(
        {Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL, Role.GATE}
    )
    scores: ClassVar[frozenset[str]] = frozenset({_DISPLACEMENT})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return memory cell movement and cosine retention by cell site."""
        taken = dict(context.readings(trace))

        if not self.requires <= context.roles:
            return DiagnosticResult(self.tag, ())

        measured: list[Measurement] = []

        for site, reading in _role_readings(Role.RECURRENT_CELL, context, taken, _REAL):
            parents = self._for(site, reading.extent("slot"))
            edges = _edges(reading, parents)
            displacement = _displacements(edges)
            retention = _cosines(edges)
            measured.append(Measurement(site, "edges", float(edges.count)))
            measured.append(_count(site, "displacements", displacement))
            measured.append(_count(site, "retentions", retention))

            if displacement.numel():
                measured.append(
                    Measurement(site, _DISPLACEMENT, float(displacement.mean()))
                )

            if retention.numel():
                measured.append(Measurement(site, _RETENTION, float(retention.mean())))

        return DiagnosticResult(self.tag, measured)


class RecurrentConnectionAnalysis(_TreeParents):
    """Measures directional continuity between observed child and parent states.

    The alignment is a cosine between post-update activations at connected tree
    slots. It describes activation continuity and does not estimate a recurrent
    weight, an eigenvalue, a pre-update state or a causal effect.
    """

    __slots__ = ()

    tag: ClassVar[str] = "recurrent-connection-analysis"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.RECURRENT_HIDDEN})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return mean child-to-parent directional alignment by hidden state site."""
        taken = dict(context.readings(trace))

        if not self.requires <= context.roles:
            return DiagnosticResult(self.tag, ())

        measured: list[Measurement] = []

        for site, reading in _role_readings(
            Role.RECURRENT_HIDDEN, context, taken, _REAL
        ):
            parents = self._for(site, reading.extent("slot"))
            edges = _edges(reading, parents)
            alignment = _cosines(edges)
            measured.append(Measurement(site, "edges", float(edges.count)))
            measured.append(_count(site, "alignments", alignment))

            if alignment.numel():
                measured.append(Measurement(site, _ALIGNMENT, float(alignment.mean())))

        return DiagnosticResult(self.tag, measured)
