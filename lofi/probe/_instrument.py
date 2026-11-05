"""Callable-component observation and intervention."""

from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, final, override

import torch

from lofi._checks import CHECKS
from lofi._frozen import FrozenValue
from lofi.nn.ports import Reading
from lofi.nn.sites import Access, Site, SiteMap
from lofi.probe._dtypes import ensure_declared
from lofi.probe._roles import ensure_publishes
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.ports import InstrumentScope, Patch
    from lofi.probe._interventions import Intervention
    from lofi.probe._ports import Probed

__all__ = [
    "Differentiated",
    "GradientSites",
    "Instrumented",
    "differentiate",
    "instrument",
]

type _Discipline = Callable[[Observation], None]
"""Specifies what a region does with a reading before it files it."""


class _Unchecked:
    """Files every reading without holding it to what its point declared."""

    __slots__ = ()

    def __call__(self, observation: Observation, /) -> None:
        """Accept one reading.

        Parameters
        ----------
        observation : Observation
            The reading to accept.
        """

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this discipline."""
        return "_Unchecked()"


_UNCHECKED: Final = _Unchecked()
"""The discipline a region opens under unless another is chosen."""


class _Checked:
    """Holds every reading to the numeric class and fill its point declared.

    Parameters
    ----------
    published : SiteMap
        The map the component publishes, which is where a declaration is read from.
    """

    __slots__ = ("_published",)

    def __init__(self, published: SiteMap) -> None:
        self._published = published

    def __call__(self, observation: Observation, /) -> None:
        """Hold one reading to what its point declared.

        Parameters
        ----------
        observation : Observation
            The reading to hold to its declaration.

        Raises
        ------
        ValueError
            If a live position of the reading lies outside the class its point
            declared, or an invalid position holds a number other than the one that
            class fixes.
        """
        ensure_declared(observation, self._published[observation.site].dtype_class)

    @override
    def __repr__(self) -> str:
        """Return how many points the discipline reads declarations from."""
        return f"_Checked({len(self._published)} sites)"


def _as_observation(reading: Reading) -> Observation:
    """Return one reading as the record a probe works in.

    Parameters
    ----------
    reading : Reading
        What was found at one point.

    Returns
    -------
    Observation
        The same numbers, detached, under the point and axes the reading names.

    Raises
    ------
    TypeError
        If the value does not have the shape of a reading.
    """
    if not isinstance(reading, Reading):
        msg = f"what was read is not a Reading: {reading!r}"
        raise TypeError(msg)

    return Observation(
        reading.site, reading.tensor, axes=reading.axes, mask=reading.mask
    )


class _Standing[ReadingT: Reading]:
    """Stores an intervention patch installed by a component scope.

    Parameters
    ----------
    intervention : Intervention
        What replaces the value at the point.
    """

    __slots__ = ("_intervention",)

    def __init__(self, intervention: Intervention) -> None:
        self._intervention = intervention

    @property
    def intervention(self) -> Intervention:
        """Return what replaces the value at the point."""
        return self._intervention

    def stands_in_for(self, reading: ReadingT, /) -> Tensor:
        """Return what stands in place of the value at one point.

        Parameters
        ----------
        reading : ReadingT
            What was found there.

        Returns
        -------
        Tensor
            The replacement, with the reading's shape and original numbers at positions
            excluded by its mask.

        Raises
        ------
        TypeError
            If the value does not have the shape of a reading.
        ValueError
            If the intervention returned a value of a different shape than the reading,
            which stands in for some other point than this one.
        """
        observed = _as_observation(reading)
        replacement = self._intervention.stands_in_for(observed)

        if tuple(replacement.shape) != tuple(observed.tensor.shape):
            msg = (
                f"{self._intervention!r} stood in for the reading at "
                f"{observed.site!r} with a value of shape "
                f"{tuple(replacement.shape)} rather than "
                f"{tuple(observed.tensor.shape)}"
            )
            raise ValueError(msg)

        if observed.mask is None:
            return replacement

        return torch.where(observed.covering(), replacement, observed.tensor)

    @override
    def __repr__(self) -> str:
        """Return the intervention this patch contains."""
        return f"_Standing({self._intervention!r})"


class Instrumented[ReadingT: Reading]:
    """Wraps an instrumented callable and its trace.

    Parameters
    ----------
    scope : InstrumentScope[ReadingT]
        The region the component opened, already entered.
    trace : Trace
        Where the readings go.
    observed : frozenset[Site]
        The points the region was opened over for reading.
    discipline : _Discipline
        What each reading is held to before it is filed, chosen once as the region
        opens. Unless one is given, every reading is accepted.

    Attributes
    ----------
    trace : Trace
        The readings the region took, empty until a pass has run.
    """

    __slots__ = ("_discipline", "_observed", "_ran", "_scope", "_trace")

    def __init__(
        self,
        scope: InstrumentScope[ReadingT],
        trace: Trace,
        *,
        observed: frozenset[Site],
        discipline: _Discipline = _UNCHECKED,
    ) -> None:
        self._scope = scope
        self._trace = trace
        self._observed = observed
        self._discipline = discipline
        self._ran = False

    @property
    def trace(self) -> Trace:
        """Return the readings the region took, empty until a pass has run."""
        return self._trace

    @property
    def ran(self) -> bool:
        """Return whether a pass has run inside the region."""
        return self._ran

    def __call__[Result](self, forward: Callable[[], Result], /) -> Result:
        """Run one pass inside the region and file what it read.

        Parameters
        ----------
        forward : Callable[[], Result]
            The zero-argument pass to run.

        Returns
        -------
        Result
            The pass result.

        Raises
        ------
        RuntimeError
            If a pass has already run inside this region.
        ValueError
            If the region reads an unopened point or stores a reading under another
            point, or, with the checks on, if a reading departs from what its point
            declared.
        """
        if self._ran:
            msg = (
                "a pass has already run in this region, and readings from two "
                "passes do not belong to one trace"
            )
            raise RuntimeError(msg)

        self._ran = True
        computed = forward()

        for site, reading in self._scope.readings.items():
            if site not in self._observed:
                msg = (
                    f"the region read {site!r}, which it was not opened over, "
                    f"so it read more of the component than was named"
                )
                raise ValueError(msg)

            observed = _as_observation(reading)

            if observed.site != site:
                msg = (
                    f"the region filed a reading from {observed.site!r} under "
                    f"{site!r}, so what was read is not where it came from"
                )
                raise ValueError(msg)

            self._discipline(observed)
            self._trace.record(observed)

        return computed

    @override
    def __repr__(self) -> str:
        """Return whether a pass has run and how much the trace holds."""
        return f"Instrumented(ran={self._ran!r}, trace={self._trace!r})"


class _Reach(NamedTuple):
    """One value a pass handed a tap, and the positions its reading covers."""

    tensor: Tensor
    covering: Tensor


class _Tap[ReadingT: Reading]:
    """Keeps every value a pass hands one point attached to the pass's graph.

    A pass may reach a point more than once when each reach's reading covers
    positions no other reach covers. The gradient at a position then belongs to
    the one reach whose reading covers it.

    Parameters
    ----------
    site : Site
        The point the tap is installed at.
    """

    __slots__ = ("_reaches", "_site")

    def __init__(self, site: Site) -> None:
        self._site = site
        self._reaches: list[_Reach] = []

    @property
    def reached(self) -> bool:
        """Return whether the pass has reached the point."""
        return bool(self._reaches)

    @property
    def attached(self) -> tuple[Tensor, ...]:
        """Return every value the pass was handed at the point, in reach order."""
        return tuple(reach.tensor for reach in self._reaches)

    def stands_in_for(self, reading: ReadingT, /) -> Tensor:
        """Return the value at the point, attached so that a gradient reaches it.

        Parameters
        ----------
        reading : ReadingT
            What was found there.

        Returns
        -------
        Tensor
            A copy of the same numbers, requiring a gradient.

        Raises
        ------
        TypeError
            If the value does not have the shape of a reading.
        RuntimeError
            If the reading is of another shape than an earlier reach's, or covers a
            position an earlier reach covered, since the gradient there would belong
            to neither reading.
        """
        covering = _as_observation(reading).covering()

        for earlier in self._reaches:
            if tuple(earlier.covering.shape) != tuple(covering.shape):
                msg = (
                    f"the pass reached {self._site!r} with a reading of shape "
                    f"{tuple(covering.shape)} after one of shape "
                    f"{tuple(earlier.covering.shape)}, so the two do not divide "
                    "one point's positions"
                )
                raise RuntimeError(msg)

            if bool((earlier.covering & covering).any()):
                msg = (
                    f"the pass reached {self._site!r} twice at one position, so a "
                    "gradient there belongs to neither reading"
                )
                raise RuntimeError(msg)

        held = reading.tensor.clone()

        if not held.requires_grad:
            held.requires_grad_()

        self._reaches.append(_Reach(held, covering))

        return held

    def gradient(self, found: Sequence[Tensor | None], /) -> Tensor:
        """Return the gradient at every position, each from the reach covering it.

        Parameters
        ----------
        found : Sequence[Tensor | None]
            The gradient at each attached value, in reach order, None where the
            scalar does not depend on it.

        Returns
        -------
        Tensor
            Of the reading's shape and dtype, zero wherever no reach covers a
            position or the scalar does not depend on the reach covering it.

        Raises
        ------
        RuntimeError
            If the pass has not reached the point.
        """
        if not self._reaches:
            msg = f"the pass has not reached {self._site!r}, so nothing is attached"
            raise RuntimeError(msg)

        total = torch.zeros_like(self._reaches[0].tensor)

        for reach, gradient in zip(self._reaches, found, strict=True):
            if gradient is not None:
                total = torch.where(reach.covering, gradient, total)

        return total

    @override
    def __repr__(self) -> str:
        """Return the point and how many times the pass reached it."""
        return f"_Tap({self._site!r}, reaches={len(self._reaches)})"


class Differentiated[ReadingT: Reading]:
    """Wraps an instrumented callable whose readings a gradient can be taken at.

    Parameters
    ----------
    region : Instrumented[ReadingT]
        The region the readings are filed through.
    taps : Mapping[Site, _Tap[ReadingT]]
        What keeps each point attached to the pass, keyed by the point.
    """

    __slots__ = ("_differentiated", "_region", "_taps")

    def __init__(
        self, region: Instrumented[ReadingT], taps: Mapping[Site, _Tap[ReadingT]]
    ) -> None:
        self._region = region
        self._taps = taps
        self._differentiated = False

    @property
    def trace(self) -> Trace:
        """Return the readings the region took, empty until a pass has run."""
        return self._region.trace

    @property
    def ran(self) -> bool:
        """Return whether a pass has run inside the region."""
        return self._region.ran

    def __call__[Result](self, forward: Callable[[], Result], /) -> Result:
        """Run one pass inside the region and file what it read.

        Parameters
        ----------
        forward : Callable[[], Result]
            The zero-argument pass to run.

        Returns
        -------
        Result
            The pass result.

        Raises
        ------
        RuntimeError
            If a pass has already run inside this region.
        ValueError
            If the pass never reached a point the region was opened over, or if the
            readings fail what the underlying region holds them to.
        """
        computed = self._region(forward)

        for site in sorted(self._taps, key=repr):
            if not self._taps[site].reached:
                msg = (
                    f"the pass never reached {site!r}, so no gradient can be "
                    "taken there"
                )
                raise ValueError(msg)

        return computed

    def gradients(self, metric: Tensor, /) -> Mapping[Site, Tensor]:
        """Return the gradient of a scalar with respect to the value at each point.

        Parameters
        ----------
        metric : Tensor
            A scalar the pass computed through the points.

        Returns
        -------
        Mapping[Site, Tensor]
            The gradient at each point the region was opened over, of the reading's
            own shape. Each position takes its gradient from the one reach whose
            reading covers it, and is zero where no reach covers it or the scalar
            does not depend on the reach that does. No other tensor the scalar was
            computed through has a gradient accumulated.

        Raises
        ------
        RuntimeError
            If no pass has run, or a gradient has already been taken.
        ValueError
            If the metric is not a scalar, or no gradient reaches it.
        """
        if not self._region.ran:
            msg = "no pass has run in this region, so there is nothing to differentiate"
            raise RuntimeError(msg)

        if self._differentiated:
            msg = "a gradient has already been taken through this region's pass"
            raise RuntimeError(msg)

        if metric.ndim != 0:
            msg = f"a gradient is taken of a scalar, not of shape {tuple(metric.shape)}"
            raise ValueError(msg)

        if not metric.requires_grad:
            msg = "no gradient reaches the metric, so none reaches the points"
            raise ValueError(msg)

        self._differentiated = True
        sites = tuple(sorted(self._taps, key=repr))
        held = tuple(value for site in sites for value in self._taps[site].attached)
        found = iter(torch.autograd.grad(metric, held, allow_unused=True))

        return MappingProxyType(
            {
                site: self._taps[site].gradient(
                    tuple(next(found) for _ in self._taps[site].attached)
                )
                for site in sites
            }
        )

    @override
    def __repr__(self) -> str:
        """Return whether a pass has run and how many points are tapped."""
        return f"Differentiated(ran={self.ran!r}, taps={len(self._taps)})"


def _resolve(published: SiteMap, named: frozenset[Site], permitted: Access) -> None:
    """Check every point named against the map, once, before a region opens.

    Parameters
    ----------
    published : SiteMap
        The map the component publishes, which is the only place a point is looked up.
    named : frozenset[Site]
        The points named for this permission.
    permitted : Access
        What those points have to permit.

    Raises
    ------
    TypeError
        If something other than a site was named.
    ValueError
        If a point is not published, or does not permit what it was named for.
    """
    for site in sorted(named, key=repr):
        if not isinstance(site, Site):
            msg = f"a point named for a region is not a Site: {site!r}"
            raise TypeError(msg)

        if site not in published:
            msg = (
                f"{site!r} is not published by this component, which publishes "
                f"{len(published)} points"
            )
            raise ValueError(msg)

        if permitted not in published[site].access:
            msg = (
                f"{site!r} permits {published[site].access!r}, so it cannot be "
                f"named for {permitted!r}"
            )
            raise ValueError(msg)


def _validated(published: SiteMap) -> SiteMap:
    """Return a tagged site map validated from its own metadata.

    Parameters
    ----------
    published : SiteMap
        The map entering the instrumentation boundary.

    Returns
    -------
    SiteMap
        The validated map.

    Raises
    ------
    TypeError
        If a component returns an untagged mapping.
    ValueError
        If the component's sites violate the inventory declared by its kind.
    """
    if not isinstance(published, SiteMap):
        msg = (
            "a component must publish a tagged SiteMap or take the explicit "
            "SiteMap.custom path"
        )
        raise TypeError(msg)

    return ensure_publishes(published)


@final
class GradientSites(FrozenValue):
    """The points of one component's map that a gradient can be taken at.

    The map and the points are checked once, here, so that any number of regions
    can be opened over them without checking either again.

    Parameters
    ----------
    published : SiteMap
        The map the component publishes.
    sites : frozenset[Site]
        The points to read and take a gradient at. Each must permit observation and
        patching and be differentiable, since a gradient reaches a point through the
        value the pass is handed there.

    Raises
    ------
    TypeError
        If the map is untagged, or a point named is not a site.
    ValueError
        If the map violates the inventory its kind declares, or a point is not
        published, does not permit observation or patching, or is declared
        non-differentiable.
    """

    _noun = "gradient site set"

    __slots__ = ("_published", "_sites")

    def __init__(self, published: SiteMap, sites: frozenset[Site]) -> None:
        validated = _validated(published)

        _resolve(validated, sites, Access.OBSERVE)
        _resolve(validated, sites, Access.PATCH)

        for site in sorted(sites, key=repr):
            if not validated[site].differentiable:
                msg = (
                    f"{site!r} is declared non-differentiable, so no gradient "
                    "reaches it"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_published", validated)
        object.__setattr__(self, "_sites", frozenset(sites))

    @property
    def published(self) -> SiteMap:
        """Return the validated map the points were checked against."""
        return self._published

    @property
    def sites(self) -> frozenset[Site]:
        """Return the points a gradient can be taken at."""
        return self._sites

    @override
    def __repr__(self) -> str:
        """Return the component and how many points a gradient can be taken at."""
        return f"GradientSites({self._published.component!r}, sites={len(self._sites)})"


@contextmanager
def instrument[ReadingT: Reading](
    model: Probed[ReadingT],
    /,
    *,
    observe: frozenset[Site] = frozenset(),
    patch: Mapping[Site, Intervention] = MappingProxyType({}),
    capacity: int | None = None,
    compress: bool = False,
) -> Iterator[Instrumented[ReadingT]]:
    """Instrument a component for observation and intervention.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component to open the region over.
    observe : frozenset[Site]
        The points to read, drawn from what the component publishes.
    patch : Mapping[Site, Intervention]
        What replaces the value at a point, keyed by the point.
    capacity : int | None
        How many readings the trace keeps, or None to keep all of them.
    compress : bool
        Whether the trace keeps floating-point readings at half precision.

    Yields
    ------
    Instrumented
        A callable for one pass and its observation trace.

    Raises
    ------
    TypeError
        If a point named is not a site, or the compression flag is not a boolean.
    ValueError
        If a point is not published by the component, or does not permit what it was
        named for, or the bound is negative or zero.
    """
    published = _validated(model.site_map())

    _resolve(published, observe, Access.OBSERVE)
    _resolve(published, frozenset(patch), Access.PATCH)

    discipline: _Discipline = _Checked(published) if CHECKS else _UNCHECKED
    patches: Mapping[Site, Patch[ReadingT]] = {
        site: _Standing(intervention) for site, intervention in patch.items()
    }
    trace = Trace(capacity=capacity, compress=compress)

    try:
        with model.instrument(observe, patches=patches) as scope:
            yield Instrumented(scope, trace, observed=observe, discipline=discipline)

    finally:
        trace.close()


@contextmanager
def differentiate[ReadingT: Reading](
    model: Probed[ReadingT],
    /,
    *,
    sites: GradientSites,
    capacity: int | None = None,
    compress: bool = False,
) -> Iterator[Differentiated[ReadingT]]:
    """Instrument a component for observation with a gradient at each point.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component to open the region over, whose map the points were checked
        against.
    sites : GradientSites
        The points to read and take a gradient at. A pass may reach a point several
        times when no two reaches' readings cover one position.
    capacity : int | None
        How many readings the trace keeps, or None to keep all of them.
    compress : bool
        Whether the trace keeps floating-point readings at half precision.

    Yields
    ------
    Differentiated
        A callable for one pass, its observation trace, and the gradients at the
        points once the pass has run.

    Raises
    ------
    TypeError
        If the compression flag is not a boolean.
    ValueError
        If the bound is negative or zero.
    """
    points = sites.sites
    discipline: _Discipline = _Checked(sites.published) if CHECKS else _UNCHECKED
    taps: dict[Site, _Tap[ReadingT]] = {site: _Tap(site) for site in points}
    trace = Trace(capacity=capacity, compress=compress)

    try:
        with model.instrument(points, patches=taps) as scope:
            yield Differentiated(
                Instrumented(scope, trace, observed=points, discipline=discipline),
                MappingProxyType(taps),
            )

    finally:
        trace.close()
