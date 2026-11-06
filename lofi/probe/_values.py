"""Observations and bounded traces."""

from collections import OrderedDict
from typing import TYPE_CHECKING, override

import torch

from lofi._checks import ensure_name, ensure_non_negative
from lofi.nn.sites import Site

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch import Tensor

__all__ = ["Observation", "ReadingCache", "Trace"]


def _named_axes(axes: Iterable[str], rank: int) -> tuple[str, ...]:
    """Return one name per axis.

    Parameters
    ----------
    axes : Iterable[str]
        The names, outermost first.
    rank : int
        How many axes the value they describe has.

    Returns
    -------
    tuple[str, ...]
        The names, unchanged.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty, if two names agree, or if the count does not match the rank.
    """
    named = tuple(
        ensure_name(f"axis {position} of an observation", name)
        for position, name in enumerate(axes)
    )

    if len(named) != rank:
        msg = f"a reading of rank {rank} needs {rank} axis names, but got {named}"
        raise ValueError(msg)

    seen: set[str] = set()

    for name in named:
        if name in seen:
            msg = f"an observation declares the axis {name!r} twice"
            raise ValueError(msg)

        seen.add(name)

    return named


def _covering_mask(mask: Tensor | None, tensor: Tensor) -> Tensor | None:
    """Return a mask that covers the outermost axes of a reading, or None.

    Parameters
    ----------
    mask : Tensor | None
        The mask to check, or None where every position holds data.
    tensor : Tensor
        The reading it covers.

    Returns
    -------
    Tensor | None
        The mask, detached, or None.

    Raises
    ------
    TypeError
        If the mask is not a tensor, or does not hold truth values.
    ValueError
        If it spans more axes than the reading has, or if an axis it spans is of a
        different extent than the reading's.
    """
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        msg = f"an observation's mask is not a tensor: {mask!r}"
        raise TypeError(msg)

    if mask.dtype is not torch.bool:
        msg = f"an observation's mask holds truth values, not {mask.dtype}"
        raise TypeError(msg)

    if mask.ndim > tensor.ndim:
        msg = (
            f"a mask over {mask.ndim} axes cannot cover a reading of rank {tensor.ndim}"
        )
        raise ValueError(msg)

    covered = tuple(tensor.shape[: mask.ndim])

    if tuple(mask.shape) != covered:
        msg = (
            f"a mask of shape {tuple(mask.shape)} does not cover the outermost "
            f"axes of a reading of shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)

    return mask.detach()


def _coordinate(axis: str, value: int, extent: int) -> int:
    """Return one selection coordinate.

    Parameters
    ----------
    axis : str
        The axis being selected on, for the message.
    value : int
        The coordinate given for it.
    extent : int
        How many positions that axis has.

    Returns
    -------
    int
        The coordinate, unchanged.

    Raises
    ------
    TypeError
        If the coordinate is not an integer, or is a bool, which reaches a coordinate
        only by mistake.
    ValueError
        If it is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"the coordinate on axis {axis!r} is not an integer: {value!r}"
        raise TypeError(msg)

    if value < 0:
        msg = f"the coordinate on axis {axis!r} counts from the start, not {value}"
        raise ValueError(msg)

    if value >= extent:
        msg = f"axis {axis!r} has {extent} positions, so {value} is beyond it"
        raise ValueError(msg)

    return value


class Observation:
    """Records a reading, its named axes, and its mask.

    Attributes
    ----------
    site : Site
        The published point the reading was taken at.
    tensor : Tensor
        The detached observed tensor.
    mask : Tensor | None
        Which positions of the outermost axes hold data, or None where all do.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    rank : int
        How many axes the reading has.

    Parameters
    ----------
    site : Site
        The published point the reading was taken at.
    tensor : Tensor
        What was found there.
    axes : Iterable[str]
        The name of each axis, outermost first, one per axis.
    mask : Tensor | None
        Which positions of the outermost axes hold data, or None where every position
        does.

    Raises
    ------
    TypeError
        If the site is not a site, the reading is not a tensor, an axis name is not a
        string, or the mask is not a tensor of truth values.
    ValueError
        If the axis names do not cover the reading's axes exactly, if two agree, or if
        the mask does not cover the reading's outermost axes.
    """

    __slots__ = ("_axes", "_mask", "_site", "_tensor")

    def __init__(
        self,
        site: Site,
        tensor: Tensor,
        *,
        axes: Iterable[str],
        mask: Tensor | None = None,
    ) -> None:
        if not isinstance(site, Site):
            msg = f"an observation's site is not a Site: {site!r}"
            raise TypeError(msg)

        if not isinstance(tensor, torch.Tensor):
            msg = f"an observation's reading is not a tensor: {tensor!r}"
            raise TypeError(msg)

        self._site = site
        self._tensor = tensor.detach()
        self._axes = _named_axes(axes, tensor.ndim)
        self._mask = _covering_mask(mask, tensor)

    @property
    def site(self) -> Site:
        """Return the published point the reading was taken at."""
        return self._site

    @property
    def tensor(self) -> Tensor:
        """Return the detached observed tensor."""
        return self._tensor

    @property
    def mask(self) -> Tensor | None:
        """Return which positions of the outermost axes hold data, or None."""
        return self._mask

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""
        return self._axes

    @property
    def rank(self) -> int:
        """Return how many axes the reading has."""
        return len(self._axes)

    def covering(self) -> Tensor:
        """Return which positions of the reading hold data, at its own shape.

        Returns
        -------
        Tensor
            Of the reading's rank and shape, holding truth values, true where the
            reading holds data.
        """
        if self._mask is None:
            return torch.ones_like(self._tensor, dtype=torch.bool)

        under = (1,) * (self._tensor.ndim - self._mask.ndim)

        return self._mask.reshape(tuple(self._mask.shape) + under).expand_as(
            self._tensor
        )

    def extent(self, axis: str) -> int:
        """Return how many positions the named axis has.

        Parameters
        ----------
        axis : str
            The axis to measure.

        Returns
        -------
        int
            Its extent.

        Raises
        ------
        KeyError
            If the observation declares no axis of that name.
        """
        return int(self._tensor.shape[self.position(axis)])

    def position(self, axis: str) -> int:
        """Return which axis of the reading the named one is.

        Parameters
        ----------
        axis : str
            The axis name to look up.

        Returns
        -------
        int
            Its position, outermost first.

        Raises
        ------
        KeyError
            If the observation declares no axis of that name.
        """
        for position, name in enumerate(self._axes):
            if name == axis:
                return position

        msg = f"no axis {axis!r} was observed here, only {self._axes}"
        raise KeyError(msg)

    def select(self, **coordinates: int) -> Observation:
        """Return the reading at one coordinate of each named axis.

        Parameters
        ----------
        **coordinates : int
            One coordinate per axis to select on, keyed by the axis name the observation
            declares.

        Returns
        -------
        Observation
            The selection, at the same site, over the axes that were not named.

        Raises
        ------
        KeyError
            If the name is not an observation axis.
        TypeError
            If a coordinate is not an integer.
        ValueError
            If a coordinate is negative or beyond its axis.
        """
        chosen = {
            axis: (self.position(axis), value) for axis, value in coordinates.items()
        }

        for axis, (offset, value) in chosen.items():
            _coordinate(axis, value, int(self._tensor.shape[offset]))

        taken = dict(chosen.values())
        index = tuple(
            taken.get(offset, slice(None)) for offset in range(self._tensor.ndim)
        )
        kept = tuple(
            name for offset, name in enumerate(self._axes) if offset not in taken
        )

        mask = self._mask

        if mask is not None:
            mask = mask[
                tuple(taken.get(offset, slice(None)) for offset in range(mask.ndim))
            ]

        return Observation(self._site, self._tensor[index], axes=kept, mask=mask)

    @override
    def __repr__(self) -> str:
        """Return the site, the axes and the extents, without the numbers."""
        return (
            f"Observation({self._site!r}, axes={self._axes!r}, "
            f"shape={tuple(self._tensor.shape)!r}, "
            f"masked={self._mask is not None!r})"
        )


def _bound(capacity: int | None) -> int | None:
    """Return how many readings a store may hold, or None for no bound.

    Parameters
    ----------
    capacity : int | None
        The entry bound, or None for an unbounded store.

    Returns
    -------
    int | None
        The bound, unchanged.

    Raises
    ------
    TypeError
        If the bound is not an integer.
    ValueError
        If the bound is negative, or is zero.
    """
    if capacity is None:
        return None

    bounded = ensure_non_negative("a reading store's capacity", capacity)

    if bounded == 0:
        msg = "a reading store holding nothing would drop every reading it took"
        raise ValueError(msg)

    return bounded


def _footprint(observation: Observation) -> int:
    """Return how many bytes one reading occupies, its mask included.

    Parameters
    ----------
    observation : Observation
        The reading to measure.

    Returns
    -------
    int
        The bytes occupied by the values and, when present, their mask.
    """
    tensor = observation.tensor
    taken = tensor.element_size() * tensor.nelement()
    mask = observation.mask

    if mask is not None:
        taken += mask.element_size() * mask.nelement()

    return taken


def _compressed(observation: Observation) -> Observation:
    """Return one reading at half precision, or unchanged where it has none.

    Parameters
    ----------
    observation : Observation
        The reading to compress.

    Returns
    -------
    Observation
        The same reading, at half precision where its numbers are floating-point and
        unchanged where they are not.
    """
    if not observation.tensor.is_floating_point():
        return observation

    return Observation(
        observation.site,
        observation.tensor.to(torch.float16),
        axes=observation.axes,
        mask=observation.mask,
    )


class ReadingCache:
    """Stores bounded readings by site.

    Parameters
    ----------
    capacity : int | None
        How many readings the store may hold, or None for no bound.
    compress : bool
        Whether floating-point readings are kept at half precision.

    Attributes
    ----------
    capacity : int | None
        How many readings the store may hold, or None for no bound.
    compresses : bool
        Whether floating-point readings are kept at half precision.
    hits : int
        How many fetches found the reading they asked for.
    misses : int
        How many did not.
    evictions : int
        How many readings were dropped to stay inside the bound.
    nbytes : int
        How many bytes the readings held right now occupy.

    Raises
    ------
    TypeError
        If the bound is not an integer, or the compression flag is not a boolean.
    ValueError
        If the bound is negative or zero.
    """

    __slots__ = (
        "_bytes",
        "_capacity",
        "_compress",
        "_evictions",
        "_hits",
        "_misses",
        "_recency",
        "_stored",
    )

    def __init__(self, *, capacity: int | None = None, compress: bool = False) -> None:
        if not isinstance(compress, bool):
            msg = f"a reading store's compression flag is not a boolean: {compress!r}"
            raise TypeError(msg)

        self._capacity = _bound(capacity)
        self._compress = compress
        self._stored: dict[Site, Observation] = {}
        self._recency: OrderedDict[Site, None] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def capacity(self) -> int | None:
        """Return how many readings the store may hold, or None for no bound."""
        return self._capacity

    @property
    def compresses(self) -> bool:
        """Return whether floating-point readings are kept at half precision."""
        return self._compress

    @property
    def hits(self) -> int:
        """Return how many fetches found the reading they asked for."""
        return self._hits

    @property
    def misses(self) -> int:
        """Return how many fetches did not find the reading they asked for."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Return how many readings were dropped to stay inside the bound."""
        return self._evictions

    @property
    def nbytes(self) -> int:
        """Return how many bytes the readings held right now occupy."""
        return self._bytes

    def put(self, observation: Observation) -> None:
        """Store one reading under the point it was read at.

        Parameters
        ----------
        observation : Observation
            The reading to store.

        Raises
        ------
        TypeError
            If the value is not an observation.
        """
        if not isinstance(observation, Observation):
            msg = f"a reading store holds observations, not {observation!r}"
            raise TypeError(msg)

        site = observation.site
        kept = _compressed(observation) if self._compress else observation

        if site in self._stored:
            self._bytes -= _footprint(self._stored[site])

        self._stored[site] = kept
        self._bytes += _footprint(kept)
        self._recency[site] = None
        self._recency.move_to_end(site)

        while self._capacity is not None and len(self._stored) > self._capacity:
            self._evict()

    def _evict(self) -> None:
        """Drop the reading that has gone longest unread."""
        stale, _ = self._recency.popitem(last=False)

        self._bytes -= _footprint(self._stored.pop(stale))
        self._evictions += 1

    def get(self, site: Site) -> Observation:
        """Return the reading held at one point, counting the fetch.

        Parameters
        ----------
        site : Site
            The point to look up.

        Returns
        -------
        Observation
            The reading held there.

        Raises
        ------
        KeyError
            If nothing is held at that point, whether because nothing was read there or
            because what was read has been evicted.
        """
        try:
            found = self._stored[site]

        except KeyError:
            self._misses += 1
            msg = f"no reading is held at {site!r}"

            raise KeyError(msg) from None

        self._hits += 1
        self._recency.move_to_end(site)

        return found

    def recency(self) -> tuple[Site, ...]:
        """Return the points held, longest unread first.

        Returns
        -------
        tuple[Site, ...]
            The points, least recently used first.
        """
        return tuple(self._recency)

    def items(self) -> tuple[tuple[Site, Observation], ...]:
        """Return stored site-observation pairs.

        Returns
        -------
        tuple[tuple[Site, Observation], ...]
            The pairs, in the order the readings arrived.
        """
        return tuple(self._stored.items())

    def __contains__[SiteT](self, site: SiteT) -> bool:
        """Return whether a reading is held without updating counters.

        Parameters
        ----------
        site : SiteT
            The point to look for.
        """
        return site in self._stored

    def __iter__(self) -> Iterator[Site]:
        """Return the points held, in the order their readings arrived."""
        return iter(self._stored)

    def __len__(self) -> int:
        """Return how many readings are held."""
        return len(self._stored)

    @override
    def __repr__(self) -> str:
        """Return the cache state and counters."""
        return (
            f"ReadingCache({len(self._stored)} of {self._capacity!r}, "
            f"bytes={self._bytes!r}, hits={self._hits!r}, "
            f"misses={self._misses!r}, evictions={self._evictions!r}, "
            f"compresses={self._compress!r})"
        )


class Trace:
    """Stores readings collected by one scope.

    Parameters
    ----------
    capacity : int | None
        How many readings to keep, or None to keep all of them.
    compress : bool
        Whether to keep floating-point readings at half precision.

    Attributes
    ----------
    closed : bool
        Whether the scope that writes this trace has finished.
    capacity : int | None
        How many readings this trace keeps, or None for all of them.
    hits : int
        How many lookups found the reading they asked for.
    misses : int
        How many did not.
    evictions : int
        How many readings were dropped to stay inside the bound.
    nbytes : int
        How many bytes the readings held right now occupy.

    Raises
    ------
    TypeError
        If the bound is not an integer, or the compression flag is not a boolean.
    ValueError
        If the bound is negative or zero.
    """

    __slots__ = ("_closed", "_readings")

    def __init__(self, *, capacity: int | None = None, compress: bool = False) -> None:
        self._readings = ReadingCache(capacity=capacity, compress=compress)
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether the scope that writes this trace has finished."""
        return self._closed

    @property
    def capacity(self) -> int | None:
        """Return how many readings this trace keeps, or None for all of them."""
        return self._readings.capacity

    @property
    def hits(self) -> int:
        """Return how many lookups found the reading they asked for."""
        return self._readings.hits

    @property
    def misses(self) -> int:
        """Return how many lookups did not find the reading they asked for."""
        return self._readings.misses

    @property
    def evictions(self) -> int:
        """Return how many readings were dropped to stay inside the bound."""
        return self._readings.evictions

    @property
    def nbytes(self) -> int:
        """Return how many bytes the readings held right now occupy."""
        return self._readings.nbytes

    def record(self, observation: Observation) -> None:
        """Add one reading to the trace.

        Parameters
        ----------
        observation : Observation
            The reading to add.

        Raises
        ------
        TypeError
            If the value is not an observation.
        RuntimeError
            If the trace has closed, or if a reading at that point is already here.
        """
        if not isinstance(observation, Observation):
            msg = f"a trace records observations, not {observation!r}"
            raise TypeError(msg)

        if self._closed:
            msg = f"this trace closed before a reading at {observation.site!r} arrived"
            raise RuntimeError(msg)

        if observation.site in self._readings:
            msg = f"this trace already holds a reading at {observation.site!r}"
            raise RuntimeError(msg)

        self._readings.put(observation)

    def close(self) -> None:
        """Freeze the trace, which the scope that owns it does as it exits."""
        self._closed = True

    def __getitem__(self, site: Site) -> Observation:
        """Return the reading taken at one point.

        Parameters
        ----------
        site : Site
            The point to look up.

        Returns
        -------
        Observation
            The reading taken there.

        Raises
        ------
        KeyError
            If nothing was read at that point, or if what was read there has since been
            dropped to keep the trace inside its bound.
        """
        try:
            return self._readings.get(site)

        except KeyError as error:
            msg = f"nothing was read at {site!r} in this trace"
            raise KeyError(msg) from error

    def __contains__[SiteT](self, site: SiteT) -> bool:
        """Return whether a reading was taken at one point.

        Parameters
        ----------
        site : SiteT
            The point to look for.
        """
        return site in self._readings

    def __iter__(self) -> Iterator[Site]:
        """Return the points read at, in the order they were read."""
        return iter(self._readings)

    def __len__(self) -> int:
        """Return how many points were read at."""
        return len(self._readings)

    def sites(self) -> tuple[Site, ...]:
        """Return the points read at, in the order they were read.

        Returns
        -------
        tuple[Site, ...]
            The points, which are the keys of the trace.
        """
        return tuple(self._readings)

    def items(self) -> tuple[tuple[Site, Observation], ...]:
        """Return recorded site-observation pairs.

        Returns
        -------
        tuple[tuple[Site, Observation], ...]
            The pairs, in the order they were read.
        """
        return self._readings.items()

    @override
    def __repr__(self) -> str:
        """Return how many points the trace holds and whether it has closed."""
        return f"Trace({len(self._readings)} sites, closed={self._closed!r})"
