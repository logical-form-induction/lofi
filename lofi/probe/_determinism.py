"""Bitwise run-to-run determinism over published observation sites."""

from typing import TYPE_CHECKING, Final, NamedTuple

import torch

from lofi._checks import ensure_non_negative
from lofi.probe._instrument import instrument

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site
    from lofi.probe._ports import Probed

__all__ = ["DeterminismReport", "SiteDeterminism", "determinism"]

_RUNS: Final = 5


class SiteDeterminism(NamedTuple):
    """The repeated calls whose observation at one site differs from the first.

    Attributes
    ----------
    site : Site
        Published point compared.
    divergent_runs : tuple[int, ...]
        One-based calls, from two through five, that differ bitwise from the first.
    """

    site: Site
    divergent_runs: tuple[int, ...]

    @property
    def deterministic(self) -> bool:
        """Return whether every call observed the same value at this site."""
        return not self.divergent_runs


class DeterminismReport(NamedTuple):
    """Bitwise agreement at each requested site, ordered by site representation.

    Attributes
    ----------
    sites : tuple[SiteDeterminism, ...]
        One result for every requested site.
    """

    sites: tuple[SiteDeterminism, ...]

    @property
    def divergent_sites(self) -> frozenset[Site]:
        """Return the sites whose repeated observations differ."""
        return frozenset(row.site for row in self.sites if not row.deterministic)

    def at(self, site: Site, /) -> SiteDeterminism:
        """Return the result for one requested site.

        Parameters
        ----------
        site : Site
            Requested point to look up.

        Raises
        ------
        KeyError
            If the site was not requested.
        """
        for row in self.sites:
            if row.site == site:
                return row

        msg = f"the determinism report has no result for {site!r}"
        raise KeyError(msg)


class _Snapshot(NamedTuple):
    """The observable parts of one reading, copied away from the component."""

    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None


def _snapshot[ReadingT: Reading, ResultT](
    model: Probed[ReadingT],
    forward: Callable[[], ResultT],
    sites: frozenset[Site],
    seed: int,
) -> dict[Site, _Snapshot]:
    """Return one seeded call's observations at every requested site."""
    with torch.random.fork_rng(), instrument(model, observe=sites) as region:
        torch.manual_seed(seed)
        region(forward)

    missing = sites.difference(region.trace.sites())

    if missing:
        ordered_missing = tuple(sorted(missing, key=repr))
        msg = f"the call did not reach the requested sites {ordered_missing!r}"
        raise ValueError(msg)

    return {
        site: _Snapshot(
            observed.tensor.clone(),
            observed.axes,
            None if observed.mask is None else observed.mask.clone(),
        )
        for site, observed in region.trace.items()
    }


def _same(first: _Snapshot, other: _Snapshot) -> bool:
    """Return whether two observations agree exactly in metadata and values."""
    if first.axes != other.axes or not _bitwise_same(first.tensor, other.tensor):
        return False

    if first.mask is None or other.mask is None:
        return first.mask is other.mask

    return _bitwise_same(first.mask, other.mask)


def _raw_bytes(tensor: Tensor) -> Tensor:
    """Return a contiguous one-dimensional view of a tensor's raw bytes."""
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _bitwise_same(first: Tensor, other: Tensor) -> bool:
    """Return whether two tensors have the same layout metadata and raw bytes."""
    return (
        first.dtype == other.dtype
        and first.device == other.device
        and tuple(first.shape) == tuple(other.shape)
        and torch.equal(_raw_bytes(first), _raw_bytes(other))
    )


def determinism[ReadingT: Reading, ResultT](
    model: Probed[ReadingT],
    forward: Callable[[], ResultT],
    /,
    *,
    sites: frozenset[Site],
    seed: int,
) -> DeterminismReport:
    """Compare five identically seeded calls at every requested site.

    Parameters
    ----------
    model : Probed[ReadingT]
        Component whose published points are observed.
    forward : Callable[[], ResultT]
        One call through the component, with its input already fixed.
    sites : frozenset[Site]
        Published points to compare, at least one.
    seed : int
        Nonnegative seed restored before each call. The caller's RNG state is restored
        after every call.

    Returns
    -------
    DeterminismReport
        Per-site bitwise disagreements with the first call.

    Raises
    ------
    ValueError
        If no site is requested, the seed is negative, or a call does not reach a
        requested site.
    TypeError
        If the seed is not an integer.
    """
    if not sites:
        msg = "a determinism diagnostic observes at least one site"
        raise ValueError(msg)

    held_seed = ensure_non_negative("a determinism seed", seed)
    ordered = tuple(sorted(sites, key=repr))
    observed = tuple(_snapshot(model, forward, sites, held_seed) for _ in range(_RUNS))
    first = observed[0]

    return DeterminismReport(
        tuple(
            SiteDeterminism(
                site,
                tuple(
                    run
                    for run, repeated in enumerate(observed[1:], start=2)
                    if not _same(first[site], repeated[site])
                ),
            )
            for site in ordered
        )
    )
