from typing import TYPE_CHECKING, NamedTuple, NoReturn, Protocol

import pytest
import torch

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn.sites import Site, SiteMap

__all__ = [
    "Instrumented",
    "Observed",
    "Publishing",
    "SiteMapSuite",
    "SiteObservation",
]


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


class Publishing(Protocol):
    def site_map(self) -> SiteMap:
        """Return the points an observer may name inside this component.

        Returns
        -------
        SiteMap
            Each published point and its specification.
        """
        ...


class Observed(Protocol):
    @property
    def axes(self) -> tuple[str, ...]: ...


class SiteObservation[T: Publishing](NamedTuple):
    """Inputs for observing a component through its site map.

    Parameters
    ----------
    build : Callable[[], T]
        Return the component under test, ready to be run.
    observe : Callable[[T], Mapping[Site, Observed]]
        Run it and return what was read at each point it exposes, keyed by the site each
        is under.
    read : Callable[[T], tuple[Tensor, ...]]
        Return the tensors a pass computes, in a fixed order, so that a reading taken
        before an observation can be compared with one taken after.
    """

    build: Callable[[], T]
    observe: Callable[[T], Mapping[Site, Observed]]
    read: Callable[[T], tuple[Tensor, ...]]


class Instrumented[T: Publishing](Protocol):
    @property
    def sites(self) -> SiteObservation[T]: ...


class SiteMapSuite:
    pytestmark = pytest.mark.conformance

    def test_the_site_map_matches_what_instrumentation_yields[T: Publishing](
        self,
        conformance: Instrumented[T],
    ) -> None:
        """Check that observing yields the declared points and disturbs nothing.

        Parameters
        ----------
        conformance : Instrumented[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a published site yields nothing, if a site yields something the map does
            not declare, if an observation's axes are not the ones its specification
            names, or if two passes bracketing an observation compute different numbers.
        """
        sites = conformance.sites
        subject = sites.build()
        published = subject.site_map()
        before = tuple(tensor.clone() for tensor in sites.read(subject))
        observed = sites.observe(subject)
        after = sites.read(subject)

        if frozenset(observed) != frozenset(published):
            _fail(
                f"the map declares {sorted(map(repr, published))} and the pass "
                f"exposed {sorted(map(repr, observed))}",
                law="the_site_map_matches_what_instrumentation_yields",
            )

        for site, reading in observed.items():
            if reading.axes != published[site].axes:
                _fail(
                    f"{site!r} declares the axes {published[site].axes} and was "
                    f"read over {reading.axes}",
                    law="the_site_map_matches_what_instrumentation_yields",
                )

        if len(after) != len(before) or not all(
            torch.equal(one, other) for one, other in zip(before, after, strict=True)
        ):
            _fail(
                "the numbers before and after an observation are not the same, "
                "so instrumenting the component moved it",
                law="the_site_map_matches_what_instrumentation_yields",
            )
