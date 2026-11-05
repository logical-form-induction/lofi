from typing import TYPE_CHECKING

import pytest

from lofi.nn.sites import ComponentId, Site, SiteMap, SiteSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.nn.ports import Observable


class _Published:
    """A test component publishing a fixed site map.

    Parameters
    ----------
    published : Mapping[Site, SiteSpec]
        The points exposed by the component.
    """

    def __init__(self, published: Mapping[Site, SiteSpec]) -> None:
        components = frozenset(site.component for site in published)

        if len(components) > 1:
            msg = f"one fixture component cannot publish points from {components}"
            raise ValueError(msg)

        component = (
            ComponentId("empty-published") if not components else next(iter(components))
        )
        self._published = SiteMap.custom(component, published)

    def site_map(self) -> SiteMap:
        """Return the component's published points."""
        return self._published


@pytest.fixture
def observable() -> Callable[[Mapping[Site, SiteSpec]], Observable]:
    """Return a factory for components publishing a supplied site map."""
    return _Published
