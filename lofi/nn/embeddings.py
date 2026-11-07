"""The shared embedding over language choices."""

from typing import TYPE_CHECKING, final, override

from torch import nn

from lofi._checks import ensure_positive
from lofi.nn.sites import Access, DTypeClass, Role, Site, SiteMap, SiteSpec

if TYPE_CHECKING:
    from torch import Tensor

    from lofi._types import ComponentId
    from lofi.terms import ChoiceIndex

__all__ = ["ChoiceEmbedding"]


@final
class ChoiceEmbedding(nn.Module):
    """One learned row for every choice in a language.

    Parameters
    ----------
    index : ChoiceIndex
        The payload-aware alphabet that owns the row order.
    width : int
        The width of every learned row.
    component : ComponentId
        The component that publishes the table as a parameter site.
    """

    def __init__(
        self,
        *,
        index: ChoiceIndex,
        width: int,
        component: ComponentId,
    ) -> None:
        super().__init__()
        self._index = index
        self._width = ensure_positive("choice embedding width", width)
        self._table = nn.Embedding(index.width, self._width, padding_idx=index.PAD)
        self._sites = SiteMap.custom(
            component,
            {
                Site(component, Role.PARAMETER, ("choice-embedding",)): SiteSpec(
                    ("choice", "width"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                )
            },
        )

    @property
    def index(self) -> ChoiceIndex:
        """Return the exact choice index this table follows."""
        return self._index

    @property
    def output_width(self) -> int:
        """Return the width of one choice row."""
        return self._width

    @property
    def weight(self) -> Tensor:
        """Return the one choice table, for declared parameter tying."""
        return self._table.weight

    @override
    def forward(self, choices: Tensor, /) -> Tensor:
        """Return the learned row of every supplied choice code."""
        return self._table(choices)

    def site_map(self) -> SiteMap:
        """Return the published choice-table parameter site."""
        return self._sites
