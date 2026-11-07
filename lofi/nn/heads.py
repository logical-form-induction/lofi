"""Signature-derived heads over the language choice axis."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

import torch
from torch import nn

from lofi.nn._values import LogDistribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.spec.layers import HeadKind
    from lofi.terms import ChoiceIndex, HoleContext, Sort

__all__ = ["TermHeads"]


@final
class TermHeads(nn.Module):
    """Map slot states to distributions over legal language choices.

    Parameters
    ----------
    index : ChoiceIndex
        The language-owned choice alphabet.
    input_width : int
        The width of each slot state.
    kind : HeadKind
        ``flat`` for one categorical, or ``factored`` for a sort-specific
        categorical over families followed by the selected family head.
    """

    def __init__(
        self,
        *,
        index: ChoiceIndex,
        input_width: int,
        kind: HeadKind,
    ) -> None:
        super().__init__()
        self._index = index
        self._width = input_width
        self._kind: HeadKind = kind
        self._flat = nn.Linear(self._width, index.width) if kind == "flat" else None
        by_sort = {
            sort: tuple(key for key in index.family_slices if key[0] == sort)
            for sort in index.signature.sorts
        }
        keys = tuple(index.family_slices)
        key_names = {key: str(at) for at, key in enumerate(keys)}
        self._family_keys = MappingProxyType(key_names)
        self._by_sort = MappingProxyType(by_sort)
        self._structures = nn.ModuleDict(
            {
                sort: nn.Linear(self._width, len(families))
                for sort, families in by_sort.items()
            }
            if kind == "factored"
            else {}
        )
        self._families = nn.ModuleDict(
            {
                key_names[key]: nn.Linear(self._width, len(codes))
                for key, codes in index.family_slices.items()
                if len(codes) > 1
            }
            if kind == "factored"
            else {}
        )

    @property
    def index(self) -> ChoiceIndex:
        """Return the choice index that determines every output column."""
        return self._index

    @property
    def input_width(self) -> int:
        """Return the width of one input state."""
        return self._width

    @property
    def kind(self) -> HeadKind:
        """Return the declared parameterization."""
        return self._kind

    @property
    def families(self) -> Mapping[tuple[Sort, str], range]:
        """Return the signature-derived choice slice of every family."""
        return self._index.family_slices

    def _flat_row(self, state: Tensor, legal: Tensor, /) -> Tensor:
        """Return one flat normalized row."""
        return self.logits(state).masked_fill(~legal, -torch.inf).log_softmax(dim=0)

    def logits(self, states: Tensor, /) -> Tensor:
        """Return unnormalized flat logits over the shared choice alphabet.

        This is the one parameterized readout used both by dynamically masked
        frontier decisions and by views that collapse payload choices to their
        operator symbols.

        Parameters
        ----------
        states : Tensor
            States with the decoder width on the innermost axis.

        Returns
        -------
        Tensor
            The same leading axes with the choice axis innermost.

        Raises
        ------
        ValueError
            If the states do not have this head's input width innermost.
        RuntimeError
            If this is a factored head, whose family normalization has no single
            unnormalized flat view.
        """
        if states.ndim < 1 or states.shape[-1] != self._width:
            msg = (
                f"term heads read width {self._width} innermost, not shape "
                f"{tuple(states.shape)}"
            )
            raise ValueError(msg)

        if self._flat is None:
            msg = "flat logits require flat term heads"
            raise RuntimeError(msg)

        return self._flat(states)

    def _factored_row(
        self,
        state: Tensor,
        legal: Tensor,
        sort: Sort,
        /,
    ) -> Tensor:
        """Return one family-factored normalized row."""
        families = self._by_sort[sort]
        available = state.new_tensor(
            [
                bool(legal[codes.start : codes.stop].any())
                for codes in (self._index.family_slices[key] for key in families)
            ],
            dtype=torch.bool,
        )
        structure = self._structures[sort](state)
        structure = structure.masked_fill(~available, -torch.inf).log_softmax(dim=0)
        result = state.new_full((self._index.width,), -torch.inf)

        for at, key in enumerate(families):
            codes = self._index.family_slices[key]
            admitted = legal[codes.start : codes.stop]

            if not bool(admitted.any()):
                continue

            name = self._family_keys[key]
            conditional = (
                self._families[name](state).masked_fill(~admitted, -torch.inf)
                if name in self._families
                else state.new_zeros((1,))
            )
            conditional = conditional.log_softmax(dim=0)
            result[codes.start : codes.stop] = structure[at] + conditional

        return result

    @override
    def forward(
        self,
        states: Tensor,
        contexts: Sequence[HoleContext],
        /,
    ) -> LogDistribution:
        """Return one normalized legal-choice row per slot state."""
        if states.ndim != 2 or states.shape[1] != self._width:
            msg = (
                f"term heads read ('term', 'width') at width {self._width}, "
                f"not shape {tuple(states.shape)}"
            )
            raise ValueError(msg)

        if states.shape[0] != len(contexts):
            msg = (
                f"{states.shape[0]} term states need the same number of hole "
                f"contexts, not {len(contexts)}"
            )
            raise ValueError(msg)

        masks = tuple(self._index.legal(context) for context in contexts)

        if any(not any(row) for row in masks):
            at = next(at for at, row in enumerate(masks) if not any(row))
            msg = f"hole context {at} admits no choice under the language bound"
            raise ValueError(msg)

        legal = torch.tensor(masks, dtype=torch.bool, device=states.device)
        rows = torch.stack(
            tuple(
                self._flat_row(state, mask)
                if self._kind == "flat"
                else self._factored_row(state, mask, context.sort)
                for state, mask, context in zip(states, legal, contexts, strict=True)
            )
        )
        return LogDistribution(
            rows,
            ("term", "choice"),
            legal,
            normalized_over="choice",
        )
