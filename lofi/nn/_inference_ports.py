"""Batch capabilities used by grounded inference responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import ReadoutId
    from lofi.lf import ArgumentRenaming, InferenceLabel

__all__ = ["InferencePairs", "JudgmentItems"]


@runtime_checkable
class InferencePairs(Protocol):
    """A batch of premise and hypothesis pairs with alignments and labels."""

    @property
    def premises(self) -> tuple[ReadoutId, ...]:
        """Return the id of each pair's premise beam, in pair order."""

    @property
    def hypotheses(self) -> tuple[ReadoutId, ...]:
        """Return the id of each pair's hypothesis beam, in pair order."""

    @property
    def alignments(self) -> tuple[tuple[ArgumentRenaming, ...], ...]:
        """Return the argument alignments each pair declares."""

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        """Return the relation each pair declares, in pair order."""


@runtime_checkable
class JudgmentItems(Protocol):
    """A pair batch carrying ordinal judgments by local pair position."""

    @property
    def judgments(self) -> Sequence[tuple[int, int]]:
        """Return local pair positions and their observed ordinal levels."""
