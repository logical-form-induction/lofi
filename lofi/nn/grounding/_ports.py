"""The single batched inference-grounding port."""

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue, Unpicklable
from lofi.nn.grounding._values import Phase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ReadoutId
    from lofi.lf import ArgumentRenaming, InferenceLabel
    from lofi.nn._values import LogDistribution
    from lofi.nn.entailment import PairedVectors
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding._values import GroundedBatch, GroundingMetadata
    from lofi.nn.sites import SiteMap

__all__ = ["GroundedPairs", "GroundingInput", "GroundingQuery", "InferenceGrounding"]


class GroundingQuery(StrEnum):
    """Name which paired proof questions a response model can score."""

    OBSERVED_LABEL = "observed-label"
    FULL = "full"


@runtime_checkable
class GroundedPairs(Protocol):
    """Pair addresses, alignments, and targets consumed by a grounding."""

    @property
    def premises(self) -> tuple[ReadoutId, ...]: ...

    @property
    def hypotheses(self) -> tuple[ReadoutId, ...]: ...

    @property
    def alignments(self) -> tuple[tuple[ArgumentRenaming, ...], ...]: ...

    @property
    def labels(self) -> tuple[InferenceLabel, ...]: ...


@final
class GroundingInput(FrozenValue, Unpicklable):
    """The published forms, pairs, vectors, and phase of one grounding call."""

    _noun = "grounding input"
    _because = "holds graph-connected tensors"
    __slots__ = (
        "_choice_rows",
        "_forms",
        "_pairs",
        "_phase",
        "_query",
        "_sentences",
    )

    def __init__(
        self,
        forms: Mapping[ReadoutId, WeightedTerms],
        /,
        *,
        choice_rows: Mapping[ReadoutId, LogDistribution],
        pairs: GroundedPairs,
        query: GroundingQuery,
        sentences: PairedVectors | None,
        phase: Phase,
    ) -> None:
        if not isinstance(phase, Phase):
            msg = f"a grounding phase is Phase, not {phase!r}"
            raise TypeError(msg)

        if not isinstance(query, GroundingQuery):
            msg = f"a grounding query is GroundingQuery, not {query!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_forms", MappingProxyType(dict(forms)))
        object.__setattr__(self, "_choice_rows", MappingProxyType(dict(choice_rows)))
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_sentences", sentences)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_query", query)

    @property
    def forms(self) -> Mapping[ReadoutId, WeightedTerms]:
        return self._forms

    @property
    def choice_rows(self) -> Mapping[ReadoutId, LogDistribution]:
        """Return payload-aware decoder marginals keyed by pair-side id."""
        return self._choice_rows

    @property
    def pairs(self) -> GroundedPairs:
        return self._pairs

    @property
    def sentences(self) -> PairedVectors | None:
        return self._sentences

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def query(self) -> GroundingQuery:
        """Return which paired proof questions the response can score."""
        return self._query


@runtime_checkable
class InferenceGrounding(Protocol):
    """Publish one label distribution for every pair in a batch."""

    @property
    def labels(self) -> tuple[InferenceLabel, ...]: ...

    @property
    def requires(self) -> frozenset[str]: ...

    @property
    def grounding_metadata(self) -> GroundingMetadata: ...

    def ground(self, batch: GroundingInput, /) -> GroundedBatch: ...

    def open_step(self) -> None: ...

    def site_map(self) -> SiteMap: ...
