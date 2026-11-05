"""Read-only finite structures supplied by declared knowledge bases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.lf.models._evaluate import Evaluator, ModelLanguage

if TYPE_CHECKING:
    from lofi.lf.models._algebra import TruthAlgebra
    from lofi.lf.models._structure import Structure

__all__ = [
    "GivenSettings",
    "GivenStructureSource",
    "KnowledgeBaseReader",
]


@runtime_checkable
class KnowledgeBaseReader(Protocol):
    """Read one declared external interpretation into a finite structure."""

    def read(
        self,
        settings: object,
        language: ModelLanguage,
        /,
        *,
        full_function_space_cap: int,
        henkin_pool: int,
    ) -> Structure: ...


@final
class GivenSettings(FrozenValue):
    """Resolved settings for one knowledge-base-backed structure source."""

    _noun = "given-source settings"
    __slots__ = (
        "_full_function_space_cap",
        "_henkin_pool",
        "_knowledge_base",
        "_knowledge_base_kind",
    )

    def __init__(
        self,
        knowledge_base_kind: str,
        knowledge_base: object,
        /,
        *,
        full_function_space_cap: int,
        henkin_pool: int,
    ) -> None:
        object.__setattr__(self, "_knowledge_base_kind", knowledge_base_kind)
        object.__setattr__(self, "_knowledge_base", knowledge_base)
        object.__setattr__(self, "_full_function_space_cap", full_function_space_cap)
        object.__setattr__(self, "_henkin_pool", henkin_pool)

    @property
    def knowledge_base_kind(self) -> str:
        return self._knowledge_base_kind

    @property
    def knowledge_base(self) -> object:
        return self._knowledge_base

    @property
    def full_function_space_cap(self) -> int:
        return self._full_function_space_cap

    @property
    def henkin_pool(self) -> int:
        return self._henkin_pool


@final
class GivenStructureSource(FrozenValue):
    """Supply one read-only intended interpretation from a knowledge base."""

    _noun = "given structure source"
    __slots__ = ("_structure",)

    def __init__(
        self,
        language: ModelLanguage,
        structure: Structure,
        algebra: TruthAlgebra,
        /,
        *,
        theory_floor: float,
    ) -> None:
        floor = ensure_finite("a theory floor", theory_floor)

        if not 0.0 <= floor <= 1.0:
            msg = f"a theory floor lies in the unit interval, not {theory_floor!r}"
            raise ValueError(msg)

        if not Evaluator(language, structure, algebra).theory_holds(floor=floor):
            msg = "the given structure does not satisfy the declared theory"
            raise ValueError(msg)

        object.__setattr__(self, "_structure", structure)

    @property
    def name(self) -> str:
        """Return the registered source name."""
        return "given"

    @property
    def evidence_label(self) -> str:
        """Name the epistemic status of scores from the intended model."""
        return "truth in an intended model"

    @property
    def trainable(self) -> bool:
        """Return false because an external interpretation is read-only."""
        return False

    def structures(self) -> tuple[Structure, ...]:
        """Return the sole intended structure."""
        return (self._structure,)

    def open_step(self) -> None:
        """Keep the read-only intended structure across optimization steps."""
