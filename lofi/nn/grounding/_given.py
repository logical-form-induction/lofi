"""Construction of a knowledge-base-backed finite-structure source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

from lofi._frozen import FrozenValue
from lofi.lf.models import (
    GivenSettings,
    GivenStructureSource,
    KnowledgeBaseReader,
    ModelLanguage,
    StructureSource,
    TruthAlgebra,
)
from lofi.registry import Registry

if TYPE_CHECKING:
    from lofi.lf.models import StructureSourceFactory

__all__ = ["GIVEN"]


@final
class _GivenFactory(FrozenValue):
    """Build a given source through its declared knowledge-base reader."""

    _noun = "structure-source factory"
    __slots__ = ()

    def build(
        self,
        settings: object,
        language: ModelLanguage,
        algebra: TruthAlgebra,
        /,
        *,
        theory_floor: float,
    ) -> StructureSource:
        if not isinstance(settings, GivenSettings):
            msg = f"the given source reads GivenSettings, not {settings!r}"
            raise TypeError(msg)

        reader = Registry[object]("knowledge_base").get(settings.knowledge_base_kind)

        if not isinstance(reader, KnowledgeBaseReader):
            msg = (
                f"knowledge-base entry {settings.knowledge_base_kind!r} names no reader"
            )
            raise TypeError(msg)

        structure = reader.read(
            settings.knowledge_base,
            language,
            full_function_space_cap=settings.full_function_space_cap,
            henkin_pool=settings.henkin_pool,
        )
        return GivenStructureSource(
            language,
            structure,
            algebra,
            theory_floor=theory_floor,
        )


GIVEN: Final[StructureSourceFactory] = _GivenFactory()
