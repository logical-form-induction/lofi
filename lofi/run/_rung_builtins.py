"""Built-in contraction-rung plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from lofi.run._boundary.experiment import (
    CorpusScaleRung,
    FallbackRung,
    LanguageRung,
    ResolutionIVRung,
    SeedRung,
    UnpreregisteredCellsRung,
)
from lofi.run._boundary.sweep import _as_point
from lofi.run._contraction import (
    DropInOrder,
    Fallback,
    PreregisteredCells,
    ResolutionIVFraction,
    SeedFloor,
)
from lofi.run._rung import RungPlugin

if TYPE_CHECKING:
    from lofi.run._contraction import Rung
    from lofi.run._rung import RungContext

__all__ = [
    "CORPUS_SCALE_PLUGIN",
    "FALLBACK_PLUGIN",
    "LANGUAGES_PLUGIN",
    "RESOLUTION_IV_PLUGIN",
    "SEEDS_PLUGIN",
    "UNPREREGISTERED_CELLS_PLUGIN",
]


def _unpreregistered_cells(
    declaration: UnpreregisteredCellsRung, context: RungContext, /
) -> Rung:
    """Keep only the axis values that realize a preregistered level."""
    del declaration
    realized: dict[str, frozenset[str]] = {}

    for key, values in context.spanned.values():
        realized[key] = realized.get(key, frozenset()) | values

    return PreregisteredCells(realized)


def _resolution_iv(declaration: ResolutionIVRung, context: RungContext, /) -> Rung:
    """Reduce the two-level axes to the declared resolution-IV fraction."""
    del context
    return ResolutionIVFraction(
        tuple(
            (generator.factor, generator.word) for generator in declaration.generators
        )
    )


def _languages(declaration: LanguageRung, context: RungContext, /) -> Rung:
    """Drop languages from one axis in the declared order."""
    del context
    return DropInOrder(declaration.kind, declaration.axis, declaration.order)


def _seeds(declaration: SeedRung, context: RungContext, /) -> Rung:
    """Reduce every cell's seeds to the declared floor.

    Raises
    ------
    ValueError
        If the floor exceeds the seeds the sweep declares or falls below the
        seeds a reported cell rests on.
    """
    if declaration.floor > context.seeds:
        msg = (
            f"the seed floor of {declaration.floor} exceeds the "
            f"{context.seeds} seeds the sweep declares"
        )
        raise ValueError(msg)

    if declaration.floor < context.minimum_seeds:
        msg = (
            f"a seed floor of {declaration.floor} is below the "
            f"{context.minimum_seeds} seeds a reported cell rests on"
        )
        raise ValueError(msg)

    return SeedFloor(declaration.floor)


def _corpus_scale(declaration: CorpusScaleRung, context: RungContext, /) -> Rung:
    """Drop corpus scales from one axis in the declared order."""
    del context
    return DropInOrder(
        declaration.kind,
        declaration.axis,
        tuple(_as_point(value) for value in declaration.order),
    )


def _fallback(declaration: FallbackRung, context: RungContext, /) -> Rung:
    """Switch the experiment to its decision point's fallback claim."""
    del declaration, context
    return Fallback()


UNPREREGISTERED_CELLS_PLUGIN: Final = RungPlugin(
    spec=UnpreregisteredCellsRung, build=_unpreregistered_cells
)
RESOLUTION_IV_PLUGIN: Final = RungPlugin(spec=ResolutionIVRung, build=_resolution_iv)
LANGUAGES_PLUGIN: Final = RungPlugin(spec=LanguageRung, build=_languages)
SEEDS_PLUGIN: Final = RungPlugin(spec=SeedRung, build=_seeds)
CORPUS_SCALE_PLUGIN: Final = RungPlugin(spec=CorpusScaleRung, build=_corpus_scale)
FALLBACK_PLUGIN: Final = RungPlugin(spec=FallbackRung, build=_fallback)
