"""Composition of run specifications from documents, profiles and overrides.

A selection names a configuration document, an optional profile, extra
overlay documents and ``key=value`` overrides, applied in that ascending
precedence through the process-entry codec, so nothing a caller selects is
dropped before validation.
"""

import json
from typing import TYPE_CHECKING, NamedTuple, cast

from lofi.run._boundary.sweep import overlay_settings
from lofi.spec import (
    RunContext,
    RunSpec,
    RunSpecCodec,
    SpecLoader,
    run_context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.spec import Composition, LeafOrigin, Provenance

__all__ = ["Selection", "compose", "composer", "resolved_document", "trace"]


class Selection(NamedTuple):
    """What one command line selects for composing a run specification."""

    config: Path
    profile: str | None
    overrides: tuple[str, ...]
    overlays: tuple[Path, ...]
    search_path: tuple[Path, ...]


def _data_root(selection: Selection, /) -> Path:
    """Return the project root whose configuration the selection composes."""
    if selection.search_path:
        return selection.search_path[0].parent

    return selection.config.parent


def resolved_document(spec: RunSpec, /) -> dict[str, JsonValue]:
    """Return the resolved document a validated run specification dumps to.

    Parameters
    ----------
    spec : RunSpec
        The validated run specification.
    """
    return cast("dict[str, JsonValue]", json.loads(spec.model_dump_json()))


def compose(selection: Selection, /) -> RunSpec:
    """Compose and validate the run specification a selection describes.

    Parameters
    ----------
    selection : Selection
        The document, profile, overlays and overrides to compose.
    """
    codec = RunSpecCodec(search_path=selection.search_path)

    return codec(
        selection.config,
        profile=selection.profile,
        overrides=selection.overrides,
        extra=overlay_settings(selection.overlays),
    )


def trace(selection: Selection, /) -> Composition[RunSpec]:
    """Compose the run specification a selection describes with its provenance.

    Parameters
    ----------
    selection : Selection
        The document, profile, overlays and overrides to compose.
    """
    return SpecLoader(search_path=selection.search_path).compose_traced(
        selection.config,
        schema=RunSpec,
        profile=selection.profile,
        overrides=selection.overrides,
        extra=overlay_settings(selection.overlays),
    )


class _PointComposer:
    """Compose one point once and retain its traced leaf provenance."""

    __slots__ = ("_compositions", "_seed_independent", "_selection")

    def __init__(self, selection: Selection) -> None:
        self._selection = selection
        self._compositions: dict[tuple[tuple[str, ...], int], Composition[RunSpec]] = {}
        self._seed_independent: dict[tuple[str, ...], Composition[RunSpec]] = {}

    def __call__(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, JsonValue]:
        """Return one resolved point and retain its provenance."""
        key = (overrides, seed)
        composition = self._compositions.get(key)

        if composition is not None:
            return resolved_document(composition.spec)

        composition = self._seed_independent.get(overrides)

        if composition is not None:
            self._compositions[key] = composition
            return resolved_document(composition.spec)

        selection = self._selection
        context = RunContext(_data_root(selection), seed)

        with run_context(context):
            composition = trace(
                Selection(
                    selection.config,
                    selection.profile,
                    (*selection.overrides, *overrides),
                    selection.overlays,
                    selection.search_path,
                )
            )
        self._compositions[key] = composition

        if not any(
            origin.expression is not None and "${lofi.seed:" in origin.expression
            for origin in composition.provenance.origins.values()
        ):
            self._seed_independent[overrides] = composition

        return resolved_document(composition.spec)

    def provenance(self, overrides: tuple[str, ...], seed: int, /) -> Provenance:
        """Return the provenance retained by the matching composition call."""
        return self._composition(overrides, seed).provenance

    def sources(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, tuple[LeafOrigin, ...]]:
        """Return every source of each path the matching composition call set."""
        return self._composition(overrides, seed).sources

    def _composition(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Composition[RunSpec]:
        """Return the composition retained for the overrides."""
        try:
            return self._compositions[(overrides, seed)]

        except KeyError:
            msg = f"no point was composed under overrides {overrides!r} at seed {seed}"
            raise LookupError(msg) from None


def composer(selection: Selection, /) -> _PointComposer:
    """Return the composition a sweep expands its points through.

    Parameters
    ----------
    selection : Selection
        The document, profile, overlays and overrides every point shares;
        a point's own overrides apply after the selection's.
    """
    return _PointComposer(selection)
