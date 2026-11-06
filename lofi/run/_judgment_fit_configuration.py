"""The judgment fit configuration kind, whose points fit judgment models.

A judgment fit experiment sweeps a judgment fit document. Its points fit in
the launching process against the designs their datasets register, train no
step a step timer prices, and add the declared model comparisons over the
persisted fits to the rows their fits recorded. The judgment capability is
optional, and this module, which imports it, is loaded only when an
experiment declares this kind.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, final

from lofi._frozen import FrozenValue
from lofi.judgment import JudgmentFitSpec, ResponseCompiler, decode_fit_manifest
from lofi.run._boundary.manifest import ManifestStatus, decode_seed_derivation
from lofi.run._experiment import (
    FIT_METRICS,
    FIT_ROLE,
    ManifestPoint,
    ResolvedConfiguration,
)
from lofi.run._manifest import FIT_MANIFEST_NAME
from lofi.run.fits import (
    HELD_OUT_PREDICTIVE_METRIC,
    FitExecutor,
    RegisteredDesigns,
    declared_model_comparison_rows,
)
from lofi.spec import RunContext, SpecLoader, run_context

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._experiment import (
        ExecutionContext,
        TracedComposer,
    )
    from lofi.run._serial import ExecutePoint
    from lofi.run._store import ExperimentStore
    from lofi.spec import Composition, LeafOrigin, Provenance

__all__ = ["JudgmentFitConfiguration"]


@final
class JudgmentFitConfiguration(FrozenValue):
    """A judgment fit document, whose points write judgment fit manifests.

    Parameters
    ----------
    kind : str
        The tag the experiment declares the document under.
    path : Path
        The judgment fit document.
    """

    _noun = "judgment-fit configuration"

    __slots__ = ("_kind", "_path")

    def __init__(self, kind: str, path: Path) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_path", path)

    @property
    def kind(self) -> str:
        """Return the tag the experiment declares the fit document under."""
        return self._kind

    @property
    def path(self) -> Path:
        """Return the judgment fit document."""
        return self._path

    @property
    def manifest_name(self) -> str:
        """Return the file a fit's directory holds its judgment fit manifest in."""
        return FIT_MANIFEST_NAME

    @property
    def enumerates(self) -> bool:
        """Return that a fit's rows are drawn under no enumeration budget."""
        return False

    @property
    def in_process(self) -> bool:
        """Return that a fit reads its designs in the launching process."""
        return True

    @property
    def timed(self) -> bool:
        """Return that a fit trains no step a step timer prices."""
        return False

    def composer(self, search_path: tuple[Path, ...], /) -> TracedComposer:
        """Return the judgment fit composition over the fit document.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against.
        """
        return _FitComposer(
            self._path,
            SpecLoader(search_path=search_path),
            data_root=search_path[0].parent,
        )

    def evaluators(
        self,
        search_path: tuple[Path, ...],
        overrides: tuple[str, ...],
        seed: int,
        /,
    ) -> Mapping[str, frozenset[str]]:
        """Return the fit's own diagnostics, the one evaluator a fit declares.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against, which the
            fit's diagnostics do not depend on.
        overrides : tuple[str, ...]
            The overrides of the point, which the diagnostics do not depend on.
        seed : int
            The run seed, which the diagnostics do not depend on.
        """
        del search_path, overrides, seed

        return MappingProxyType({FIT_ROLE: frozenset(FIT_METRICS)})

    def roles(self, config: Mapping[str, JsonValue], /) -> frozenset[str]:
        """Return the fit's diagnostic role, which every fit reports under.

        Parameters
        ----------
        config : Mapping[str, JsonValue]
            The point's resolved fit, which the role does not depend on.
        """
        del config

        return frozenset({FIT_ROLE})

    def manifest_point(self, data: bytes, /) -> ManifestPoint:
        """Return the fit, seed and source a judgment fit manifest records.

        A judgment fit manifest is written only once its fit completes, so it
        records a completed point.

        Parameters
        ----------
        data : bytes
            The manifest's bytes.
        """
        manifest = decode_fit_manifest(data)

        return ManifestPoint(
            ResolvedConfiguration(manifest.fit.model_dump_json()),
            decode_seed_derivation(manifest.seed_derivation),
            ManifestStatus.COMPLETED,
            manifest.git_revision,
            None,
        )

    def executor(
        self, context: ExecutionContext, points: Sequence[SweepPoint], /
    ) -> ExecutePoint:
        """Return the fit executor over the registered designs, every point checked.

        Parameters
        ----------
        context : ExecutionContext
            The store, environment and failure policy the fits execute
            against.
        points : Sequence[SweepPoint]
            The fits the executor will be given.

        Raises
        ------
        PointExecutionError
            If a fit cannot be executed; nothing is executed.
        """
        fits = FitExecutor(
            context.store,
            designs=RegisteredDesigns(),
            compiler=ResponseCompiler.installed(),
            environment=context.environment,
            infrastructure_attempts=context.failure.infrastructure_attempts,
        )

        for point in points:
            fits.check(point)

        return fits

    def identities(self, points: Sequence[SweepPoint], /) -> Mapping[str, JsonValue]:
        """Return no identities, since a fit trains no blueprint.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The experiment's expanded fits.
        """
        del points

        return MappingProxyType({})

    def result_rows(
        self,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the fits' rows with the declared model comparisons in place.

        A fit's own held-out predictive density is reported through the model
        comparison it enters, so it is replaced by every complete comparison
        over the persisted fits.

        Parameters
        ----------
        rows : Sequence[Mapping[str, JsonValue]]
            The rows the completed fits recorded, in expansion order.
        points : Sequence[SweepPoint]
            The experiment's expanded fits.
        store : ExperimentStore
            The store the fits execute under.
        """
        return (
            *(row for row in rows if row.get("metric") != HELD_OUT_PREDICTIVE_METRIC),
            *declared_model_comparison_rows(points, store),
        )


@final
class _FitComposer:
    """Compose one judgment fit and retain its traced leaf provenance."""

    __slots__ = ("_compositions", "_data_root", "_loader", "_path")

    def __init__(self, path: Path, loader: SpecLoader, *, data_root: Path) -> None:
        self._path = path
        self._loader = loader
        self._data_root = data_root
        self._compositions: dict[
            tuple[tuple[str, ...], int], Composition[JudgmentFitSpec]
        ] = {}

    def __call__(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, JsonValue]:
        """Return one resolved fit and retain its provenance."""
        with run_context(RunContext(self._data_root, seed)):
            composition = self._loader.compose_traced(
                self._path, schema=JudgmentFitSpec, overrides=overrides
            )
        self._compositions[(overrides, seed)] = composition

        return cast(
            "dict[str, JsonValue]", json.loads(composition.spec.model_dump_json())
        )

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
    ) -> Composition[JudgmentFitSpec]:
        """Return the composition retained for the overrides."""
        try:
            return self._compositions[(overrides, seed)]

        except KeyError:
            msg = f"no fit was composed under overrides {overrides!r} at seed {seed}"
            raise LookupError(msg) from None
