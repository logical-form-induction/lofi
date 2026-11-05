"""The baseline-fit configuration kind, whose points fit baseline systems.

A baseline-fit experiment sweeps a baseline-fit document. Its points fit the
registered baseline system in the launching process, train no step a step
timer prices, and report the rows their fits persisted.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, final

from lofi._frozen import FrozenValue
from lofi.run._boundary.baselines import (
    BASELINE_FIT_MANIFEST_NAME,
    BASELINE_ROLE,
    BaselineFitCodec,
    BaselineFitSpec,
    decode_baseline_manifest,
)
from lofi.run._boundary.experiment import BaselineFitExperimentConfig
from lofi.run._boundary.manifest import ManifestStatus, decode_seed_derivation
from lofi.run._experiment import ManifestPoint, ResolvedConfiguration
from lofi.run._experiment_config import ExperimentConfigPlugin
from lofi.run.baselines import BaselineExecutor, RegisteredBaselines
from lofi.spec import RunContext, run_context

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._experiment import (
        ExecutionContext,
        ExperimentConfiguration,
        TracedComposer,
    )
    from lofi.run._serial import ExecutePoint
    from lofi.run._store import ExperimentStore
    from lofi.spec import Composition, LeafOrigin, Provenance

__all__ = ["PLUGIN", "BaselineFitConfiguration"]


@final
class BaselineFitConfiguration(FrozenValue):
    """A baseline-fit document whose points persist baseline report rows.

    Parameters
    ----------
    kind : str
        The tag the experiment declares the document under.
    path : Path
        The baseline-fit document.
    """

    _noun = "baseline-fit configuration"
    __slots__ = ("_kind", "_path")

    def __init__(self, kind: str, path: Path) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_path", path)

    @property
    def kind(self) -> str:
        """Return the tag the experiment declares the baseline-fit document under."""
        return self._kind

    @property
    def path(self) -> Path:
        """Return the baseline-fit document."""
        return self._path

    @property
    def manifest_name(self) -> str:
        """Return the file holding a completed baseline-fit manifest."""
        return BASELINE_FIT_MANIFEST_NAME

    @property
    def enumerates(self) -> bool:
        """Return that a baseline fit claims no grammar-enumeration budget."""
        return False

    @property
    def in_process(self) -> bool:
        """Return that a baseline fit runs in the launching process."""
        return True

    @property
    def timed(self) -> bool:
        """Return that a baseline fit trains no step a step timer prices."""
        return False

    def composer(self, search_path: tuple[Path, ...], /) -> TracedComposer:
        """Return composition over the baseline-fit document."""
        return _BaselineComposer(
            self._path,
            BaselineFitCodec(search_path=search_path),
            data_root=search_path[0].parent,
        )

    def evaluators(
        self,
        search_path: tuple[Path, ...],
        overrides: tuple[str, ...],
        seed: int,
        /,
    ) -> Mapping[str, frozenset[str]]:
        """Return the metrics the selected baseline system reports."""
        with run_context(RunContext(search_path[0].parent, seed)):
            fit = BaselineFitCodec(search_path=search_path)(
                self._path, overrides=overrides
            )
        return MappingProxyType({BASELINE_ROLE: frozenset(fit.system.metrics())})

    def roles(self, config: Mapping[str, JsonValue], /) -> frozenset[str]:
        """Return the baseline role of a resolved fit."""
        del config
        return frozenset({BASELINE_ROLE})

    def manifest_point(self, data: bytes, /) -> ManifestPoint:
        """Return the completed point recorded by a baseline-fit manifest."""
        manifest = decode_baseline_manifest(data)
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
        """Return the executor of the registered baseline systems, every fit checked.

        Raises
        ------
        PointExecutionError
            If a fit cannot be executed; nothing is executed.
        """
        fitted = BaselineExecutor(
            context.store,
            operation=RegisteredBaselines(),
            environment=context.environment,
            infrastructure_attempts=context.failure.infrastructure_attempts,
        )

        for point in points:
            fitted.check(point)

        return fitted

    def identities(self, points: Sequence[SweepPoint], /) -> Mapping[str, JsonValue]:
        """Return no identities, since a baseline fit trains no blueprint."""
        del points

        return MappingProxyType({})

    def result_rows(
        self,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the rows the baseline fits persisted, as persisted."""
        del points, store

        return tuple(rows)


@final
class _BaselineComposer:
    """Compose baseline fits and retain each point's traced sources."""

    __slots__ = ("_codec", "_compositions", "_data_root", "_path")

    def __init__(
        self,
        path: Path,
        codec: BaselineFitCodec,
        *,
        data_root: Path,
    ) -> None:
        self._path = path
        self._codec = codec
        self._data_root = data_root
        self._compositions: dict[
            tuple[tuple[str, ...], int], Composition[BaselineFitSpec]
        ] = {}

    def __call__(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, JsonValue]:
        """Return one resolved baseline fit and retain its provenance."""
        with run_context(RunContext(self._data_root, seed)):
            composition = self._codec.compose_traced(self._path, overrides=overrides)
        self._compositions[(overrides, seed)] = composition
        return cast(
            "dict[str, JsonValue]", json.loads(composition.spec.model_dump_json())
        )

    def provenance(self, overrides: tuple[str, ...], seed: int, /) -> Provenance:
        """Return the sources retained by the matching composition call."""
        return self._composition(overrides, seed).provenance

    def sources(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, tuple[LeafOrigin, ...]]:
        """Return every source of each resolved baseline-fit leaf."""
        return self._composition(overrides, seed).sources

    def _composition(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Composition[BaselineFitSpec]:
        """Return the composition retained for the overrides and seed."""
        try:
            return self._compositions[(overrides, seed)]
        except KeyError:
            msg = (
                f"no baseline fit was composed under overrides {overrides!r} "
                f"at seed {seed}"
            )
            raise LookupError(msg) from None


def _baseline_fit_configuration(
    declaration: BaselineFitExperimentConfig, path: Path, /
) -> ExperimentConfiguration:
    """Build the installed baseline-fit configuration."""
    return BaselineFitConfiguration(declaration.kind, path)


PLUGIN: Final = ExperimentConfigPlugin(
    spec=BaselineFitExperimentConfig,
    build=_baseline_fit_configuration,
)
"""The installed baseline-fit configuration kind."""
