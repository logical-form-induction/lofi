"""The active learning configuration kind, each of whose points is one run.

An active learning experiment sweeps an active learning document. Each point
runs its rounds in the launching process, where a collector may await the
responses to a fielded batch, trains no step a step timer prices, and reports
the rows its rounds recorded.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, final

from lofi._frozen import FrozenValue
from lofi.run._boundary.active import (
    ACTIVE_LEARNING_MANIFEST_NAME,
    ACTIVE_LEARNING_METRICS,
    ACTIVE_LEARNING_ROLE,
    decode_active_learning_manifest,
)
from lofi.run._boundary.manifest import ManifestStatus, decode_seed_derivation
from lofi.run._experiment import ManifestPoint, ResolvedConfiguration
from lofi.run.active import ActiveLearningExecutor, ActiveLearningInputs
from lofi.spec import RunContext, SpecLoader, load_plugins, run_context
from lofi.spec.acquisition import ActiveLearningSpec

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

__all__ = ["ActiveLearningConfiguration"]


@final
class ActiveLearningConfiguration(FrozenValue):
    """An active learning document, each of whose points is one run of it.

    Parameters
    ----------
    kind : str
        The tag the experiment declares the document under.
    path : Path
        The active learning document.
    """

    _noun = "active-learning configuration"
    __slots__ = ("_kind", "_path")

    def __init__(self, kind: str, path: Path) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_path", path)

    @property
    def kind(self) -> str:
        """Return the tag the experiment declares the active learning document under."""
        return self._kind

    @property
    def path(self) -> Path:
        """Return the active learning document."""
        return self._path

    @property
    def manifest_name(self) -> str:
        """Return the file a run's directory holds its manifest in."""
        return ACTIVE_LEARNING_MANIFEST_NAME

    @property
    def enumerates(self) -> bool:
        """Return that a run's rows are drawn under no enumeration budget."""
        return False

    @property
    def in_process(self) -> bool:
        """Return that a run answers its collector in the launching process."""
        return True

    @property
    def timed(self) -> bool:
        """Return that a run trains no step a step timer prices."""
        return False

    def composer(self, search_path: tuple[Path, ...], /) -> TracedComposer:
        """Return the composition over the active learning document."""
        load_plugins()
        return _ActiveComposer(
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
        """Return the round metrics every run reports, under its one role."""
        del search_path, overrides, seed
        return MappingProxyType(
            {ACTIVE_LEARNING_ROLE: frozenset(ACTIVE_LEARNING_METRICS)}
        )

    def roles(self, config: Mapping[str, JsonValue], /) -> frozenset[str]:
        """Return the role every run reports its rounds under."""
        del config
        return frozenset({ACTIVE_LEARNING_ROLE})

    def manifest_point(self, data: bytes, /) -> ManifestPoint:
        """Return the document, seed and status an active learning manifest records.

        Raises
        ------
        ValueError
            If the bytes are not an active learning manifest, or it records no
            experiment seed derivation.
        """
        manifest = decode_active_learning_manifest(data)
        derivation = manifest.seed_derivation
        if derivation is None:
            msg = "an experiment's active-learning manifest records no seed derivation"
            raise ValueError(msg)
        return ManifestPoint(
            ResolvedConfiguration(manifest.document.model_dump_json()),
            decode_seed_derivation(derivation),
            (
                ManifestStatus.COMPLETED
                if manifest.status == "completed"
                else ManifestStatus.KILLED
            ),
            manifest.git_revision,
            None,
            awaiting_responses=manifest.status == "awaiting-responses",
        )

    def executor(
        self, context: ExecutionContext, points: Sequence[SweepPoint], /
    ) -> ExecutePoint:
        """Return the executor of the runs over the registered inputs, each checked.

        Raises
        ------
        PointExecutionError
            If a run cannot be executed; nothing is executed.
        """
        learning = ActiveLearningExecutor(
            context.store,
            environment=context.environment,
            infrastructure_attempts=context.failure.infrastructure_attempts,
            inputs=ActiveLearningInputs(),
        )

        for point in points:
            learning.check(point)

        return learning

    def identities(self, points: Sequence[SweepPoint], /) -> Mapping[str, JsonValue]:
        """Return no identities, since a run of rounds trains no blueprint."""
        del points

        return MappingProxyType({})

    def result_rows(
        self,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the rows the runs' rounds recorded, as recorded."""
        del points, store

        return tuple(rows)


@final
class _ActiveComposer:
    """Compose one active learning document and retain its traced provenance."""

    __slots__ = ("_compositions", "_data_root", "_loader", "_path")

    def __init__(self, path: Path, loader: SpecLoader, *, data_root: Path) -> None:
        self._path = path
        self._loader = loader
        self._data_root = data_root
        self._compositions: dict[
            tuple[tuple[str, ...], int], Composition[ActiveLearningSpec]
        ] = {}

    def __call__(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, JsonValue]:
        """Return one resolved active learning document and retain its provenance."""
        with run_context(RunContext(self._data_root, seed)):
            composition = self._loader.compose_traced(
                self._path, schema=ActiveLearningSpec, overrides=overrides
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
    ) -> Composition[ActiveLearningSpec]:
        """Return the composition retained for the overrides."""
        try:
            return self._compositions[(overrides, seed)]

        except KeyError:
            msg = (
                f"no active-learning document was composed under overrides "
                f"{overrides!r} at seed {seed}"
            )
            raise LookupError(msg) from None
