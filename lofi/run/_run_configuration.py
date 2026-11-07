"""The run configuration kind, whose points train and are scored held out.

A run experiment sweeps a run document. Each point trains through the
training operations it is given, may run in a scheduler task as well as in
the launching process, is priced off its own timed training step, and is
frozen with the grammar, codec, corpus and component identities its training
blueprint computes.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

from lofi._frozen import FrozenValue
from lofi.registry import RegistryError
from lofi.run._boundary.experiment import RunExperimentConfig
from lofi.run._boundary.manifest import decode_manifest, decode_seed_derivation
from lofi.run._boundary.sweep import resolve_point
from lofi.run._compose import Selection, composer
from lofi.run._executor import PointExecutor
from lofi.run._experiment import ManifestPoint, ResolvedConfiguration
from lofi.run._experiment_config import ExperimentConfigPlugin
from lofi.run._manifest import MANIFEST_NAME
from lofi.run._training import plan_training
from lofi.spec import EvaluationData, RunContext, RunSpecCodec, run_context

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
    from lofi.spec import EvaluatorSpec, RunSpec

__all__ = ["PLUGIN", "RunConfiguration"]


def _run_evaluator_metrics(
    run: RunSpec, declaration: EvaluatorSpec, /
) -> tuple[str, ...]:
    """Resolve evaluator rows against the label inventory the run's data owns."""
    if declaration.evaluation_data() is EvaluationData.INFERENCE_PAIRS:
        return declaration.metrics(run.data.inference_labels())

    return declaration.metrics()


@final
class RunConfiguration(FrozenValue):
    """A run document, whose points train and write experiment manifests.

    Parameters
    ----------
    kind : str
        The tag the experiment declares the document under.
    path : Path
        The run document.
    """

    _noun = "run configuration"

    __slots__ = ("_kind", "_path")

    def __init__(self, kind: str, path: Path) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_path", path)

    @property
    def kind(self) -> str:
        """Return the tag the experiment declares the run document under."""
        return self._kind

    @property
    def path(self) -> Path:
        """Return the run document."""
        return self._path

    @property
    def manifest_name(self) -> str:
        """Return the file a run directory holds its experiment manifest in."""
        return MANIFEST_NAME

    @property
    def enumerates(self) -> bool:
        """Return that a run's held-out rows are decoded under its grammar's budget."""
        return True

    @property
    def in_process(self) -> bool:
        """Return that a run's points also execute in scheduler tasks."""
        return False

    @property
    def timed(self) -> bool:
        """Return that a run's points train the step a step timer prices."""
        return True

    def composer(self, search_path: tuple[Path, ...], /) -> TracedComposer:
        """Return the run-specification composition over the run document.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against.
        """
        return composer(
            Selection(
                config=self._path,
                profile=None,
                overrides=(),
                overlays=(),
                search_path=search_path,
            )
        )

    def evaluators(
        self,
        search_path: tuple[Path, ...],
        overrides: tuple[str, ...],
        seed: int,
        /,
    ) -> Mapping[str, frozenset[str]]:
        """Return the metrics each evaluator the run document declares reports.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against.
        overrides : tuple[str, ...]
            The overrides of the point whose evaluators are read.
        seed : int
            The run seed used while composing that point.
        """
        with run_context(RunContext(search_path[0].parent, seed)):
            run = RunSpecCodec(search_path=search_path)(self._path, overrides=overrides)

        return MappingProxyType(
            {
                role: frozenset(_run_evaluator_metrics(run, declaration))
                for role, declaration in run.evaluators.items()
            }
        )

    def roles(self, config: Mapping[str, JsonValue], /) -> frozenset[str]:
        """Return the roles a resolved run configuration declares evaluators under.

        Parameters
        ----------
        config : Mapping[str, JsonValue]
            The point's resolved run configuration.
        """
        evaluators = config.get("evaluators")

        return frozenset(evaluators) if isinstance(evaluators, dict) else frozenset()

    def manifest_point(self, data: bytes, /) -> ManifestPoint:
        """Return the run, seed, status and source an experiment manifest records.

        Parameters
        ----------
        data : bytes
            The manifest's bytes.
        """
        manifest = decode_manifest(data)

        return ManifestPoint(
            ResolvedConfiguration(manifest.run_spec.model_dump_json()),
            decode_seed_derivation(manifest.seed_derivation),
            manifest.status,
            manifest.git_revision,
            manifest.signature_fingerprint,
        )

    def executor(
        self, context: ExecutionContext, points: Sequence[SweepPoint], /
    ) -> ExecutePoint:
        """Return the store-bound executor that trains and scores each run point.

        Parameters
        ----------
        context : ExecutionContext
            The store, environment, failure policy and training operations the
            points execute against.
        points : Sequence[SweepPoint]
            The points the executor will be given; a run point is validated
            when it executes, so none is refused here.
        """
        del points

        return PointExecutor(
            context.store.runs,
            codec=RunSpecCodec(),
            operations=context.training,
            infrastructure_attempts=context.failure.infrastructure_attempts,
            failure_policy=context.failure,
            environment=context.environment,
        )

    def identities(self, points: Sequence[SweepPoint], /) -> Mapping[str, JsonValue]:
        """Return the identities training records for each point, by identifier.

        Every point is planned as training plans it, and its grammar, codec
        and corpus identities and its component fingerprints by role are the
        ones the point's manifest records when it trains.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The experiment's expanded points.

        Raises
        ------
        ValueError
            If a point cannot be planned for training.
        """
        codec = RunSpecCodec()
        identities: dict[str, JsonValue] = {}

        for point in points:
            try:
                blueprint = plan_training(resolve_point(point, codec))

            except (RegistryError, TypeError, ValueError) as refused:
                msg = (
                    f"the point {point.point_id} cannot be planned for training: "
                    f"{refused}"
                )
                raise ValueError(msg) from refused

            components: dict[str, JsonValue] = dict(blueprint.component_fingerprints)
            identities[point.point_id] = {
                "grammar_fingerprint": blueprint.grammar_fingerprint,
                "codec_fingerprint": blueprint.codec_fingerprint,
                "corpus_revision": blueprint.corpus_revision,
                "component_fingerprints": components,
            }

        return MappingProxyType(identities)

    def result_rows(
        self,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the held-out rows the run points recorded, as recorded.

        Parameters
        ----------
        rows : Sequence[Mapping[str, JsonValue]]
            The rows the completed points recorded, in expansion order.
        points : Sequence[SweepPoint]
            The experiment's expanded points, which add no row.
        store : ExperimentStore
            The store the points execute under, which adds no row.
        """
        del points, store

        return tuple(rows)


def _run_configuration(
    declaration: RunExperimentConfig, path: Path, /
) -> ExperimentConfiguration:
    """Build the installed run configuration."""
    return RunConfiguration(declaration.kind, path)


PLUGIN: Final = ExperimentConfigPlugin(
    spec=RunExperimentConfig, build=_run_configuration
)
"""The installed run configuration kind."""
