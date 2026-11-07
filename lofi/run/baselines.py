"""Serial execution and persistence of external baseline fits."""

from typing import TYPE_CHECKING, Protocol, cast, final

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    BaselineBuildContext,
    BaselineMeasurement,
    BaselineResult,
    BaselineSystemOperation,
    config_digest,
)
from lofi.registry import plugin_for
from lofi.run._boundary.baselines import (
    BASELINE_FIT_MANIFEST_NAME,
    BASELINE_ROLE,
    BaselineFitManifest,
    BaselineFitSpec,
    decode_baseline_fit,
    encode_baseline_manifest,
)
from lofi.run._boundary.manifest import (
    ManifestStatus,
    RecordedRows,
    encode_seed_derivation,
)
from lofi.run._executor import PointExecutionError, execute_to_outcome
from lofi.run._manifest import (
    replace_file,
    write_rows,
)
from lofi.run._store import RunOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.registry import ComponentPlugin
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._environment import ExecutionEnvironment
    from lofi.run._store import ExperimentStore
    from lofi.spec.components.base import ConfiguredSpec

__all__ = [
    "BaselineExecutor",
    "BaselineFitOperation",
    "RegisteredBaselines",
]


class BaselineFitOperation(Protocol):
    """Fit and score one validated baseline declaration."""

    def __call__(
        self,
        fit: BaselineFitSpec,
        root: Path,
        seed: int,
        /,
    ) -> BaselineResult:
        """Return the fit's measurements and source content identities."""
        ...


@final
class RegisteredBaselines:
    """Construct and execute the baseline system registered by a fit."""

    __slots__ = ()

    def __call__(
        self,
        fit: BaselineFitSpec,
        root: Path,
        seed: int,
        /,
    ) -> BaselineResult:
        """Build and run one validated baseline fit through its exact plugin."""
        configured = cast("ConfiguredSpec[object]", fit.system)
        plugin = cast(
            "ComponentPlugin[object, BaselineBuildContext, BaselineSystemOperation]",
            plugin_for(configured),
        )
        operation = plugin.construct(configured, BaselineBuildContext(root, seed))
        return operation()


def _measurement_row(
    measurement: BaselineMeasurement,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
    convention_hash: str,
    conventions: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    """Return one complete report row after validating its scalar fields."""
    for label, value in (
        ("baseline family", measurement.family),
        ("baseline system", measurement.system),
        ("baseline metric", measurement.metric),
        ("baseline citation", measurement.cites),
        ("baseline reason", measurement.reason),
        ("baseline unit", measurement.unit),
    ):
        ensure_name(label, value)
    ensure_positive("a baseline sample count", measurement.n)
    if isinstance(measurement.value, float):
        ensure_finite("a baseline measurement", measurement.value)

    return {
        "freeze_id": freeze_id,
        "config_hash": config_digest(point.config),
        "point_id": point.point_id,
        "task": BASELINE_ROLE,
        "family": measurement.family,
        "system": measurement.system,
        "metric": measurement.metric,
        "arm": measurement.arm,
        "cites": measurement.cites,
        "reason": measurement.reason,
        "value": measurement.value,
        "n": measurement.n,
        "unit": measurement.unit,
        "seed": seed,
        "convention_hash": convention_hash,
        "conventions": dict(sorted(conventions.items())),
        **Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec(),
    }


@final
class BaselineExecutor:
    """Execute baseline fit points serially through an experiment store."""

    __slots__ = (
        "_checked",
        "_environment",
        "_failure_policy",
        "_infrastructure_attempts",
        "_operation",
        "_store",
    )

    def __init__(
        self,
        store: ExperimentStore,
        *,
        operation: BaselineFitOperation,
        environment: ExecutionEnvironment,
        infrastructure_attempts: int,
    ) -> None:
        self._store = store
        self._operation = operation
        self._environment = environment
        self._failure_policy = store.experiment.failure
        self._infrastructure_attempts = ensure_positive(
            "baseline infrastructure attempts", infrastructure_attempts
        )
        self._checked: dict[str, BaselineFitSpec] = {}

    @property
    def store(self) -> ExperimentStore:
        """Return the experiment store this executor writes through."""
        return self._store

    def check(self, point: SweepPoint, /) -> BaselineFitSpec:
        """Validate one point as a baseline fit once per executor."""
        checked = self._checked.get(point.point_id)
        if checked is not None:
            return checked
        try:
            fit = decode_baseline_fit(point.config)
        except ValueError as refused:
            msg = f"the baseline fit of point {point.point_id!r} is refused: {refused}"
            raise PointExecutionError(msg) from refused
        self._checked[point.point_id] = fit
        return fit

    def _fit(
        self,
        point: SweepPoint,
        fit: BaselineFitSpec,
        directory: Path,
        seed: int,
        /,
    ) -> RunOutcome:
        """Fit, validate and persist one baseline point."""
        result = self._operation(fit, self._store.experiment.config.path.parent, seed)
        measurements = result.measurements
        declared = fit.system.metrics()
        observed = tuple(measurement.metric for measurement in measurements)
        if observed != declared:
            msg = (
                f"baseline system {fit.system.kind!r} declared metrics {declared!r} "
                f"but emitted {observed!r}"
            )
            raise ValueError(msg)

        conventions: dict[str, JsonValue] = dict(fit.system.conventions())
        conventions["source_digests"] = dict(result.source_digests)
        convention_hash = config_digest(conventions)
        derived = self._store.experiment.seed_policy.point(point.point_id, point.seed)
        freeze_id = self._store.freeze_id
        rows = tuple(
            _measurement_row(
                measurement,
                point=point,
                freeze_id=freeze_id,
                seed=seed,
                convention_hash=convention_hash,
                conventions=conventions,
            )
            for measurement in measurements
        )
        write_rows(directory, RecordedRows(self._store.runs.freeze_id, derived, rows))
        manifest = BaselineFitManifest(
            fit=fit,
            source_digests=dict(result.source_digests),
            seed_derivation=encode_seed_derivation(derived),
            package_versions=dict(self._environment.packages),
            git_revision=self._environment.git_revision,
        )
        replace_file(
            directory / BASELINE_FIT_MANIFEST_NAME,
            encode_baseline_manifest(manifest),
        )
        return RunOutcome(directory, ManifestStatus.COMPLETED, cached=False)

    def __call__(self, point: SweepPoint, /) -> RunOutcome:
        """Fit one baseline point unless its completed result is cached."""
        fit = self.check(point)
        directory = self._store.directory(point)
        derived = self._store.experiment.seed_policy.point(point.point_id, point.seed)
        self._store.claim(point)
        if self._store.is_completed(point):
            return RunOutcome(directory, ManifestStatus.COMPLETED, cached=True)
        return execute_to_outcome(
            lambda _retry: self._fit(point, fit, directory, derived.root),
            directory,
            numerical=(),
            derivation=derived,
            infrastructure_attempts=self._infrastructure_attempts,
            failure_policy=self._failure_policy,
            halve_batch=lambda: False,
        )
