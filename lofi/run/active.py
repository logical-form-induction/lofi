"""Active learning runs driven from their declared documents.

An active learning document is checked once at its run boundary: its
judgment fit document is composed and must declare MAP inference and an ordinal
response model reading exactly the predictor columns the trained point fills;
it may declare no held-out comparison, dynamic range gate or table ablation,
which an active learning run would not perform. The candidate pool is built
there too, and every attribute expression the document declares, its
acquisition's allocation and uniqueness constraints and its start's
stratification, and the expression of every factor it declares, is checked
against every pool item,
so one naming an attribute the items do not carry is refused before anything
runs. A declared response encoding
is read there once and must have the response's levels; every bead reader of
the run, its start and its collector alike, reads through it. The run's
performance is
measured by the cross-validation the active learning document declares, under
the splitter installed under its tag. A checked run then restores the trained
point and publishes its readouts over the element of every pool item the
document maps to the point's input role,
and obtains the judgments its first round starts from: bead judgments, read
through the bead judgment source under the declared encoding, or the
collector's answers to a seeded draw from every declared stratum, collected as
round 0. It then runs one round
after another: each refits the response model by MAP to the judgments so far,
scores the unjudged candidates, selects a batch by the declared acquisition
rule, and collects the batch through the declared collector, until the
stopping rule holds, the pool cannot fill another batch, or the declared number
of rounds is reached.

Every stream of a run derives from its run seed: the MAP initializations, the
initial draw, the cross-validation split, the agreement bootstrap and the
collector's own draws.
A run outside an experiment takes the seed it is given. Inside an experiment
each sweep point is one run under the seed the experiment derives for it, and
its rounds are its report rows: agreement, cross-validated rank correlation,
the batch's variance in predicted response, and each stratum's share of the
batch with the performance it was allocated by and whether that performance
was floored at zero.
"""

import json
from hashlib import file_digest
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, final

from lofi.eval import NO_ENUMERATION_BUDGET, UNFROZEN, Approximation, config_digest
from lofi.judgment import (
    ActiveLearningResult,
    AttributeExpression,
    AwaitedBatch,
    BoundedResponseSpec,
    CollectorContext,
    CoverageGate,
    ItemFactors,
    JudgmentFitSpec,
    MapInference,
    MarginalLikelihoodError,
    PointFittingError,
    ResponseCompiler,
    SplitterFactory,
    acquisition_rule_for,
    active_collector_for,
    collect_initial,
    item_key,
    lexicon_generator_for,
    refuse_uncarried,
    run_active_learning,
)
from lofi.registry import Registry, RegistryError
from lofi.run._active_fit import ActiveItemColumns, MapActiveFitter
from lofi.run._active_inputs import (
    Pool,
    pool_readouts,
    read_bead_judgments,
    read_encoding,
    read_pool,
)
from lofi.run._boundary.active import (
    ACTIVE_LEARNING_MANIFEST_NAME,
    ACTIVE_LEARNING_ROLE,
    ActiveLearningManifest,
    compose_active_learning,
    decode_active_learning_document,
    encode_active_learning_manifest,
)
from lofi.run._boundary.manifest import (
    ManifestStatus,
    RecordedRows,
    encode_seed_derivation,
)
from lofi.run._environment import execution_environment
from lofi.run._executor import (
    PointExecutionError,
    PreparationError,
    execute_to_outcome,
)
from lofi.run._manifest import (
    replace_file,
    write_rows,
)
from lofi.run._store import RunOutcome
from lofi.spec import (
    ConfigDocumentError,
    ConfigError,
    SpecLoader,
    config_search_path,
    derive_seed,
    load_plugins,
)
from lofi.spec.acquisition import BeadInitialSettings, CollectedInitialSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from bead.items.item import Item

    from lofi._types import JsonValue
    from lofi.judgment import (
        AcquisitionRule,
        BeadResponseEncoding,
        LexiconGenerator,
        ResponseProgram,
    )
    from lofi.run._active_inputs import BeadJudgments
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._environment import ExecutionEnvironment
    from lofi.run._seed import PointSeed
    from lofi.run._store import ExperimentStore
    from lofi.spec.acquisition import (
        ActiveLearningSettings,
        ActiveLearningSpec,
        PointReadoutSettings,
        TokenizerSettings,
    )
    from lofi.spec.lexicon_generators import LexiconGeneratorSpec

__all__ = [
    "ACTIVE_LEARNING_COMMAND",
    "BOOTSTRAP_STREAM",
    "COLLECTOR_STREAM",
    "CROSS_VALIDATION_STREAM",
    "INITIAL_STREAM",
    "MAP_STREAM",
    "ROWS_NAME",
    "ActiveLearning",
    "ActiveLearningCommand",
    "ActiveLearningExecutor",
    "ActiveLearningInputs",
    "ActiveLearningOutcome",
    "PointPredictors",
    "active_learn",
    "round_rows",
]

MAP_STREAM: Final = "active-map"
"""The stream every MAP initialization of a run derives from."""

BOOTSTRAP_STREAM: Final = "agreement-bootstrap"
"""The stream the agreement interval of every round derives from."""

COLLECTOR_STREAM: Final = "collector"
"""The stream a collector's own draws derive from."""

CROSS_VALIDATION_STREAM: Final = "cross-validation"
"""The stream the cross-validation split of every round derives from."""

INITIAL_STREAM: Final = "initial-draw"
"""The stream a collected start's per-stratum draw derives from."""

_SPLITTER_FAMILY: Final = "splitter"
"""The registry family the declared cross-validation's splitter is installed in."""

ROWS_NAME: Final = "rows.json"
"""The file a run outside an experiment records its round rows in."""


class PointPredictors(Protocol):
    """Read the predictor columns a trained point fills for every pool item."""

    def __call__(
        self,
        run: Path,
        items: Sequence[Item],
        inputs: Mapping[str, str],
        tokenizer: TokenizerSettings,
        readouts: Mapping[str, PointReadoutSettings],
        /,
    ) -> Mapping[str, Mapping[str, float]]:
        """Return each item's declared predictor readouts, by item identity."""
        ...


@final
class ActiveLearningInputs:
    """The operations a run builds its pool and its predictors through.

    Parameters
    ----------
    generator_for : Callable[[LexiconGeneratorSpec], LexiconGenerator]
        Builds each lexicon generator an item-construction document declares.
    predictors : PointPredictors
        Reads the predictors a trained point's run directory fills; by default
        the point is restored and its forward pass published over the pool.
    """

    __slots__ = ("_generator_for", "_predictors")

    def __init__(
        self,
        *,
        generator_for: Callable[
            [LexiconGeneratorSpec], LexiconGenerator
        ] = lexicon_generator_for,
        predictors: PointPredictors = pool_readouts,
    ) -> None:
        self._generator_for = generator_for
        self._predictors = predictors

    def pool(self, path: Path, /) -> Pool:
        """Construct the candidate pool an item-construction document declares."""
        return read_pool(path, generator_for=self._generator_for)

    def predictors(
        self,
        run: Path,
        items: Sequence[Item],
        inputs: Mapping[str, str],
        tokenizer: TokenizerSettings,
        readouts: Mapping[str, PointReadoutSettings],
        /,
    ) -> Mapping[str, Mapping[str, float]]:
        """Return the predictor columns the trained point fills for the pool."""
        return self._predictors(run, items, inputs, tokenizer, readouts)


class ActiveLearningOutcome(NamedTuple):
    """What one run of an active learning document produced and read."""

    result: ActiveLearningResult
    initial_judgments: int
    initial_skipped: int
    source_digests: Mapping[str, str]


def _digest(path: Path, /) -> str:
    """Return a file's SHA-256 content identity."""
    with path.open("rb") as stream:
        return f"sha256:{file_digest(stream, 'sha256').hexdigest()}"


def _file(root: Path, declared: str, role: str, /) -> Path:
    """Return a declared file under the document's directory."""
    path = root / declared
    if not path.is_file():
        msg = f"the {role} {declared!r} is not a file under {root}"
        raise ValueError(msg)
    return path


@final
class ActiveLearning:
    """One checked active learning document, ready to run.

    Parameters
    ----------
    document : ActiveLearningSpec
        The validated document.
    root : Path
        The directory every declared path is relative to.
    inputs : ActiveLearningInputs
        The operations the pool and predictors are built through.

    Raises
    ------
    ValueError
        If a declared file is absent, the judgment fit is not one an
        active learning run can refit, the pool cannot be constructed, or a
        declared expression or factor names an attribute a pool item does not
        carry.
    UnknownTagError
        If no splitter or acquisition rule is installed under its tag.
    """

    __slots__ = (
        "_dataset",
        "_document",
        "_encoding",
        "_factors",
        "_gate",
        "_inputs",
        "_levels",
        "_paths",
        "_plan",
        "_pool",
        "_program",
        "_root",
        "_rule",
        "_settings",
        "_splitter",
        "_unseen",
    )

    def __init__(
        self,
        document: ActiveLearningSpec,
        root: Path,
        /,
        *,
        inputs: ActiveLearningInputs,
    ) -> None:
        settings = document.settings()
        paths = {
            "pool": _file(root, settings.pool, "item-construction document"),
            "fit": _file(root, settings.fit, "judgment-fit document"),
        }
        if isinstance(settings.initial, BeadInitialSettings):
            paths["responses"] = _file(
                root, settings.initial.responses, "initial judgments"
            )
        if settings.encoding is not None:
            paths["encoding"] = _file(root, settings.encoding, "response encoding")
        run = root / settings.predictors.run
        if not run.is_dir():
            msg = f"the trained point {settings.predictors.run!r} is not a directory"
            raise ValueError(msg)
        load_plugins()
        fit = (
            SpecLoader(search_path=config_search_path(root))
            .compose_traced(paths["fit"], schema=JudgmentFitSpec)
            .spec
        )
        inference = _check_fit(fit, settings)
        validation = settings.cross_validation
        unseen = validation.unseen_participant

        self._document = document
        self._settings = settings
        self._paths = paths
        self._inputs = inputs
        self._program: ResponseProgram = ResponseCompiler.installed().compile(
            fit.response
        )
        response = fit.response
        if isinstance(response, BoundedResponseSpec):
            msg = "an active-learning run scores ordinal responses, not bounded ones"
            raise TypeError(msg)
        self._plan = inference
        self._levels = response.levels
        self._encoding = (
            None
            if settings.encoding is None
            else _read_encoding(paths["encoding"], response.levels)
        )
        self._splitter = Registry[SplitterFactory](_SPLITTER_FAMILY).get(
            validation.splitter
        )
        self._gate = CoverageGate(validation.coverage_threshold, policy=unseen)
        self._unseen = unseen
        self._dataset = fit.dataset
        self._root = root
        self._rule = acquisition_rule_for(settings.acquisition)
        pool = inputs.pool(paths["pool"])
        for source in pool.sources[1:]:
            paths[f"pool:{source.name}"] = source
        self._pool = _checked_pool(pool.items, settings, self._rule)
        self._factors = ItemFactors(settings.factors, self._pool)

    @property
    def document(self) -> ActiveLearningSpec:
        """Return the validated document."""
        return self._document

    @property
    def settings(self) -> ActiveLearningSettings:
        """Return the document's resolved settings."""
        return self._settings

    @property
    def dataset(self) -> str:
        """Return the identifier the fit document gives the collected judgments."""
        return self._dataset

    def run(self, seed: int, directory: Path, /) -> ActiveLearningOutcome:
        """Run every round the inputs and the collector allow.

        Parameters
        ----------
        seed : int
            The run seed every stream derives from.
        directory : Path
            The directory the run keeps its rounds under.
        """
        settings = self._settings
        pool = self._pool
        predictors = settings.predictors
        run = self._root / predictors.run
        readouts = self._inputs.predictors(
            run, pool, predictors.inputs, predictors.tokenizer, predictors.readouts
        )
        factors = self._factors
        columns = {
            item_key(item): ActiveItemColumns(
                self._rule.stratum(item),
                factors.levels(item_key(item)),
                readouts[item_key(item)],
            )
            for item in pool
        }
        validation = settings.cross_validation
        fitter = MapActiveFitter(
            self._program,
            self._plan,
            levels=self._levels,
            factors=factors.names,
            columns=tuple(predictors.readouts),
            items=columns,
            splitter=self._splitter(
                validation.held_out_unit,
                folds=validation.folds,
                seed=derive_seed(seed, CROSS_VALIDATION_STREAM),
            ),
            gate=self._gate,
            unseen=self._unseen,
            seed=derive_seed(seed, MAP_STREAM),
        )
        directory.mkdir(parents=True, exist_ok=True)
        collector = active_collector_for(
            settings.collector,
            CollectorContext(
                directory,
                self._root,
                pool,
                {item: values.readouts for item, values in columns.items()},
                levels=self._levels,
                seed=derive_seed(seed, COLLECTOR_STREAM),
                encoding=self._encoding,
            ),
        )
        digests = {
            path.relative_to(self._root).as_posix(): _digest(path)
            for path in (*self._paths.values(), *collector.sources)
        }
        trained = run / "manifest.json"
        if trained.is_file():
            digests[trained.relative_to(self._root).as_posix()] = _digest(trained)
        sources = dict(sorted(digests.items()))

        match settings.initial:
            case CollectedInitialSettings() as declared:
                collected = collect_initial(
                    declared,
                    pool,
                    collector,
                    seed=derive_seed(seed, INITIAL_STREAM),
                )
                if isinstance(collected, AwaitedBatch):
                    awaiting = ActiveLearningResult(
                        (), (), awaiting=collected, stop_reason=None
                    )
                    return ActiveLearningOutcome(awaiting, 0, 0, sources)
                initial, skipped = collected, 0
            case BeadInitialSettings():
                read = self._read_bead()
                initial, skipped = read.judgments, read.skipped

        result = run_active_learning(
            settings,
            pool,
            initial,
            fitter,
            collector,
            self._rule,
            bootstrap_seed=derive_seed(seed, BOOTSTRAP_STREAM),
        )
        return ActiveLearningOutcome(result, len(initial), skipped, sources)

    def _read_bead(self) -> BeadJudgments:
        """Return the bead judgments a bead start declares.

        Raises
        ------
        ValueError
            If the document declares no response encoding.
        """
        if self._encoding is None:
            msg = "a bead start reads responses, and the run has no response encoding"
            raise ValueError(msg)
        return read_bead_judgments(self._paths["responses"], self._encoding)


def _checked_pool(
    pool: tuple[Item, ...], settings: ActiveLearningSettings, rule: AcquisitionRule, /
) -> tuple[Item, ...]:
    """Return the pool once every declared rule expression reads it.

    Raises
    ------
    ValueError
        If an expression cannot be read from an item.
    """
    expressions = list(rule.expressions)
    if isinstance(settings.initial, CollectedInitialSettings):
        expressions.append(
            AttributeExpression(
                settings.initial.stratify_by, role="an initial stratification"
            )
        )
    refuse_uncarried(expressions, pool)
    return pool


def _read_encoding(path: Path, levels: int, /) -> BeadResponseEncoding:
    """Return the declared response encoding, which has the fit's levels.

    Raises
    ------
    ValueError
        If the encoding's levels are not the fit's response's.
    """
    encoding = read_encoding(path)
    if encoding.n_levels != levels:
        msg = (
            f"the response encoding {encoding.name!r} has {encoding.n_levels} "
            f"levels and the fit's response has {levels}"
        )
        raise ValueError(msg)
    return encoding


def _check_fit(
    fit: JudgmentFitSpec, settings: ActiveLearningSettings, /
) -> MapInference:
    """Return the MAP inference of a fit an active learning run can refit.

    Raises
    ------
    ValueError
        If the fit declares other inference, a section the run does not
        perform, or a response reading other columns than the point fills.
    """
    match fit.inference:
        case MapInference() as inference:
            pass
        case declared:
            msg = (
                "an active-learning run refits its response model by MAP, and the "
                f"fit declares {declared.method.value!r} inference"
            )
            raise ValueError(msg)
    unperformed = tuple(
        name
        for name, declared in (
            ("comparison", fit.comparison is not None),
            ("dynamic_range", fit.dynamic_range is not None),
            ("ablation", bool(fit.ablation)),
        )
        if declared
    )
    if unperformed:
        msg = f"an active-learning run performs no {', '.join(unperformed)} of its fit"
        raise ValueError(msg)
    read = frozenset(fit.response.readouts)
    filled = frozenset(settings.predictors.readouts)
    if read != filled:
        msg = (
            f"the fit's response reads {sorted(read)!r} and the trained point fills "
            f"{sorted(filled)!r}"
        )
        raise ValueError(msg)
    return inference


def round_rows(
    learning: ActiveLearning,
    result: ActiveLearningResult,
    /,
    *,
    identity: Mapping[str, JsonValue],
    seed: int,
) -> tuple[dict[str, JsonValue], ...]:
    """Return the report rows of every recorded round.

    Parameters
    ----------
    learning : ActiveLearning
        The checked run the rounds belong to.
    result : ActiveLearningResult
        The rounds recorded.
    identity : Mapping[str, JsonValue]
        The columns naming the run, such as its freeze, point and configuration.
    seed : int
        The run seed.
    """
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[dict[str, JsonValue]] = []

    for round_ in result.rounds:
        base: dict[str, JsonValue] = {
            **identity,
            "task": ACTIVE_LEARNING_ROLE,
            "round": round_.index,
            "seed": seed,
            "dataset": learning.dataset,
            "stopped": round_.stopped,
            "stop_reason": round_.stop_reason,
            **approximation,
        }
        size = len(round_.batch.item_ids)
        rows.append(
            {
                **base,
                "metric": "agreement",
                "value": round_.agreement,
                "lower": round_.agreement_interval[0],
                "upper": round_.agreement_interval[1],
                "n": round_.judged,
            }
        )
        rows.append(
            {
                **base,
                "metric": "rank-correlation",
                "value": round_.rank_correlation,
                "lower": round_.rank_interval[0],
                "upper": round_.rank_interval[1],
                "n": round_.judged,
            }
        )
        rows.append(
            {
                **base,
                "metric": "batch-variance",
                "value": round_.batch.variance,
                "n": size,
            }
        )
        batch = round_.batch
        rows.extend(
            {
                **base,
                "metric": "stratum-share",
                "stratum": stratum,
                "performance": batch.performance.get(stratum),
                "floored": stratum in batch.floored,
                "value": share,
                "n": size,
            }
            for stratum, share in sorted(batch.shares.items())
        )

    return tuple(rows)


def _manifest(
    learning: ActiveLearning,
    outcome: ActiveLearningOutcome,
    *,
    seed: int,
    derived: PointSeed | None,
    environment: ExecutionEnvironment,
) -> ActiveLearningManifest:
    """Return the manifest one run records."""
    awaiting = outcome.result.awaiting
    return ActiveLearningManifest(
        document=learning.document,
        status="completed" if awaiting is None else "awaiting-responses",
        rounds=len(outcome.result.rounds),
        awaiting=None if awaiting is None else awaiting.index,
        stop_reason=outcome.result.stop_reason,
        initial_judgments=outcome.initial_judgments,
        initial_skipped=outcome.initial_skipped,
        source_digests=dict(outcome.source_digests),
        seed=seed,
        seed_derivation=None if derived is None else encode_seed_derivation(derived),
        package_versions=dict(environment.packages),
        git_revision=environment.git_revision,
    )


@final
class ActiveLearningExecutor:
    """Execute the active learning points of one launch through an experiment store.

    A point awaiting the responses to a batch is recorded as awaiting them, and
    executing it again resumes it.

    Parameters
    ----------
    store : ExperimentStore
        The store each point's directory lies under.
    environment : ExecutionEnvironment
        The package versions and source revision every manifest records.
    infrastructure_attempts : int
        The attempts each point receives across launches.
    inputs : ActiveLearningInputs
        The operations every run builds its pool and predictors through.
    """

    __slots__ = (
        "_checked",
        "_environment",
        "_infrastructure_attempts",
        "_inputs",
        "_store",
    )

    def __init__(
        self,
        store: ExperimentStore,
        *,
        environment: ExecutionEnvironment,
        infrastructure_attempts: int,
        inputs: ActiveLearningInputs,
    ) -> None:
        self._store = store
        self._environment = environment
        self._infrastructure_attempts = infrastructure_attempts
        self._inputs = inputs
        self._checked: dict[str, ActiveLearning] = {}

    @property
    def store(self) -> ExperimentStore:
        """Return the store this executor writes through."""
        return self._store

    def check(self, point: SweepPoint, /) -> ActiveLearning:
        """Validate one point as an active learning run once per executor.

        Raises
        ------
        PointExecutionError
            If the point's document or the documents it names are refused.
        """
        checked = self._checked.get(point.point_id)
        if checked is not None:
            return checked
        try:
            learning = ActiveLearning(
                decode_active_learning_document(point.config),
                self._store.experiment.config.path.parent,
                inputs=self._inputs,
            )
        except (
            ConfigDocumentError,
            ConfigError,
            RegistryError,
            TypeError,
            ValueError,
        ) as refused:
            msg = (
                f"the active-learning run of point {point.point_id!r} is refused: "
                f"{refused}"
            )
            raise PointExecutionError(msg) from refused
        self._checked[point.point_id] = learning
        return learning

    def _run(
        self,
        point: SweepPoint,
        learning: ActiveLearning,
        directory: Path,
        derived: PointSeed,
        /,
    ) -> RunOutcome:
        """Run one point, then record its rows and manifest."""
        try:
            outcome = learning.run(derived.root, directory)
        except PointFittingError, MarginalLikelihoodError:
            raise
        except (RegistryError, ValueError) as refused:
            raise PreparationError(str(refused)) from refused
        rows = round_rows(
            learning,
            outcome.result,
            identity={
                "freeze_id": self._store.freeze_id,
                "config_hash": config_digest(point.config),
                "point_id": point.point_id,
            },
            seed=derived.root,
        )
        write_rows(directory, RecordedRows(self._store.runs.freeze_id, derived, rows))
        manifest = _manifest(
            learning,
            outcome,
            seed=derived.root,
            derived=derived,
            environment=self._environment,
        )
        replace_file(
            directory / ACTIVE_LEARNING_MANIFEST_NAME,
            encode_active_learning_manifest(manifest),
        )
        status = (
            ManifestStatus.COMPLETED
            if outcome.result.awaiting is None
            else ManifestStatus.KILLED
        )
        return RunOutcome(directory, status, cached=False)

    def __call__(self, point: SweepPoint, /) -> RunOutcome:
        """Run one point unless its completed result is recorded."""
        learning = self.check(point)
        directory = self._store.directory(point)
        derived = self._store.experiment.seed_policy.point(point.point_id, point.seed)
        self._store.claim(point)
        if self._store.is_completed(point):
            return RunOutcome(directory, ManifestStatus.COMPLETED, cached=True)
        return execute_to_outcome(
            lambda _retry: self._run(point, learning, directory, derived),
            directory,
            numerical=(PointFittingError, MarginalLikelihoodError),
            derivation=derived,
            infrastructure_attempts=self._infrastructure_attempts,
            failure_policy=self._store.experiment.failure,
            halve_batch=lambda: False,
        )


def active_learn(
    document: Path,
    seed: int,
    out: Path,
    environment: ExecutionEnvironment,
    /,
    *,
    inputs: ActiveLearningInputs,
) -> ActiveLearningManifest:
    """Run one active learning document outside any experiment.

    The rounds are kept under ``out``, which records the run's manifest and its
    round rows, reported against no freeze. Running the document again with
    the same seed and directory resumes a run awaiting responses.

    Raises
    ------
    ConfigError
        If the document is refused.
    ValueError
        If a document it names is refused, or a round cannot be run.
    """
    learning = ActiveLearning(
        compose_active_learning(document), document.parent, inputs=inputs
    )
    outcome = learning.run(seed, out)
    rows = round_rows(
        learning,
        outcome.result,
        identity={
            "freeze_id": UNFROZEN,
            "config_hash": config_digest(
                json.loads(learning.document.model_dump_json())
            ),
        },
        seed=seed,
    )
    manifest = _manifest(
        learning, outcome, seed=seed, derived=None, environment=environment
    )
    replace_file(
        out / ACTIVE_LEARNING_MANIFEST_NAME, encode_active_learning_manifest(manifest)
    )
    recorded = {"rows": [dict(row) for row in rows]}
    replace_file(
        out / ROWS_NAME,
        (json.dumps(recorded, allow_nan=False, indent=2) + "\n").encode(),
    )
    return manifest


@final
class ActiveLearningCommand:
    """Run one active learning document from the command line.

    Parameters
    ----------
    inputs : ActiveLearningInputs
        The operations every run builds its pool and predictors through.
    """

    __slots__ = ("_inputs",)

    def __init__(self, inputs: ActiveLearningInputs, /) -> None:
        self._inputs = inputs

    def __call__(
        self, document: Path, seed: int, out: Path, /
    ) -> ActiveLearningManifest:
        """Run the document under the seed, keeping its rounds under ``out``."""
        return active_learn(
            document, seed, out, execution_environment(), inputs=self._inputs
        )


ACTIVE_LEARNING_COMMAND: Final = ActiveLearningCommand(ActiveLearningInputs())
"""The command line run of a document through the installed generators."""
