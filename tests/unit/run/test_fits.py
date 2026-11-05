import json
import math
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never
from uuid import UUID

import didactic.api as dx
import pyarrow.parquet as pq
import pytest
from bead.items.item import Item
from tests.unit.run.test__experiment import (
    a_judgment_experiment_document,
    write_experiment,
)

from lofi.eval import config_digest
from lofi.judgment import (
    LINEAR,
    POSITION,
    AttributedItem,
    BeadJudgmentSource,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    CalibrationCampaign,
    CalibrationSeeds,
    Choice,
    ClosedScale,
    ColumnInventory,
    ColumnLevel,
    ComparisonSpec,
    ConditionalLikelihood,
    CoverageGate,
    Cumulative,
    DesignCodec,
    DesignProvenance,
    DynamicRangeCriterion,
    FactorTerm,
    FitSchedule,
    FixedDiscrimination,
    GuideFit,
    InferenceMethod,
    ItemFactors,
    JointTypeTensor,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    MapFitter,
    MapInference,
    MarginalLikelihoodError,
    MarginalPointPredictor,
    MaskedLMGeneratorSpec,
    MaskedLMLexiconGenerator,
    MemorizedCells,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    NutsInference,
    PointFittingError,
    PosteriorFittingError,
    PredictedJudgmentSource,
    PriorBlock,
    QuiversGuideFitter,
    QuiversNutsFitter,
    RandomIntercept,
    ReadBoundedDesign,
    ReadCorpusDesign,
    ReadDesign,
    ReplicateRecord,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    TableAblationSpec,
    TableColumns,
    TableRows,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
    VariationalInference,
    construct_items,
    decode_fit_manifest,
    decode_item_construction,
    design_of,
    installed_linkings,
    linear_steps,
    splitter_for,
)
from lofi.judgment.compiler import PARTICIPANT
from lofi.lf import InferenceLabel
from lofi.registry import UnknownTagError
from lofi.run import (
    FIT_MANIFEST_NAME,
    ExecutionEnvironment,
    ExperimentSeed,
    ExperimentStore,
    FailureKind,
    ManifestStatus,
    PointExecutionError,
    ReseededRunError,
    RunStore,
    SweepPoint,
    fits,
    installed_packages,
    read_binding,
    read_experiment,
    read_failure,
    read_rows,
)
from lofi.run.fits import (
    POINT_ESTIMATE_NAME,
    POSTERIOR_NAME,
    FitExecutor,
    RegisteredDesigns,
    declared_model_comparison_rows,
)
from lofi.run.results import ROWS_NAME, write_results
from lofi.spec import HeldOutParticipantPolicy
from lofi.spec.judgment_sources import PredictedJudgmentSourceSettings
from lofi.spec.optimizers.adam import AdamOptimizerSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.judgment import (
        BoundProgram,
        InferenceText,
        JudgmentDesign,
        JudgmentFitManifest,
        NutsPlan,
        PointFit,
        PointObjective,
        PosteriorFit,
        ResponseProgram,
    )
    from lofi.run import Experiment
    from lofi.run.fits import JudgmentDesigns
    from lofi.spec.lexicon_generators import LexiconGeneratorSpec

LEVELS: Final = 7
SCORE: Final = "score"
DATASET: Final = "tiny-ratings"
FREEZE: Final = "freeze/1"
SEED: Final = ExperimentSeed(7)
ENVIRONMENT: Final = ExecutionEnvironment(installed_packages(), "0123abcd", "cpu")


PROVENANCE: Final = DesignProvenance(
    tag=DATASET,
    checkpoint=None,
    checkpoint_dtype=None,
    unigram_digest=None,
    corpus_digests={},
    seed=None,
    participants=None,
    items=None,
    levels=None,
    population_digest=None,
    producer_run=None,
    producer_point=None,
)


class TinyDesigns:
    def __init__(self) -> None:
        self.resolved: list[str] = []

    def read(self, dataset: str, /) -> ReadDesign:
        self.resolved.append(dataset)
        codec = DesignCodec(LEVELS, factors=(), readouts=(SCORE,))
        design = codec.encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                (participant + 2 * item) % LEVELS,
                RowColumns(factors={}, readouts={SCORE: -1.5 + item}),
            )
            for participant in range(2)
            for item in range(4)
        )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


def a_fit(*, calibration: JsonValue = None) -> dict[str, JsonValue]:
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    return {
        "dataset": DATASET,
        "response": {
            "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
        },
        "inference": json.loads(
            NutsInference(warmup=5, retained=8, chains=2).model_dump_json()
        ),
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": calibration,
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }


def a_map_fit(
    *, marginal: bool = False, marginal_group: str = PARTICIPANT
) -> dict[str, JsonValue]:
    """Return a small conditional or truncated-marginal MAP fit."""
    objective = (
        TruncatedMarginalLikelihood(
            group=marginal_group,
            radius=3.0,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-7,
            quadrature_points=5,
            quadrature_seed=19,
        )
        if marginal
        else ConditionalLikelihood()
    )
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=objective,
        predictor=(RandomIntercept(group=marginal_group),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    document = a_fit()
    document["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    document["inference"] = json.loads(
        MapInference(
            optimizer=AdamOptimizerSpec(learning_rate=0.01),
            steps=2,
            initial_scale=0.1,
        ).model_dump_json()
    )

    return document


def an_experiment(
    tmp_path: Path,
    *,
    root: int = 7,
    freeze_id: str = FREEZE,
    numerical: Literal["final", "reseed"] = "final",
) -> Experiment:
    document = a_judgment_experiment_document()
    document["seed_policy"] = {"root": root}
    document["freeze_id"] = freeze_id
    failure = document["failure"]

    if not isinstance(failure, dict):
        msg = "the experiment fixture has no failure-policy body"
        raise TypeError(msg)

    failure["numerical"] = numerical
    folder = tmp_path / f"repository-{root}-{freeze_id.replace('/', '-')}"

    return read_experiment(write_experiment(folder, document, name="fits"))


def an_executor(
    store: RunStore, experiment: Experiment, *, designs: JudgmentDesigns | None = None
) -> FitExecutor:
    return FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=TinyDesigns() if designs is None else designs,
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=3,
    )


class BeadRecord(NamedTuple):
    """One response row at bead's judgment-source boundary."""

    annotator_id: str
    item_id: str
    question_name: str
    response_label: str


class BeadEncoding(NamedTuple):
    """One response encoding at bead's judgment-source boundary."""

    name: str
    n_levels: int
    labels: tuple[str, ...]


class BeadDesigns:
    """Expose one bead source through the public fit design boundary."""

    def __init__(self) -> None:
        labels = ("unlikely", "uncertain", "likely")
        records = tuple(
            BeadRecord(
                f"participant-{participant}",
                f"bead-item-{item}",
                "inference",
                labels[(participant + item) % len(labels)],
            )
            for participant in range(2)
            for item in range(4)
        )
        items = tuple(
            AttributedItem(
                f"bead-item-{item}", {"condition": ("control", "target")[item % 2]}, {}
            )
            for item in range(4)
        )
        self._source = BeadJudgmentSource(
            records,
            BeadEncoding("inference", len(labels), labels),
            factors=ItemFactors({"condition": "item['condition']"}, items),
        )

    def read(self, dataset: str, /) -> ReadDesign:
        return ReadDesign(design_of(self._source), PROVENANCE.with_(tag=dataset))


class _FixtureFill:
    """Complete the fixture prompt with the two fixture verbs."""

    def __call__(self, prompt: str, /, *, top_k: int) -> tuple[dict[str, object], ...]:
        assert top_k == 2
        assert prompt == "Someone <mask> someone."
        return (
            {"token_str": "admired", "score": 0.8},
            {"token_str": "called", "score": 0.3},
        )


def _fixture_generator(
    declaration: LexiconGeneratorSpec, /
) -> MaskedLMLexiconGenerator:
    assert isinstance(declaration, MaskedLMGeneratorSpec)
    settings = declaration.settings()
    return MaskedLMLexiconGenerator(settings, _FixtureFill())


ITEMS: Final = Path(__file__).resolve().parents[2] / "data" / "judgment" / "items"
"""The bead resources a declared item construction reads."""


class ConstructedBeadDesigns:
    """Expose a declared bead item construction to the fit boundary."""

    def __init__(self) -> None:
        declaration = decode_item_construction((ITEMS / "items.json").read_bytes())
        items = construct_items(declaration, ITEMS, generator_for=_fixture_generator)
        labels = ("unlikely", "uncertain", "likely")
        records = tuple(
            BeadRecord(
                f"participant-{participant}",
                str(item.id),
                "inference",
                labels[(participant + item_index) % len(labels)],
            )
            for participant in range(2)
            for item_index, item in enumerate(items)
        )
        self.item_ids = frozenset(str(item.id) for item in items)
        self._source = BeadJudgmentSource(
            records,
            BeadEncoding("inference", len(labels), labels),
            factors=ItemFactors(
                {"structure": "item['structure']", "verb": "item['verb']"}, items
            ),
        )

    def read(self, dataset: str, /) -> ReadDesign:
        return ReadDesign(design_of(self._source), PROVENANCE.with_(tag=dataset))


class _PredictedPoint:
    @property
    def response_levels(self) -> int:
        return 3

    @property
    def relation_labels(self) -> tuple[InferenceLabel, ...]:
        return (
            InferenceLabel.ENTAILS,
            InferenceLabel.CONTRADICTS,
            InferenceLabel.NEUTRAL,
        )

    def predict_responses(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        rows = (
            (0.8, 0.1, 0.1),
            (0.1, 0.8, 0.1),
            (0.1, 0.1, 0.8),
            (0.7, 0.2, 0.1),
        )
        return rows[: len(pairs)]

    def predict_relations(
        self, pairs: Sequence[InferenceText], /
    ) -> tuple[tuple[float, ...], ...]:
        return self.predict_responses(pairs)


class PredictedDesigns:
    """Expose a producer point's predictions to the fit boundary."""

    def __init__(self) -> None:
        items = tuple(
            Item(
                id=UUID(int=position + 1),
                item_template_id=UUID(int=0),
                rendered_elements={
                    "context": f"person-{position} admires someone",
                    "claim": f"person-{position} likes someone",
                },
            )
            for position in range(4)
        )
        self._source = PredictedJudgmentSource(
            PredictedJudgmentSourceSettings(
                "runs/synthetic",
                "point-17",
                "items.json",
                {"premise": "context", "hypothesis": "claim"},
                {},
                "inference",
                ("entails", "contradicts", "neutral"),
                "lofi-point-17",
            ),
            items,
            _PredictedPoint(),
        )

    def read(self, dataset: str, /) -> ReadDesign:
        return ReadDesign(design_of(self._source), self._source.provenance(dataset))


def a_bead_fit(*, factor: str = "condition") -> dict[str, JsonValue]:
    """Return a small fit over the bead source's three-level design."""
    response = ResponseSpec(
        family=Cumulative(),
        levels=3,
        objective=ConditionalLikelihood(),
        predictor=(FactorTerm(name=factor),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    document = a_fit()
    document["dataset"] = "bead-inference"
    document["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    return document


def a_predicted_fit() -> dict[str, JsonValue]:
    """Return a small threshold-only fit over producer predictions."""
    response = ResponseSpec(
        family=Cumulative(),
        levels=3,
        objective=ConditionalLikelihood(),
        predictor=(RandomIntercept(group=PARTICIPANT),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    document = a_fit()
    document["dataset"] = "predicted-inference"
    document["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    return document


def test_a_bead_source_fits_from_a_document_with_item_ids_preserved(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_bead_fit(), seed=0)

    outcome = an_executor(store, experiment, designs=BeadDesigns())(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    predictions = tuple(
        row for row in recorded.rows if row["metric"] == "response-prediction"
    )
    assert {row["item"] for row in predictions} == {
        "bead-item-0",
        "bead-item-1",
        "bead-item-2",
        "bead-item-3",
    }


def test_a_declared_judgment_item_set_trains_through_the_fit_document(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_bead_fit(factor="structure"), seed=0)
    designs = ConstructedBeadDesigns()

    outcome = an_executor(store, experiment, designs=designs)(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    predictions = tuple(
        row for row in recorded.rows if row["metric"] == "response-prediction"
    )
    assert {row["item"] for row in predictions} == designs.item_ids


def test_a_predicted_design_records_its_producer_in_the_fit_manifest(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_predicted_fit(), seed=0)

    outcome = an_executor(store, experiment, designs=PredictedDesigns())(point)
    manifest = fit_manifest(store, point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert manifest.design.producer_run == "runs/synthetic"
    assert manifest.design.producer_point == "point-17"


def fit_manifest(store: RunStore, point: SweepPoint) -> JudgmentFitManifest:
    return decode_fit_manifest(
        (store.directory(point) / FIT_MANIFEST_NAME).read_bytes()
    )


def test_a_fit_point_records_its_diagnostic_rows_and_its_manifest(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=1)
    derived = experiment.seed_policy.point(point.point_id, point.seed)

    outcome = an_executor(store, experiment)(point)
    manifest = fit_manifest(store, point)
    posterior = store.directory(point) / POSTERIOR_NAME
    recorded = read_rows(store.directory(point))
    binding = read_binding(store.directory(point))

    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False)
    assert manifest.seed_derivation.experiment_root == derived.experiment.root
    assert manifest.seed_derivation.point_id == derived.point_id
    assert manifest.seed_derivation.seed == derived.seed
    assert manifest.seed_derivation.root == derived.root
    assert manifest.git_revision == "0123abcd"
    assert manifest.design == PROVENANCE
    assert manifest.artifact.name == POSTERIOR_NAME
    assert manifest.artifact.sha256 == sha256(posterior.read_bytes()).hexdigest()
    assert json.loads(manifest.fit.model_dump_json()) == point.config
    assert binding is not None
    assert binding.freeze_id == experiment.freeze_id
    assert recorded is not None
    assert recorded.freeze_id == experiment.freeze_id
    assert recorded.seed_derivation == derived
    diagnostics = [
        row
        for row in recorded.rows
        if row["metric"] in {"divergences", "max-rhat", "min-ess"}
    ]
    assert [row["metric"] for row in diagnostics] == [
        "divergences",
        "max-rhat",
        "min-ess",
    ]
    assert sum(row["metric"] == "response-prediction" for row in recorded.rows) == 8
    assert sum(row["metric"] == "split-half-ceiling" for row in recorded.rows) == 1

    for row in recorded.rows:
        assert (row["budget_kind"], row["budget_depth"], row["budget_beam"]) == (
            "none",
            None,
            None,
        )
        assert row["freeze_id"] == experiment.freeze_id
        assert row["config_hash"] == config_digest(point.config)
        assert (row["point_id"], row["seed"]) == (point.point_id, derived.root)

    assert {row["n"] for row in diagnostics} == {16}


@pytest.mark.parametrize("inference", ["conditional", "marginal"])
def test_a_map_fit_records_its_point_artifact_and_objective(
    tmp_path: Path,
    inference: Literal["conditional", "marginal"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objectives: list[str] = []
    fitted = MapFitter.fit

    def recorded(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        objectives.append(type(objective).__name__)
        return fitted(fitter, objective)

    monkeypatch.setattr(MapFitter, "fit", recorded)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_map_fit(marginal=inference == "marginal"), seed=0)
    outcome = an_executor(store, an_experiment(tmp_path))(point)
    directory = store.directory(point)
    artifact = directory / POINT_ESTIMATE_NAME
    manifest = fit_manifest(store, point)
    rows = read_rows(directory)

    assert outcome.status is ManifestStatus.COMPLETED
    assert objectives == [
        "MarginalPointObjective"
        if inference == "marginal"
        else "ConditionalPointObjective"
    ]
    assert artifact.is_file()
    assert not (directory / POSTERIOR_NAME).exists()
    assert manifest.artifact.name == POINT_ESTIMATE_NAME
    assert manifest.artifact.sha256 == sha256(artifact.read_bytes()).hexdigest()
    assert rows is not None
    [diagnostic] = [row for row in rows.rows if row["metric"] == "map-log-posterior"]
    assert diagnostic["n"] == 2
    assert diagnostic["objective"] == (
        "truncated_marginal" if inference == "marginal" else "conditional"
    )


def test_a_map_fit_reseeds_a_typed_numerical_retry_and_then_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds: list[int] = []
    fitted = MapFitter.fit

    def once(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        seeds.append(fitter.seed)

        if len(seeds) == 1:
            msg = "MAP fitting produced a nonfinite gradient at step 1"
            raise PointFittingError(msg)

        return fitted(fitter, objective)

    monkeypatch.setattr(MapFitter, "fit", once)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    experiment = an_experiment(tmp_path, numerical="reseed")
    point = SweepPoint(a_map_fit(), seed=0)
    executor = an_executor(store, experiment)

    first = executor(point)
    again = executor(point)

    assert (first.status, first.cached) == (ManifestStatus.COMPLETED, False)
    assert (again.status, again.cached) == (ManifestStatus.COMPLETED, True)
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_a_marginal_likelihood_error_reseeds_map_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds: list[int] = []
    fitted = MapFitter.fit

    def once(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        seeds.append(fitter.seed)

        if len(seeds) == 1:
            msg = "the marginal integration failed numerically"
            raise MarginalLikelihoodError(msg)

        return fitted(fitter, objective)

    monkeypatch.setattr(MapFitter, "fit", once)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    experiment = an_experiment(tmp_path, numerical="reseed")
    point = SweepPoint(a_map_fit(marginal=True), seed=0)
    outcome = an_executor(store, experiment)(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_a_map_objective_is_constructed_before_the_point_directory_is_claimed(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_map_fit(marginal=True, marginal_group="item"), seed=0)

    with pytest.raises(ValueError, match="observed participant index"):
        an_executor(store, an_experiment(tmp_path))(point)

    assert not store.directory(point).exists()


def test_a_recorded_fit_is_served_without_fitting_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)
    an_executor(store, experiment)(point)

    def unfitted(fitter: QuiversNutsFitter, program: BoundProgram, /) -> Never:
        del fitter, program
        msg = "a recorded fit was drawn again"
        raise AssertionError(msg)

    monkeypatch.setattr(QuiversNutsFitter, "fit", unfitted)
    outcome = an_executor(store, experiment)(point)

    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, True)


def test_a_fit_completed_under_another_freeze_is_fitted_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)
    an_executor(store, an_experiment(tmp_path))(point)
    fit = QuiversNutsFitter.fit
    drawn: list[int] = []

    def counted(fitter: QuiversNutsFitter, program: BoundProgram, /) -> PosteriorFit:
        drawn.append(1)

        return fit(fitter, program)

    monkeypatch.setattr(QuiversNutsFitter, "fit", counted)
    later = an_experiment(tmp_path, freeze_id="freeze/2")
    outcome = an_executor(store, later)(point)
    refitted = RunStore(store.root, freeze_id="freeze/2", seed=SEED).directory(point)
    recorded = read_rows(refitted)
    kept = read_rows(store.directory(point))

    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False)
    assert outcome.directory == refitted != store.directory(point)
    assert drawn == [1]
    assert recorded is not None
    assert recorded.freeze_id == "freeze/2"
    assert kept is not None
    assert kept.freeze_id == FREEZE
    assert (store.directory(point) / FIT_MANIFEST_NAME).is_file()


def test_the_nuts_seed_is_derived_from_the_point_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans: list[NutsPlan] = []
    fit = QuiversNutsFitter.fit

    def recorded(fitter: QuiversNutsFitter, program: BoundProgram, /) -> PosteriorFit:
        plans.append(fitter.plan)

        return fit(fitter, program)

    monkeypatch.setattr(QuiversNutsFitter, "fit", recorded)
    point = SweepPoint(a_fit(), seed=0)

    an_executor(
        RunStore(tmp_path / "a", freeze_id=FREEZE, seed=SEED), an_experiment(tmp_path)
    )(point)
    an_executor(
        RunStore(tmp_path / "b", freeze_id=FREEZE, seed=SEED), an_experiment(tmp_path)
    )(point)
    an_executor(
        RunStore(tmp_path / "c", freeze_id=FREEZE, seed=SEED),
        an_experiment(tmp_path, root=8),
    )(point)

    first, again, other = plans
    assert first.seed == again.seed != other.seed
    assert (first.warmup, first.retained, first.chains) == (5, 8, 2)


def test_a_sampler_failure_is_a_recorded_numerical_failure_with_its_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def failing(fitter: QuiversNutsFitter, program: BoundProgram, /) -> Never:
        del fitter, program
        attempts.append(1)
        msg = "NUTS posterior fitting failed: every retained transition diverged"
        raise PosteriorFittingError(msg)

    monkeypatch.setattr(QuiversNutsFitter, "fit", failing)
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)

    first = an_executor(store, experiment)(point)
    again = an_executor(store, experiment)(point)
    failure = read_failure(store.directory(point))

    assert (first.status, again.cached) == (ManifestStatus.KILLED, True)
    assert attempts == [1]
    assert failure is not None
    assert failure.kind is FailureKind.NUMERICAL
    assert "diverged" in failure.diagnostic
    assert failure.seed_derivation == experiment.seed_policy.point(
        point.point_id, point.seed
    )
    assert not (store.directory(point) / FIT_MANIFEST_NAME).exists()


def test_an_exhausted_fit_is_refitted_and_recorded_as_an_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def exhausted(fitter: QuiversNutsFitter, program: BoundProgram, /) -> Never:
        del fitter, program
        attempts.append(1)
        msg = "CUDA out of memory"
        raise MemoryError(msg)

    monkeypatch.setattr(QuiversNutsFitter, "fit", exhausted)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)

    outcome = an_executor(store, an_experiment(tmp_path))(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert len(attempts) == 3
    assert failure is not None
    assert failure.kind is FailureKind.INFRASTRUCTURE


GUIDE_LEARNING_RATE: Final = 0.05
CALIBRATION: Final[dict[str, JsonValue]] = {
    "replicates": 2,
    "guide_iterations": 20,
    "guide_learning_rate": GUIDE_LEARNING_RATE,
}


def a_calibrated_fit() -> dict[str, JsonValue]:
    fit = a_fit(calibration=CALIBRATION)
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),
            RandomIntercept(group=PARTICIPANT),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    fit["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }

    return fit


def calibration_rows(store: RunStore, point: SweepPoint) -> list[dict[str, JsonValue]]:
    rows = read_rows(store.directory(point))

    assert rows is not None

    return [dict(row) for row in rows.rows if "summary" in row]


class SeededCalibrationReport:
    """A calibration record exposing its caller-owned rank stream."""

    def __init__(self, rank_seed: int, *, clears_floor: bool = True) -> None:
        self.replicates = 2
        self.long_rows: tuple[dict[str, JsonValue], ...] = (
            {"summary": "seed-probe", "rank_seed": rank_seed},
            {"summary": "cutpoint_recovery", "clears_floor": clears_floor},
        )


class SeededCampaign:
    """A calibration campaign reducing to a seeded report and recording no replicate."""

    def __init__(self, report: SeededCalibrationReport) -> None:
        self.records: tuple[ReplicateRecord, ...] = ()
        self.failed: tuple[ReplicateRecord, ...] = ()
        self.report = report


def seeded_calibration(
    compiled: ResponseProgram,
    design: JudgmentDesign,
    *,
    replicates: int,
    guide_iterations: int,
    guide_learning_rate: float,
    plan: NutsPlan,
    seeds: CalibrationSeeds,
) -> SeededCampaign:
    """Return a deterministic campaign of the rank stream selected for a fit."""
    del compiled, design, replicates, guide_iterations, guide_learning_rate, plan
    return SeededCampaign(SeededCalibrationReport(seeds.rank))


def unfitted_calibration(*arguments: object, **keywords: object) -> CalibrationCampaign:
    """Return a campaign in which some method failed every replicate."""
    del arguments, keywords
    return CalibrationCampaign(
        tuple(
            ReplicateRecord(
                replicate=replicate,
                seed=replicate,
                nuts=None,
                evidence_bounds={InferenceMethod.STRUCTURED: -1.0},
                failures={
                    InferenceMethod.NUTS: "a chain diverged",
                    InferenceMethod.MEAN_FIELD: "a scale is not finite",
                },
            )
            for replicate in range(2)
        ),
        None,
    )


def test_a_calibrated_fit_records_the_calibration_of_its_replicates(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_calibrated_fit(), seed=0)
    derived = experiment.seed_policy.point(point.point_id, point.seed)

    outcome = an_executor(store, experiment)(point)
    rows = read_rows(store.directory(point))
    calibrated = calibration_rows(store, point)

    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False)
    assert rows is not None
    metrics = [row["metric"] for row in rows.rows if "metric" in row]
    assert metrics[:3] == ["divergences", "max-rhat", "min-ess"]
    assert metrics.count("response-prediction") == 8
    assert metrics.count("split-half-ceiling") == 1
    assert {row["summary"] for row in calibrated} == {
        "sbc",
        "sbc_rank_bin",
        "coverage",
        "cutpoint_recovery",
        "replicate",
    }
    replicates = [row for row in calibrated if row["summary"] == "replicate"]
    assert [(row["replicate"], row["method"]) for row in replicates] == [
        (replicate, method)
        for replicate in range(2)
        for method in ("nuts", "mean_field", "structured")
    ]
    assert all(row["retained"] is True for row in replicates)
    assert all(row["failure"] is None for row in replicates)
    assert all(
        isinstance(row["max_rhat"], float)
        for row in replicates
        if row["method"] == "nuts"
    )
    assert all(
        isinstance(row["evidence_bound"], float)
        for row in replicates
        if row["method"] != "nuts"
    )
    assert {row["method"] for row in calibrated if row["summary"] == "coverage"} == {
        "nuts",
        "structured",
        "mean_field",
    }
    assert {
        row["replicates"] for row in calibrated if row["summary"] != "replicate"
    } == {2}

    for row in calibrated:
        assert row["freeze_id"] == experiment.freeze_id
        assert row["config_hash"] == config_digest(point.config)
        assert (row["point_id"], row["seed"], row["n"]) == (
            point.point_id,
            derived.root,
            2,
        )
        assert row["residual_measured"] is False
        assert (row["budget_kind"], row["budget_depth"], row["budget_beam"]) == (
            "none",
            None,
            None,
        )


def test_calibration_rows_are_reproduced_by_the_root_and_change_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lofi.run.fits.prior_predictive_calibration", seeded_calibration
    )
    point = SweepPoint(a_calibrated_fit(), seed=0)
    first, again, other = (
        RunStore(tmp_path / "first", freeze_id=FREEZE, seed=SEED),
        RunStore(tmp_path / "again", freeze_id=FREEZE, seed=SEED),
        RunStore(tmp_path / "other", freeze_id=FREEZE, seed=SEED),
    )

    an_executor(first, an_experiment(tmp_path))(point)
    an_executor(again, an_experiment(tmp_path))(point)
    an_executor(other, an_experiment(tmp_path, root=8))(point)

    assert calibration_rows(first, point) == calibration_rows(again, point)
    assert calibration_rows(first, point) != calibration_rows(other, point)


def test_a_calibration_over_shared_thresholds_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unfitted(fitter: QuiversNutsFitter, program: BoundProgram, /) -> Never:
        del fitter, program
        msg = "a refused calibration was fitted"
        raise AssertionError(msg)

    monkeypatch.setattr(QuiversNutsFitter, "fit", unfitted)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(calibration=CALIBRATION), seed=0)

    with pytest.raises(PointExecutionError, match="do not vary over participants"):
        an_executor(store, an_experiment(tmp_path))(point)

    assert not store.directory(point).exists()


def test_a_calibration_below_the_recovery_floor_is_recorded_with_its_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def below_floor(
        *arguments: object, seeds: CalibrationSeeds, **keywords: object
    ) -> SeededCampaign:
        del arguments, keywords
        return SeededCampaign(SeededCalibrationReport(seeds.rank, clears_floor=False))

    monkeypatch.setattr(fits, "prior_predictive_calibration", below_floor)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_calibrated_fit(), seed=0)

    outcome = an_executor(store, an_experiment(tmp_path))(point)
    recovery = [
        row
        for row in calibration_rows(store, point)
        if row["summary"] == "cutpoint_recovery"
    ]

    assert outcome.status is ManifestStatus.COMPLETED
    assert [row["clears_floor"] for row in recovery] == [False]
    assert (store.directory(point) / FIT_MANIFEST_NAME).exists()


def test_a_calibration_retaining_no_replicate_is_a_recorded_numerical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fits, "prior_predictive_calibration", unfitted_calibration)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_calibrated_fit(), seed=0)

    outcome = an_executor(store, an_experiment(tmp_path))(point)
    failure = read_failure(store.directory(point))

    assert outcome.status is ManifestStatus.KILLED
    assert failure is not None
    assert failure.kind is FailureKind.NUMERICAL
    assert "no calibration replicate of 2 was fitted by every method" in (
        failure.diagnostic
    )
    assert "replicate 1: nuts (a chain diverged)" in failure.diagnostic
    assert "mean_field (a scale is not finite)" in failure.diagnostic
    assert read_rows(store.directory(point)) is None
    assert not (store.directory(point) / FIT_MANIFEST_NAME).exists()


def test_a_calibration_fits_its_guides_at_the_declared_learning_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates: list[float] = []

    def recorded(
        compiled: ResponseProgram,
        design: JudgmentDesign,
        *,
        replicates: int,
        guide_iterations: int,
        guide_learning_rate: float,
        plan: NutsPlan,
        seeds: CalibrationSeeds,
    ) -> SeededCampaign:
        rates.append(guide_learning_rate)
        return seeded_calibration(
            compiled,
            design,
            replicates=replicates,
            guide_iterations=guide_iterations,
            guide_learning_rate=guide_learning_rate,
            plan=plan,
            seeds=seeds,
        )

    monkeypatch.setattr(fits, "prior_predictive_calibration", recorded)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    fit = a_calibrated_fit()
    fit["calibration"] = {**CALIBRATION, "guide_learning_rate": 0.125}

    an_executor(store, an_experiment(tmp_path))(SweepPoint(fit, seed=0))

    assert rates == [0.125]


def test_a_fit_recorded_under_another_experiment_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(fitter: QuiversNutsFitter, program: BoundProgram, /) -> Never:
        del fitter, program
        msg = "NUTS posterior fitting failed"
        raise PosteriorFittingError(msg)

    monkeypatch.setattr(QuiversNutsFitter, "fit", failing)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)
    an_executor(store, an_experiment(tmp_path, root=8))(point)

    with pytest.raises(ReseededRunError) as refused:
        an_executor(store, an_experiment(tmp_path))(point)

    assert refused.value.point_id == point.point_id


def test_an_unregistered_dataset_is_refused_by_the_registered_designs() -> None:
    with pytest.raises(UnknownTagError, match="no-such-ratings"):
        RegisteredDesigns().read("no-such-ratings")


def test_a_launch_reads_each_dataset_once_for_every_point_it_fits(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    designs = TinyDesigns()
    executor = an_executor(store, experiment, designs=designs)
    points = [SweepPoint(a_fit(), seed=seed) for seed in (0, 1)]

    for point in points:
        executor(point)

    manifests = [fit_manifest(store, point) for point in points]

    assert designs.resolved == [DATASET]
    assert manifests[0].design == manifests[1].design == PROVENANCE

    an_executor(store, experiment, designs=designs)(SweepPoint(a_fit(), seed=2))

    assert designs.resolved == [DATASET, DATASET]


class ByListDesigns:
    """Read a small crossed design carrying list, verb and frame factors."""

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(
            LEVELS,
            factors=("list", "verb", "frame"),
            readouts=(SCORE,),
        )
        design = codec.encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{group}-{verb}-{frame}-{participant}",
                (group + verb + frame + participant) % LEVELS,
                RowColumns(
                    factors={
                        "list": f"list-{group}",
                        "verb": ("think", "tell")[verb],
                        "frame": ("_ NP", "_ S_that")[frame],
                    },
                    readouts={SCORE: float(group + verb - frame)},
                ),
            )
            for group in range(2)
            for verb in range(2)
            for frame in range(2)
            for participant in range(2)
        )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


class MixedCoverageDesigns:
    """Read list folds containing both seen and new held-out participants."""

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(LEVELS, factors=("list",))
        records = (
            JudgmentRow(
                "participant-shared",
                "item-0",
                2,
                RowColumns(factors={"list": "list-0"}, readouts={}),
            ),
            JudgmentRow(
                "participant-shared",
                "item-1",
                4,
                RowColumns(factors={"list": "list-1"}, readouts={}),
            ),
            JudgmentRow(
                "participant-list-0",
                "item-2",
                1,
                RowColumns(factors={"list": "list-0"}, readouts={}),
            ),
            JudgmentRow(
                "participant-list-1",
                "item-3",
                5,
                RowColumns(factors={"list": "list-1"}, readouts={}),
            ),
        )

        return ReadDesign(codec.encode(records), PROVENANCE.with_(tag=dataset))


POOLED_RECORDS: Final = tuple(
    JudgmentRow(
        f"participant-{participant}",
        f"item-{item}",
        (participant + 3 * item) % LEVELS,
        RowColumns(factors={}, readouts={}),
    )
    for participant in range(3)
    for item in range(4)
)
"""Rows whose response-grain folds pool every participant in each fold."""


class PooledParticipantDesigns:
    """Read crossed rows whose held-out responses come from seen participants."""

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(LEVELS, factors=())

        return ReadDesign(codec.encode(POOLED_RECORDS), PROVENANCE.with_(tag=dataset))


def a_response_grain_comparison(
    objective: Literal["conditional", "truncated_marginal"],
) -> tuple[dict[str, JsonValue], ComparisonSpec]:
    """Return one MAP candidate holding out responses in two folds."""
    document = a_map_fit(marginal=objective == "truncated_marginal")
    comparison = ComparisonSpec(
        held_out_unit="response",
        readout_units={},
        splitter="kfold",
        folds=2,
        seed=23,
        coverage_threshold=0.0,
        unseen_participant=(
            HeldOutParticipantPolicy.MARGINAL
            if objective == "truncated_marginal"
            else HeldOutParticipantPolicy.PLUG_IN
        ),
        models=ModelComparisonSpec(
            candidates=("conditional", "truncated_marginal"),
            candidate=objective,
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )
    document["comparison"] = dumped(comparison)

    return document, comparison


@pytest.mark.parametrize("objective", ["conditional", "truncated_marginal"])
def test_a_pooled_held_out_fold_records_each_participants_density_and_count(
    tmp_path: Path, objective: Literal["conditional", "truncated_marginal"]
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    document, comparison = a_response_grain_comparison(objective)
    point = SweepPoint(document, seed=0)
    outcome = an_executor(
        store,
        an_experiment(tmp_path),
        designs=PooledParticipantDesigns(),
    )(point)
    recorded = read_rows(store.directory(point))
    split = splitter_for(comparison).split(
        POOLED_RECORDS,
        CoverageGate(
            comparison.coverage_threshold, policy=comparison.unseen_participant
        ),
    )

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    densities = [
        row
        for row in recorded.rows
        if row["metric"] == "held-out-predictive-log-density"
    ]
    expected: dict[tuple[int, str], int] = {}

    for fold in split.folds:
        for row in fold.held_out:
            key = (fold.index, row.participant)
            expected[key] = expected.get(key, 0) + 1

    assert {
        (row["fold"], row["participant"]): row["n"] for row in densities
    } == expected
    assert len(densities) == len(expected)
    assert len({participant for _, participant in expected}) == 3
    assert {row["participant_regime"] for row in densities} == {"seen-participant"}
    assert all(
        isinstance(row["value"], float) and math.isfinite(row["value"])
        for row in densities
    )


class ChoiceCoverageDesigns:
    """Read list folds with linked readouts on two displayed alternatives."""

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(
            2,
            factors=("list",),
            readouts=(SCORE,),
            alternatives=2,
        )
        records = tuple(
            JudgmentRow(
                participant,
                f"trial-{position}",
                position % 2,
                RowColumns(
                    factors={"list": f"list-{position % 2}"},
                    readouts={},
                    alternative_readouts={
                        SCORE: (-1.0 + 0.2 * position, 1.0 - 0.1 * position)
                    },
                ),
                (f"choice-{position}-left", f"choice-{position}-right"),
            )
            for position, participant in enumerate(
                (
                    "participant-shared",
                    "participant-shared",
                    "participant-list-0",
                    "participant-list-1",
                )
            )
        )

        return ReadDesign(codec.encode(records), PROVENANCE.with_(tag=dataset))


def a_map_comparison_fit(
    objective: Literal["conditional", "truncated_marginal"],
    *,
    policy: HeldOutParticipantPolicy | None,
    coverage_threshold: float = 0.0,
) -> dict[str, JsonValue]:
    """Return one MAP candidate with a declared by-list comparison."""
    document = a_map_fit(marginal=objective == "truncated_marginal")
    document["comparison"] = dumped(
        ComparisonSpec(
            held_out_unit="list",
            readout_units={},
            splitter="leave-one-group-out",
            folds=None,
            seed=23,
            coverage_threshold=coverage_threshold,
            unseen_participant=policy,
            models=ModelComparisonSpec(
                candidates=("conditional", "truncated_marginal"),
                candidate=objective,
                criterion=ModelComparisonCriterion.PSIS_LOO,
                weighting=ModelWeighting.STACKING,
                pareto_fraction=0.1,
                pareto_threshold=0.7,
            ),
        )
    )

    return document


def a_choice_comparison_fit() -> dict[str, JsonValue]:
    """Return a conditional MAP comparison over a linked forced choice."""
    document = a_map_fit()
    response = ResponseSpec(
        family=Choice(),
        levels=2,
        objective=ConditionalLikelihood(),
        predictor=(LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(),
    )
    document["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    comparison = ComparisonSpec(
        held_out_unit="list",
        readout_units={SCORE: "item"},
        splitter="leave-one-group-out",
        folds=None,
        seed=23,
        coverage_threshold=0.0,
        unseen_participant=HeldOutParticipantPolicy.PLUG_IN,
        models=ModelComparisonSpec(
            candidates=("conditional", "twin"),
            candidate="conditional",
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )
    document["comparison"] = dumped(comparison)

    return document


def test_a_map_fold_refit_preserves_linked_forced_choice_rows(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_choice_comparison_fit(), seed=0)
    outcome = an_executor(
        store,
        an_experiment(tmp_path),
        designs=ChoiceCoverageDesigns(),
    )(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    densities = tuple(
        row
        for row in recorded.rows
        if row["metric"] == "held-out-predictive-log-density"
    )
    assert len(densities) == 4
    assert {row["objective"] for row in densities} == {"conditional"}


def test_a_map_comparison_executes_both_candidates_and_scores_every_declared_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted_objectives: list[str] = []
    seeds: list[int] = []
    fitted = MapFitter.fit

    def recorded(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        seeds.append(fitter.seed)
        fitted_objectives.append(type(objective).__name__)
        return fitted(fitter, objective)

    monkeypatch.setattr(MapFitter, "fit", recorded)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = an_executor(
        store,
        an_experiment(tmp_path),
        designs=MixedCoverageDesigns(),
    )
    declarations = (
        ("conditional", HeldOutParticipantPolicy.PLUG_IN),
        ("truncated_marginal", HeldOutParticipantPolicy.MARGINAL),
    )
    points = tuple(
        SweepPoint(a_map_comparison_fit(objective, policy=policy), seed=0)
        for objective, policy in declarations
    )
    densities: list[Mapping[str, JsonValue]] = []

    for point in points:
        outcome = executor(point)
        directory = store.directory(point)
        recorded_rows = read_rows(directory)
        manifest = fit_manifest(store, point)

        assert outcome.status is ManifestStatus.COMPLETED
        assert recorded_rows is not None
        assert (directory / POINT_ESTIMATE_NAME).is_file()
        assert manifest.artifact.name == POINT_ESTIMATE_NAME
        densities.extend(
            row
            for row in recorded_rows.rows
            if row["metric"] == "held-out-predictive-log-density"
        )

    assert fitted_objectives.count("ConditionalPointObjective") == 3
    assert fitted_objectives.count("MarginalPointObjective") == 3
    assert len(seeds) == 6
    assert len(set(seeds)) == 6
    assert {
        (
            row["candidate"],
            row["objective"],
            row["convention"],
            row["fold"],
            row["participant_regime"],
        )
        for row in densities
    } == {
        (
            objective,
            objective,
            policy.value if regime == "new-participant" else None,
            fold,
            regime,
        )
        for objective, policy in declarations
        for fold in range(2)
        for regime in ("seen-participant", "new-participant")
    }
    assert all(row["n"] == 1 for row in densities)
    assert all(
        isinstance(row["value"], float) and math.isfinite(row["value"])
        for row in densities
    )


def test_a_fold_admitted_without_a_policy_reports_its_unpredicted_participants(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(
        a_map_comparison_fit("conditional", policy=None, coverage_threshold=1.0),
        seed=0,
    )

    outcome = an_executor(
        store, an_experiment(tmp_path), designs=MixedCoverageDesigns()
    )(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    densities = [
        row for row in recorded.rows if row["metric"] == fits.HELD_OUT_PREDICTIVE_METRIC
    ]
    unpredicted = [
        row
        for row in recorded.rows
        if row["metric"] == fits.HELD_OUT_UNPREDICTED_METRIC
    ]
    assert {row["participant_regime"] for row in densities} == {"seen-participant"}
    assert [
        (row["fold"], row["participant_regime"], row["value"], row["n"])
        for row in unpredicted
    ] == [(0, "new-participant", 1.0, 1), (1, "new-participant", 1.0, 1)]
    assert all(row["convention"] is None for row in unpredicted)


@pytest.mark.parametrize(
    ("objective", "policy"),
    [
        ("conditional", HeldOutParticipantPolicy.PLUG_IN),
        ("truncated_marginal", HeldOutParticipantPolicy.MARGINAL),
    ],
)
def test_a_declared_policy_predicts_every_unseen_participant_below_the_threshold(
    tmp_path: Path,
    objective: Literal["conditional", "truncated_marginal"],
    policy: HeldOutParticipantPolicy,
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(
        a_map_comparison_fit(objective, policy=policy, coverage_threshold=1.0),
        seed=0,
    )

    outcome = an_executor(
        store, an_experiment(tmp_path), designs=MixedCoverageDesigns()
    )(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    metrics = {row["metric"] for row in recorded.rows}
    unseen = [
        row
        for row in recorded.rows
        if row["metric"] == fits.HELD_OUT_PREDICTIVE_METRIC
        and row["participant_regime"] == "new-participant"
    ]

    assert fits.HELD_OUT_UNPREDICTED_METRIC not in metrics
    assert [(row["fold"], row["convention"]) for row in unseen] == [
        (0, policy.value),
        (1, policy.value),
    ]


def executed_map_comparison(
    tmp_path: Path,
) -> tuple[FitExecutor, tuple[SweepPoint, SweepPoint]]:
    """Execute both MAP candidates and return their shared executor and points."""
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = an_executor(
        store,
        an_experiment(tmp_path),
        designs=MixedCoverageDesigns(),
    )
    points = (
        SweepPoint(
            a_map_comparison_fit(
                "conditional",
                policy=HeldOutParticipantPolicy.PLUG_IN,
            ),
            seed=0,
        ),
        SweepPoint(
            a_map_comparison_fit(
                "truncated_marginal",
                policy=HeldOutParticipantPolicy.MARGINAL,
            ),
            seed=0,
        ),
    )

    for point in points:
        assert executor(point).status is ManifestStatus.COMPLETED

    return executor, points


def comparison_result_rows(path: Path) -> list[dict[str, JsonValue]]:
    """Read the held-out comparison rows from a written results table."""
    return [
        row
        for row in pq.read_table(path / ROWS_NAME).to_pylist()
        if row["metric"] == "held-out-predictive-log-density"
    ]


def test_written_results_emit_one_verified_complete_map_candidate_set(
    tmp_path: Path,
) -> None:
    executor, points = executed_map_comparison(tmp_path)
    directory = tmp_path / "results"

    write_results(
        directory,
        points=points,
        store=executor.store,
        today=date(2026, 11, 1),
    )
    rows = comparison_result_rows(directory)

    assert len(rows) == 8
    assert {row["candidate"] for row in rows} == {
        "conditional",
        "truncated_marginal",
    }
    assert {
        candidate: sum(row["candidate"] == candidate for row in rows)
        for candidate in ("conditional", "truncated_marginal")
    } == {
        "conditional": 4,
        "truncated_marginal": 4,
    }


def test_written_results_gate_an_incomplete_map_candidate_set(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = an_executor(
        store,
        an_experiment(tmp_path),
        designs=MixedCoverageDesigns(),
    )
    points = (
        SweepPoint(
            a_map_comparison_fit(
                "conditional",
                policy=HeldOutParticipantPolicy.PLUG_IN,
            ),
            seed=0,
        ),
        SweepPoint(
            a_map_comparison_fit(
                "truncated_marginal",
                policy=HeldOutParticipantPolicy.MARGINAL,
            ),
            seed=0,
        ),
    )
    assert executor(points[0]).status is ManifestStatus.COMPLETED
    directory = tmp_path / "results"

    write_results(
        directory,
        points=points,
        store=executor.store,
        today=date(2026, 11, 1),
    )

    assert comparison_result_rows(directory) == []


def test_written_results_gate_a_map_candidate_with_a_missing_artifact(
    tmp_path: Path,
) -> None:
    executor, points = executed_map_comparison(tmp_path)
    (executor.store.directory(points[1]) / POINT_ESTIMATE_NAME).unlink()
    directory = tmp_path / "results"

    write_results(
        directory,
        points=points,
        store=executor.store,
        today=date(2026, 11, 1),
    )

    assert comparison_result_rows(directory) == []


def test_written_results_refuse_a_tampered_map_candidate_artifact(
    tmp_path: Path,
) -> None:
    executor, points = executed_map_comparison(tmp_path)
    artifact = executor.store.directory(points[1]) / POINT_ESTIMATE_NAME
    artifact.write_bytes(b"tampered point estimate")

    with pytest.raises(ValueError, match="SHA-256"):
        write_results(
            tmp_path / "results",
            points=points,
            store=executor.store,
            today=date(2026, 11, 1),
        )


def test_a_marginal_map_comparison_uses_the_declared_plug_in_policy(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(
        a_map_comparison_fit(
            "truncated_marginal",
            policy=HeldOutParticipantPolicy.PLUG_IN,
        ),
        seed=0,
    )
    outcome = an_executor(
        store,
        an_experiment(tmp_path),
        designs=MixedCoverageDesigns(),
    )(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    densities = tuple(
        row
        for row in recorded.rows
        if row["metric"] == "held-out-predictive-log-density"
    )
    assert len(densities) == 4
    assert {row["convention"] for row in densities} == {None, "plug_in"}


def test_a_marginal_prediction_error_reseeds_the_complete_map_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    fit_seeds: list[int] = []
    predicted = MarginalPointPredictor.probabilities
    fitted = MapFitter.fit

    def record_seed(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        fit_seeds.append(fitter.seed)
        return fitted(fitter, objective)

    def once(predictor: MarginalPointPredictor, design: JudgmentDesign, /) -> Tensor:
        nonlocal calls
        calls += 1

        if calls == 1:
            msg = "the marginal prediction failed numerically"
            raise MarginalLikelihoodError(msg)

        return predicted(predictor, design)

    monkeypatch.setattr(MarginalPointPredictor, "probabilities", once)
    monkeypatch.setattr(MapFitter, "fit", record_seed)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(
        a_map_comparison_fit(
            "truncated_marginal",
            policy=HeldOutParticipantPolicy.MARGINAL,
        ),
        seed=0,
    )
    outcome = an_executor(
        store,
        an_experiment(tmp_path, numerical="reseed"),
        designs=MixedCoverageDesigns(),
    )(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert calls == 5
    assert len(fit_seeds) == 5
    assert fit_seeds[0] != fit_seeds[2]


@pytest.mark.parametrize("probability", [0.0, float("nan")])
def test_a_nonfinite_predictive_density_reseeds_the_complete_map_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
) -> None:
    calls = 0
    fit_seeds: list[int] = []
    predicted = MarginalPointPredictor.probabilities
    fitted = MapFitter.fit

    def record_seed(fitter: MapFitter, objective: PointObjective, /) -> PointFit:
        fit_seeds.append(fitter.seed)
        return fitted(fitter, objective)

    def once(predictor: MarginalPointPredictor, design: JudgmentDesign, /) -> Tensor:
        nonlocal calls
        calls += 1

        if calls == 1:
            return predictor.parameters.new_full(
                (design.n_rows, design.response_levels), probability
            )

        return predicted(predictor, design)

    monkeypatch.setattr(MarginalPointPredictor, "probabilities", once)
    monkeypatch.setattr(MapFitter, "fit", record_seed)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(
        a_map_comparison_fit(
            "truncated_marginal",
            policy=HeldOutParticipantPolicy.MARGINAL,
        ),
        seed=0,
    )
    outcome = an_executor(
        store,
        an_experiment(tmp_path, numerical="reseed"),
        designs=MixedCoverageDesigns(),
    )(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert calls == 5
    assert len(fit_seeds) == 5
    assert fit_seeds[0] != fit_seeds[2]


class DegenerateGrammaticalityDesigns:
    """Read a root-mass pair with one constant log-odds value."""

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        log_half = math.log(0.5)
        design = DesignCodec(
            LEVELS,
            factors=(),
            readouts=("p_gram", "p_gram_complement"),
        ).encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                (participant + item) % LEVELS,
                RowColumns(
                    factors={},
                    readouts={
                        "p_gram": log_half,
                        "p_gram_complement": log_half,
                    },
                ),
            )
            for participant in range(2)
            for item in range(4)
        )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


class SpreadGrammaticalityDesigns:
    """Read a root-mass pair with a declared spread of log-odds."""

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        odds = (-3.0, -1.0, 1.0, 3.0)
        design = DesignCodec(
            LEVELS,
            factors=(),
            readouts=("p_gram", "p_gram_complement"),
        ).encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                (participant + item) % LEVELS,
                RowColumns(
                    factors={},
                    readouts={
                        "p_gram": -math.log1p(math.exp(-odds[item])),
                        "p_gram_complement": -math.log1p(math.exp(odds[item])),
                    },
                ),
            )
            for participant in range(2)
            for item in range(4)
        )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


def a_grammaticality_fit(*, minimum_interquartile_range: float) -> dict[str, JsonValue]:
    """Return a fit gated by the declared root grammaticality range."""
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(
                readout="p_gram",
                linking=LinkingSpec(
                    family="logit",
                    columns={"complement": "p_gram_complement"},
                ),
            ),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    document = a_variational_fit()
    document["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    document["dynamic_range"] = {
        "readout": "p_gram",
        "criterion": json.loads(
            DynamicRangeCriterion(
                registered_at="2020-01-01T00:00:00+00:00",
                minimum_interquartile_range=minimum_interquartile_range,
                ceiling=2.0,
                maximum_ceiling_fraction=0.5,
            ).model_dump_json()
        ),
    }

    return document


def test_a_degenerate_declared_dynamic_range_is_refused_before_launch(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_grammaticality_fit(minimum_interquartile_range=0.1), seed=0)
    executor = FitExecutor(
        ExperimentStore(store.root, an_experiment(tmp_path)),
        designs=DegenerateGrammaticalityDesigns(),
        compiler=ResponseCompiler.installed(),
        environment=ENVIRONMENT,
        infrastructure_attempts=1,
    )

    with pytest.raises(PointExecutionError, match="interquartile range"):
        executor(point)

    assert not store.directory(point).exists()


def test_a_passing_dynamic_range_is_recorded_with_the_fit(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_grammaticality_fit(minimum_interquartile_range=2.0), seed=0)
    executor = FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=SpreadGrammaticalityDesigns(),
        compiler=ResponseCompiler.installed(),
        environment=ENVIRONMENT,
        infrastructure_attempts=1,
    )

    executor.check(point)
    assert not store.directory(point).exists()

    outcome = executor(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    [report] = [row for row in recorded.rows if row["metric"] == "dynamic-range"]
    assert recorded.rows[0] == report
    assert report["readout"] == "p_gram"
    assert report["value"] == 3.0
    assert report["n"] == 4
    assert report["inventory"] == DATASET
    assert report["minimum"] == -3.0
    assert report["lower_quartile"] == -1.5
    assert report["median"] == 0.0
    assert report["upper_quartile"] == 1.5
    assert report["maximum"] == 3.0
    assert report["ceiling_fraction"] == 0.25
    assert report["criterion_registered_at"] == "2020-01-01T00:00:00+00:00"
    assert report["criterion_minimum_interquartile_range"] == 2.0
    assert report["criterion_ceiling"] == 2.0
    assert report["criterion_maximum_ceiling_fraction"] == 0.5
    assert report["passed"] is True
    assert report["scale"] == "log-odds"


def by_list_ablation_fit() -> dict[str, JsonValue]:
    """Return a fit declaring LOGO by list and two table-ablation rungs."""
    document = a_variational_fit()
    schedule = FitSchedule(
        optimizer=AdamOptimizerSpec(learning_rate=0.02),
        steps=2,
        batch_size=4,
        initial_scale=0.25,
    )
    inventory = ColumnInventory(
        primitive_types=("NP", "S_that"),
        null_type="NONE",
        levels=(
            ColumnLevel(label="_ NP", arguments=("NP",)),
            ColumnLevel(label="_ S_that", arguments=("S_that",)),
        ),
    )
    comparison = ComparisonSpec(
        held_out_unit="list",
        readout_units={SCORE: "item"},
        splitter="leave-one-group-out",
        folds=None,
        seed=23,
        coverage_threshold=0.0,
        unseen_participant=None,
        models=ModelComparisonSpec(
            candidates=("declared", "twin"),
            candidate="declared",
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )
    rungs = (
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think", "tell")),
            columns=TableColumns(attribute="frame", inventory=inventory),
            width=2,
            schedule=schedule,
            lift=MemorizedCells(),
        ),
        TableAblationSpec(
            rows=TableRows(attribute="verb", levels=("think", "tell")),
            columns=TableColumns(attribute="frame", inventory=inventory),
            width=2,
            schedule=schedule,
            lift=JointTypeTensor(),
        ),
    )
    document["comparison"] = dumped(comparison)
    document["ablation"] = [dumped(rung) for rung in rungs]

    return document


def test_a_by_list_ablation_runs_from_the_fit_document_and_writes_rows(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(by_list_ablation_fit(), seed=0)
    executor = FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=ByListDesigns(),
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=1,
    )

    outcome = executor(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    metrics = [row["metric"] for row in recorded.rows]
    predictions = [
        row for row in recorded.rows if row["metric"] == "response-prediction"
    ]

    assert metrics.count("evidence-bound") == 1
    assert metrics.count("response-prediction") == 16
    assert metrics.count("split-half-ceiling") == 1
    assert metrics.count("held-out-participant-fraction") == 2
    assert metrics.count("table-ablation-log-density") == 4
    assert all(
        isinstance(row["distribution"], list) and len(row["distribution"]) == LEVELS
        for row in predictions
    )
    assert all(
        isinstance(row["soft_label"], list) and len(row["soft_label"]) == LEVELS
        for row in predictions
    )
    assert {row["agreement"] for row in predictions} <= {"low", "medium", "high"}
    assert (store.directory(point) / POSTERIOR_NAME).is_file()
    assert {
        (row["fold"], row["rung"])
        for row in recorded.rows
        if row["metric"] == "table-ablation-log-density"
    } == {(fold, rung) for fold in range(2) for rung in ("memorized", "joint_type")}


def test_a_complete_candidate_set_is_compared_from_its_persisted_pointwise_fits(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = FitExecutor(
        ExperimentStore(store.root, experiment),
        designs=ByListDesigns(),
        compiler=ResponseCompiler({LINEAR: linear_steps}),
        environment=ENVIRONMENT,
        infrastructure_attempts=1,
    )
    documents = []

    for candidate in ("declared", "twin"):
        document = by_list_ablation_fit()
        comparison = document["comparison"]

        if not isinstance(comparison, dict):
            msg = "the comparison fixture has no declaration body"
            raise TypeError(msg)

        models = comparison["models"]

        if not isinstance(models, dict):
            msg = "the comparison fixture has no model-comparison body"
            raise TypeError(msg)

        models["candidate"] = candidate
        document["ablation"] = []
        inference = document["inference"]

        if not isinstance(inference, dict):
            msg = "the comparison fixture has no inference body"
            raise TypeError(msg)

        inference["draws"] = 100
        documents.append(document)

    points = tuple(SweepPoint(document, seed=0) for document in documents)

    for point in points:
        assert executor(point).status is ManifestStatus.COMPLETED

    rows = declared_model_comparison_rows(points, executor.store)

    assert [row["metric"] for row in rows].count("psis-loo-elpd") == 2
    assert [row["metric"] for row in rows].count("psis-loo-elpd-delta") == 1
    assert {row.get("model") for row in rows if row["kind"] == "model"} == {
        "declared",
        "twin",
    }


def test_the_threshold_check_runs_once_per_point_before_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[str | None] = []
    directories: list[bool] = []
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_calibrated_fit(), seed=0)

    def counted(compiled: ResponseProgram, /) -> None:
        checked.append(compiled.threshold_group)
        directories.append(store.directory(point).exists())

    monkeypatch.setattr(fits, "ensure_participant_thresholds", counted)
    monkeypatch.setattr(fits, "prior_predictive_calibration", unfitted_calibration)
    executor = an_executor(store, an_experiment(tmp_path))

    executor.check(point)
    executor(point)

    assert checked == [PARTICIPANT]
    assert directories == [False]


P_GRAM: Final = "p_gram"
P_NOT: Final = "p_not"
LM: Final = "lm"
TOKENS: Final = "tokens"
UNIGRAM: Final = "unigram"
READOUTS: Final = (SCORE, P_GRAM, P_NOT, LM, TOKENS, UNIGRAM)
LINKED: Final = {
    "linear": LinkTerm(readout=SCORE, linking=LinkingSpec(family="linear")),
    "logit": LinkTerm(
        readout=P_GRAM,
        linking=LinkingSpec(family="logit", columns={"complement": P_NOT}),
    ),
    "morcela": LinkTerm(
        readout=LM,
        linking=LinkingSpec(
            family="morcela", columns={"length": TOKENS, "frequency": UNIGRAM}
        ),
    ),
    "probability": LinkTerm(readout=P_GRAM, linking=LinkingSpec(family="probability")),
}


def readouts(row: int, at: int = 0) -> dict[str, float]:
    mass = 0.1 + 0.8 * (((row * 7 + at * 3) % 11) / 11.0)

    return {
        SCORE: -1.5 + (row + at) % 4,
        P_GRAM: math.log(mass),
        P_NOT: math.log1p(-mass),
        LM: -10.0 - (row + at) % 6,
        TOKENS: 3.0 + (row + at) % 5,
        UNIGRAM: -40.0 - 2.0 * ((row + at) % 7),
    }


class EveryKindOfDesign:
    """Reads an ordinal, a forced-choice or a tied-choice design."""

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        rows = range(16)
        alternatives = {"choice": (2, 2), "ties": (3, 2)}

        if dataset == "bounded":
            scale = ClosedScale(0.0, 100.0)
            design = BoundedDesignCodec(readouts=READOUTS).encode(
                BoundedJudgmentRow(
                    f"participant-{row % 4}",
                    f"item-{row}",
                    float(5 + 5 * (row % 16)),
                    scale,
                    RowColumns(factors={}, readouts=readouts(row)),
                )
                for row in rows
            )

            return ReadBoundedDesign(design, PROVENANCE.with_(tag=dataset))

        if dataset in alternatives:
            levels, shown = alternatives[dataset]
            design = DesignCodec(levels, readouts=READOUTS, alternatives=shown).encode(
                JudgmentRow(
                    f"participant-{row % 4}",
                    f"trial-{row}",
                    row % levels,
                    RowColumns(
                        {},
                        {},
                        {
                            name: tuple(readouts(row, at)[name] for at in range(shown))
                            for name in READOUTS
                        },
                    ),
                    tuple(f"item-{row}-{at}" for at in range(shown)),
                )
                for row in rows
            )
        else:
            design = DesignCodec(LEVELS, readouts=READOUTS).encode(
                JudgmentRow(
                    f"participant-{row % 4}",
                    f"item-{row}",
                    row % LEVELS,
                    RowColumns(factors={}, readouts=readouts(row)),
                )
                for row in rows
            )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


def dumped(model: dx.Model) -> JsonValue:
    return json.loads(model.model_dump_json())


def a_response(family: str, term: LinkTerm) -> tuple[str, dict[str, JsonValue]]:
    """Return the dataset a family is fit to and an unvalidated response over a term."""
    if family == "bounded":
        return "bounded", {
            "family": {"family": "bounded", "scale": 0.2},
            "predictor": [
                dumped(term),
                dumped(RandomIntercept(group=PARTICIPANT)),
            ],
            "identification": {"constraint": "identity_link_on_logit"},
        }

    declared: dict[str, tuple[str, dict[str, JsonValue], int, PriorBlock]] = {
        "cumulative": (
            "ordinal",
            {"family": "cumulative"},
            LEVELS,
            PriorBlock(thresholds=ThresholdPriors()),
        ),
        "choice": ("choice", {"family": "choice"}, 2, PriorBlock()),
        "tied-choice": (
            "ties",
            {"family": "choice", "ties": True},
            3,
            PriorBlock(thresholds=ThresholdPriors()),
        ),
        "metric": ("ordinal", {"family": "metric"}, LEVELS, PriorBlock()),
        "rank-consistent": (
            "ordinal",
            {"family": "rank_consistent", "scheme": "corn"},
            LEVELS,
            PriorBlock(),
        ),
    }
    dataset, declaration, levels, priors = declared[family]
    shown = (
        [dumped(FactorTerm(name=POSITION))] if declaration["family"] == "choice" else []
    )

    return dataset, {
        "family": declaration,
        "levels": levels,
        "objective": {"objective": "conditional"},
        "predictor": [dumped(term), *shown],
        "identification": {"constraint": "identity_link_on_logit"},
        "priors": dumped(priors),
    }


FAMILIES: Final = (
    "cumulative",
    "choice",
    "tied-choice",
    "bounded",
    "metric",
    "rank-consistent",
)


def test_the_linked_terms_cover_every_installed_linking_family() -> None:
    assert sorted(LINKED) == list(installed_linkings().available())


@pytest.mark.parametrize("linking", sorted(LINKED))
@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_and_linking_fits_from_a_document_or_is_refused_at_validation(
    tmp_path: Path, family: str, linking: str
) -> None:
    dataset, response = a_response(family, LINKED[linking])
    document: dict[str, JsonValue] = {
        "dataset": dataset,
        "response": {
            (
                "ResponseDeclaration_boundedresponsespec"
                if family == "bounded"
                else "ResponseDeclaration_responsespec"
            ): response
        },
        "inference": {
            "inference": "svi",
            "guide": "structured",
            "iterations": 5,
            "draws": 2,
            "learning_rate": 0.05,
        },
        "elicitation": {"method": "participants", "version": "1"},
        "calibration": None,
        "comparison": None,
        "dynamic_range": None,
        "ablation": [],
        "interannotator": None,
    }
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    executor = FitExecutor(
        ExperimentStore(store.root, an_experiment(tmp_path)),
        designs=EveryKindOfDesign(),
        compiler=ResponseCompiler.installed(),
        environment=ENVIRONMENT,
        infrastructure_attempts=1,
    )
    point = SweepPoint(document, seed=0)

    if family in {"metric", "rank-consistent"} or (
        family == "bounded" and linking != "logit"
    ):
        with pytest.raises(dx.ValidationError):
            executor(point)

        assert not store.directory(point).exists()

        return

    outcome = executor(point)
    recorded = read_rows(store.directory(point))

    assert (outcome.status, outcome.cached) == (ManifestStatus.COMPLETED, False)
    assert recorded is not None
    metrics = [row["metric"] for row in recorded.rows]

    if family == "bounded":
        assert metrics == ["evidence-bound"]
    else:
        levels = {"choice": 2, "tied-choice": 3}.get(family, LEVELS)
        predictions = [
            row for row in recorded.rows if row["metric"] == "response-prediction"
        ]

        assert metrics[0] == "evidence-bound"
        assert len(predictions) == 16
        assert metrics.count("split-half-ceiling") == 1
        assert all(
            isinstance(row["distribution"], list) and len(row["distribution"]) == levels
            for row in predictions
        )
        assert all(
            isinstance(row["soft_label"], list) and len(row["soft_label"]) == levels
            for row in predictions
        )


def a_variational_fit(
    guide: str = "structured", *, learning_rate: float = GUIDE_LEARNING_RATE
) -> dict[str, JsonValue]:
    fit = a_fit()
    fit["inference"] = json.loads(
        VariationalInference(
            guide=InferenceMethod(guide),
            iterations=20,
            draws=6,
            learning_rate=learning_rate,
        ).model_dump_json()
    )

    return fit


def test_a_variational_fit_fits_its_guide_at_the_declared_learning_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates: list[float] = []
    fit = QuiversGuideFitter.fit

    def recorded(fitter: QuiversGuideFitter, program: BoundProgram, /) -> GuideFit:
        rates.append(fitter.learning_rate)

        return fit(fitter, program)

    monkeypatch.setattr(QuiversGuideFitter, "fit", recorded)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_variational_fit(learning_rate=0.125), seed=0)

    outcome = an_executor(store, an_experiment(tmp_path))(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert rates == [0.125]


def test_a_variational_fit_records_its_bound_from_its_own_seed_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds: list[tuple[InferenceMethod, int, int, int]] = []
    fit = QuiversGuideFitter.fit

    def recorded(fitter: QuiversGuideFitter, program: BoundProgram, /) -> GuideFit:
        seeds.append((fitter.method, fitter.iterations, fitter.draws, fitter.seed))

        return fit(fitter, program)

    monkeypatch.setattr(QuiversGuideFitter, "fit", recorded)
    experiment = an_experiment(tmp_path)
    point = SweepPoint(a_variational_fit(), seed=0)
    stores = [
        RunStore(tmp_path / name, freeze_id=FREEZE, seed=SEED)
        for name in ("first", "again")
    ]

    for store in stores:
        outcome = an_executor(store, experiment)(point)

        assert outcome.status is ManifestStatus.COMPLETED

    an_executor(
        RunStore(tmp_path / "other", freeze_id=FREEZE, seed=SEED),
        an_experiment(tmp_path, root=8),
    )(point)
    first, again = (read_rows(store.directory(point)) for store in stores)
    derived = experiment.seed_policy.point(point.point_id, point.seed)
    streams = derived.policy(accelerator=None, samplers=(fits.GUIDE_STREAM,))

    assert first is not None
    assert again is not None
    assert first.rows == again.rows
    metrics = [row["metric"] for row in first.rows]
    assert (metrics[0], first.rows[0]["n"]) == ("evidence-bound", 6)
    assert metrics.count("response-prediction") == 8
    assert metrics.count("split-half-ceiling") == 1
    assert seeds[0] == (
        InferenceMethod.STRUCTURED,
        20,
        6,
        streams.sampler_seed(fits.GUIDE_STREAM),
    )
    assert seeds[0] == seeds[1] != seeds[2]
    assert fit_manifest(stores[0], point).fit.inference.method is (
        InferenceMethod.STRUCTURED
    )


def test_two_guides_declared_by_a_document_fit_differently(tmp_path: Path) -> None:
    experiment = an_experiment(tmp_path)
    bounds = []

    for guide in ("structured", "mean_field"):
        store = RunStore(tmp_path / guide, freeze_id=FREEZE, seed=SEED)
        point = SweepPoint(a_variational_fit(guide), seed=0)
        an_executor(store, experiment)(point)
        recorded = read_rows(store.directory(point))

        assert recorded is not None

        bounds.append(recorded.rows[0]["value"])

    assert bounds[0] != bounds[1]


def test_a_response_that_draws_no_parameter_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    fit = a_fit()
    response = ResponseSpec(
        family=Choice(),
        levels=2,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(
                readout=SCORE, linking=LinkingSpec(family=LINEAR, fixed={"slope": 1.0})
            ),
        ),
        identification=FixedDiscrimination(value=1.0),
    )
    fit["response"] = {
        "ResponseDeclaration_responsespec": json.loads(response.model_dump_json())
    }
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(fit, seed=0)

    with pytest.raises(PointExecutionError, match="draws no parameter"):
        an_executor(store, an_experiment(tmp_path))(point)

    assert not store.directory(point).exists()


class RaterDesigns:
    """Read four participants who each rate the same five items."""

    def read(self, dataset: str, /) -> ReadDesign:
        codec = DesignCodec(LEVELS, factors=(), readouts=(SCORE,))
        design = codec.encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                min(LEVELS - 1, item + (participant * item) % 3),
                RowColumns(factors={}, readouts={SCORE: -1.5 + item}),
            )
            for participant in range(4)
            for item in range(5)
        )

        return ReadDesign(design, PROVENANCE.with_(tag=dataset))


def test_a_fit_reports_only_the_declared_interannotator_coefficients(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    fit = a_fit()
    fit["interannotator"] = {
        "coefficients": ["spearman"],
        "replicates": 2000,
        "level": 0.9,
    }
    point = SweepPoint(fit, seed=0)

    outcome = an_executor(store, experiment, designs=RaterDesigns())(point)
    recorded = read_rows(store.directory(point))

    assert outcome.status is ManifestStatus.COMPLETED
    assert recorded is not None
    ceilings = [
        row for row in recorded.rows if str(row["metric"]).startswith("interannotator-")
    ]
    assert [row["metric"] for row in ceilings] == ["interannotator-spearman"]
    (spearman,) = ceilings
    assert spearman["n"] == 6
    assert spearman["level"] == 0.9
    assert isinstance(spearman["value"], float)
    lower, upper = spearman["lower"], spearman["upper"]
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= spearman["value"] <= upper


def test_a_fit_without_an_interannotator_declaration_reports_no_ceiling(
    tmp_path: Path,
) -> None:
    experiment = an_experiment(tmp_path)
    store = RunStore(tmp_path / "store", freeze_id=FREEZE, seed=SEED)
    point = SweepPoint(a_fit(), seed=0)

    an_executor(store, experiment, designs=RaterDesigns())(point)
    recorded = read_rows(store.directory(point))

    assert recorded is not None
    assert not any(
        str(row["metric"]).startswith("interannotator-") for row in recorded.rows
    )
