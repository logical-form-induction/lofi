import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import didactic.api as dx
import pytest
import yaml
from torch import nn

from lofi.data import SelectionReport
from lofi.data._selection import CategoryBudget, RankBand
from lofi.data.synthetic.spec.inference import (
    InferenceNoiseRateSpec,
    InferenceNoiseSpec,
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
    UniformLabelConfusionSpec,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lexicon import ClampReport
from lofi.lf import RELATION_LABELS
from lofi.run import (
    RandomStreams,
    RunPhase,
    SeedPolicy,
    SweepPoint,
    plan_training,
    resolve_device_policy,
)
from lofi.run._boundary.manifest import MeasuredRole, measure_point
from lofi.run._boundary.metrics import (
    METRICS_NAME,
    CountsValue,
    CountValue,
    FlagValue,
    HistogramRecord,
    MeasuresValue,
    MeasureValue,
    MetricsRecord,
    MetricsRecorder,
    TextValue,
    _epoch,
    _occupancy,
    contributed_record,
    corpus_record,
    inference_record,
    metrics_row,
    read_metrics,
    selection_record,
    write_metrics,
)
from lofi.spec import RunSpecCodec, config_search_path
from lofi.spec.constraint_sources import ClampSpec
from lofi.terms import Identity
from lofi.train import (
    BestCheckpointId,
    CheckpointId,
    EpochReport,
    EvalResult,
    FittingReport,
    ParserFittingPlan,
    ResumeCheckpointId,
    ResumePosition,
    TrainResult,
    prepare_training,
)
from lofi.training_plugins import TrainingUnit

if TYPE_CHECKING:
    from lofi.data.synthetic import Histogram, SyntheticCorpus, SyntheticDataset
    from lofi.data.synthetic.training import FormTypingTarget
    from lofi.run import TrainingBlueprint
    from lofi.spec import RunSpec
    from lofi.train import PreparedTraining

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))
INFERENCE = ROOT / "lofi/configs/data/datasets/train/dataset/synthetic-inference.yaml"
INFERENCE_SEMANTICS = (
    ROOT
    / "tests/acceptance/documents/data/datasets/train/dataset/source"
    / "interpretation/related-factors.yaml"
)


def composed(directory: Path, *selections: str) -> RunSpec:
    """Compose the packaged parser run with further group selections."""
    document = directory / "composed.yaml"
    chosen = "".join(f"  - {selection}\n" for selection in selections)
    document.write_text(
        f"defaults:\n  - synthetic-parser\n{chosen}", encoding="utf-8", newline="\n"
    )

    return RunSpecCodec(search_path=(CONFIG.parent, *config_search_path(directory)))(
        document
    )


def drawn(run_spec: RunSpec) -> SyntheticCorpus:
    """Return the corpus a run's data section draws."""
    planned = plan_training(run_spec)
    return cast("SyntheticDataset", planned.lower_dataset("train")).corpus


def counted(histogram: Histogram) -> HistogramRecord:
    """Return what a histogram's record should hold."""
    return HistogramRecord(
        counts={str(value): count for value, count in histogram.counts.items()},
        total=histogram.total,
        mean=histogram.mean if histogram.total else None,
    )


def test_a_corpus_record_holds_every_report_the_corpus_achieved(
    tmp_path: Path,
) -> None:
    corpus = drawn(
        composed(
            tmp_path,
            "data.datasets.train.dataset.corruption.proportion: quarter",
            "data.datasets.train.dataset.corruption.operators: swap-deletion",
            "data.datasets.train.dataset.noise.grammatical: tenth",
            "data.datasets.train.dataset.noise.ungrammatical: tenth",
        )
    )
    record = corpus_record(corpus)
    corruption = corpus.corruption
    noise = corpus.noise

    assert corruption is not None
    assert noise is not None
    assert record.length.achieved == counted(corpus.length.achieved)
    assert record.length.resampled == corpus.length.resampled
    assert record.lexicon.forms == counted(corpus.lexicon.forms)
    assert record.lexicon.classes == {
        category: counted(histogram)
        for category, histogram in sorted(corpus.lexicon.classes.items())
    }
    assert record.lexicon.realized_size == corpus.lexicon.realized_size
    assert record.lexicon.realized_vocabulary_size == (
        corpus.lexicon.realized_vocabulary_size
    )
    assert record.lexicon.token_shares == dict(
        sorted(corpus.lexicon.token_shares.items())
    )
    assert record.lexicon.rank_frequency_exponent == (
        corpus.lexicon.rank_frequency_exponent
    )
    assert record.lexicon.coupling == corpus.lexicon.coupling
    assert record.lexicon.concordance == corpus.lexicon.concordance
    assert record.inventory.categories == dict(
        sorted(corpus.inventory.categories.items())
    )
    assert record.inventory.argument_counts == dict(
        sorted(corpus.inventory.argument_counts.items())
    )
    assert record.inventory.composition_orders == dict(
        sorted(corpus.inventory.composition_orders.items())
    )
    assert record.inventory.max_arguments == corpus.inventory.max_arguments
    assert record.inventory.max_order == corpus.inventory.max_order
    assert record.inventory.arguments == counted(corpus.inventory.arguments)
    assert record.inventory.orders == counted(corpus.inventory.orders)
    assert record.inventory.primitives == corpus.inventory.primitives
    assert record.inventory.realized_primitive_count == (
        corpus.inventory.realized_primitive_count
    )
    assert record.ambiguity.derivations == counted(corpus.ambiguity.derivations)
    assert record.corruption is not None
    assert record.corruption.corrupted == corruption.corrupted
    assert record.corruption.achieved_proportion == corruption.achieved_proportion
    assert record.corruption.accepted == dict(sorted(corruption.accepted.items()))
    assert record.noise is not None
    assert record.noise.flip_count == noise.flip_count
    assert record.noise.achieved_rate == noise.achieved_rate
    assert record.scope is None
    assert record.ill_typing is None
    assert type(record).model_validate_json(record.model_dump_json()) == record


def test_a_corpus_declared_without_corruption_or_noise_records_neither(
    tmp_path: Path,
) -> None:
    record = corpus_record(drawn(composed(tmp_path)))

    assert record.corruption is None
    assert record.noise is None
    assert record.length.achieved.total == len(drawn(composed(tmp_path)).examples)


def test_equal_declarations_give_byte_equal_corpus_records(tmp_path: Path) -> None:
    run_spec = composed(
        tmp_path,
        "data.datasets.train.dataset.noise.grammatical: tenth",
        "data.datasets.train.dataset.noise.ungrammatical: tenth",
    )
    first = corpus_record(drawn(run_spec)).model_dump_json()
    second = corpus_record(drawn(run_spec)).model_dump_json()
    other = corpus_record(
        drawn(
            run_spec.with_(
                data=run_spec.data.with_(
                    datasets={
                        "train": run_spec.data.datasets["train"].with_(
                            dataset=run_spec.data.datasets["train"].dataset.with_(
                                seed=6
                            )
                        )
                    }
                )
            )
        )
    ).model_dump_json()

    assert first == second
    assert first != other


def inference_declaration(
    *, noisy: bool, semantic: bool = False
) -> SyntheticInferenceDecl:
    """Return the packaged inference dataset, with label noise if asked."""
    document = yaml.safe_load(INFERENCE.read_text(encoding="utf-8"))
    if semantic:
        document["source"]["interpretation"] = yaml.safe_load(
            INFERENCE_SEMANTICS.read_text(encoding="utf-8")
        )
    declaration = SyntheticInferenceDecl.model_validate(document)

    if not noisy:
        return declaration

    return declaration.with_(
        label_noise=InferenceNoiseSpec(
            rates=tuple(
                InferenceNoiseRateSpec(relation=label, rate=0.3)
                for label in RELATION_LABELS
            ),
            confusion=UniformLabelConfusionSpec(),
            tolerance=1.0,
        )
    )


def test_an_inference_record_holds_both_corpora_and_the_pair_reports() -> None:
    inference = SyntheticInferenceLowerer().lower(
        inference_declaration(noisy=False),
        ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    record = inference_record(inference)

    assert record.premise == corpus_record(inference.premises)
    assert record.hypothesis == corpus_record(inference.hypotheses)
    assert record.inference.labels == dict(sorted(inference.inference.labels.items()))
    assert record.inference.total == inference.inference.total
    assert record.inference.satisfied is inference.inference.satisfied()
    assert record.complexity.overlap == counted(inference.complexity.overlap)
    assert set(record.complexity.premise) == set(inference.complexity.premise)
    assert record.confusion is None
    assert record.semantic is None


def test_an_inference_record_measures_one_semantic_joint_over_both_sides() -> None:
    inference = SyntheticInferenceLowerer().lower(
        inference_declaration(noisy=False, semantic=True),
        ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    report = inference.semantic
    record = inference_record(inference).semantic

    assert report is not None
    assert record is not None
    assert record.joint.token_counts == dict(sorted(report.joint.token_counts.items()))
    assert record.per_form_meanings == counted(report.per_form_meanings)
    assert record.items == dict(sorted(report.items.items()))
    assert record.item_shares == dict(sorted(report.item_shares.items()))
    assert record.cells == {
        category: dict(sorted(counts.items()))
        for category, counts in sorted(report.cells.items())
    }


def test_an_inference_record_holds_the_confusion_of_noised_labels() -> None:
    inference = SyntheticInferenceLowerer().lower(
        inference_declaration(noisy=True),
        ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    record = inference_record(inference)
    noise = inference.noise

    assert noise is not None
    assert record.confusion is not None
    assert record.confusion.flip_count == noise.flip_count
    assert record.confusion.decided == noise.decided
    assert record.confusion.matrix == {
        label: dict(sorted(row.items()))
        for label, row in sorted(noise.matrix().items())
    }


def test_a_selection_record_holds_the_budget_by_category() -> None:
    report = SelectionReport(
        3,
        7,
        0.5,
        {"NP": CategoryBudget(2, 5, 0.25), "S\\NP": CategoryBudget(1, 2, 0.125)},
        concordance=None,
        selector=RankBand(0.0, 1.0),
    )
    record = selection_record(
        report,
        ClampSpec().settings(),
        ClampReport(2, 1),
        Identity(),
    )

    assert (record.types_pinned, record.tokens_covered, record.mass_share) == (
        3,
        7,
        0.5,
    )
    assert list(record.by_category) == ["NP", "S\\NP"]
    assert record.by_category["NP"].tokens_covered == 5
    assert record.concordance is None
    assert record.clamp.mode == "hard"
    assert record.clamp.penalty is None
    assert record.clamp.gradient_scale == 1.0
    assert record.clamp.matching == "identity"
    assert record.clamp.dropped_clamps == 2
    assert record.clamp.skipped_items == 1


def test_a_contributed_report_records_each_kind_of_measurement() -> None:
    record = contributed_record(
        {
            "flag": True,
            "count": 3,
            "measure": 0.5,
            "text": "closed",
            "counts": {"b": 2, "a": 1},
            "measures": {"x": 1, "y": 0.25},
        }
    )

    assert list(record) == ["count", "counts", "flag", "measure", "measures", "text"]
    assert record["flag"] == FlagValue(value=True)
    assert record["count"] == CountValue(value=3)
    assert record["measure"] == MeasureValue(value=0.5)
    assert record["text"] == TextValue(value="closed")
    assert record["counts"] == CountsValue(value={"a": 1, "b": 2})
    assert record["measures"] == MeasuresValue(value={"x": 1.0, "y": 0.25})


def test_a_contributed_table_of_non_numbers_or_an_empty_name_is_refused() -> None:
    with pytest.raises(TypeError, match="not a number"):
        contributed_record({"table": cast("dict[str, int]", {"a": "b"})})

    with pytest.raises(TypeError, match="not a number"):
        contributed_record({"table": {"a": True}})

    with pytest.raises(ValueError, match="empty"):
        contributed_record({"": 1})


def prepared_run(directory: Path) -> tuple[TrainingBlueprint, PreparedTraining]:
    """Prepare the packaged parser run with corruption and noise."""
    run_spec = composed(
        directory,
        "data.datasets.train.dataset.corruption.proportion: quarter",
        "data.datasets.train.dataset.corruption.operators: swap-deletion",
        "data.datasets.train.dataset.noise.grammatical: tenth",
        "data.datasets.train.dataset.noise.ungrammatical: tenth",
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()

    return blueprint, prepare_training(
        blueprint, streams, resolve_device_policy(run_spec)
    )


def residual_key(prepared: PreparedTraining) -> tuple[str, str]:
    """Return the first loss term's tag and the key its residual is logged under."""
    keys = cast("ParserFittingPlan", prepared.plan).step.assembly.residual_keys
    tag = next(iter(keys))

    return tag, keys[tag]


def an_epoch(epoch: int, key: str, residuals: tuple[float, ...]) -> EpochReport:
    """Return a completed epoch whose steps report the given residuals."""
    return EpochReport(
        epoch,
        tuple(
            TrainResult(
                epoch * len(residuals) + position + 1,
                2,
                metrics={key: residual, "optimizer/stepped": 1.0},
            )
            for position, residual in enumerate(residuals)
        ),
        EvalResult(4, metrics={"loss/grammaticality": 1.0 / (epoch + 1)}),
    )


def test_a_prevalidation_epoch_records_no_validation_metrics() -> None:
    report = EpochReport(
        0,
        (TrainResult(1, 2, metrics={"optimizer/stepped": 1.0}),),
        None,
    )

    recorded = _epoch(report)

    assert recorded.validation_items is None
    assert recorded.validation is None


def a_report(epochs: tuple[EpochReport, ...], *, stopped_early: bool) -> FittingReport:
    """Return a parser fitting report over the given epochs."""
    steps = sum(len(epoch.results) for epoch in epochs)

    return FittingReport(
        epochs,
        (),
        steps,
        2 * steps,
        unit=TrainingUnit.EPOCHS,
        constraint_selection=None,
        best_checkpoint=BestCheckpointId(CheckpointId("best-0")),
        latest_checkpoint=ResumeCheckpointId(CheckpointId("latest-0")),
        stopped_early=stopped_early,
    )


def target_report(
    epochs: tuple[EpochReport, ...],
    *,
    stop: Literal["target-met", "target-not-met"],
    accuracy: float,
) -> FittingReport:
    """Return a parser fitting report closed by its held-out target."""
    steps = sum(len(epoch.results) for epoch in epochs)

    return FittingReport(
        epochs,
        (),
        steps,
        2 * steps,
        unit=TrainingUnit.EPOCHS,
        constraint_selection=None,
        best_checkpoint=BestCheckpointId(CheckpointId("best-0")),
        latest_checkpoint=ResumeCheckpointId(CheckpointId("latest-0")),
        stopped_early=stop == "target-met",
        stop_reason=stop,
        achieved_accuracy=accuracy,
    )


@pytest.fixture(scope="module")
def run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, TrainingBlueprint, PreparedTraining]:
    directory = tmp_path_factory.mktemp("prepared")
    blueprint, prepared = prepared_run(directory)

    return directory, blueprint, prepared


def test_preparation_writes_the_point_and_every_report_it_drew(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    plan = cast("ParserFittingPlan", prepared.plan)
    recorder = MetricsRecorder(
        tmp_path, SweepPoint(json.loads(blueprint.configuration))
    )
    recorder.timed(RunPhase.PREPARATION, 1.5)
    recorder.prepared(blueprint, prepared)
    record = read_metrics(tmp_path)
    owned = prepared.model.role_parameters()
    corpus = plan.corpus

    assert record is not None
    assert (tmp_path / METRICS_NAME).is_file()
    assert record.stage == "prepared"
    assert record.phases == {"preparation": 1.5}
    assert record.peak_memory_bytes is None
    assert set(record.point.roles) == set(prepared.model.modules)
    assert record.point.roles["parser"].kind == "inside-outside"
    assert record.point.roles["controller"].parameters == sum(
        parameter.numel() for parameter in owned["controller"]
    )
    assert record.point.parameter_count == sum(
        parameter.numel() for parameter in prepared.model.named_parameters.values()
    )
    assert record.point.corpus_fingerprint == blueprint.corpus_revision
    assert corpus is not None
    assert record.corpus == corpus_record(corpus)
    assert record.selection is None
    assert record.bucketing is not None
    assert record.bucketing.training.items == plan.items_per_epoch
    assert record.bucketing.training.batches == len(plan.batches)
    assert record.bucketing.training.live == sum(
        len(words) for batch in plan.batches for words in batch.words
    )
    assert record.bucketing.training.capacity == sum(
        len(batch.words) * max(len(words) for words in batch.words)
        for batch in plan.batches
    )
    assert record.bucketing.training.padded == (
        record.bucketing.training.capacity - record.bucketing.training.live
    )
    assert record.bucketing.validation.batches == len(plan.fit_batches)
    assert 0.0 < record.bucketing.training.rate <= 1.0
    assert record.bucketing.training.form_typing is None
    assert record.bucketing.validation.form_typing is None
    assert record.fitting is None
    assert record.residuals == {}
    assert "compile" in record.contributed
    assert record.contributed["compile"]["congruence"] == TextValue(
        value=plan.grammar.grammar.matching.name
    )
    assert record.contributed["compile"]["codec-slots"] == CountValue(
        value=plan.grammar.language.codec.at(
            plan.grammar.language.category_sort
        ).capacity
    )
    assert record.contributed["compile"]["approximation-residual-mass"] == (
        MeasureValue(value=0.75)
    )
    assert record.contributed["compile"]["approximation-overrides"] == TextValue(
        value="{}"
    )


def test_pair_bucketing_counts_each_side_row_in_its_padded_capacity() -> None:
    class PairBatch:
        words = (("one", "two"), ("three",))
        form_typing: tuple[tuple[FormTypingTarget, ...], ...] = ((), ())

        def __len__(self) -> int:
            return 1

    record = _occupancy((PairBatch(),))

    assert record.items == 1
    assert record.live == 3
    assert record.capacity == 4
    assert record.padded == 1


def test_closing_writes_fitting_residuals_contributions_and_peak_memory(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    tag, key = residual_key(prepared)
    recorder = MetricsRecorder(
        tmp_path, SweepPoint(json.loads(blueprint.configuration))
    )
    recorder.prepared(blueprint, prepared)
    recorder.timed(RunPhase.FITTING, 2.0)
    recorder.timed(RunPhase.FITTING, 0.5)
    recorder.fitted(
        a_report((an_epoch(0, key, (0.25, 0.75)),), stopped_early=True),
        ResumePosition(0, 0),
    )
    recorder.contribute("parser", {"cells": 4, "exhausted": False})
    recorder.closed(peak_memory_bytes=1024)
    record = read_metrics(tmp_path)

    assert record is not None
    assert record.stage == "closed"
    assert record.phases == {"fitting": 2.5}
    assert record.peak_memory_bytes == 1024
    assert record.fitting is not None
    assert record.fitting.stop == "early-stopping"
    assert record.fitting.stopped_epoch == 0
    assert record.fitting.best_checkpoint == "best-0"
    assert record.fitting.latest_checkpoint == "latest-0"
    assert [step.metrics[key] for step in record.fitting.epochs[0].steps] == [
        0.25,
        0.75,
    ]
    assert record.fitting.epochs[0].validation == {"loss/grammaticality": 1.0}
    assert record.fitting.epochs[0].validation_items == 4
    assert record.residuals[tag].steps == 2
    assert record.residuals[tag].mean == 0.5
    assert record.residuals[tag].maximum == 0.75
    assert record.residuals[tag].last == 0.75
    assert record.contributed["parser"] == {
        "cells": CountValue(value=4),
        "exhausted": FlagValue(value=False),
    }
    assert "compile" in record.contributed


@pytest.mark.parametrize(
    ("stop", "accuracy"),
    [("target-met", 0.95), ("target-not-met", 0.5)],
)
def test_fitting_metrics_record_the_held_out_target_outcome(
    tmp_path: Path,
    run: tuple[Path, TrainingBlueprint, PreparedTraining],
    stop: Literal["target-met", "target-not-met"],
    accuracy: float,
) -> None:
    _, blueprint, prepared = run
    _, key = residual_key(prepared)
    recorder = MetricsRecorder(
        tmp_path, SweepPoint(json.loads(blueprint.configuration))
    )
    recorder.prepared(blueprint, prepared)
    recorder.fitted(
        target_report((an_epoch(0, key, (0.25,)),), stop=stop, accuracy=accuracy),
        ResumePosition(0, 0),
    )
    recorder.closed(peak_memory_bytes=None)
    record = read_metrics(tmp_path)

    assert record is not None
    assert record.fitting is not None
    assert record.fitting.stop == stop
    assert record.fitting.achieved_accuracy == accuracy


def test_a_report_contributed_twice_under_one_name_is_refused(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    recorder = MetricsRecorder(
        tmp_path, SweepPoint(json.loads(blueprint.configuration))
    )
    recorder.prepared(blueprint, prepared)
    recorder.contribute("grounding", {"variant": "pairwise"})

    with pytest.raises(ValueError, match="already contributed"):
        recorder.contribute("grounding", {"variant": "pairwise"})


def test_closing_before_preparation_is_recorded_is_refused(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path, SweepPoint({"run": {"name": "unprepared"}}))

    with pytest.raises(RuntimeError, match="preparation"):
        recorder.closed(peak_memory_bytes=None)

    assert read_metrics(tmp_path) is None


def test_a_resumed_attempt_carries_earlier_epochs_and_phases(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    tag, key = residual_key(prepared)
    point = SweepPoint(json.loads(blueprint.configuration))
    first = MetricsRecorder(tmp_path, point)
    first.timed(RunPhase.PREPARATION, 1.0)
    first.prepared(blueprint, prepared)
    first.fitted(
        a_report(
            (an_epoch(0, key, (0.5,)), an_epoch(1, key, (0.25,))), stopped_early=False
        ),
        ResumePosition(0, 0),
    )
    first.timed(RunPhase.FITTING, 3.0)
    first.closed(peak_memory_bytes=None)

    second = MetricsRecorder(tmp_path, point)
    second.timed(RunPhase.PREPARATION, 1.0)
    second.prepared(blueprint, prepared)
    second.fitted(
        a_report((an_epoch(1, key, (0.125,)),), stopped_early=False),
        ResumePosition(1, 2),
    )
    second.timed(RunPhase.FITTING, 1.0)
    second.closed(peak_memory_bytes=None)
    record = read_metrics(tmp_path)

    assert record is not None
    assert record.phases == {"fitting": 4.0, "preparation": 2.0}
    assert record.fitting is not None
    assert [epoch.epoch for epoch in record.fitting.epochs] == [0, 1]
    assert [
        step.metrics[key] for epoch in record.fitting.epochs for step in epoch.steps
    ] == [0.5, 0.125]
    assert record.fitting.stop == "horizon"
    assert record.fitting.stopped_epoch == 1
    assert record.residuals[tag].steps == 2


def test_an_attempt_with_nothing_left_to_fit_keeps_the_earlier_fitting(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    _, key = residual_key(prepared)
    point = SweepPoint(json.loads(blueprint.configuration))
    first = MetricsRecorder(tmp_path, point)
    first.prepared(blueprint, prepared)
    first.fitted(
        a_report((an_epoch(0, key, (0.5,)),), stopped_early=False), ResumePosition(0, 0)
    )
    first.closed(peak_memory_bytes=None)
    earlier = read_metrics(tmp_path)

    second = MetricsRecorder(tmp_path, point)
    second.prepared(blueprint, prepared)
    second.closed(peak_memory_bytes=None)
    record = read_metrics(tmp_path)

    assert earlier is not None
    assert record is not None
    assert record.fitting == earlier.fitting


def test_resuming_distance_steps_keeps_only_the_steps_before_the_checkpoint(
    tmp_path: Path, run: tuple[Path, TrainingBlueprint, PreparedTraining]
) -> None:
    _, blueprint, prepared = run
    point = SweepPoint(json.loads(blueprint.configuration))

    def stepped(step: int, *, closes: bool) -> TrainResult:
        return TrainResult(step, 1, metrics={"optimizer/stepped": float(closes)})

    def report(results: tuple[TrainResult, ...], steps: int) -> FittingReport:
        return FittingReport(
            (),
            results,
            steps,
            steps,
            unit=TrainingUnit.STEPS,
            constraint_selection=None,
            best_checkpoint=None,
            latest_checkpoint=ResumeCheckpointId(CheckpointId("latest")),
            stopped_early=False,
        )

    first = MetricsRecorder(tmp_path, point)
    first.prepared(blueprint, prepared)
    first.fitted(
        report(
            (
                stepped(0, closes=False),
                stepped(1, closes=True),
                stepped(1, closes=False),
                stepped(2, closes=True),
                stepped(2, closes=False),
            ),
            2,
        ),
        ResumePosition(0, 0),
    )
    first.closed(peak_memory_bytes=None)

    second = MetricsRecorder(tmp_path, point)
    second.prepared(blueprint, prepared)
    second.fitted(
        report((stepped(1, closes=False), stepped(2, closes=True)), 2),
        ResumePosition(1, 2),
    )
    second.closed(peak_memory_bytes=None)
    record = read_metrics(tmp_path)

    assert record is not None
    assert record.fitting is not None
    assert record.fitting.stopped_epoch is None
    assert record.fitting.best_checkpoint is None
    assert [
        (step.step, step.metrics["optimizer/stepped"]) for step in record.fitting.steps
    ] == [(0, 0.0), (1, 1.0), (1, 0.0), (2, 1.0)]


def a_record(point_id_seed: int = 0) -> MetricsRecord:
    """Return a closed metrics record over a two-role model and a drawn corpus."""
    run_spec = PACKAGED_CODEC(CONFIG)
    planned = plan_training(run_spec)
    corpus = cast("SyntheticDataset", planned.lower_dataset("train")).corpus
    parser = nn.Linear(2, 3)

    return MetricsRecord(
        stage="closed",
        point=measure_point(
            SweepPoint({"run": {"name": "record"}}, seed=point_id_seed),
            {"parser": MeasuredRole("inside-outside", tuple(parser.parameters()))},
            corpus_fingerprint="corpus:a",
        ),
        phases={"fitting": 1.0},
        peak_memory_bytes=None,
        corpus=corpus_record(corpus),
        inference=None,
        selection=None,
        bucketing=None,
        residuals={},
        fitting=None,
        contributed={"grounding": {"variant": TextValue(value="pairwise")}},
    )


def test_a_written_record_reads_back_equal(tmp_path: Path) -> None:
    record = a_record()
    write_metrics(tmp_path / "run", record)

    assert read_metrics(tmp_path / "run") == record
    assert read_metrics(tmp_path) is None


def test_a_file_that_holds_no_record_is_refused(tmp_path: Path) -> None:
    (tmp_path / METRICS_NAME).write_text("{}", encoding="utf-8", newline="\n")

    with pytest.raises(dx.ValidationError):
        read_metrics(tmp_path)


def test_a_record_flattens_to_its_scalars_by_path() -> None:
    record = a_record()
    row = metrics_row(record)
    corpus = record.corpus

    assert corpus is not None
    assert row["stage"] == "closed"
    assert row["point/parameter_count"] == 9
    assert row["point/roles/parser/parameters"] == 9
    assert row["point/roles/parser/kind"] == "inside-outside"
    assert row["phases/fitting"] == 1.0
    assert row["peak_memory_bytes"] is None
    assert row["corpus/length/resampled"] == corpus.length.resampled
    assert row["corpus/lexicon/coupling"] == corpus.lexicon.coupling
    assert row["corpus/lexicon/realized_vocabulary_size"] == (
        corpus.lexicon.realized_vocabulary_size
    )
    assert row["corpus/inventory/realized_primitive_count"] == (
        corpus.inventory.realized_primitive_count
    )
    assert row["contributed/grounding/variant/value"] == "pairwise"
    assert "corpus/inventory/primitives" not in row
    assert all(not isinstance(value, list | dict) for value in row.values())
