"""Executable known-grammar supertagging conditions."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi import train
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    TrainingBlueprint,
    plan_training,
    resolve_device_policy,
)
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.held_out import evaluate_held_out
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import RunSpecCodec, config_search_path

if TYPE_CHECKING:
    from lofi.eval import EvaluationArtifact, Report
    from lofi.spec import RunSpec
    from lofi.train import ParserFittingPlan

ROOT: Final = Path(__file__).resolve().parents[2]
PROJECT: Final = ROOT / "tests/integration/fixtures/supertag-conditions"
S1: Final = ROOT / "experiments/s1"
TREATMENTS: Final = (
    "induction",
    "tag-dictionary",
    "fully-supervised",
    "prange",
    "inventory-seeded",
    "inventory-seeded-tag-dictionary",
    "inventory-seeded-fully-supervised",
    "inventory-seeded-prange",
)
SEED: Final = 17


class _RecordingSink:
    """Retain emitted reports while discarding side artifacts."""

    def __init__(self) -> None:
        self.reports: list[Report] = []

    def emit(self, report: Report, /) -> None:
        self.reports.append(report)

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        del role, artifact


def _declared(treatment: str, /) -> RunSpec:
    """Compose one packaged condition through the public document surface."""
    codec = RunSpecCodec(search_path=(S1, *config_search_path(PROJECT)))
    return codec(PROJECT / f"{treatment}.yaml")


def _prepared(
    run_spec: RunSpec, /
) -> tuple[train.PreparedTraining, RandomStreams, TrainingBlueprint]:
    """Return one seeded condition preparation and its blueprint."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(SEED, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = train.prepare_training(
        blueprint, streams, resolve_device_policy(run_spec)
    )
    return prepared, streams, blueprint


@pytest.mark.integration
@pytest.mark.parametrize("treatment", TREATMENTS)
def test_a_condition_trains_and_scores_the_section_zero_fixture(
    treatment: str,
    tmp_path: Path,
) -> None:
    run_spec = _declared(treatment)
    prepared, streams, blueprint = _prepared(run_spec)
    plan = cast("ParserFittingPlan", prepared.plan)
    retained = tuple(
        item_id
        for batch in plan.retained_batches["validation"]
        for item_id in batch.item_ids
    )
    stages = tuple(
        staged[2]
        for epoch in range(plan.epochs)
        if (staged := plan.stage_at(epoch)) is not None
    )
    expected_steps = sum(task.optimizer_steps for task in stages)

    assert retained
    assert all(str(item_id).startswith("wsj_0000.") for item_id in retained)
    assert all(task.batches > 0 for task in stages)
    if treatment in {"induction", "inventory-seeded"}:
        lexical = next(task for task in stages if task.name == "lexical-constraints")
        assert lexical.optimizer_steps == 0

    store = train.DirectoryStore(tmp_path / "checkpoints")
    template = _training_checkpoint_manifest(
        blueprint,
        prepared,
        streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    fitted = train.fit_training(
        prepared,
        store,
        template,
        streams,
        train.ResumePosition(0, 0),
        sinks=(),
    )
    sink = _RecordingSink()
    reports = evaluate_held_out(
        prepared,
        run_spec,
        freeze_id="freeze/1",
        seed=SEED,
        sink=sink,
    )

    assert fitted.optimizer_steps == expected_steps
    assert tuple(sink.reports) == reports
    assert {report.rows[0].task for report in reports} == {
        "lexical",
        "structural",
        "types",
    }
    assert all(row.split == "validation" for report in reports for row in report.rows)
    assert all(
        len({row.convention_hash for row in report.rows}) == 1 for report in reports
    )
    assert all(row.convention_hash for report in reports for row in report.rows)


@pytest.mark.integration
def test_a_zero_step_lexical_stage_resumes_at_its_epoch_boundary(
    tmp_path: Path,
) -> None:
    codec = RunSpecCodec(search_path=(S1, *config_search_path(PROJECT)))
    run_spec = codec(
        PROJECT / "induction.yaml",
        overrides=("trainer.base.checkpoints.keep=2",),
    )
    prepared, streams, blueprint = _prepared(run_spec)
    store = train.DirectoryStore(tmp_path / "checkpoints")
    template = _training_checkpoint_manifest(
        blueprint,
        prepared,
        streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    first_fit = train.fit_training(
        prepared,
        store,
        template,
        streams,
        train.ResumePosition(0, 0),
        sinks=(),
    )
    assert first_fit.epochs[0].validation is None
    boundary = next(
        handle
        for handle in store.list()
        if (fitting := store.manifest(handle).fitting) is not None
        and fitting.kind == "resume-latest"
        and fitting.completed_epochs == 1
    )
    boundary_fitting = store.manifest(boundary).fitting
    assert boundary_fitting is not None
    assert boundary_fitting.best is None
    assert boundary_fitting.best_checkpoint is None
    recovered, recovered_streams, recovered_blueprint = _prepared(run_spec)
    recovered_template = _training_checkpoint_manifest(
        recovered_blueprint,
        recovered,
        recovered_streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    checkpoint = store.load(boundary, recovered_template)
    position = train.resume_from_checkpoint(
        checkpoint,
        recovered_template,
        recovered.model.prepare_restore,
        recovered.optimization.prepare_restore,
        recovered_streams,
    )
    continued = train.fit_training(
        recovered,
        store,
        checkpoint.manifest,
        recovered_streams,
        position,
        sinks=(),
    )

    assert position == train.ResumePosition(0, 0)
    assert continued.optimizer_steps == 1
    final = store.manifest(continued.latest_checkpoint)
    assert not train.fitting_remains(
        recovered,
        final,
        recovered_streams,
        train.ResumePosition(final.step, final.data_position),
    )


@pytest.mark.integration
def test_the_cfg_control_fits_short_strings_and_scores_deeper_ones(
    tmp_path: Path,
) -> None:
    run_spec = _declared("cfg-collapse")
    prepared, streams, blueprint = _prepared(run_spec)
    plan = cast("ParserFittingPlan", prepared.plan)
    corpus = plan.corpus

    assert corpus is not None
    assert corpus.shape is not None
    achieved: dict[str, set[int]] = {}
    for example, measurement in zip(
        corpus.examples, corpus.shape.measurements, strict=True
    ):
        split = example.item.split
        if split is not None:
            achieved.setdefault(split, set()).add(measurement.depth)
    assert achieved == {"train": {2}, "validation": {4}}
    assert {len(words) for batch in plan.batches for words in batch.words} == {2}
    assert {
        len(words)
        for batch in plan.retained_batches["validation"]
        for words in batch.words
    } == {4}

    store = train.DirectoryStore(tmp_path / "checkpoints")
    template = _training_checkpoint_manifest(
        blueprint,
        prepared,
        streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    fitted = train.fit_training(
        prepared,
        store,
        template,
        streams,
        train.ResumePosition(0, 0),
        sinks=(),
    )
    reports = evaluate_held_out(
        prepared,
        run_spec,
        freeze_id="freeze/1",
        seed=SEED,
        sink=_RecordingSink(),
    )

    assert fitted.optimizer_steps > 0
    assert {row.task for report in reports for row in report.rows} == {"lexical"}
    assert all(row.split == "validation" for report in reports for row in report.rows)
    assert all(row.convention_hash for report in reports for row in report.rows)
