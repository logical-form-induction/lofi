from typing import TYPE_CHECKING, Final

from tests.acceptance.driver import composed

from lofi import train
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.held_out import evaluate_held_out
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec.components.data import InferenceLabeledDatasetSpec

SEED: Final = 17

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.eval import EvaluationArtifact, Report


class RecordingSink:
    def __init__(self) -> None:
        self.reports: list[Report] = []

    def emit(self, report: Report, /) -> None:
        self.reports.append(report)

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        del role, artifact


def test_inference_document_reaches_all_declared_report_rows(tmp_path: Path) -> None:
    run_spec = composed("inference-evaluator.yaml")
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(SEED, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = train.prepare_training(
        blueprint, streams, resolve_device_policy(run_spec)
    )
    template = _training_checkpoint_manifest(
        blueprint,
        prepared,
        streams,
        CHECKPOINT_RECORDS,
        step=0,
        data_position=0,
    )
    train.fit_training(
        prepared,
        train.DirectoryStore(tmp_path / "checkpoints"),
        template,
        streams,
        train.ResumePosition(0, 0),
        sinks=(),
    )
    sink = RecordingSink()

    reports = evaluate_held_out(
        prepared,
        run_spec,
        freeze_id="freeze/1",
        seed=SEED,
        sink=sink,
    )

    assert tuple(sink.reports) == reports
    assert len(reports) == 1
    dataset = run_spec.data.datasets["train"].dataset
    evaluator = run_spec.evaluators["inference"]
    assert isinstance(dataset, InferenceLabeledDatasetSpec)
    assert tuple(row.metric for row in reports[0].rows) == tuple(
        evaluator.metrics(dataset.inference_labels())
    )
    assert len(reports[0].rows) == 14
    assert all("micro" not in row.metric for row in reports[0].rows)
    for row in reports[0].rows:
        if row.value is None or row.n < 2:
            assert row.ci_lo is None
            assert row.ci_hi is None
        else:
            assert row.ci_lo is not None
            assert row.ci_hi is not None
