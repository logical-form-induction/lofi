from typing import TYPE_CHECKING, Final

from tests.acceptance.driver import composed

from lofi import train
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.run._boundary.manifest import _training_checkpoint_manifest
from lofi.run.held_out import evaluate_held_out
from lofi.run.operations import CHECKPOINT_RECORDS
from lofi.spec import Dependency

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.eval import EvaluationArtifact, Report

SEED: Final = 17


class _RecordingSink:
    def __init__(self) -> None:
        self.reports: list[Report] = []

    def emit(self, report: Report, /) -> None:
        self.reports.append(report)

    def emit_artifact(self, role: str, artifact: EvaluationArtifact, /) -> None:
        del role, artifact


def test_a_logical_form_point_reports_every_declared_category_row(
    tmp_path: Path,
) -> None:
    run_spec = composed("lf-category-metrics.yaml")
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
    sink = _RecordingSink()

    reports = evaluate_held_out(
        prepared,
        run_spec,
        freeze_id="freeze/1",
        seed=SEED,
        sink=sink,
    )

    assert tuple(sink.reports) == reports
    assert len(reports) == 3
    by_role = {report.rows[0].task: report for report in reports}
    category_rows = by_role["category-side"].rows
    assert run_spec.evaluators["category-side"].dependencies() == (
        Dependency("category-terms", "terms-readout"),
    )
    assert tuple(row.metric for row in category_rows) == tuple(
        run_spec.evaluators["category-side"].metrics()
    )
    assert len({row.convention_hash for row in category_rows}) == 1
    assert run_spec.evaluators["expert"].dependencies() == (
        Dependency("category-terms", "terms-readout"),
    )
    assert tuple(row.metric for row in by_role["expert"].rows) == (
        "expert-rating-alpha",
        "expert-error-wrong-argument-direction",
        "expert-error-wrong-arity",
        "expert-error-wrong-result",
    )
    assert tuple(row.metric for row in by_role["logical-form"].rows) == (
        "logical-form-exact",
        "one-to-one-accuracy",
        "random-relabeling-floor",
        "fiber-relative-recovery",
        "fixed-vocabulary-accuracy",
        "fiber-floor",
        "fiber-size",
    )
    assert all(
        row.value is not None
        for row in category_rows
        if not row.metric.startswith("constructor-")
    )
