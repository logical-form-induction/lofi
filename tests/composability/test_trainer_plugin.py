"""A trainer installed by a third-party distribution trains from a run document."""

import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_plugin(root: Path, module: str) -> None:
    source = """from typing import Literal

from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.training_plugins import TrainerPlugin
from lofi.training_plugins.parser import PLUGIN as PARSER


class DelegatingTrainerSpec(TrainerSpec):
    kind: Literal["third-party-delegating"] = "third-party-delegating"
    delegate: ParserTrainerSpec

    def refuse_epoch(self, occurrences):
        self.delegate.refuse_epoch(occurrences)


def plan(declaration, run_spec):
    return PARSER.plan(declaration.delegate, run_spec)


def present(declaration, run_spec):
    return PARSER.presenter(declaration.delegate, run_spec)


def build(declaration, context):
    return PARSER.prepare(declaration.delegate, context)


def price(declaration):
    return PARSER.price(declaration.delegate)


PLUGIN = TrainerPlugin(
    spec=DelegatingTrainerSpec,
    plan=plan,
    present=present,
    build=build,
    price=price,
)
"""
    (root / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = root / "lofi_trainer_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lofi-trainer-plugin\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.trainer]\nthird-party-delegating = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )


def test_a_third_party_trainer_selected_by_a_document_trains(tmp_path: Path) -> None:
    module = f"lofi_trainer_plugin_{tmp_path.name.replace('-', '_')}"
    _write_plugin(tmp_path, module)
    document = tmp_path / "run.yaml"
    document.write_text(
        "defaults:\n"
        "  - run/synthetic-parser\n"
        "trainer:\n"
        "  kind: third-party-delegating\n"
        "  delegate:\n"
        "    kind: parser\n"
        "    dataset: train\n"
        "    epochs: 1\n"
        "    batch_size: 2\n"
        "    accumulation_steps: 2\n"
        "    batch_sampler: {kind: length-bucket, bucket_factor: 100, "
        "reshuffle_each_epoch: false}\n"
        "    early_stopping:\n"
        "      metric: loss/grammaticality\n"
        "      mode: min\n"
        "      patience: 2\n"
        "      minimum_delta: 0.0\n"
        "      keep_best: 1\n"
        "    checkpoints:\n"
        "      interval: 1\n"
        "      keep: 1\n",
        encoding="utf-8",
        newline="\n",
    )
    script = r"""import json
import sys
from pathlib import Path

from lofi.run import (
    CHECKPOINTS_NAME,
    MetricsRecorder,
    RandomStreams,
    SeedPolicy,
    Selection,
    SweepPoint,
    operations,
    compose,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import config_search_path, load_plugins
from lofi.train import DirectoryStore

root = Path(sys.argv[1])
loaded = load_plugins()
assert "third-party-delegating" in loaded

run_spec = compose(
    Selection(
        config=root / "run.yaml",
        profile=None,
        overrides=(),
        overlays=(),
        search_path=config_search_path(root),
    )
)
assert type(run_spec.trainer).__name__ == "DelegatingTrainerSpec"

blueprint = plan_training(run_spec)
streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
streams.apply()
operations.TRAINING_OPERATIONS.execute(
    blueprint,
    root / "run",
    streams,
    resolve_device_policy(run_spec),
    scoring=None,
    sinks=(),
    recorder=MetricsRecorder(
        root / "run", SweepPoint(json.loads(blueprint.configuration))
    ),
)
checkpoints = DirectoryStore(root / "run" / CHECKPOINTS_NAME).list()
assert checkpoints, "the third-party trainer saved no checkpoint"
print("third-party trainer trained")
"""
    python_path = os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=script,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": python_path},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "third-party trainer trained\n"
    assert module not in sys.modules
