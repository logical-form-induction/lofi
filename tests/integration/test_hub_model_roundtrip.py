"""Installed-boundary round trip for one complete offline model bundle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from lofi.data import LicenseRef, Licensing, Provenance, Reproducibility
from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.run import ExperimentSeed, RandomStreams, plan_training, resolve_device_policy
from lofi.run.hub import ModelRelease, export_hub_model
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpec, RunSpecCodec
from lofi.train import ChartForward, ParserFittingPlan, prepare_training

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.train import PreparedTraining

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
DERIVATION = ExperimentSeed(317).point("integration-hub", 0)


def _prediction_digest(prepared: PreparedTraining, /) -> tuple[int, str]:
    """Return the item count and exact serialized parser scores."""
    plan = prepared.plan
    assert isinstance(plan, ParserFittingPlan)
    assert isinstance(plan.forward, ChartForward)

    for module in prepared.model.modules.values():
        module.eval()

    scores: list[float] = []

    with torch.no_grad():
        for batch in (*plan.batches, *plan.fit_batches):
            assert isinstance(batch, SyntheticTrainingBatch)
            published = plan.forward.publish(batch)
            assert published.scores is not None
            scores.extend(published.scores.tensor.detach().cpu().tolist())

    payload = json.dumps(scores, allow_nan=False, separators=(",", ":")).encode()

    return len(scores), sha256(payload).hexdigest()


def _twenty_item_run() -> RunSpec:
    """Return the packaged parser run with exactly twenty fixed occurrences."""
    document = cast(
        "dict[str, JsonValue]",
        json.loads(CODEC(CONFIG).model_dump_json()),
    )
    data = cast("dict[str, JsonValue]", document["data"])
    datasets = cast("dict[str, JsonValue]", data["datasets"])
    train = cast("dict[str, JsonValue]", datasets["train"])
    dataset = cast("dict[str, JsonValue]", train["dataset"])
    dataset["size"] = 20

    return RunSpec.model_validate(document)


def test_a_model_bundle_round_trips_twenty_predictions_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    run_spec = _twenty_item_run()
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    baseline = _prediction_digest(prepared)

    with torch.no_grad():
        for position, parameter in enumerate(prepared.model.parameters, start=1):
            update = torch.linspace(
                -0.25 * position,
                0.25 * position,
                parameter.numel(),
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape_as(parameter)
            parameter.add_(update)

    release = ModelRelease(
        "lofi-twenty-item-parser",
        ("lofi/fixed-synthetic-20",),
        ("en",),
        Provenance(
            "generated",
            Licensing(
                "CC-BY-4.0",
                (LicenseRef("CC-BY-4.0", "fixed-synthetic-20"),),
            ),
            (),
            Reproducibility(
                "https://github.com/logical-form-induction/lofi",
                "0123456789abcdef",
                "pytest --run-integration",
                "uv",
                DERIVATION.root,
            ),
        ),
        grammar_identity=blueprint.grammar_fingerprint,
        codec_identity=blueprint.codec_fingerprint,
        action_set_identity=blueprint.action_set_fingerprint,
        corpus_revision=blueprint.corpus_revision,
        pretrained_resources={},
    )
    bundle = tmp_path / "bundle"
    export_hub_model(bundle, release, blueprint, prepared, DERIVATION)
    expected = _prediction_digest(prepared)
    assert expected != baseline
    isolated = tmp_path / "isolated"
    cache = tmp_path / "empty-cache"
    isolated.mkdir()
    cache.mkdir()
    script = r"""
import json
import socket
import sys
from hashlib import sha256
from pathlib import Path

import torch

from lofi.run.hub import load_hub_bundle
from lofi.train import ParserFittingPlan

def refuse(*args, **kwargs):
    del args, kwargs
    raise AssertionError("loading an offline bundle opened a socket")

socket.socket = refuse
socket.create_connection = refuse
bundle = Path(sys.argv[1])
loaded = load_hub_bundle(bundle)
assert loaded.release.model_name == "lofi-twenty-item-parser"
assert loaded.release.provenance.source_method == "generated"
plan = loaded.prepared.plan
assert isinstance(plan, ParserFittingPlan)
for module in loaded.prepared.model.modules.values():
    module.eval()
scores = []
with torch.no_grad():
    retained = tuple(
        batch for batches in plan.retained_batches.values() for batch in batches
    )
    for batch in (*plan.batches, *retained):
        published = plan.forward.publish(batch)
        assert published.scores is not None
        scores.extend(published.scores.tensor.detach().cpu().tolist())
payload = json.dumps(scores, allow_nan=False, separators=(",", ":")).encode()
print(json.dumps({"count": len(scores), "digest": sha256(payload).hexdigest()}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_CACHE": str(cache),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-", str(bundle)],
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=isolated,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    found = json.loads(completed.stdout)
    assert expected[0] == found["count"] == 20
    assert expected[1] == found["digest"]
