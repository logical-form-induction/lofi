import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec, config_search_path
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from lofi.nn.parsers import DeepInsideOutside
    from lofi.nn.ports import Controller
    from lofi.spec import RunSpec
    from lofi.train import PreparedTraining

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))


def _paired(
    directory: Path,
    encoder: str,
    identity: str,
    application: str,
) -> RunSpec:
    """Compose the packaged parser run over packaged component fragments."""
    document = directory / f"{encoder}-{identity}-{application}.yaml"
    document.write_text(
        "defaults:\n"
        "  - synthetic-parser\n"
        f"  - model.encoders.terms: {encoder}\n"
        f"  - model.decoders.identity: {identity}\n"
        f"  - model.decoders.application: {application}\n",
        encoding="utf-8",
        newline="\n",
    )

    extra: dict[str, int] = {
        "trainer.accumulation_steps": 1,
        "model.encoders.terms.width": 4,
    }
    if encoder == "transformer":
        extra["model.encoders.terms.heads"] = 1

    return RunSpecCodec(search_path=(CONFIG.parent, *config_search_path(directory)))(
        document, extra=extra
    )


def _prepared(run_spec: RunSpec) -> PreparedTraining:
    """Prepare one composed run on its declared device."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(blueprint, streams, resolve_device_policy(run_spec))


def test_packaged_declaration_composes_and_prepares_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_spec = PACKAGED_CODEC(CONFIG)
    policy = resolve_device_policy(run_spec)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, policy)
    plan = prepared.plan
    controller = cast("Controller", prepared.model.modules["controller"])
    parser = cast("DeepInsideOutside", prepared.model.modules["parser"])

    assert isinstance(plan, ParserFittingPlan)
    assert prepared.model.policy is policy
    assert controller.action_set is parser.rules.action_set
    accumulation = cast("int", run_spec.trainer.accumulation_steps)

    assert plan.steps_per_epoch * accumulation == len(plan.batches)


@pytest.mark.parametrize(
    ("encoder", "identity", "application"),
    [
        ("rnn", "rnn", "rnn"),
        ("hrr", "rnn", "rnn"),
        ("linear", "rnn", "rnn"),
        ("transformer", "rnn", "rnn"),
    ],
)
def test_every_shipped_encoder_trains_with_a_frontier_decoder_from_fragments(
    tmp_path: Path,
    encoder: str,
    identity: str,
    application: str,
) -> None:
    prepared = _prepared(_paired(tmp_path, encoder, identity, application))
    plan = cast("ParserFittingPlan", prepared.plan)
    decoder = prepared.model.modules["application"]
    before = tuple(parameter.detach().clone() for parameter in decoder.parameters())
    batch = plan.batches[0]

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    result = plan.step.run(Featurized(batch, 0)).result

    assert result.step == 1
    assert result.metrics["optimizer/loss_finite"] == 1.0
    assert all(math.isfinite(value) for value in result.metrics.values())
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(before, decoder.parameters(), strict=True)
    )
