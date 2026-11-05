"""The declared chart checkpointing width on a long-sentence point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import RunSpecCodec, config_search_path
from lofi.train import ChartForward, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from lofi.spec import RunSpec

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
LENGTH = 16


def _composed(directory: Path, *overrides: str) -> RunSpec:
    """Compose the packaged parser run over long sentences and a chart-only loss."""
    document = directory / "long.yaml"
    document.write_text(
        """defaults:
  - synthetic-parser
loss:
  terms:
    - objective:
        kind: grammaticality
        readout: sentences
""",
        encoding="utf-8",
        newline="\n",
    )

    return RunSpecCodec(search_path=(CONFIG.parent, *config_search_path(directory)))(
        document,
        overrides=(
            f"data.datasets.train.dataset.length.min_length={LENGTH}",
            f"data.datasets.train.dataset.length.max_length={LENGTH}",
            "trainer.accumulation_steps=1",
            "model.controller.hidden=32",
            *overrides,
        ),
    )


def _step(run_spec: RunSpec, /) -> tuple[float, int, dict[str, torch.Tensor], int]:
    """Return one batch's loss, the peak bytes held for backward and the gradients."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    prepared = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    plan = cast("ParserFittingPlan", prepared.plan)
    batch = plan.batches[0]
    held = 0
    peak = 0

    assert isinstance(plan.forward, ChartForward)
    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal held, peak
        held += tensor.numel() * tensor.element_size()
        peak = max(peak, held)

        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        total = plan.step.assembly.evaluate(plan.forward.publish(batch), step=0).total

    total.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in prepared.model.named_parameters.items()
        if parameter.grad is not None
    }

    return total.item(), peak, gradients, max(len(words) for words in batch.words)


@pytest.mark.integration
def test_the_checkpointing_policy_lowers_peak_memory_at_equal_outputs(
    tmp_path: Path,
) -> None:
    held_loss, held_peak, held_gradients, length = _step(_composed(tmp_path))
    loss, peak, gradients, _ = _step(
        _composed(tmp_path, "device_policy.checkpoint_from_width=2")
    )

    assert length == LENGTH
    assert peak < held_peak
    assert loss == held_loss
    assert gradients.keys() == held_gradients.keys()
    assert all(
        torch.allclose(gradients[name], held_gradients[name], rtol=0.0, atol=1e-6)
        for name in gradients
    )


@pytest.mark.integration
def test_a_later_width_checkpoints_less_of_the_chart(tmp_path: Path) -> None:
    peaks = [
        _step(_composed(tmp_path, f"device_policy.checkpoint_from_width={width}"))[1]
        for width in (LENGTH, LENGTH // 2, 2)
    ]

    assert peaks[0] > peaks[1] > peaks[2]
