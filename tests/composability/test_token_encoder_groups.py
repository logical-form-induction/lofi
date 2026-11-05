"""Every packaged token-encoder option trains the packaged parser run.

The options that load pretrained weights read them through a stand-in loader, a
small randomly initialized stack over the run's own vocabulary, so no checkpoint
is downloaded and each option's declaration is still what selects the stack.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.pretrained_stand_ins import StandInLoader

from lofi.data.synthetic.training import SyntheticTrainingBatch
from lofi.nn.tokens.pretrained import (
    ENCODER_ONLY,
    T5_ENCODER,
    EncoderArchitecturePlugin,
)
from lofi.nn.tokens.window import ContextOverflowError
from lofi.run import (
    RandomStreams,
    SeedPolicy,
    packaged_options,
    plan_training,
    resolve_device_policy,
)
from lofi.spec import RunSpecCodec, config_search_path
from lofi.train import Featurized, ParserFittingPlan, ParserStep, prepare_training

if TYPE_CHECKING:
    from lofi.nn.tokens.pretrained import PretrainedLoader, PretrainedTokenEncoder
    from lofi.spec import RunSpec
    from lofi.train import PreparedTraining

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
SLOT: Final = "model.token_encoder"
WORDS: Final = tuple(f"word-{index}" for index in range(8))
PRETRAINED: Final = (
    "eurobert",
    "llm2vec",
    "mmbert",
    "modernbert",
    "mt5",
    "pretrained",
    "xlm-roberta",
)

# the static option names vectors outside the repository, so its run reads the
# vectors the packaged run ships
OVERRIDES: Final = {
    "static": (f"{SLOT}.vectors=lofi.resource:run/synthetic-parser.vec",),
}


def _composed(
    directory: Path, option: str, *overrides: str, loss: str | None = None
) -> RunSpec:
    """Compose the packaged parser run over one packaged token-encoder option.

    Parameters
    ----------
    directory : Path
        Where the composing document is written.
    option : str
        The packaged token-encoder option selected.
    *overrides : str
        Dotted overrides applied after the selections.
    loss : str | None
        A packaged loss option replacing the run's own, or None to keep it.
    """
    document = directory / f"{option}.yaml"
    selected = "" if loss is None else f"  - loss: {loss}\n"
    cache = (
        f"artifacts:\n  root: {directory / 'artifacts'}\n"
        if option in PRETRAINED
        else ""
    )
    document.write_text(
        f"defaults:\n  - synthetic-parser\n  - {SLOT}: {option}\n{selected}{cache}",
        encoding="utf-8",
        newline="\n",
    )

    return RunSpecCodec(search_path=(CONFIG.parent, *config_search_path(directory)))(
        document,
        overrides=(
            "trainer.accumulation_steps=1",
            *OVERRIDES.get(option, ()),
            *overrides,
        ),
    )


def _prepared(run_spec: RunSpec, *, pretrained: bool) -> PreparedTraining:
    """Prepare one composed run, its pretrained weights read through a stand-in."""
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(13, samplers=blueprint.sampler_streams))
    streams.apply()
    resources: dict[str, PretrainedLoader] = (
        {"token_encoder": StandInLoader(WORDS, width=6, seed=3)} if pretrained else {}
    )

    return prepare_training(
        blueprint,
        streams,
        resolve_device_policy(run_spec),
        pretrained_resources=resources,
    )


def test_the_packaged_token_encoder_options_are_the_ones_trained_here() -> None:
    assert set(packaged_options(SLOT)) == {"learned", "static", *PRETRAINED}


@pytest.mark.parametrize("option", packaged_options(SLOT))
def test_every_packaged_token_encoder_option_trains_a_parser_step(
    tmp_path: Path, option: str
) -> None:
    prepared = _prepared(_composed(tmp_path, option), pretrained=option in PRETRAINED)
    plan = cast("ParserFittingPlan", prepared.plan)
    parameters = prepared.model.parameters
    before = tuple(parameter.detach().clone() for parameter in parameters)
    batch = plan.batches[0]

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    result = plan.step.run(Featurized(batch, 0)).result

    assert result.step == 1
    assert result.metrics["optimizer/loss_finite"] == 1.0
    assert result.metrics["tokens/truncated_items"] == 0.0
    assert all(math.isfinite(value) for value in result.metrics.values())
    assert any(
        not torch.equal(initial, trained)
        for initial, trained in zip(before, parameters, strict=True)
    )


@pytest.mark.parametrize(
    ("option", "architecture"),
    [
        ("mt5", T5_ENCODER),
        ("xlm-roberta", ENCODER_ONLY),
    ],
)
def test_a_packaged_option_builds_the_stack_its_document_declares(
    tmp_path: Path, option: str, architecture: EncoderArchitecturePlugin
) -> None:
    prepared = _prepared(_composed(tmp_path, option), pretrained=True)
    encoder = cast("PretrainedTokenEncoder", prepared.model.modules["token_encoder"])

    assert encoder.architecture is architecture
    assert all(
        parameter.dtype is prepared.model.policy.dtype
        for parameter in encoder.parameters()
    )


def test_an_item_overflowing_the_declared_window_is_reported_in_the_step_metrics(
    tmp_path: Path,
) -> None:
    run_spec = _composed(
        tmp_path,
        "mt5",
        f"{SLOT}.window.rows=2",
        "trainer.early_stopping.metric=loss/chart-scale",
        loss="chart-scale",
    )
    prepared = _prepared(run_spec, pretrained=True)
    plan = cast("ParserFittingPlan", prepared.plan)
    batch = plan.batches[0]
    overflowing = sum(1 for words in batch.words if len(words) > 2)

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    result = plan.step.run(Featurized(batch, 0)).result

    assert overflowing > 0
    assert result.metrics["tokens/truncated_items"] == float(overflowing)
    assert result.metrics["optimizer/loss_finite"] == 1.0


def test_a_window_declared_to_refuse_overflow_stops_the_step(tmp_path: Path) -> None:
    run_spec = _composed(
        tmp_path, "mt5", f"{SLOT}.window.rows=2", f"{SLOT}.window.overflow=refuse"
    )
    prepared = _prepared(run_spec, pretrained=True)
    plan = cast("ParserFittingPlan", prepared.plan)
    batch = next(batch for batch in plan.batches if max(map(len, batch.words)) > 2)

    assert isinstance(plan.step, ParserStep)
    assert isinstance(batch, SyntheticTrainingBatch)

    with pytest.raises(ContextOverflowError, match="the window holds 2"):
        plan.step.run(Featurized(batch, 0))
