"""Tests for the alternating-trainer plugin."""

from pathlib import Path

import pytest

from lofi.data import Relational
from lofi.mixing import RoundRobinMixingSpec
from lofi.spec import RunSpec, RunSpecCodec
from lofi.spec.trainers.composed import AlternatingTrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.spec.trainers.steps import GroundedStepSpec, ParserStepSpec
from lofi.training_plugins import (
    FeaturizerMismatchError,
    TrainingLength,
    TrainingUnit,
)
from lofi.training_plugins.alternating import PLUGIN

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


def declaration() -> tuple[AlternatingTrainerSpec, RunSpec]:
    """Return an alternating declaration beside its complete run declaration."""
    run_spec = PACKAGED_CODEC(CONFIG)
    trainer = run_spec.trainer
    assert isinstance(trainer, ParserTrainerSpec)
    base = trainer.with_(accumulation_steps=1)
    declared = AlternatingTrainerSpec(
        base=base,
        tasks=(
            ParserStepSpec(name="first", dataset="train"),
            ParserStepSpec(name="second", dataset="train"),
        ),
        mixing=RoundRobinMixingSpec(),
    )
    return declared, run_spec


def test_the_alternating_plugin_reaches_every_task_sampler() -> None:
    declared, run_spec = declaration()

    assert PLUGIN.spec is AlternatingTrainerSpec
    assert PLUGIN.plan(declared, run_spec) == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_the_alternating_plugin_prices_the_shared_epoch_horizon() -> None:
    declared, _ = declaration()

    assert PLUGIN.price(declared) == TrainingLength(
        declared.base.epochs,
        TrainingUnit.EPOCHS,
    )


def test_a_task_presenting_pairs_to_a_sentence_featurizer_is_refused() -> None:
    declared, run_spec = declaration()
    mixed = declared.with_(
        tasks=(
            ParserStepSpec(name="sentences", dataset="train"),
            GroundedStepSpec(name="pairs", dataset="train"),
        )
    )

    with pytest.raises(FeaturizerMismatchError) as refused:
        PLUGIN.plan(mixed, run_spec)

    assert refused.value.trainer == "alternating"
    assert refused.value.presented is Relational
