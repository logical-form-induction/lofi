from pathlib import Path

from lofi.spec import RunSpecCodec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.training_plugins import TrainingLength, TrainingUnit
from lofi.training_plugins.distance import PLUGIN

CONFIG = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-distance.yaml"
)


def test_the_distance_plugin_checks_and_names_its_sampler_stream() -> None:
    run_spec = RunSpecCodec()(CONFIG)

    assert PLUGIN.spec is DistanceTrainerSpec
    assert PLUGIN.plan(run_spec.trainer, run_spec) == (
        "training:train:distance-batches",
    )


def test_the_distance_plugin_prices_a_run_at_its_declared_steps() -> None:
    trainer = RunSpecCodec()(CONFIG).trainer

    assert isinstance(trainer, DistanceTrainerSpec)
    assert PLUGIN.price(trainer.with_(steps=9)) == TrainingLength(9, TrainingUnit.STEPS)
