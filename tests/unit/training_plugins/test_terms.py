from pathlib import Path

import pytest

from lofi.spec import RunSpecCodec
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec
from lofi.spec.trainers.terms import TermsTrainerSpec
from lofi.training_plugins import TrainingLength, TrainingUnit
from lofi.training_plugins.terms import PLUGIN

CONFIG = (
    Path(__file__).resolve().parents[2] / "acceptance/documents/rule-pair-unary.yaml"
)


def test_the_terms_plugin_checks_and_names_its_sampler_stream() -> None:
    run_spec = RunSpecCodec()(CONFIG)

    assert PLUGIN.spec is TermsTrainerSpec
    assert PLUGIN.plan(run_spec.trainer, run_spec) == ("training:train:terms-batches",)


def test_the_terms_plugin_refuses_an_undeclared_space() -> None:
    run_spec = RunSpecCodec()(CONFIG)
    trainer = run_spec.trainer

    assert isinstance(trainer, TermsTrainerSpec)

    with pytest.raises(ValueError, match=r"undeclared spaces \['missing'\]"):
        PLUGIN.plan(trainer.with_(spaces=("missing",)), run_spec)


def test_the_terms_plugin_prices_a_run_at_its_declared_steps() -> None:
    trainer = RunSpecCodec()(CONFIG).trainer

    assert isinstance(trainer, TermsTrainerSpec)
    assert PLUGIN.price(trainer.with_(steps=9)) == TrainingLength(9, TrainingUnit.STEPS)


def test_an_accuracy_target_adds_a_distinct_held_out_stream_and_sets_the_price() -> (
    None
):
    run_spec = RunSpecCodec()(CONFIG)
    trainer = run_spec.trainer

    assert isinstance(trainer, TermsTrainerSpec)
    targeted = trainer.with_(
        steps=None,
        stop=AccuracyStoppingSpec(
            metric="loss/categories/autoencoding/exact_match",
            target=0.95,
            max_steps=9,
        ),
    )

    assert PLUGIN.plan(targeted, run_spec) == (
        "training:train:terms-batches",
        "training:train:terms-held-out-batches",
    )
    assert PLUGIN.price(targeted) == TrainingLength(9, TrainingUnit.STEPS)


def test_an_unknown_target_metric_is_refused_during_allocation_free_planning() -> None:
    run_spec = RunSpecCodec()(CONFIG)
    trainer = run_spec.trainer

    assert isinstance(trainer, TermsTrainerSpec)
    targeted = trainer.with_(
        steps=None,
        stop=AccuracyStoppingSpec(metric="unknown", target=0.95, max_steps=9),
    )

    with pytest.raises(ValueError, match="not evaluated"):
        PLUGIN.plan(targeted, run_spec)
