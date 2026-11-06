import copy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi.data import Unlabeled
from lofi.registry import Registry, RegistryError
from lofi.spec import RunSpecCodec
from lofi.spec.trainers.distance import DistanceTrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.training_plugins import TrainerPlugin, trainer_plugin
from lofi.training_plugins.distance import PLUGIN as DISTANCE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data import SupervisionArm
    from lofi.spec import RunSpec

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
DISTANCE_CONFIG = CONFIG.with_name("synthetic-distance.yaml")
PACKAGED_CODEC = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))


def test_a_trainer_declaration_resolves_to_its_typed_plugin() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    declaration = run_spec.trainer
    plugin = trainer_plugin(declaration)

    assert plugin.spec is ParserTrainerSpec
    assert plugin.plan(declaration, run_spec) == (
        "training:train:validation-split",
        "training:train:training-batches",
        "training:train:validation-batches",
    )


def test_a_trainer_plugin_is_frozen_and_names_its_declaration() -> None:
    plugin = trainer_plugin(PACKAGED_CODEC(CONFIG).trainer)

    assert copy.copy(plugin) is plugin
    assert repr(plugin) == "TrainerPlugin(spec=ParserTrainerSpec)"

    with pytest.raises(AttributeError):
        plugin.__setattr__("_spec", object)


def test_a_trainer_entry_must_name_a_trainer_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PACKAGED_CODEC(CONFIG).trainer
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: object())

    with pytest.raises(RegistryError, match="not a plugin"):
        trainer_plugin(trainer)


def test_a_trainer_entry_must_build_the_declared_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PACKAGED_CODEC(CONFIG).trainer
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: DISTANCE)

    with pytest.raises(
        RegistryError, match="builds DistanceTrainerSpec, not ParserTrainerSpec"
    ):
        trainer_plugin(trainer)


def test_a_trainer_plugin_refuses_a_declaration_of_another_variant() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)

    with pytest.raises(
        RegistryError,
        match="the trainer plugin builds DistanceTrainerSpec, not ParserTrainerSpec",
    ):
        DISTANCE.plan(run_spec.trainer, run_spec)


def test_the_published_trainer_plugin_has_typed_callbacks() -> None:
    plugin = trainer_plugin(PACKAGED_CODEC(CONFIG).trainer)

    assert isinstance(plugin, TrainerPlugin)
    assert callable(plugin.planner)
    assert callable(plugin.presenter)
    assert callable(plugin.builder)
    assert callable(plugin.pricer)


def test_a_plugin_refuses_a_declaration_of_another_variant() -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    distance = trainer_plugin(RunSpecCodec()(DISTANCE_CONFIG).trainer)

    with pytest.raises(RegistryError, match="builds DistanceTrainerSpec"):
        distance.price(run_spec.trainer)

    with pytest.raises(RegistryError, match="builds DistanceTrainerSpec"):
        distance.plan(run_spec.trainer, run_spec)


def test_a_trainer_entry_naming_another_variants_plugin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distance = trainer_plugin(RunSpecCodec()(DISTANCE_CONFIG).trainer)
    trainer = PACKAGED_CODEC(CONFIG).trainer
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: distance)

    with pytest.raises(RegistryError, match="builds DistanceTrainerSpec"):
        trainer_plugin(trainer)


def _present_sentences(
    declaration: DistanceTrainerSpec, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    del run_spec

    return {declaration.dataset: frozenset({Unlabeled})}


def test_a_trainer_presenting_to_a_dataset_without_a_featurizer_is_refused() -> None:
    run_spec = RunSpecCodec()(DISTANCE_CONFIG)
    presenting = TrainerPlugin(
        spec=DistanceTrainerSpec,
        plan=DISTANCE.planner,
        present=_present_sentences,
        build=DISTANCE.builder,
        price=DISTANCE.pricer,
    )

    assert run_spec.data.datasets["train"].featurizer is None

    with pytest.raises(ValueError, match="declares no featurizer"):
        presenting.plan(run_spec.trainer, run_spec)
