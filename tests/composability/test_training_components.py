"""Composition tests for installed training components."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

from didactic.settings import load_document
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, SequentialLR

from lofi.nn.schedules import ScheduleContext
from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.spec import OptimizerSpec, SchedulerSpec, TrainerSpec
from lofi.spec._plugins import load_plugins

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.spec import ConfiguredSpec
    from lofi.spec.trainers.parser import ParserTrainerSpec


def installed(family: str) -> tuple[str, ...]:
    """Return the tags a family's entry-point group lists."""
    registry = Registry[object](family)

    return registry.available()


def built(declaration: OptimizerSpec | SchedulerSpec, context: object) -> object:
    """Build one declared component through the plugin its family lists."""
    configured = cast("ConfiguredSpec[object]", declaration)

    return plugin_for(configured).construct(configured, context)


def test_packaged_training_fragments_validate_discover_construct_and_schedule() -> None:
    loaded = load_plugins()
    trainer_document = load_document(section_root("trainer") / "parser.yaml")
    optimizer_document = load_document(section_root("optimizer") / "adamw.yaml")
    scheduler_document = load_document(section_root("scheduler") / "cosine.yaml")
    trainer = cast(
        "ParserTrainerSpec",
        TrainerSpec.model_validate_json(json.dumps(trainer_document)),
    )
    optimizer_spec = OptimizerSpec.model_validate_json(json.dumps(optimizer_document))
    scheduler_spec = SchedulerSpec.model_validate_json(json.dumps(scheduler_document))
    parameter = nn.Parameter(nn.Linear(1, 1).weight.detach().clone())
    optimizer = cast("AdamW", built(optimizer_spec, (parameter,)))
    scheduler = cast(
        "LRScheduler",
        built(
            scheduler_spec,
            ScheduleContext(optimizer, epochs=trainer.epochs, steps_per_epoch=3),
        ),
    )

    assert {"parser", "adamw", "adam", "sgd", "cosine", "linear", "constant"} <= set(
        loaded
    )
    assert packaged_options("trainer") == (
        "distance",
        "parser",
        "terms",
    )
    assert packaged_options("optimizer") == ("adam", "adamw", "sgd")
    assert packaged_options("scheduler") == ("constant", "cosine", "linear")
    assert type(optimizer) is AdamW
    assert optimizer.param_groups[0]["params"] == [parameter]
    assert installed("optimizer") == ("adam", "adamw", "sgd")
    assert installed("scheduler") == ("constant", "cosine", "linear")
    assert installed("trainer") == (
        "alternating",
        "distance",
        "parser",
        "staged",
        "terms",
    )
    assert scheduler.state_dict()["T_max"] == trainer.epochs * 3


def test_scheduler_horizon_counts_optimizer_steps_not_accumulation_microbatches() -> (
    None
):
    trainer = cast(
        "ParserTrainerSpec",
        TrainerSpec.model_validate(
            {
                "kind": "parser",
                "dataset": "train",
                "epochs": 2,
                "batch_size": 8,
                "accumulation_steps": 4,
                "batch_sampler": {
                    "kind": "length-bucket",
                    "bucket_factor": 100,
                    "reshuffle_each_epoch": False,
                },
                "early_stopping": {
                    "metric": "loss/parser",
                    "mode": "min",
                    "patience": 3,
                    "minimum_delta": 0.0,
                    "keep_best": 1,
                },
                "checkpoints": {"interval": 1, "keep": 1},
            }
        ),
    )
    optimizer = AdamW(nn.Linear(2, 1).parameters(), lr=0.01)
    scheduler = built(
        SchedulerSpec.model_validate({"kind": "cosine", "warmup_steps": 1}),
        ScheduleContext(optimizer, epochs=trainer.epochs, steps_per_epoch=3),
    )

    assert type(scheduler) is SequentialLR
    state = scheduler.state_dict()
    parts = cast("list[dict[str, JsonValue]]", state["_schedulers"])
    assert state["_milestones"] == [1]
    assert parts[1]["T_max"] == trainer.epochs * 3 - 1
    assert parts[1]["T_max"] != trainer.epochs * 3 * trainer.accumulation_steps - 1


def test_training_components_discover_and_round_trip_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    probe = """
import json
from didactic.settings import load_document
from torch import nn
from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.spec import OptimizerSpec, SchedulerSpec, TrainerSpec
from lofi.spec._plugins import load_plugins
from lofi.nn.schedules import ScheduleContext

loaded = load_plugins()
families = (
    ("trainer", TrainerSpec),
    ("optimizer", OptimizerSpec),
    ("scheduler", SchedulerSpec),
)
round_trips = {}
specs = {}
for slot, family in families:
    options = packaged_options(slot)
    name = "parser" if slot == "trainer" else options[0]
    document = load_document(section_root(slot) / f"{name}.yaml")
    declared = family.model_validate_json(json.dumps(document))
    encoded = declared.model_dump_json()
    round_trips[slot] = family.model_validate_json(encoded).model_dump_json() == encoded
    specs[slot] = declared

optimizer = plugin_for(specs["optimizer"]).construct(
    specs["optimizer"], tuple(nn.Linear(2, 1).parameters())
)
scheduler = plugin_for(specs["scheduler"]).construct(
    specs["scheduler"],
    ScheduleContext(optimizer, epochs=specs["trainer"].epochs, steps_per_epoch=2),
)
optimizer_registry = Registry("optimizer")
trainer_registry = Registry("trainer")
print(json.dumps({
    "loaded": loaded,
    "optimizer_available": optimizer_registry.available(),
    "trainer_available": trainer_registry.available(),
    "options": {slot: packaged_options(slot) for slot, _ in families},
    "round_trips": round_trips,
    "optimizer": type(optimizer).__name__,
    "scheduler": type(scheduler).__name__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    observed = cast("dict[str, JsonValue]", json.loads(completed.stdout))
    loaded = cast("list[str]", observed["loaded"])

    assert {"parser", "adamw", "cosine"} <= set(loaded)
    assert observed["optimizer_available"] == ["adam", "adamw", "sgd"]
    assert observed["trainer_available"] == [
        "alternating",
        "distance",
        "parser",
        "staged",
        "terms",
    ]
    assert observed["options"] == {
        "trainer": ["distance", "parser", "terms"],
        "optimizer": ["adam", "adamw", "sgd"],
        "scheduler": ["constant", "cosine", "linear"],
    }
    assert observed["round_trips"] == {
        "trainer": True,
        "optimizer": True,
        "scheduler": True,
    }
    assert observed["optimizer"] == "Adam"
    assert observed["scheduler"] == "ConstantLR"
