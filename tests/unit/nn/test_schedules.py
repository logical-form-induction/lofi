import copy
import pickle
from itertools import pairwise
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from lofi.nn.schedules import (
    CONSTANT_PLUGIN,
    COSINE_PLUGIN,
    LINEAR_PLUGIN,
    ScheduleContext,
)
from lofi.registry import Registry
from lofi.spec.schedulers.constant import ConstantSchedulerSpec
from lofi.spec.schedulers.cosine import CosineSchedulerSpec
from lofi.spec.schedulers.linear import LinearSchedulerSpec

if TYPE_CHECKING:
    from lofi.spec import ConfiguredSpec

BASE_RATE: Final = 0.01
EPOCHS: Final = 4
STEPS_PER_EPOCH: Final = 25
TOTAL_STEPS: Final = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS: Final = 10


@pytest.fixture
def optimizer() -> AdamW:
    return AdamW(nn.Linear(4, 4).parameters(), lr=BASE_RATE)


def built(
    optimizer: AdamW,
    *,
    warmup_steps: int,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> LRScheduler:
    return COSINE_PLUGIN.construct(
        CosineSchedulerSpec(warmup_steps=warmup_steps),
        ScheduleContext(optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch),
    )


def test_installed_metadata_lists_every_scheduler_plugin() -> None:
    registry = Registry[object]("scheduler")

    assert registry.get("cosine") is COSINE_PLUGIN
    assert registry.get("linear") is LINEAR_PLUGIN
    assert registry.get("constant") is CONSTANT_PLUGIN
    assert registry.available() == ("constant", "cosine", "linear")


def test_a_context_derives_the_horizon_from_the_epochs_and_their_steps(
    optimizer: AdamW,
) -> None:
    context = ScheduleContext(optimizer, epochs=EPOCHS, steps_per_epoch=7)

    assert context.optimizer is optimizer
    assert context.total_steps == EPOCHS * 7


def test_a_context_is_frozen_and_refuses_serialization(optimizer: AdamW) -> None:
    context = ScheduleContext(optimizer, epochs=1, steps_per_epoch=1)

    assert copy.deepcopy(context) is context

    with pytest.raises(AttributeError, match="immutable"):
        context.__setattr__("_epochs", 2)

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(context)


def test_cosine_without_warmup_spans_the_derived_horizon(optimizer: AdamW) -> None:
    scheduler = built(optimizer, warmup_steps=0)

    assert type(scheduler) is CosineAnnealingLR
    assert scheduler.T_max == TOTAL_STEPS
    assert scheduler.eta_min == 0.0


def test_the_declared_default_warmup_reaches_the_built_schedule(
    optimizer: AdamW,
) -> None:
    scheduler = COSINE_PLUGIN.construct(
        CosineSchedulerSpec(),
        ScheduleContext(optimizer, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH),
    )

    assert type(scheduler) is CosineAnnealingLR


def test_warmup_and_cosine_return_sequential_lr(optimizer: AdamW) -> None:
    scheduler = built(optimizer, warmup_steps=WARMUP_STEPS)

    assert type(scheduler) is SequentialLR

    state = scheduler.state_dict()
    parts = cast("list[dict[str, object]]", state["_schedulers"])

    assert state["_milestones"] == [WARMUP_STEPS]
    assert parts[0]["start_factor"] == 1e-8
    assert parts[0]["total_iters"] == WARMUP_STEPS
    assert parts[1]["T_max"] == TOTAL_STEPS - WARMUP_STEPS


def test_the_warmup_starts_from_the_declared_fraction(optimizer: AdamW) -> None:
    scheduler = COSINE_PLUGIN.construct(
        CosineSchedulerSpec(warmup_steps=WARMUP_STEPS, warmup_start_factor=0.25),
        ScheduleContext(optimizer, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH),
    )

    assert scheduler.get_last_lr() == pytest.approx([BASE_RATE * 0.25])


def test_one_warmup_step_still_builds_a_warmup(optimizer: AdamW) -> None:
    scheduler = built(optimizer, warmup_steps=1)
    state = scheduler.state_dict()

    assert state["_milestones"] == [1]


@pytest.mark.parametrize("warmup_steps", [TOTAL_STEPS, TOTAL_STEPS + 1])
def test_a_warmup_leaving_no_decay_step_is_refused(
    optimizer: AdamW, warmup_steps: int
) -> None:
    with pytest.raises(ValueError, match="leaves no step"):
        built(optimizer, warmup_steps=warmup_steps)


def test_a_warmup_one_short_of_the_horizon_leaves_one_decay_step(
    optimizer: AdamW,
) -> None:
    scheduler = built(optimizer, warmup_steps=TOTAL_STEPS - 1)
    parts = cast("list[dict[str, object]]", scheduler.state_dict()["_schedulers"])

    assert parts[1]["T_max"] == 1


def test_warmup_starts_near_zero(optimizer: AdamW) -> None:
    scheduler = built(optimizer, warmup_steps=WARMUP_STEPS)

    assert scheduler.get_last_lr() == pytest.approx([BASE_RATE * 1e-8])


def test_cosine_reaches_its_minimum_at_the_derived_horizon(
    optimizer: AdamW,
) -> None:
    scheduler = built(optimizer, warmup_steps=0)

    for _ in range(TOTAL_STEPS):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == pytest.approx([0.0], abs=1e-12)


def test_composed_schedule_warms_then_anneals(optimizer: AdamW) -> None:
    scheduler = built(optimizer, warmup_steps=WARMUP_STEPS)

    assert scheduler.get_last_lr() == pytest.approx([BASE_RATE * 1e-8])

    for _ in range(WARMUP_STEPS):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == pytest.approx([BASE_RATE])

    for _ in range(TOTAL_STEPS - WARMUP_STEPS):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == pytest.approx([0.0], abs=1e-12)


class TestCosinePlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(
        self, optimizer: AdamW
    ) -> ConfiguredComponentFixture:
        context = ScheduleContext(
            optimizer, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH
        )

        def observe(component: object) -> dict[str, bool]:
            scheduler = cast("SequentialLR", component)
            parts = cast(
                "list[dict[str, object]]", scheduler.state_dict()["_schedulers"]
            )

            return {
                "drives the context's optimizer": scheduler.optimizer is optimizer,
                "warms for the declared steps": parts[0]["total_iters"] == WARMUP_STEPS,
                "anneals over the rest of the horizon": parts[1]["T_max"]
                == TOTAL_STEPS - WARMUP_STEPS,
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                CosineSchedulerSpec(warmup_steps=WARMUP_STEPS),
            ),
            context=context,
            expected_type=SequentialLR,
            observe=observe,
            unchanged=lambda: context.total_steps == TOTAL_STEPS,
        )


def context_for(optimizer: AdamW) -> ScheduleContext:
    return ScheduleContext(optimizer, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)


def rates(scheduler: LRScheduler, optimizer: AdamW, steps: int) -> list[float]:
    """Return the learning rate before each of a number of optimizer steps."""
    seen: list[float] = []

    for _ in range(steps):
        seen.append(float(scheduler.get_last_lr()[0]))
        optimizer.step()
        scheduler.step()

    seen.append(float(scheduler.get_last_lr()[0]))

    return seen


def test_linear_without_warmup_decays_to_the_final_factor_at_the_horizon(
    optimizer: AdamW,
) -> None:
    scheduler = LINEAR_PLUGIN.construct(
        LinearSchedulerSpec(final_factor=0.25), context_for(optimizer)
    )
    seen = rates(scheduler, optimizer, TOTAL_STEPS)

    assert type(scheduler) is LinearLR
    assert seen[0] == pytest.approx(BASE_RATE)
    assert seen[TOTAL_STEPS // 2] == pytest.approx(BASE_RATE * 0.625)
    assert seen[-1] == pytest.approx(BASE_RATE * 0.25)
    assert all(later <= earlier for earlier, later in pairwise(seen))


def test_linear_with_warmup_warms_then_decays(optimizer: AdamW) -> None:
    scheduler = LINEAR_PLUGIN.construct(
        LinearSchedulerSpec(warmup_steps=WARMUP_STEPS, warmup_start_factor=0.5),
        context_for(optimizer),
    )
    seen = rates(scheduler, optimizer, TOTAL_STEPS)

    assert type(scheduler) is SequentialLR
    assert seen[0] == pytest.approx(BASE_RATE * 0.5)
    assert seen[WARMUP_STEPS] == pytest.approx(BASE_RATE)
    assert seen[-1] == pytest.approx(0.0, abs=1e-12)


def test_constant_keeps_the_declared_rate_after_its_warmup(optimizer: AdamW) -> None:
    flat = CONSTANT_PLUGIN.construct(ConstantSchedulerSpec(), context_for(optimizer))

    assert type(flat) is ConstantLR
    assert rates(flat, optimizer, TOTAL_STEPS) == pytest.approx(
        [BASE_RATE] * (TOTAL_STEPS + 1)
    )


def test_constant_with_warmup_reaches_and_holds_the_declared_rate() -> None:
    optimizer = AdamW(nn.Linear(4, 4).parameters(), lr=BASE_RATE)
    scheduler = CONSTANT_PLUGIN.construct(
        ConstantSchedulerSpec(warmup_steps=WARMUP_STEPS, warmup_start_factor=0.1),
        context_for(optimizer),
    )
    seen = rates(scheduler, optimizer, TOTAL_STEPS)

    assert seen[0] == pytest.approx(BASE_RATE * 0.1)
    assert seen[WARMUP_STEPS:] == pytest.approx(
        [BASE_RATE] * (TOTAL_STEPS - WARMUP_STEPS + 1)
    )


def test_a_warmup_spanning_the_horizon_is_refused_by_the_linear_schedule(
    optimizer: AdamW,
) -> None:
    with pytest.raises(ValueError, match="leaves no step"):
        LINEAR_PLUGIN.construct(
            LinearSchedulerSpec(warmup_steps=TOTAL_STEPS), context_for(optimizer)
        )


def test_a_warmup_spanning_the_horizon_is_refused_by_the_constant_schedule(
    optimizer: AdamW,
) -> None:
    with pytest.raises(ValueError, match="leaves no step"):
        CONSTANT_PLUGIN.construct(
            ConstantSchedulerSpec(warmup_steps=TOTAL_STEPS), context_for(optimizer)
        )


class TestLinearPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(
        self, optimizer: AdamW
    ) -> ConfiguredComponentFixture:
        context = context_for(optimizer)

        def observe(component: object) -> dict[str, bool]:
            scheduler = cast("LinearLR", component)

            return {
                "drives the context's optimizer": scheduler.optimizer is optimizer,
                "decays over the horizon": scheduler.total_iters == TOTAL_STEPS,
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", LinearSchedulerSpec()),
            context=context,
            expected_type=LinearLR,
            observe=observe,
            unchanged=lambda: context.total_steps == TOTAL_STEPS,
        )


class TestConstantPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(
        self, optimizer: AdamW
    ) -> ConfiguredComponentFixture:
        context = context_for(optimizer)

        def observe(component: object) -> dict[str, bool]:
            scheduler = cast("ConstantLR", component)

            return {
                "drives the context's optimizer": scheduler.optimizer is optimizer,
                "holds the declared rate": scheduler.get_last_lr() == [BASE_RATE],
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", ConstantSchedulerSpec()),
            context=context,
            expected_type=ConstantLR,
            observe=observe,
            unchanged=lambda: context.total_steps == TOTAL_STEPS,
        )
