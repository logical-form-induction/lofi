from copy import copy, deepcopy

import didactic.api as dx
import pytest

from lofi.control_variate import (
    ControlVariate,
    ControlVariatePlugin,
    ControlVariateSpec,
    MovingControlVariate,
    MovingControlVariateSpec,
)
from lofi.registry import Registry


def test_a_moving_baseline_tracks_the_mean_decided_reward() -> None:
    baseline = MovingControlVariate(decay=0.75, initial=0.25)

    baseline.observe((0.5, 1.0))

    assert baseline.value == pytest.approx(0.375)
    assert isinstance(baseline, ControlVariate)
    assert repr(baseline) == "MovingControlVariate(decay=0.75, value=0.375)"


def test_a_moving_baseline_ignores_a_step_with_no_decided_reward() -> None:
    baseline = MovingControlVariate(decay=0.5, initial=0.25)

    baseline.observe(())

    assert baseline.value == pytest.approx(0.25)


def test_a_moving_baseline_declaration_owns_its_initial_value() -> None:
    declared = MovingControlVariateSpec(decay=0.8)

    assert declared.settings() == (0.8, 0.0)


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("inf")])
def test_a_moving_baseline_refuses_an_invalid_decay(decay: float) -> None:
    with pytest.raises(dx.ValidationError, match="control-variate decay"):
        MovingControlVariateSpec(decay=decay)


def test_a_moving_baseline_spec_is_the_only_boundary_value() -> None:
    declared = MovingControlVariateSpec(decay=0.5, initial=-1.0)

    assert copy(declared) == declared
    assert deepcopy(declared) == declared


def test_the_tagged_root_declares_no_fallback_settings_operation() -> None:
    assert "settings" not in ControlVariateSpec.__dict__


def test_the_registered_moving_plugin_builds_a_moving_baseline() -> None:
    plugin = Registry[object]("control_variate").get("moving")

    assert isinstance(plugin, ControlVariatePlugin)

    built = plugin.construct(MovingControlVariateSpec(decay=0.5).settings())

    assert isinstance(built, MovingControlVariate)
    assert built.value == 0.0


def test_a_baseline_plugin_refuses_settings_of_another_variant() -> None:
    plugin = Registry[object]("control_variate").get("moving")

    assert isinstance(plugin, ControlVariatePlugin)

    with pytest.raises(TypeError, match="reads MovingControlVariateSettings"):
        plugin.construct(object())
