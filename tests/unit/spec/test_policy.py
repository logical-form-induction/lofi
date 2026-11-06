import math

import didactic.api as dx
import pytest

from lofi.spec.policy import (
    ComputeDtype,
    DevicePolicySpec,
    GradScalerSpec,
    MatmulPrecision,
)


def policy(**changes: str | bool | int | GradScalerSpec) -> DevicePolicySpec:
    values: dict[str, str | bool | int | GradScalerSpec] = {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    }
    values.update(changes)

    return DevicePolicySpec.model_validate(values)


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps", "xpu"])
def test_device_declarations_preserve_their_full_spelling(device: str) -> None:
    assert policy(device=device).device == device


def test_declared_values_decode_to_enum_members() -> None:
    declared = policy(compute_dtype="float16", matmul_precision="medium")

    assert declared.compute_dtype is ComputeDtype.FLOAT16
    assert declared.matmul_precision is MatmulPrecision.MEDIUM


def test_a_policy_with_a_scaler_round_trips_through_json() -> None:
    declared = policy(
        compute_dtype="float16",
        scaler=GradScalerSpec(
            init_scale=1024.0,
            growth_factor=3.0,
            backoff_factor=0.25,
            growth_interval=16,
        ),
    )

    restored = DevicePolicySpec.model_validate_json(declared.model_dump_json())

    assert restored == declared
    assert restored.scaler == declared.scaler


def test_gradient_scaler_defaults_are_valid() -> None:
    scaler = GradScalerSpec()

    assert scaler.init_scale > 0.0
    assert scaler.growth_factor > 1.0
    assert 0.0 < scaler.backoff_factor < 1.0
    assert scaler.growth_interval > 0


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_a_nonpositive_or_nonfinite_initial_scale_is_refused(value: float) -> None:
    with pytest.raises(dx.ValidationError):
        GradScalerSpec(init_scale=value)


@pytest.mark.parametrize("value", [1.0, 0.0, -1.0, math.inf, math.nan])
def test_an_invalid_growth_factor_is_refused(value: float) -> None:
    with pytest.raises(dx.ValidationError):
        GradScalerSpec(growth_factor=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, math.inf, math.nan])
def test_an_invalid_backoff_factor_is_refused(value: float) -> None:
    with pytest.raises(dx.ValidationError):
        GradScalerSpec(backoff_factor=value)


@pytest.mark.parametrize("value", [0, -1])
def test_a_nonpositive_growth_interval_is_refused(value: int) -> None:
    with pytest.raises(dx.ValidationError):
        GradScalerSpec(growth_interval=value)


def test_bfloat16_refuses_a_gradient_scaler() -> None:
    with pytest.raises(dx.ValidationError, match="bfloat16 computation"):
        policy(compute_dtype="bfloat16", scaler=GradScalerSpec())


def test_bfloat16_without_a_scaler_is_valid() -> None:
    assert policy(compute_dtype="bfloat16").scaler is None


def test_policy_values_are_immutable() -> None:
    declared = policy()

    with pytest.raises(AttributeError, match="immutable"):
        declared.__setattr__("device", "cuda:1")


@pytest.mark.parametrize(
    ("field", "value"),
    [("compute_dtype", "float128"), ("matmul_precision", "fast")],
)
def test_an_unknown_declared_value_is_refused(field: str, value: str) -> None:
    values = policy().model_dump()
    values[field] = value

    with pytest.raises(dx.ValidationError):
        DevicePolicySpec.model_validate(values)


def test_a_policy_declares_its_activation_memory_budget() -> None:
    assert policy(activation_memory_bytes=40_000_000_000).activation_memory_bytes == (
        40_000_000_000
    )


@pytest.mark.parametrize("value", [0, -1])
def test_a_nonpositive_activation_memory_budget_is_refused(value: int) -> None:
    with pytest.raises(dx.ValidationError, match="activation memory budget"):
        policy(activation_memory_bytes=value)


def test_a_policy_without_an_activation_memory_budget_is_refused() -> None:
    values = policy().model_dump()
    del values["activation_memory_bytes"]

    with pytest.raises(dx.ValidationError):
        DevicePolicySpec.model_validate(values)


def test_an_unknown_scaler_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        GradScalerSpec.model_validate({"unknown": True})


def test_an_unknown_policy_setting_is_refused() -> None:
    values = policy().model_dump() | {"unknown": True}

    with pytest.raises(dx.ValidationError):
        DevicePolicySpec.model_validate(values)


def test_a_policy_checkpoints_no_chart_width_unless_it_declares_one() -> None:
    assert policy().checkpoint_from_width is None
    assert policy(checkpoint_from_width=6).checkpoint_from_width == 6


@pytest.mark.parametrize("width", [1, 0, -1, True])
def test_a_checkpointing_width_below_two_is_refused(width: int) -> None:
    with pytest.raises(dx.ValidationError, match="checkpoint_from_width"):
        policy(checkpoint_from_width=width)
