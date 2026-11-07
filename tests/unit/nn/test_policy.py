from typing import override

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn.policy import DeviceKind, DevicePolicy, ensure_placed, place

AUTOCASTING_KINDS = tuple(
    kind for kind in DeviceKind if kind.autocast_device_type is not None
)


def policy_on(
    kind: DeviceKind, autocast_dtype: torch.dtype | None = None
) -> DevicePolicy:
    return DevicePolicy(torch.device(kind.value), torch.float32, autocast_dtype)


@pytest.mark.parametrize("kind", list(DeviceKind))
def test_every_declared_kind_builds_a_policy(kind: DeviceKind) -> None:
    policy = policy_on(kind)

    assert policy.kind is kind
    assert policy.device.type == kind.value


def test_an_undeclared_device_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="undeclared device 'hpu'"):
        DevicePolicy(torch.device("hpu"), torch.float32)


def test_the_cpu_region_enables_no_other_device_type() -> None:
    policy = DevicePolicy(torch.device("cpu"), torch.float32, torch.bfloat16)
    others = [
        kind.autocast_device_type
        for kind in AUTOCASTING_KINDS
        if kind is not DeviceKind.CPU and kind.autocast_device_type is not None
    ]

    with policy.autocast():
        assert torch.is_autocast_enabled("cpu")
        assert [torch.is_autocast_enabled(other) for other in others] == [False] * len(
            others
        )


@pytest.mark.filterwarnings("ignore:CUDA is not available")
@pytest.mark.parametrize(
    "kind", [kind for kind in AUTOCASTING_KINDS if kind is not DeviceKind.CPU]
)
def test_an_accelerator_region_never_takes_the_cpu_path(kind: DeviceKind) -> None:
    policy = policy_on(kind, torch.bfloat16)

    with policy.autocast():
        assert not torch.is_autocast_enabled("cpu")


def test_autocast_computes_in_the_policys_autocast_dtype() -> None:
    policy = DevicePolicy(torch.device("cpu"), torch.float32, torch.bfloat16)
    left = torch.randn(4, 4, device=policy.device, dtype=policy.dtype)
    right = torch.randn(4, 4, device=policy.device, dtype=policy.dtype)

    with policy.autocast():
        assert torch.is_autocast_enabled("cpu")
        assert (left @ right).dtype is torch.bfloat16

        with policy.autocast(enabled=False):
            assert not torch.is_autocast_enabled("cpu")
            assert (left @ right).dtype is torch.float32

    assert not torch.is_autocast_enabled("cpu")
    assert (left @ right).dtype is torch.float32


def test_a_policy_without_an_autocast_dtype_computes_in_full_precision() -> None:
    policy = DevicePolicy(torch.device("cpu"), torch.float32)
    left = torch.randn(4, 4, device=policy.device, dtype=policy.dtype)
    right = torch.randn(4, 4, device=policy.device, dtype=policy.dtype)

    assert not policy.autocast_enabled

    with policy.autocast():
        assert not torch.is_autocast_enabled("cpu")
        assert (left @ right).dtype is torch.float32


def test_a_device_without_autocast_still_yields_a_usable_region() -> None:
    policy = policy_on(DeviceKind.META)

    assert policy.autocast_device_type is None

    with policy.autocast():
        assert torch.zeros(3, device=policy.device, dtype=policy.dtype).is_meta


def test_autocast_on_a_device_that_has_none_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="no autocast implementation"):
        policy_on(DeviceKind.META, torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_an_integral_parameter_dtype_is_a_construction_error(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="parameter dtype"):
        DevicePolicy(torch.device("cpu"), dtype)


def test_an_integral_autocast_dtype_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="autocast dtype"):
        DevicePolicy(torch.device("cpu"), torch.float32, torch.int32)


def test_a_policy_reads_back_what_it_was_built_from_and_holds_nothing_else() -> None:
    policy = DevicePolicy(torch.device("cpu"), torch.float32)

    assert policy.device.type == "cpu"
    assert policy.dtype is torch.float32
    assert policy.autocast_dtype is None
    assert policy.checkpoint_from_width is None
    assert policy.kind is DeviceKind.CPU
    assert not hasattr(policy, "__dict__")


def test_equality_is_placement_and_precision() -> None:
    cpu = torch.device("cpu")
    policy = DevicePolicy(cpu, torch.float32, torch.bfloat16)

    assert policy == DevicePolicy(cpu, torch.float32, torch.bfloat16)
    assert policy != DevicePolicy(cpu, torch.float64, torch.bfloat16)
    assert policy != DevicePolicy(cpu, torch.float32)
    assert policy != DevicePolicy(torch.device("meta"), torch.float32)
    assert policy != torch.float32

    seen = {policy: "one", DevicePolicy(cpu, torch.float32, torch.bfloat16): "again"}

    assert seen[policy] == "again"


@settings(max_examples=50)
@given(
    kind=st.sampled_from(AUTOCASTING_KINDS),
    index=st.integers(min_value=0, max_value=15),
)
def test_the_autocast_device_type_ignores_the_device_index(
    kind: DeviceKind, index: int
) -> None:
    indexed = DevicePolicy(torch.device(kind.value, index), torch.float32)

    assert indexed.device.index == index
    assert indexed.autocast_device_type == policy_on(kind).autocast_device_type


DOUBLE = DevicePolicy(torch.device("cpu"), torch.float64)


class Stack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = torch.nn.Linear(2, 2)
        self.register_buffer("gain", torch.ones(2))
        self.register_buffer("steps", torch.zeros(2, dtype=torch.int64))

    @override
    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.inner(rows) * self.gain


class Latecomer(Stack):
    def grow(self) -> None:
        self.later = torch.nn.Linear(2, 2)


def test_placing_a_module_moves_what_it_owns_to_the_run() -> None:
    module = Stack()

    placed = place(module, DOUBLE)

    assert placed is module
    assert [parameter.dtype for parameter in module.parameters()] == [torch.float64] * 2
    assert module.get_buffer("gain").dtype is torch.float64
    assert module.get_buffer("gain").device.type == "cpu"


def test_placing_a_module_leaves_a_counting_buffer_in_its_own_type() -> None:
    module = place(Stack(), DOUBLE)

    assert module.get_buffer("steps").dtype is torch.int64


def test_placing_a_module_a_second_time_changes_nothing() -> None:
    module = place(Stack(), DOUBLE)
    weights = module.inner.weight.detach().clone()

    again = place(module, DOUBLE)

    assert again is module
    assert torch.equal(module.inner.weight, weights)
    assert ensure_placed(module, DOUBLE) is module


def test_a_module_still_in_the_default_precision_is_refused() -> None:
    with pytest.raises(ValueError, match=r"inner\.weight is held in"):
        ensure_placed(Stack(), DOUBLE)


def test_a_module_left_on_another_device_is_refused() -> None:
    elsewhere = place(Stack(), DevicePolicy(torch.device("meta"), torch.float64))

    with pytest.raises(ValueError, match=r"inner\.weight is on meta"):
        ensure_placed(elsewhere, DOUBLE)


def test_a_submodule_registered_after_placement_is_refused() -> None:
    module = place(Latecomer(), DOUBLE)

    module.grow()

    with pytest.raises(ValueError, match=r"later\.weight is held in"):
        ensure_placed(module, DOUBLE)


def test_placing_a_module_again_takes_a_late_submodule_with_it() -> None:
    module = place(Latecomer(), DOUBLE)

    module.grow()

    assert ensure_placed(place(module, DOUBLE), DOUBLE) is module


def test_an_index_a_family_hands_out_is_not_read_as_a_mismatch() -> None:
    indexed = DevicePolicy(torch.device("cpu", 0), torch.float64)

    module = place(Stack(), indexed)

    assert module.inner.weight.device.index is None
    assert ensure_placed(module, indexed) is module


def test_a_module_placed_on_a_run_computes_in_that_run_s_precision() -> None:
    module = place(Stack(), DOUBLE)

    rows = torch.ones(3, 2, dtype=DOUBLE.dtype, device=DOUBLE.device)

    assert module(rows).dtype is torch.float64


def test_a_checkpointing_width_is_part_of_the_policy_and_its_identity() -> None:
    cpu = torch.device("cpu")
    policy = DevicePolicy(cpu, torch.float32, checkpoint_from_width=4)

    assert policy.checkpoint_from_width == 4
    assert policy == DevicePolicy(cpu, torch.float32, checkpoint_from_width=4)
    assert policy != DevicePolicy(cpu, torch.float32, checkpoint_from_width=5)
    assert policy != DevicePolicy(cpu, torch.float32)
    assert hash(policy) == hash(
        DevicePolicy(cpu, torch.float32, checkpoint_from_width=4)
    )
    assert "checkpoint_from_width=4" in repr(policy)


@pytest.mark.parametrize("width", [1, 0, -2])
def test_a_checkpointing_width_below_two_is_a_construction_error(width: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        DevicePolicy(torch.device("cpu"), torch.float32, checkpoint_from_width=width)
