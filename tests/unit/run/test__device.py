import subprocess
import sys
from typing import Literal, NamedTuple

import pytest
import torch

from lofi.nn.policy import DeviceKind
from lofi.run import (
    device_scope,
    resolve_device_policy,
)
from lofi.run._device import accelerator_of
from lofi.spec import (
    ComputeDtype,
    DatasetSpec,
    DevicePolicySpec,
    FeaturizerSpec,
    LossSpec,
    MatmulPrecision,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    TrainerSpec,
)


class _ParserSpec(ParserSpec):
    kind: Literal["device-policy-parser"] = "device-policy-parser"


class _LossSpec(LossSpec):
    kind: Literal["device-policy-loss"] = "device-policy-loss"


class _DatasetSpec(DatasetSpec):
    kind: Literal["device-policy-dataset"] = "device-policy-dataset"


class _FeaturizerSpec(FeaturizerSpec):
    kind: Literal["device-policy-featurizer"] = "device-policy-featurizer"


class _TrainerSpec(TrainerSpec):
    kind: Literal["device-policy-trainer"] = "device-policy-trainer"


class _OptimizerSpec(OptimizerSpec):
    kind: Literal["device-policy-optimizer"] = "device-policy-optimizer"


class _PolicyCalls(NamedTuple):
    devices: list[str]
    deterministic: list[bool]
    matmul: list[str]


@pytest.fixture
def policy_calls(monkeypatch: pytest.MonkeyPatch) -> _PolicyCalls:
    calls = _PolicyCalls([], [], [])
    device = torch.device

    def counted_device(declaration: str) -> torch.device:
        calls.devices.append(declaration)
        return device(declaration)

    monkeypatch.setattr(torch, "device", counted_device)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        calls.deterministic.append,
    )
    monkeypatch.setattr(
        torch,
        "set_float32_matmul_precision",
        calls.matmul.append,
    )

    return calls


def _run(
    *,
    device: str = "cpu",
    compute_dtype: ComputeDtype = ComputeDtype.FLOAT32,
    deterministic: bool = True,
    matmul_precision: MatmulPrecision = MatmulPrecision.HIGH,
    checkpoint_from_width: int | None = None,
) -> RunSpec:
    declaration = DevicePolicySpec(
        device=device,
        compute_dtype=compute_dtype,
        deterministic=deterministic,
        matmul_precision=matmul_precision,
        activation_memory_bytes=8_000_000_000,
        checkpoint_from_width=checkpoint_from_width,
    )

    return RunSpec.model_validate(
        {
            "parse": {"retain": "forest", "reachability": "all"},
            "release": {"checkpoint": "resume-latest"},
            "model": {
                "language": {"from": "data"},
                "parser": {"kind": "device-policy-parser"},
            },
            "loss": {"terms": [{"objective": {"kind": "device-policy-loss"}}]},
            "data": {
                "datasets": {
                    "train": {
                        "dataset": {"kind": "device-policy-dataset"},
                        "featurizer": {"kind": "device-policy-featurizer"},
                    }
                }
            },
            "trainer": {"kind": "device-policy-trainer"},
            "optimizer": {"kind": "device-policy-optimizer"},
            "device_policy": declaration.model_dump(),
        }
    )


@pytest.mark.parametrize(
    ("compute_dtype", "parameter_dtype", "autocast_dtype"),
    [
        (ComputeDtype.FLOAT16, torch.float32, torch.float16),
        (ComputeDtype.BFLOAT16, torch.float32, torch.bfloat16),
        (ComputeDtype.FLOAT32, torch.float32, None),
        (ComputeDtype.FLOAT64, torch.float64, None),
    ],
)
def test_resolution_uses_the_explicit_precision_table(
    compute_dtype: ComputeDtype,
    parameter_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
    policy_calls: _PolicyCalls,
) -> None:
    policy = resolve_device_policy(_run(compute_dtype=compute_dtype))

    assert policy.dtype is parameter_dtype
    assert policy.autocast_dtype is autocast_dtype
    assert policy_calls.devices == ["cpu"]
    assert policy_calls.deterministic == []
    assert policy_calls.matmul == []


class _Accelerator(NamedTuple):
    type: str


def _present(monkeypatch: pytest.MonkeyPatch, family: str | None, count: int) -> None:
    """Report the machine as holding ``count`` devices of one accelerator family."""
    accelerator = None if family is None else _Accelerator(family)

    def current_accelerator(*, check_available: bool = False) -> _Accelerator | None:
        del check_available
        return accelerator

    monkeypatch.setattr(torch.accelerator, "current_accelerator", current_accelerator)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: count)


@pytest.mark.parametrize(
    ("declaration", "family", "count", "held"),
    [
        ("cuda", None, 0, "has no accelerator"),
        ("cuda:0", "mps", 1, "accelerator is mps"),
        ("mps", "cuda", 2, "accelerator is cuda"),
        ("cuda:2", "cuda", 2, "holds 2 cuda devices"),
    ],
)
def test_resolution_refuses_a_device_the_machine_does_not_hold(
    declaration: str,
    family: str | None,
    count: int,
    held: str,
    policy_calls: _PolicyCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _present(monkeypatch, family, count)

    with pytest.raises(ValueError, match=f"is not present: this machine.* {held}"):
        resolve_device_policy(_run(device=declaration))

    assert policy_calls.deterministic == []
    assert policy_calls.matmul == []


@pytest.mark.parametrize("declaration", ["cpu", "cpu:3", "meta"])
def test_resolution_needs_no_accelerator_for_the_processor_or_meta_device(
    declaration: str, policy_calls: _PolicyCalls, monkeypatch: pytest.MonkeyPatch
) -> None:
    _present(monkeypatch, None, 0)

    assert resolve_device_policy(_run(device=declaration)).device.type in {
        "cpu",
        "meta",
    }
    assert policy_calls.devices == [declaration]


@pytest.mark.parametrize(
    ("kind", "declaration", "index"),
    [
        (DeviceKind.CPU, "cpu:0", 0),
        (DeviceKind.CUDA, "cuda:1", 1),
        (DeviceKind.MPS, "mps:2", 2),
        (DeviceKind.XPU, "xpu:3", 3),
        (DeviceKind.META, "meta", None),
    ],
)
def test_resolution_is_total_over_the_declared_device_families(
    kind: DeviceKind,
    declaration: str,
    index: int | None,
    policy_calls: _PolicyCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _present(monkeypatch, kind.value, 4)
    policy = resolve_device_policy(_run(device=declaration))

    assert policy.kind is kind
    assert policy.device.index == index
    assert policy_calls.devices == [declaration]


def test_resolution_rejects_an_unknown_device_family(
    policy_calls: _PolicyCalls,
) -> None:
    for declaration in ("hpu:4", "unknown:5"):
        with pytest.raises(ValueError, match="undeclared device"):
            resolve_device_policy(_run(device=declaration))

    assert policy_calls.devices == ["hpu:4", "unknown:5"]
    assert policy_calls.deterministic == []
    assert policy_calls.matmul == []


def test_importing_the_run_package_does_not_import_torch() -> None:
    source = "import sys; import lofi.run; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.stdout.strip() == "False"


def test_device_scope_restores_every_process_global_after_return_and_error() -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()

    try:
        torch.use_deterministic_algorithms(mode=False, warn_only=True)
        torch.set_float32_matmul_precision("medium")

        with device_scope(
            _run(
                deterministic=True,
                matmul_precision=MatmulPrecision.HIGHEST,
            )
        ) as policy:
            assert policy.device.type == "cpu"
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.get_float32_matmul_precision() == "highest"

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_float32_matmul_precision() == "medium"

        failure = "scope failure"

        with (
            pytest.raises(RuntimeError, match=failure),
            device_scope(_run(deterministic=True)),
        ):
            raise RuntimeError(failure)

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_float32_matmul_precision() == "medium"

    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)


def test_device_scope_rolls_back_when_a_setup_setter_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    set_precision = torch.set_float32_matmul_precision
    calls: list[str] = []

    try:
        torch.use_deterministic_algorithms(mode=False, warn_only=True)
        set_precision("medium")

        def reject_highest(
            value: Literal["highest", "high", "medium"],
        ) -> None:
            calls.append(value)

            if value == "highest":
                msg = "matmul setup failed"
                raise RuntimeError(msg)

            set_precision(value)

        monkeypatch.setattr(torch, "set_float32_matmul_precision", reject_highest)

        with (
            pytest.raises(RuntimeError, match="matmul setup failed"),
            device_scope(
                _run(
                    deterministic=True,
                    matmul_precision=MatmulPrecision.HIGHEST,
                )
            ),
        ):
            pytest.fail("a failed setup cannot enter the scope body")

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_float32_matmul_precision() == "medium"
        assert calls == ["highest", "medium"]

    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        set_precision(precision)


@pytest.mark.parametrize("width", [None, 2, 7])
def test_resolution_carries_the_declared_checkpointing_width(
    width: int | None, policy_calls: _PolicyCalls
) -> None:
    policy = resolve_device_policy(_run(checkpoint_from_width=width))

    assert policy.checkpoint_from_width == width
    assert policy_calls.devices == ["cpu"]


@pytest.mark.parametrize(
    ("device", "accelerator"),
    [("cpu", None), ("meta", None), ("mps", "mps"), ("cuda:1", "cuda:1")],
)
def test_a_declared_device_names_its_accelerator_unless_every_host_holds_it(
    device: str, accelerator: str | None
) -> None:
    assert accelerator_of(device) == accelerator
