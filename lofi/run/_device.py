"""Runtime device-policy resolution."""

from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Final, assert_never

from lofi.spec.policy import ComputeDtype

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    import torch

    from lofi.nn.policy import DevicePolicy
    from lofi.spec import RunSpec

__all__ = ["accelerator_of", "device_scope", "resolve_device_policy"]


_HOSTLESS: Final = frozenset({"cpu", "meta"})
"""The device families every machine holds, which need no accelerator."""


def accelerator_of(device: str, /) -> str | None:
    """Return the accelerator a declared device names, or None for a host device.

    Parameters
    ----------
    device : str
        The device a policy declares, such as ``cpu``, ``mps`` or ``cuda:1``.

    Returns
    -------
    str | None
        The device itself when its family is an accelerator, or None when it
        is a family every machine holds.
    """
    return None if device.partition(":")[0] in _HOSTLESS else device


def _refuse_absent(library: ModuleType, declared: str, device: torch.device, /) -> None:
    """Refuse an accelerator device the machine running the run does not hold.

    Parameters
    ----------
    library : ModuleType
        The tensor library.
    declared : str
        The device name the policy declares.
    device : torch.device
        The device that name spells.

    Raises
    ------
    ValueError
        If the machine's accelerator is of another family, or holds fewer devices
        than the declared index requires.
    """
    if device.type in _HOSTLESS:
        return

    accelerator = library.accelerator.current_accelerator(check_available=True)

    if accelerator is None:
        msg = f"device {declared!r} is not present: this machine has no accelerator"
        raise ValueError(msg)

    if accelerator.type != device.type:
        msg = (
            f"device {declared!r} is not present: this machine's accelerator is "
            f"{accelerator.type}"
        )
        raise ValueError(msg)

    count = library.accelerator.device_count()

    if device.index is not None and device.index >= count:
        msg = (
            f"device {declared!r} is not present: this machine holds {count} "
            f"{device.type} devices"
        )
        raise ValueError(msg)


def resolve_device_policy(run_spec: RunSpec, /) -> DevicePolicy:
    """Resolve the device and precision policy for one run.

    Parameters
    ----------
    run_spec : RunSpec
        Validated run configuration.

    Returns
    -------
    DevicePolicy
        Runtime placement and precision policy.

    Raises
    ------
    ValueError
        If the device family is unsupported, the machine holds no such device, or
        the precision policy is incompatible with the device.
    """
    torch = import_module("torch")
    neural_policy = import_module("lofi.nn.policy")
    declaration = run_spec.device_policy

    try:
        device = torch.device(declaration.device)
    except (RuntimeError, ValueError) as error:
        family = declaration.device.partition(":")[0]
        declared = ", ".join(str(member.value) for member in neural_policy.DeviceKind)
        msg = f"undeclared device {family!r}; declared: {declared}"
        raise ValueError(msg) from error

    try:
        neural_policy.DeviceKind(device.type)
    except ValueError as error:
        declared = ", ".join(str(member.value) for member in neural_policy.DeviceKind)
        msg = f"undeclared device {device.type!r}; declared: {declared}"
        raise ValueError(msg) from error

    _refuse_absent(torch, declaration.device, device)

    match declaration.compute_dtype:
        case ComputeDtype.FLOAT16:
            parameter_dtype = torch.float32
            autocast_dtype = torch.float16
        case ComputeDtype.BFLOAT16:
            parameter_dtype = torch.float32
            autocast_dtype = torch.bfloat16
        case ComputeDtype.FLOAT32:
            parameter_dtype = torch.float32
            autocast_dtype = None
        case ComputeDtype.FLOAT64:
            parameter_dtype = torch.float64
            autocast_dtype = None
        case unreachable:
            assert_never(unreachable)

    return neural_policy.DevicePolicy(
        device,
        parameter_dtype,
        autocast_dtype,
        checkpoint_from_width=declaration.checkpoint_from_width,
    )


@contextmanager
def device_scope(run_spec: RunSpec, /) -> Iterator[DevicePolicy]:
    """Apply one run's process-wide numeric settings and restore them on exit.

    Parameters
    ----------
    run_spec : RunSpec
        Validated run configuration.

    Yields
    ------
    DevicePolicy
        Runtime placement and precision policy.

    Raises
    ------
    ValueError
        If the device family is unsupported, the machine holds no such device, or
        the precision policy is incompatible with the device.
    """
    torch = import_module("torch")
    declaration = run_spec.device_policy
    policy = resolve_device_policy(run_spec)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_precision = torch.get_float32_matmul_precision()

    try:
        torch.use_deterministic_algorithms(declaration.deterministic, warn_only=False)
        torch.set_float32_matmul_precision(declaration.matmul_precision.value)
        yield policy

    finally:
        try:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

        finally:
            torch.set_float32_matmul_precision(matmul_precision)
