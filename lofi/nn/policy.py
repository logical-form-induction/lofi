"""Tensor placement and precision policies."""

from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Self, override

import torch

if TYPE_CHECKING:
    from torch.nn import Module

__all__ = ["DeviceKind", "DevicePolicy", "ensure_placed", "place"]


class DeviceKind(Enum):
    """Identifies a supported device family."""

    CPU = ("cpu", "cpu")
    CUDA = ("cuda", "cuda")
    MPS = ("mps", "mps")
    XPU = ("xpu", "xpu")
    META = ("meta", None)

    #: The device type torch's autocast takes, or None where the family has no
    #: autocast implementation at all.
    autocast_device_type: str | None

    def __new__(cls, device_type: str, autocast_device_type: str | None) -> Self:
        """Return the member whose value is the device type torch reports.

        Parameters
        ----------
        device_type : str
            The device type this family answers to.
        autocast_device_type : str | None
            The device type its autocast takes, or None where it has none.
        """
        member = object.__new__(cls)
        member._value_ = device_type
        member.autocast_device_type = autocast_device_type

        return member


class DevicePolicy:
    """Resolves tensor placement and precision for one run.

    Parameters
    ----------
    device : torch.device
        Where parameters and activations live, index included.
    dtype : torch.dtype
        The floating point type parameters are stored in.
    autocast_dtype : torch.dtype | None
        The floating point type mixed precision computes in, or None to run with
        autocast off.
    checkpoint_from_width : int | None
        The narrowest span width whose chart cells are recomputed during the
        backward pass rather than held from the forward pass, or None to hold
        every cell's activations.

    Raises
    ------
    ValueError
        If the device family is not one the stack declares, if either dtype is not a
        floating point type, if autocast is asked of a family that has none, or if
        the checkpointing width is below two.
    """

    __slots__ = (
        "_autocast_dtype",
        "_checkpoint_from_width",
        "_device",
        "_dtype",
        "_kind",
    )

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        autocast_dtype: torch.dtype | None = None,
        *,
        checkpoint_from_width: int | None = None,
    ) -> None:
        try:
            kind = DeviceKind(device.type)

        except ValueError as error:
            declared = ", ".join(str(member.value) for member in DeviceKind)
            msg = f"undeclared device {device.type!r}; declared: {declared}"

            raise ValueError(msg) from error

        if not dtype.is_floating_point:
            msg = f"parameter dtype must be a floating point type: {dtype}"

            raise ValueError(msg)

        if autocast_dtype is not None:
            if not autocast_dtype.is_floating_point:
                msg = f"autocast dtype must be a floating point type: {autocast_dtype}"

                raise ValueError(msg)

            if kind.autocast_device_type is None:
                msg = f"device {device.type!r} has no autocast implementation"

                raise ValueError(msg)

        if checkpoint_from_width is not None and checkpoint_from_width < 2:
            msg = (
                f"a chart's cells are composed from width two, so checkpointing "
                f"starts at a width of at least two, not {checkpoint_from_width}"
            )

            raise ValueError(msg)

        self._device = device
        self._dtype = dtype
        self._autocast_dtype = autocast_dtype
        self._checkpoint_from_width = checkpoint_from_width
        self._kind = kind

    @property
    def device(self) -> torch.device:
        """Return the resolved torch device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating point type parameters are stored in."""
        return self._dtype

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """Return the type mixed precision computes in, or None with autocast off."""
        return self._autocast_dtype

    @property
    def checkpoint_from_width(self) -> int | None:
        """Return the narrowest span width whose cells are recomputed, or None."""
        return self._checkpoint_from_width

    @property
    def kind(self) -> DeviceKind:
        """Return the declared family this run's device belongs to."""
        return self._kind

    @property
    def autocast_device_type(self) -> str | None:
        """Return the device type torch's autocast takes for this run's device."""
        return self._kind.autocast_device_type

    @property
    def autocast_enabled(self) -> bool:
        """Return whether this run computes under autocast."""
        return self._autocast_dtype is not None

    def autocast(self, *, enabled: bool = True) -> AbstractContextManager[None]:
        """Return the autocast region for this run's device.

        Parameters
        ----------
        enabled : bool
            Whether autocast applies inside the region.
        """
        device_type = self._kind.autocast_device_type

        if device_type is None or (enabled and self._autocast_dtype is None):
            return nullcontext()

        if not enabled:
            return torch.autocast(device_type=device_type, enabled=False)

        return torch.autocast(device_type=device_type, dtype=self._autocast_dtype)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether both policies place and compute identically.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not DevicePolicy:
            return NotImplemented

        return (
            self._device == other._device
            and self._dtype == other._dtype
            and self._autocast_dtype == other._autocast_dtype
            and self._checkpoint_from_width == other._checkpoint_from_width
        )

    @override
    def __hash__(self) -> int:
        """Return the hash of the placement and precision this policy names."""
        return hash(
            (
                DevicePolicy,
                self._device,
                self._dtype,
                self._autocast_dtype,
                self._checkpoint_from_width,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return (
            f"{type(self).__name__}(device={self._device!r}, dtype={self._dtype!r}, "
            f"autocast_dtype={self._autocast_dtype!r}, "
            f"checkpoint_from_width={self._checkpoint_from_width!r})"
        )


def _elsewhere(where: torch.device, wanted: torch.device) -> bool:
    """Return whether a tensor is placed somewhere other than a run asked for.

    Parameters
    ----------
    where : torch.device
        The device a tensor is on.
    wanted : torch.device
        The device the run named.
    """
    if where.type != wanted.type:
        return True

    return (
        wanted.index is not None
        and where.index is not None
        and where.index != wanted.index
    )


def ensure_placed[ModuleT: Module](module: ModuleT, policy: DevicePolicy, /) -> ModuleT:
    """Return a module.

    Parameters
    ----------
    module : ModuleT
        The module to check, along with every submodule it owns.
    policy : DevicePolicy
        Where this run's tensors are placed and what precision they are stored in.

    Returns
    -------
    ModuleT
        The module that was passed in.

    Raises
    ------
    ValueError
        If a parameter or a buffer is off the run's device, or if a floating one is
        stored in another precision.
    """
    for name, tensor in chain(module.named_parameters(), module.named_buffers()):
        if _elsewhere(tensor.device, policy.device):
            msg = (
                f"{name} is on {tensor.device} while this run is on "
                f"{policy.device}. a module is placed once, whole, by the runner"
            )

            raise ValueError(msg)

        if tensor.is_floating_point() and tensor.dtype is not policy.dtype:
            msg = (
                f"{name} is held in {tensor.dtype} while this run holds "
                f"parameters in {policy.dtype}. a module is placed once, whole, "
                f"by the runner"
            )

            raise ValueError(msg)

    return module


def place[ModuleT: Module](module: ModuleT, policy: DevicePolicy, /) -> ModuleT:
    """Move a whole module onto a run's device and precision.

    Parameters
    ----------
    module : ModuleT
        The module to place, with every submodule it will own already built.
    policy : DevicePolicy
        Where this run's tensors are placed and what precision they are stored in.

    Returns
    -------
    ModuleT
        The module that was passed in, now placed.

    Raises
    ------
    ValueError
        If a tensor still violates the run's device or precision policy after placement.
    """
    return ensure_placed(module.to(device=policy.device, dtype=policy.dtype), policy)
