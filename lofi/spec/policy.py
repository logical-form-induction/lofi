"""Device and precision configuration."""

import enum
from typing import Self

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)

__all__ = [
    "ComputeDtype",
    "DevicePolicySpec",
    "GradScalerSpec",
    "MatmulPrecision",
]


class ComputeDtype(enum.StrEnum):
    """Enumerates declared floating-point compute types."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class MatmulPrecision(enum.StrEnum):
    """Enumerates declared float32 matrix-multiplication precision levels."""

    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"


class GradScalerSpec(dx.Model, extra="forbid"):
    """Declares dynamic gradient-scaling parameters."""

    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    @dx.validates("init_scale")
    @classmethod
    def _init_scale_is_positive(cls, value: float) -> float:
        """Return a positive finite initial scale."""
        scale = ensure_finite("initial gradient scale", value)

        if scale <= 0.0:
            msg = f"initial gradient scale must be positive: {value!r}"
            raise ValueError(msg)

        return scale

    @dx.validates("growth_factor")
    @classmethod
    def _growth_factor_exceeds_one(cls, value: float) -> float:
        """Return a finite growth factor greater than one."""
        factor = ensure_finite("gradient-scale growth factor", value)

        if factor <= 1.0:
            msg = f"gradient-scale growth factor must exceed one: {value!r}"
            raise ValueError(msg)

        return factor

    @dx.validates("backoff_factor")
    @classmethod
    def _backoff_factor_is_fractional(cls, value: float) -> float:
        """Return a finite backoff factor strictly between zero and one."""
        factor = ensure_finite("gradient-scale backoff factor", value)

        if not 0.0 < factor < 1.0:
            msg = (
                "gradient-scale backoff factor must lie between zero and one: "
                f"{value!r}"
            )
            raise ValueError(msg)

        return factor

    @dx.validates("growth_interval")
    @classmethod
    def _growth_interval_is_positive(cls, value: int) -> int:
        """Return a positive growth interval."""
        interval = ensure_non_negative("gradient-scale growth interval", value)

        if interval == 0:
            msg = "gradient-scale growth interval must be positive: 0"
            raise ValueError(msg)

        return interval


class DevicePolicySpec(dx.Model, extra="forbid"):
    """Declares device, precision and memory settings for one run.

    ``activation_memory_bytes`` is the peak bytes one training step may hold
    on the declared device. Pricing names a stack whose measured peak exceeds
    it, and an array launch leaves that stack's points out. On a device whose
    allocator keeps a peak count, training measures every step against it and
    stops the run with ``ActivationMemoryExceededError`` at the first step
    above it. The processor keeps no such count, so there the bound prices and
    sizes an array task's memory request but no step is measured. ``scaler`` is
    the dynamic gradient scaling every trainer steps under, or None for
    unscaled steps. ``checkpoint_from_width`` is the
    narrowest span width whose chart cells are recomputed during the backward
    pass rather than held from the forward pass, trading compute for activation
    memory on long inputs; None holds every cell's activations.
    """

    device: str
    compute_dtype: ComputeDtype
    deterministic: bool
    matmul_precision: MatmulPrecision
    activation_memory_bytes: int
    scaler: GradScalerSpec | None = None
    checkpoint_from_width: int | None = None

    @dx.validates("checkpoint_from_width")
    @classmethod
    def _checkpointing_starts_at_a_composed_width(cls, value: int | None) -> int | None:
        """Return a checkpointing width of at least two, or None."""
        if value is None:
            return None

        if ensure_positive("checkpoint_from_width", value) < 2:
            msg = (
                f"chart cells are composed from width two, so checkpoint_from_width "
                f"is at least two, not {value}"
            )
            raise ValueError(msg)

        return value

    @dx.validates("device")
    @classmethod
    def _device_is_named(cls, value: str) -> str:
        """Return a nonempty device declaration."""
        return ensure_name("device", value)

    @dx.validates("activation_memory_bytes")
    @classmethod
    def _activation_memory_is_positive(cls, value: int) -> int:
        """Return a positive activation-memory ceiling."""
        return ensure_positive("activation memory budget", value)

    @dx.model_validator()
    def _bfloat16_has_no_scaler(self) -> Self:
        """Return a policy whose bfloat16 computation has no scaler."""
        if self.compute_dtype is ComputeDtype.BFLOAT16 and self.scaler is not None:
            msg = "bfloat16 computation does not use a gradient scaler"
            raise ValueError(msg)

        return self
