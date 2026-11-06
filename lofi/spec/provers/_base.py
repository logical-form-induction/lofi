"""Shared declarations for registered entailment provers."""

from typing import Literal, NamedTuple, Self

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.spec.components.base import ComponentSpec
from lofi.spec.provers._budget import ProofBudgetSpec

__all__ = [
    "LabelingProverSpec",
    "PolymorphismPolicy",
    "ProverServiceSettings",
    "ProverSpec",
]

type PolymorphismPolicy = Literal["refuse", "monomorphize", "erase"]
"""How a prover translation treats a polymorphic constant occurrence."""


class ProverServiceSettings(NamedTuple):
    """Process and translation settings shared by every prover family."""

    timeout: float
    workers: int
    batch_size: int
    cache_size: int | None
    memory_mb: int | None
    cpu_seconds: int | None
    seed: int | None
    polymorphism: PolymorphismPolicy
    certificates: frozenset[str]


class ProverSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one registered entailment prover and its proof service."""

    timeout: float = 30.0
    workers: int = 1
    batch_size: int = 64
    cache_size: int | None = None
    memory_mb: int | None = None
    cpu_seconds: int | None = None
    seed: int | None = None
    polymorphism: PolymorphismPolicy = "refuse"
    certificates: tuple[str, ...] = ()

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists prover plugins."""
        return "prover"

    @dx.validates("timeout")
    @classmethod
    def _timeout_is_positive(cls, timeout: float) -> float:
        """Return a finite positive per-call timeout."""
        held = ensure_finite("a prover timeout", timeout)

        if held <= 0.0:
            msg = f"a prover timeout is positive: {timeout!r}"
            raise ValueError(msg)

        return held

    @dx.validates("workers")
    @classmethod
    def _workers_are_positive(cls, workers: int) -> int:
        """Return a positive worker-pool size."""
        return ensure_positive("a prover worker count", workers)

    @dx.validates("batch_size")
    @classmethod
    def _batch_size_is_positive(cls, batch_size: int) -> int:
        """Return a positive dispatch batch size."""
        return ensure_positive("a prover batch size", batch_size)

    @dx.model_validator()
    def _bounded_resources_are_positive(self) -> Self:
        """Return a declaration whose optional resource limits are positive."""
        for field, value in (
            ("cache_size", self.cache_size),
            ("memory_mb", self.memory_mb),
            ("cpu_seconds", self.cpu_seconds),
        ):
            if value is not None:
                ensure_positive(f"a prover {field.replace('_', ' ')}", value)

        if self.seed is not None:
            ensure_non_negative("a prover seed", self.seed)

        return self

    @dx.validates("certificates")
    @classmethod
    def _certificates_are_named(cls, certificates: tuple[str, ...]) -> tuple[str, ...]:
        """Return distinct open certificate capability names."""
        held = tuple(
            ensure_name("a prover certificate", certificate)
            for certificate in certificates
        )

        if len(set(held)) != len(held):
            msg = "a prover lists a certificate twice"
            raise ValueError(msg)

        return held

    def service_settings(self) -> ProverServiceSettings:
        """Return common immutable process and translation settings."""
        return ProverServiceSettings(
            timeout=self.timeout,
            workers=self.workers,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            memory_mb=self.memory_mb,
            cpu_seconds=self.cpu_seconds,
            seed=self.seed,
            polymorphism=self.polymorphism,
            certificates=frozenset(self.certificates),
        )


class LabelingProverSpec(dx.Model, extra="forbid"):
    """Declare proof service resources spent while producing data labels."""

    budget: ProofBudgetSpec = dx.field(default_factory=ProofBudgetSpec)
