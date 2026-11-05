"""Incremental SMT prover declaration."""

from types import MappingProxyType
from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec.provers._base import (
    ProverServiceSettings,
    ProverSpec,
)

__all__ = ["SmtProverSettings", "SmtProverSpec"]


class SmtProverSettings(NamedTuple):
    """Resolved settings of an incremental SMT prover."""

    engine: str
    executable: str | None
    logic: str
    options: MappingProxyType[str, str]
    prefix: tuple[str, ...] | None
    arguments: tuple[str, ...]
    terminate_grace: float
    service: ProverServiceSettings


class SmtProverSpec(ProverSpec, extra="forbid"):
    """Declare a long-lived entailment oracle using a registered SMT engine.

    Attributes
    ----------
    engine : str
        The registered solver-engine name.
    executable : str | None
        The solver's path, or None to find the engine on the search path.
    logic : str
        The SMT-LIB logic each worker sets.
    options : dict[str, str]
        SMT-LIB options each worker sets after its logic.
    prefix : tuple[str, ...] | None
        The arguments that lead the command, replacing the engine's default
        prefix, or None to run the engine's default.
    arguments : tuple[str, ...]
        Arguments that follow the prefix and the resource options.
    terminate_grace : float
        Seconds a stopped worker has to exit after the terminate signal
        before it is killed.
    """

    kind: Literal["smt"] = "smt"
    engine: str = "z3"
    executable: str | None = None
    logic: str = "ALL"
    options: dict[str, str] = dx.field(default_factory=dict)
    prefix: tuple[str, ...] | None = None
    arguments: tuple[str, ...] = ()
    terminate_grace: float = 0.2

    @dx.validates("engine", "logic")
    @classmethod
    def _names_are_nonempty(cls, value: str) -> str:
        """Return a nonempty engine or SMT-LIB logic name."""
        return ensure_name("an SMT setting", value)

    @dx.validates("options")
    @classmethod
    def _options_are_named(cls, options: dict[str, str]) -> dict[str, str]:
        """Return named SMT-LIB options."""
        return {
            ensure_name("an SMT option", name): ensure_name(
                f"SMT option {name!r} value", value
            )
            for name, value in options.items()
        }

    @dx.validates("prefix")
    @classmethod
    def _prefix_is_named(cls, prefix: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Return nonempty prefix arguments."""
        if prefix is None:
            return None
        return tuple(ensure_name("an SMT prefix argument", item) for item in prefix)

    @dx.validates("arguments")
    @classmethod
    def _arguments_are_named(cls, arguments: tuple[str, ...]) -> tuple[str, ...]:
        """Return nonempty engine arguments."""
        return tuple(
            ensure_name("an SMT engine argument", argument) for argument in arguments
        )

    @dx.validates("terminate_grace")
    @classmethod
    def _grace_is_positive(cls, terminate_grace: float) -> float:
        """Return a finite positive grace period."""
        held = ensure_finite("an SMT terminate grace period", terminate_grace)

        if held <= 0.0:
            msg = f"an SMT terminate grace period is positive: {terminate_grace!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> SmtProverSettings:
        """Return the resolved prover settings."""
        service = self.service_settings()
        return SmtProverSettings(
            self.engine,
            self.executable,
            self.logic,
            MappingProxyType(dict(self.options)),
            self.prefix,
            self.arguments,
            self.terminate_grace,
            service,
        )
