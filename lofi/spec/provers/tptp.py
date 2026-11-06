"""TPTP prover declaration."""

from pathlib import PurePath
from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.provers._base import (
    ProverServiceSettings,
    ProverSpec,
)

__all__ = ["TptpProverSettings", "TptpProverSpec"]


class TptpProverSettings(NamedTuple):
    """Resolved settings of a TPTP prover."""

    engine: str
    executable: str | None
    dialect: Literal["fof", "tff", "thf"]
    prefix: tuple[str, ...] | None
    options: tuple[str, ...]
    problem_file: str
    service: ProverServiceSettings


class TptpProverSpec(ProverSpec, extra="forbid"):
    """Declare a sandboxed entailment oracle using a registered TPTP engine.

    Attributes
    ----------
    engine : str
        The registered prover-engine name.
    executable : str | None
        The prover's path, or None to find the engine on the search path.
    dialect : Literal["fof", "tff", "thf"]
        Whether problems are untyped, first-order typed or higher-order typed.
    prefix : tuple[str, ...] | None
        The arguments that lead the command, replacing the engine's default
        prefix, or None to run the engine's default.
    options : tuple[str, ...]
        Arguments that follow the prefix and the resource options.
    problem_file : str
        The file name, inside the call's sandbox directory, that holds the
        problem and is passed to the prover as its last argument.
    """

    kind: Literal["tptp"] = "tptp"
    engine: str = "eprover"
    executable: str | None = None
    dialect: Literal["fof", "tff", "thf"] = "fof"
    prefix: tuple[str, ...] | None = None
    options: tuple[str, ...] = ()
    problem_file: str = "problem.p"

    @dx.validates("engine")
    @classmethod
    def _engine_is_named(cls, engine: str) -> str:
        """Return an open registered engine name."""
        return ensure_name("a TPTP engine", engine)

    @dx.validates("prefix")
    @classmethod
    def _prefix_is_named(cls, prefix: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Return nonempty prefix arguments."""
        if prefix is None:
            return None
        return tuple(ensure_name("a TPTP prefix argument", item) for item in prefix)

    @dx.validates("options")
    @classmethod
    def _options_are_named(cls, options: tuple[str, ...]) -> tuple[str, ...]:
        """Return nonempty engine options."""
        return tuple(ensure_name("a TPTP engine option", option) for option in options)

    @dx.validates("problem_file")
    @classmethod
    def _problem_file_is_a_file_name(cls, problem_file: str) -> str:
        """Return one plain file name inside the sandbox directory."""
        held = ensure_name("a TPTP problem file", problem_file)

        if PurePath(held).name != held or held in {".", ".."}:
            msg = f"a TPTP problem file is one plain file name: {problem_file!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> TptpProverSettings:
        """Return the resolved prover settings."""
        service = self.service_settings()
        return TptpProverSettings(
            self.engine,
            self.executable,
            self.dialect,
            self.prefix,
            self.options,
            self.problem_file,
            service,
        )
