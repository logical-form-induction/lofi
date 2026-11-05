"""Sandboxed typed entailment through TPTP provers."""

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.lf import (
    Abstain,
    CounterSatisfiable,
    Proof,
    Proved,
    ProverBound,
    Refuted,
    Verdict,
)
from lofi.lf.provers._capability import TranslatorCapability, joint_capability
from lofi.lf.provers._sandbox import sandbox_environment
from lofi.lf.provers._translate import (
    TPTP_DIALECT_CAPABILITIES,
    FirstOrderTranslationError,
    PolymorphismPolicy,
    TptpDialect,
    tptp_problem,
)
from lofi.lf.provers._tree import ProcessTree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.lf.provers._translate import _Language
    from lofi.terms import Term

__all__ = [
    "EPROVER",
    "LEO3",
    "VAMPIRE",
    "TptpEngine",
    "TptpOracle",
]


class _TptpEngineIdentity(NamedTuple):
    name: str
    dialects: tuple[TptpDialect, ...]
    prefix: tuple[str, ...]
    serves: tuple[tuple[TptpDialect, TranslatorCapability], ...]
    proved: tuple[str, ...]
    refuted: tuple[str, ...]
    timeout: tuple[str, ...]
    resource: tuple[str, ...]
    memory_option: str | None
    cpu_option: str | None
    seed_option: str | None


@final
class TptpEngine(FrozenValue):
    """One registered TPTP process and SZS interpretation."""

    _noun = "TPTP engine"
    __slots__ = (
        "_cpu_option",
        "_dialects",
        "_memory_option",
        "_name",
        "_prefix",
        "_proved",
        "_refuted",
        "_resource",
        "_seed_option",
        "_serves",
        "_timeout",
    )

    def __init__(
        self,
        name: str,
        *,
        dialects: tuple[TptpDialect, ...],
        prefix: tuple[str, ...] = (),
        proved: tuple[str, ...] = ("Theorem", "Unsatisfiable"),
        refuted: tuple[str, ...] = ("CounterSatisfiable", "Satisfiable"),
        timeout: tuple[str, ...] = ("Timeout",),
        resource: tuple[str, ...] = ("ResourceOut", "MemoryOut"),
        memory_option: str | None = None,
        cpu_option: str | None = None,
        seed_option: str | None = None,
        serves: Mapping[TptpDialect, TranslatorCapability] | None = None,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("a TPTP engine", name))
        object.__setattr__(self, "_dialects", frozenset(dialects))
        object.__setattr__(self, "_prefix", tuple(prefix))
        object.__setattr__(self, "_proved", frozenset(proved))
        object.__setattr__(self, "_refuted", frozenset(refuted))
        object.__setattr__(self, "_timeout", frozenset(timeout))
        object.__setattr__(self, "_resource", frozenset(resource))
        object.__setattr__(self, "_memory_option", memory_option)
        object.__setattr__(self, "_cpu_option", cpu_option)
        object.__setattr__(self, "_seed_option", seed_option)
        derived = {dialect: TPTP_DIALECT_CAPABILITIES[dialect] for dialect in dialects}
        object.__setattr__(
            self,
            "_serves",
            MappingProxyType(derived if serves is None else dict(serves)),
        )

        if set(self._serves) != set(self._dialects):
            msg = "a TPTP engine declares one served fragment per dialect"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return the registry tag and default executable name."""
        return self._name

    @property
    def prefix(self) -> tuple[str, ...]:
        """Return the arguments a declaration with no prefix of its own runs."""
        return self._prefix

    def check_dialect(self, dialect: TptpDialect, /) -> None:
        """Refuse a dialect this engine does not advertise."""
        if dialect not in self._dialects:
            msg = (
                f"TPTP engine {self._name!r} does not serve dialect {dialect!r}; "
                f"served dialects are {sorted(self._dialects)!r}"
            )
            raise ValueError(msg)

    def serves(self, dialect: TptpDialect, /) -> TranslatorCapability:
        """Return the fragment served for one checked dialect."""
        self.check_dialect(dialect)
        return joint_capability(
            TPTP_DIALECT_CAPABILITIES[dialect], self._serves[dialect]
        )

    def arguments(
        self,
        executable: str,
        problem: Path,
        /,
        *,
        prefix: tuple[str, ...],
        options: tuple[str, ...],
        memory_mb: int | None,
        cpu_seconds: int | None,
        seed: int | None,
    ) -> tuple[str, ...]:
        """Return the argument vector of one declared process.

        The prefix leads, then the engine's resource options, the declared
        options and the problem path.
        """
        resources: list[str] = []

        for field, template, value in (
            ("memory_mb", self._memory_option, memory_mb),
            ("cpu_seconds", self._cpu_option, cpu_seconds),
            ("seed", self._seed_option, seed),
        ):
            if value is None:
                continue

            if template is None:
                msg = f"TPTP engine {self._name!r} does not support resource {field!r}"
                raise ValueError(msg)

            resources.append(template.format(value=value))

        return executable, *prefix, *resources, *options, str(problem)

    def verdict(
        self,
        output: str,
        status: str | None,
        /,
        *,
        returncode: int,
        certificates: frozenset[str],
    ) -> Verdict:
        """Map one completed SZS report to typed evidence."""
        if status is not None and status in self._proved:
            detail = (
                _certificate(output, status)
                if "proof" in certificates
                else f"SZS status {status}"
            )
            return Proved(Proof(self._name, detail))

        if status is not None and status in self._refuted:
            return Refuted(
                CounterSatisfiable(
                    self._name,
                    status,
                    _certificate(output, status),
                )
            )

        if status is not None and status in self._timeout:
            return Abstain(output or status, cause="timeout")

        if status is not None and status in self._resource:
            return Abstain(output or status, cause="gave-up")

        reason = output.strip() or (
            f"{self._name} exited with status {returncode} without an SZS status"
        )
        return Abstain(reason, cause="gave-up")

    def _identity(self) -> _TptpEngineIdentity:
        return _TptpEngineIdentity(
            self._name,
            tuple(sorted(self._dialects)),
            self._prefix,
            tuple(sorted(self._serves.items())),
            tuple(sorted(self._proved)),
            tuple(sorted(self._refuted)),
            tuple(sorted(self._timeout)),
            tuple(sorted(self._resource)),
            self._memory_option,
            self._cpu_option,
            self._seed_option,
        )

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is TptpEngine and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((TptpEngine, self._identity()))


EPROVER = TptpEngine(
    "eprover",
    dialects=("fof", "tff"),
    prefix=("--auto", "--proof-object=1", "--tstp-format"),
    cpu_option="--cpu-limit={value}",
    memory_option="--memory-limit={value}",
)
"""The packaged E prover command."""

VAMPIRE = TptpEngine(
    "vampire",
    dialects=("fof", "tff", "thf"),
    prefix=("--mode", "casc"),
    cpu_option="--time_limit={value}",
    memory_option="--memory_limit={value}",
    seed_option="--random_seed={value}",
)
"""The packaged Vampire command."""

LEO3 = TptpEngine("leo3", dialects=("thf",))
"""The packaged Leo-III command."""


def _executable(engine: TptpEngine, supplied: str | None, /) -> str:
    """Resolve an executable once at the oracle boundary."""
    candidate = shutil.which(engine.name) if supplied is None else supplied

    if candidate is None:
        msg = f"TPTP engine {engine.name!r} is not installed"
        raise FileNotFoundError(msg)

    path = Path(candidate).expanduser().resolve()

    if not path.is_file() or not os.access(path, os.X_OK):
        msg = f"TPTP executable is not executable: {str(path)!r}"
        raise ValueError(msg)

    return str(path)


def _status(output: str, /) -> str | None:
    """Return the first SZS status word in prover output."""
    for line in output.splitlines():
        _, marker, rest = line.partition("SZS status")
        words = rest.split()

        if marker and words:
            return words[0]

    return None


def _certificate(output: str, status: str, /) -> str:
    """Return the reproducible part of a prover's report.

    The certificate is the SZS output the prover printed, from the first
    ``SZS output start`` line to the last ``SZS output end`` line, or the SZS
    status line when the prover printed no output. Timings, process
    identifiers and strategy seeds the prover reports outside them are left
    out, so deciding one problem twice yields one certificate.
    """
    lines = output.splitlines()
    starts = [at for at, line in enumerate(lines) if "SZS output start" in line]
    ends = [at for at, line in enumerate(lines) if "SZS output end" in line]

    if starts and ends and starts[0] <= ends[-1]:
        return "\n".join(lines[starts[0] : ends[-1] + 1])

    return f"SZS status {status}"


@final
class TptpOracle:
    """An FOF, TFF or THF entailment oracle over a declared TPTP engine."""

    __slots__ = (
        "_bound",
        "_certificates",
        "_command",
        "_dialect",
        "_engine",
        "_language",
        "_polymorphism",
        "_problem_file",
        "_starts",
        "_starts_lock",
    )

    def __init__(
        self,
        language: _Language,
        /,
        *,
        engine: TptpEngine,
        dialect: TptpDialect,
        executable: str | None,
        bound: ProverBound,
        prefix: tuple[str, ...] | None,
        options: tuple[str, ...],
        problem_file: str,
        memory_mb: int | None,
        cpu_seconds: int | None,
        seed: int | None,
        polymorphism: PolymorphismPolicy,
        certificates: frozenset[str],
    ) -> None:
        """Bind a TPTP process configuration to one declared language."""
        engine.check_dialect(dialect)
        bound.check(language.profile)
        self._language = language
        self._engine = engine
        self._dialect = dialect
        resolved_executable = _executable(engine, executable)
        self._command = engine.arguments(
            resolved_executable,
            Path(problem_file),
            prefix=engine.prefix if prefix is None else prefix,
            options=options,
            memory_mb=memory_mb,
            cpu_seconds=cpu_seconds,
            seed=seed,
        )
        self._problem_file = problem_file
        self._bound = bound
        self._polymorphism = polymorphism
        self._certificates = certificates
        self._starts = 0
        self._starts_lock = Lock()

    @property
    def bound(self) -> ProverBound:
        """Return the widest logical profile served by this oracle."""
        return self._bound

    @property
    def process_starts(self) -> int:
        """Return how many sandboxed prover processes have been started."""
        with self._starts_lock:
            return self._starts

    def close(self) -> None:
        """Release resources; TPTP decisions hold no persistent process."""

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        """Decide one entailment within a hard subprocess deadline."""
        held = tuple(theory)
        clock = ensure_finite("a timeout", timeout)

        if clock <= 0.0:
            msg = f"a timeout is a positive number of seconds: {timeout!r}"
            raise ValueError(msg)

        return self._settle(premise, hypothesis, held, clock)

    def _settle(
        self,
        premise: Term,
        hypothesis: Term,
        theory: tuple[Term, ...],
        timeout: float,
    ) -> Verdict:
        """Translate and run one unremembered problem."""
        started = monotonic()
        try:
            problem = tptp_problem(
                self._language,
                premise,
                hypothesis,
                theory=theory,
                dialect=self._dialect,
                polymorphism=self._polymorphism,
            )
        except FirstOrderTranslationError as error:
            return Abstain(str(error), cause="out-of-fragment")

        remaining = timeout - (monotonic() - started)

        if remaining <= 0.0:
            return Abstain(
                "the TPTP deadline elapsed during translation", cause="timeout"
            )

        with TemporaryDirectory(prefix="lofi-tptp-") as directory:
            path = Path(directory) / self._problem_file
            path.write_text(problem, encoding="utf-8", newline="\n")
            with (
                subprocess.Popen(
                    self._command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    cwd=directory,
                    env=sandbox_environment(directory),
                    close_fds=True,
                    start_new_session=True,
                ) as process,
                ProcessTree(process.pid) as tree,
            ):
                with self._starts_lock:
                    self._starts += 1

                try:
                    stdout, stderr = process.communicate(timeout=remaining)
                except subprocess.TimeoutExpired:
                    tree.kill()
                    process.communicate()
                    return Abstain(
                        "the TPTP prover exceeded its wall clock", cause="timeout"
                    )

        output = "\n".join(part for part in (stdout, stderr) if part)

        return self._engine.verdict(
            output,
            _status(output),
            returncode=process.returncode,
            certificates=self._certificates,
        )
