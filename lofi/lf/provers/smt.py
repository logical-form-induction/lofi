"""Long-lived incremental first-order entailment through SMT solvers."""

import os
import shutil
import subprocess
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.lf import Abstain, CounterSatisfiable, Proof, Proved, ProverBound, Refuted
from lofi.lf.provers._capability import (
    FIRST_ORDER_CAPABILITY,
    TranslatorCapability,
    joint_capability,
)
from lofi.lf.provers._sandbox import sandbox_environment
from lofi.lf.provers._translate import PolymorphismPolicy, smt_query, truth_type

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import IO

    from lofi.lf import Verdict
    from lofi.lf.provers._translate import _Language
    from lofi.terms import Term

__all__ = ["CVC5", "Z3", "SmtEngine", "SmtOracle"]


@final
class SmtEngine(FrozenValue):
    """One registered incremental SMT-LIB command."""

    _noun = "SMT engine"
    __slots__ = (
        "_cpu_option",
        "_memory_option",
        "_name",
        "_prefix",
        "_seed_option",
        "_serves",
    )

    def __init__(
        self,
        name: str,
        prefix: tuple[str, ...],
        *,
        serves: TranslatorCapability = FIRST_ORDER_CAPABILITY,
        memory_option: str | None = None,
        cpu_option: str | None = None,
        seed_option: str | None = None,
    ) -> None:
        object.__setattr__(self, "_name", ensure_name("an SMT engine", name))
        object.__setattr__(
            self,
            "_prefix",
            tuple(ensure_name("an SMT engine argument", value) for value in prefix),
        )
        object.__setattr__(self, "_serves", serves)
        object.__setattr__(self, "_memory_option", memory_option)
        object.__setattr__(self, "_cpu_option", cpu_option)
        object.__setattr__(self, "_seed_option", seed_option)

    @property
    def name(self) -> str:
        """Return the registry tag and default executable name."""
        return self._name

    @property
    def prefix(self) -> tuple[str, ...]:
        """Return the arguments a declaration with no prefix of its own runs."""
        return self._prefix

    def arguments(
        self,
        executable: str,
        extra: tuple[str, ...],
        /,
        *,
        prefix: tuple[str, ...],
        memory_mb: int | None,
        cpu_seconds: int | None,
        seed: int | None,
    ) -> tuple[str, ...]:
        """Return the argument vector of one incremental worker.

        The prefix leads, then the engine's resource options and the declared
        extra arguments.
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
                msg = f"SMT engine {self._name!r} does not support resource {field!r}"
                raise ValueError(msg)

            resources.append(template.format(value=value))

        return executable, *prefix, *resources, *extra

    @property
    def serves(self) -> TranslatorCapability:
        """Return the fragment jointly served by this engine and SMT translation."""
        return joint_capability(FIRST_ORDER_CAPABILITY, self._serves)

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is SmtEngine and (
            self._name,
            self._prefix,
            self._serves,
            self._memory_option,
            self._cpu_option,
            self._seed_option,
        ) == (
            other._name,
            other._prefix,
            other._serves,
            other._memory_option,
            other._cpu_option,
            other._seed_option,
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                SmtEngine,
                self._name,
                self._prefix,
                self._serves,
                self._memory_option,
                self._cpu_option,
                self._seed_option,
            )
        )


Z3 = SmtEngine(
    "z3",
    ("-in", "-smt2"),
    memory_option="-memory:{value}",
    cpu_option="-T:{value}",
    seed_option="smt.random_seed={value}",
)
"""The packaged incremental Z3 command."""

CVC5 = SmtEngine(
    "cvc5",
    ("--lang=smt2", "--incremental", "--produce-models", "-"),
    seed_option="--seed={value}",
)
"""The packaged incremental cvc5 command."""


def _executable(engine: SmtEngine, supplied: str | None, /) -> str:
    """Resolve an executable once at the oracle boundary."""
    candidate = shutil.which(engine.name) if supplied is None else supplied

    if candidate is None:
        msg = f"SMT engine {engine.name!r} is not installed"
        raise FileNotFoundError(msg)

    path = Path(candidate).expanduser().resolve()

    if not path.is_file() or not os.access(path, os.X_OK):
        msg = f"SMT executable is not executable: {str(path)!r}"
        raise ValueError(msg)

    return str(path)


@final
class _Responses:
    """The standard output of one solver worker, read as it arrives.

    A pipe cannot be awaited with a deadline on every platform, so a thread
    reads the worker's output into a queue, and a read waits on the queue
    until its deadline instead.
    """

    __slots__ = ("_buffer", "_chunks", "_ended")

    def __init__(self, stream: IO[bytes], /) -> None:
        self._chunks: SimpleQueue[bytes] = SimpleQueue()
        self._buffer = bytearray()
        self._ended = False
        Thread(target=self._pump, args=(stream,), daemon=True).start()

    def _pump(self, stream: IO[bytes], /) -> None:
        """Queue each chunk of output as it arrives, then an empty chunk."""
        descriptor = stream.fileno()

        while chunk := os.read(descriptor, 4096):
            self._chunks.put(chunk)

        self._chunks.put(b"")

    def _arrive(self, deadline: float, /) -> bool:
        """Take the next chunk of output, or return False at the deadline."""
        remaining = deadline - monotonic()

        if remaining <= 0.0:
            return False

        try:
            chunk = self._chunks.get(timeout=remaining)

        except Empty:
            return False

        if chunk:
            self._buffer.extend(chunk)

        else:
            self._ended = True

        return True

    def line(self, timeout: float, /) -> str | None:
        """Return the next response line, or None if none arrives in time.

        Once the worker's output has ended, the rest of it is returned, which
        is empty when nothing is left.
        """
        deadline = monotonic() + timeout

        while b"\n" not in self._buffer and not self._ended:
            if not self._arrive(deadline):
                return None

        cut = self._buffer.find(b"\n") + 1 or len(self._buffer)
        line = bytes(self._buffer[:cut])
        del self._buffer[:cut]

        return line.decode()

    def block(self, timeout: float, /) -> str | None:
        """Return one atom or balanced SMT-LIB response, or None if none arrives.

        None is also returned when the worker's output ends before the
        response is complete.
        """
        deadline = monotonic() + timeout

        while True:
            text = self._buffer.decode()
            stripped = text.strip()

            if stripped:
                opened = "(" in stripped
                balance = stripped.count("(") - stripped.count(")")

                if (not opened and "\n" in text) or (opened and balance <= 0):
                    self._buffer.clear()
                    return stripped

            if self._ended or not self._arrive(deadline):
                return None


@final
class SmtOracle:
    """An entailment oracle with one incremental solver process."""

    __slots__ = (
        "_bound",
        "_certificates",
        "_closed",
        "_command",
        "_declared",
        "_engine",
        "_language",
        "_lock",
        "_logic",
        "_options",
        "_polymorphism",
        "_process",
        "_responses",
        "_sandbox",
        "_starts",
        "_terminate_grace",
    )

    def __init__(
        self,
        language: _Language,
        /,
        *,
        engine: SmtEngine,
        executable: str | None,
        bound: ProverBound,
        logic: str,
        options: Mapping[str, str],
        prefix: tuple[str, ...] | None,
        arguments: tuple[str, ...],
        terminate_grace: float,
        polymorphism: PolymorphismPolicy,
        certificates: frozenset[str],
        memory_mb: int | None,
        cpu_seconds: int | None,
        seed: int | None,
    ) -> None:
        """Bind one incremental solver worker to a declared language."""
        bound.check(language.profile)
        self._language = language
        self._engine = engine
        resolved_executable = _executable(engine, executable)
        self._command = engine.arguments(
            resolved_executable,
            arguments,
            prefix=engine.prefix if prefix is None else prefix,
            memory_mb=memory_mb,
            cpu_seconds=cpu_seconds,
            seed=seed,
        )
        truth_type(language)
        self._bound = bound
        self._logic = ensure_name("an SMT-LIB logic", logic)
        held_options = dict(options)

        if "model" in certificates:
            held_options["produce-models"] = "true"

        if "proof" in certificates:
            held_options["produce-proofs"] = "true"

        self._options = tuple(sorted(held_options.items()))
        self._terminate_grace = terminate_grace
        self._polymorphism = polymorphism
        self._certificates = certificates
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: _Responses | None = None
        self._declared: set[str] = set()
        self._sandbox = TemporaryDirectory(prefix="lofi-smt-")
        self._starts = 0
        self._closed = False

    @property
    def bound(self) -> ProverBound:
        """Return the exact first-order profile served by this oracle."""
        return self._bound

    @property
    def process_starts(self) -> int:
        """Return how many incremental solver processes have been started."""
        return self._starts

    def close(self) -> None:
        """Stop this worker's incremental solver process."""
        with self._lock:
            if self._closed:
                return

            self._stop()
            self._sandbox.cleanup()
            self._closed = True

    def _stop(self) -> None:
        """Stop the process while the oracle lock is held."""
        process = self._process
        self._process = None
        self._responses = None
        self._declared.clear()

        if process is None:
            return

        if process.stdin is not None:
            process.stdin.close()

        if process.poll() is not None:
            return

        process.terminate()

        try:
            process.wait(timeout=self._terminate_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _worker(self) -> tuple[subprocess.Popen[bytes], _Responses]:
        """Return the live incremental worker, starting it once if needed."""
        process = self._process
        responses = self._responses

        if self._closed:
            msg = "the SMT oracle is closed"
            raise RuntimeError(msg)

        if process is not None and responses is not None and process.poll() is None:
            return process, responses

        process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._sandbox.name,
            env=sandbox_environment(self._sandbox.name),
            close_fds=True,
            start_new_session=True,
        )

        if process.stdout is None:
            msg = "the SMT worker has no standard output"
            raise RuntimeError(msg)

        responses = _Responses(process.stdout)
        self._process = process
        self._responses = responses
        self._starts += 1
        self._declared.clear()
        self._write(
            process,
            (
                f"(set-logic {self._logic})",
                *(f"(set-option :{name} {value})" for name, value in self._options),
            ),
        )
        return process, responses

    @staticmethod
    def _write(process: subprocess.Popen[bytes], lines: Sequence[str], /) -> None:
        """Write complete SMT commands to a live worker."""
        stream = process.stdin

        if stream is None:
            msg = "the SMT worker has no standard input"
            raise RuntimeError(msg)

        stream.write(("\n".join(lines) + "\n").encode())
        stream.flush()

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        """Decide one entailment within a hard worker-response deadline."""
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
        """Translate and submit one unremembered incremental query."""
        started = monotonic()
        query = smt_query(
            self._language,
            premise,
            hypothesis,
            theory=theory,
            polymorphism=self._polymorphism,
        )

        with self._lock:
            process, responses = self._worker()
            fresh = tuple(
                declaration
                for declaration in query.declarations
                if declaration not in self._declared
            )
            self._write(process, fresh)
            self._declared.update(fresh)
            self._write(process, ("(push 1)", *query.assertions, "(check-sat)"))
            remaining = timeout - (monotonic() - started)

            if remaining <= 0.0:
                self._stop()
                return Abstain(
                    "the SMT deadline elapsed before a response", cause="timeout"
                )

            response = responses.line(remaining)

            if response is None:
                self._stop()
                return Abstain(
                    "the SMT solver exceeded its wall clock", cause="timeout"
                )

            answer = response.strip()

            detail = answer

            if answer == "sat" and "model" in self._certificates:
                self._write(process, ("(get-model)",))
                certificate_time = timeout - (monotonic() - started)
                model = (
                    None
                    if certificate_time <= 0.0
                    else responses.block(certificate_time)
                )

                if model is None:
                    self._stop()
                    return Abstain(
                        "the SMT deadline elapsed before the requested model",
                        cause="timeout",
                    )

                detail = model.strip()
            elif answer == "unsat" and "proof" in self._certificates:
                self._write(process, ("(get-proof)",))
                certificate_time = timeout - (monotonic() - started)
                proof = (
                    None
                    if certificate_time <= 0.0
                    else responses.block(certificate_time)
                )

                if proof is None:
                    self._stop()
                    return Abstain(
                        "the SMT deadline elapsed before the requested proof",
                        cause="timeout",
                    )

                detail = proof.strip()

            if answer in ("sat", "unsat", "unknown"):
                self._write(process, ("(pop 1)",))
            else:
                self._stop()

        return self._verdict(answer, detail)

    def _verdict(self, answer: str, detail: str, /) -> Verdict:
        """Map one SMT response and its requested evidence to a verdict."""
        if answer == "unsat":
            return Proved(Proof(self._engine.name, detail))

        if answer == "sat":
            return Refuted(CounterSatisfiable(self._engine.name, "sat", detail))

        if answer == "unknown":
            return Abstain("the SMT solver returned unknown", cause="gave-up")

        return Abstain(
            answer or "the SMT solver closed without a response",
            cause="gave-up",
        )
