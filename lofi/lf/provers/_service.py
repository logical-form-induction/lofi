"""Budgeted, memoized dispatch for symbolic entailment questions."""

from concurrent.futures import Future, ThreadPoolExecutor
from queue import SimpleQueue
from time import monotonic, thread_time
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, override, runtime_checkable

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.lf._verdict import (
    ABSTENTION_CAUSES,
    Abstain,
    AbstentionCause,
    Graded,
    Proof,
    Proved,
    Refuted,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.lf._ports import EntailmentOracle
    from lofi.lf._profile import LogicProfile
    from lofi.lf._verdict import Verdict
    from lofi.terms import Term

__all__ = ["ProofBudget", "ProofService", "ProofSite", "Questions", "ServiceReport"]

type _MemoKey = tuple[str, Term, Term, tuple[Term, ...]]
type _Settled = Proved | Refuted | Graded


@final
class ProofSite(FrozenValue):
    """Stable identity of one consumer of a shared proof service."""

    _noun = "proof site"
    __slots__ = ("_name",)

    def __init__(self, name: str, /) -> None:
        object.__setattr__(self, "_name", ensure_name("a proof site", name))

    @property
    def name(self) -> str:
        """Return the declared use-site name."""
        return self._name

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is ProofSite and self._name == other._name

    @override
    def __hash__(self) -> int:
        return hash((ProofSite, self._name))

    @override
    def __repr__(self) -> str:
        return f"ProofSite({self._name!r})"


class _SiteAccounting:
    """Mutable lifetime and current-step accounting for one proof site."""

    __slots__ = (
        "abstentions",
        "calls",
        "cpu",
        "hits",
        "lifetime_cpu",
        "process_starts",
        "questions",
        "started",
    )

    def __init__(self) -> None:
        self.lifetime_cpu = 0.0
        self.open_step()

    def open_step(self) -> None:
        """Reset this site's current-step counters."""
        self.calls = 0
        self.hits = 0
        self.cpu = 0.0
        self.questions = 0
        self.abstentions = dict.fromkeys(ABSTENTION_CAUSES, 0)
        self.process_starts = 0
        self.started = monotonic()


@runtime_checkable
class _ProcessOracle(Protocol):
    """An oracle that can report how many solver processes it started."""

    @property
    def process_starts(self) -> int:
        """Return cumulative solver-process starts."""


@runtime_checkable
class _ClosableOracle(Protocol):
    """An oracle owning resources released with the service."""

    def close(self) -> None:
        """Release the oracle's resources."""


def _close_oracles(oracles: tuple[EntailmentOracle, ...], /) -> None:
    """Close each resource-owning oracle identity at most once."""
    closed: set[int] = set()

    for oracle in oracles:
        identity = id(oracle)

        if identity not in closed and isinstance(oracle, _ClosableOracle):
            oracle.close()
            closed.add(identity)


def _validate_oracles(oracles: tuple[EntailmentOracle, ...], workers: int, /) -> None:
    """Require one distinct worker per oracle and one common prover."""
    if len({id(oracle) for oracle in oracles}) != workers:
        msg = "a proof-service factory returns one distinct oracle per worker"
        raise ValueError(msg)

    provers = {oracle.bound.prover for oracle in oracles}

    if len(provers) != 1:
        msg = f"a proof service's workers use one prover, not {sorted(provers)!r}"
        raise ValueError(msg)


@final
class Questions(FrozenValue):
    """Distinct entailment questions sharing one background theory."""

    _noun = "entailment questions"
    __slots__ = ("_goals", "_premises", "_theory")

    def __init__(
        self,
        premises: tuple[Term, ...],
        goals: tuple[Term, ...],
        /,
        *,
        theory: tuple[Term, ...] = (),
    ) -> None:
        if len(premises) != len(goals):
            msg = (
                "entailment questions pair every premise with one goal: "
                f"{len(premises)} premises and {len(goals)} goals"
            )
            raise ValueError(msg)

        seen: set[tuple[Term, Term]] = set()
        distinct: list[tuple[Term, Term]] = []

        for pair in zip(premises, goals, strict=True):
            if pair not in seen:
                seen.add(pair)
                distinct.append(pair)

        object.__setattr__(self, "_premises", tuple(pair[0] for pair in distinct))
        object.__setattr__(self, "_goals", tuple(pair[1] for pair in distinct))
        object.__setattr__(self, "_theory", tuple(theory))

    @property
    def premises(self) -> tuple[Term, ...]:
        """Return distinct premises in first-occurrence order."""
        return self._premises

    @property
    def goals(self) -> tuple[Term, ...]:
        """Return the goal paired with each premise."""
        return self._goals

    @property
    def theory(self) -> tuple[Term, ...]:
        """Return the common theory assumed by every question."""
        return self._theory

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Questions
            and self._premises == other._premises
            and self._goals == other._goals
            and self._theory == other._theory
        )

    @override
    def __hash__(self) -> int:
        return hash((Questions, self._premises, self._goals, self._theory))

    @override
    def __repr__(self) -> str:
        return (
            f"Questions({self._premises!r}, {self._goals!r}, theory={self._theory!r})"
        )


def _optional_count(field: str, value: int | None, /) -> int | None:
    return None if value is None else ensure_non_negative(field, value)


def _optional_seconds(field: str, value: float | None, /) -> float | None:
    if value is None:
        return None

    held = ensure_finite(field, value)

    if held < 0.0:
        msg = f"{field} is non-negative: {value!r}"
        raise ValueError(msg)

    return held


@final
class ProofBudget(FrozenValue):
    """The only resource budget governing a proof-service use site."""

    _noun = "proof budget"
    __slots__ = ("_calls", "_cpu", "_wall")

    def __init__(
        self,
        *,
        calls_per_step: int | None = None,
        wall_per_step: float | None = None,
        cpu_seconds: float | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_calls",
            _optional_count("proof calls per step", calls_per_step),
        )
        object.__setattr__(
            self,
            "_wall",
            _optional_seconds("proof wall seconds per step", wall_per_step),
        )
        object.__setattr__(
            self,
            "_cpu",
            _optional_seconds("proof CPU seconds", cpu_seconds),
        )

    @property
    def calls_per_step(self) -> int | None:
        return self._calls

    @property
    def wall_per_step(self) -> float | None:
        return self._wall

    @property
    def cpu_seconds(self) -> float | None:
        """Return this site's cumulative CPU-time limit."""
        return self._cpu

    def _identity(self) -> tuple[int | None, float | None, float | None]:
        return self._calls, self._wall, self._cpu

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is ProofBudget and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((ProofBudget, self._identity()))

    @override
    def __repr__(self) -> str:
        return (
            f"ProofBudget(calls_per_step={self._calls!r}, "
            f"wall_per_step={self._wall!r}, "
            f"cpu_seconds={self._cpu!r})"
        )


@final
class ServiceReport(FrozenValue):
    """Cumulative proof throughput, memo use, abstentions, and CPU time."""

    _noun = "proof-service report"
    __slots__ = (
        "_abstentions",
        "_calls",
        "_cpu",
        "_hits",
        "_process_starts",
        "_questions",
        "_wall",
    )

    def __init__(
        self,
        calls: int,
        memo_hits: int,
        abstentions: Mapping[AbstentionCause, int],
        cpu_seconds: float,
        questions: int,
        wall_seconds: float,
        process_starts: int | None,
    ) -> None:
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_hits", memo_hits)
        object.__setattr__(self, "_abstentions", MappingProxyType(dict(abstentions)))
        object.__setattr__(self, "_cpu", cpu_seconds)
        object.__setattr__(self, "_questions", questions)
        object.__setattr__(self, "_wall", wall_seconds)
        object.__setattr__(self, "_process_starts", process_starts)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def memo_hits(self) -> int:
        return self._hits

    @property
    def abstentions(self) -> Mapping[AbstentionCause, int]:
        return self._abstentions

    @property
    def cpu_seconds(self) -> float:
        return self._cpu

    @property
    def questions(self) -> int:
        """Return distinct questions presented to the service."""
        return self._questions

    @property
    def wall_seconds(self) -> float:
        """Return elapsed service lifetime in seconds."""
        return self._wall

    @property
    def questions_per_second(self) -> float:
        """Return distinct questions presented per elapsed wall second."""
        return 0.0 if self._wall == 0.0 else self._questions / self._wall

    @property
    def memo_hit_rate(self) -> float:
        """Return the decided-query share answered by the memo."""
        decided = self._calls + self._hits
        return 0.0 if decided == 0 else self._hits / decided

    @property
    def process_starts(self) -> int | None:
        """Return solver process starts, or None for a process-free oracle."""
        return self._process_starts

    def _identity(
        self,
    ) -> tuple[
        int,
        int,
        tuple[tuple[str, int], ...],
        float,
        int,
        float,
        int | None,
    ]:
        return (
            self._calls,
            self._hits,
            tuple(sorted(self._abstentions.items())),
            self._cpu,
            self._questions,
            self._wall,
            self._process_starts,
        )

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is ServiceReport and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((ServiceReport, self._identity()))

    @override
    def __repr__(self) -> str:
        return (
            f"ServiceReport(calls={self._calls!r}, memo_hits={self._hits!r}, "
            f"abstentions={dict(self._abstentions)!r}, cpu_seconds={self._cpu!r}, "
            f"questions={self._questions!r}, wall_seconds={self._wall!r}, "
            f"process_starts={self._process_starts!r})"
        )


@final
class ProofService:
    """The single process-local dispatcher for one oracle and logic profile."""

    __slots__ = (
        "_abstentions",
        "_batch_size",
        "_cache_size",
        "_calls",
        "_closed",
        "_cpu",
        "_executor",
        "_hits",
        "_memo",
        "_oracles",
        "_pool",
        "_profile",
        "_questions",
        "_sites",
        "_started",
        "_timeout",
        "_workers",
    )

    def __init__(
        self,
        oracle: Callable[[], EntailmentOracle],
        /,
        *,
        profile: LogicProfile,
        workers: int,
        batch_size: int,
        timeout: float,
        cache_size: int | None = None,
    ) -> None:
        held_workers = ensure_non_negative("proof-service workers", workers)
        held_batch = ensure_non_negative("proof-service batch size", batch_size)
        held_timeout = ensure_finite("a proof-call timeout", timeout)
        held_cache = _optional_count("proof-service cache size", cache_size)

        if held_workers == 0:
            msg = "a proof service has at least one worker"
            raise ValueError(msg)

        if held_batch == 0:
            msg = "a proof service has a positive batch size"
            raise ValueError(msg)

        if held_timeout <= 0.0:
            msg = f"a proof-call timeout is positive: {timeout!r}"
            raise ValueError(msg)

        if held_cache == 0:
            msg = "a proof-service cache size is positive when declared"
            raise ValueError(msg)

        built: list[EntailmentOracle] = []

        try:
            for _ in range(held_workers):
                instance = oracle()
                built.append(instance)
                instance.bound.check(profile)

            _validate_oracles(tuple(built), held_workers)
        except Exception:
            _close_oracles(tuple(built))
            raise

        oracles = tuple(built)

        self._oracles = oracles
        self._pool: SimpleQueue[EntailmentOracle] = SimpleQueue()

        for instance in oracles:
            self._pool.put(instance)

        self._profile = profile
        self._workers = held_workers
        self._batch_size = held_batch
        self._cache_size = held_cache
        self._timeout = held_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=held_workers,
            thread_name_prefix="lofi-proof",
        )
        self._memo: dict[_MemoKey, _Settled] = {}
        self._abstentions: dict[AbstentionCause, int] = dict.fromkeys(
            ABSTENTION_CAUSES, 0
        )
        self._calls = 0
        self._hits = 0
        self._cpu = 0.0
        self._questions = 0
        self._started = monotonic()
        self._closed = False
        self._sites: dict[ProofSite, _SiteAccounting] = {}

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def batch_size(self) -> int:
        """Return the most questions submitted in one dispatch wave."""
        return self._batch_size

    @property
    def timeout(self) -> float:
        """Return the wall-clock limit of one proof call."""
        return self._timeout

    @property
    def cache_size(self) -> int | None:
        """Return the memo capacity, or None when it is unbounded."""
        return self._cache_size

    @property
    def prover(self) -> str:
        """Return the identity of the oracle this service dispatches."""
        return self._oracles[0].bound.prover

    @property
    def report(self) -> ServiceReport:
        """Return cumulative service-lifetime telemetry."""
        return ServiceReport(
            self._calls,
            self._hits,
            self._abstentions,
            self._cpu,
            self._questions,
            monotonic() - self._started,
            self._process_starts(),
        )

    def step_report(self, site: ProofSite, /) -> ServiceReport:
        """Return telemetry for one site since its most recent step boundary."""
        accounting = self._site(site)
        return ServiceReport(
            accounting.calls,
            accounting.hits,
            accounting.abstentions,
            accounting.cpu,
            accounting.questions,
            monotonic() - accounting.started,
            accounting.process_starts if self._process_starts() is not None else None,
        )

    def _process_starts(self) -> int | None:
        """Return cumulative process starts when every worker reports them."""
        process_oracles = tuple(
            oracle for oracle in self._oracles if isinstance(oracle, _ProcessOracle)
        )
        return (
            sum(oracle.process_starts for oracle in process_oracles)
            if len(process_oracles) == len(self._oracles)
            else None
        )

    def open_step(self, site: ProofSite, /) -> None:
        """Reset one site's step counters without clearing its CPU use or memo."""
        self._require_open()
        self._site(site).open_step()

    def close(self) -> None:
        """Join and release the service's worker pool."""
        if self._closed:
            return

        self._executor.shutdown(wait=True, cancel_futures=True)

        _close_oracles(self._oracles)

        self._closed = True

    def decide(
        self,
        questions: Questions,
        /,
        *,
        budget: ProofBudget,
        site: ProofSite,
    ) -> tuple[Verdict, ...]:
        """Answer distinct questions from the memo and one site's budget."""
        self._require_open()
        accounting = self._site(site)
        self._questions += len(questions.premises)
        accounting.questions += len(questions.premises)
        results: list[Verdict | None] = [None] * len(questions.premises)
        pairs = tuple(zip(questions.premises, questions.goals, strict=True))
        dispatch_size = min(self._batch_size, self._workers)

        for start in range(0, len(pairs), dispatch_size):
            process_starts = self._process_starts()
            pending: list[
                tuple[int, _MemoKey, Future[tuple[Verdict, float]], float]
            ] = []

            for offset, (premise, goal) in enumerate(
                pairs[start : start + dispatch_size]
            ):
                index = start + offset
                key = (self._profile.digest, premise, goal, questions.theory)
                remembered = self._memo.get(key)

                if remembered is not None:
                    self._memo.pop(key)
                    self._memo[key] = remembered
                    self._hits += 1
                    accounting.hits += 1
                    results[index] = remembered
                    continue

                cause = self._spent(accounting, budget)

                if cause is not None:
                    results[index] = self._abstain(cause, budget)
                    self._abstentions[cause] = self._abstentions.get(cause, 0) + 1
                    accounting.abstentions[cause] = (
                        accounting.abstentions.get(cause, 0) + 1
                    )
                    continue

                wall = self._wall_remaining(accounting, budget)
                timeout = self._timeout if wall is None else min(self._timeout, wall)
                future = self._executor.submit(
                    self._call,
                    premise,
                    goal,
                    questions.theory,
                    timeout,
                )
                accounting.calls += 1
                self._calls += 1
                pending.append((index, key, future, monotonic() + timeout))

            for index, key, future, deadline in pending:
                wait = max(0.0, deadline - monotonic())
                wall = self._wall_remaining(accounting, budget)

                if wall is not None:
                    wait = min(wait, wall)

                try:
                    verdict, cpu = future.result(timeout=wait)
                except TimeoutError:
                    verdict = self._abstain("timeout", budget)
                    cpu = 0.0
                except (OSError, RuntimeError) as error:
                    verdict = Abstain(
                        f"the proof worker failed with {type(error).__name__}: {error}",
                        cause="gave-up",
                    )
                    cpu = 0.0

                self._cpu += cpu
                accounting.cpu += cpu
                accounting.lifetime_cpu += cpu
                results[index] = verdict

                if isinstance(verdict, (Proved, Refuted, Graded)):
                    if (
                        self._cache_size is not None
                        and len(self._memo) >= self._cache_size
                    ):
                        self._memo.pop(next(iter(self._memo)))

                    self._memo[key] = verdict
                elif isinstance(verdict, Abstain):
                    self._abstentions[verdict.cause] = (
                        self._abstentions.get(verdict.cause, 0) + 1
                    )
                    accounting.abstentions[verdict.cause] = (
                        accounting.abstentions.get(verdict.cause, 0) + 1
                    )

            ended_starts = self._process_starts()
            if process_starts is not None and ended_starts is not None:
                accounting.process_starts += ended_starts - process_starts

        settled: list[Verdict] = []

        for result in results:
            if result is None:
                msg = "a dispatched proof question produced no verdict"
                raise RuntimeError(msg)

            settled.append(result)

        return tuple(settled)

    def equivalent(
        self,
        left: Term,
        right: Term,
        /,
        *,
        budget: ProofBudget,
        site: ProofSite,
        theory: tuple[Term, ...] = (),
    ) -> Verdict:
        """Decide mutual entailment through this service and its shared budget.

        The reverse question is dispatched only after the forward question is
        proved. A refutation or abstention therefore settles equivalence
        without spending a second call.

        Parameters
        ----------
        left : Term
            One closed formula.
        right : Term
            The other closed formula.
        budget : ProofBudget
            The one use-site budget governing both directions.
        site : ProofSite
            The stable identity whose counters both directions spend.
        theory : tuple[Term, ...]
            Further closed formulas both directions assume.
        """
        forward = self.decide(
            Questions((left,), (right,), theory=theory), budget=budget, site=site
        )[0]

        if not (
            isinstance(forward, Proved)
            or (isinstance(forward, Graded) and forward.degree == 1.0)
        ):
            return forward

        backward = self.decide(
            Questions((right,), (left,), theory=theory), budget=budget, site=site
        )[0]

        if not (
            isinstance(backward, Proved)
            or (isinstance(backward, Graded) and backward.degree == 1.0)
        ):
            return backward

        if isinstance(forward, Graded) or isinstance(backward, Graded):
            forward_detail = (
                forward.proof.derivation
                if isinstance(forward, Proved)
                else forward.detail
            )
            backward_detail = (
                backward.proof.derivation
                if isinstance(backward, Proved)
                else backward.detail
            )
            return Graded(
                "mutual-entailment",
                1.0,
                f"forward: {forward_detail}\nbackward: {backward_detail}",
            )

        return Proved(
            Proof(
                "mutual-entailment",
                f"forward ({forward.proof.system}):\n{forward.proof.derivation}\n"
                f"backward ({backward.proof.system}):\n{backward.proof.derivation}",
                (*forward.proof.unsat_core, *backward.proof.unsat_core),
            )
        )

    def _call(
        self,
        premise: Term,
        goal: Term,
        theory: tuple[Term, ...],
        timeout: float,
    ) -> tuple[Verdict, float]:
        started = thread_time()
        oracle = self._pool.get()

        try:
            verdict = oracle.entails(
                premise,
                goal,
                theory=theory,
                timeout=timeout,
            )
            return verdict, thread_time() - started
        finally:
            self._pool.put(oracle)

    def _spent(
        self, accounting: _SiteAccounting, budget: ProofBudget, /
    ) -> AbstentionCause | None:
        if (
            budget.calls_per_step is not None
            and accounting.calls >= budget.calls_per_step
        ):
            return "budget-spent"

        if (
            budget.cpu_seconds is not None
            and accounting.lifetime_cpu >= budget.cpu_seconds
        ):
            return "budget-spent"

        wall = self._wall_remaining(accounting, budget)

        if wall is not None and wall <= 0.0:
            return "timeout"

        return None

    def _wall_remaining(
        self, accounting: _SiteAccounting, budget: ProofBudget, /
    ) -> float | None:
        if budget.wall_per_step is None:
            return None

        return max(0.0, budget.wall_per_step - (monotonic() - accounting.started))

    def _site(self, site: ProofSite, /) -> _SiteAccounting:
        if not isinstance(site, ProofSite):
            msg = f"a proof-service site is a ProofSite, not {site!r}"
            raise TypeError(msg)

        accounting = self._sites.get(site)
        if accounting is None:
            accounting = _SiteAccounting()
            self._sites[site] = accounting
        return accounting

    def _abstain(self, cause: AbstentionCause, budget: ProofBudget) -> Abstain:
        reason = (
            f"the proof budget is spent: {budget!r}"
            if cause == "budget-spent"
            else f"the proof step wall clock is spent: {budget!r}"
        )
        return Abstain(reason, cause=cause)

    def _require_open(self) -> None:
        if self._closed:
            msg = "a closed proof service cannot dispatch questions"
            raise RuntimeError(msg)
