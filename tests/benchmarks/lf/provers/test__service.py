"""Measures a long-lived proof worker against process-per-question dispatch."""

from statistics import median
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Final

from tests.language import language
from tests.verdicts import refuted

from lofi.lf import ProverBound
from lofi.lf.provers import ProofBudget, ProofService, ProofSite, Questions

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.lf import Verdict
    from lofi.terms import Term


LANGUAGE: Final = language(tuple(f"p{index}" for index in range(17)))
PREMISES: Final = tuple(LANGUAGE.constant(f"p{index}") for index in range(8))
GOALS: Final = tuple(LANGUAGE.constant(f"p{index + 8}") for index in range(8))
QUESTIONS: Final = Questions(PREMISES, GOALS)
STARTUP_SECONDS: Final = 0.002
ROUNDS: Final = 5
SITE: Final = ProofSite("benchmark")


class _ProcessOracle:
    """A process-shaped fixture whose construction pays worker startup cost."""

    def __init__(self) -> None:
        sleep(STARTUP_SECONDS)
        self._bound = ProverBound("process-fixture", LANGUAGE.profile.fragment)

    @property
    def bound(self) -> ProverBound:
        return self._bound

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        del premise, hypothesis, theory, timeout
        return refuted("fixture")


def _pooled() -> None:
    service = ProofService(
        _ProcessOracle,
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=len(PREMISES),
        timeout=1.0,
    )

    try:
        service.decide(QUESTIONS, budget=ProofBudget(), site=SITE)
    finally:
        service.close()


def _per_question() -> None:
    for premise, goal in zip(PREMISES, GOALS, strict=True):
        oracle = _ProcessOracle()
        oracle.entails(premise, goal, timeout=1.0)


def _median_seconds(call: Callable[[], None], /) -> float:
    samples: list[float] = []

    for _ in range(ROUNDS):
        started = perf_counter()
        call()
        samples.append(perf_counter() - started)

    return median(samples)


def test_a_worker_pool_beats_process_per_question_on_one_question_set() -> None:
    pooled = _median_seconds(_pooled)
    per_question = _median_seconds(_per_question)

    assert pooled < per_question * 0.5
