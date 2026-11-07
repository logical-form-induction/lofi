import shutil
import sys
from typing import TYPE_CHECKING

import pytest

from lofi.lf import (
    Abstain,
    CounterSatisfiable,
    LogicFragment,
    Proof,
    Proved,
    ProverBound,
    ProverUnsupportedError,
    Refuted,
)
from lofi.lf.provers import CVC5, Z3, SmtEngine, SmtOracle
from lofi.lf.provers._capability import HIGHER_ORDER_CAPABILITY

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lofi.spec.language import LanguageAssembly


def _bound(engine: SmtEngine, language: LanguageAssembly) -> ProverBound:
    serves = engine.serves
    profile = language.profile
    return ProverBound(
        f"smt:{engine.name}",
        LogicFragment(
            order=serves.order,
            quantifiers=profile.fragment.quantifiers if serves.quantifiers else (),
            connectives=profile.fragment.connectives,
            functions=serves.functions,
            equality=serves.equality,
        ),
        world_types=(
            profile.world_types if "*" in serves.world_types else serves.world_types
        ),
    )


def _oracle(
    language: LanguageAssembly,
    engine: SmtEngine,
    executable: str | None,
    *,
    certificates: frozenset[str] = frozenset(),
    prefix: tuple[str, ...] | None = None,
    terminate_grace: float = 0.2,
) -> SmtOracle:
    return SmtOracle(
        language,
        engine=engine,
        executable=sys.executable if executable is not None else None,
        bound=_bound(engine, language),
        logic="ALL",
        options={},
        prefix=(
            (executable, *(engine.prefix if prefix is None else prefix))
            if executable is not None
            else prefix
        ),
        arguments=(),
        terminate_grace=terminate_grace,
        polymorphism="refuse",
        certificates=certificates,
        memory_mb=None,
        cpu_seconds=None,
        seed=None,
    )


def test_an_smt_engine_builds_its_declared_process_arguments() -> None:
    assert Z3.prefix == ("-in", "-smt2")
    assert Z3.arguments(
        "/solver",
        ("--extra",),
        prefix=Z3.prefix,
        memory_mb=512,
        cpu_seconds=7,
        seed=11,
    ) == (
        "/solver",
        "-in",
        "-smt2",
        "-memory:512",
        "-T:7",
        "smt.random_seed=11",
        "--extra",
    )


def test_an_smt_engine_refuses_a_resource_it_does_not_serve() -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        CVC5.arguments(
            "/solver",
            (),
            prefix=CVC5.prefix,
            memory_mb=512,
            cpu_seconds=None,
            seed=None,
        )


def test_an_engine_cannot_widen_the_smt_translator_fragment() -> None:
    engine = SmtEngine("fixture", (), serves=HIGHER_ORDER_CAPABILITY)

    assert engine.serves.order == "first"
    assert engine.serves.world_types == ()


@pytest.mark.parametrize("engine", [Z3, CVC5])
def test_smt_decides_repeated_questions_and_enforces_deadline(
    first_order_language: LanguageAssembly,
    engine: SmtEngine,
) -> None:
    if shutil.which(engine.name) is None:
        pytest.skip(f"{engine.name} is not installed")

    language = first_order_language
    premise = language.read("(all x. dog(x) => mammal(x)) & dog(rex)")
    hypothesis = language.read("mammal(rex)")
    oracle = _oracle(language, engine, None)

    try:
        proved = oracle.entails(premise, hypothesis, timeout=5.0)
        refuted = oracle.entails(hypothesis, premise, timeout=5.0)

        assert isinstance(proved, Proved)
        assert isinstance(refuted, Refuted)
        assert isinstance(proved.proof, Proof)
        assert proved.proof.system == engine.name
        assert isinstance(refuted.evidence, CounterSatisfiable)
        assert refuted.evidence.system == engine.name
        assert oracle.entails(premise, hypothesis, timeout=5.0) == proved
        assert oracle.entails(hypothesis, premise, timeout=5.0) == refuted

        timed_out = oracle.entails(
            premise,
            language.read("dog(rex)"),
            theory=(premise,) * 256,
            timeout=0.001,
        )

        assert isinstance(timed_out, Abstain)
        assert timed_out.cause == "timeout"
    finally:
        oracle.close()


def _answering(*answer: str) -> str:
    """Return a solver that runs an answer each time it is asked to check."""
    body = "".join(f"        {line}\n" for line in answer)

    return (
        "import sys\n"
        "for line in sys.stdin:\n"
        '    if line.rstrip("\\n") == "(check-sat)":\n'
        f"{body}"
    )


def test_an_unknown_answer_abstains_as_a_give_up(
    first_order_language: LanguageAssembly, python_program: Callable[[str], str]
) -> None:
    language = first_order_language
    oracle = _oracle(
        language, Z3, python_program(_answering('print("unknown", flush=True)'))
    )

    try:
        verdict = oracle.entails(
            language.read("dog(rex)"), language.read("mammal(rex)"), timeout=5.0
        )
    finally:
        oracle.close()

    assert isinstance(verdict, Abstain)
    assert verdict.cause == "gave-up"


def test_a_late_answer_never_decides_the_next_query(
    first_order_language: LanguageAssembly,
    python_program: Callable[[str], str],
    tmp_path: Path,
) -> None:
    language = first_order_language
    marker = tmp_path / "started"
    first_worker_is_slow = (
        "import pathlib\n"
        "import time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "slow = not marker.exists()\n"
        "marker.touch()\n"
    ) + _answering(
        "if slow:",
        "    time.sleep(5)",
        'print("unsat" if slow else "sat", flush=True)',
    )
    oracle = _oracle(language, Z3, python_program(first_worker_is_slow))
    premise = language.read("dog(rex)")
    hypothesis = language.read("mammal(rex)")

    try:
        late = oracle.entails(premise, hypothesis, timeout=1.0)
        next_verdict = oracle.entails(premise, hypothesis, timeout=10.0)
    finally:
        oracle.close()

    assert isinstance(late, Abstain)
    assert late.cause == "timeout"
    assert isinstance(next_verdict, Refuted)


def test_a_nonpositive_deadline_is_refused(
    first_order_language: LanguageAssembly, python_program: Callable[[str], str]
) -> None:
    language = first_order_language
    oracle = _oracle(
        language, Z3, python_program(_answering('print("sat", flush=True)'))
    )

    try:
        with pytest.raises(ValueError, match="positive"):
            oracle.entails(
                language.read("dog(rex)"), language.read("dog(rex)"), timeout=0.0
            )
    finally:
        oracle.close()


def test_an_intensional_language_is_refused(
    intensional_language: LanguageAssembly, python_program: Callable[[str], str]
) -> None:
    with pytest.raises(ProverUnsupportedError) as refused:
        _oracle(intensional_language, Z3, python_program(_answering('print("sat")')))

    assert refused.value.field == "world_types"


def test_a_requested_smt_model_is_persisted_as_counterevidence(
    first_order_language: LanguageAssembly,
    python_program: Callable[[str], str],
) -> None:
    executable = python_program(
        _answering(
            'print("sat", flush=True)',
            'if sys.stdin.readline().rstrip("\\n") == "(get-model)":',
            '    print("(model", flush=True)',
            '    print("  (define-fun lofi_rex () lofi_sort_e lofi_rex)", flush=True)',
            '    print(")", flush=True)',
        )
    )
    language = first_order_language
    oracle = _oracle(language, Z3, executable, certificates=frozenset(("model",)))

    try:
        verdict = oracle.entails(
            language.read("dog(rex)"),
            language.read("mammal(rex)"),
            timeout=5.0,
        )
    finally:
        oracle.close()

    assert isinstance(verdict, Refuted)
    assert isinstance(verdict.evidence, CounterSatisfiable)
    assert verdict.evidence.detail.startswith("(model")
    assert "define-fun lofi_rex" in verdict.evidence.detail


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [(None, ["--default"]), (("--declared",), ["--declared"]), ((), [])],
)
def test_a_declared_prefix_replaces_the_engine_prefix(
    first_order_language: LanguageAssembly,
    python_program: Callable[[str], str],
    prefix: tuple[str, ...] | None,
    expected: list[str],
) -> None:
    language = first_order_language
    engine = SmtEngine("fixture", ("--default",))
    solver = f"import sys\nexact = sys.argv[1:] == {expected!r}\n" + _answering(
        'print("unsat" if exact else "unknown", flush=True)'
    )
    oracle = _oracle(language, engine, python_program(solver), prefix=prefix)

    try:
        verdict = oracle.entails(
            language.read("dog(rex)"), language.read("mammal(rex)"), timeout=5.0
        )
    finally:
        oracle.close()

    assert isinstance(verdict, Proved)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows terminates without a signal handler"
)
@pytest.mark.parametrize(("grace", "finished"), [(5.0, True), (0.01, False)])
def test_a_stopped_worker_has_its_declared_grace_period_to_finish(
    first_order_language: LanguageAssembly,
    python_program: Callable[[str], str],
    tmp_path: Path,
    *,
    grace: float,
    finished: bool,
) -> None:
    language = first_order_language
    marker = tmp_path / "finished"
    solver = (
        "import pathlib\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})\n"
        "answered = False\n"
        "for line in sys.stdin:\n"
        '    if line.rstrip("\\n") == "(check-sat)" and not answered:\n'
        '        print("unknown", flush=True)\n'
        "        answered = True\n"
        "signal.sigwait({signal.SIGTERM})\n"
        "time.sleep(0.5)\n"
        f"pathlib.Path({str(marker)!r}).touch()\n"
    )
    oracle = _oracle(language, Z3, python_program(solver), terminate_grace=grace)
    premise = language.read("dog(rex)")
    hypothesis = language.read("mammal(rex)")

    try:
        ready = oracle.entails(premise, hypothesis, timeout=30.0)
        verdict = oracle.entails(premise, hypothesis, timeout=0.5)
    finally:
        oracle.close()

    assert isinstance(ready, Abstain)
    assert isinstance(verdict, Abstain)
    assert verdict.cause == "timeout"
    assert oracle.process_starts == 1
    assert marker.exists() is finished
