import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

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
from lofi.lf.provers import (
    EPROVER,
    TPTP_DIALECT_CAPABILITIES,
    VAMPIRE,
    TptpDialect,
    TptpEngine,
    TptpOracle,
)
from lofi.lf.provers._capability import (
    FIRST_ORDER_CAPABILITY,
    HIGHER_ORDER_CAPABILITY,
    TranslatorCapability,
)
from lofi.spec.language import ConstantSpec, LanguageIso, LogicSpec, registered_language

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.spec.language import LanguageAssembly


@pytest.mark.parametrize("dialect", ["fof", "tff"])
@pytest.mark.parametrize("engine", [EPROVER, VAMPIRE])
def test_tptp_decides_repeated_questions_and_enforces_deadline(
    first_order_language: LanguageAssembly,
    engine: TptpEngine,
    dialect: TptpDialect,
) -> None:
    if shutil.which(engine.name) is None:
        pytest.skip(f"{engine.name} is not installed")

    language = first_order_language
    premise = language.read("(all x. dog(x) => mammal(x)) & dog(rex)")
    hypothesis = language.read("mammal(rex)")
    oracle = _oracle(language, None, engine=engine, dialect=dialect)
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


def _bound(
    engine: TptpEngine,
    dialect: TptpDialect,
    language: LanguageAssembly,
) -> ProverBound:
    serves = engine.serves(dialect)
    profile = language.profile
    return ProverBound(
        f"tptp:{engine.name}:{dialect}",
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


_THEOREM: Final = "print('% SZS status Theorem')\n"
"""A prover that proves every problem."""


def _oracle(
    language: LanguageAssembly,
    executable: str | None,
    *,
    engine: TptpEngine = EPROVER,
    dialect: TptpDialect = "fof",
    prefix: tuple[str, ...] | None = None,
    options: tuple[str, ...] = (),
    problem_file: str = "problem.p",
) -> TptpOracle:
    return TptpOracle(
        language,
        engine=engine,
        dialect=dialect,
        executable=executable,
        bound=_bound(engine, dialect, language),
        prefix=prefix,
        options=options,
        problem_file=problem_file,
        memory_mb=None,
        cpu_seconds=None,
        seed=None,
        polymorphism="refuse",
        certificates=frozenset(),
    )


def test_a_tptp_engine_builds_its_declared_process_arguments() -> None:
    engine = TptpEngine(
        "fixture",
        dialects=("fof",),
        prefix=("--mode=fixture",),
        memory_option="--memory={value}",
        cpu_option="--cpu={value}",
        seed_option="--seed={value}",
    )

    assert engine.prefix == ("--mode=fixture",)
    assert engine.arguments(
        "/prover",
        Path("problem.p"),
        prefix=("--declared",),
        options=("--extra",),
        memory_mb=512,
        cpu_seconds=7,
        seed=11,
    ) == (
        "/prover",
        "--declared",
        "--memory=512",
        "--cpu=7",
        "--seed=11",
        "--extra",
        "problem.p",
    )


def test_a_tptp_engine_refuses_a_resource_it_does_not_serve() -> None:
    with pytest.raises(ValueError, match="seed"):
        EPROVER.arguments(
            "/prover",
            Path("problem.p"),
            prefix=EPROVER.prefix,
            options=(),
            memory_mb=None,
            cpu_seconds=None,
            seed=11,
        )


def test_an_engine_cannot_widen_a_first_order_tptp_translator() -> None:
    engine = TptpEngine(
        "fixture",
        dialects=("fof",),
        serves={"fof": HIGHER_ORDER_CAPABILITY},
    )

    assert engine.serves("fof").order == "first"
    assert engine.serves("fof").world_types == ()


def _decide(oracle: TptpOracle, language: LanguageAssembly) -> object:
    return oracle.entails(
        language.read("dog(rex)"), language.read("mammal(rex)"), timeout=5.0
    )


@pytest.mark.parametrize(
    ("report", "cause"),
    [
        ("% SZS status Timeout for problem.p", "timeout"),
        ("% SZS status ResourceOut for problem.p", "gave-up"),
        ("% SZS status MemoryOut for problem.p", "gave-up"),
        ("% SZS status GaveUp for problem.p", "gave-up"),
        ("the prover printed no status", "gave-up"),
    ],
)
def test_undecided_prover_reports_abstain_with_their_cause(
    first_order_language: LanguageAssembly,
    script: Callable[[str], str],
    report: str,
    cause: str,
) -> None:
    oracle = _oracle(first_order_language, script(f"print({report!r})"))
    verdict = _decide(oracle, first_order_language)

    assert isinstance(verdict, Abstain)
    assert verdict.cause == cause


@pytest.mark.parametrize(
    ("first", "later", "decided"),
    [("Theorem", "CounterSatisfiable", Proved), ("Satisfiable", "Theorem", Refuted)],
)
def test_the_first_reported_status_decides(
    first_order_language: LanguageAssembly,
    script: Callable[[str], str],
    first: str,
    later: str,
    decided: type,
) -> None:
    executable = script(
        f"print('% SZS status {first} for problem.p')\n"
        f"print('% SZS status {later} for problem.p')\n"
    )

    assert isinstance(
        _decide(_oracle(first_order_language, executable), first_order_language),
        decided,
    )


def test_a_nonpositive_deadline_is_refused(
    first_order_language: LanguageAssembly, script: Callable[[str], str]
) -> None:
    oracle = _oracle(first_order_language, script(_THEOREM))

    with pytest.raises(ValueError, match="positive"):
        oracle.entails(
            first_order_language.read("dog(rex)"),
            first_order_language.read("dog(rex)"),
            timeout=0.0,
        )


def test_an_intensional_language_is_refused(
    intensional_language: LanguageAssembly, script: Callable[[str], str]
) -> None:
    with pytest.raises(ProverUnsupportedError) as refused:
        _oracle(intensional_language, script(_THEOREM))

    assert refused.value.field == "world_types"


def test_thf_runs_a_second_order_problem_through_the_declared_engine(
    script: Callable[[str], str],
) -> None:
    declared = registered_language("neo-davidsonian").with_(
        constants=(
            *registered_language("neo-davidsonian").constants,
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
            ConstantSpec(name="fido", type="e"),
        ),
        logic=LogicSpec(order="second"),
    )
    language = LanguageIso().forward(declared)
    executable = script(
        "import pathlib\n"
        "import sys\n"
        'problem = pathlib.Path(sys.argv[-1]).read_text(encoding="utf-8")\n'
        'if any(line.startswith("thf(") for line in problem.splitlines()):\n'
        "    print('% SZS status Theorem')\n"
    )
    oracle = _oracle(language, executable, engine=VAMPIRE, dialect="thf")
    premise = language.read("(forall P:(e -> t). P(rex) => P(fido)) & dog(rex)")

    assert isinstance(
        oracle.entails(premise, language.read("dog(fido)"), timeout=5.0),
        Proved,
    )
    assert oracle.bound.fragment.order == "higher"


def test_a_term_outside_tptp_translation_abstains_without_starting_a_process(
    first_order_language: LanguageAssembly,
    script: Callable[[str], str],
) -> None:
    oracle = _oracle(
        first_order_language,
        script("raise SystemExit(99)"),
        engine=VAMPIRE,
        dialect="thf",
    )
    polymorphic = first_order_language.read(r"(\x. dog(rex))(rex)")

    verdict = oracle.entails(polymorphic, polymorphic, timeout=5.0)

    assert isinstance(verdict, Abstain)
    assert verdict.cause == "out-of-fragment"
    assert "monomorphic type" in verdict.reason
    assert oracle.process_starts == 0


def test_first_order_tptp_refuses_a_second_order_profile_by_field(
    script: Callable[[str], str],
) -> None:
    language = LanguageIso().forward(
        registered_language("neo-davidsonian").with_(logic=LogicSpec(order="second"))
    )

    with pytest.raises(ProverUnsupportedError) as refused:
        TptpOracle(
            language,
            engine=VAMPIRE,
            dialect="tff",
            executable=script(_THEOREM),
            bound=_bound(VAMPIRE, "tff", language),
            prefix=None,
            options=(),
            problem_file="problem.p",
            memory_mb=None,
            cpu_seconds=None,
            seed=None,
            polymorphism="refuse",
            certificates=frozenset(),
        )

    assert refused.value.field == "order"


_ARGUMENTS: Final = (
    "import sys\n"
    "print('% SZS status Theorem' if sys.argv[1:-1] == {expected!r} "
    "else '% SZS status GaveUp')\n"
)
"""A prover that proves only under one exact argument vector."""


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (None, ["--default", "--extra"]),
        (("--declared",), ["--declared", "--extra"]),
        ((), ["--extra"]),
    ],
)
def test_a_declared_prefix_replaces_the_engine_prefix_and_options_follow_it(
    first_order_language: LanguageAssembly,
    script: Callable[[str], str],
    prefix: tuple[str, ...] | None,
    expected: list[str],
) -> None:
    engine = TptpEngine("fixture", dialects=("fof",), prefix=("--default",))
    oracle = _oracle(
        first_order_language,
        script(_ARGUMENTS.format(expected=expected)),
        engine=engine,
        prefix=prefix,
        options=("--extra",),
    )

    assert isinstance(_decide(oracle, first_order_language), Proved)


def test_the_oracle_writes_the_problem_under_its_declared_file_name(
    first_order_language: LanguageAssembly,
    script: Callable[[str], str],
) -> None:
    executable = script(
        "import pathlib\n"
        "import sys\n"
        "problem = pathlib.Path(sys.argv[-1])\n"
        "print('% SZS status Theorem' if sys.argv[-1] == 'query.p' "
        "and problem.is_file() else '% SZS status GaveUp')\n"
    )
    oracle = _oracle(first_order_language, executable, problem_file="query.p")

    assert isinstance(_decide(oracle, first_order_language), Proved)


def test_each_dialect_declares_the_fragment_its_translation_serves() -> None:
    assert TPTP_DIALECT_CAPABILITIES == {
        "fof": FIRST_ORDER_CAPABILITY,
        "tff": FIRST_ORDER_CAPABILITY,
        "thf": HIGHER_ORDER_CAPABILITY,
    }


@pytest.mark.parametrize("dialect", ["fof", "tff", "thf"])
def test_an_engine_serves_each_dialect_its_declared_fragment(
    dialect: TptpDialect,
) -> None:
    engine = TptpEngine("fixture", dialects=(dialect,))
    declared: TranslatorCapability = TPTP_DIALECT_CAPABILITIES[dialect]

    assert engine.serves(dialect) == declared
