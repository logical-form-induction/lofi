import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Final

from scripts.fixtures.make_chart_fixture import (
    PinnedFingerprint,
    fixture_grammar,
    sample_sentences,
    write_fixture,
)

from lofi import _checks

if TYPE_CHECKING:
    import pytest
    from tests.benchmarks.conftest import PinnedChart

CONFTEST: Final = Path(__file__).with_name("conftest.py")
ROOT_CONFTEST: Final = Path(__file__).parents[1] / "conftest.py"
TREE: Final = Path(__file__).resolve().parents[2]

TIMED: Final = textwrap.dedent(
    """
    def test_timed(benchmark):
        benchmark(sum, range(100))


    def test_plain():
        pass
    """
)

PINNED: Final = textwrap.dedent(
    """
    from lofi.deduce import ParsePolicy, Reachability, Retain, SpanFree, Viterbi, parse


    class Unweighted:
        lowering = SpanFree()

        def axiom_weight(self, position, entry, /):
            del position, entry

            return 0.0

        def edge_weight(self, rule, inputs, output, span, split, /):
            del rule, inputs, output, span, split

            return 0.0


    def test_pinned(pinned_chart):
        assert len(pinned_chart.sentences) == 3
        assert pinned_chart.grammar.fingerprint == pinned_chart.pinned.fingerprint

        for sentence in pinned_chart.sentences:
            forest = parse(
                sentence,
                rules=pinned_chart.grammar.rules,
                source=pinned_chart.lexicon,
                semiring=Viterbi(),
                scorer=Unweighted(),
                goal=pinned_chart.grammar.goal,
                unary_depth=pinned_chart.grammar.budget.unary_rounds,
                policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
            )

            assert tuple(pinned_chart.grammar.goal.roots(forest))
    """
)


FLOOR: Final = textwrap.dedent(
    """
    from conftest import measure_floor


    def test_floor():
        readings = iter([0.0, 3.0, 10.0, 11.0, 20.0, 22.0])
        calls = []
        floor = measure_floor(
            lambda: next(readings), lambda: calls.append(None), rounds=3, iterations=4
        )

        assert floor == 0.25
        assert len(calls) == 12
    """
)

BUDGETED: Final = textwrap.dedent(
    """
    def test_budgeted(benchmark, wall_clock_budget):
        benchmark(sum, range(100))

        if wall_clock_budget.enforced:
            assert benchmark.stats.stats.min <= 0.0
    """
)

SHARED: Final = textwrap.dedent(
    """
    seen = []


    def test_first(harness_floor):
        seen.append(harness_floor)


    def test_second(harness_floor):
        assert harness_floor is seen[0]
    """
)


def _session(
    pytester: pytest.Pytester,
    source: str,
    *options: str,
) -> pytest.RunResult:
    """Run one test module under a copy of the benchmark conftest."""
    pytester.makepyprojecttoml("[tool.pytest.ini_options]\n")
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(source)

    return pytester.runpytest_inprocess("-p", "no:cacheprovider", "-rs", *options)


def _tree_session(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    *options: str,
    module: str = TIMED,
) -> pytest.RunResult:
    """Run a benchmark module in a fresh interpreter under both conftests."""
    monkeypatch.delenv("LOFI_CHECK", raising=False)
    pytester.makepyprojecttoml(
        f'[tool.pytest.ini_options]\npythonpath = ["{TREE.as_posix()}"]\n',
    )
    pytester.makeconftest(ROOT_CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(
        **{
            "benchmarks/conftest": CONFTEST.read_text(encoding="utf-8"),
            "benchmarks/test_timed": module,
        }
    )

    return pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", "-rs", "benchmarks", *options
    )


def _write_pinned(pytester: pytest.Pytester) -> tuple[Path, Path]:
    """Write a three-sentence pinned fixture where the copied conftest looks."""
    pack = fixture_grammar("cfg-toy")
    fixtures = pytester.path / "fixtures"
    sentences_path = fixtures / "chart_200.jsonl"
    fingerprint_path = fixtures / "chart_200.fingerprint"
    write_fixture(
        pack,
        sample_sentences(pack, 3, 20, 7),
        7,
        sentences_path,
        fingerprint_path,
    )

    return sentences_path, fingerprint_path


def test_the_harness_floor_is_a_positive_fraction_of_a_microsecond(
    harness_floor: float,
) -> None:
    assert 0 < harness_floor < 1e-6


def test_the_harness_floor_is_the_fastest_round_divided_by_its_calls(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, FLOOR)

    result.assert_outcomes(passed=1)


def test_the_harness_floor_is_measured_once_per_session(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, SHARED)

    result.assert_outcomes(passed=2)


def test_a_checked_build_refuses_to_time_and_names_the_flag(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    result = _session(pytester, TIMED, "--benchmark-only")

    result.assert_outcomes(errors=1, skipped=1)
    assert "LOFI_CHECK" in result.stdout.str()


def test_a_checked_build_still_runs_the_tests_that_time_nothing(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    result = _session(pytester, TIMED)

    result.assert_outcomes(errors=1, passed=1)
    assert "test_timed" in result.stdout.str()


def test_an_unchecked_build_times_the_benchmark(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    result = _session(pytester, TIMED, "--benchmark-only")

    result.assert_outcomes(passed=1, skipped=1)


def test_a_skipped_benchmark_is_not_refused_on_a_checked_build(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    result = _session(pytester, TIMED, "--benchmark-skip")

    result.assert_outcomes(passed=1, skipped=1)


def test_a_benchmark_only_run_is_unchecked_unless_the_flag_is_set(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _tree_session(pytester, monkeypatch, "--benchmark-only")

    result.assert_outcomes(passed=1, skipped=1)


def test_an_ordinary_run_is_checked_unless_the_flag_is_set(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _tree_session(pytester, monkeypatch)

    result.assert_outcomes(errors=1, passed=1)
    assert "LOFI_CHECK" in result.stdout.str()


def test_a_wall_clock_budget_holds_by_default(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _tree_session(pytester, monkeypatch, "--benchmark-only", module=BUDGETED)

    result.assert_outcomes(failed=1)
    assert "benchmark.stats.stats.min <= 0.0" in result.stdout.str()


def test_a_session_that_declares_wall_clock_budgets_off_still_times_the_benchmark(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _tree_session(
        pytester,
        monkeypatch,
        "--benchmark-only",
        "-o",
        "wall_clock_budgets=false",
        "--benchmark-json=timed.json",
        module=BUDGETED,
    )

    result.assert_outcomes(passed=1)
    timed = json.loads((pytester.path / "timed.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in timed["benchmarks"]] == ["test_budgeted"]


def test_the_pinned_chart_parses_under_the_grammar_it_records(
    pytester: pytest.Pytester,
) -> None:
    _write_pinned(pytester)
    result = _session(pytester, PINNED)

    result.assert_outcomes(passed=1)


def test_an_absent_pinned_chart_skips_and_names_the_files(
    pytester: pytest.Pytester,
) -> None:
    result = _session(pytester, PINNED)

    result.assert_outcomes(skipped=1)
    assert "chart_200.jsonl" in result.stdout.str()
    assert "chart_200.fingerprint" in result.stdout.str()


def test_a_record_naming_an_unknown_pack_fails_the_pinned_chart(
    pytester: pytest.Pytester,
) -> None:
    _sentences_path, fingerprint_path = _write_pinned(pytester)
    recorded = PinnedFingerprint.read(fingerprint_path)
    recorded._replace(grammar="nope").write(fingerprint_path)
    result = _session(pytester, PINNED)

    result.assert_outcomes(errors=1)
    assert "unknown grammar pack 'nope'" in result.stdout.str()


def test_a_grammar_that_moved_fails_the_pinned_chart_with_both_values(
    pytester: pytest.Pytester,
) -> None:
    _sentences_path, fingerprint_path = _write_pinned(pytester)
    recorded = PinnedFingerprint.read(fingerprint_path)
    recorded._replace(fingerprint="stale").write(fingerprint_path)
    result = _session(pytester, PINNED)

    result.assert_outcomes(errors=1)
    assert "grammar fingerprint" in result.stdout.str()
    assert "'stale'" in result.stdout.str()
    assert repr(recorded.fingerprint) in result.stdout.str()


def test_a_sentence_file_that_moved_fails_the_pinned_chart(
    pytester: pytest.Pytester,
) -> None:
    sentences_path, _fingerprint_path = _write_pinned(pytester)

    with sentences_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"tokens":["the","cat"]}\n')

    result = _session(pytester, PINNED)

    result.assert_outcomes(errors=1)
    assert "sentence digest" in result.stdout.str()


def test_the_in_tree_pinned_chart_holds_two_hundred_sentences(
    pinned_chart: PinnedChart,
) -> None:
    assert len(pinned_chart.sentences) == 200
    assert pinned_chart.grammar.fingerprint == pinned_chart.pinned.fingerprint
