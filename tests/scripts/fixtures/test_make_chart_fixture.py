import json
from hashlib import sha256
from typing import TYPE_CHECKING, Final

import pytest
from scripts.fixtures.make_chart_fixture import (
    FIXTURE_GRAMMARS,
    FingerprintMismatchError,
    FixtureGrammar,
    FixtureMismatchError,
    PinnedFingerprint,
    SentenceDigestMismatchError,
    ShapeMismatchError,
    check_pinned_fixture,
    fixture_grammar,
    main,
    read_sentences,
    sample_sentences,
    write_fixture,
)

from lofi.data.synthetic import (
    SyntheticGenerator,
    UniformLength,
    UnproductiveGrammarError,
)
from lofi.deduce import ParsePolicy, Reachability, Retain, SpanFree, Viterbi, parse
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.deduce import Cell
    from lofi.judge import Rule
    from lofi.lexicon import Entry

SEED: Final = 7
COUNT: Final = 12
LENGTH: Final = 8


def _atom(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def _pack(
    name: str,
    productions: tuple[tuple[str, str, str, str], ...],
    lexicon: dict[str, tuple[str, ...]],
    goal: str,
) -> FixtureGrammar:
    nonterminals = tuple(
        dict.fromkeys(
            [
                *(symbol for rule in productions for symbol in rule[1:]),
                *lexicon,
                goal,
            ]
        )
    )

    return FixtureGrammar(
        GrammarDecl(
            name=name,
            signature=SignatureSpec(
                name=name, primitives=tuple(nonterminals), syntax="s-expression"
            ),
            rules=RuleSetSpec(
                name=name,
                rules=tuple(
                    PatternRuleSpec(
                        name=rule,
                        premises=(_atom(left), _atom(right)),
                        conclusion=_atom(parent),
                    )
                    for rule, parent, left, right in productions
                ),
            ),
            lexicon=InlineLexiconSpec(
                entries=tuple(
                    LexicalEntrySpec(
                        form=form, category=f'("#const[Cat]" "{preterminal}")'
                    )
                    for preterminal, forms in lexicon.items()
                    for form in forms
                )
            ),
            goal=RootGoalSpec(categories=(f'("#const[Cat]" "{goal}")',)),
            budget=BudgetSpec(unary_rounds=1, closure_rounds=2, closure_nodes=16),
        )
    )


BRANCHING: Final = _pack(
    "branching", (("s-ab", "S", "A", "B"),), {"A": ("a", "b"), "B": ("c",)}, "S"
)
TINY: Final = _pack(
    "tiny", (("s-aa", "S", "A", "A"), ("s-ss", "S", "S", "S")), {"A": ("a",)}, "S"
)


class Unweighted:
    """Scores every axiom and edge at log one."""

    __slots__ = ()
    lowering = SpanFree()

    def axiom_weight[T](self, position: int, entry: Entry[T], /) -> float:
        del position, entry

        return 0.0

    def edge_weight[T](
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split

        return 0.0


def _generate(directory: Path, *arguments: str) -> tuple[Path, Path]:
    assert main(["--grammar", "cfg-toy", "--seed", str(SEED), *arguments]) == 0

    return directory / f"chart_{COUNT}.jsonl", directory / f"chart_{COUNT}.fingerprint"


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("fixture")

    return _generate(
        directory,
        "--count",
        str(COUNT),
        "--length",
        str(LENGTH),
        "--out",
        str(directory),
    )


@pytest.fixture
def copied(fixture: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    """Copy the module's fixture so a test can edit it."""
    sentences_path = tmp_path / fixture[0].name
    fingerprint_path = tmp_path / fixture[1].name
    sentences_path.write_bytes(fixture[0].read_bytes())
    fingerprint_path.write_bytes(fixture[1].read_bytes())

    return sentences_path, fingerprint_path


def test_two_runs_at_one_seed_are_byte_identical(tmp_path: Path) -> None:
    arguments = ("--count", str(COUNT), "--length", str(LENGTH))
    first = _generate(tmp_path / "one", *arguments, "--out", str(tmp_path / "one"))
    second = _generate(tmp_path / "two", *arguments, "--out", str(tmp_path / "two"))

    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_the_sentences_are_the_synthetic_generators_draw_at_the_seed() -> None:
    grammar = fixture_grammar("cfg-toy")
    corpus = SyntheticGenerator(
        grammar.assemble(),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    ).sample(COUNT, seed=SEED, length=UniformLength(LENGTH, LENGTH))

    assert sample_sentences(grammar, COUNT, LENGTH, SEED) == tuple(
        tuple(token.surface for token in example.item.tokens)
        for example in corpus.examples
    )
    assert sample_sentences(grammar, COUNT, LENGTH, SEED) != sample_sentences(
        grammar, COUNT, LENGTH, SEED + 1
    )


def test_the_fixture_holds_the_requested_count_at_the_requested_length(
    fixture: tuple[Path, Path],
) -> None:
    sentences = read_sentences(fixture[0])

    assert len(sentences) == COUNT
    assert all(len(sentence) == LENGTH for sentence in sentences)
    assert len(set(sentences)) > 1


def test_every_line_is_one_json_object_holding_only_tokens(
    fixture: tuple[Path, Path],
) -> None:
    lines = fixture[0].read_text(encoding="utf-8").splitlines()

    assert len(lines) == COUNT
    assert all(set(json.loads(line)) == {"tokens"} for line in lines)
    assert read_sentences(fixture[0]) == tuple(
        tuple(json.loads(line)["tokens"]) for line in lines
    )


def test_every_drawn_sentence_parses_to_the_goal_under_the_compiled_grammar(
    fixture: tuple[Path, Path],
) -> None:
    assembly = fixture_grammar("cfg-toy").assemble()
    grammar = assembly.grammar

    for sentence in read_sentences(fixture[0]):
        forest = parse(
            sentence,
            rules=grammar.rules,
            source=assembly.lexicon,
            semiring=Viterbi(),
            scorer=Unweighted(),
            goal=grammar.goal,
            unary_depth=grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )

        assert tuple(grammar.goal.roots(forest))


def test_the_toy_pack_compiles_to_twenty_rules_over_fifteen_categories() -> None:
    toy = fixture_grammar("cfg-toy")
    assembly = toy.assemble()
    knobs = SyntheticGenerator(
        assembly,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    ).knobs
    lexicon = assembly.lexicon

    assert toy.name == toy.declaration.name == "cfg-toy"
    assert len(assembly.grammar.rules) == 20
    assert knobs.primitive_count == 15
    assert knobs.combinators == tuple(rule.name for rule in assembly.grammar.rules)
    assert len(lexicon.forms) == 28
    assert all(len(assembly.lexicon.entries(form)) == 1 for form in lexicon.forms)


def test_an_unknown_grammar_pack_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown grammar pack 'ccg-huge'"):
        fixture_grammar("ccg-huge")

    with pytest.raises(ValueError, match="unknown grammar pack 'ccg-huge'"):
        main(["--grammar", "ccg-huge", "--seed", "1", "--out", str(tmp_path)])

    assert not list(tmp_path.iterdir())


def test_every_listed_pack_is_found_by_its_own_name() -> None:
    assert [fixture_grammar(grammar.name) for grammar in FIXTURE_GRAMMARS] == list(
        FIXTURE_GRAMMARS
    )


def test_the_fingerprint_file_records_the_grammar_seed_digest_and_shape(
    fixture: tuple[Path, Path],
) -> None:
    pinned = PinnedFingerprint.read(fixture[1])

    assert pinned.grammar == "cfg-toy"
    assert pinned.seed == SEED
    assert (
        pinned.fingerprint == fixture_grammar("cfg-toy").assemble().grammar.fingerprint
    )
    assert pinned.digest == sha256(fixture[0].read_bytes()).hexdigest()
    assert pinned.count == COUNT
    assert pinned.lengths == {LENGTH: COUNT}
    assert json.loads(fixture[1].read_text(encoding="utf-8"))["lengths"] == {
        str(LENGTH): COUNT
    }


def test_the_check_passes_on_a_fresh_fixture_and_returns_its_record(
    fixture: tuple[Path, Path],
) -> None:
    grammar = fixture_grammar("cfg-toy").assemble().grammar

    assert check_pinned_fixture(grammar, *fixture) == PinnedFingerprint.read(fixture[1])


def test_a_changed_grammar_fails_the_check_naming_both_fingerprints(
    fixture: tuple[Path, Path],
) -> None:
    toy = fixture_grammar("cfg-toy").declaration
    changed = (
        FixtureGrammar(
            toy.with_(
                rules=toy.rules.with_(
                    rules=(*toy.rules.rules[:1], *toy.rules.rules[2:])
                )
            )
        )
        .assemble()
        .grammar
    )
    recorded = PinnedFingerprint.read(fixture[1]).fingerprint

    assert changed.fingerprint != recorded

    with pytest.raises(FingerprintMismatchError) as caught:
        check_pinned_fixture(changed, *fixture)

    assert isinstance(caught.value, FixtureMismatchError)
    assert caught.value.recorded == recorded
    assert caught.value.found == changed.fingerprint
    assert recorded in str(caught.value)
    assert changed.fingerprint in str(caught.value)


def test_an_edited_sentence_file_fails_the_check_naming_both_digests(
    copied: tuple[Path, Path],
) -> None:
    grammar = fixture_grammar("cfg-toy").assemble().grammar
    recorded = PinnedFingerprint.read(copied[1]).digest

    with copied[0].open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"tokens":["the","cat","sees","the","dog"]}\n')

    edited = sha256(copied[0].read_bytes()).hexdigest()

    with pytest.raises(SentenceDigestMismatchError) as caught:
        check_pinned_fixture(grammar, *copied)

    assert caught.value.recorded == recorded
    assert caught.value.found == edited
    assert recorded in str(caught.value)
    assert edited in str(caught.value)


def test_an_edited_count_fails_the_check_even_when_the_digest_is_repinned(
    copied: tuple[Path, Path],
) -> None:
    grammar = fixture_grammar("cfg-toy").assemble().grammar
    pinned = PinnedFingerprint.read(copied[1])
    pinned._replace(count=COUNT + 1, lengths={LENGTH: COUNT + 1}).write(copied[1])

    with pytest.raises(ShapeMismatchError) as caught:
        check_pinned_fixture(grammar, *copied)

    assert isinstance(caught.value, FixtureMismatchError)
    assert "sentence count" in str(caught.value)
    assert caught.value.recorded == str(COUNT + 1)
    assert caught.value.found == str(COUNT)


def test_an_edited_histogram_fails_the_check_even_when_the_digest_is_repinned(
    copied: tuple[Path, Path],
) -> None:
    grammar = fixture_grammar("cfg-toy").assemble().grammar
    pinned = PinnedFingerprint.read(copied[1])
    pinned._replace(lengths={LENGTH - 1: 1, LENGTH: COUNT - 1}).write(copied[1])

    with pytest.raises(ShapeMismatchError) as caught:
        check_pinned_fixture(grammar, *copied)

    assert "length histogram" in str(caught.value)
    assert caught.value.recorded == repr({LENGTH - 1: 1, LENGTH: COUNT - 1})
    assert caught.value.found == repr({LENGTH: COUNT})


def test_a_record_whose_histogram_disagrees_with_its_count_is_refused(
    copied: tuple[Path, Path],
) -> None:
    pinned = PinnedFingerprint.read(copied[1])
    pinned._replace(count=COUNT + 1).write(copied[1])

    with pytest.raises(ValueError, match=f"records {COUNT + 1} sentences"):
        PinnedFingerprint.read(copied[1])


def test_a_fingerprint_file_missing_a_field_is_refused(
    copied: tuple[Path, Path],
) -> None:
    copied[1].write_text('{"fingerprint": "abc"}\n', encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="pinned fingerprint record"):
        PinnedFingerprint.read(copied[1])


def test_the_record_round_trips_through_its_file(tmp_path: Path) -> None:
    pinned = PinnedFingerprint("cfg-toy", 3, "f" * 64, "d" * 64, 5, {3: 2, 20: 3})
    pinned.write(tmp_path / "chart.fingerprint")

    assert PinnedFingerprint.read(tmp_path / "chart.fingerprint") == pinned


def test_sampling_draws_every_derivation_of_a_branching_goal() -> None:
    drawn = set(sample_sentences(BRANCHING, 40, 2, SEED))

    assert drawn == {("a", "c"), ("b", "c")}


def test_sampling_yields_the_goal_language_at_every_length() -> None:
    assert set(sample_sentences(TINY, 5, 4, SEED)) == {("a", "a", "a", "a")}
    assert set(sample_sentences(TINY, 5, 6, SEED)) == {("a",) * 6}


def test_sampling_refuses_a_length_the_goal_cannot_derive() -> None:
    with pytest.raises(UnproductiveGrammarError, match="at least 3 and at most 3"):
        sample_sentences(TINY, 1, 3, SEED)


@pytest.mark.parametrize(("count", "length"), [(0, 4), (-1, 4), (1, 0), (1, -2)])
def test_sampling_refuses_a_non_positive_count_or_length(
    count: int,
    length: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_sentences(TINY, count, length, SEED)


def test_write_fixture_records_what_it_wrote(tmp_path: Path) -> None:
    sentences = sample_sentences(BRANCHING, 3, 2, SEED)
    paths = (tmp_path / "chart_3.jsonl", tmp_path / "chart_3.fingerprint")
    pinned = write_fixture(BRANCHING, sentences, SEED, *paths)

    assert read_sentences(paths[0]) == sentences
    assert PinnedFingerprint.read(paths[1]) == pinned
    assert pinned == PinnedFingerprint(
        "branching",
        SEED,
        BRANCHING.assemble().grammar.fingerprint,
        sha256(paths[0].read_bytes()).hexdigest(),
        3,
        {2: 3},
    )


def test_main_prints_both_paths_and_the_recorded_fingerprint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentences, fingerprint = _generate(
        tmp_path, "--count", str(COUNT), "--length", "6", "--out", str(tmp_path)
    )

    assert capsys.readouterr().out == (
        f"{sentences}\n{fingerprint}\n{PinnedFingerprint.read(fingerprint).fingerprint}\n"
    )
    assert all(len(sentence) == 6 for sentence in read_sentences(sentences))


def test_the_default_length_is_the_working_point(tmp_path: Path) -> None:
    sentences, _ = _generate(tmp_path, "--count", str(COUNT), "--out", str(tmp_path))

    assert all(len(sentence) == 20 for sentence in read_sentences(sentences))
