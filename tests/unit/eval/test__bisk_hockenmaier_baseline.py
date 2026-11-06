"""Tests for the registered Bisk--Hockenmaier baseline operation."""

import json
from typing import TYPE_CHECKING, Final
from unittest.mock import patch

import didactic.api as dx
import pytest

from lofi._types import ItemId
from lofi.data import Token
from lofi.data.corpora.ccgbank import MarkedCategoryAlgebra
from lofi.deduce import LogInside, Viterbi, parse
from lofi.eval import BaselineBuildContext
from lofi.eval._bisk_hockenmaier_baseline import (
    _AutoSentence,
    _gold_category,
    _grammar,
    _in_length_band,
    _inducer,
    _item,
    _root_feeders,
    build_bisk_hockenmaier_baseline,
)
from lofi.eval._bisk_hockenmaier_baseline import (
    _reference as _read_reference,
)
from lofi.spec import CombinatorSetSpec
from lofi.spec.baselines.bisk_hockenmaier import (
    BiskHockenmaierBaselineSpec,
    bisk_hockenmaier_grammar,
)
from lofi.spec.components.run import ParseSpec, TopKPrunerSpec, ViterbiSemiringSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.spec.grammar import GrammarDecl

_AUTO: Final = (
    "ID={identifier} PARSER=GOLD NUMPARSE=1\n"
    "(<T N 0 2> (<L N NN NN dogs N>) (<L N\\N . . . N\\N>) )\n"
)
_DEPENDENCIES: Final = (
    "1\tdogs\t_\tNN\t_\t_\t_\t_\t0\troot\n2\t.\t_\t.\t_\t_\t_\t_\t1\tpunct\n"
)
_METRICS: Final = (
    "supertagging-length-10",
    "dependencies-length-10",
    "supertagging-length-20",
    "dependencies-length-20",
    "supertagging-all-lengths",
    "dependencies-all-lengths",
)


def _reference(score: float) -> dict[str, JsonValue]:
    return {
        "source": {
            "title": "Synthetic Figure 4(a) reference",
            "authors": ["Test Author"],
            "venue": "Test",
            "url": "https://example.test/bisk-hockenmaier",
            "figure": "Figure 4(a)",
        },
        "metric": "unlabeled directed attachment",
        "training_band": "wsj10",
        "evaluation_section": 23,
        "points": [
            {"metric": "dependencies-length-10", "score_percent": score},
            {"metric": "dependencies-length-20", "score_percent": score},
            {"metric": "dependencies-all-lengths", "score_percent": score},
        ],
    }


def _reduced_grammar() -> GrammarDecl:
    grammar = bisk_hockenmaier_grammar()
    return grammar.with_(budget=grammar.budget.with_(max_category_arity=3))


def _fixture(root: Path, score: float) -> BiskHockenmaierBaselineSpec:
    ccgbank = root / "ccgbank"
    development = ccgbank / "00"
    training = ccgbank / "02"
    evaluation = ccgbank / "23"
    development.mkdir(parents=True)
    training.mkdir()
    evaluation.mkdir()
    (development / "wsj_0000.auto").write_text(
        _AUTO.format(identifier="wsj_0000.1"), encoding="utf-8", newline="\n"
    )
    (training / "wsj_0200.auto").write_text(
        _AUTO.format(identifier="wsj_0200.1"), encoding="utf-8", newline="\n"
    )
    (evaluation / "wsj_2300.auto").write_text(
        _AUTO.format(identifier="wsj_2300.1"), encoding="utf-8", newline="\n"
    )
    (root / "evaluation.conll").write_text(
        _DEPENDENCIES, encoding="utf-8", newline="\n"
    )
    (root / "reference.json").write_text(
        json.dumps(_reference(score)), encoding="utf-8", newline="\n"
    )
    return BiskHockenmaierBaselineSpec(
        ccgbank_source="ccgbank",
        dependency_gold_source="evaluation.conll",
        reference_source="reference.json",
        reference_tolerance=0.0,
        training_sections=(2,),
        training_max_length=1,
        grammar=_reduced_grammar(),
        nominal_category="N",
        sentence_category="S",
        coordination_category="conj",
        gold_sentence_atom="S",
        gold_coordination_atom="conj",
        nominal_tags=frozenset({"NN"}),
        verbal_tags=frozenset({"VB", "VBG"}),
        coordination_tags=frozenset({"CC"}),
        other_tags=frozenset({"JJ", "."}),
        punctuation_tags=frozenset({"."}),
        headed_or_modifier_arguments=("N", "S"),
        sentence_arguments=("N",),
        second_order_modifiers=False,
        max_arity=1,
        category_induction_steps=1,
        derived_constituent_pass=False,
        em_max_iterations=1,
        convergence_tolerance=1e-6,
    )


def _wsj_declaration() -> BiskHockenmaierBaselineSpec:
    return BiskHockenmaierBaselineSpec(
        ccgbank_source="ccgbank",
        dependency_gold_source="evaluation.conll",
        reference_source="reference.json",
        reference_tolerance=0.0,
        nominal_category="N",
        sentence_category="S",
        coordination_category="conj",
        gold_sentence_atom="S",
        gold_coordination_atom="conj",
        nominal_tags=frozenset({"NN", "NNS", "NNP", "NNPS"}),
        verbal_tags=frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}),
        coordination_tags=frozenset({"CC"}),
        other_tags=frozenset({"JJ", "RB", "."}),
        punctuation_tags=frozenset({"."}),
        headed_or_modifier_arguments=("N", "S"),
        sentence_arguments=("N",),
        em_max_iterations=50,
        convergence_tolerance=1e-6,
    )


def test_the_packaged_grammar_derives_a_five_token_wsj_sentence() -> None:
    settings = _wsj_declaration().settings()
    grammar = _grammar(settings)
    forms = ("Ms.", "Haag", "plays", "Elianti", ".")
    sentence = _AutoSentence(
        2,
        ItemId("wsj_0200.2"),
        forms,
        ("NNP", "NNP", "VBZ", "NNP", "."),
        tuple(Token(position, form) for position, form in enumerate(forms)),
        ("N/N", "N", "S\\N/N", "N", "."),
    )

    item = _item(
        sentence, grammar, settings, _inducer(settings, grammar, settings.conventions)
    )
    assembly = grammar.assembly
    roots = {
        assembly.syntax.render(assembly.store, key.reading)
        for key in item.chart.items_at((0, len(forms)))
    }

    assert "TOP" in roots


def test_the_baseline_runs_real_auto_and_conll_with_one_punctuation_mask(
    tmp_path: Path,
) -> None:
    declared = _fixture(tmp_path, 0.0)

    result = build_bisk_hockenmaier_baseline(
        declared.settings(), BaselineBuildContext(tmp_path, 7)
    )()
    measurements = result.measurements

    assert tuple(row.metric for row in measurements) == _METRICS
    assert tuple(row.value for row in measurements) == (1.0, 0.0) * 3
    assert tuple(row.n for row in measurements) == (1,) * 6
    assert set(result.source_digests) == {
        "ccgbank-auto",
        "dependency-gold",
        "published-reference",
    }


def test_vbg_does_not_force_the_sentence_root_and_long_items_stay_in_all(
    tmp_path: Path,
) -> None:
    settings = _fixture(tmp_path, 0.0).settings()
    grammar = _grammar(settings)
    forms = tuple(f"word-{position}" for position in range(21))
    sentence = _AutoSentence(
        23,
        ItemId("wsj_2300.2"),
        forms,
        ("NN",) * 21,
        tuple(Token(position, form) for position, form in enumerate(forms)),
        ("N",) * 21,
    )

    assert _root_feeders(("VBG",), grammar, settings) is None
    assert _root_feeders(("VBG", "VB"), grammar, settings) == {grammar.sentence}
    assert not _in_length_band(sentence, 20, settings)
    assert _in_length_band(sentence, None, settings)


def test_the_reference_gate_refuses_a_planted_figure_mismatch(tmp_path: Path) -> None:
    declared = _fixture(tmp_path, 100.0)
    operation = build_bisk_hockenmaier_baseline(
        declared.settings(), BaselineBuildContext(tmp_path, 7)
    )

    with pytest.raises(
        ValueError, match="Figure 4\\(a\\) reference gate failed"
    ) as refused:
        operation()

    for metric in (
        "dependencies-length-10",
        "dependencies-length-20",
        "dependencies-all-lengths",
    ):
        assert metric in str(refused.value)


def test_the_reference_boundary_refuses_reordered_metrics(tmp_path: Path) -> None:
    document = _reference(0.0)
    points = document["points"]
    if not isinstance(points, list):
        msg = "the reference fixture has no points"
        raise TypeError(msg)
    points.reverse()
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="incomplete or reordered"):
        _read_reference(path, 23)


def test_source_content_identity_changes_when_bytes_change_at_the_same_paths(
    tmp_path: Path,
) -> None:
    declared = _fixture(tmp_path, 0.0)
    operation = build_bisk_hockenmaier_baseline(
        declared.settings(), BaselineBuildContext(tmp_path, 7)
    )
    original = operation().source_digests

    auto = tmp_path / "ccgbank/00/wsj_0000.auto"
    auto.write_text(
        auto.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n"
    )
    changed_auto = operation().source_digests

    dependency = tmp_path / "evaluation.conll"
    dependency.write_text(
        dependency.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n"
    )
    changed_dependency = operation().source_digests

    reference = tmp_path / "reference.json"
    reference.write_text(
        reference.read_text(encoding="utf-8").replace('"metric":', '  "metric":', 1),
        encoding="utf-8",
        newline="\n",
    )
    changed_reference = operation().source_digests

    assert changed_auto["ccgbank-auto"] != original["ccgbank-auto"]
    assert changed_auto["dependency-gold"] == original["dependency-gold"]
    assert changed_dependency["dependency-gold"] != changed_auto["dependency-gold"]
    assert (
        changed_reference["published-reference"]
        != changed_dependency["published-reference"]
    )


@pytest.mark.parametrize(
    ("parse", "semiring"),
    [
        (None, LogInside),
        (
            ParseSpec(
                retain="forest",
                reachability="roots",
                pruner=TopKPrunerSpec(width=1),
                semiring=ViterbiSemiringSpec(),
            ),
            Viterbi,
        ),
    ],
)
def test_the_declared_parse_controls_the_baseline_support_parse(
    tmp_path: Path,
    parse: ParseSpec | None,
    semiring: type[LogInside | Viterbi],
) -> None:
    fixture = _fixture(tmp_path, 0.0)
    declared = fixture if parse is None else fixture.with_(parse=parse)
    settings = declared.settings()
    grammar = _grammar(settings)
    forms = ("dogs", ".")
    sentence = _AutoSentence(
        23,
        ItemId("wsj_2300.1"),
        forms,
        ("NN", "."),
        tuple(Token(position, form) for position, form in enumerate(forms)),
        ("N", "N\\N"),
    )
    item = _item(
        sentence,
        grammar,
        settings,
        _inducer(settings, grammar, settings.conventions),
    )

    assert type(item.chart.semiring) is semiring
    assert declared.conventions()["parse"] == declared.parse.model_dump()


@pytest.mark.parametrize("threads", [1, 3])
def test_the_declared_threads_reach_the_baseline_support_parse(
    tmp_path: Path, threads: int
) -> None:
    settings = _fixture(tmp_path, 0.0).with_(threads=threads).settings()
    grammar = _grammar(settings)
    forms = ("dogs", ".")
    sentence = _AutoSentence(
        23,
        ItemId("wsj_2300.1"),
        forms,
        ("NN", "."),
        tuple(Token(position, form) for position, form in enumerate(forms)),
        ("N", "N\\N"),
    )
    inducer = _inducer(settings, grammar, settings.conventions)

    with patch("lofi.eval._bisk_hockenmaier_baseline.parse", wraps=parse) as parser:
        _item(sentence, grammar, settings, inducer)

    assert parser.call_args is not None
    assert parser.call_args.kwargs["threads"] == threads


def test_a_baseline_parse_that_keeps_no_forest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(dx.ValidationError, match="packed forest"):
        _fixture(tmp_path, 0.0).with_(
            parse=ParseSpec(retain="inside", reachability="roots")
        )


def test_the_declared_grammar_supplies_the_support_parse_rules(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, 0.0)
    grammar = fixture.grammar
    combinators, *others = grammar.rules.rules
    assert isinstance(combinators, CombinatorSetSpec)
    declared = fixture.with_(
        grammar=grammar.with_(
            rules=grammar.rules.with_(
                rules=(combinators.with_(composition=(1, 2)), *others)
            )
        )
    )
    settings = declared.settings()
    built = _grammar(settings)
    forms = ("dogs", ".")
    sentence = _AutoSentence(
        23,
        ItemId("wsj_2300.1"),
        forms,
        ("NN", "."),
        tuple(Token(position, form) for position, form in enumerate(forms)),
        ("N", "N\\N"),
    )

    with patch("lofi.eval._bisk_hockenmaier_baseline.parse", wraps=parse) as parser:
        _item(
            sentence, built, settings, _inducer(settings, built, settings.conventions)
        )

    assert parser.call_args is not None
    names = {rule.name for rule in parser.call_args.kwargs["rules"]}
    assert "composition:/:2:harmonic:steedman" in names
    assert "composition:/:1:harmonic:steedman" in names


def test_the_support_grammar_raises_only_to_the_declared_targets(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, 0.0)
    built = _grammar(fixture.settings())
    assembly = built.assembly
    rules = {rule.name: rule for rule in assembly.grammar.rules}

    raised = {
        assembly.syntax.render(assembly.store, category)
        for name in sorted(built.raising_rules)
        for category in rules[name].apply((built.nominal,))
    }

    assert raised == {"S/(S\\N)", "S\\(S/N)", "N/(N\\N)", "N\\(N/N)"}


@pytest.mark.parametrize("category", ["S\\N", "S/(S\\N)", "TOP", "N"])
def test_the_baseline_coordinates_raised_composed_and_root_categories(
    tmp_path: Path, category: str
) -> None:
    built = _grammar(_fixture(tmp_path, 0.0).settings())
    assembly = built.assembly
    rules = {rule.name: rule for rule in assembly.grammar.rules}
    conjunct = assembly.syntax.parse(assembly.store, category)

    marked = tuple(rules["coordination-mark"].apply((built.coordination, conjunct)))

    assert tuple(assembly.syntax.render(assembly.store, term) for term in marked) == (
        f"@conj({category})",
    )
    assert tuple(rules["coordination-merge"].apply((conjunct, marked[0]))) == (
        conjunct,
    )
    assert rules["coordination-mark"].admits((built.coordination, conjunct), marked[0])
    assert rules["coordination-merge"].admits((conjunct, marked[0]), conjunct)


def test_gold_categories_erase_to_the_declared_gold_atoms(tmp_path: Path) -> None:
    declared = _fixture(tmp_path, 0.0).with_(
        gold_sentence_atom="TOP", gold_coordination_atom="CC"
    )
    grammar = _grammar(declared.settings())
    algebra = MarkedCategoryAlgebra()

    assert _gold_category("TOP[dcl]", grammar, algebra) == grammar.sentence
    assert _gold_category("CC", grammar, algebra) == grammar.coordination
    assert _gold_category("S", grammar, algebra) == grammar.nominal
    assert _gold_category("conj", grammar, algebra) == grammar.nominal
