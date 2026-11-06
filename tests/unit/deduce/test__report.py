from __future__ import annotations

import json
from itertools import chain, product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override
from unittest.mock import patch

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.deduce import (
    Counting,
    DerivationCountRow,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    UnitEdges,
    derivation_count_report,
    parse,
)
from lofi.judge import NoNormalForm, RuleSet
from lofi.lexicon import Entry
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.terms import TargetSet, Term

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Cell
    from lofi.judge import (
        NormalForm,
        NormalFormDecisionTable,
        Rule,
    )
    from lofi.lexicon import Clamp
    from lofi.terms import Matching


_STEPS: Final = (
    ("application-f-a", "F", "A", "S"),
    ("application-g-a", "G", "A", "A"),
    ("composition-f-g", "F", "G", "F"),
)
_COMPILED: Final = declared_rules(
    ccg_signature("S", "F", "G", "A", name="derivation-report"),
    RuleSetSpec(
        name="derivation-report",
        rules=tuple(
            PatternRuleSpec(
                name=name,
                premises=(
                    ConstantPatternSpec(name=left),
                    ConstantPatternSpec(name=right),
                ),
                conclusion=ConstantPatternSpec(name=output),
                head_policy="non-absorbing",
            )
            for name, left, right, output in _STEPS
        ),
    ),
)
MATCHING: Final[Matching[Term]] = _COMPILED.rule_set.matching


def _category(name: str) -> Term:
    """Return one declared report-fixture category."""
    return _COMPILED.context.syntax.parse(_COMPILED.context.store, name)


class RedundancyNormalForm:
    """Keeps the right-associated representative of one ambiguity."""

    @property
    def name(self) -> str:
        """Return the normal-form name."""
        return "right-associated"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return lexical, composition, and other tags."""
        return frozenset({"L", "C", "O"})

    def lexical_tag(self) -> str:
        """Return the lexical tag."""
        return "L"

    def decisions(self, rules: Sequence[Rule[Term]], /) -> NormalFormDecisionTable:
        """Defer every finite tag state to exact step admission."""
        alphabet = tuple(sorted(self.alphabet))

        return tuple(
            MappingProxyType(
                dict.fromkeys(product(alphabet, repeat=rule.arity), "premises")
            )
            for rule in rules
        )

    def tag(self, rule: Rule[Term], children: tuple[str, ...], /) -> str:
        """Return the tag for a rule conclusion.

        Parameters
        ----------
        rule
            Applied rule.
        children
            Premise tags.
        """
        del children

        return "C" if rule.name.startswith("composition") else "O"

    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Return whether a step is the selected representative.

        Parameters
        ----------
        rule
            Applied rule.
        children
            Premise tags.
        inputs
            Premise terms.
        output
            Rule conclusion.
        """
        del output

        return len(children) == len(inputs) == rule.arity and not (
            rule.name.startswith("application") and children[0] == "C"
        )


class OverRestrictiveNormalForm(RedundancyNormalForm):
    """Rejects every application step."""

    @property
    @override
    def name(self) -> str:
        """Return the normal-form name."""
        return "over-restrictive"

    @override
    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Return whether a step is not application.

        Parameters
        ----------
        rule
            Applied rule.
        children
            Premise tags.
        inputs
            Premise terms.
        output
            Rule conclusion.
        """
        return super().admits(rule, children, inputs, output) and (
            not rule.name.startswith("application")
        )


class ToySource:
    """A repeatable finite lexical source."""

    __slots__ = ()

    @property
    def is_re_callable(self) -> bool:
        """Return whether enumeration may be repeated."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return no clamps."""
        return ()

    def axioms(
        self,
        tokens: Sequence[str],
        /,
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        """Enumerate the declared lexical categories.

        Parameters
        ----------
        tokens
            Sentence tokens.

        Returns
        -------
        Iterable of tuple
            Position, entry, and a deliberately nonunit source weight.
        """
        categories = {"f": _category("F"), "g": _category("G"), "a": _category("A")}

        return tuple(
            (position, Entry(token, categories[token]), 7)
            for position, token in enumerate(tokens)
            if token in categories
        )


class DrainingSource(ToySource):
    """A lexical source that cannot be read repeatedly."""

    @property
    @override
    def is_re_callable(self) -> bool:
        """Return whether enumeration may be repeated, which it may not."""
        return False

    @override
    def axioms(
        self,
        tokens: Sequence[str],
        /,
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        """Refuse lexical enumeration.

        Parameters
        ----------
        tokens
            Sentence tokens.
        """
        del tokens
        msg = "a nonrepeatable source was enumerated"
        raise AssertionError(msg)


class UnitScorer:
    """Assigns unit weights to chart edges."""

    lowering = UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> int:
        """Return unit lexical weight.

        Parameters
        ----------
        position
            Token position.
        entry
            Lexical entry.
        """
        del position, entry

        return 1

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> int:
        """Return unit rule weight.

        Parameters
        ----------
        rule
            Applied rule.
        inputs
            Premise terms.
        output
            Rule conclusion.
        span
            Conclusion span.
        split
            Binary split point.
        """
        del rule, inputs, output, span, split

        return 1


APPLICATION: Final[tuple[Rule[Term], ...]] = _COMPILED.rule_set.rules[:2]
COMPOSITION: Final[Rule[Term]] = _COMPILED.rule_set.rules[2]
NORMAL_FORM: Final[NormalForm[Term]] = RedundancyNormalForm()
SOURCE: Final = ToySource()
GOAL: Final = RootGoal(TargetSet((_category("S"),), MATCHING))
APPLICATION_ONLY: Final = RuleSet(
    "application-only",
    APPLICATION,
    MATCHING,
    NORMAL_FORM,
)
WITH_COMPOSITION: Final = RuleSet(
    "with-composition",
    (*APPLICATION, COMPOSITION),
    MATCHING,
    NORMAL_FORM,
)
VOCABULARY: Final = ("a", "f", "g")
MAX_SENTENCE_LENGTH: Final = 3
SENTENCES: Final = tuple(
    chain.from_iterable(
        product(VOCABULARY, repeat=length)
        for length in range(1, MAX_SENTENCE_LENGTH + 1)
    )
)


def _goal_terms(
    inventory: RuleSet[Term],
    normal_form: NormalForm[Term],
    tokens: tuple[str, ...],
) -> frozenset[Term]:
    """Return matching-keyed goal terms from the real chart.

    Parameters
    ----------
    inventory
        Rule inventory.
    normal_form
        Normal form to apply.
    tokens
        Sentence tokens.
    """
    selected = RuleSet(
        inventory.name,
        inventory.rules,
        inventory.matching,
        normal_form,
    )
    chart = parse(
        tokens,
        rules=selected,
        source=SOURCE,
        semiring=Counting(),
        scorer=UnitScorer(),
        goal=GOAL,
        unary_depth=0,
        threads=1,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )

    return frozenset(
        inventory.matching.key(chart.term(key)) for key in GOAL.roots(chart)
    )


def _assert_complete(normal_form: NormalForm[Term]) -> None:
    """Require goal-set equality for every declared sentence.

    Parameters
    ----------
    normal_form
        Normal form to check.
    """
    for tokens in SENTENCES:
        assert _goal_terms(WITH_COMPOSITION, normal_form, tokens) == _goal_terms(
            WITH_COMPOSITION,
            NoNormalForm(),
            tokens,
        )


def _row_spec(row: DerivationCountRow) -> dict[str, JsonValue]:
    """Return a report row as JSON-compatible fields.

    Parameters
    ----------
    row
        Report row.
    """
    return {
        "inventory_name": row.inventory_name,
        "sentence_position": row.sentence_position,
        "tokens": list(row.tokens),
        "normal_form_count": row.normal_form_count,
        "unconstrained_count": row.unconstrained_count,
        "retention_ratio": row.retention_ratio,
    }


def test_real_chart_checks_every_sentence_through_length_three() -> None:
    _assert_complete(NORMAL_FORM)


def test_real_chart_census_detects_an_over_restrictive_filter() -> None:
    with pytest.raises(AssertionError):
        _assert_complete(OverRestrictiveNormalForm())


def test_count_report_matches_the_committed_fixture() -> None:
    rows = derivation_count_report(
        (APPLICATION_ONLY, WITH_COMPOSITION),
        (("f", "a"), ("f", "g", "a"), ("unknown",)),
        source=SOURCE,
        goal=GOAL,
        unary_depth=0,
        threads=1,
    )
    fixture = Path(__file__).with_name("fixtures") / "derivation_counts.json"

    assert [_row_spec(row) for row in rows] == json.loads(
        fixture.read_text(encoding="utf-8")
    )
    assert any(
        row.retention_ratio is not None and 0.0 < row.retention_ratio < 1.0
        for row in rows
    )


def test_count_report_preserves_order_and_snapshots_sentences() -> None:
    inventories = [WITH_COMPOSITION, APPLICATION_ONLY]
    sentences = [["f", "a"], ["unknown"]]
    original = [sentence.copy() for sentence in sentences]
    rows = derivation_count_report(
        inventories,
        sentences,
        source=SOURCE,
        goal=GOAL,
        unary_depth=0,
        threads=1,
    )

    assert inventories == [WITH_COMPOSITION, APPLICATION_ONLY]
    assert sentences == original
    assert WITH_COMPOSITION.rules == (*APPLICATION, COMPOSITION)
    assert WITH_COMPOSITION.normal_form is NORMAL_FORM
    assert GOAL.categories == (_category("S"),)

    sentences[0].append("g")

    assert sentences[1] == original[1]
    assert tuple((row.inventory_name, row.sentence_position) for row in rows) == (
        ("with-composition", 0),
        ("with-composition", 1),
        ("application-only", 0),
        ("application-only", 1),
    )
    assert rows[0].tokens == tuple(original[0])
    assert rows[1].normal_form_count == rows[1].unconstrained_count == 0
    assert rows[1].retention_ratio is None


def test_count_report_returns_immutable_rows() -> None:
    (row,) = derivation_count_report(
        (APPLICATION_ONLY,),
        (("f", "a"),),
        source=SOURCE,
        goal=GOAL,
        unary_depth=0,
        threads=1,
    )

    with pytest.raises(AttributeError):
        row.__setattr__("normal_form_count", 2)


def test_count_report_rejects_duplicate_inventory_names() -> None:
    duplicate = RuleSet(
        APPLICATION_ONLY.name,
        (*APPLICATION, COMPOSITION),
        MATCHING,
        NORMAL_FORM,
    )

    with pytest.raises(ValueError, match="duplicate names"):
        derivation_count_report(
            (APPLICATION_ONLY, duplicate),
            (("f", "a"),),
            source=SOURCE,
            goal=GOAL,
            unary_depth=0,
            threads=1,
        )


def test_count_report_rejects_a_nonrepeatable_source_before_enumeration() -> None:
    with pytest.raises(ValueError, match="repeated enumeration"):
        derivation_count_report(
            (APPLICATION_ONLY,),
            (("f", "a"),),
            source=DrainingSource(),
            goal=GOAL,
            unary_depth=0,
            threads=1,
        )


def test_count_report_parses_with_the_declared_threads() -> None:
    with patch("lofi.deduce._report.parse", wraps=parse) as parser:
        derivation_count_report(
            (APPLICATION_ONLY,),
            (("f", "a"),),
            source=SOURCE,
            goal=GOAL,
            unary_depth=0,
            threads=2,
        )

    assert parser.call_count == 2
    assert {call.kwargs["threads"] for call in parser.call_args_list} == {2}


def test_count_report_rejects_a_non_positive_thread_count() -> None:
    with pytest.raises(ValueError, match="threads"):
        derivation_count_report(
            (APPLICATION_ONLY,),
            (("f", "a"),),
            source=SOURCE,
            goal=GOAL,
            unary_depth=0,
            threads=0,
        )
