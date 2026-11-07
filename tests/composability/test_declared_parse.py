"""Declared goals and chart components compose into one symbolic parse."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
from didactic.settings import load_document

from lofi.deduce import (
    CellPruner,
    CoverGoal,
    ParsePolicy,
    Retain,
    RootGoal,
    UnitEdges,
    Viterbi,
    parse,
)
from lofi.run import section_root
from lofi.spec.components.run import ParseSpec
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.deduce import Cell, Semiring
    from lofi.judge import Rule
    from lofi.lexicon import Entry
    from lofi.terms import Term


GRAMMAR: Final = Path(__file__).parent / "fixtures" / "constructors" / "c01.yaml"


class _UnitScorer:
    """Assign the multiplicative identity to every lexical and rule step."""

    @property
    def lowering(self) -> UnitEdges:
        """Return the native unit-edge scorer."""
        return UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        """Return the log-space multiplicative identity."""
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the log-space multiplicative identity."""
        del rule, inputs, output, span, split
        return 0.0


def _document(path: Path, /) -> dict[str, JsonValue]:
    """Load one mapping-valued configuration document."""
    return cast("dict[str, JsonValue]", load_document(path))


@pytest.mark.parametrize("parse_name", ["exhaustive", "viterbi-beam"])
@pytest.mark.parametrize("goal_kind", ["root", "cover"])
def test_one_grammar_document_sweeps_goals_pruning_and_semirings(
    parse_name: str,
    goal_kind: str,
) -> None:
    """Lower each declared point and parse the fixture with no identity branch."""
    document = _document(GRAMMAR)
    document["goal"] = (
        {"kind": "root", "categories": ["S"], "root_weights": [-0.25]}
        if goal_kind == "root"
        else {
            "kind": "cover",
            "categories": ["S"],
            "max_fragments": 2,
            "penalty": 0.75,
            "penalty_form": "after-first",
            "tiling": "lexical",
        }
    )
    declaration = GrammarDecl.model_validate(document)
    assembly = GrammarIso().forward(
        declaration,
        registered_language("propositional"),
    )
    grammar = assembly.grammar
    settings = ParseSpec.model_validate(
        _document(section_root("parse") / f"{parse_name}.yaml")
    ).build()
    tokens = ("w0", "w1") if goal_kind == "root" else ("w0", "w3")
    chart = parse(
        tokens,
        rules=grammar.rules,
        source=assembly.lexicon,
        semiring=cast("Semiring[float]", settings.semiring),
        scorer=_UnitScorer(),
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        pruner=cast("CellPruner[Term, float] | None", settings.pruner),
        prune_lexical=settings.prune_lexical,
        strategy=settings.strategy,
        span_index=settings.span_index,
        chart_store=settings.chart_store,
        policy=cast("ParsePolicy[Literal[Retain.FOREST]]", settings.policy),
    )
    forest = chart
    groups = grammar.goal.root_groups(forest)

    assert groups

    if isinstance(grammar.goal, RootGoal):
        (root,) = groups[0]
        assert grammar.goal.root_weight(root) == -0.25
        assert tuple(forest.derivations(root))
    elif isinstance(grammar.goal, CoverGoal):
        assert len(groups) == 2
        assert grammar.goal.combination_weight(len(groups)) == -0.75
    else:
        pytest.fail(f"unexpected goal {grammar.goal!r}")

    if parse_name == "viterbi-beam":
        assert isinstance(forest.semiring, Viterbi)
        assert settings.pruner is not None
