"""Tests for the induced-grammar dependency convention."""

import json
import subprocess
import sys
from copy import copy, deepcopy
from pathlib import Path
from pickle import dumps, loads
from typing import TYPE_CHECKING, NamedTuple, cast, override

import pytest
from tests.unit.data.deps.test__projection import (
    DependencyProjectorConformanceTests,
    ProjectorCase,
)

from lofi.data import Dependency, Derivation, DerivationNode, Token, TokenSpan
from lofi.data.deps import (
    BISK_HOCKENMAIER_CONVENTION,
    BiskHockenmaierDependencyProjector,
    CCGCategoryAlgebra,
    CCGDependencyConvention,
    CCGDependencyProjector,
    CCGProjectionReport,
    CoindexedPath,
    InducedDependencyState,
    project_bisk_hockenmaier_dependencies,
    project_dependencies,
)
from lofi.spec import HeadPolicy, HeadSelection

if TYPE_CHECKING:
    from lofi._types import JsonValue

FIXTURE = Path(__file__).with_name("fixtures") / "bisk_hockenmaier.json"


class FixtureAlgebra:
    """Structural operations declared by the golden fixture.

    Parameters
    ----------
    functors
        Result and argument categories keyed by functor.
    head_sharing
        Functors whose result takes its argument's head.
    """

    __slots__ = ("_functors", "_head_sharing")

    def __init__(
        self,
        functors: dict[str, tuple[str, str]],
        head_sharing: frozenset[str],
    ) -> None:
        self._functors = dict(functors)
        self._head_sharing = head_sharing

    def is_functor(self, category: str, /) -> bool:
        """Return whether the fixture declares a category as a functor.

        Parameters
        ----------
        category
            Category to inspect.
        """
        return category in self._functors

    def result(self, category: str, /) -> str:
        """Return a fixture functor's result.

        Parameters
        ----------
        category
            Declared functor category.
        """
        return self._functors[category][0]

    def argument(self, category: str, /) -> str:
        """Return a fixture functor's argument.

        Parameters
        ----------
        category
            Declared functor category.
        """
        return self._functors[category][1]

    def equivalent(self, left: str, right: str, /) -> bool:
        """Return whether two fixture categories are equal.

        Parameters
        ----------
        left
            First category.
        right
            Second category.
        """
        return left == right

    def shares_head(self, category: str, /) -> bool:
        """Return whether a fixture functor passes its argument head.

        Parameters
        ----------
        category
            Category to inspect.
        """
        return category in self._head_sharing

    def coindices(self, category: str, /) -> tuple[CoindexedPath, ...]:
        """Return no coindices, since the fixture declares none.

        Parameters
        ----------
        category
            Category to inspect.
        """
        del category
        return ()


class GoldenCase(NamedTuple):
    """One fixture derivation and its expected dependency arcs."""

    name: str
    derivation: Derivation[str]
    tokens: tuple[Token, ...]
    selections: dict[str, HeadSelection]
    arcs: tuple[tuple[int, int], ...]


class GoldenFixture(NamedTuple):
    """The fixture algebra and projection cases."""

    algebra: CCGCategoryAlgebra[str]
    coordination_marks: frozenset[str]
    cases: tuple[GoldenCase, ...]


def _golden_fixture() -> GoldenFixture:
    """Load the hand-written induced-dependency fixture."""
    document = cast(
        "dict[str, JsonValue]",
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )
    functor_values = cast("dict[str, JsonValue]", document["functors"])
    functors: dict[str, tuple[str, str]] = {}
    head_sharing: set[str] = set()

    for category, value in functor_values.items():
        declaration = cast("dict[str, JsonValue]", value)
        functors[category] = (
            cast("str", declaration["result"]),
            cast("str", declaration["argument"]),
        )

        if cast("bool", declaration["shares_head"]):
            head_sharing.add(category)

    algebra = FixtureAlgebra(functors, frozenset(head_sharing))
    cases: list[GoldenCase] = []

    for value in cast("list[JsonValue]", document["cases"]):
        declaration = cast("dict[str, JsonValue]", value)
        nodes: dict[int, DerivationNode[str]] = {}

        for node_value in cast("list[JsonValue]", declaration["nodes"]):
            node = cast("dict[str, JsonValue]", node_value)
            span = cast("list[int]", node["span"])
            nodes[cast("int", node["id"])] = DerivationNode(
                TokenSpan(span[0], span[1]),
                cast("str", node["label"]),
                rule=cast("str | None", node.get("rule")),
                children=tuple(cast("list[int]", node.get("children", []))),
            )

        selections = {
            name: HeadSelection(
                cast("HeadPolicy", parts[0]),
                cast("int | None", parts[1]),
            )
            for name, raw_parts in cast(
                "dict[str, JsonValue]",
                declaration["selections"],
            ).items()
            for parts in (cast("list[JsonValue]", raw_parts),)
        }
        tokens = tuple(
            Token(position, surface)
            for position, surface in enumerate(cast("list[str]", declaration["tokens"]))
        )
        arcs = tuple(
            (parts[0], parts[1])
            for raw_parts in cast("list[JsonValue]", declaration["arcs"])
            for parts in (cast("list[int]", raw_parts),)
        )
        cases.append(
            GoldenCase(
                cast("str", declaration["name"]),
                Derivation(cast("int", declaration["root"]), nodes),
                tokens,
                selections,
                arcs,
            )
        )

    return GoldenFixture(
        algebra,
        frozenset(cast("list[str]", document["coordination_marks"])),
        tuple(cases),
    )


class TestBiskHockenmaierDependencyProjector(
    DependencyProjectorConformanceTests[
        str,
        Dependency,
        InducedDependencyState[str],
    ]
):
    """Protocol conformance for the induced dependency projector."""

    @override
    def make_projector_case(
        self,
    ) -> ProjectorCase[str, Dependency, InducedDependencyState[str]]:
        """Build a lexical and unary projector case."""
        fixture = _golden_fixture()
        return ProjectorCase(
            BiskHockenmaierDependencyProjector(
                fixture.algebra,
                fixture.coordination_marks,
            ),
            DerivationNode(TokenSpan(0, 1), "N"),
            Token(0, "birds"),
            DerivationNode(
                TokenSpan(0, 1),
                "NP",
                rule="type-change",
                children=(0,),
            ),
            "type-change",
            HeadSelection("pass-through", 0),
            (0,),
        )


def test_five_golden_derivations_match_arc_for_arc() -> None:
    fixture = _golden_fixture()
    projector = BiskHockenmaierDependencyProjector(
        fixture.algebra,
        fixture.coordination_marks,
    )

    assert len(fixture.cases) == 5

    for case in fixture.cases:
        projection = project_bisk_hockenmaier_dependencies(
            case.derivation,
            case.tokens,
            case.selections,
            projector,
        )
        reached = tuple(
            (dependency.head, dependency.dependent)
            for dependency in projection.dependencies
        )

        assert reached == case.arcs, case.name
        assert all(head != dependent for head, dependent in reached)
        assert sum(head == 0 for head, _ in reached) == 1
        assert all(
            dependency.label == ("root" if dependency.head == 0 else "dep")
            for dependency in projection.dependencies
        )


def test_modifier_direction_and_provenance_differ_from_ccgbank() -> None:
    fixture = _golden_fixture()
    modifier = next(case for case in fixture.cases if case.name == "modifier inversion")
    primary_projector = CCGDependencyProjector(
        fixture.algebra,
        CCGDependencyConvention("ccgbank"),
    )
    primary = project_dependencies(
        modifier.derivation,
        modifier.tokens,
        modifier.selections,
        primary_projector,
    )
    secondary_projector = BiskHockenmaierDependencyProjector(
        fixture.algebra,
        fixture.coordination_marks,
    )
    secondary = secondary_projector.complete(
        project_dependencies(
            modifier.derivation,
            modifier.tokens,
            modifier.selections,
            secondary_projector,
        )
    )

    assert tuple(
        (dependency.predicate_position + 1, dependency.argument_position + 1)
        for dependency in primary.dependencies
    ) == ((1, 2),)
    assert tuple(
        (dependency.head, dependency.dependent)
        for dependency in secondary.dependencies
        if dependency.head != 0
    ) == ((2, 1),)
    assert primary_projector.report(primary).convention_name == "ccgbank"
    assert secondary_projector.report(secondary) == CCGProjectionReport(
        "bisk-hockenmaier-conll-2008-coverage-secondary",
        0,
        0,
    )
    assert secondary_projector.convention is BISK_HOCKENMAIER_CONVENTION
    assert BISK_HOCKENMAIER_CONVENTION.templates == ()


def test_coordination_requires_mark_then_merge_and_one_root() -> None:
    fixture = _golden_fixture()
    projector = BiskHockenmaierDependencyProjector(
        fixture.algebra,
        fixture.coordination_marks,
    )
    left = projector.lexical(
        DerivationNode(TokenSpan(0, 1), "N"),
        Token(0, "cats"),
    )
    coordinator = projector.lexical(
        DerivationNode(TokenSpan(1, 2), "conj"),
        Token(1, "and"),
    )
    right = projector.lexical(
        DerivationNode(TokenSpan(2, 3), "N"),
        Token(2, "dogs"),
    )
    merge_node = DerivationNode(
        TokenSpan(0, 3),
        "N",
        rule="coordination-merge",
        children=(0, 1),
    )

    with pytest.raises(ValueError, match="requires one marked right child"):
        projector.step(
            merge_node,
            "coordination-merge",
            (left, right),
            HeadSelection("left-conjunct", 0),
        )

    marked = projector.step(
        DerivationNode(
            TokenSpan(1, 3),
            "N[conj]",
            rule="coordination-mark",
            children=(0, 1),
        ),
        "coordination-mark",
        (coordinator, right),
        HeadSelection("non-absorbing", 1),
    )

    with pytest.raises(ValueError, match="requires non-absorbing"):
        projector.step(
            DerivationNode(
                TokenSpan(1, 3),
                "N[conj]",
                rule="coordination-mark",
                children=(0, 1),
            ),
            "coordination-mark",
            (coordinator, right),
            HeadSelection("functor", 1),
        )

    with pytest.raises(ValueError, match="unmerged coordination marker"):
        projector.complete(marked)

    merged = projector.step(
        merge_node,
        "coordination-merge",
        (left, marked),
        HeadSelection("left-conjunct", 0),
    )
    completed = projector.complete(merged)

    with pytest.raises(ValueError, match="already has a root arc"):
        projector.complete(completed)


def test_an_undeclared_nonabsorbing_rule_is_not_a_coordination_mark() -> None:
    fixture = _golden_fixture()
    projector = BiskHockenmaierDependencyProjector(
        fixture.algebra,
        fixture.coordination_marks,
    )
    punctuation = projector.lexical(
        DerivationNode(TokenSpan(0, 1), ","),
        Token(0, ","),
    )
    word = projector.lexical(
        DerivationNode(TokenSpan(1, 2), "N"),
        Token(1, "birds"),
    )
    projected = projector.step(
        DerivationNode(
            TokenSpan(0, 2),
            "N",
            rule="punctuation-coordination-mark",
            children=(0, 1),
        ),
        "punctuation-coordination-mark",
        (punctuation, word),
        HeadSelection("non-absorbing", 1),
    )
    completed = projector.complete(projected)

    assert projected.state is not None
    assert projected.state.coordinator_heads == ()
    assert completed.dependencies == (Dependency(0, 2, "root"),)


def test_induced_projection_values_are_frozen_hashable_and_pickleable() -> None:
    fixture = _golden_fixture()
    projector = BiskHockenmaierDependencyProjector(
        fixture.algebra,
        fixture.coordination_marks,
    )
    state = projector.lexical(
        DerivationNode(TokenSpan(0, 1), "N"),
        Token(0, "birds"),
    ).state

    assert state is not None
    assert copy(state) is state
    assert deepcopy(state) is state
    assert loads(dumps(state)) == state
    assert len({state, loads(dumps(state))}) == 1
    assert copy(projector) is projector
    assert deepcopy(projector) is projector

    with pytest.raises(AttributeError, match="immutable"):
        state.__setattr__("_category", "S")

    with pytest.raises(TypeError, match="not an integer"):
        InducedDependencyState("N", (True,), state.report)

    with pytest.raises(ValueError, match="negative"):
        InducedDependencyState("N", (-1,), state.report)

    with pytest.raises(ValueError, match="has no name"):
        BiskHockenmaierDependencyProjector(fixture.algebra, frozenset({""}))


def test_importing_the_induced_projector_loads_no_corpus_or_judge_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import lofi.data.deps._induced; "
                "print('lofi.data.corpora' in sys.modules); "
                "print('lofi.judge' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.stdout.splitlines() == ["False", "False"]
