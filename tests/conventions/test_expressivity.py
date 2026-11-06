"""No composition order is bounded by the object, only ever by a run that says so.

The category half of that convention is carried by ``ClosureReport.category_set``,
which reads an exhausted closure as ``unknown`` rather than as ``finite``. The order
half is carried here, in two parts. The first is the readout clause: a readout that
chooses among a closed set of composition orders makes the learned calculus finite
however unbounded the symbolic rule is, so the scan below reads every module under
``lofi/nn`` and refuses an enum, a literal, a collection or a name that stands for such
a set. The second is the run clause: a search that stopped at a bound says so, so that
an empty parse under a rule admitting every finite order stays distinguishable from a
sentence the grammar cannot build.
"""

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanFree
from lofi.deduce import (
    LogInside,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    TopKPruner,
    parse,
)
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec import ApplicationSpec, CompositionSpec, RuleSetSpec
from lofi.spec.grammar import (
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    OperatorSpec,
)
from lofi.terms import (
    TargetSet,
    Term,
)

if TYPE_CHECKING:
    from lofi.deduce import Cell, PackedForest
    from lofi.judge import Rule

TREE: Final = Path(__file__).resolve().parents[2]

READOUTS: Final = TREE / "lofi" / "nn"
"""The package whose output spaces the readout clause covers."""

ENUM_BASES: Final = frozenset({"Enum", "Flag", "IntEnum", "IntFlag", "StrEnum"})
"""Bases that make a class body a closed inventory."""

BUILDERS: Final = frozenset({"frozenset", "list", "set", "tuple"})
"""Calls that wrap a display without changing what it enumerates."""

ORDER_NAME: Final = re.compile(
    r"(?:^|_)(?:orders|max_order|min_order|n_orders|num_orders|order_vocab"
    r"|order_logits|order_head|order_axis|composition_order|composition_orders"
    r"|composition_degree|combinator_order|combinator_orders)(?:$|_)",
    re.IGNORECASE,
)
"""Names standing for a set of composition orders rather than for a sequence.

A bare ``order`` is excluded on purpose. In this package it is overwhelmingly the
sorting sense, as in a permutation returned by ``argsort`` or the position of a column,
and flagging it would make the clause unreadable rather than stronger.
"""

B_NOTATION: Final = re.compile(r"^B\d+$", re.IGNORECASE)
"""Steedman's name for composition at one order, as an enum member or a literal."""

MINIMUM_MODULES: Final = 80
"""How many readout modules the scan reaches before its silence means anything."""

MINIMUM_INVENTORIES: Final = 250
"""How many closed inventories the scan collects before its silence means anything."""


class Inventory(NamedTuple):
    """One closed set of names a module declares.

    Attributes
    ----------
    module : str
        The module's path relative to the tree.
    line : int
        The line the inventory is declared on.
    kind : str
        Whether the inventory is an enum, a literal, or a collection display.
    name : str
        The name the inventory is bound to, or ``Literal`` for an annotation.
    members : tuple[str, ...]
        The members, rendered as text.
    """

    module: str
    line: int
    kind: str
    name: str
    members: tuple[str, ...]

    def reads_as_orders(self) -> bool:
        """Return whether this inventory enumerates composition orders.

        Returns
        -------
        bool
            Whether the name says the members are orders, or the members are
            written in Steedman's B notation.
        """
        if ORDER_NAME.search(self.name):
            return True

        return sum(bool(B_NOTATION.match(member)) for member in self.members) >= 2


def _display(node: ast.expr) -> tuple[str, ...]:
    """Return the constant members of a collection display.

    Parameters
    ----------
    node : ast.expr
        The expression a name is bound to.

    Returns
    -------
    tuple[str, ...]
        The constants the display holds, empty where the expression enumerates
        nothing.
    """
    if isinstance(node, ast.Call):
        named = isinstance(node.func, ast.Name) and node.func.id in BUILDERS

        return _display(node.args[0]) if named and len(node.args) == 1 else ()

    if not isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return ()

    return tuple(
        str(element.value) for element in node.elts if isinstance(element, ast.Constant)
    )


def _targets(node: ast.Assign | ast.AnnAssign) -> tuple[ast.expr, ...]:
    """Return the expressions an assignment binds.

    Parameters
    ----------
    node : ast.Assign | ast.AnnAssign
        The assignment to read.

    Returns
    -------
    tuple[ast.expr, ...]
        The assignment's targets.
    """
    return (node.target,) if isinstance(node, ast.AnnAssign) else tuple(node.targets)


def _enum_members(body: list[ast.stmt]) -> tuple[str, ...]:
    """Return the names a class body assigns, which are an enum's members.

    Parameters
    ----------
    body : list[ast.stmt]
        The statements of the class body.

    Returns
    -------
    tuple[str, ...]
        The assigned names, in declaration order.
    """
    return tuple(
        target.id
        for statement in body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        for target in _targets(statement)
        if isinstance(target, ast.Name)
    )


def _literal_members(node: ast.Subscript) -> tuple[str, ...]:
    """Return the values a ``Literal`` annotation admits.

    Parameters
    ----------
    node : ast.Subscript
        The ``Literal[...]`` subscript.

    Returns
    -------
    tuple[str, ...]
        The admitted values, rendered as text.
    """
    held = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)

    return tuple(
        str(element.value) for element in held if isinstance(element, ast.Constant)
    )


def inventories(source: str, module: str) -> tuple[Inventory, ...]:
    """Return every closed inventory one module declares.

    Parameters
    ----------
    source : str
        The module's text.
    module : str
        The name the inventories are reported under.

    Returns
    -------
    tuple[Inventory, ...]
        The enums, literal annotations and constant collection displays, in the
        order the walk reaches them.
    """
    found: list[Inventory] = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id in ENUM_BASES for base in node.bases
        ):
            members = _enum_members(node.body)
            found.append(Inventory(module, node.lineno, "enum", node.name, members))

        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "Literal"
        ):
            admitted = _literal_members(node)
            found.append(Inventory(module, node.lineno, "literal", "Literal", admitted))

        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            held = _display(node.value)

            found.extend(
                Inventory(module, node.lineno, "collection", target.id, held)
                for target in _targets(node)
                if held and isinstance(target, ast.Name)
            )

    return tuple(found)


def bound_names(source: str, module: str) -> tuple[tuple[str, int, str], ...]:
    """Return every name one module binds, with where it binds it.

    A binding is an assignment target, an attribute assigned to, a parameter, or the
    name of a function or class. Reading a name is not binding it, so a module that
    passes an order along is not caught here and a module that holds one is.

    Parameters
    ----------
    source : str
        The module's text.
    module : str
        The name the bindings are reported under.

    Returns
    -------
    tuple[tuple[str, int, str], ...]
        The module, the line, and the bound name, in the order the walk reaches
        them.
    """
    found: list[tuple[str, int, str]] = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.append((module, node.lineno, node.id))

        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            found.append((module, node.lineno, node.attr))

        if isinstance(node, ast.arg):
            found.append((module, node.lineno, node.arg))

        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            found.append((module, node.lineno, node.name))

    return tuple(found)


def readout_modules() -> tuple[Path, ...]:
    """Return the readout modules the clause covers, in path order.

    Returns
    -------
    tuple[Path, ...]
        Every Python file below the neural package, caches excluded.
    """
    return tuple(
        found
        for found in sorted(READOUTS.rglob("*.py"))
        if "__pycache__" not in found.parts
    )


def _scanned() -> tuple[tuple[Inventory, ...], tuple[tuple[str, int, str], ...]]:
    """Return the inventories and the bound names of every readout module.

    Returns
    -------
    tuple[tuple[Inventory, ...], tuple[tuple[str, int, str], ...]]
        The inventories, then the bindings.
    """
    held: list[Inventory] = []
    names: list[tuple[str, int, str]] = []

    for module in readout_modules():
        source = module.read_text(encoding="utf-8")
        relative = str(module.relative_to(TREE))
        held.extend(inventories(source, relative))
        names.extend(bound_names(source, relative))

    return tuple(held), tuple(names)


SCANNED_INVENTORIES, SCANNED_NAMES = _scanned()

FORBIDDEN: Final = '''
"""A readout whose output space is a closed set of composition orders."""

from enum import Enum
from typing import Literal

import torch
from torch import nn

COMPOSITION_ORDERS = ("B0", "B1", "B2")
SPINE_ORDERS = (0, 1, 2, 3)


class CompositionOrder(Enum):
    B0 = 0
    B1 = 1
    B2 = 2


type Degree = Literal["B0", "B1", "B2"]


class OrderHead(nn.Module):
    def __init__(self, width: int, *, max_order: int = 2) -> None:
        super().__init__()
        self._score = nn.Linear(width, max_order + 1)

    def weigh(self, cells: torch.Tensor) -> torch.Tensor:
        return self._score(cells).log_softmax(dim=-1)
'''
"""A readout of the kind the clause refuses, which keeps the scan honest.

Nothing imports it. It is the input the detectors run over below, so that a scan
returning nothing reads as a tree holding no such readout rather than as a detector
that stopped working.
"""

SORTING: Final = """
order = weights.argsort(descending=True)


def rank(order: int) -> int:
    return order
"""
"""The sorting sense of the word, which the clause leaves alone."""


def test_the_scan_reaches_the_readouts_it_claims_to_cover() -> None:
    kinds = {found.kind for found in SCANNED_INVENTORIES}

    assert len(readout_modules()) >= MINIMUM_MODULES
    assert len(SCANNED_INVENTORIES) >= MINIMUM_INVENTORIES
    assert len(SCANNED_NAMES) >= len(SCANNED_INVENTORIES)
    assert kinds >= {"collection", "enum"}


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("collection", "COMPOSITION_ORDERS"),
        ("collection", "SPINE_ORDERS"),
        ("enum", "CompositionOrder"),
        ("literal", "Literal"),
    ],
)
def test_the_detector_reads_a_closed_order_inventory_in_each_form(
    kind: str, name: str
) -> None:
    found = [
        held
        for held in inventories(FORBIDDEN, "forbidden.py")
        if held.kind == kind and held.name == name
    ]

    assert found
    assert all(held.reads_as_orders() for held in found)


def test_the_detector_reads_an_order_ceiling_a_readout_holds() -> None:
    held = {
        name
        for _, _, name in bound_names(FORBIDDEN, "forbidden.py")
        if ORDER_NAME.search(name)
    }

    assert sorted(held) == ["COMPOSITION_ORDERS", "SPINE_ORDERS", "max_order"]


def test_the_detector_leaves_the_sorting_sense_of_order_alone() -> None:
    held = [name for _, _, name in bound_names(SORTING, "sorting.py")]

    assert "order" in held
    assert [name for name in held if ORDER_NAME.search(name)] == []


def test_no_readout_output_space_is_a_closed_inventory_of_composition_orders() -> None:
    closed = [found for found in SCANNED_INVENTORIES if found.reads_as_orders()]

    assert closed == []


def test_no_readout_binds_a_name_standing_for_a_composition_order_ceiling() -> None:
    ceilings = [
        (module, line, name)
        for module, line, name in SCANNED_NAMES
        if ORDER_NAME.search(name)
    ]

    assert ceilings == []


FORWARD_COMPOSITION: Final = CompositionSpec(
    direction="unrestricted",
    order=None,
    view="/",
)
BACKWARD_COMPOSITION: Final = CompositionSpec(
    direction="unrestricted",
    order=None,
    view="\\",
)
ADOPTED: Final = declared_rules(
    ccg_signature(
        "N",
        "S",
        "conj",
        operators=(OperatorSpec(name="N[conj]", sort="Cat", erases_to="N"),),
        conjunctions=ConjunctionsSpec(
            markers=("conj",),
            pairs=(ConjunctionPairSpec(unmarked="N", marked="N[conj]"),),
        ),
        name="expressivity-categories",
    ),
    RuleSetSpec(
        name="expressivity-rules",
        rules=(
            ApplicationSpec(view="/"),
            ApplicationSpec(view="\\"),
            FORWARD_COMPOSITION,
            BACKWARD_COMPOSITION,
        ),
    ),
    matching=FeatureErasureMatchingSpec(),
)
STORE: Final = ADOPTED.context.store
SYNTAX: Final = ADOPTED.context.syntax


def category(text: str) -> Term:
    """Return the category a CCG source string denotes.

    Parameters
    ----------
    text : str
        The category in CCG surface syntax.

    Returns
    -------
    Term
        The interned category term.
    """
    return SYNTAX.parse(STORE, text)


RULES: Final = ADOPTED.rule_set
"""A rule set with composition over every finite order."""
LEXICON: Final = StaticLexicon(
    (
        Entry("takes", category("S/N")),
        Entry("of", category("((N/N)/N)/N")),
        Entry("thing", category("N")),
        Entry("thing", category("S"), weight=1.0),
    )
)
GOAL: Final = RootGoal(TargetSet((category("S"),), RULES.matching))
TOKENS: Final = ("takes", "of", "thing", "thing", "thing")


class FlatScorer:
    """Weighs every rule application alike, so only the lexicon ranks a cell."""

    __slots__ = ()
    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        """Return the entry's own weight.

        Parameters
        ----------
        position : int
            Where the token stands.
        entry : Entry[Term]
            The lexical entry.

        Returns
        -------
        float
            The entry's log weight.
        """
        del position

        return entry.weight

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the same weight for every rule application.

        Parameters
        ----------
        rule : Rule[Term]
            The rule applied.
        inputs : tuple[Term, ...]
            The premise categories.
        output : Term
            The conclusion.
        span : Cell
            The span the conclusion covers.
        split : int | None
            The split point, for a binary step.

        Returns
        -------
        float
            Zero, in the log-inside algebra.
        """
        del rule, inputs, output, span, split

        return 0.0


def _parsed(beam: int | None) -> PackedForest[Term, float]:
    """Parse the fixture sentence under an optional cell beam.

    Parameters
    ----------
    beam : int | None
        How many items a cell may keep, or None to keep every item.

    Returns
    -------
    PackedForest[Term, float]
        The forest the sweep built.
    """
    return parse(
        TOKENS,
        rules=RULES,
        source=LEXICON,
        semiring=LogInside(),
        scorer=FlatScorer(),
        goal=GOAL,
        unary_depth=0,
        pruner=None if beam is None else TopKPruner(beam),
        prune_lexical=beam is not None,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def _roots(forest: PackedForest[Term, float]) -> tuple[Term, ...]:
    """Return the goal categories a forest reached.

    Parameters
    ----------
    forest : PackedForest[Term, float]
        The forest to read.

    Returns
    -------
    tuple[Term, ...]
        One category per accepted root.
    """
    return tuple(forest.term(root) for root in GOAL.roots(forest))


def test_the_grammar_declaring_the_schema_bounds_no_order_of_its_own() -> None:
    schema = [
        rule for rule in RULES if rule.name == "composition:/:any:harmonic:unrestricted"
    ]

    assert FORWARD_COMPOSITION.order is None
    assert BACKWARD_COMPOSITION.order is None
    assert len(schema) == 1
    assert schema[0].to_spec()["order"] is None


def test_an_unbounded_run_reaches_the_goal_and_reports_no_truncation() -> None:
    forest = _parsed(None)

    assert _roots(forest) == (category("S"),)
    assert not forest.truncated


def test_a_beam_that_empties_the_goal_reports_the_truncation_it_caused() -> None:
    forest = _parsed(1)

    assert _roots(forest) == ()
    assert forest.truncated


def test_a_beam_wide_enough_for_the_cell_reaches_the_goal_and_reports_nothing() -> None:
    forest = _parsed(2)

    assert _roots(forest) == (category("S"),)
    assert not forest.truncated
