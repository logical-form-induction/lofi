import ast
import subprocess
import sys
from copy import copy, deepcopy
from pathlib import Path
from pickle import dumps, loads
from typing import TYPE_CHECKING, Final, NamedTuple, Never, override

import pytest

from lofi.data import Dependency, Derivation, DerivationNode, Token, TokenSpan
from lofi.data.deps._projection import (
    AmbiguousProjectionStepError,
    DependencyProjector,
    HeadProjection,
    UnrecoverableProjectionStepError,
    project_dependencies,
)
from lofi.spec import HeadSelection

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence


class RecordingProjector:
    """A recording dependency projector with declared head selection."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def lexical(
        self,
        node: DerivationNode[str],
        token: Token,
        /,
    ) -> HeadProjection[Never, Never]:
        self.events.append(("lexical", node.label))
        return HeadProjection((token.index,))

    def step(
        self,
        node: DerivationNode[str],
        rule_name: str,
        children: tuple[HeadProjection[Never, Never], ...],
        selection: HeadSelection,
        /,
    ) -> HeadProjection[Never, Never]:
        self.events.append(("step", rule_name))
        dependencies = tuple(
            dependency for child in children for dependency in child.dependencies
        )

        if selection.slot is None:
            return HeadProjection((node.span.start,), dependencies)

        return HeadProjection(children[selection.slot].heads, dependencies)


class ProjectorCase[L: Hashable, D, S](NamedTuple):
    """A reusable dependency-projector conformance case."""

    projector: DependencyProjector[L, D, S]
    leaf: DerivationNode[L]
    token: Token
    step: DerivationNode[L]
    rule_name: str
    selection: HeadSelection
    expected_heads: tuple[int, ...]


class DependencyProjectorConformanceTests[L: Hashable, D, S]:
    """Dependency-projector conformance tests."""

    def make_projector_case(self) -> ProjectorCase[L, D, S]:
        raise NotImplementedError

    def test_the_projector_implements_both_fold_operations(self) -> None:
        case = self.make_projector_case()
        lexical = case.projector.lexical(case.leaf, case.token)
        projected = case.projector.step(
            case.step,
            case.rule_name,
            (lexical,),
            case.selection,
        )

        assert isinstance(case.projector, DependencyProjector)
        assert lexical.heads == (case.token.index,)
        assert projected.heads == case.expected_heads


class TestRecordingDependencyProjector(
    DependencyProjectorConformanceTests[str, Never, Never]
):
    @override
    def make_projector_case(self) -> ProjectorCase[str, Never, Never]:
        return ProjectorCase(
            RecordingProjector(),
            DerivationNode(TokenSpan(0, 1), "N"),
            Token(0, "word"),
            DerivationNode(
                TokenSpan(0, 1),
                "NP",
                rule="unary",
                children=(0,),
            ),
            "unary",
            HeadSelection("pass-through", 0),
            (0,),
        )


class FixedRecovery:
    """A recovery relation with fixed rule names."""

    __slots__ = ("_names", "calls")

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def recover(
        self,
        inputs: Sequence[str],
        output: str,
        /,
    ) -> tuple[str, ...]:
        self.calls.append((tuple(inputs), output))
        return self._names


DERIVATION: Final = Derivation(
    10,
    {
        10: DerivationNode(
            TokenSpan(0, 3),
            "ROOT",
            rule="ternary",
            children=(7, 2, 9),
        ),
        7: DerivationNode(TokenSpan(0, 1), "A"),
        2: DerivationNode(TokenSpan(1, 2), "B", rule="nullary"),
        9: DerivationNode(
            TokenSpan(2, 3),
            "C'",
            rule="unary",
            children=(8,),
        ),
        8: DerivationNode(TokenSpan(2, 3), "C"),
    },
)

SELECTIONS: Final[Mapping[str, HeadSelection]] = {
    "nullary": HeadSelection("pass-through", None),
    "unary": HeadSelection("pass-through", 0),
    "ternary": HeadSelection("functor", 2),
}
TOKENS: Final = (Token(0, "a"), Token(1, "b"), Token(2, "c"))


def test_the_fold_visits_children_in_declared_order_and_uses_exact_slots() -> None:
    projector = RecordingProjector()

    result = project_dependencies(DERIVATION, TOKENS, SELECTIONS, projector)

    assert result.heads == (2,)
    assert projector.events == [
        ("lexical", "A"),
        ("step", "nullary"),
        ("lexical", "C"),
        ("step", "unary"),
        ("step", "ternary"),
    ]


def test_the_fold_refuses_a_missing_selection_without_a_default() -> None:
    selections = {name: value for name, value in SELECTIONS.items() if name != "unary"}

    with pytest.raises(KeyError, match=r"unary.*no head selection"):
        project_dependencies(DERIVATION, TOKENS, selections, RecordingProjector())


def test_the_fold_refuses_a_nonlexical_node_without_a_rule_name() -> None:
    derivation = Derivation(
        1,
        {
            1: DerivationNode(TokenSpan(0, 1), "NP", children=(0,)),
            0: DerivationNode(TokenSpan(0, 1), "N"),
        },
    )

    with pytest.raises(UnrecoverableProjectionStepError, match=r"no step recovery"):
        project_dependencies(
            derivation,
            (Token(0, "n"),),
            {},
            RecordingProjector(),
        )


def test_an_unnamed_step_recovers_by_labels_and_auto_head_child() -> None:
    derivation = Derivation(
        2,
        {
            2: DerivationNode(
                TokenSpan(0, 2),
                "C",
                children=(0, 1),
                features={"head": "1"},
            ),
            0: DerivationNode(TokenSpan(0, 1), "A"),
            1: DerivationNode(TokenSpan(1, 2), "B"),
        },
    )
    recovery = FixedRecovery(("left", "right"))
    projector = RecordingProjector()

    result = project_dependencies(
        derivation,
        (Token(0, "a"), Token(1, "b")),
        {
            "left": HeadSelection("functor", 0),
            "right": HeadSelection("functor", 1),
        },
        projector,
        recovery=recovery,
    )

    assert result.heads == (1,)
    assert recovery.calls == [(("A", "B"), "C")]
    assert projector.events[-1] == ("step", "right")


def test_an_explicit_rule_name_bypasses_recovery() -> None:
    recovery = FixedRecovery(("other",))

    result = project_dependencies(
        DERIVATION,
        TOKENS,
        SELECTIONS,
        RecordingProjector(),
        recovery=recovery,
    )

    assert result.heads == (2,)
    assert recovery.calls == []


def test_an_unnamed_step_distinguishes_zero_and_ambiguous_recovery() -> None:
    derivation = Derivation(
        2,
        {
            2: DerivationNode(
                TokenSpan(0, 2),
                "C",
                children=(0, 1),
                features={"head": "0"},
            ),
            0: DerivationNode(TokenSpan(0, 1), "A"),
            1: DerivationNode(TokenSpan(1, 2), "B"),
        },
    )
    tokens = (Token(0, "a"), Token(1, "b"))

    with pytest.raises(UnrecoverableProjectionStepError, match="no recovered rule"):
        project_dependencies(
            derivation,
            tokens,
            {"right": HeadSelection("functor", 1)},
            RecordingProjector(),
            recovery=FixedRecovery(("right",)),
        )

    with pytest.raises(AmbiguousProjectionStepError, match="ambiguous recovered"):
        project_dependencies(
            derivation,
            tokens,
            {
                "first": HeadSelection("functor", 0),
                "second": HeadSelection("functor", 0),
            },
            RecordingProjector(),
            recovery=FixedRecovery(("first", "second")),
        )


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ({}, "no AUTO head"),
        ({"head": "left"}, "noninteger AUTO head"),
        ({"head": "-1"}, "head child -1"),
        ({"head": "2"}, "head child 2"),
    ],
)
def test_an_unnamed_step_validates_its_auto_head_child(
    features: Mapping[str, str],
    message: str,
) -> None:
    derivation = Derivation(
        2,
        {
            2: DerivationNode(
                TokenSpan(0, 2),
                "C",
                children=(0, 1),
                features=features,
            ),
            0: DerivationNode(TokenSpan(0, 1), "A"),
            1: DerivationNode(TokenSpan(1, 2), "B"),
        },
    )

    with pytest.raises(ValueError, match=message):
        project_dependencies(
            derivation,
            (Token(0, "a"), Token(1, "b")),
            {"rule": HeadSelection("functor", 0)},
            RecordingProjector(),
            recovery=FixedRecovery(("rule",)),
        )


@pytest.mark.parametrize(
    ("selection", "child_count", "message"),
    [
        (HeadSelection("pass-through", 0), 0, "nullary.*slot 0"),
        (HeadSelection("pass-through", None), 1, "positive-arity.*no head slot"),
        (HeadSelection("pass-through", 1), 1, "slot 1.*arity 1"),
    ],
)
def test_the_fold_rechecks_a_selection_against_the_observed_arity(
    selection: HeadSelection,
    child_count: int,
    message: str,
) -> None:
    derivation = (
        Derivation(
            0,
            {0: DerivationNode(TokenSpan(0, 1), "N", rule="subject")},
        )
        if child_count == 0
        else Derivation(
            1,
            {
                1: DerivationNode(
                    TokenSpan(0, 1),
                    "NP",
                    rule="subject",
                    children=(0,),
                ),
                0: DerivationNode(TokenSpan(0, 1), "N"),
            },
        )
    )

    with pytest.raises(ValueError, match=message):
        project_dependencies(
            derivation,
            (Token(0, "n"),),
            {"subject": selection},
            RecordingProjector(),
        )


def test_a_head_projection_is_a_frozen_structural_value() -> None:
    dependency = Dependency(2, 1, "argument", 1)
    projection = HeadProjection((2,), (dependency,))

    assert projection.heads == (2,)
    assert projection.dependencies == (dependency,)
    assert projection == HeadProjection((2,), (dependency,))
    assert len({projection, HeadProjection((2,), (dependency,))}) == 1
    assert repr(projection) == f"HeadProjection((2,), ({dependency!r},), None)"
    assert copy(projection) is projection
    assert deepcopy(projection) is projection
    assert loads(dumps(projection)) == projection

    with pytest.raises(AttributeError, match="immutable"):
        projection.__setattr__("_heads", (0,))


@pytest.mark.parametrize(
    ("heads", "error", "message"),
    [
        ((True,), TypeError, "not an integer"),
        ((-1,), ValueError, "negative"),
    ],
)
def test_a_head_projection_rejects_invalid_head_positions(
    heads: tuple[int, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        HeadProjection(heads)


def test_the_fold_imports_no_judge_module_or_concrete_rule() -> None:
    path = Path("lofi/data/deps/_projection.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    fold_nodes = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_selection_for", "project_dependencies"}
    )

    assert all(not name.startswith("lofi.judge") for name in imports)
    assert not any(
        isinstance(member, ast.Call)
        and isinstance(member.func, ast.Name)
        and member.func.id == "isinstance"
        for function in fold_nodes
        for member in ast.walk(function)
    )


def test_importing_the_fold_does_not_load_the_rule_engine() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "from lofi.data.deps import project_dependencies; "
                "assert callable(project_dependencies); "
                "assert not any(name == 'lofi.judge' or "
                "name.startswith('lofi.judge.') for name in sys.modules)"
            ),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
