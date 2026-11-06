"""Interpretation conformance laws."""

from itertools import product
from typing import TYPE_CHECKING, Final, NamedTuple
from unittest import SkipTest

from hypothesis import given, settings

from lofi._checks import (
    LawViolationError,
    ensure_non_negative,
    fail_law,
)
from lofi.interpret._derive import axiom_entries, is_axiom, read, step_context

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

    from lofi.deduce import Derivation
    from lofi.interpret._ports import Interpretation, Lexicalization
    from lofi.lexicon import Entry

__all__ = [
    "INTERPRETATION_LAWS",
    "InterpretationFixture",
    "check_interpretation_laws",
    "compositionality",
    "declared_partiality",
    "packing_honesty",
]

_MIN_EXAMPLES: Final = 50
"""Minimum generated examples per interpretation law."""

type _Tree[Category, Weight] = Derivation[Category, Weight]
"""A derivation a law reads, named so generated examples resolve lazily."""


class InterpretationFixture[
    Category,
    Value,
    Meaning = Category,
    Weight = float,
](NamedTuple):
    """Inputs for interpretation-law checks."""

    interpretation: Interpretation[Category, Value, Meaning, Weight]
    lexicalization: Lexicalization[Category, Meaning]
    derivations: SearchStrategy[Derivation[Category, Weight]]
    same_item_derivations: SearchStrategy[
        tuple[Derivation[Category, Weight], Derivation[Category, Weight]]
    ]
    max_examples: int = 100


def _budget(max_examples: int) -> settings:
    """Return Hypothesis settings for an interpretation law."""
    checked = ensure_non_negative("max_examples", max_examples)

    return settings(max_examples=max(_MIN_EXAMPLES, checked), deadline=None)


def _nodes[Category, Weight](
    derivation: Derivation[Category, Weight],
) -> tuple[Derivation[Category, Weight], ...]:
    """Return every node of a derivation."""
    found: list[Derivation[Category, Weight]] = []
    pending = [derivation]

    while pending:
        node = pending.pop()
        found.append(node)
        pending.extend(node.children)

    return tuple(found)


def _read[Category, Value, Meaning, Weight](
    derivation: Derivation[Category, Weight],
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
    *,
    law: str,
) -> tuple[Value | None, ...]:
    """Read a derivation and report exceptions as law violations."""
    try:
        return read(derivation, fixture.interpretation, fixture.lexicalization)

    except Exception as exc:
        message = f"{law}: reading {derivation!r} raised {exc!r}"

        raise LawViolationError(message) from exc


def _axiom[Category, Value, Meaning, Weight](
    entry: Entry[Category, Meaning],
    interpretation: Interpretation[Category, Value, Meaning, Weight],
    *,
    law: str,
) -> Value | None:
    """Interpret an axiom and report exceptions as law violations."""
    try:
        return interpretation.axiom(entry)

    except Exception as exc:
        message = f"{law}: interpreting lexical axiom {entry!r} raised {exc!r}"

        raise LawViolationError(message) from exc


def _entries[Category, Value, Meaning, Weight](
    node: Derivation[Category, Weight],
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
    *,
    law: str,
) -> tuple[Entry[Category, Meaning], ...]:
    """Return the entries an axiom stands for, reporting failures as violations."""
    try:
        return axiom_entries(node, fixture.lexicalization)

    except Exception as exc:
        message = f"{law}: resolving lexical axiom {node!r} raised {exc!r}"

        raise LawViolationError(message) from exc


def _children[Category, Value, Meaning, Weight](
    step: Derivation[Category, Weight],
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
    *,
    law: str,
) -> tuple[tuple[Value, ...], ...]:
    """Return each fully defined choice of a step's child values."""
    below = tuple(_read(child, fixture, law=law) for child in step.children)

    return tuple(
        tuple(value for value in choice if value is not None)
        for choice in product(*below)
        if all(value is not None for value in choice)
    )


def _step[Category, Value, Meaning, Weight](
    step: Derivation[Category, Weight],
    children: tuple[Value, ...],
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
    *,
    law: str,
) -> Value | None:
    """Interpret a step and report exceptions as law violations."""
    try:
        context = step_context(step, fixture.lexicalization)
        return fixture.interpretation.step(context, children)

    except Exception as exc:
        message = f"{law}: interpreting {step.edge.rule_name!r} raised {exc!r}"

        raise LawViolationError(message) from exc


def compositionality[Category, Value, Meaning, Weight](
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
) -> None:
    """Check that equal local inputs always receive equal values."""
    interpretation = fixture.interpretation
    exercised: list[str] = []

    @given(fixture.derivations)
    @_budget(fixture.max_examples)
    def law(derivation: _Tree[Category, Weight]) -> None:
        for node in _nodes(derivation):
            if is_axiom(node):
                for entry in _entries(node, fixture, law="compositionality"):
                    first = _axiom(entry, interpretation, law="compositionality")
                    second = _axiom(entry, interpretation, law="compositionality")
                    exercised.append("lexical axiom")

                    if first != second:
                        detail = (
                            f"lexical axiom {entry!r} maps to {first!r} and then "
                            f"{second!r}"
                        )

                        fail_law(detail, law="compositionality")

                continue

            for children in _children(node, fixture, law="compositionality"):
                first = _step(node, children, fixture, law="compositionality")
                second = _step(node, children, fixture, law="compositionality")
                exercised.append(node.edge.rule_name)

                if first != second:
                    detail = (
                        f"{node.edge.rule_name!r} maps {children!r} at order "
                        f"{node.edge.order} to {first!r} and then {second!r}"
                    )

                    fail_law(detail, law="compositionality")

    law()

    if not exercised:
        fail_law("no generated derivation had a local input", law="compositionality")


def declared_partiality[Category, Value, Meaning, Weight](
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
) -> None:
    """Check that support declarations match defined step values."""
    interpretation = fixture.interpretation
    exercised: list[str] = []

    @given(fixture.derivations)
    @_budget(fixture.max_examples)
    def law(derivation: _Tree[Category, Weight]) -> None:
        for node in _nodes(derivation):
            if is_axiom(node):
                continue

            name = node.edge.rule_name

            for children in _children(node, fixture, law="declared partiality"):
                try:
                    supported = interpretation.supports(
                        fixture.lexicalization.rule(name)
                    )

                except Exception as exc:
                    message = (
                        f"declared partiality: checking support for "
                        f"{name!r} raised {exc!r}"
                    )

                    raise LawViolationError(message) from exc

                value = _step(node, children, fixture, law="declared partiality")
                exercised.append(name)

                if supported == (value is not None):
                    continue

                state = "supported" if supported else "unsupported"
                detail = f"{name!r} is {state} but produced {value!r}"

                fail_law(detail, law="declared partiality")

    law()

    if not exercised:
        fail_law(
            "no generated derivation had a readable step", law="declared partiality"
        )


def packing_honesty[Category, Value, Meaning, Weight](
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
) -> None:
    """Check equal values for derivations declared to share one item.

    Raises
    ------
    SkipTest
        If the interpretation refines packed items.
    LawViolationError
        If evaluation fails or packed derivations receive different values.
    """
    interpretation = fixture.interpretation

    if interpretation.refines_items:
        msg = f"{interpretation.name!r} refines packed items"
        raise SkipTest(msg)

    @given(fixture.same_item_derivations)
    @_budget(fixture.max_examples)
    def law(
        pair: tuple[_Tree[Category, Weight], _Tree[Category, Weight]],
    ) -> None:
        left, right = pair
        first = _read(left, fixture, law="packing honesty")
        second = _read(right, fixture, law="packing honesty")

        if first == second:
            return

        detail = f"one packed item receives {first!r} and {second!r}"

        fail_law(detail, law="packing honesty")

    law()


INTERPRETATION_LAWS: Final = (
    compositionality,
    declared_partiality,
    packing_honesty,
)
"""Interpretation laws in conformance order."""


def check_interpretation_laws[Category, Value, Meaning, Weight](
    fixture: InterpretationFixture[Category, Value, Meaning, Weight],
) -> None:
    """Run every interpretation law."""
    for law in INTERPRETATION_LAWS:
        law(fixture)
