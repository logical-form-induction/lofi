"""Tests for installed ill-typing operators."""

from random import Random
from typing import TYPE_CHECKING, Final

from tests.unit.data.synthetic.conftest import flat_generator, scoped_generator

from lofi.data.synthetic._injection import (
    ARGUMENT_SWAP_PLUGIN,
    OVER_APPLICATION_PLUGIN,
    SORT_MISMATCH_PLUGIN,
    IllTypingContext,
    IllTypingOperator,
)
from lofi.data.synthetic.spec.operators import (
    ArgumentSwapSpec,
    OverApplicationSpec,
    SortMismatchSpec,
)
from lofi.terms import positions

if TYPE_CHECKING:
    from tests.unit.data.synthetic.conftest import Generator

    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.terms import Term

    type Interpretation = InterpretationFunction

SCOPED: Final = "∃x. book(x) & ∃y. student(y) & read(y)(x)"
FLAT: Final = "love(john)(mary)"


def _interpretation(generator: Generator) -> Interpretation:
    held = generator.interpretation
    if held is None:
        msg = "the fixture generator declares no interpretation"
        raise AssertionError(msg)
    return held


def _refused(
    interpretation: Interpretation,
    operator: IllTypingOperator,
    text: str,
    *,
    seed: int = 0,
) -> Term:
    candidate = operator.apply(Random(seed), interpretation.read_text(text))
    if candidate is None:
        msg = f"{text!r} does not admit {operator.name!r}"
        raise AssertionError(msg)
    return candidate


def test_every_operator_turns_a_well_typed_form_into_a_refused_one() -> None:
    interpretation = _interpretation(scoped_generator())
    context = IllTypingContext(interpretation)
    source = interpretation.read_text(SCOPED)
    operators = (
        ARGUMENT_SWAP_PLUGIN.construct(ArgumentSwapSpec(), context),
        OVER_APPLICATION_PLUGIN.construct(OverApplicationSpec(), context),
        SORT_MISMATCH_PLUGIN.construct(SortMismatchSpec(), context),
    )

    for operator in operators:
        candidate = _refused(interpretation, operator, SCOPED)

        assert candidate is not source
        assert interpretation.check(source).is_well_typed
        assert not interpretation.check(candidate).is_well_typed
        assert interpretation.failure(candidate) is not None


def test_an_argument_swap_exchanges_the_children_of_one_application() -> None:
    interpretation = _interpretation(flat_generator())
    operator = ARGUMENT_SWAP_PLUGIN.construct(
        ArgumentSwapSpec(), IllTypingContext(interpretation)
    )
    source = interpretation.read_text(FLAT)
    candidate = _refused(interpretation, operator, FLAT)
    swapped = [
        path
        for path in positions(source, lambda node: node.op.name == "eval")
        if _at(candidate, path).args == _at(source, path).args[::-1]
        and _at(candidate, path).args != _at(source, path).args
    ]

    assert len(swapped) == 1
    assert candidate.size == source.size


def test_an_over_application_adds_one_application() -> None:
    interpretation = _interpretation(flat_generator())
    operator = OVER_APPLICATION_PLUGIN.construct(
        OverApplicationSpec(), IllTypingContext(interpretation)
    )
    source = interpretation.read_text(FLAT)
    candidate = _refused(interpretation, operator, FLAT)

    assert len(positions(candidate, lambda node: node.op.name == "eval")) == (
        len(positions(source, lambda node: node.op.name == "eval")) + 1
    )


def test_a_sort_mismatch_uses_declared_binder_structure() -> None:
    interpretation = _interpretation(scoped_generator())
    operator = SORT_MISMATCH_PLUGIN.construct(
        SortMismatchSpec(), IllTypingContext(interpretation)
    )
    source = interpretation.read_text(SCOPED)
    candidate = _refused(interpretation, operator, SCOPED)
    variable = f"#bvar[{interpretation.language.term_sort}]"

    assert len(positions(candidate, lambda node: node.op.name == variable)) > len(
        positions(source, lambda node: node.op.name == variable)
    )
    assert candidate.size <= source.size


def test_operators_decline_forms_without_the_required_structure() -> None:
    interpretation = _interpretation(flat_generator())
    context = IllTypingContext(interpretation)
    constant = interpretation.read_text("john")
    flat = interpretation.read_text(FLAT)

    assert (
        ARGUMENT_SWAP_PLUGIN.construct(ArgumentSwapSpec(), context).apply(
            Random(0), constant
        )
        is None
    )
    assert (
        OVER_APPLICATION_PLUGIN.construct(OverApplicationSpec(), context).apply(
            Random(0), constant
        )
        is None
    )
    assert (
        SORT_MISMATCH_PLUGIN.construct(SortMismatchSpec(), context).apply(
            Random(0), flat
        )
        is None
    )


def _at(term: Term, path: tuple[int, ...]) -> Term:
    held = term
    for step in path:
        held = held.args[step]
    return held
