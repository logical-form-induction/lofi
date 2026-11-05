from typing import TYPE_CHECKING, Final, override
from unittest import SkipTest

import pytest
from hypothesis import strategies as st
from tests.conformance.interpretation import InterpretationSuite
from tests.support.derivations import FormLexicon, axiom, step
from tests.support.rules import declared_rules

from lofi._checks import LawViolationError
from lofi.interpret import Interpretation, StepContext
from lofi.interpret.laws import (
    InterpretationFixture,
    check_interpretation_laws,
    packing_honesty,
)
from lofi.lexicon import Entry
from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import SignatureSpec

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

    from lofi.deduce import Derivation
    from lofi.judge import Rule
    from lofi.terms import Term


_X: Final = TermPatternSpec(name="X", sort="Cat")
_Y: Final = TermPatternSpec(name="Y", sort="Cat")
RULES: Final = declared_rules(
    SignatureSpec(
        name="interpretation-laws", primitives=("left", "right"), syntax="s-expression"
    ),
    RuleSetSpec(
        name="interpretation-laws",
        rules=(
            PatternRuleSpec(
                name="pack",
                premises=(_X, _Y),
                conclusion=ConstantPatternSpec(name="right"),
            ),
            PatternRuleSpec(name="gap", premises=(_X,), conclusion=_X),
        ),
    ),
)
STORE: Final = RULES.context.store
LEFT: Final = STORE.const("Cat", "left")
RIGHT: Final = STORE.const("Cat", "right")
PACK, GAP = RULES.rule_set.rules

type _Tree = Derivation[Term, float]
"""A generated fixture derivation."""

ENTRIES: Final = (Entry("x", LEFT), Entry("y", RIGHT), Entry("z", LEFT))
LEXICON: Final = FormLexicon((PACK, GAP), ENTRIES)
X: Final = axiom(ENTRIES[0])
Y: Final = axiom(ENTRIES[1], 1)
Z: Final = axiom(ENTRIES[2], 2)


def _extend(
    children: st.SearchStrategy[_Tree],
) -> st.SearchStrategy[_Tree]:
    """Return recursive derivations over the two fixture rules."""
    unary = st.builds(
        lambda child, weight: step(GAP, (child,), LEFT, weight),
        children,
        st.floats(allow_nan=False, allow_infinity=False),
    )
    binary = st.builds(
        lambda left, right, weight, order: step(
            PACK,
            (left, right),
            RIGHT,
            weight,
            order,
        ),
        children,
        children,
        st.floats(allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=3),
    )

    return st.one_of(unary, binary)


DERIVATIONS: Final = st.recursive(
    st.sampled_from((X, Y, Z)),
    _extend,
    max_leaves=8,
)
SAME_ITEM: Final = st.builds(
    lambda first, second, root: (
        step(PACK, (step(PACK, (X, Y), RIGHT, 0.0, first), Z), RIGHT, 0.0, root),
        step(PACK, (X, step(PACK, (Y, Z), RIGHT, 0.0, second)), RIGHT, 0.0, root),
    ),
    st.integers(min_value=0, max_value=9),
    st.integers(min_value=0, max_value=9),
    st.integers(min_value=0, max_value=9),
)
SAME_ITEM_SHAPES: Final = st.just((Y, step(PACK, (X, Y), RIGHT, 0.0)))


class _Count:
    """Counts defined lexical leaves."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "count"

    @property
    def refines_items(self) -> bool:
        """Return whether counts refine packed items."""
        return False

    def supports(self, rule: Rule[Term], /) -> bool:
        """Return whether the fixture gives the rule a value."""
        return rule.name != "gap"

    def axiom(self, entry: Entry[Term], /) -> int:
        """Count one lexical leaf."""
        del entry

        return 1

    def step(
        self,
        context: StepContext[Term],
        children: tuple[int, ...],
        /,
    ) -> int | None:
        """Return one value for every defined rule."""
        del children

        return None if context.rule.name == "gap" else 1


class _RefiningCount(_Count):
    """Count interpretation declared to refine packed items."""

    __slots__ = ()

    @property
    @override
    def refines_items(self) -> bool:
        """Return whether counts refine packed items."""
        return True

    @override
    def step(
        self,
        context: StepContext[Term],
        children: tuple[int, ...],
        /,
    ) -> int | None:
        """Add child counts for a defined rule."""
        return None if context.rule.name == "gap" else sum(children)


_MODULE_STATE = [0]
_AXIOM_STATE = [0]


class _ModuleState(_Count):
    """Interpretation whose result depends on module state."""

    __slots__ = ()

    @override
    def step(
        self,
        context: StepContext[Term],
        children: tuple[int, ...],
        /,
    ) -> int | None:
        """Read and change module state when interpreting a step."""
        if context.rule.name == "gap":
            return None

        _MODULE_STATE[0] += 1

        return sum(children) + _MODULE_STATE[0]


class _AxiomState(_Count):
    """Interpretation whose lexical value depends on module state."""

    __slots__ = ()

    @override
    def axiom(self, entry: Entry[Term], /) -> int:
        """Read and change module state for a lexical axiom."""
        del entry
        _AXIOM_STATE[0] += 1

        return _AXIOM_STATE[0]


class _Missing(_RefiningCount):
    """Interpretation missing a value it declares supported."""

    __slots__ = ()

    @override
    def supports(self, rule: Rule[Term], /) -> bool:
        """Declare every fixture rule supported."""
        del rule

        return True

    @override
    def step(
        self,
        context: StepContext[Term],
        children: tuple[int, ...],
        /,
    ) -> None:
        """Return no value despite the support declaration."""
        del context, children


def _fixture(
    interpretation: Interpretation[Term, int],
    *,
    pairs: SearchStrategy[
        tuple[Derivation[Term, float], Derivation[Term, float]]
    ] = SAME_ITEM,
) -> InterpretationFixture[Term, int]:
    """Return the shared law fixture for an interpretation."""
    return InterpretationFixture(interpretation, LEXICON, DERIVATIONS, pairs)


def test_a_complete_stub_satisfies_the_interpretation_protocol() -> None:
    interpretation: Interpretation[Term, int] = _Count()

    assert isinstance(interpretation, Interpretation)
    assert interpretation.name == "count"


def test_the_laws_accept_a_total_nonrefining_interpretation() -> None:
    check_interpretation_laws(_fixture(_Count()))


def test_the_laws_report_refining_packing_checks_as_skipped() -> None:
    with pytest.raises(SkipTest, match="refines packed items"):
        check_interpretation_laws(_fixture(_RefiningCount(), pairs=SAME_ITEM_SHAPES))


def test_compositionality_rejects_an_interpretation_that_reads_module_state() -> None:
    _MODULE_STATE[0] = 0

    with pytest.raises(LawViolationError, match="compositionality"):
        check_interpretation_laws(_fixture(_ModuleState()))


def test_compositionality_rejects_a_stateful_axiom_interpretation() -> None:
    _AXIOM_STATE[0] = 0

    with pytest.raises(
        LawViolationError,
        match=r"compositionality.*lexical axiom",
    ):
        check_interpretation_laws(_fixture(_AxiomState()))


def test_partiality_rejects_a_supported_rule_with_no_value() -> None:
    with pytest.raises(LawViolationError, match="supported but produced None"):
        check_interpretation_laws(_fixture(_Missing()))


def test_packing_honesty_rejects_two_values_for_one_declared_item() -> None:
    _MODULE_STATE[0] = 0

    with pytest.raises(LawViolationError, match="one packed item receives"):
        packing_honesty(_fixture(_ModuleState(), pairs=SAME_ITEM_SHAPES))


@pytest.fixture
def interpretation_conformance() -> InterpretationFixture[Term, int]:
    """Return the complete interpretation fixture."""
    return _fixture(_Count())


class TestCountInterpretation(InterpretationSuite):
    """Interpretation contract for the complete fixture."""
