from typing import TYPE_CHECKING, Final

import pytest

from lofi.deduce import NativeTermRule
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    TypeRaisingSpec,
)
from lofi.spec._rule_laws import LawGrammar
from lofi.terms import RepresentativeKernel, Term, TermRuleKernel

if TYPE_CHECKING:
    from lofi.judge import Rule

GRAMMAR: Final = LawGrammar("direct")
X: Final = GRAMMAR.atom("x")
Y: Final = GRAMMAR.atom("y")
Z: Final = GRAMMAR.atom("z0")
FUNCTORS, FORWARD = GRAMMAR.functor("forward")
APPLICATION: Final = GRAMMAR.rule(ApplicationSpec(view="fwd"))
COMPOSITION: Final = GRAMMAR.rule(
    CompositionSpec(view="fwd", order=1, direction="steedman")
)
RAISING: Final = GRAMMAR.rule(TypeRaisingSpec(view="fwd", arguments=None))
PATTERN: Final = GRAMMAR.rule(
    PatternRuleSpec(
        name="x-to-y",
        premises=(ConstantPatternSpec(name="x"),),
        conclusion=ConstantPatternSpec(name="y"),
    )
)


def _kernel(rule: Rule[Term]) -> TermRuleKernel:
    assert isinstance(rule, NativeTermRule)
    return rule.lowering


def test_a_pattern_kernel_executes_its_declared_relation() -> None:
    """Apply one pattern relation through its sole native implementation."""
    kernel = _kernel(PATTERN)

    assert tuple(kernel.apply((X,))) == (Y,)
    assert kernel.admits((X,), Y)
    assert not kernel.admits((Y,), Y)


def test_a_composition_kernel_reports_witnesses_order_and_join_keys() -> None:
    """Delegate the checking directions of composition to the native kernel."""
    kernel = _kernel(COMPOSITION)
    principal = FUNCTORS.make(FORWARD, X, Y)
    container = FUNCTORS.make(FORWARD, Y, Z)
    output = FUNCTORS.make(FORWARD, X, Z)

    assert tuple(kernel.apply((principal, container))) == (output,)
    assert kernel.witnesses((principal, container), output) == (
        (0, 1, FORWARD, (FORWARD,)),
    )
    assert kernel.applied_order((principal, container), output) == 1
    assert kernel.applied_order((principal, container), X) is None
    assert set(kernel.join_keys(principal, 0)) & set(kernel.join_keys(container, 1))


def test_a_kernel_residuates_the_one_missing_premise() -> None:
    """Recover a missing premise marked with ``None``."""
    principal = FUNCTORS.make(FORWARD, X, Y)

    assert _kernel(APPLICATION).residuate(X, (principal, None)) == (Y,)
    assert _kernel(APPLICATION).residuate(X, (None, None)) == ()


def test_open_raising_admits_and_residuates_without_enumerating() -> None:
    """Check an open-target raising rule natively."""
    kernel = _kernel(RAISING)
    raised = FUNCTORS.make(FORWARD, X, FUNCTORS.make(FUNCTORS.opposite(FORWARD), X, Y))

    assert tuple(kernel.apply((Y,))) == ()
    assert kernel.admits((Y,), raised)
    assert kernel.residuate(raised, (None,)) == (Y,)


def test_a_kernel_without_a_residual_refuses_residuation() -> None:
    """Refuse a direction the pattern kernel does not declare."""
    with pytest.raises(ValueError, match="residuation"):
        _kernel(PATTERN).residuate(Y, (None,))


def test_representatives_compile_identity_core_and_finite_refinement() -> None:
    """Expose every current native representative function publicly."""
    identity = RepresentativeKernel.identity()
    core = RepresentativeKernel.core()
    refined = identity.refine(((X, Y),))
    product = RepresentativeKernel.product(core, identity)
    nested = RepresentativeKernel.product(product, core)
    subsumption = RepresentativeKernel.subsumption()

    assert identity.key(Y) is Y
    assert refined.key(X) is Y
    assert refined.key(Z) is Z
    assert product.key(X) is X
    assert nested.key(Y) is Y
    assert "subsumption" in repr(subsumption)


def test_a_term_rule_kernel_is_frozen() -> None:
    """Refuse replacement of native kernel state."""
    kernel = _kernel(APPLICATION)

    with pytest.raises(AttributeError, match="immutable"):
        kernel.__setattr__("native", kernel.native)
