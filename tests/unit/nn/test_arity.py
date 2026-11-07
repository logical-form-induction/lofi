from copy import copy, deepcopy
from math import sqrt
from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance.arity_scheme import (
    ARITY_SCHEME_LAWS,
    AritySchemeConformance,
    AritySchemeSuite,
)
from tests.support.rules import declared_rules

from lofi._checks import LawViolationError
from lofi.deduce import RootGoal
from lofi.formalism import (
    Budget,
    Formalism,
    FormalismDeclaration,
    Grammar,
    TheoryArtifact,
)
from lofi.judge import ClosureReport, capacity_report
from lofi.nn._values import Encoded
from lofi.nn.arity import (
    SHIPPED_RANKS,
    ArityBound,
    ArityHeadScheme,
    ArityUnsupportedError,
    RoleFillerScheme,
    reached_operators,
)
from lofi.nn.ports import ArityScheme
from lofi.spec import (
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import FunctorSpec, SignatureSpec
from lofi.terms import (
    Operator,
    Signature,
    TargetSet,
    Term,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.spec import RuleSpec


ATOM: Final = Operator("a", "T")
UNARY: Final = Operator("raise", "T", arg_sorts=("T",))
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"))
TERNARY: Final = Operator("triple", "T", arg_sorts=("T", "T", "T"))

SIGNATURE: Final = Signature(
    "arity-test",
    ("T",),
    (ATOM, UNARY, BINARY, TERNARY),
).with_holes()
UNARY_SIGNATURE: Final = Signature(
    "unary-arity-test",
    ("T",),
    (ATOM, UNARY),
).with_holes()
DERIVED_SIGNATURE: Final = Signature(
    "derived-arity-test",
    ("T",),
    (ATOM, UNARY, BINARY),
).with_holes()

BINARY_CLOSURE: Final = (ATOM, UNARY, BINARY)
WIDTH: Final = 24
CLOSURE_DEPTH: Final = 1
CLOSURE_BUDGET: Final = 8


def _scheme_classes() -> tuple[type[RoleFillerScheme | ArityHeadScheme], ...]:
    return (RoleFillerScheme, ArityHeadScheme)


def _build(
    kind: type[RoleFillerScheme | ArityHeadScheme],
    reached: Sequence[Operator] = BINARY_CLOSURE,
    *,
    width: int = WIDTH,
    bound: ArityBound | None = None,
) -> RoleFillerScheme | ArityHeadScheme:
    torch.manual_seed(11)

    return kind(width=width, reached=reached, bound=bound)


def _state(seed: int, *, width: int = WIDTH) -> Encoded:
    generator = torch.Generator().manual_seed(seed)

    return Encoded(torch.randn(width, generator=generator), ("width",))


def test_the_default_bound_ships_the_declared_ranks() -> None:
    assert ArityBound().ranks == tuple(sorted(SHIPPED_RANKS))
    assert ArityBound().ceiling == max(SHIPPED_RANKS)


def test_a_bound_derives_its_ceiling_and_reason_from_the_ranks_it_ships() -> None:
    wider = ArityBound((1, 2, 3))

    assert wider.ceiling == 3
    assert "at most 3" in wider.reason
    assert "at most 2" in ArityBound().reason
    assert "at most 3" not in ArityBound().reason


def test_a_bound_refuses_a_rank_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="zero"):
        ArityBound((0, 2))

    with pytest.raises(ValueError, match="negative"):
        ArityBound((-1,))

    with pytest.raises(ValueError, match="at least one rank"):
        ArityBound(())


def test_a_bound_is_frozen_and_equal_on_the_ranks_it_ships() -> None:
    bound = ArityBound((2, 1, 2))

    assert bound.ranks == (1, 2)
    assert bound == ArityBound((1, 2))
    assert hash(bound) == hash(ArityBound((1, 2)))
    assert bound != ArityBound((1,))
    assert repr(bound) == "ArityBound((1, 2))"

    with pytest.raises(AttributeError, match="immutable"):
        bound.__setattr__("_ranks", (3,))

    with pytest.raises(AttributeError, match="immutable"):
        bound.__delattr__("_ranks")

    assert copy(bound) is bound
    assert deepcopy(bound) is bound


def test_the_served_ranks_are_the_positive_arities_the_closure_reaches() -> None:
    assert ArityBound().served(BINARY_CLOSURE) == (1, 2)
    assert ArityBound().served((ATOM,)) == ()
    assert ArityBound().served((BINARY, ATOM, BINARY)) == (2,)


def test_a_declared_arity_the_closure_never_reaches_costs_nothing() -> None:
    assert ArityBound().served(BINARY_CLOSURE) == (1, 2)
    assert TERNARY in SIGNATURE.operators

    for kind in _scheme_classes():
        scheme = _build(kind)

        assert scheme.ranks == (1, 2)


def test_a_reached_arity_above_the_ceiling_is_refused_with_the_declared_reason() -> (
    None
):
    bound = ArityBound()

    with pytest.raises(ArityUnsupportedError) as refusal:
        bound.served((*BINARY_CLOSURE, TERNARY))

    message = str(refusal.value)

    assert bound.reason in message
    assert "arity 3" in message
    assert "'triple'" in message


def test_a_reached_arity_the_family_skips_is_refused_as_well() -> None:
    bound = ArityBound((2,))

    with pytest.raises(ArityUnsupportedError) as refusal:
        bound.served((ATOM, UNARY, BINARY))

    assert bound.reason in str(refusal.value)
    assert "arity 1" in str(refusal.value)


def test_widening_the_shipped_ranks_admits_the_arity_the_default_refuses() -> None:
    assert ArityBound((1, 2, 3)).served((*BINARY_CLOSURE, TERNARY)) == (1, 2, 3)

    for kind in _scheme_classes():
        scheme = _build(kind, (*BINARY_CLOSURE, TERNARY), bound=ArityBound((1, 2, 3)))

        assert scheme.ranks == (1, 2, 3)
        assert len(scheme.factor(_state(3), arity=3)) == 3


def test_no_two_slots_of_three_served_ranks_share_a_module() -> None:
    wider = ArityBound((1, 2, 3))
    reached = (*BINARY_CLOSURE, TERNARY)
    torch.manual_seed(11)
    holographic = RoleFillerScheme(width=WIDTH, reached=reached, bound=wider)
    recursive = _build(ArityHeadScheme, reached, bound=wider)
    roles = tuple(
        holographic.role(arity, position)
        for arity in holographic.ranks
        for position in range(arity)
    )

    assert len(roles) == sum(holographic.ranks)

    for first, role in enumerate(roles):
        for other in roles[first + 1 :]:
            assert not torch.equal(role, other)

    parent = _state(24)
    second_of_two = recursive.factor(parent, arity=2)[1]
    first_of_three = recursive.factor(parent, arity=3)[0]

    assert not torch.allclose(second_of_two.tensor, first_of_three.tensor)


def test_role_filler_binding_inverts_once_the_magnitudes_leave_their_default() -> None:
    torch.manual_seed(11)
    scheme = RoleFillerScheme(width=WIDTH, reached=BINARY_CLOSURE)
    optimizer = torch.optim.SGD(scheme.parameters(), lr=0.5)
    loss = (
        torch.stack(tuple(scheme.role(2, position) for position in range(2)))
        .pow(2)
        .sum()
    )

    loss.backward()
    optimizer.step()

    moved = scheme.role(2, 0).detach()

    assert not torch.allclose(moved.abs(), torch.ones_like(moved))

    filler = _state(25)
    empty = Encoded(torch.zeros(WIDTH), ("width",))

    for position in range(2):
        children = [empty, empty]
        children[position] = filler
        bundle = scheme.combine(tuple(children))
        recovered = scheme.factor(bundle, arity=2)[position].tensor.detach()

        assert torch.allclose(recovered, filler.tensor, atol=1e-4)


def test_a_scheme_refuses_an_unserved_arity_at_construction() -> None:
    for kind in _scheme_classes():
        with pytest.raises(ArityUnsupportedError, match="at most 2"):
            _build(kind, (*BINARY_CLOSURE, TERNARY))


def test_a_scheme_refuses_a_width_that_is_not_positive() -> None:
    for kind in _scheme_classes():
        with pytest.raises(ValueError, match="zero"):
            _build(kind, width=0)


def test_a_scheme_satisfies_the_published_protocol() -> None:
    for kind in _scheme_classes():
        assert isinstance(_build(kind), ArityScheme)


def test_factoring_emits_one_child_state_per_argument_position() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        parent = _state(5)

        assert scheme.factor(parent, arity=0) == ()
        assert len(scheme.factor(parent, arity=1)) == 1
        assert len(scheme.factor(parent, arity=2)) == 2

        for child in scheme.factor(parent, arity=2):
            assert child.axes == parent.axes
            assert child.tensor.shape == parent.tensor.shape


def test_factoring_refuses_a_state_of_the_wrong_shape_even_at_arity_zero() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)

        with pytest.raises(ValueError, match="innermost"):
            scheme.factor(Encoded(torch.zeros(WIDTH), ("symbol",)), arity=0)

        assert scheme.factor(_state(4), arity=0) == ()


def test_factoring_refuses_a_negative_arity_and_an_unserved_one() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)

        with pytest.raises(ValueError, match="negative"):
            scheme.factor(_state(6), arity=-1)

        with pytest.raises(ArityUnsupportedError, match="not at 3"):
            scheme.factor(_state(6), arity=3)


def test_factoring_refuses_a_state_of_the_wrong_shape() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)

        with pytest.raises(ValueError, match="innermost"):
            scheme.factor(Encoded(torch.zeros(WIDTH), ("symbol",)), arity=1)

        with pytest.raises(ValueError, match="wide"):
            scheme.factor(Encoded(torch.zeros(WIDTH + 1), ("width",)), arity=1)


def test_a_child_state_differs_by_argument_position_and_by_arity() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        parent = _state(7)
        left, right = scheme.factor(parent, arity=2)
        (only,) = scheme.factor(parent, arity=1)

        assert not torch.allclose(left.tensor, right.tensor)
        assert not torch.allclose(left.tensor, only.tensor)


def test_combination_refuses_an_unserved_rank() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        child = _state(8)

        with pytest.raises(ArityUnsupportedError, match="not at 3"):
            scheme.combine((child, child, child))

        with pytest.raises(ArityUnsupportedError, match="not at 0"):
            scheme.combine(())


def test_combination_refuses_children_that_disagree() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        wide = Encoded(torch.zeros(2, WIDTH), ("cell", "width"))

        with pytest.raises(ValueError, match="share their axes"):
            scheme.combine((wide, Encoded(torch.zeros(3, WIDTH), ("cell", "width"))))

        with pytest.raises(ValueError, match="share their axes"):
            scheme.combine((wide, Encoded(torch.zeros(2, WIDTH), ("item", "width"))))


def test_a_combined_state_is_valid_only_where_every_child_is() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        left = Encoded(
            torch.zeros(3, WIDTH),
            ("cell", "width"),
            torch.tensor([True, True, False]),
        )
        right = Encoded(
            torch.zeros(3, WIDTH),
            ("cell", "width"),
            torch.tensor([True, False, True]),
        )
        combined = scheme.combine((left, right))

        assert combined.mask is not None
        assert combined.mask.tolist() == [True, False, False]
        assert scheme.combine((left, left)).mask is not None
        assert scheme.combine((_state(9), _state(10))).mask is None


def test_a_role_is_never_zero_and_differs_by_position_and_arity() -> None:
    scheme = RoleFillerScheme(width=WIDTH, reached=BINARY_CLOSURE)
    left = scheme.role(2, 0)
    right = scheme.role(2, 1)

    assert bool((left != 0.0).all())
    assert bool((right != 0.0).all())
    assert not torch.equal(left, right)
    assert not torch.equal(left, scheme.role(1, 0))

    with pytest.raises(ArityUnsupportedError, match="not at 3"):
        scheme.role(3, 0)

    with pytest.raises(ValueError, match="no argument at 2"):
        scheme.role(2, 2)


def test_role_filler_binding_inverts_exactly_at_every_served_rank() -> None:
    scheme = _build(RoleFillerScheme)
    filler = _state(12)
    empty = Encoded(torch.zeros(WIDTH), ("width",))

    (recovered,) = scheme.factor(scheme.combine((filler,)), arity=1)

    assert torch.allclose(recovered.tensor, filler.tensor, atol=1e-5)

    for position in range(2):
        children = [empty, empty]
        children[position] = filler
        bundle = scheme.combine(tuple(children))
        unbound = scheme.factor(bundle, arity=2)[position]

        assert torch.allclose(unbound.tensor, filler.tensor, atol=1e-5)


def test_a_bundle_holds_its_norm_steady_as_the_rank_grows() -> None:
    scheme = _build(RoleFillerScheme)
    filler = _state(13)
    empty = Encoded(torch.zeros(WIDTH), ("width",))
    alone = scheme.combine((filler,))
    paired = scheme.combine((filler, empty))

    expected = float(filler.tensor.norm())

    assert float(alone.tensor.detach().norm()) == pytest.approx(expected, abs=1e-4)
    assert float(paired.tensor.detach().norm()) == pytest.approx(
        expected / sqrt(2), abs=1e-4
    )


def test_role_filler_unbinding_favors_the_filler_it_was_bound_with() -> None:
    scheme = RoleFillerScheme(width=512, reached=BINARY_CLOSURE)
    left = _state(14, width=512)
    right = _state(15, width=512)
    bundle = scheme.combine((left, right))
    unbound = scheme.factor(bundle, arity=2)

    for position, filler in enumerate((left, right)):
        other = right if position == 0 else left
        recovered = unbound[position].tensor.detach()
        near = torch.nn.functional.cosine_similarity(recovered, filler.tensor, dim=0)
        far = torch.nn.functional.cosine_similarity(recovered, other.tensor, dim=0)

        assert float(near) > float(far)


def test_role_filler_cross_talk_follows_the_magnitudes_the_roles_hold() -> None:
    torch.manual_seed(13)
    scheme = RoleFillerScheme(width=512, reached=BINARY_CLOSURE)
    optimizer = torch.optim.SGD(scheme.parameters(), lr=0.5)
    loss = scheme.role(2, 1).abs().sum() - scheme.role(2, 0).abs().sum()

    loss.backward()
    optimizer.step()

    left = scheme.role(2, 0).detach()
    right = scheme.role(2, 1).detach()
    ratio = float(right.abs().mean() / left.abs().mean())

    assert ratio < 0.5

    first = _state(61, width=512)
    second = _state(62, width=512)
    recovered = scheme.factor(scheme.combine((first, second)), arity=2)
    into_first = recovered[0].tensor.detach() - first.tensor
    into_second = recovered[1].tensor.detach() - second.tensor

    assert float(into_first.norm()) == pytest.approx(
        ratio * float(second.tensor.norm()), rel=1e-3
    )
    assert float(into_second.norm()) == pytest.approx(
        float(first.tensor.norm()) / ratio, rel=1e-3
    )
    assert float(into_second.norm()) > float(into_first.norm())

    near = torch.nn.functional.cosine_similarity(
        recovered[0].tensor.detach(), first.tensor, dim=0
    )
    far = torch.nn.functional.cosine_similarity(
        recovered[1].tensor.detach(), second.tensor, dim=0
    )

    assert float(near) > float(far)


def test_the_head_family_holds_one_head_per_position_of_each_reached_arity() -> None:
    scheme = _build(ArityHeadScheme)
    names = {name for name, _ in scheme.named_parameters()}

    assert sum(1 for name in names if name.startswith("_down.")) == 2 * 3
    assert sum(1 for name in names if name.startswith("_up.")) == 2 * 2


def test_a_scheme_builds_nothing_for_an_arity_the_closure_never_reaches() -> None:
    binary_only = _build(ArityHeadScheme, (ATOM, BINARY))
    reaching_both = _build(ArityHeadScheme, BINARY_CLOSURE)
    fewer = sum(1 for _ in binary_only.parameters())
    more = sum(1 for _ in reaching_both.parameters())

    assert binary_only.ranks == (2,)
    assert fewer < more

    with pytest.raises(ArityUnsupportedError, match="not at 1"):
        binary_only.factor(_state(16), arity=1)


def test_gradients_cross_factoring_and_combination() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        tensor = _state(17).tensor.detach().clone().requires_grad_()
        parent = Encoded(tensor, ("width",))
        total = torch.stack(
            tuple(
                scheme.combine(scheme.factor(parent, arity=rank)).tensor.sum()
                for rank in scheme.ranks
            )
        ).sum()

        total.backward()

        assert tensor.grad is not None
        assert bool(tensor.grad.isfinite().all())

        for parameter in scheme.parameters():
            assert parameter.grad is not None
            assert bool(parameter.grad.isfinite().all())


def test_a_checkpoint_restores_the_role_signs() -> None:
    torch.manual_seed(21)
    written = RoleFillerScheme(width=WIDTH, reached=BINARY_CLOSURE)
    torch.manual_seed(22)
    read = RoleFillerScheme(width=WIDTH, reached=BINARY_CLOSURE)

    assert not torch.equal(written.role(2, 0), read.role(2, 0))

    read.load_state_dict(written.state_dict())

    assert torch.equal(written.role(2, 0), read.role(2, 0))


def _arity_signature(*, binary: bool) -> SignatureSpec:
    """Declare the atom, the unary raising and optionally the binary pairing."""
    unary = ConstructorSpec(
        name=UNARY.name,
        kind=("Cat", "Cat"),
        notation=PrefixConstructorNotationSpec(spelling=UNARY.name),
    )
    pair = ConstructorSpec(
        name=BINARY.name,
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling="*", precedence=10, associativity="left"
        ),
    )

    return SignatureSpec(
        name="derived-arity-test" if binary else "unary-arity-test",
        primitives=(ATOM.name,),
        constructors=(unary, pair) if binary else (unary,),
        functors=(
            FunctorSpec(
                name=BINARY.name,
                constructor=BINARY.name,
                argument_side="either",
                opposite=BINARY.name,
            ),
        )
        if binary
        else (),
    )


UNARY_RAISE: Final = PatternRuleSpec(
    name="unary-raise",
    premises=(TermPatternSpec(name="X", sort="Cat"),),
    conclusion=NodePatternSpec(
        operator=FixedOperatorSpec(name=UNARY.name),
        arguments=(TermPatternSpec(name="X", sort="Cat"),),
    ),
)
"""The unary rule rewriting any category to its raised constructor."""


def _grammar(
    signature: SignatureSpec, rules: tuple[RuleSpec, ...]
) -> tuple[Grammar, ClosureReport[Term]]:
    """Compile a grammar whose closure is computed from its own rules.

    Parameters
    ----------
    signature : SignatureSpec
    rules : tuple[RuleSpec, ...]
    """
    adopted = declared_rules(signature, RuleSetSpec(name="arity-rules", rules=rules))
    store = adopted.context.store
    root = adopted.context.syntax.parse(store, ATOM.name)
    ruleset = adopted.rule_set
    matching = ruleset.matching
    declaration = FormalismDeclaration(
        ruleset.closure((root,), max_depth=CLOSURE_DEPTH, budget=CLOSURE_BUDGET),
        TheoryArtifact.from_specification(store.signature.name, b"{}"),
        1,
        Budget(
            depth=5,
            mass=1.0,
            beam=8,
            unary_rounds=1,
            threads=1,
            beta_fuel=1_000,
            max_category_arity=None,
            max_category_size=None,
        ),
        capacity_report(ruleset),
        lexicalized=True,
    )

    targets = TargetSet((root,), matching)
    formalism = Formalism(
        store.signature.name,
        store,
        ruleset,
        targets,
        RootGoal(targets),
        declaration,
    )

    return formalism.compile(), formalism.closure


def _derived_grammar() -> tuple[Grammar, ClosureReport[Term]]:
    return _grammar(
        _arity_signature(binary=True),
        (
            UNARY_RAISE,
            TypeRaisingSpec(view=BINARY.name, targets=(ATOM.name,), arguments=None),
        ),
    )


def _unary_grammar() -> tuple[Grammar, ClosureReport[Term]]:
    return _grammar(_arity_signature(binary=False), (UNARY_RAISE,))


def test_both_rules_fire_and_the_closure_reaches_each_of_their_operators() -> None:
    _grammar_of_both, closure = _derived_grammar()
    names = [op.name for op in reached_operators(closure)]
    atom = closure.reached[0].op.name

    assert closure.never_applicable == ()
    assert set(names) == {atom, UNARY.name, BINARY.name}
    assert names[0] == atom


def test_the_operators_a_closure_reaches_are_read_off_its_terms() -> None:
    store = TermStore(SIGNATURE)
    nested = store.op(UNARY, store.op(BINARY, store.op(ATOM), store.op(ATOM)))
    closure = ClosureReport((store.op(ATOM), nested), saturated=True, steps=1, budget=4)

    assert reached_operators(closure) == (ATOM, UNARY, BINARY)
    empty: ClosureReport[Term] = ClosureReport((), saturated=True, steps=0, budget=0)

    assert reached_operators(empty) == ()


def test_the_branching_a_scheme_pays_for_is_computed_from_the_closure() -> None:
    _derived, derived_closure = _derived_grammar()
    _unary, unary_closure = _unary_grammar()
    reached = reached_operators(derived_closure)
    unary_only = reached_operators(unary_closure)

    assert TERNARY in SIGNATURE.operators
    assert TERNARY not in reached

    for kind in _scheme_classes():
        assert _build(kind, reached).ranks == (1, 2)
        assert _build(kind, unary_only).ranks == (1,)


def test_a_closure_reaching_only_nullary_operators_serves_no_rank() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind, (ATOM,))
        state = _state(18)

        assert scheme.ranks == ()
        assert scheme.factor(state, arity=0) == ()

        with pytest.raises(ArityUnsupportedError, match="no combination"):
            scheme.combine((state,))

        with pytest.raises(ArityUnsupportedError, match="so none at rank 1"):
            scheme.factor(state, arity=1)


def test_combination_refuses_a_child_shaped_for_another_scheme() -> None:
    for kind in _scheme_classes():
        scheme = _build(kind)
        child = _state(19)
        narrow = Encoded(torch.ones(1), ("width",))

        with pytest.raises(ValueError, match="reads states 24 wide, not 1 wide"):
            scheme.combine((narrow, narrow))

        with pytest.raises(ValueError, match="reads states 24 wide, not 1 wide"):
            scheme.combine((child, narrow))

        with pytest.raises(ValueError, match="innermost"):
            scheme.combine((Encoded(torch.ones(WIDTH), ("symbol",)),))


def _conformance(
    kind: type[RoleFillerScheme | ArityHeadScheme],
    reached: Sequence[Operator] = BINARY_CLOSURE,
) -> AritySchemeConformance:
    return AritySchemeConformance(lambda: _build(kind, reached))


class DroppingScheme(RoleFillerScheme):
    """A scheme that emits one child however many argument positions there are."""

    @override
    def factor(self, parent: Encoded, /, *, arity: int) -> tuple[Encoded, ...]:
        return super().factor(parent, arity=arity)[:1]


class TestTheRoleFillerSchemeConforms(AritySchemeSuite):
    @pytest.fixture
    def conformance(self) -> AritySchemeConformance:
        return _conformance(RoleFillerScheme)


class TestTheArityHeadSchemeConforms(AritySchemeSuite):
    @pytest.fixture
    def conformance(self) -> AritySchemeConformance:
        return _conformance(ArityHeadScheme)


class TestASchemeServingOneRankConforms(AritySchemeSuite):
    @pytest.fixture
    def conformance(self) -> AritySchemeConformance:
        return _conformance(ArityHeadScheme, (ATOM, UNARY))


def test_the_suite_names_every_arity_scheme_law() -> None:
    assert len(ARITY_SCHEME_LAWS) == 6


def test_the_suite_rejects_a_scheme_that_drops_a_child() -> None:
    with pytest.raises(LawViolationError, match="factored into 1 children"):
        AritySchemeSuite().test_factoring_emits_one_state_per_argument_position(
            _conformance(DroppingScheme)
        )
