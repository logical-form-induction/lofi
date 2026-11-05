from itertools import product
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

import pytest
from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import (
    HOLE,
    AllOf,
    CompositionRole,
    EisnerNormalForm,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    NormalForm,
    NormalFormRole,
)
from lofi.judge.laws import NormalFormFixture, check_normal_form_laws
from lofi.spec import ApplicationSpec, CompositionSpec, RuleSetSpec
from lofi.spec.grammar import FeatureErasureMatchingSpec, OperatorSpec
from lofi.terms import Term

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge import (
        CompositionRestriction,
        Direction,
        NormalFormDecisionTable,
        Rule,
        RuleProbe,
        RuleRestriction,
        Slot,
    )
    from lofi.terms import OpName


type Role = Literal["forward", "backward"]
"""Directed composition role."""


class Exact:
    """Integer-identity congruence."""

    @property
    def name(self) -> str:
        """Return the congruence name."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Report that representatives are available."""
        return True

    def key(self, term: int, /) -> int:
        """Select an integer as its own representative."""
        return term

    def equivalent(self, left: int, right: int, /) -> bool:
        """Compare two integers."""
        return left == right


class FunctorView:
    """Named forward and backward slashes."""

    def __init__(self, forward: OpName, backward: OpName) -> None:
        """Store the two distinguished slashes."""
        self._forward = forward
        self._backward = backward

    @property
    def forward_slash(self) -> OpName:
        """Return the forward slash."""
        return self._forward

    @property
    def backward_slash(self) -> OpName:
        """Return the backward slash."""
        return self._backward


class StepRule:
    """Binary integer rule with normal-form metadata."""

    def __init__(
        self,
        kind: str = "other",
        role: Role = "forward",
        order: int | None = 0,
        *,
        applied: int | None = None,
        self_opposite: bool = False,
        direction: CompositionRestriction = "unrestricted",
        crossed: bool = False,
    ) -> None:
        """Store the rule declaration and composition metadata."""
        backward = "fwd" if self_opposite else "bwd"
        self._kind = kind
        self._direction = direction
        self._crossed = crossed
        self._order = order
        self._applied = order if applied is None else applied
        self._functors = FunctorView("fwd", backward)
        self._slash = "fwd" if role == "forward" else backward

    @property
    def name(self) -> str:
        """Return the rule name."""
        return f"{self._kind}:{self._slash}:{self._order}"

    @property
    def arity(self) -> int:
        """Return the binary arity."""
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the forward and checking directions."""
        return frozenset({"forward", "check"})

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return no residuable slots."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Report complete forward enumeration."""
        return True

    @property
    def matching(self) -> Exact:
        """Return exact integer matching."""
        return Exact()

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    @property
    def functors(self) -> FunctorView:
        """Return the functor view."""
        return self._functors

    @property
    def order(self) -> int | None:
        """Return the declared order, or ``None`` for every finite order."""
        return self._order

    @property
    def slash(self) -> OpName:
        """Return the principal slash."""
        return self._slash

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the structural role under test."""
        if self._kind == "composition":
            direction = (
                "undirected"
                if self._functors.forward_slash == self._functors.backward_slash
                else "forward"
                if self._slash == self._functors.forward_slash
                else "backward"
            )
            return NormalFormRole(
                composition=CompositionRole(
                    direction,
                    principal_slot=0 if direction != "backward" else 1,
                    order=self._order,
                    crossed=self._crossed,
                )
            )
        if self._kind == "type-raising":
            direction = (
                "undirected"
                if self._functors.forward_slash == self._functors.backward_slash
                else "forward"
                if self._slash == "fwd"
                else "backward"
            )
            return NormalFormRole(raising=direction)
        if self._kind == "coordination":
            return NormalFormRole(coordination=True)
        return NormalFormRole()

    def applied_order(self, inputs: tuple[int, ...], output: int, /) -> int | None:
        """Return the order this step is declared to have used."""
        del inputs, output

        return self._applied

    def apply(self, inputs: tuple[int, ...], /) -> Iterator[int]:
        """Add a premise pair."""
        if len(inputs) == self.arity:
            yield sum(inputs)

    def admits(self, inputs: tuple[int, ...], output: int, /) -> bool:
        """Check the integer addition relation."""
        return len(inputs) == self.arity and sum(inputs) == output

    def residuate(self, output: int, known: tuple[Slot[int], ...], /) -> Iterator[int]:
        """Solve either missing addend."""
        if len(known) != self.arity or known.count(HOLE) != 1:
            return

        for term in known:
            if isinstance(term, int):
                yield output - term

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {"rule": self._kind, "direction": self._direction}


class UnaryStepRule(StepRule):
    """Unary rule with normal-form metadata."""

    @property
    @override
    def arity(self) -> int:
        """Return the unary arity."""
        return 1


@pytest.mark.parametrize(
    "normal_form",
    [
        NoNormalForm[int](),
        EisnerNormalForm[int](),
        HockenmaierBiskNormalForm[int](),
        AllOf[int]((EisnerNormalForm(), HockenmaierBiskNormalForm())),
    ],
    ids=lambda normal_form: normal_form.name,
)
def test_each_filter_has_a_finite_alphabet_and_is_structural(
    normal_form: NormalForm[int],
) -> None:
    assert isinstance(normal_form, NormalForm)
    assert normal_form.alphabet
    assert isinstance(normal_form.alphabet, frozenset)
    assert normal_form.lexical_tag() in normal_form.alphabet


@pytest.mark.parametrize(
    "normal_form",
    [
        NoNormalForm[int](),
        EisnerNormalForm[int](),
        HockenmaierBiskNormalForm[int](),
        AllOf[int]((EisnerNormalForm(), HockenmaierBiskNormalForm())),
    ],
    ids=lambda normal_form: normal_form.name,
)
def test_the_decision_table_agrees_with_every_term_independent_decision(
    normal_form: NormalForm[int],
) -> None:
    rules: tuple[Rule[int], ...] = (
        StepRule(),
        StepRule("composition", order=0),
        StepRule("composition", order=1),
        StepRule("composition", order=None, applied=1),
        StepRule("composition", order=1, direction="steedman"),
        UnaryStepRule("type-raising"),
        UnaryStepRule("ccg-coordination-phase"),
    )
    table = normal_form.decisions(rules)

    assert len(table) == len(rules)

    for rule, row in zip(rules, table, strict=True):
        assert set(row) == set(product(normal_form.alphabet, repeat=rule.arity))

        for children, decision in row.items():
            if decision != "premises":
                inputs = (1,) * rule.arity
                expected = normal_form.admits(rule, children, inputs, sum(inputs))

                assert (decision == "admit") is expected


def test_no_normal_form_keeps_every_complete_step() -> None:
    normal_form = NoNormalForm[int]()
    rule = StepRule()

    assert normal_form.alphabet == frozenset({"OTHER"})
    assert normal_form.tag(rule, ("OTHER", "OTHER")) == "OTHER"
    assert normal_form.admits(rule, ("OTHER", "OTHER"), (1, 2), 100)
    assert not normal_form.admits(rule, ("OTHER",), (1, 2), 3)


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("order", range(3))
def test_eisner_tags_only_positive_composition_results(role: Role, order: int) -> None:
    normal_form = EisnerNormalForm[int]()
    rule = StepRule("composition", role, order)
    expected = "OTHER" if order == 0 else ("FC" if role == "forward" else "BC")

    assert normal_form.tag(rule, ("OTHER", "OTHER")) == expected


@pytest.mark.parametrize(
    ("role", "children", "expected"),
    [
        ("forward", ("FC", "OTHER"), False),
        ("forward", ("OTHER", "FC"), True),
        ("forward", ("BC", "OTHER"), True),
        ("backward", ("OTHER", "BC"), False),
        ("backward", ("BC", "OTHER"), True),
        ("backward", ("OTHER", "FC"), True),
    ],
)
@pytest.mark.parametrize("order", [0, 2], ids=("application", "composition"))
def test_eisner_blocks_same_direction_composition_in_the_primary_slot(
    role: Role,
    children: tuple[str, str],
    order: int,
    *,
    expected: bool,
) -> None:
    normal_form = EisnerNormalForm[int]()
    rule = StepRule("composition", role, order)

    assert normal_form.admits(rule, children, (1, 2), 3) is expected


def test_directional_filters_treat_self_opposite_composition_as_undirected() -> None:
    rule = StepRule("composition", order=1, self_opposite=True)
    eisner = EisnerNormalForm[int]()
    hockenmaier_bisk = HockenmaierBiskNormalForm[int]()

    assert eisner.tag(rule, ("OTHER", "OTHER")) == "UC"
    assert hockenmaier_bisk.tag(rule, ("OTHER", "OTHER")) == "UC1"


@pytest.mark.parametrize(
    ("normal_form", "primary"),
    [
        (EisnerNormalForm[int](), "FC"),
        (HockenmaierBiskNormalForm[int](), "FC1"),
    ],
    ids=("eisner", "hockenmaier-bisk"),
)
@pytest.mark.parametrize("order", [0, 1], ids=("application", "composition"))
def test_crossed_composition_can_be_included_or_excluded(
    normal_form: NormalForm[int], primary: str, order: int
) -> None:
    crossed = StepRule("composition", "forward", order, crossed=True)

    if isinstance(normal_form, EisnerNormalForm):
        excluded = EisnerNormalForm[int](include_crossed=False)
        assert excluded.admits(crossed, (primary, "OTHER"), (1, 2), 3)

    assert not normal_form.admits(crossed, (primary, "OTHER"), (1, 2), 3)


def test_hockenmaier_bisk_declares_degree_raising_and_coordination_tags() -> None:
    normal_form = HockenmaierBiskNormalForm[int]()

    assert normal_form.name == "hockenmaier-bisk"
    assert normal_form.alphabet == frozenset(
        {
            "OTHER",
            "FTR",
            "BTR",
            "UTR",
            "COORD",
            "FC",
            "FC1",
            "FCn",
            "BC",
            "BC1",
            "BCn",
            "UC",
            "UC1",
            "UCn",
        }
    )
    assert normal_form.tag(StepRule("composition", "forward", 0), ()) == "OTHER"
    assert normal_form.tag(StepRule("type-raising"), ()) == "FTR"
    assert normal_form.tag(StepRule("type-raising", "backward"), ()) == "BTR"
    assert normal_form.tag(StepRule("coordination"), ()) == "COORD"
    assert normal_form.tag(StepRule("pattern"), ()) == "OTHER"
    assert normal_form.tag(StepRule(), ()) == "OTHER"


@pytest.mark.parametrize("track_child_order", [False, True])
def test_hockenmaier_bisk_tags_undirected_raising_on_its_own(
    *, track_child_order: bool
) -> None:
    normal_form = HockenmaierBiskNormalForm[int](track_child_order=track_child_order)
    raising = StepRule("type-raising", self_opposite=True)
    application = StepRule("composition", order=0, self_opposite=True)
    (row,) = normal_form.decisions((application,))

    assert "UTR" in normal_form.alphabet
    assert normal_form.tag(raising, ()) == "UTR"
    assert row[("UTR", "OTHER")] == "premises"
    assert row[("FTR", "OTHER")] == "admit"
    assert row[("BTR", "OTHER")] == "admit"


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize(
    ("order", "suffix"), [(1, "1"), (2, "n"), (9, "n"), (None, "")]
)
def test_hockenmaier_bisk_tags_composition_by_direction_and_degree(
    role: Role, order: int | None, suffix: str
) -> None:
    normal_form = HockenmaierBiskNormalForm[int]()
    expected = ("FC" if role == "forward" else "BC") + suffix

    assert normal_form.tag(StepRule("composition", role, order), ()) == expected


def test_exact_hockenmaier_bisk_tags_retain_bounded_child_orders() -> None:
    normal_form = HockenmaierBiskNormalForm[int](
        order_bound=3,
        track_child_order=True,
    )

    assert {"FC:1", "FC:2", "FC:3", "FC:>3"} <= normal_form.alphabet
    assert normal_form.tag(StepRule("composition", "forward", 2), ()) == "FC:2"
    assert normal_form.tag(StepRule("composition", "forward", 4), ()) == "FC:>3"


def test_hockenmaier_bisk_constraints_can_be_disabled_independently() -> None:
    primary = HockenmaierBiskNormalForm[int](
        order_bound=3,
        track_child_order=True,
        composition_primary=False,
    )
    raising = HockenmaierBiskNormalForm[int](coordination_raising=False)

    assert primary.admits(
        StepRule("composition", "forward", 1),
        ("FC:2", "OTHER"),
        (1, 2),
        3,
    )
    assert raising.admits(UnaryStepRule("type-raising"), ("COORD",), (1,), 2)


def test_hockenmaier_bisk_raises_anything_but_a_coordination() -> None:
    normal_form = HockenmaierBiskNormalForm[int]()
    rule = UnaryStepRule("type-raising")

    assert normal_form.admits(rule, ("OTHER",), (1,), 2)
    assert normal_form.admits(rule, ("FTR",), (1,), 2)
    assert normal_form.admits(rule, ("FCn",), (1,), 2)
    assert not normal_form.admits(rule, ("COORD",), (1,), 2)


class ConstraintCase(NamedTuple):
    """One bounded normal-form admission case."""

    order: int
    primary: str
    secondary: str
    expected: bool


CONSTRAINT_CASES: Final = (
    ConstraintCase(0, "FC1", "OTHER", expected=False),
    ConstraintCase(0, "FCn", "OTHER", expected=False),
    ConstraintCase(1, "FCn", "OTHER", expected=False),
    ConstraintCase(1, "FC", "OTHER", expected=False),
    ConstraintCase(2, "FC1", "OTHER", expected=False),
    ConstraintCase(9, "FC1", "OTHER", expected=False),
    ConstraintCase(2, "FCn", "OTHER", expected=True),
    ConstraintCase(2, "FC", "OTHER", expected=True),
    ConstraintCase(1, "BC1", "OTHER", expected=True),
    ConstraintCase(0, "OTHER", "FCn", expected=True),
    ConstraintCase(2, "OTHER", "FCn", expected=True),
    ConstraintCase(1, "FTR", "BC1", expected=True),
    ConstraintCase(1, "FTR", "BC", expected=True),
    ConstraintCase(2, "FTR", "BCn", expected=True),
    ConstraintCase(2, "FTR", "OTHER", expected=True),
    ConstraintCase(0, "BTR", "OTHER", expected=True),
)


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("case", CONSTRAINT_CASES)
def test_hockenmaier_bisk_applies_each_order_and_raising_constraint(
    role: Role, case: ConstraintCase
) -> None:
    normal_form = HockenmaierBiskNormalForm[int]()
    translate = str.maketrans({"F": "B", "B": "F"}) if role == "backward" else None
    primary = (
        case.primary.translate(translate) if translate is not None else case.primary
    )
    secondary = (
        case.secondary.translate(translate) if translate is not None else case.secondary
    )
    children = (primary, secondary) if role == "forward" else (secondary, primary)
    rule = StepRule("composition", role, case.order)

    assert normal_form.admits(rule, children, (1, 2), 3) is case.expected


@pytest.mark.parametrize(
    ("applied", "expected"),
    [(0, False), (1, False), (2, True), (7, True)],
)
def test_hockenmaier_bisk_reads_the_order_a_step_derived_when_none_is_declared(
    applied: int,
    *,
    expected: bool,
) -> None:
    normal_form = HockenmaierBiskNormalForm[int]()
    rule = StepRule("composition", "forward", None, applied=applied)

    assert normal_form.admits(rule, ("FC", "OTHER"), (1, 2), 3) is expected


def test_hockenmaier_bisk_admits_a_step_whose_order_it_cannot_read() -> None:
    normal_form = HockenmaierBiskNormalForm[int]()
    rule = StepRule("composition", "forward", None, applied=None)

    assert normal_form.admits(rule, ("FC", "OTHER"), (1, 2), 3)


def test_all_of_encodes_and_projects_the_product_alphabet() -> None:
    eisner = EisnerNormalForm[int]()
    bounded = HockenmaierBiskNormalForm[int]()
    normal_form = AllOf[int]((eisner, bounded))
    lexical = normal_form.lexical_tag()
    first_order = StepRule("composition", "forward", 1)
    application = StepRule("composition", "forward", 0)
    composed = normal_form.tag(first_order, (lexical, lexical))

    assert normal_form.normal_forms == (eisner, bounded)
    assert len(normal_form.alphabet) == len(eisner.alphabet) * len(bounded.alphabet)
    assert lexical == '["OTHER","OTHER"]'
    assert composed == '["FC","FC1"]'
    assert not normal_form.admits(application, (composed, lexical), (1, 2), 3)


class LastChildTag:
    """Filter with delimiter-bearing tags."""

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "delimiter-tags"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return two delimiter-bearing tags."""
        return frozenset({"a", "a,b"})

    def lexical_tag(self) -> str:
        """Return the first tag."""
        return "a"

    def decisions(self, rules: Sequence[Rule[int]], /) -> NormalFormDecisionTable:
        """Admit every finite child-tag combination."""
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "admit")
            for rule in rules
        )

    def tag(self, rule: Rule[int], children: tuple[str, ...], /) -> str:
        """Copy the last child tag."""
        del rule

        return children[-1] if children else "a"

    def admits(
        self,
        rule: Rule[int],
        children: tuple[str, ...],
        inputs: tuple[int, ...],
        output: int,
        /,
    ) -> bool:
        """Admit every step."""
        del rule, children, inputs, output

        return True


def test_all_of_encoding_is_collision_free_and_rejects_unknown_child_tags() -> None:
    one = LastChildTag()
    normal_form = AllOf[int]((one, one))

    assert len(normal_form.alphabet) == 4

    with pytest.raises(ValueError, match="unknown conjunction tag"):
        normal_form.tag(StepRule(), ("not-a-product-tag",))


def test_empty_all_of_is_the_structurally_complete_identity_filter() -> None:
    normal_form = AllOf[int](())
    rule = StepRule()

    assert normal_form.name == "all-of"
    assert normal_form.alphabet == frozenset({"[]"})
    assert normal_form.lexical_tag() == "[]"
    assert normal_form.tag(rule, ("[]", "[]")) == "[]"
    assert normal_form.admits(rule, ("[]", "[]"), (1, 2), 3)
    assert not normal_form.admits(rule, ("[]",), (1, 2), 3)


def test_stateful_normal_forms_are_frozen_and_copy_their_components() -> None:
    stateless = HockenmaierBiskNormalForm[int]()
    held: list[NormalForm[int]] = [stateless]
    conjunction = AllOf(held)
    held.clear()

    assert conjunction.normal_forms == (stateless,)

    with pytest.raises(TypeError):
        vars(stateless)

    with pytest.raises(AttributeError, match="immutable"):
        conjunction.__setattr__("_name", "changed")


PLAIN_RULE: Final[Rule[int]] = StepRule()
PLAIN_RULES: Final = (PLAIN_RULE,)
TERMS: Final = st.integers(min_value=0, max_value=5)
TOKENS: Final = st.lists(st.just("x"), min_size=1, max_size=4).map(tuple)


def _reachable(
    normal_form: NormalForm[int] | None, tokens: tuple[str, ...]
) -> frozenset[int]:
    """Enumerate sums over a one-category toy grammar."""
    lexical = "" if normal_form is None else normal_form.lexical_tag()
    cells: dict[tuple[int, int], set[tuple[int, str]]] = {
        (index, index + 1): {(1, lexical)} for index in range(len(tokens))
    }

    for width in range(2, len(tokens) + 1):
        for start in range(len(tokens) - width + 1):
            end = start + width
            here: set[tuple[int, str]] = set()

            for split in range(start + 1, end):
                for left, left_tag in cells[start, split]:
                    for right, right_tag in cells[split, end]:
                        children = (left_tag, right_tag)
                        inputs = (left, right)

                        for output in PLAIN_RULE.apply(inputs):
                            if normal_form is None:
                                here.add((output, ""))
                            elif normal_form.admits(
                                PLAIN_RULE, children, inputs, output
                            ):
                                tag = normal_form.tag(PLAIN_RULE, children)
                                here.add((output, tag))

            cells[start, end] = here

    return frozenset(value for value, _ in cells[0, len(tokens)])


@pytest.mark.parametrize(
    "normal_form",
    [
        NoNormalForm[int](),
        EisnerNormalForm[int](),
        HockenmaierBiskNormalForm[int](),
        AllOf[int]((EisnerNormalForm(), HockenmaierBiskNormalForm())),
    ],
    ids=lambda normal_form: normal_form.name,
)
def test_each_filter_holds_to_all_normal_form_laws(
    normal_form: NormalForm[int],
) -> None:
    check_normal_form_laws(
        NormalFormFixture(normal_form, PLAIN_RULES, TERMS, TOKENS, _reachable, 30)
    )


RAISING: Final = declared_rules(
    ccg_signature(
        "t", "x", "z", operators=(OperatorSpec(name="t[b]", sort="Cat", erases_to="t"),)
    ),
    RuleSetSpec(
        name="raising",
        rules=(
            ApplicationSpec(view="/"),
            CompositionSpec(view="/", order=1, direction="unrestricted"),
        ),
    ),
    matching=FeatureErasureMatchingSpec(),
)
APPLICATION, COMPOSITION = RAISING.rule_set


def category(text: str) -> Term:
    return RAISING.context.syntax.parse(RAISING.context.store, text)


@pytest.mark.parametrize(("result", "refused"), [("t", True), ("t[b]", False)])
def test_a_raised_functor_is_refused_in_application_only_when_lowering_agrees(
    result: str, *, refused: bool
) -> None:
    raised = category("t/(t\\x)")
    functor = category(f"{result}\\x")
    (output,) = APPLICATION.apply((raised, functor))
    normal_form = HockenmaierBiskNormalForm[Term]()

    assert output == category("t")
    assert normal_form.admits(
        APPLICATION, ("FTR", "OTHER"), (raised, functor), output
    ) is (not refused)


@pytest.mark.parametrize(("result", "refused"), [("t", True), ("t[b]", False)])
def test_a_raised_primary_meets_a_higher_backward_composition_only_when_lowering_agrees(
    result: str, *, refused: bool
) -> None:
    raised = category("t/(t\\x)")
    container = category(f"({result}\\x)/z")
    (output,) = COMPOSITION.apply((raised, container))
    normal_form = HockenmaierBiskNormalForm[Term]()

    assert normal_form.admits(
        COMPOSITION, ("FTR", "BCn"), (raised, container), output
    ) is (not refused)
    assert normal_form.admits(COMPOSITION, ("FTR", "BC1"), (raised, container), output)


@pytest.mark.parametrize(
    ("normal_form", "below"),
    [
        *((EisnerNormalForm[int](), tag) for tag in ("FC", "BC")),
        *(
            (HockenmaierBiskNormalForm[int](), tag)
            for tag in ("FC", "BC", "FC1", "BCn", "FTR", "COORD")
        ),
    ],
)
def test_a_coercion_resets_the_tag_of_the_item_it_concludes(
    normal_form: NormalForm[int], below: str
) -> None:
    coercion = UnaryStepRule("pattern")

    assert normal_form.admits(coercion, (below,), (1,), 2)
    assert normal_form.tag(coercion, (below,)) == "OTHER"
