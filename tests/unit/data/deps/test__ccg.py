from copy import copy, deepcopy
from pickle import dumps, loads
from typing import Final, NamedTuple, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.unit.data.deps.test__projection import (
    DependencyProjectorConformanceTests,
    ProjectorCase,
)

from lofi.data import DerivationNode, Token, TokenSpan
from lofi.data.deps import (
    ArgumentSlot,
    CCGCategoryAlgebra,
    CCGDependencyConvention,
    CCGDependencyProjector,
    CCGDependencyTemplate,
    CCGProjectionReport,
    CoindexedPath,
    DecoratedCategory,
    HeadProjection,
    LexicalHead,
    ProjectedDependency,
)
from lofi.spec import HeadSelection


class Category(NamedTuple):
    """A small structural CCG category."""

    name: str
    result: Category | None = None
    argument: Category | None = None
    head_sharing: bool = False
    coindexed: tuple[CoindexedPath, ...] = ()


class CategoryAlgebra:
    """Synthetic structural category operations."""

    def is_functor(self, category: Category, /) -> bool:
        return category.result is not None

    def result(self, category: Category, /) -> Category:
        if category.result is None:
            raise ValueError(category)
        return category.result

    def argument(self, category: Category, /) -> Category:
        if category.argument is None:
            raise ValueError(category)
        return category.argument

    def equivalent(self, left: Category, right: Category, /) -> bool:
        return left == right

    def shares_head(self, category: Category, /) -> bool:
        return category.head_sharing

    def coindices(self, category: Category, /) -> tuple[CoindexedPath, ...]:
        return category.coindexed


class CategoryAlgebraCase[C](NamedTuple):
    """A reusable CCG category-algebra conformance case."""

    algebra: CCGCategoryAlgebra[C]
    atom: C
    functor: C
    result: C
    argument: C
    head_sharing: C
    nonsharing: C


class CCGCategoryAlgebraConformanceTests[C]:
    """CCG category-algebra conformance tests."""

    def make_category_algebra_case(self) -> CategoryAlgebraCase[C]:
        raise NotImplementedError

    def test_the_algebra_distinguishes_atoms_and_exposes_functor_parts(self) -> None:
        case = self.make_category_algebra_case()

        assert isinstance(case.algebra, CCGCategoryAlgebra)
        assert not case.algebra.is_functor(case.atom)
        assert case.algebra.is_functor(case.functor)
        assert case.algebra.equivalent(
            case.algebra.result(case.functor),
            case.result,
        )
        assert case.algebra.equivalent(
            case.algebra.argument(case.functor),
            case.argument,
        )

    def test_the_algebra_declares_head_sharing_structurally(self) -> None:
        case = self.make_category_algebra_case()

        assert case.algebra.shares_head(case.head_sharing)
        assert not case.algebra.shares_head(case.nonsharing)


def _functor(
    result: Category,
    argument: Category,
    *,
    head_sharing: bool = False,
    coindexed: tuple[CoindexedPath, ...] = (),
) -> Category:
    """Build a synthetic functor category."""
    return Category(
        f"({result.name}|{argument.name})",
        result,
        argument,
        head_sharing,
        coindexed,
    )


N: Final = Category("N")
NP: Final = Category("NP")
S: Final = Category("S")
S_BACK_NP: Final = _functor(S, NP)
TRANSITIVE: Final = _functor(S_BACK_NP, NP)
DETERMINER: Final = _functor(NP, N, head_sharing=True)
MODIFIER: Final = _functor(S_BACK_NP, S_BACK_NP, head_sharing=True)
COMPOSED: Final = _functor(S_BACK_NP, NP)
S_OVER_NP: Final = _functor(S, NP)
NP_OVER_N: Final = _functor(NP, N)
S_OVER_N: Final = _functor(S, N)
DOUBLE_NOUN_BUILDER: Final = _functor(NP_OVER_N, N)
DOUBLE_COMPOSED: Final = _functor(S_OVER_N, N)
RAISED_NP: Final = _functor(S, S_BACK_NP)
SUBJECT_SHARING: Final = _functor(S_BACK_NP, S_BACK_NP)
CONTROL_RESULT: Final = _functor(S_BACK_NP, S_BACK_NP)
OBJECT_CONTROL: Final = _functor(CONTROL_RESULT, NP)
ALGEBRA: Final = CategoryAlgebra()
PROJECTOR: Final = CCGDependencyProjector(ALGEBRA)
SUBJECT_TEMPLATE: Final = CCGDependencyTemplate(
    "subject-sharing",
    SUBJECT_SHARING,
    (
        CoindexedPath(("result", "argument"), 1),
        CoindexedPath(("argument", "argument"), 1),
    ),
)
OBJECT_TEMPLATE: Final = CCGDependencyTemplate(
    "object-control",
    OBJECT_CONTROL,
    (
        CoindexedPath(("argument",), 1),
        CoindexedPath(("result", "argument", "argument"), 1),
    ),
)
CONVENTION: Final = CCGDependencyConvention(
    "synthetic-ccgbank",
    (SUBJECT_TEMPLATE, OBJECT_TEMPLATE),
)


class TestSyntheticCategoryAlgebra(CCGCategoryAlgebraConformanceTests[Category]):
    @override
    def make_category_algebra_case(self) -> CategoryAlgebraCase[Category]:
        return CategoryAlgebraCase(
            ALGEBRA,
            N,
            TRANSITIVE,
            S_BACK_NP,
            NP,
            DETERMINER,
            TRANSITIVE,
        )


class TestCCGDependencyProjector(
    DependencyProjectorConformanceTests[
        Category,
        ProjectedDependency[Category],
        DecoratedCategory[Category],
    ]
):
    @override
    def make_projector_case(
        self,
    ) -> ProjectorCase[
        Category,
        ProjectedDependency[Category],
        DecoratedCategory[Category],
    ]:
        return ProjectorCase(
            PROJECTOR,
            DerivationNode(TokenSpan(0, 1), NP),
            Token(0, "Kim"),
            DerivationNode(
                TokenSpan(0, 1),
                RAISED_NP,
                rule="type-raising",
                children=(0,),
            ),
            "type-raising",
            HeadSelection("pass-through", 0),
            (0,),
        )


def _lexical(
    category: Category,
    position: int,
    form: str,
    projector: CCGDependencyProjector[Category] = PROJECTOR,
) -> HeadProjection[ProjectedDependency[Category], DecoratedCategory[Category]]:
    """Project one synthetic lexical node."""
    return projector.lexical(
        DerivationNode(TokenSpan(position, position + 1), category),
        Token(position, form),
    )


def _step(
    output: Category,
    rule: str,
    children: tuple[
        HeadProjection[ProjectedDependency[Category], DecoratedCategory[Category]],
        ...,
    ],
    selection: HeadSelection,
    *,
    projector: CCGDependencyProjector[Category] = PROJECTOR,
) -> HeadProjection[ProjectedDependency[Category], DecoratedCategory[Category]]:
    """Project one synthetic nonlexical step."""
    node = DerivationNode(
        TokenSpan(0, sum(max(len(child.heads), 1) for child in children)),
        output,
        rule=rule,
        children=tuple(range(len(children))),
    )
    return projector.step(node, rule, children, selection)


def test_the_recursive_state_numbers_result_spine_arguments_inside_out() -> None:
    projection = _lexical(TRANSITIVE, 3, "ate")
    state = projection.state

    assert state is not None
    assert state.open_slots == (1, 2)
    assert tuple(head.form for head in state.heads) == ("ate",)
    assert state.result is not None
    assert state.result.heads == state.heads
    assert state.argument is not None
    assert state.argument.heads == ()
    assert state.argument.slot == 2
    assert state.argument.governors == ((state.heads[0], 2),)
    assert state.result.argument is not None
    assert state.result.argument.slot == 1
    assert state.result.argument.governors == ((state.heads[0], 1),)


def test_application_emits_a_dependency_and_retains_the_predicate_head() -> None:
    result = _step(
        S_BACK_NP,
        "forward-application",
        (_lexical(TRANSITIVE, 0, "ate"), _lexical(NP, 1, "apples")),
        HeadSelection("functor", 0),
    )

    assert result.heads == (0,)
    assert result.state is not None
    assert result.state.open_slots == (1,)
    assert result.dependencies == (
        ProjectedDependency(
            LexicalHead(0, "ate", TRANSITIVE),
            LexicalHead(1, "apples", NP),
            2,
            "forward-application",
        ),
    )


def test_backward_application_uses_the_physical_functor_slot() -> None:
    result = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), _lexical(S_BACK_NP, 1, "slept")),
        HeadSelection("functor", 1),
    )

    assert result.heads == (1,)
    assert result.dependencies[0].predicate_form == "slept"
    assert result.dependencies[0].argument_form == "Kim"
    assert result.dependencies[0].slot == 1


def test_determiners_emit_their_dependency_but_pass_the_noun_head() -> None:
    result = _step(
        NP,
        "determiner-application",
        (_lexical(DETERMINER, 0, "the"), _lexical(N, 1, "buck")),
        HeadSelection("functor", 0),
    )

    dependency = result.dependencies[0]
    assert result.heads == (1,)
    assert dependency.predicate_position == 0
    assert dependency.predicate_form == "the"
    assert dependency.argument_position == 1
    assert dependency.argument_form == "buck"
    assert dependency.raw_category == DETERMINER
    assert dependency.slot == 1
    assert dependency.rule_id == "determiner-application"


def test_modifier_composition_passes_the_argument_head_and_keeps_both_arcs() -> None:
    composed = _step(
        COMPOSED,
        "forward-composition",
        (_lexical(MODIFIER, 0, "just"), _lexical(TRANSITIVE, 1, "passing")),
        HeadSelection("functor", 0),
    )
    applied = _step(
        S_BACK_NP,
        "forward-application",
        (composed, _lexical(NP, 2, "buck")),
        HeadSelection("functor", 0),
    )

    assert composed.heads == (1,)
    assert applied.heads == (1,)
    assert tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in applied.dependencies
    ) == (("just", 2, "passing"), ("passing", 2, "buck"))


def test_composition_preserves_the_remaining_arguments_lexical_governor() -> None:
    composed = _step(
        S_OVER_N,
        "forward-composition",
        (_lexical(S_OVER_NP, 0, "needs"), _lexical(NP_OVER_N, 1, "a")),
        HeadSelection("functor", 0),
    )
    state = composed.state

    assert state is not None
    assert state.argument is not None
    assert state.argument.governors == ((LexicalHead(1, "a", NP_OVER_N), 1),)

    applied = _step(
        S,
        "forward-application",
        (composed, _lexical(N, 2, "name")),
        HeadSelection("functor", 0),
    )

    assert applied.heads == (0,)
    assert tuple(
        (dependency.predicate_form, dependency.argument_form)
        for dependency in applied.dependencies
    ) == (("needs", "a"), ("a", "name"))


def test_generalized_composition_preserves_each_remaining_governor() -> None:
    composed = _step(
        DOUBLE_COMPOSED,
        "forward-composition-two",
        (
            _lexical(S_OVER_NP, 0, "needs"),
            _lexical(DOUBLE_NOUN_BUILDER, 1, "pairs"),
        ),
        HeadSelection("functor", 0),
    )
    state = composed.state

    assert state is not None
    assert state.open_slots == (1, 2)
    assert state.argument is not None
    assert state.argument.governors == (
        (LexicalHead(1, "pairs", DOUBLE_NOUN_BUILDER), 2),
    )

    first = _step(
        S_OVER_N,
        "forward-application-outer",
        (composed, _lexical(N, 2, "names")),
        HeadSelection("functor", 0),
    )
    second = _step(
        S,
        "forward-application-inner",
        (first, _lexical(N, 3, "labels")),
        HeadSelection("functor", 0),
    )

    assert tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in second.dependencies
    ) == (
        ("needs", 1, "pairs"),
        ("pairs", 2, "names"),
        ("pairs", 1, "labels"),
    )


def test_coordination_concatenates_head_lists_monotonically() -> None:
    first = _lexical(NP, 0, "Kim")
    second = _lexical(NP, 1, "Lee")
    result = _step(
        NP,
        "coordination",
        (first, second),
        HeadSelection("left-conjunct", 0),
    )

    assert result.heads == (0, 1)
    assert len(result.heads) >= len(first.heads)
    assert len(result.heads) >= len(second.heads)
    assert result.dependencies == ()


@pytest.mark.parametrize("output", [RAISED_NP, S])
def test_unary_type_rules_preserve_heads_without_emitting(output: Category) -> None:
    source = _lexical(NP, 0, "Kim")
    result = _step(output, "unary", (source,), HeadSelection("pass-through", 0))

    assert len(result.heads) == len(source.heads) == 1
    assert result.dependencies == ()


@settings(max_examples=10)
@given(
    st.lists(
        st.tuples(
            st.sampled_from((NP, TRANSITIVE, DETERMINER, MODIFIER)),
            st.text(max_size=12),
        ),
        min_size=50,
        max_size=50,
    )
)
def test_generated_head_states_retain_position_form_and_category(
    declarations: list[tuple[Category, str]],
) -> None:
    for position, (category, form) in enumerate(declarations):
        state = _lexical(category, position, form).state

        assert state is not None
        assert state.heads == (LexicalHead(position, form, category),)


def test_projection_values_are_frozen_hashable_and_pickleable() -> None:
    predicate = LexicalHead(0, "ate", TRANSITIVE)
    argument = LexicalHead(1, "apples", NP)
    dependency = ProjectedDependency(predicate, argument, 2, "application")
    slot = ArgumentSlot(2, ((predicate, 2),), (0, 1))
    state = DecoratedCategory(TRANSITIVE, (predicate,), slot=slot)

    assert copy(dependency) is dependency
    assert deepcopy(state) is state
    assert loads(dumps(predicate)) == predicate
    assert loads(dumps(slot)) == slot
    assert loads(dumps(dependency)) == dependency
    assert loads(dumps(state)) == state
    assert len({dependency, loads(dumps(dependency))}) == 1

    with pytest.raises(AttributeError, match="immutable"):
        dependency.__setattr__("_slot", 1)


@pytest.mark.parametrize("slot", [True, 0, -1])
def test_a_projected_dependency_requires_a_positive_integer_slot(
    slot: int,
) -> None:
    predicate = LexicalHead(0, "ate", TRANSITIVE)
    argument = LexicalHead(1, "apples", NP)

    with pytest.raises((TypeError, ValueError), match="slot"):
        ProjectedDependency(predicate, argument, slot, "application")


@pytest.mark.parametrize("index", [True, 0, -1])
def test_an_argument_slot_requires_a_positive_integer_index(index: int) -> None:
    with pytest.raises((TypeError, ValueError), match="slot"):
        ArgumentSlot(index, ())


@pytest.mark.parametrize("predicate", ["will", "seems"])
def test_subject_sharing_emits_matrix_and_embedded_subject_arcs(
    predicate: str,
) -> None:
    projector = CCGDependencyProjector(ALGEBRA, CONVENTION)
    shared = _lexical(SUBJECT_SHARING, 1, predicate, projector)
    shared_state = shared.state

    assert shared_state is not None
    assert loads(dumps(shared_state)).projection_report == CCGProjectionReport(
        "synthetic-ccgbank",
        2,
        1,
    )

    unary = _step(
        SUBJECT_SHARING,
        "unary",
        (shared,),
        HeadSelection("pass-through", 0),
        projector=projector,
    )
    unary_state = unary.state

    assert unary_state is not None
    assert unary_state.result is not None
    assert unary_state.result.argument is not None
    assert unary_state.argument is not None
    assert unary_state.argument.argument is not None
    assert unary_state.result.argument.coindex == (1, 1)
    assert unary_state.argument.argument.coindex == (1, 1)

    complemented = _step(
        S_BACK_NP,
        "forward-application",
        (unary, _lexical(S_BACK_NP, 2, "sleep", projector)),
        HeadSelection("functor", 0),
        projector=projector,
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim", projector), complemented),
        HeadSelection("functor", 1),
        projector=projector,
    )

    assert tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in completed.dependencies
    ) == (
        (predicate, 2, "sleep"),
        (predicate, 1, "Kim"),
        ("sleep", 1, "Kim"),
    )
    assert projector.report(completed) == CCGProjectionReport(
        "synthetic-ccgbank",
        2,
        1,
    )


def test_object_control_emits_the_embedded_predicate_to_controller_arc() -> None:
    projector = CCGDependencyProjector(ALGEBRA, CONVENTION)
    controlled = _step(
        CONTROL_RESULT,
        "object-application",
        (
            _lexical(OBJECT_CONTROL, 1, "permit", projector),
            _lexical(NP, 2, "managers", projector),
        ),
        HeadSelection("functor", 0),
        projector=projector,
    )
    complemented = _step(
        S_BACK_NP,
        "vp-application",
        (controlled, _lexical(S_BACK_NP, 3, "retain", projector)),
        HeadSelection("functor", 0),
        projector=projector,
    )
    completed = _step(
        S,
        "subject-application",
        (_lexical(NP, 0, "executives", projector), complemented),
        HeadSelection("functor", 1),
        projector=projector,
    )

    assert tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in completed.dependencies
    ) == (
        ("permit", 3, "managers"),
        ("permit", 2, "retain"),
        ("retain", 1, "managers"),
        ("permit", 1, "executives"),
    )


def test_composition_preserves_shared_subject_governors() -> None:
    projector = CCGDependencyProjector(ALGEBRA, CONVENTION)
    composed = _step(
        TRANSITIVE,
        "forward-composition",
        (
            _lexical(SUBJECT_SHARING, 1, "will", projector),
            _lexical(TRANSITIVE, 2, "eat", projector),
        ),
        HeadSelection("functor", 0),
        projector=projector,
    )
    object_applied = _step(
        S_BACK_NP,
        "object-application",
        (composed, _lexical(NP, 3, "apples", projector)),
        HeadSelection("functor", 0),
        projector=projector,
    )
    completed = _step(
        S,
        "subject-application",
        (_lexical(NP, 0, "Kim", projector), object_applied),
        HeadSelection("functor", 1),
        projector=projector,
    )

    assert tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in completed.dependencies
    ) == (
        ("will", 2, "eat"),
        ("eat", 2, "apples"),
        ("will", 1, "Kim"),
        ("eat", 1, "Kim"),
    )


def test_reports_distinguish_absent_empty_and_nonmatching_conventions() -> None:
    lexical_node = DerivationNode(TokenSpan(0, 1), TRANSITIVE)
    token = Token(0, "eat")
    absent = CCGDependencyProjector(ALGEBRA)
    empty = CCGDependencyProjector(
        ALGEBRA,
        CCGDependencyConvention("empty"),
    )
    unmatched = CCGDependencyProjector(
        ALGEBRA,
        CCGDependencyConvention("unmatched", (SUBJECT_TEMPLATE,)),
    )

    absent_projection = absent.lexical(lexical_node, token)
    empty_projection = empty.lexical(lexical_node, token)

    assert absent.report(absent_projection) == CCGProjectionReport(
        None,
        0,
        0,
    )
    assert empty.report(empty_projection) == CCGProjectionReport(
        "empty",
        0,
        0,
    )
    assert absent_projection.state == empty_projection.state
    assert hash(absent_projection.state) == hash(empty_projection.state)
    assert unmatched.report(
        unmatched.lexical(lexical_node, token),
    ) == CCGProjectionReport("unmatched", 1, 0)


def test_a_projection_report_keeps_the_convention_that_produced_its_state() -> None:
    source = CCGDependencyProjector(ALGEBRA, CONVENTION)
    projected = source.lexical(
        DerivationNode(TokenSpan(0, 1), SUBJECT_SHARING),
        Token(0, "will"),
    )
    unrelated = CCGDependencyProjector(
        ALGEBRA,
        CCGDependencyConvention("unrelated"),
    )

    assert unrelated.report(projected) == CCGProjectionReport(
        "synthetic-ccgbank",
        2,
        1,
    )
    assert CCGDependencyProjector(ALGEBRA).report(projected) == source.report(projected)


def test_overlapping_templates_are_refused_without_merging_their_coindices() -> None:
    overlap = CCGDependencyConvention(
        "overlap",
        (
            SUBJECT_TEMPLATE,
            CCGDependencyTemplate(
                "other-subject-sharing",
                SUBJECT_SHARING,
                SUBJECT_TEMPLATE.coindices,
            ),
        ),
    )

    with pytest.raises(ValueError, match="overlapping templates"):
        CCGDependencyProjector(ALGEBRA, overlap)


def test_invalid_template_paths_are_refused_at_projector_construction() -> None:
    invalid = CCGDependencyConvention(
        "invalid-path",
        (
            CCGDependencyTemplate(
                "invalid-path",
                SUBJECT_SHARING,
                (
                    CoindexedPath(("argument", "argument", "argument"), 1),
                    CoindexedPath(("result", "argument", "argument"), 1),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not name an argument node"):
        CCGDependencyProjector(ALGEBRA, invalid)


def test_nonequivalent_coindexed_arguments_are_refused_at_construction() -> None:
    incompatible = CCGDependencyConvention(
        "incompatible",
        (
            CCGDependencyTemplate(
                "incompatible",
                OBJECT_CONTROL,
                (
                    CoindexedPath(("argument",), 1),
                    CoindexedPath(("result", "argument"), 1),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="nonequivalent argument categories"):
        CCGDependencyProjector(ALGEBRA, incompatible)


def _arcs(
    projection: HeadProjection[
        ProjectedDependency[Category], DecoratedCategory[Category]
    ],
) -> tuple[tuple[str, int, str], ...]:
    """Return each dependency as predicate form, slot and argument form."""
    return tuple(
        (dependency.predicate_form, dependency.slot, dependency.argument_form)
        for dependency in projection.dependencies
    )


SUBJECT_PATHS: Final = (
    CoindexedPath(("result", "argument"), 7),
    CoindexedPath(("argument", "argument"), 7),
)
MARKED_AUXILIARY: Final = _functor(S_BACK_NP, S_BACK_NP, coindexed=SUBJECT_PATHS)
MARKED_ADVERB: Final = _functor(
    S_BACK_NP, S_BACK_NP, head_sharing=True, coindexed=SUBJECT_PATHS
)
NOUN_POSTMODIFIER: Final = _functor(NP, NP, head_sharing=True)
RELATIVE_RESULT: Final = _functor(NP, NP, head_sharing=True)
RELATIVE: Final = _functor(
    RELATIVE_RESULT,
    S_OVER_NP,
    coindexed=(
        CoindexedPath(("result", "argument"), 3),
        CoindexedPath(("argument", "argument"), 3),
    ),
)


def test_a_type_changed_adjunct_keeps_its_subject_binding() -> None:
    adjunct = _step(
        MODIFIER,
        "type-changing",
        (_lexical(S_BACK_NP, 2, "smiling"),),
        HeadSelection("pass-through", 0),
    )
    modified = _step(
        S_BACK_NP,
        "backward-application",
        (_lexical(S_BACK_NP, 1, "left"), adjunct),
        HeadSelection("functor", 1),
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), modified),
        HeadSelection("functor", 1),
    )

    assert modified.heads == (1,)
    assert modified.dependencies == ()
    assert _arcs(completed) == (("left", 1, "Kim"), ("smiling", 1, "Kim"))


def test_a_type_changed_noun_postmodifier_takes_the_noun_as_subject() -> None:
    postmodifier = _step(
        NOUN_POSTMODIFIER,
        "type-changing",
        (_lexical(S_BACK_NP, 1, "seen"),),
        HeadSelection("pass-through", 0),
    )
    completed = _step(
        NP,
        "backward-application",
        (_lexical(NP, 0, "dogs"), postmodifier),
        HeadSelection("functor", 1),
    )

    assert completed.heads == (0,)
    assert _arcs(completed) == (("seen", 1, "dogs"),)


def test_category_coindices_come_from_the_algebra_without_a_convention() -> None:
    complemented = _step(
        S_BACK_NP,
        "forward-application",
        (_lexical(MARKED_AUXILIARY, 1, "will"), _lexical(S_BACK_NP, 2, "sleep")),
        HeadSelection("functor", 0),
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), complemented),
        HeadSelection("functor", 1),
    )

    assert complemented.heads == (1,)
    assert _arcs(completed) == (
        ("will", 2, "sleep"),
        ("will", 1, "Kim"),
        ("sleep", 1, "Kim"),
    )


def test_a_coindexed_modifier_still_passes_its_argument_head() -> None:
    adverb = _lexical(MARKED_ADVERB, 1, "soon")
    adverb_state = adverb.state

    assert adverb_state is not None
    assert adverb_state.shares_head
    assert adverb_state.result is not None
    assert adverb_state.result.argument is not None
    assert adverb_state.result.argument.governors == ()

    modified = _step(
        S_BACK_NP,
        "forward-application",
        (adverb, _lexical(S_BACK_NP, 2, "sleep")),
        HeadSelection("functor", 0),
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), modified),
        HeadSelection("functor", 1),
    )

    assert modified.heads == (2,)
    assert _arcs(completed) == (("soon", 2, "sleep"), ("sleep", 1, "Kim"))


def test_an_extracted_argument_keeps_each_governors_own_slot() -> None:
    raised = _step(
        RAISED_NP,
        "type-raising",
        (_lexical(NP, 2, "I"),),
        HeadSelection("pass-through", 0),
    )
    clause = _step(
        S_OVER_NP,
        "forward-composition",
        (raised, _lexical(TRANSITIVE, 3, "saw")),
        HeadSelection("functor", 0),
    )
    relative = _step(
        RELATIVE_RESULT,
        "forward-application",
        (_lexical(RELATIVE, 1, "that"), clause),
        HeadSelection("functor", 0),
    )
    completed = _step(
        NP,
        "backward-application",
        (_lexical(NP, 0, "man"), relative),
        HeadSelection("functor", 1),
    )

    assert raised.heads == (2,)
    assert clause.heads == (3,)
    assert _arcs(clause) == (("saw", 1, "I"),)
    assert completed.heads == (0,)
    assert _arcs(completed) == (
        ("saw", 1, "I"),
        ("that", 2, "saw"),
        ("that", 1, "man"),
        ("saw", 2, "man"),
    )


def test_coordinated_functors_share_their_arguments() -> None:
    coordinated = _step(
        TRANSITIVE,
        "coordination",
        (_lexical(TRANSITIVE, 1, "saw"), _lexical(TRANSITIVE, 3, "heard")),
        HeadSelection("left-conjunct", 0),
    )
    applied = _step(
        S_BACK_NP,
        "forward-application",
        (coordinated, _lexical(NP, 4, "Lee")),
        HeadSelection("functor", 0),
    )

    assert coordinated.heads == (1, 3)
    assert applied.heads == (1, 3)
    assert _arcs(applied) == (("saw", 2, "Lee"), ("heard", 2, "Lee"))


def test_a_dependency_reached_twice_is_emitted_once() -> None:
    projector = CCGDependencyProjector(ALGEBRA, CONVENTION)
    controlled = _step(
        CONTROL_RESULT,
        "object-application",
        (
            _lexical(OBJECT_CONTROL, 1, "permit", projector),
            _lexical(NP, 2, "managers", projector),
        ),
        HeadSelection("functor", 0),
        projector=projector,
    )
    complemented = _step(
        S_BACK_NP,
        "vp-application",
        (controlled, _lexical(S_BACK_NP, 3, "retain", projector)),
        HeadSelection("functor", 0),
        projector=projector,
    )

    assert _arcs(complemented).count(("permit", 3, "managers")) == 1


@pytest.mark.parametrize(
    "governors",
    [((LexicalHead(0, "ate", TRANSITIVE), 0),), ((LexicalHead(0, "ate", N), -1),)],
)
def test_an_argument_slot_requires_positive_governing_slots(
    governors: tuple[tuple[LexicalHead[Category], int], ...],
) -> None:
    with pytest.raises(ValueError, match="governing slot"):
        ArgumentSlot(1, governors)


@pytest.mark.parametrize(
    ("coindex", "error"),
    [((-1, 1), ValueError), ((0, 0), ValueError), ((True, 1), TypeError)],
)
def test_an_argument_slot_requires_a_valid_coindex(
    coindex: tuple[int, int],
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="coindex"):
        ArgumentSlot(1, (), coindex)


def test_an_atomic_decorated_category_cannot_share_a_head() -> None:
    with pytest.raises(ValueError, match="cannot share"):
        DecoratedCategory(NP, (), shares_head=True)


def test_only_a_matching_result_makes_a_category_type_raised() -> None:
    raised = _step(
        RAISED_NP,
        "type-raising",
        (_lexical(NP, 0, "Kim"),),
        HeadSelection("pass-through", 0),
    )
    changed = _step(
        _functor(N, S_BACK_NP),
        "type-changing",
        (_lexical(NP, 0, "Kim"),),
        HeadSelection("pass-through", 0),
    )

    assert raised.state is not None
    assert raised.state.shares_head
    assert changed.state is not None
    assert not changed.state.shares_head


def test_a_routed_binding_lands_on_the_innermost_equivalent_argument() -> None:
    changed = _step(
        _functor(S, TRANSITIVE),
        "type-changing",
        (_lexical(S_BACK_NP, 0, "smiled"),),
        HeadSelection("pass-through", 0),
    )
    state = changed.state

    assert state is not None
    assert state.argument is not None
    assert state.argument.governors == ()
    assert state.argument.argument is not None
    assert state.argument.argument.governors == ()
    assert state.argument.result is not None
    assert state.argument.result.argument is not None
    assert state.argument.result.argument.governors == (
        (LexicalHead(0, "smiled", S_BACK_NP), 1),
    )


def test_an_unroutable_binding_stays_on_the_restructured_argument() -> None:
    noun_taking = _functor(S, N)
    changed = _step(
        _functor(S_BACK_NP, noun_taking),
        "type-changing",
        (_lexical(TRANSITIVE, 0, "ate"),),
        HeadSelection("pass-through", 0),
    )
    applied = _step(
        S_BACK_NP,
        "forward-application",
        (changed, _lexical(noun_taking, 1, "needs")),
        HeadSelection("functor", 0),
    )

    assert _arcs(applied) == (("ate", 2, "needs"),)


def test_a_raised_argument_fills_the_slot_it_composes_into() -> None:
    raised = _step(
        _functor(S_BACK_NP, TRANSITIVE),
        "type-raising",
        (_lexical(NP, 2, "Lee"),),
        HeadSelection("pass-through", 0),
    )
    builder = _functor(TRANSITIVE, N)
    composed = _step(
        _functor(S_BACK_NP, N),
        "forward-composition",
        (raised, _lexical(builder, 1, "gave")),
        HeadSelection("functor", 0),
    )
    state = composed.state

    assert _arcs(composed) == (("gave", 2, "Lee"),)
    assert state is not None
    assert state.result is not None
    assert state.result.argument is not None
    assert state.result.argument.heads == (LexicalHead(2, "Lee", NP),)


def test_a_governed_argument_raised_by_a_relabel_is_governed_once() -> None:
    raised_object = _functor(S_BACK_NP, TRANSITIVE)
    changed = _step(
        _functor(S_BACK_NP, raised_object),
        "type-changing",
        (_lexical(TRANSITIVE, 1, "saw"),),
        HeadSelection("pass-through", 0),
    )
    state = changed.state
    raised = _step(
        raised_object,
        "type-raising",
        (_lexical(NP, 2, "Lee"),),
        HeadSelection("pass-through", 0),
    )
    applied = _step(
        S_BACK_NP,
        "forward-application",
        (changed, raised),
        HeadSelection("functor", 0),
    )

    assert state is not None
    assert state.argument is not None
    assert state.argument.governors == ((LexicalHead(1, "saw", TRANSITIVE), 2),)
    assert state.argument.argument is not None
    assert state.argument.argument.argument is not None
    assert state.argument.argument.argument.governors == ()
    assert _arcs(applied) == (("saw", 2, "Lee"),)


def test_a_coindex_survives_a_step_that_fills_its_node() -> None:
    negated = _step(
        MARKED_AUXILIARY,
        "backward-composition",
        (_lexical(MARKED_AUXILIARY, 1, "will"), _lexical(MARKED_ADVERB, 2, "not")),
        HeadSelection("functor", 1),
    )
    state = negated.state
    complemented = _step(
        S_BACK_NP,
        "forward-application",
        (negated, _lexical(S_BACK_NP, 3, "sleep")),
        HeadSelection("functor", 0),
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), complemented),
        HeadSelection("functor", 1),
    )

    assert state is not None
    assert state.result is not None
    assert state.result.argument is not None
    assert state.result.argument.coindex == (1, 7)
    assert _arcs(completed) == (
        ("not", 2, "will"),
        ("will", 2, "sleep"),
        ("will", 1, "Kim"),
        ("sleep", 1, "Kim"),
    )


def test_a_raised_subject_composed_into_an_auxiliary_reaches_its_complement() -> None:
    raised = _step(
        RAISED_NP,
        "type-raising",
        (_lexical(NP, 0, "Kim"),),
        HeadSelection("pass-through", 0),
    )
    clause = _step(
        _functor(S, S_BACK_NP),
        "forward-composition",
        (raised, _lexical(MARKED_AUXILIARY, 1, "will")),
        HeadSelection("functor", 0),
    )
    completed = _step(
        S,
        "forward-application",
        (clause, _lexical(S_BACK_NP, 2, "sleep")),
        HeadSelection("functor", 0),
    )

    assert _arcs(completed) == (
        ("will", 1, "Kim"),
        ("will", 2, "sleep"),
        ("sleep", 1, "Kim"),
    )


def test_a_coindex_inside_the_result_of_an_argument_is_unified() -> None:
    controller = _functor(
        S_BACK_NP,
        TRANSITIVE,
        coindexed=(
            CoindexedPath(("result", "argument"), 4),
            CoindexedPath(("argument", "result", "argument"), 4),
        ),
    )
    complemented = _step(
        S_BACK_NP,
        "forward-application",
        (_lexical(controller, 1, "tried"), _lexical(TRANSITIVE, 2, "saw")),
        HeadSelection("functor", 0),
    )
    completed = _step(
        S,
        "backward-application",
        (_lexical(NP, 0, "Kim"), complemented),
        HeadSelection("functor", 1),
    )

    assert _arcs(completed) == (
        ("tried", 2, "saw"),
        ("tried", 1, "Kim"),
        ("saw", 1, "Kim"),
    )


def test_a_functor_whose_argument_has_no_slot_is_refused() -> None:
    head = LexicalHead(0, "odd", S_OVER_NP)
    unnumbered = HeadProjection(
        (0,),
        (),
        DecoratedCategory(
            S_OVER_NP, (head,), DecoratedCategory(S, (head,)), DecoratedCategory(NP, ())
        ),
    )

    with pytest.raises(ValueError, match="unnumbered argument"):
        _step(
            S,
            "forward-application",
            (unnumbered, _lexical(NP, 1, "Kim")),
            HeadSelection("functor", 0),
        )


def test_a_head_passing_functor_yields_only_its_argument_daughters_heads() -> None:
    noun_modifier = _functor(NP, NP, head_sharing=True)
    which = LexicalHead(1, "which", noun_modifier)
    filler = LexicalHead(4, "dollars", NP)
    filled = HeadProjection(
        (1,),
        (),
        DecoratedCategory(
            noun_modifier,
            (which,),
            DecoratedCategory(NP, (which,)),
            DecoratedCategory(NP, (filler,), slot=ArgumentSlot(1, ((which, 1),))),
            shares_head=True,
        ),
    )
    modified = _step(
        NP,
        "backward-application",
        (_lexical(NP, 0, "time"), filled),
        HeadSelection("functor", 1),
    )
    completed = _step(
        S,
        "forward-application",
        (_lexical(S_OVER_NP, 5, "for"), modified),
        HeadSelection("functor", 0),
    )

    assert modified.heads == (0,)
    assert modified.state is not None
    assert tuple(head.form for head in modified.state.heads) == ("time",)
    assert [arc for arc in _arcs(completed) if arc[0] == "for"] == [("for", 1, "time")]
