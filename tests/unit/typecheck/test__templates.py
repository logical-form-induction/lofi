import copy
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.language import language

from lofi.deduce import CoverGoal
from lofi.lexicon import Clamp, Entry
from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    LanguageAssembly,
    LanguageIso,
)
from lofi.terms import Identity
from lofi.typecheck import (
    CategoryProjection,
    ClampedGoal,
    ConstraintTemplate,
    FixedSpan,
    FormPattern,
    ResolvedConstraint,
    SpanAccessor,
    TokenPosition,
    TypeTemplate,
    UnknownPrimitiveError,
    WholeInput,
    read_templates,
    type_template,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.terms import Term
    from lofi.typecheck import TypeShape


TOKENS: Final = ("jo", "saw", "which", "thing", "and", "which", "thing")
"""The committed input every constraint below is applied to."""


@pytest.fixture
def types() -> LanguageAssembly:
    """Return the shipped language whose store holds semantic types."""
    return language()


def arrow(
    declared: LanguageAssembly,
    domain: Term,
    codomain: Term,
) -> Term:
    """Return the function type from one type to another."""
    return declared.store.op(
        declared.store.signature.by_name(declared.function_type),
        domain,
        codomain,
    )


def primitive(declared: LanguageAssembly, name: str) -> Term:
    """Return the primitive type a name declares."""
    return declared.read_type(name)


def test_a_primitive_template_resolves_to_the_primitive(
    types: LanguageAssembly,
) -> None:
    assert type_template("e").resolve(types) is primitive(types, "e")


def test_a_two_member_template_resolves_to_a_function_type(
    types: LanguageAssembly,
) -> None:
    built = type_template(["e", "t"]).resolve(types)

    assert built is arrow(types, primitive(types, "e"), primitive(types, "t"))


def test_a_template_of_three_members_nests_to_the_right(
    types: LanguageAssembly,
) -> None:
    built = type_template(["e", "e", "t"]).resolve(types)
    expected = arrow(
        types,
        primitive(types, "e"),
        arrow(types, primitive(types, "e"), primitive(types, "t")),
    )

    assert built is expected
    assert built is not arrow(
        types,
        arrow(types, primitive(types, "e"), primitive(types, "e")),
        primitive(types, "t"),
    )


def test_a_nested_list_becomes_a_subtree(types: LanguageAssembly) -> None:
    built = type_template([["e", "t"], ["e", "t"]]).resolve(types)
    predicate = arrow(types, primitive(types, "e"), primitive(types, "t"))

    assert built is arrow(types, predicate, predicate)


def test_the_shape_records_what_the_template_was_written_as() -> None:
    assert type_template("e").shape == "e"
    assert type_template(["e", ["v", "t"]]).shape == ("e", ("v", "t"))


def test_a_template_naming_an_undeclared_primitive_is_refused(
    types: LanguageAssembly,
) -> None:
    with pytest.raises(UnknownPrimitiveError) as raised:
        type_template(["s", "t"]).resolve(types)

    assert raised.value.primitive == "s"
    assert raised.value.declared == ("e", "t", "v")


def test_a_template_naming_a_variable_operator_is_refused(
    types: LanguageAssembly,
) -> None:
    with pytest.raises(UnknownPrimitiveError) as raised:
        type_template("#fvar").resolve(types)

    assert raised.value.primitive == "#fvar"
    assert "#fvar" not in raised.value.declared


def test_a_malformed_template_is_refused() -> None:
    with pytest.raises(ValueError, match="has 1 members"):
        type_template(["e"])

    with pytest.raises(TypeError, match="neither a primitive nor a list"):
        type_template(7)

    with pytest.raises(ValueError, match="the type at the template is empty"):
        type_template("")

    with pytest.raises(TypeError, match=r"the type at the template\[0\]"):
        type_template([None, "t"])


def test_a_template_built_directly_validates_its_shape() -> None:
    with pytest.raises(ValueError, match="has 1 members"):
        TypeTemplate(("e",))

    with pytest.raises(TypeError, match="neither a name nor a tuple"):
        TypeTemplate(cast("str", ["e", "t"]))

    with pytest.raises(ValueError, match="a template's primitive"):
        TypeTemplate("")


def test_a_malformed_nested_member_is_refused_rather_than_flattened(
    types: LanguageAssembly,
) -> None:
    shallow: TypeShape = ("e", ("t",))
    deep: TypeShape = ("e", ("t", ("e",)))

    with pytest.raises(ValueError, match="has 1 members"):
        TypeTemplate(shallow)

    with pytest.raises(ValueError, match="has 1 members"):
        TypeTemplate(deep)

    with pytest.raises(TypeError, match="neither a name nor a tuple"):
        TypeTemplate(cast("TypeShape", ("e", 7)))

    with pytest.raises(TypeError, match="neither a name nor a tuple"):
        TypeTemplate(cast("TypeShape", ("e", ("t", ["e", "t"]))))

    with pytest.raises(ValueError, match="a template's primitive"):
        TypeTemplate(("e", ("", "t")))

    well_formed = TypeTemplate(("e", ("v", "t")))

    assert well_formed.shape == ("e", ("v", "t"))
    assert well_formed.resolve(types) is arrow(
        types,
        primitive(types, "e"),
        arrow(types, primitive(types, "v"), primitive(types, "t")),
    )


def test_a_malformed_nested_member_is_refused_as_data_is_read() -> None:
    with pytest.raises(ValueError, match=r"the template\[1\] has 1 members"):
        type_template(["e", ["t"]])

    with pytest.raises(ValueError, match=r"the template\[1\]\[1\] has 1 members"):
        type_template(["e", ["t", ["e"]]])

    with pytest.raises(TypeError, match=r"the type at the template\[1\]\[0\]"):
        type_template(["e", [None, "t"]])

    with pytest.raises(ValueError, match=r"the template\[1\]\[0\] is empty"):
        type_template(["e", ["", "t"]])


def test_named_templates_are_read_from_decoded_data() -> None:
    declared: dict[str, JsonValue] = {"noun": ["e", "t"], "entity": "e"}
    found = read_templates(declared)

    assert found == {
        "noun": TypeTemplate(("e", "t")),
        "entity": TypeTemplate("e"),
    }
    assert list(found) == ["noun", "entity"]


def test_a_blank_name_in_a_template_mapping_is_refused() -> None:
    with pytest.raises(ValueError, match="a name in the lexicon"):
        read_templates({"": "e"}, within="the lexicon")

    with pytest.raises(ValueError, match="'noun' in the lexicon"):
        read_templates({"noun": ["e"]}, within="the lexicon")


def test_a_constraint_resolves_to_the_clamp_a_caller_writes_today(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(FixedSpan(0, 2), (type_template(["e", "t"]),))
    resolved = template.resolve(types)
    permitted = arrow(types, primitive(types, "e"), primitive(types, "t"))

    assert resolved.clamps(TOKENS) == (Clamp((0, 2), (permitted,)),)
    assert resolved.permitted == frozenset({permitted})
    assert not resolved.inverted


def test_an_accessed_clamp_admits_what_the_literal_clamp_admits(
    types: LanguageAssembly,
) -> None:
    predicate = arrow(types, primitive(types, "e"), primitive(types, "t"))
    entity = primitive(types, "e")
    resolved = ConstraintTemplate(
        FixedSpan(0, 1), (type_template(["e", "t"]),)
    ).resolve(types)
    clamp = resolved.clamps(TOKENS)[0]

    assert clamp.covers(0)
    assert not clamp.covers(1)
    assert clamp.admits(Entry("jo", predicate), Identity(), Identity())
    assert not clamp.admits(Entry("jo", entity), Identity(), Identity())


def test_each_fixed_shape_clamps_the_span_it_covers(
    types: LanguageAssembly,
) -> None:
    truth = type_template("t")
    covered = {
        TokenPosition(1): (1, 2),
        FixedSpan(1, 7): (1, 7),
        WholeInput(): (0, 7),
    }

    for accessor, span in covered.items():
        resolved = ConstraintTemplate(accessor, (truth,)).resolve(types)

        assert resolved.clamps(TOKENS) == (Clamp(span, (primitive(types, "t"),)),)


def test_a_constraint_reads_back_everything_it_was_declared_over(
    types: LanguageAssembly,
) -> None:
    truth = type_template("t")
    accessors: tuple[SpanAccessor, ...] = (
        TokenPosition(1),
        FixedSpan(1, 7),
        WholeInput(),
        FormPattern(("which", None)),
    )

    for accessor in accessors:
        for inverted in (False, True):
            template = ConstraintTemplate(accessor, (truth,), inverted=inverted)
            resolved = template.resolve(types)

            assert template.accessor == accessor
            assert template.permitted == frozenset({truth})
            assert template.inverted is inverted
            assert resolved.accessor == accessor
            assert resolved.permitted == frozenset({primitive(types, "t")})
            assert resolved.inverted is inverted


def test_a_pattern_constraint_clamps_every_span_it_matches(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(FormPattern(("which", None)), (type_template("e"),))
    fixed = (TokenPosition(1), FixedSpan(1, 7), WholeInput())
    clamps = template.resolve(types).clamps(TOKENS)
    matched = frozenset(clamp.span for clamp in clamps)
    covered = frozenset(span for accessor in fixed for span in accessor.spans(TOKENS))

    assert tuple(clamp.span for clamp in clamps) == ((2, 4), (5, 7))
    assert not matched & covered


def test_the_clamp_family_takes_the_clamps_an_accessor_produces(
    types: LanguageAssembly,
) -> None:
    resolved = ConstraintTemplate(
        FormPattern(("which", None)), (type_template("e"),)
    ).resolve(types)
    clamps = resolved.clamps(TOKENS)
    goal = ClampedGoal(CoverGoal(1), clamps, CategoryProjection(), Identity())

    assert goal.clamps == clamps
    assert tuple(clamp.span for clamp in goal.clamps) == ((2, 4), (5, 7))
    assert goal.clamps[0].category == frozenset({primitive(types, "e")})


def _arrowed_language() -> LanguageAssembly:
    base = language()
    source = LanguageIso().backward(base)
    return LanguageIso().forward(
        source.with_(
            name="arrowed-types",
            function_type="=>",
            constructors=(
                ConstructorSpec(
                    name="=>",
                    kind=("Ty", "Ty", "Ty"),
                    notation=InfixConstructorNotationSpec(
                        spelling="=>", precedence=1, associativity="right"
                    ),
                ),
            ),
            constants=tuple(
                constant.with_(type=constant.type.replace("->", "=>"))
                for constant in source.constants
            ),
        )
    )


def test_a_constraint_uses_the_declared_function_type_constructor() -> None:
    declared = _arrowed_language()
    template = ConstraintTemplate(FixedSpan(0, 2), (type_template(["e", "t"]),))
    expected = declared.read_type("e => t")
    resolved = template.resolve(declared)

    assert resolved.permitted == frozenset({expected})
    assert resolved.clamps(TOKENS) == (Clamp((0, 2), (expected,)),)


def test_an_inverted_constraint_carries_its_direction_into_its_clamps(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(FixedSpan(0, 2), (type_template("e"),), inverted=True)
    resolved = template.resolve(types)
    clamps = resolved.clamps(TOKENS)

    assert template.inverted
    assert resolved.inverted
    assert all(clamp.negate for clamp in clamps)


def test_a_constraint_over_several_types_permits_all_of_them(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(
        FixedSpan(0, 2), (type_template("e"), type_template(["e", "t"]))
    )
    resolved = template.resolve(types)

    assert template.permitted == {type_template("e"), type_template(["e", "t"])}
    assert resolved.permitted == {
        primitive(types, "e"),
        arrow(types, primitive(types, "e"), primitive(types, "t")),
    }
    assert resolved.clamps(TOKENS)[0].category == resolved.permitted


def test_the_same_constraint_serves_inputs_of_different_lengths(
    types: LanguageAssembly,
) -> None:
    resolved = ConstraintTemplate(WholeInput(), (type_template("t"),)).resolve(types)

    assert resolved.clamps(TOKENS)[0].span == (0, 7)
    assert resolved.clamps(TOKENS[:3])[0].span == (0, 3)


def test_an_absent_primitive_is_refused_when_the_constraint_resolves(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(FixedSpan(0, 2), (type_template(["s", "t"]),))

    with pytest.raises(UnknownPrimitiveError, match="'s'"):
        template.resolve(types)


def test_a_constraint_permitting_nothing_is_refused() -> None:
    empty: tuple[Term, ...] = ()

    with pytest.raises(ValueError, match="constraint template permits no type"):
        ConstraintTemplate(WholeInput(), ())

    with pytest.raises(ValueError, match="resolved constraint permits no type"):
        ResolvedConstraint(WholeInput(), empty)


def test_a_constraint_reaching_past_its_input_is_refused(
    types: LanguageAssembly,
) -> None:
    resolved = ConstraintTemplate(FixedSpan(5, 8), (type_template("t"),)).resolve(types)

    with pytest.raises(ValueError, match="reaches past"):
        resolved.clamps(TOKENS)


def test_a_pattern_matching_nothing_clamps_nothing(
    types: LanguageAssembly,
) -> None:
    resolved = ConstraintTemplate(
        FormPattern(("nowhere",)), (type_template("t"),)
    ).resolve(types)

    assert resolved.clamps(TOKENS) == ()


def test_templates_compare_on_what_they_declare(
    types: LanguageAssembly,
) -> None:
    truth = type_template("t")
    entity = type_template("e")

    assert TypeTemplate("t") == truth
    assert TypeTemplate("t") != entity
    assert TypeTemplate("t") != "t"
    assert ConstraintTemplate(WholeInput(), (truth,)) == ConstraintTemplate(
        WholeInput(), (truth,)
    )
    assert ConstraintTemplate(WholeInput(), (truth,)) != ConstraintTemplate(
        WholeInput(), (entity,)
    )
    assert ConstraintTemplate(WholeInput(), (truth,)) != ConstraintTemplate(
        FixedSpan(0, 7), (truth,)
    )
    assert ConstraintTemplate(WholeInput(), (truth,)) != ConstraintTemplate(
        WholeInput(), (truth,), inverted=True
    )
    assert ConstraintTemplate(WholeInput(), (truth,)) != truth

    resolved = ConstraintTemplate(WholeInput(), (truth,)).resolve(types)

    assert resolved == ConstraintTemplate(WholeInput(), (truth,)).resolve(types)
    assert resolved != ConstraintTemplate(WholeInput(), (entity,)).resolve(types)
    assert resolved != ConstraintTemplate(
        WholeInput(), (truth,), inverted=True
    ).resolve(types)
    assert resolved != ConstraintTemplate(FixedSpan(0, 7), (truth,)).resolve(types)
    assert resolved != truth


def test_templates_hash_with_what_they_declare(
    types: LanguageAssembly,
) -> None:
    truth = type_template("t")
    resolved = ConstraintTemplate(WholeInput(), (truth,)).resolve(types)

    assert len({TypeTemplate("t"), truth, type_template("e")}) == 2
    assert (
        len(
            {
                ConstraintTemplate(WholeInput(), (truth,)),
                ConstraintTemplate(WholeInput(), (truth,)),
                ConstraintTemplate(WholeInput(), (truth,), inverted=True),
            }
        )
        == 2
    )
    assert hash(resolved) == hash(
        ConstraintTemplate(WholeInput(), (truth,)).resolve(types)
    )


def test_templates_render_as_what_they_declare(
    types: LanguageAssembly,
) -> None:
    truth = type_template("t")

    assert repr(TypeTemplate(("e", "t"))) == "TypeTemplate(('e', 't'))"
    assert repr(ConstraintTemplate(WholeInput(), (truth,))) == (
        "ConstraintTemplate(WholeInput(), permitted=1, inverted=False)"
    )
    assert repr(ConstraintTemplate(WholeInput(), (truth,)).resolve(types)) == (
        "ResolvedConstraint(WholeInput(), permitted=1, inverted=False)"
    )


def test_templates_are_immutable_and_copy_to_themselves(
    types: LanguageAssembly,
) -> None:
    template = ConstraintTemplate(WholeInput(), (type_template("t"),))
    resolved = template.resolve(types)

    with pytest.raises(AttributeError, match="constraint template"):
        template.__setattr__("_inverted", True)

    with pytest.raises(AttributeError, match="resolved constraint"):
        resolved.__setattr__("_inverted", True)

    with pytest.raises(AttributeError, match="type template"):
        type_template("t").__setattr__("_shape", "e")

    assert copy.copy(template) is template
    assert copy.deepcopy(resolved) is resolved
