from importlib.metadata import EntryPoint
from json import dumps, loads

import pytest
from tests.support.installed import install

from lofi.lf import LogicProfileError
from lofi.spec import (
    ConstructorSpec,
    InfixConstructorNotationSpec,
    MixfixConstructorNotationSpec,
    PrefixConstructorNotationSpec,
)
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantInferenceError,
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
    LogicSpec,
    canonical_bytes,
    registered_language,
)
from lofi.spec.provers import LabelingProverSpec, TruthTableProverSpec


@pytest.fixture
def logical_constant_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the packaged semantics without relying on editable metadata."""
    install(
        monkeypatch,
        "logical_constant",
        (
            EntryPoint(
                name="most",
                value="lofi.lf._semantics:MOST",
                group="lofi.logical_constant",
            ),
        ),
    )


def test_lowering_builds_one_many_sorted_store_and_round_trips() -> None:
    source = registered_language("neo-davidsonian")
    converter = LanguageIso()
    language = converter.forward(source)

    assert language.store.signature.sorts == (
        source.term_sort,
        source.type_sort,
        source.category_sort,
    )
    assert language.type_of(language.read("\\x. x")).op.sort == language.type_sort
    assert language.checker.store is language.store
    assert canonical_bytes(converter.backward(language)) == canonical_bytes(source)


def test_logic_owns_its_prover_labeling_budget_and_variable_inventory() -> None:
    source = registered_language("propositional").with_(
        logic=LogicSpec(
            equality=False,
            variables=("u", "v", "event"),
            prover=TruthTableProverSpec(atoms=7),
            labeling=LabelingProverSpec(),
        )
    )
    language = LanguageIso().forward(source)

    assert language.profile.variables == ("u", "v", "event")
    assert source.logic.prover.atoms == 7
    assert source.logic.labeling.budget.calls_per_step is None


def test_the_declared_variable_inventory_bounds_names_drawn_by_the_printer() -> None:
    source = registered_language("propositional").with_(
        logic=LogicSpec(equality=False, variables=("u",))
    )
    language = LanguageIso().forward(source)

    assert language.print(language.read(r"\x. x")) == r"\u. u"

    with pytest.raises(ValueError, match="variable inventory is exhausted"):
        language.print(language.read(r"\x. \y. x"))


def test_world_and_index_types_are_explicit_atomic_resources() -> None:
    source = registered_language("propositional").with_(
        atomic_types=(
            AtomicTypeSpec(name="e"),
            AtomicTypeSpec(name="v", kind="event"),
            AtomicTypeSpec(name="t", kind="proposition"),
            AtomicTypeSpec(name="s", kind="world"),
            AtomicTypeSpec(name="i", kind="index"),
        )
    )
    profile = LanguageIso().forward(source).profile

    assert profile.world_types == frozenset(("s",))
    assert profile.proposition_types == frozenset(("t",))
    assert profile.index_types == frozenset(("i",))
    assert "intensional" not in LogicSpec.__field_specs__


def test_logic_owns_the_semantics_negation_and_label_map() -> None:
    source = registered_language("propositional")
    profile = (
        LanguageIso()
        .forward(
            source.with_(
                logic=source.logic.with_(
                    semantics="kripke",
                    negation="not",
                    label_map="strict",
                )
            )
        )
        .profile
    )

    assert profile.semantics == "kripke"
    assert profile.negation == "not"
    assert profile.label_map.name == "strict"


def test_first_order_logic_refuses_a_binder_over_relations_by_field() -> None:
    source = registered_language("neo-davidsonian")
    relation_binder = ConstantSpec(
        name="every_relation",
        type="((e -> t) -> t) -> t",
        classes=("quantifier", "universal"),
        notation=BinderNotationSpec(
            ascii="every_relation",
            unicode="every_relation",
            bodies=1,
        ),
    )

    with pytest.raises(LogicProfileError) as refused:
        LanguageIso().forward(
            source.with_(constants=(*source.constants, relation_binder))
        )

    assert refused.value.field == "order"


def test_logic_with_equality_disabled_refuses_an_equality_operator() -> None:
    source = registered_language("neo-davidsonian")

    with pytest.raises(LogicProfileError) as refused:
        LanguageIso().forward(source.with_(logic=source.logic.with_(equality=False)))

    assert refused.value.field == "equality"


def test_logic_with_functions_disabled_refuses_a_function_operator() -> None:
    source = registered_language("propositional")
    function = ConstantSpec(name="successor", type="e -> e")

    with pytest.raises(LogicProfileError) as refused:
        LanguageIso().forward(
            source.with_(
                constants=(*source.constants, function),
                logic=source.logic.with_(functions=False),
            )
        )

    assert refused.value.field == "functions"


def test_canonical_bytes_are_the_eager_language_envelope_bytes() -> None:
    source = registered_language("neo-davidsonian")
    sorted_keys = dumps(
        loads(source.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    ).encode()

    assert canonical_bytes(source) == source.compile_envelope.canonical_declaration
    assert canonical_bytes(source) == sorted_keys
    assert (
        canonical_bytes(LanguageDecl.model_validate_json(canonical_bytes(source)))
        == sorted_keys
    )


def test_lowering_builds_one_choice_index_and_one_codec_per_sort() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    choices = language.codec.choices

    assert choices.signature is language.store.signature
    assert choices.depth_bounds == {"Ty": 4, "Tm": 8, "Cat": 4}
    assert language.codec.index is language.codec.at("Ty").index
    assert language.codec.index is language.codec.at("Tm").index
    assert language.codec.at("Ty") is language.codec.at("Ty")
    assert language.codec.at("Tm") is language.codec.at("Tm")
    assert language.codec.at("Ty") is not language.codec.at("Tm")
    assert language.codec.at("Ty").capacity == 15
    assert language.codec.at("Tm").capacity == 24
    assert choices.index("#const[Ty]", 0) != choices.index("#const[Ty]", 1)


def test_declared_operator_families_do_not_define_their_structural_roles() -> None:
    source = registered_language("neo-davidsonian").with_(
        binder_family="bind",
        constant_family="const",
        variable_family="var",
    )
    language = LanguageIso().forward(source)
    signature = language.store.signature
    term = language.read("\\x. x")

    assert signature.constant(language.type_sort).family == "const"
    assert signature.free(language.type_sort).family == "var"
    assert signature.bound(language.term_sort).family == "var"
    assert signature.by_name(source.abstraction).family == "bind"
    assert signature.is_constant(signature.constant(language.type_sort))
    assert signature.is_free_variable(signature.free(language.type_sort))
    assert signature.is_bound_variable(signature.bound(language.term_sort))
    assert signature.is_binder(term.op)
    assert language.type_of(term).op.sort == language.type_sort


def _inferred_language() -> LanguageDecl:
    source = registered_language("neo-davidsonian")
    constants = tuple(
        constant.with_(type="infer", quantifies=())
        if constant.name == "and"
        else constant
        for constant in source.constants
    )
    return source.with_(
        constants=(
            *constants,
            ConstantSpec(name="john", type="e"),
            ConstantSpec(name="sleep", type="e -> t"),
            ConstantSpec(name="snore", type="infer", classes=("predicate",)),
            ConstantSpec(
                name="most",
                type="infer",
                classes=("quantifier",),
                notation=BinderNotationSpec(ascii="most", unicode="most", bodies=2),
            ),
        )
    )


def test_unresolved_constants_are_refused_without_typed_uses() -> None:
    with pytest.raises(ConstantInferenceError, match="and"):
        LanguageIso().forward(_inferred_language())


def test_the_declared_function_type_types_abstractions_beside_other_constructors() -> (
    None
):
    source = LanguageDecl(
        name="pairs",
        atomic_types=(AtomicTypeSpec(name="e"), AtomicTypeSpec(name="t")),
        constructors=(
            ConstructorSpec(
                name="pair",
                kind=("Ty", "Ty", "Ty"),
                notation=PrefixConstructorNotationSpec(spelling="pair"),
            ),
            ConstructorSpec(
                name="fn",
                kind=("Ty", "Ty", "Ty"),
                notation=InfixConstructorNotationSpec(
                    spelling="fn", precedence=1, associativity="right"
                ),
            ),
            ConstructorSpec(
                name="choice",
                kind=("Ty", "Ty", "Ty", "Ty"),
                notation=MixfixConstructorNotationSpec(
                    parts=("if ", " then ", " else ", " end"), precedence=2
                ),
            ),
        ),
        constants=(
            ConstantSpec(name="p", type="e fn t"),
            ConstantSpec(name="both", type="pair(e, e) fn t"),
            ConstantSpec(name="choose", type="if e then t else e end fn t"),
        ),
        binders=BinderSpec(style="curry"),
        function_type="fn",
    )
    language = LanguageIso().forward(source)

    assert language.function_type == "fn"
    assert language.print_type(language.type_of(language.read(r"\x. p(x)"))) == (
        "e fn t"
    )
    assert language.print_type(language.scheme("both").body) == "pair(e, e) fn t"
    assert language.print_type(language.scheme("choose").body) == (
        "if e then t else e end fn t"
    )


def test_the_profile_draws_connectives_and_quantifiers_from_declared_classes() -> None:
    source = LanguageDecl(
        name="classes",
        atomic_types=(AtomicTypeSpec(name="e"), AtomicTypeSpec(name="t")),
        constructors=(
            ConstructorSpec(
                name="->",
                kind=("Ty", "Ty", "Ty"),
                notation=InfixConstructorNotationSpec(
                    spelling="->", precedence=1, associativity="right"
                ),
            ),
        ),
        constants=(
            ConstantSpec(
                name="same",
                type="e -> e -> t",
                classes=("sentential", "equality"),
            ),
            ConstantSpec(name="also", type="t -> t -> t", classes=("conjunction",)),
            ConstantSpec(
                name="each",
                type="(e -> t) -> t",
                classes=("quantifier", "universal"),
            ),
            ConstantSpec(name="p", type="e -> t", classes=("predicate",)),
        ),
        binders=BinderSpec(style="curry"),
    )
    fragment = LanguageIso().forward(source).profile.fragment

    assert fragment.connectives == {"also"}
    assert fragment.quantifiers == {"each"}


def test_meaning_postulates_are_read_typed_and_stored_once() -> None:
    source = registered_language("neo-davidsonian").with_(
        constants=(
            *registered_language("neo-davidsonian").constants,
            ConstantSpec(name="dog", type="e -> t", classes=("predicate",)),
            ConstantSpec(name="mammal", type="e -> t", classes=("predicate",)),
        ),
        logic=LogicSpec(
            postulates=("forall x. dog(x) => mammal(x)",),
        ),
    )
    language = LanguageIso().forward(source)

    assert tuple(language.print(formula) for formula in language.profile.theory) == (
        "forall x0. dog(x0) => mammal(x0)",
    )


def test_a_defined_logical_constant_is_typed_and_stored_as_a_closed_term() -> None:
    source = registered_language("neo-davidsonian")
    constants = tuple(
        constant.with_(
            classes=("sentential",),
            definition=r"\p:t. \q:t. ~(~p | ~q)",
        )
        if constant.name == "and"
        else constant
        for constant in source.constants
    )
    language = LanguageIso().forward(source.with_(constants=constants))

    definition = language.profile.definitions["and"]

    assert definition.is_closed
    assert language.print_type(language.type_of(definition)) == "t -> t -> t"
    assert "and" in language.profile.logical_constants


def test_a_definition_cannot_refer_to_a_defined_constant() -> None:
    source = registered_language("neo-davidsonian")
    constants = tuple(
        constant.with_(classes=("sentential",), definition=r"\p:t. \q:t. p & q")
        if constant.name == "and"
        else constant
        for constant in source.constants
    )

    with pytest.raises(ValueError, match="must use primitives"):
        LanguageIso().forward(source.with_(constants=constants))


def test_a_semantics_plugin_must_match_its_constants_declared_arity(
    logical_constant_semantics: None,
) -> None:
    del logical_constant_semantics
    source = registered_language("neo-davidsonian")
    most = ConstantSpec(
        name="most",
        type="(e -> t) -> (e -> t) -> t",
        classes=("quantifier",),
        notation=BinderNotationSpec(ascii="most", unicode="most", bodies=2),
        semantics="most",
    )
    language = LanguageIso().forward(
        source.with_(
            constants=(*source.constants, most),
            logic=LogicSpec(order="higher"),
        )
    )

    assert language.profile.logical_semantics["most"].name == "most"

    with pytest.raises(ValueError, match=r"has arity 1.*needs 2"):
        LanguageIso().forward(
            source.with_(
                constants=(*source.constants, most.with_(type="(e -> t) -> t")),
                logic=LogicSpec(order="higher"),
            )
        )
