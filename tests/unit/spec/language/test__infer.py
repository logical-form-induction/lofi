import pytest

from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantInferenceError,
    ConstantSpec,
    LanguageDecl,
    LanguageError,
    LanguageIso,
    registered_language,
)
from lofi.spec.language._infer import infer_declaration
from lofi.spec.language._lowering import lower_language


def _with(*constants: ConstantSpec, inferred_and: bool = False) -> LanguageDecl:
    source = registered_language("neo-davidsonian")
    return source.with_(
        constants=(
            *(
                constant.with_(type="infer", quantifies=())
                if inferred_and and constant.name == "and"
                else constant
                for constant in source.constants
            ),
            ConstantSpec(name="john", type="e"),
            ConstantSpec(name="sleep", type="e -> t"),
            *constants,
        )
    )


def _types(source: LanguageDecl) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        constant.name: (constant.type, constant.quantifies)
        for constant in source.constants
    }


MOST = ConstantSpec(
    name="most",
    type="infer",
    classes=("quantifier",),
    notation=BinderNotationSpec(ascii="most", unicode="most", bodies=2),
)


def test_uses_assign_principal_types_to_logical_and_nonlogical_constants() -> None:
    resolved = _types(
        infer_declaration(
            _with(ConstantSpec(name="snore", type="infer"), MOST, inferred_and=True),
            (
                ("snore(john)", "t"),
                ("snore(john) & sleep(john)", "t"),
                ("most x.(snore(x), sleep(x))", "t"),
            ),
            lower_language,
        )
    )

    assert resolved["snore"] == ("e -> t", ())
    assert resolved["and"] == ("t -> t -> t", ())
    assert resolved["most"] == ("(e -> t) -> (e -> t) -> t", ())
    assert resolved["sleep"] == ("e -> t", ())


def test_a_function_use_and_an_individual_use_are_refused_naming_both() -> None:
    with pytest.raises(ConstantInferenceError) as refused:
        infer_declaration(
            _with(ConstantSpec(name="bad", type="infer")),
            (("bad(john)", "t"), ("sleep(john)", "t"), ("bad", "e")),
            lower_language,
        )

    assert refused.value.constant == "bad"
    assert "use 'bad(john)' demands 'e -> t'" in str(refused.value)
    assert "use 'bad' demands 'e'" in str(refused.value)


def test_uses_at_two_atomic_types_are_refused_rather_than_generalized() -> None:
    with pytest.raises(ConstantInferenceError, match="bad") as refused:
        infer_declaration(
            _with(
                ConstantSpec(name="bad", type="infer"),
                ConstantSpec(name="run", type="v"),
            ),
            (("bad(john)", "t"), ("bad(run)", "t")),
            lower_language,
        )

    assert "'bad(john)'" in str(refused.value)
    assert "'bad(run)'" in str(refused.value)
    assert "'e -> t'" in str(refused.value)
    assert "'v -> t'" in str(refused.value)


def test_occurrences_within_one_use_share_the_constraints_of_that_use() -> None:
    resolved = _types(
        infer_declaration(
            _with(ConstantSpec(name="pick", type="infer")),
            (("\\x. sleep(pick(x)) & sleep(x)", "e -> t"),),
            lower_language,
        )
    )

    assert resolved["pick"] == ("e -> e", ())


def test_an_unconstrained_type_is_quantified_avoiding_declared_atoms() -> None:
    source = LanguageDecl(
        name="atoms-named-like-variables",
        atomic_types=(AtomicTypeSpec(name="a"), AtomicTypeSpec(name="t")),
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
            ConstantSpec(name="same", type="x -> x -> t", quantifies=("x",)),
            ConstantSpec(name="holds", type="a -> t"),
            ConstantSpec(name="loose", type="infer"),
            ConstantSpec(name="pinned", type="infer"),
        ),
        binders=BinderSpec(style="curry"),
    )
    resolved = _types(
        infer_declaration(
            source,
            (("same(loose)(loose)", "t"), ("\\y. holds(pinned(y))", "t -> t")),
            lower_language,
        )
    )

    assert resolved["loose"] == ("b", ("b",))
    assert resolved["pinned"] == ("t -> a", ())
    assert (
        LanguageIso()
        .forward(
            source.with_(
                constants=tuple(
                    constant.with_(
                        type=resolved[constant.name][0],
                        quantifies=resolved[constant.name][1],
                    )
                    for constant in source.constants
                )
            )
        )
        .scheme("loose")
        .quantified
    )


def test_type_variables_of_separate_uses_do_not_constrain_each_other() -> None:
    source = LanguageDecl(
        name="uses-apart",
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
            ConstantSpec(name="id", type="x -> x", quantifies=("x",)),
            ConstantSpec(name="neg", type="t -> t"),
            ConstantSpec(name="pinned", type="infer"),
            ConstantSpec(name="loose", type="infer"),
        ),
        binders=BinderSpec(style="curry"),
    )
    resolved = _types(
        infer_declaration(
            source,
            (("pinned(id)", "t"), ("loose(id)", "t"), ("pinned(neg)", "t")),
            lower_language,
        )
    )

    assert resolved["pinned"] == ("(t -> t) -> t", ())
    assert resolved["loose"] == ("(a -> a) -> t", ("a",))


def test_an_inferred_constant_no_use_mentions_is_refused() -> None:
    with pytest.raises(ConstantInferenceError, match=r"snore.*no declared use"):
        infer_declaration(
            _with(ConstantSpec(name="snore", type="infer")),
            (("john & john", "t"), ("sleep(john)", "t")),
            lower_language,
        )


def test_a_use_of_the_wrong_type_is_refused_naming_the_use() -> None:
    with pytest.raises(LanguageError, match="'sleep\\(snore\\)' has type 't', not 'e'"):
        infer_declaration(
            _with(ConstantSpec(name="snore", type="infer")),
            (("snore(john)", "t"), ("sleep(snore)", "e")),
            lower_language,
        )


def test_the_converter_lowers_the_resolved_declaration() -> None:
    converter = LanguageIso()
    language = converter.infer(
        _with(ConstantSpec(name="snore", type="infer")),
        (("snore(john) & sleep(john)", "t"),),
    )

    assert language.print_type(language.scheme("snore").body) == "e -> t"
    assert all(
        constant.type != "infer" for constant in converter.backward(language).constants
    )
