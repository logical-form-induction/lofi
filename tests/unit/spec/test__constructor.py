import didactic.api as dx
import pytest

from lofi.spec import (
    ConstructorSpec,
    InfixConstructorNotationSpec,
    MixfixConstructorNotationSpec,
    PrefixConstructorNotationSpec,
)


def test_a_constructor_derives_arity_and_builds_its_operator() -> None:
    declaration = ConstructorSpec(
        name="pair",
        kind=("Tm", "Tm", "Tm"),
        notation=PrefixConstructorNotationSpec(spelling="pair"),
    )

    operator = declaration.build()

    assert declaration.arity == 2
    assert operator.name == "pair"
    assert operator.arg_sorts == ("Tm", "Tm")
    assert operator.sort == "Tm"
    assert operator.family == "constructor"


def test_infix_and_mixfix_notations_validate_their_parse_boundaries() -> None:
    infix = InfixConstructorNotationSpec(
        spelling="->", precedence=10, associativity="right"
    )
    mixfix = MixfixConstructorNotationSpec(
        parts=("if ", " then ", " else "), precedence=5
    )

    assert infix.style == "infix"
    assert mixfix.parts == ("if ", " then ", " else ")

    with pytest.raises(dx.ValidationError, match="leading part"):
        MixfixConstructorNotationSpec(parts=("", " then "), precedence=5)


def test_a_constructor_builds_its_declared_notation() -> None:
    infix = ConstructorSpec(
        name="/",
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling="/", precedence=10, associativity="left"
        ),
    )
    mixfix = ConstructorSpec(
        name="pair",
        kind=("Tm", "Tm", "Tm"),
        notation=MixfixConstructorNotationSpec(parts=("<", "|", ">"), precedence=5),
    )

    assert infix.build_notation().name == "/"
    assert mixfix.build_notation().name == "pair"
    infix.refuse_notation_arity()
    mixfix.refuse_notation_arity()


def test_a_notation_refuses_a_constructor_it_cannot_write() -> None:
    ternary = ConstructorSpec(
        name="tri",
        kind=("Tm", "Tm", "Tm", "Tm"),
        notation=InfixConstructorNotationSpec(
            spelling="~", precedence=1, associativity="none"
        ),
    )
    short = ConstructorSpec(
        name="pair",
        kind=("Tm", "Tm", "Tm"),
        notation=MixfixConstructorNotationSpec(parts=("<", ">"), precedence=5),
    )

    with pytest.raises(ValueError, match="not binary"):
        ternary.refuse_notation_arity()

    with pytest.raises(ValueError, match="needs 3 parts"):
        short.refuse_notation_arity()
