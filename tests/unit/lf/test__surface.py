from typing import Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.lf import Application, Atom, Expression, Lambda, LogicalFormParseError
from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantSpec,
    InfixNotationSpec,
    LanguageAssembly,
    LanguageDecl,
    LanguageIso,
    PrefixNotationSpec,
)


def _language() -> LanguageAssembly:
    return LanguageIso().forward(
        LanguageDecl(
            name="surface",
            atomic_types=tuple(AtomicTypeSpec(name=name) for name in ("e", "t")),
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
                ConstantSpec(name="p", type="e -> t"),
                ConstantSpec(
                    name="and",
                    type="t -> t -> t",
                    classes=("conjunction",),
                    notation=InfixNotationSpec(
                        ascii="&", unicode="∧", precedence=4, associativity="left"
                    ),
                ),
                ConstantSpec(
                    name="not",
                    type="t -> t",
                    classes=("negation",),
                    notation=PrefixNotationSpec(ascii="-", unicode="¬", precedence=2),
                ),
                ConstantSpec(
                    name="every",
                    type="(e -> t) -> t",
                    classes=("quantifier",),
                    notation=BinderNotationSpec(ascii="every", unicode="∀", bodies=1),
                ),
            ),
            binders=BinderSpec(style="curry"),
        )
    )


def test_generated_notation_round_trips_open_terms_and_precedence() -> None:
    language = _language()
    term = language.read("every x. p(x) & p(y) & p(x)")

    assert language.read(language.print(term)).key() == term.key()


def test_an_abstraction_in_function_position_is_parenthesized() -> None:
    language = _language()
    term = language.read(r"(\x. p(x))(y)")
    written = language.print(term)

    assert written == r"(\x0. p(x0))(y)"
    assert language.read(written).key() == term.key()


def test_undeclared_notation_is_refused() -> None:
    with pytest.raises(LogicalFormParseError):
        _language().read("~p(x)")


def test_an_inner_binder_shadows_an_outer_binder_of_the_same_name() -> None:
    language = _language()
    shadowed = language.read(r"\x. \x. p(x)")

    assert shadowed.key() == language.read(r"\y. \x. p(x)").key()
    assert shadowed.key() != language.read(r"\x. \y. p(x)").key()


def test_a_looser_prefix_operand_of_an_infix_is_parenthesized() -> None:
    language = _language()
    term = language.read("(-p(x)) & p(y)")
    written = language.print(term)

    assert written == "(- p(x)) & p(y)"
    assert language.read(written).key() == term.key()
    assert language.read("-p(x) & p(y)").key() != term.key()


def test_binder_constant_without_abstraction_uses_application_syntax() -> None:
    language = _language()
    every = language.constant("every")
    predicate = language.constant("p")
    term = language.apply(every, predicate)

    assert language.print(term) == "every(p)"


def test_a_binder_operand_of_an_infix_is_parenthesized() -> None:
    language = _language()
    term = language.read("(every x. p(x)) & p(y)")
    written = language.print(term)

    assert written == "(every x0. p(x0)) & p(y)"
    assert language.read(written).key() == term.key()


def test_printed_bound_names_avoid_free_variables_of_the_term() -> None:
    language = _language()
    term = language.read(r"every y. p(y) & p(x0)")
    written = language.print(term)

    assert "x0" in written
    assert language.read(written).key() == term.key()


_SYMBOLS: Final = ("&", "|", "+", "*", "%", "^", "#", "@", "!", "?")
_BINDER_WORDS: Final = (
    "all",
    "every",
    "forall",
    "exists",
    "some",
    "most",
    "iota",
    "the",
)
_FREE: Final = ("u", "w", "x0", "x1")


@st.composite
def _declarations(draw: st.DrawFn) -> LanguageDecl:
    """Draw a language whose notation varies in kind, precedence and shape."""
    count = draw(st.integers(min_value=1, max_value=6))
    symbols = iter(draw(st.permutations(_SYMBOLS)))
    constants = [ConstantSpec(name="c", type="t")]

    for at in range(count):
        kind = draw(st.sampled_from(("prefix", "infix", "binder", "plain")))
        precedence = draw(st.integers(min_value=1, max_value=12))

        if kind == "prefix":
            token = next(symbols)
            notation = PrefixNotationSpec(
                ascii=token, unicode=token, precedence=precedence
            )
        elif kind == "infix":
            token = next(symbols)
            notation = InfixNotationSpec(
                ascii=token,
                unicode=token,
                precedence=precedence,
                associativity=draw(st.sampled_from(("left", "right"))),
            )
        elif kind == "binder":
            notation = BinderNotationSpec(
                ascii=f"q{at}",
                unicode=f"q{at}",
                bodies=draw(st.integers(min_value=1, max_value=3)),
            )
        else:
            notation = None

        constants.append(ConstantSpec(name=f"k{at}", type="t", notation=notation))

    return LanguageDecl(
        name="drawn",
        atomic_types=(AtomicTypeSpec(name="t"),),
        constructors=(
            ConstructorSpec(
                name="->",
                kind=("Ty", "Ty", "Ty"),
                notation=InfixConstructorNotationSpec(
                    spelling="->", precedence=1, associativity="right"
                ),
            ),
        ),
        constants=tuple(constants),
        binders=BinderSpec(style="curry"),
    )


def _expressions(language: LanguageDecl) -> st.SearchStrategy[Expression]:
    """Draw surface expressions over a language's declared notation."""
    names = tuple(constant.name for constant in language.constants)
    leaves = st.sampled_from((*names, *_FREE, "y")).map(Atom)
    arities = {
        constant.name: (
            1
            if constant.notation is None or constant.notation.kind == "prefix"
            else 2
            if constant.notation.kind == "infix"
            else -cast("int", constant.notation.bodies)
        )
        for constant in language.constants
        if constant.name != "c"
    }

    def extend(inner: st.SearchStrategy[Expression]) -> st.SearchStrategy[Expression]:
        def applied(name: str, arguments: list[Expression], bound: str) -> Expression:
            arity = arities[name]
            held: Expression = Atom(name)

            if arity < 0:
                for argument in arguments[:-arity]:
                    held = Application(held, Lambda(bound, None, argument))

                return held

            for argument in arguments[:arity]:
                held = Application(held, argument)

            return held

        return st.one_of(
            st.builds(Application, inner, inner),
            st.builds(Lambda, st.sampled_from(("y", "x0")), st.none(), inner),
            st.builds(
                applied,
                st.sampled_from(tuple(arities)),
                st.lists(inner, min_size=3, max_size=3),
                st.sampled_from(("y", "x1")),
            ),
        )

    return st.recursive(leaves, extend, max_leaves=12)


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_generated_notations_print_terms_their_parser_reads_back(
    data: st.DataObject,
) -> None:
    declaration = data.draw(_declarations())
    language = LanguageIso().forward(declaration)
    syntax = language.syntax
    term = syntax.desugar(data.draw(_expressions(declaration))).term
    written = language.print(term)

    assert syntax.desugar(syntax.parse(written)).term.key() == term.key(), written


@settings(max_examples=60, deadline=None)
@given(declaration=_declarations())
def test_only_declared_notation_is_read(declaration: LanguageDecl) -> None:
    syntax = LanguageIso().forward(declaration).syntax
    declared = {
        constant.notation.ascii
        for constant in declaration.constants
        if constant.notation is not None
    }

    for symbol in _SYMBOLS:
        if symbol not in declared:
            with pytest.raises(LogicalFormParseError):
                syntax.parse(f"c {symbol} c")

            with pytest.raises(LogicalFormParseError):
                syntax.parse(f"{symbol} c")

    for word in _BINDER_WORDS:
        with pytest.raises(LogicalFormParseError):
            syntax.parse(f"{word} y. c")
