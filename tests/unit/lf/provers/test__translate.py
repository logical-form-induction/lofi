import re
from typing import TYPE_CHECKING

import pytest

from lofi.lf.provers import FirstOrderTranslationError
from lofi.lf.provers._translate import smt_query, tptp_problem, truth_type
from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderNotationSpec,
    BinderSpec,
    ConstantSpec,
    InfixNotationSpec,
    LanguageDecl,
    LanguageIso,
    LogicSpec,
    PrefixNotationSpec,
    registered_language,
)

if TYPE_CHECKING:
    from lofi.spec.language import LanguageAssembly


def test_translation_uses_declared_roles_instead_of_constant_names(
    first_order_language: LanguageAssembly,
) -> None:
    language = first_order_language
    premise = language.read("all x. dog(x) => mammal(x)")
    hypothesis = language.read("mammal(rex)")
    fof = tptp_problem(language, premise, hypothesis, dialect="fof")
    smt = smt_query(language, premise, hypothesis)

    assert "! [X0] :" in fof
    assert "=>" in fof
    assert "$tType" not in fof
    assert any("declare-fun" in declaration for declaration in smt.declarations)
    assert any("forall" in assertion for assertion in smt.assertions)


def test_typed_translation_infers_an_inner_binder_in_its_outer_scope(
    first_order_language: LanguageAssembly,
) -> None:
    language = first_order_language
    nested = language.read("all x. dog(x) => all y. mammal(y) => dog(x)")
    tff = tptp_problem(language, nested, nested, dialect="tff")
    smt = smt_query(language, nested, nested)

    assert tff.count("! [X") == 4
    assert any(assertion.count("forall") == 2 for assertion in smt.assertions)


def test_typed_tptp_declares_the_language_types_of_the_used_constants(
    first_order_language: LanguageAssembly,
) -> None:
    language = first_order_language
    tff = tptp_problem(
        language,
        language.read("all x. dog(x) => mammal(x)"),
        language.read("mammal(rex)"),
        dialect="tff",
    )
    lines = tff.splitlines()

    assert sum("$tType" in line for line in lines) == 1
    assert any(line.endswith("> $o).") and "dog" in line for line in lines)
    assert any(": lofi_sort_e" in line and "rex" in line for line in lines)
    assert any("! [X0: lofi_sort_e" in line for line in lines)
    assert lines[-1].startswith("tff(conjecture, conjecture,")


def _renamed() -> LanguageAssembly:
    return LanguageIso().forward(
        LanguageDecl(
            name="renamed-logic",
            atomic_types=(AtomicTypeSpec(name="ent"), AtomicTypeSpec(name="prop")),
            constructors=(
                ConstructorSpec(
                    name="to",
                    kind=("Ty", "Ty", "Ty"),
                    notation=InfixConstructorNotationSpec(
                        spelling="to", precedence=1, associativity="right"
                    ),
                ),
            ),
            constants=(
                ConstantSpec(name="owns", type="ent to ent to prop"),
                ConstantSpec(name="ann", type="ent"),
                ConstantSpec(
                    name="nicht",
                    type="prop to prop",
                    classes=("negation",),
                    notation=PrefixNotationSpec(ascii="-", unicode="-", precedence=5),
                ),
                ConstantSpec(
                    name="oder",
                    type="prop to prop to prop",
                    classes=("disjunction",),
                    notation=InfixNotationSpec(
                        ascii="+", unicode="+", precedence=2, associativity="left"
                    ),
                ),
                ConstantSpec(
                    name="some",
                    type="(ent to prop) to prop",
                    classes=("existential",),
                    notation=BinderNotationSpec(ascii="some", unicode="some"),
                ),
            ),
            binders=BinderSpec(style="church"),
            function_type="to",
        )
    )


def test_a_renamed_logic_translates_through_its_declared_roles_and_types() -> None:
    language = _renamed()
    premise = language.read("some x:ent. owns(ann)(x) + -owns(x)(ann)")
    query = smt_query(language, premise, premise)
    tff = tptp_problem(language, premise, premise, dialect="tff")

    assert truth_type(language) == "prop"
    assert "(declare-fun" in "".join(query.declarations)
    assert any("(exists ((x0 " in assertion for assertion in query.assertions)
    assert any("(or " in assertion for assertion in query.assertions)
    assert "? [X0: lofi_sort_ent" in tff
    assert "> $o)." in tff


def test_logical_constants_must_share_one_truth_type() -> None:
    declaration = LanguageDecl(
        name="split-truth",
        atomic_types=(AtomicTypeSpec(name="t"), AtomicTypeSpec(name="u")),
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
            ConstantSpec(name="neg", type="t -> t", classes=("negation",)),
            ConstantSpec(name="neg2", type="u -> u", classes=("negation",)),
        ),
        binders=BinderSpec(style="curry"),
    )

    with pytest.raises(FirstOrderTranslationError, match="one truth type"):
        truth_type(LanguageIso().forward(declaration))


def test_thf_emits_quantification_over_predicates() -> None:
    declared = registered_language("neo-davidsonian").with_(
        name="second-order",
        constants=(
            *registered_language("neo-davidsonian").constants,
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
            ConstantSpec(name="fido", type="e"),
        ),
        logic=LogicSpec(order="second"),
    )
    language = LanguageIso().forward(declared)
    premise = language.read("(forall P:(e -> t). P(rex) => P(fido)) & dog(rex)")
    problem = tptp_problem(
        language,
        premise,
        language.read("dog(fido)"),
        dialect="thf",
    )

    assert "thf(" in problem
    assert "! [X0: (lofi_sort_e" in problem
    assert "lofi_dog" in problem
    assert problem.splitlines()[-1].startswith("thf(conjecture, conjecture,")


def test_declared_meaning_postulates_are_emitted_before_the_question() -> None:
    declared = registered_language("neo-davidsonian").with_(
        constants=(
            *registered_language("neo-davidsonian").constants,
            ConstantSpec(name="dog", type="e -> t"),
            ConstantSpec(name="mammal", type="e -> t"),
            ConstantSpec(name="rex", type="e"),
        ),
        logic=LogicSpec(postulates=("forall x. dog(x) => mammal(x)",)),
    )
    language = LanguageIso().forward(declared)
    problem = tptp_problem(
        language,
        language.read("dog(rex)"),
        language.read("mammal(rex)"),
        dialect="tff",
    )

    assert "tff(axiom_0, axiom," in problem
    assert "tff(axiom_1, axiom," in problem
    assert problem.index("tff(axiom_0") < problem.index("tff(conjecture")


def test_thf_unfolds_a_defined_connective_without_declaring_a_symbol() -> None:
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
    language = LanguageIso().forward(
        source.with_(
            constants=(
                *constants,
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="q", type="t"),
            )
        )
    )

    problem = tptp_problem(
        language,
        language.read("p & q"),
        language.read("p & q"),
        dialect="thf",
    )

    assert "lofi_and" not in problem
    assert "(^ [X" in problem
    assert " | " in problem


def test_thf_emits_a_postulated_generalized_quantifier_plugin() -> None:
    source = registered_language("neo-davidsonian")
    language = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(name="dog", type="e -> t"),
                ConstantSpec(name="mammal", type="e -> t"),
                ConstantSpec(
                    name="most",
                    type="(e -> t) -> (e -> t) -> t",
                    classes=("quantifier",),
                    notation=BinderNotationSpec(ascii="most", unicode="most", bodies=2),
                    semantics="most",
                ),
            ),
            logic=LogicSpec(
                order="higher",
                postulates=("forall x. dog(x) => dog(x)",),
            ),
        )
    )
    formula = language.read("most x.(dog(x), mammal(x))")
    problem = tptp_problem(language, formula, formula, dialect="thf")

    assert "lofi_most" in problem
    assert problem.count("lofi_most") >= 3
    assert "@ (^ [X" in problem


def test_monomorphization_translates_a_polymorphic_constant_instance() -> None:
    source = registered_language("neo-davidsonian")
    language = LanguageIso().forward(
        source.with_(
            constants=(
                *source.constants,
                ConstantSpec(name="rex", type="e"),
                ConstantSpec(name="id", type="a -> a", quantifies=("a",)),
            )
        )
    )
    formula = language.read("id(rex) = rex")

    tff = tptp_problem(
        language,
        formula,
        formula,
        dialect="tff",
        polymorphism="monomorphize",
    )
    smt = smt_query(
        language,
        formula,
        formula,
        polymorphism="monomorphize",
    )

    assert "lofi_id" in tff
    assert re.search(r"lofi_id[^:]*: (lofi_sort_e[^ ]*) > \1", tff) is not None
    assert any("lofi_id" in declaration for declaration in smt.declarations)
