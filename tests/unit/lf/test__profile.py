import pytest

from lofi.lf import (
    LogicFragment,
    LogicProfileError,
    ProverBound,
    ProverUnsupportedError,
)
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language


def test_profile_derives_constant_inventories_from_declared_classes() -> None:
    declaration = registered_language("propositional").with_(
        constants=(
            *registered_language("propositional").constants,
            ConstantSpec(name="rain", type="t", classes=("predicate",)),
        )
    )
    language = LanguageIso().forward(declaration)

    assert language.profile.application == declaration.application
    assert language.profile.constant_classes["and"] == {
        "sentential",
        "conjunction",
    }
    assert language.profile.fragment.connectives >= {"and", "not"}


def test_fragment_inclusion_uses_declared_names_without_a_fixed_inventory() -> None:
    wide = LogicFragment(
        order="higher", quantifiers=("every", "most"), connectives=("nand",)
    )
    narrow = LogicFragment(order="first", quantifiers=("every",), connectives=("nand",))

    assert wide.admits(narrow)
    assert not narrow.admits(wide)


def test_profile_refuses_an_invalid_domain_by_field() -> None:
    language = LanguageIso().forward(registered_language("propositional"))

    with pytest.raises(LogicProfileError) as refused:
        type(language.profile)(
            language.store,
            formula_sort=language.term_sort,
            application=language.calculus.app,
            classes=language.constant_classes,
            fragment=language.profile.fragment,
            type_kinds=language.profile.type_kinds,
            variables=language.profile.variables,
            semantics_name=language.profile.semantics,
            negation=language.profile.negation,
            label_map=language.profile.label_map,
            domains={"e": 0},
            roles=language.profile.logical_roles,
        )

    assert refused.value.field == "domains"


def test_prover_bound_names_the_first_unsupported_profile_field() -> None:
    language = LanguageIso().forward(registered_language("propositional"))
    narrow = ProverBound(
        "oracle",
        LogicFragment(order="first"),
    )

    with pytest.raises(ProverUnsupportedError) as refused:
        narrow.check(language.profile)

    assert refused.value.field == "connectives"
