from lofi.spec.language import BinderSpec, registered_language, resolve_language


def test_shipped_languages_declare_their_logical_constants() -> None:
    full = registered_language("neo-davidsonian")
    propositional = registered_language("propositional")

    assert {constant.name for constant in full.constants} >= {"forall", "exists"}
    assert all(
        "quantifier" not in constant.classes for constant in propositional.constants
    )
    assert all(
        "equality" not in constant.classes for constant in propositional.constants
    )
    assert full.binders == propositional.binders == BinderSpec()
    assert not propositional.logic.equality


def test_a_registered_name_resolves_at_a_language_boundary() -> None:
    declaration = registered_language("propositional")

    assert resolve_language("propositional") is declaration
    assert resolve_language(declaration) is declaration
