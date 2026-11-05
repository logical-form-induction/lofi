from lofi.spec.language import LanguageIso, registered_language


def test_assembly_constructs_constants_and_applications_in_its_store() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    proposition = language.constant("not")
    applied = language.apply(proposition, language.store.fvar("Tm", "p"))

    assert language.store.contains(applied)
    assert applied.op.name == language.calculus.app
