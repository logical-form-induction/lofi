from lofi import spec
from lofi.spec import language
from lofi.spec.language import LanguageDecl, LanguageIso


def test_language_package_exports_the_declared_boundary() -> None:
    assert language.LanguageDecl is LanguageDecl
    assert language.LanguageIso is LanguageIso
    assert hasattr(spec, "language")
    assert not hasattr(language, "neo_davidsonian")
    assert not hasattr(language, "propositional_language")
