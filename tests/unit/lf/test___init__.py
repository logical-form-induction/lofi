from lofi import lf
from lofi.lf._codec import term_codec
from lofi.lf._renaming import ArgumentRenaming
from lofi.lf._surface import Desugared, LogicalFormParseError, SurfaceSyntax


def test_exports_the_current_language_operations() -> None:
    assert lf.Desugared is Desugared
    assert lf.LogicalFormParseError is LogicalFormParseError
    assert lf.SurfaceSyntax is SurfaceSyntax
    assert lf.term_codec is term_codec
    assert lf.ArgumentRenaming is ArgumentRenaming


def test_does_not_export_superseded_global_language_routes() -> None:
    for name in (
        "LF_OPERATOR_INDEX",
        "LF_SIGNATURE",
        "TYPE_SIGNATURE",
        "BinderPresentation",
        "DIALECTS",
        "Dialect",
        "EmissionError",
        "EmissionSort",
        "desugar",
        "emit",
        "emit_problem",
        "parse",
        "print_ascii",
        "print_annotated",
        "print_unicode",
    ):
        assert name not in vars(lf)
