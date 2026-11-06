from lofi.lf import _codec, term_codec
from lofi.spec.language import LanguageIso, registered_language
from lofi.terms import CompleteTree, PreorderArityLayout


def test_builds_a_fresh_codec_from_the_lowered_language_signature() -> None:
    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    layouts = (
        CompleteTree(
            language.store.signature,
            language.store.signature.operators,
            depth=2,
        ),
        PreorderArityLayout(5),
    )

    for layout in layouts:
        codec = term_codec(language, layout)

        assert codec.index.signature is language.store.signature
        assert codec.layout is layout


def test_no_process_global_operator_index_exists() -> None:
    assert "LF_OPERATOR_INDEX" not in vars(_codec)
