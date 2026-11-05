from lofi.spec.lexicon_generators import GENERATOR_FAMILY, LexiconGeneratorSpec


def test_every_generator_declaration_names_the_generator_family() -> None:
    assert LexiconGeneratorSpec.family() == GENERATOR_FAMILY == "lexicon_generator"
