from lofi.nn import tokens


def test_the_package_initializer_does_not_publish_implementation_objects() -> None:
    assert not hasattr(tokens, "__all__")
    assert not hasattr(tokens, "TokenVocabulary")
    assert not hasattr(tokens, "PretrainedTokenEncoder")
    assert not hasattr(tokens, "MeanSubwordPooler")
