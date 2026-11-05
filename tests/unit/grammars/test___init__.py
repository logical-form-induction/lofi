from lofi import grammars


def test_the_package_has_no_cross_formalism_exports() -> None:
    assert grammars.__all__ == []
