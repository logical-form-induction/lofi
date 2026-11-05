from hashlib import sha256
from typing import Final, cast

import didactic.api as dx
import pytest

from lofi.spec import ArchivedLanguage, IndexedLanguage

DECLARATION: Final = b'{"name":"toy"}'
SPECIFICATION: Final = b"theory one"
ROUTE: Final = dx.LoweringRoute("decl-1", "native-1")


def theory() -> IndexedLanguage:
    return IndexedLanguage(
        "decl-1->native-1",
        SPECIFICATION,
        sha256(SPECIFICATION).hexdigest(),
        ROUTE,
    )


def test_an_indexed_declaration_holds_its_declaration_and_theory() -> None:
    language = ArchivedLanguage(DECLARATION, theory())

    assert language == ArchivedLanguage(DECLARATION, theory())
    assert hash(language) == hash(ArchivedLanguage(DECLARATION, theory()))
    assert language.declaration == DECLARATION
    assert language.theory == theory()
    assert "ArchivedLanguage" in repr(language)


@pytest.mark.parametrize("declaration", [b"", cast("bytes", "text")])
def test_an_empty_or_nonbyte_declaration_is_refused(declaration: bytes) -> None:
    with pytest.raises((TypeError, ValueError)):
        ArchivedLanguage(declaration, theory())


def test_a_nonindexed_theory_is_refused() -> None:
    with pytest.raises(TypeError, match="IndexedLanguage"):
        ArchivedLanguage(DECLARATION, cast("IndexedLanguage", ROUTE))
