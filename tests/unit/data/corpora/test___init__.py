import pytest
from tests.conformance.corpus_reader import imports_no_tensor_library

from lofi._checks import LawViolationError
from lofi.data import corpora
from lofi.data.corpora import conll2008


def test_every_reader_shipped_today_imports_with_the_tensor_library_refused() -> None:
    imports_no_tensor_library("lofi.data.corpora")


def test_the_vocabulary_a_reader_emits_imports_with_it_refused_too() -> None:
    imports_no_tensor_library("lofi.data")


def test_conll2008_names_are_published_at_the_corpus_boundary() -> None:
    assert all(name in corpora.__all__ for name in conll2008.__all__)
    assert all(
        getattr(corpora, name) is getattr(conll2008, name) for name in conll2008.__all__
    )


def test_the_law_reports_a_package_that_does_reach_the_tensor_library() -> None:
    with pytest.raises(LawViolationError, match="did not import with torch refused"):
        imports_no_tensor_library("lofi.nn")
