"""Tests for the type-logical grammar package exports."""

from lofi.grammars import tlg
from lofi.grammars.tlg import ModalMatching


def test_the_package_exports_only_the_modal_matching_runtime() -> None:
    assert tlg.__all__ == ["ModalMatching"]
    assert tlg.ModalMatching is ModalMatching
