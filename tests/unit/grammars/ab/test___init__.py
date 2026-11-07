"""Tests for the applicative grammar package boundary."""

from lofi.grammars import ab


def test_the_package_adds_no_public_python_declaration_route() -> None:
    assert not tuple(name for name in vars(ab) if not name.startswith("_"))
