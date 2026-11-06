"""Tests for installed multiple-comparison correction plugins."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest
from tests.support.installed import install

from lofi.registry import UnknownTagError
from lofi.run import BonferroniSpec, correction
from lofi.stats import Bonferroni

BONFERRONI_POINT = EntryPoint(
    name="bonferroni",
    value="lofi.run._correction_builtins:BONFERRONI_PLUGIN",
    group="lofi.correction",
)


class _UntaggedCorrection:
    @property
    def kind(self) -> object:
        return 7


def test_a_correction_resolves_through_its_installed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "correction", (BONFERRONI_POINT,))

    assert isinstance(correction(BonferroniSpec()), Bonferroni)


def test_a_correction_declaration_requires_a_string_tag() -> None:
    with pytest.raises(TypeError, match="no string tag"):
        correction(_UntaggedCorrection())


def test_a_correction_kind_requires_exactly_one_installed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "correction", ())

    with pytest.raises(UnknownTagError, match="'bonferroni'"):
        correction(BonferroniSpec())
