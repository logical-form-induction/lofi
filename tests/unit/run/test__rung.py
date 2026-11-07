from importlib.metadata import EntryPoint
from types import MappingProxyType

import pytest
from tests.support.installed import install

from lofi.registry import RegistryError, UnknownTagError
from lofi.run import FallbackRung, SeedRung
from lofi.run._contraction import Fallback
from lofi.run._rung import RungContext, contraction_rung

CONTEXT = RungContext(MappingProxyType({}), seeds=3, minimum_seeds=1)


def _only(name: str, value: str) -> tuple[EntryPoint, ...]:
    return (EntryPoint(name=name, value=value, group="lofi.contraction_rung"),)


def test_a_declared_rung_is_lowered_by_its_installed_plugin() -> None:
    assert isinstance(contraction_rung(FallbackRung(), CONTEXT), Fallback)


def test_a_rung_without_exactly_one_installed_plugin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "contraction_rung", ())

    with pytest.raises(UnknownTagError, match="'fallback'"):
        contraction_rung(FallbackRung(), CONTEXT)


def test_a_plugin_reading_another_declaration_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        "contraction_rung",
        _only("seeds", "lofi.run._rung_builtins:FALLBACK_PLUGIN"),
    )

    with pytest.raises(RegistryError, match="builds FallbackRung, not SeedRung"):
        contraction_rung(SeedRung(floor=1), CONTEXT)
