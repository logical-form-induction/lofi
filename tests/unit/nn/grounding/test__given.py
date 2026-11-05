from __future__ import annotations

from typing import TYPE_CHECKING

from tests.language import language

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Frame,
    GivenSettings,
    GivenStructureSource,
    ModelLanguage,
    Structure,
)
from lofi.nn.grounding._given import GIVEN
from lofi.registry import Registry

if TYPE_CHECKING:
    import pytest


class Reader:
    def read(
        self,
        settings: object,
        declared: ModelLanguage,
        /,
        *,
        full_function_space_cap: int,
        henkin_pool: int,
    ) -> Structure:
        assert settings == ("fixture",)
        assert full_function_space_cap == 8
        assert henkin_pool == 2

        return Structure(
            "fixture",
            Frame(
                declared.store,
                function_type=declared.function_type,
                truth_type="t",
                carriers={"e": 1, "v": 1},
                full_function_space_cap=8,
                henkin_pool=2,
                seed=0,
            ),
            {},
        )


def test_the_given_factory_resolves_the_declared_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = language()

    def reader(registry: Registry[object], tag: str) -> object:
        assert registry.family == "knowledge_base"
        assert tag == "fixture"
        return Reader()

    monkeypatch.setattr(Registry, "get", reader)
    source = GIVEN.build(
        GivenSettings(
            "fixture",
            ("fixture",),
            full_function_space_cap=8,
            henkin_pool=2,
        ),
        declared,
        PRODUCT.build(
            implication=RESIDUUM,
            quantifier=EXTREME,
            equality=CRISP,
            margin=0.0,
        ),
        theory_floor=1.0,
    )

    assert isinstance(source, GivenStructureSource)
    assert source.structures()[0].name == "fixture"
