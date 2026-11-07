from typing import TYPE_CHECKING

import pytest
from tests.conformance.sweep import SweepConformance, SweepFixture

from lofi.run import SweepPoint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue


class NullExpander:
    __slots__ = ()

    @staticmethod
    def render(seed: int) -> bytes:
        return str(seed).encode()

    @staticmethod
    def identity(config: Mapping[str, JsonValue]) -> str:
        return repr(config)

    @staticmethod
    def deduplicate(points: Sequence[SweepPoint]) -> tuple[SweepPoint, ...]:
        return tuple(points)


@pytest.mark.xfail(strict=True, reason="the null expander violates every sweep law")
class TestNullExpander(SweepConformance):
    @pytest.fixture
    def sweep_fixture(self) -> SweepFixture:
        null = NullExpander()
        points = (
            SweepPoint({"model": {"width": 64}, "seed": 1}),
            SweepPoint({"model": {"width": 128}, "seed": 1}),
        )

        return SweepFixture(null.render, null.identity, null.deduplicate, points)
