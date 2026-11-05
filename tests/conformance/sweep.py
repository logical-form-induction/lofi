from typing import TYPE_CHECKING, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.run import SweepPoint

__all__ = ["SWEEP_LAWS", "SweepConformance", "SweepFixture"]


class SweepFixture(NamedTuple):
    render: Callable[[int], bytes]
    identity: Callable[[Mapping[str, JsonValue]], str]
    deduplicate: Callable[[Sequence[SweepPoint]], tuple[SweepPoint, ...]]
    points: tuple[SweepPoint, ...]


def _fail(detail: str, *, law: str) -> NoReturn:
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def expansion_is_process_stable(fixture: SweepFixture) -> None:
    law = "expansion_is_process_stable"
    first = fixture.render(1)
    second = fixture.render(2)

    if not first:
        _fail("the expansion is empty", law=law)

    if first != second:
        _fail("two hash seeds produced different bytes", law=law)


def identity_is_key_order_invariant(fixture: SweepFixture) -> None:
    law = "identity_is_key_order_invariant"
    left: dict[str, JsonValue] = {"model": {"width": 64, "depth": 2}, "seed": 3}
    right: dict[str, JsonValue] = {"seed": 3, "model": {"depth": 2, "width": 64}}

    if fixture.identity(left) != fixture.identity(right):
        _fail("mapping order changed the point identifier", law=law)


def overlapping_sweeps_collapse(fixture: SweepFixture) -> None:
    law = "overlapping_sweeps_collapse"

    if not fixture.points:
        _fail("the fixture supplies no point", law=law)

    combined = (*fixture.points, fixture.points[0])
    unique = fixture.deduplicate(combined)

    if len(unique) != len(fixture.points) or set(unique) != set(fixture.points):
        _fail("an overlapping point did not collapse", law=law)


SWEEP_LAWS = (
    expansion_is_process_stable,
    identity_is_key_order_invariant,
    overlapping_sweeps_collapse,
)


class SweepConformance:
    pytestmark = pytest.mark.conformance

    def test_expansion_is_process_stable(self, sweep_fixture: SweepFixture) -> None:
        expansion_is_process_stable(sweep_fixture)

    def test_identity_is_key_order_invariant(self, sweep_fixture: SweepFixture) -> None:
        identity_is_key_order_invariant(sweep_fixture)

    def test_overlapping_sweeps_collapse(self, sweep_fixture: SweepFixture) -> None:
        overlapping_sweeps_collapse(sweep_fixture)
