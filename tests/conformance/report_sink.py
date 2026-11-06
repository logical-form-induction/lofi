"""Shared laws for sinks that receive completed report values."""

from typing import TYPE_CHECKING, ClassVar, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ReportSinkConformance", "ReportSinkFixture"]


class ReportSinkFixture[T](NamedTuple):
    """One sink and two emissions for the shared sink laws.

    Parameters
    ----------
    name : str
        Sink name.
    emit : Callable[[T], None]
        Emit one report value.
    read : Callable[[], tuple[T, ...]]
        Return retained report values in emission order.
    first : T
        First report value.
    second : T
        Distinct second report value.
    retains : bool
        Whether the sink retains emitted values.
    """

    name: str
    emit: Callable[[T], None]
    read: Callable[[], tuple[T, ...]]
    first: T
    second: T
    retains: bool


class ReportSinkConformance:
    """Laws shared by report sinks that retain or discard their input."""

    pytestmark: ClassVar[pytest.MarkDecorator] = pytest.mark.conformance

    def test_the_sink_has_a_nonempty_name[T](
        self, report_sink_fixture: ReportSinkFixture[T]
    ) -> None:
        assert report_sink_fixture.name

    def test_one_emission_is_complete[T](
        self, report_sink_fixture: ReportSinkFixture[T]
    ) -> None:
        report_sink_fixture.emit(report_sink_fixture.first)
        expected = (report_sink_fixture.first,) if report_sink_fixture.retains else ()

        assert report_sink_fixture.read() == expected

    def test_repeated_emissions_retain_their_order[T](
        self, report_sink_fixture: ReportSinkFixture[T]
    ) -> None:
        report_sink_fixture.emit(report_sink_fixture.first)
        report_sink_fixture.emit(report_sink_fixture.second)
        expected = (
            (report_sink_fixture.first, report_sink_fixture.second)
            if report_sink_fixture.retains
            else ()
        )

        assert report_sink_fixture.read() == expected
