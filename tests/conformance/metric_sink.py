from typing import TYPE_CHECKING, Final

import pytest

from lofi.run import (
    metric_keys_are_supplied_as_data,
    telemetry_backends_are_confined_to_run,
    unit_tests_do_not_import_telemetry_backends,
)

if TYPE_CHECKING:
    from lofi.run import MetricSinkFixture

__all__ = ["METRIC_SINK_CONFORMANCE_LAWS", "MetricSinkConformance"]


class MetricSinkConformance:
    pytestmark = pytest.mark.conformance

    def test_unit_tests_do_not_import_telemetry_backends(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        unit_tests_do_not_import_telemetry_backends(metric_sink_fixture)

    def test_telemetry_backends_are_confined_to_run(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        telemetry_backends_are_confined_to_run(metric_sink_fixture)

    def test_metric_keys_are_supplied_as_data(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        metric_keys_are_supplied_as_data(metric_sink_fixture)


METRIC_SINK_CONFORMANCE_LAWS: Final = (
    MetricSinkConformance.test_unit_tests_do_not_import_telemetry_backends,
    MetricSinkConformance.test_telemetry_backends_are_confined_to_run,
    MetricSinkConformance.test_metric_keys_are_supplied_as_data,
)
