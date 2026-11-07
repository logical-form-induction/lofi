import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, override

import pytest
from tests.conformance.metric_sink import (
    METRIC_SINK_CONFORMANCE_LAWS,
    MetricSinkConformance,
)

from lofi.run import METRIC_SINK_LAWS, MetricSinkFixture, NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue


class BackendImportingSink:
    def __init__(self) -> None:
        self.backend = import_module("wandb")

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        del step

        if mapping:
            message = "this sink requires one logging method per loss"
            raise TypeError(message)

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        del name, rows

    def log_loss(self, value: float, step: int) -> None:
        del value, step

    def close(self) -> None:
        return


def _bad_tree(root: Path) -> Path:
    unit = root / "tests" / "unit"
    lower = root / "lofi" / "train"
    unit.mkdir(parents=True)
    lower.mkdir(parents=True)
    (unit / "test_backend.py").write_text(
        "import wandb\n", encoding="utf-8", newline="\n"
    )
    (lower / "backend.py").write_text(
        "from wandb import log\n", encoding="utf-8", newline="\n"
    )

    return root


class TestNullMetricSink(MetricSinkConformance):
    @pytest.fixture
    def metric_sink_fixture(self) -> MetricSinkFixture:
        return MetricSinkFixture(
            NullSink(),
            Path(__file__).parents[2],
            frozenset({"wandb", "mlflow", "trackio"}),
            ("loss/syntax", "loss/semantics"),
        )


class TestBackendImportingMetricSink(MetricSinkConformance):
    @pytest.fixture
    def metric_sink_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> MetricSinkFixture:
        monkeypatch.setitem(sys.modules, "wandb", ModuleType("wandb"))

        return MetricSinkFixture(
            BackendImportingSink(),
            _bad_tree(tmp_path),
            frozenset({"wandb", "mlflow", "trackio"}),
            ("loss/syntax", "loss/semantics"),
        )

    @pytest.mark.xfail(
        strict=True,
        reason="a unit test imports the wandb backend",
    )
    @override
    def test_unit_tests_do_not_import_telemetry_backends(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        super().test_unit_tests_do_not_import_telemetry_backends(metric_sink_fixture)

    @pytest.mark.xfail(
        strict=True,
        reason="the train package imports the wandb backend",
    )
    @override
    def test_telemetry_backends_are_confined_to_run(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        super().test_telemetry_backends_are_confined_to_run(metric_sink_fixture)

    @pytest.mark.xfail(
        strict=True,
        reason="the sink requires a logging method per loss",
    )
    @override
    def test_metric_keys_are_supplied_as_data(
        self, metric_sink_fixture: MetricSinkFixture
    ) -> None:
        super().test_metric_keys_are_supplied_as_data(metric_sink_fixture)


def test_the_suite_binds_one_method_to_each_metric_sink_law() -> None:
    bound = {
        method.__name__.removeprefix("test_") for method in METRIC_SINK_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in METRIC_SINK_LAWS}
