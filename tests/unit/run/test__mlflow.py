from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest
from tests.conformance.metric_sink import MetricSinkConformance

from lofi.run import MetricSinkFixture, MissingTelemetryBackendError, SinkContext
from lofi.run import _mlflow as mlflow_module
from lofi.run._mlflow import PLUGIN, MlflowSink
from lofi.spec.sinks.mlflow import MlflowSinkSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from lofi._types import JsonValue

POINT = SinkContext("0123abcd")


class FakeMetric(NamedTuple):
    key: str
    value: float
    timestamp: int
    step: int


class FakeExperiment(NamedTuple):
    experiment_id: str


class FakeRunInfo(NamedTuple):
    run_id: str


class FakeRun(NamedTuple):
    info: FakeRunInfo


class FakeClient:
    def __init__(self, experiments: Mapping[str, str]) -> None:
        self.tracking_uri: str | None = None
        self.experiments = dict(experiments)
        self.runs: list[tuple[str, str | None]] = []
        self.metrics: list[tuple[str, str, float, int]] = []
        self.tables: list[tuple[str, dict[str, tuple[JsonValue, ...]], str]] = []
        self.terminated: list[str] = []

    def get_experiment_by_name(self, name: str) -> FakeExperiment | None:
        if name not in self.experiments:
            return None

        return FakeExperiment(self.experiments[name])

    def create_experiment(self, name: str) -> str:
        self.experiments[name] = f"created-{name}"

        return self.experiments[name]

    def create_run(self, experiment_id: str, *, run_name: str | None) -> FakeRun:
        self.runs.append((experiment_id, run_name))

        return FakeRun(FakeRunInfo(f"run-{len(self.runs)}"))

    def log_batch(self, run_id: str, *, metrics: Sequence[FakeMetric]) -> None:
        self.metrics.extend(
            (run_id, metric.key, metric.value, metric.step) for metric in metrics
        )

    def log_table(
        self,
        run_id: str,
        data: Mapping[str, Sequence[JsonValue]],
        artifact_file: str,
    ) -> None:
        self.tables.append(
            (
                run_id,
                {name: tuple(values) for name, values in data.items()},
                artifact_file,
            )
        )

    def set_terminated(self, run_id: str) -> None:
        self.terminated.append(run_id)


class FakeModule:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.MlflowClient = self.connect
        self.Metric = FakeMetric

    def connect(self, *, tracking_uri: str | None) -> FakeClient:
        self.client.tracking_uri = tracking_uri

        return self.client


class TestMlflowSink(MetricSinkConformance):
    @pytest.fixture
    def metric_sink_fixture(self) -> MetricSinkFixture:
        return MetricSinkFixture(
            MlflowSink(FakeClient({}), FakeMetric, "run-1"),
            Path(__file__).parents[3],
            frozenset({"wandb", "mlflow", "trackio"}),
            ("loss/syntax", "loss/semantics"),
        )


def _loading(
    monkeypatch: pytest.MonkeyPatch, client: FakeClient, loaded: list[str]
) -> None:
    module = FakeModule(client)

    def load(name: str) -> FakeModule:
        loaded.append(name)

        return module

    monkeypatch.setattr(mlflow_module.importlib, "import_module", load)


def test_the_plugin_loads_only_mlflow_and_logs_to_one_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient({"grammar": "7"})
    loaded: list[str] = []
    _loading(monkeypatch, client, loaded)

    sink = PLUGIN.construct(
        MlflowSinkSpec(
            experiment_name="grammar",
            run_name="trial",
            tracking_uri="file:///tmp/mlruns",
        ),
        POINT,
    )
    sink.log_scalars({"loss": 0.25, "loss/coherence": 0.5}, 4)
    sink.log_table("scores", ({"split": "dev", "score": 0.5},))
    sink.close()

    assert loaded == ["mlflow", "mlflow.entities"]
    assert client.tracking_uri == "file:///tmp/mlruns"
    assert client.runs == [("7", "trial")]
    assert client.metrics == [
        ("run-1", "loss", 0.25, 4),
        ("run-1", "loss/coherence", 0.5, 4),
    ]
    assert client.tables == [
        ("run-1", {"score": (0.5,), "split": ("dev",)}, "scores.json")
    ]
    assert client.terminated == ["run-1"]


def test_an_absent_experiment_is_created_and_the_run_named_after_its_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient({"other": "3"})
    _loading(monkeypatch, client, [])

    PLUGIN.construct(MlflowSinkSpec(experiment_name="grammar"), POINT)

    assert client.tracking_uri is None
    assert client.experiments == {"other": "3", "grammar": "created-grammar"}
    assert client.runs == [("created-grammar", "0123abcd")]


def test_an_unavailable_selected_client_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(mlflow_module.importlib, "import_module", missing)

    with pytest.raises(MissingTelemetryBackendError, match=r"lofi\[tracking\]"):
        PLUGIN.construct(MlflowSinkSpec(experiment_name="grammar"), POINT)
