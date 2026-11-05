from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest
from tests.conformance.metric_sink import MetricSinkConformance

from lofi.run import MetricSinkFixture, MissingTelemetryBackendError, SinkContext
from lofi.run import _wandb as wandb_module
from lofi.run._wandb import PLUGIN
from lofi.spec.sinks.wandb import WandbSinkSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from lofi._types import JsonValue


POINT = SinkContext("0123abcd")


class FakeTable(NamedTuple):
    columns: tuple[str, ...]
    data: tuple[tuple[JsonValue, ...], ...]


class FakeTableFactory:
    def __call__(
        self,
        *,
        columns: Sequence[str],
        data: Sequence[Sequence[JsonValue]],
    ) -> FakeTable:
        return FakeTable(tuple(columns), tuple(tuple(row) for row in data))


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float | FakeTable], int | None]] = []
        self.finished = False

    def log(
        self,
        data: Mapping[str, float | FakeTable],
        *,
        step: int | None = None,
    ) -> None:
        self.logged.append((dict(data), step))

    def finish(self) -> None:
        self.finished = True


class FakeClient:
    def __init__(self) -> None:
        self.Table = FakeTableFactory()
        self.run = FakeRun()
        self.initialized: tuple[str, str | None, str | None] | None = None

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
    ) -> FakeRun:
        self.initialized = (project, entity, name)

        return self.run


class TestWandbSink(MetricSinkConformance):
    @pytest.fixture
    def metric_sink_fixture(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> MetricSinkFixture:
        client = FakeClient()

        def load(name: str) -> FakeClient:
            del name

            return client

        monkeypatch.setattr(wandb_module.importlib, "import_module", load)

        return MetricSinkFixture(
            PLUGIN.construct(WandbSinkSpec(project="grammar"), POINT),
            Path(__file__).parents[3],
            frozenset({"wandb", "mlflow", "trackio"}),
            ("loss/syntax", "loss/semantics"),
        )


def test_the_factory_loads_and_initializes_only_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    loaded: list[str] = []

    def load(name: str) -> FakeClient:
        loaded.append(name)

        return client

    monkeypatch.setattr(wandb_module.importlib, "import_module", load)
    sink = PLUGIN.construct(
        WandbSinkSpec(project="grammar", entity="lab", run_name="trial"), POINT
    )
    sink.log_scalars({"loss": 0.25}, 4)
    sink.log_table("scores", ({"split": "dev", "score": 0.5},))
    sink.close()

    assert loaded == ["wandb"]
    assert client.initialized == ("grammar", "lab", "trial")
    assert client.run.logged == [
        ({"loss": 0.25}, 4),
        (
            {
                "scores": FakeTable(
                    ("score", "split"),
                    ((0.5, "dev"),),
                )
            },
            None,
        ),
    ]
    assert client.run.finished


def test_an_unavailable_selected_client_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(wandb_module.importlib, "import_module", missing)

    with pytest.raises(MissingTelemetryBackendError, match=r"lofi\[tracking\]"):
        PLUGIN.construct(WandbSinkSpec(project="grammar"), POINT)


def test_a_run_the_declaration_leaves_unnamed_is_named_after_its_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(wandb_module.importlib, "import_module", lambda _: client)

    PLUGIN.construct(WandbSinkSpec(project="grammar"), POINT)

    assert client.initialized == ("grammar", None, "0123abcd")
