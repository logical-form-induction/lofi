from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi._checks import LawViolationError
from lofi.run import METRIC_SINK_LAWS, MetricSinkFixture, NullSink
from lofi.run import unit_tests_do_not_import_telemetry_backends as unit_import_law

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue


class RecordingMetricSink:
    def __init__(self) -> None:
        self.scalars: list[tuple[dict[str, float], int]] = []

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        self.scalars.append((dict(mapping), step))

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        del name, rows

    def close(self) -> None:
        return


def _keys_from_loss_registry(loss_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"loss/{name}" for name in loss_names)


_REGISTRY_KEYS = _keys_from_loss_registry(("syntax", "semantics"))


def _fixture(root: Path) -> MetricSinkFixture:
    return MetricSinkFixture(
        NullSink(),
        root,
        frozenset({"wandb", "mlflow", "trackio"}),
        _REGISTRY_KEYS,
    )


def test_the_null_sink_satisfies_every_metric_sink_law() -> None:
    root = Path(__file__).parents[3]
    fixture = _fixture(root)

    for law in METRIC_SINK_LAWS:
        law(fixture)


@pytest.mark.parametrize(
    "source",
    [
        'import importlib as il\nil.import_module("wandb")\n',
        'from importlib import import_module as load\nload("wandb")\n',
    ],
)
def test_the_import_law_follows_importlib_aliases(tmp_path: Path, source: str) -> None:
    unit = tmp_path / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_backend.py").write_text(source, encoding="utf-8", newline="\n")

    with pytest.raises(LawViolationError, match="unit tests import telemetry"):
        unit_import_law(_fixture(tmp_path))


@pytest.mark.parametrize(
    "source",
    [
        """
def import_module(name: str) -> str:
    return name

import_module("wandb")
""",
        """
class Loader:
    def import_module(self, name: str) -> str:
        return name

Loader().import_module("wandb")
""",
    ],
)
def test_the_import_law_ignores_unrelated_callables(
    tmp_path: Path, source: str
) -> None:
    unit = tmp_path / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_helper.py").write_text(source, encoding="utf-8", newline="\n")

    unit_import_law(_fixture(tmp_path))


def test_the_metric_key_law_rejects_an_empty_registry_key_set() -> None:
    fixture = MetricSinkFixture(NullSink(), Path(__file__).parents[3], frozenset(), ())

    with pytest.raises(LawViolationError, match="registry supplied no metric keys"):
        METRIC_SINK_LAWS[2](fixture)


def test_the_metric_key_law_passes_registry_keys_as_scalar_data(tmp_path: Path) -> None:
    sink = RecordingMetricSink()
    fixture = MetricSinkFixture(sink, tmp_path, frozenset(), _REGISTRY_KEYS)

    METRIC_SINK_LAWS[2](fixture)

    assert sink.scalars == [({"loss/syntax": 1.0, "loss/semantics": 2.0}, 3)]
