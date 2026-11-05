import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

import pytest
from tests.support.installed import installed

from lofi.registry import Registry
from lofi.run import MetricSink, SinkContext, resolve_metric_sinks
from lofi.spec.sinks.mlflow import MlflowSinkSpec
from lofi.spec.sinks.trackio import TrackioSinkSpec
from lofi.spec.sinks.wandb import WandbSinkSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.spec import SinkSpec

type FamilyTag = tuple[str, str]
type SinkFactory = Callable[[Mapping[str, JsonValue]], MetricSink]

_GROUP_PREFIX: Final = "lofi."

_SINK: Final = "sink"
_WANDB: Final = "wandb"
_TRACKIO: Final = "trackio"
_MLFLOW: Final = "mlflow"

_BROKEN_TAG: Final = "broken-local-member"


_STEP: Final = 3
_SCALARS: Final = MappingProxyType({"loss/semantics": 0.5, "loss/syntax": 0.25})
_TABLE_NAME: Final = "scores"
_ROWS: Final = (
    {"split": "dev", "score": 0.5, "arm": "b"},
    {"split": "test", "arm": "a"},
)
_COLUMNS: Final = ("arm", "score", "split")
_TABLE_DATA: Final = (("b", 0.5, "dev"), ("a", None, "test"))


class _DiscoveredMember(Protocol):
    """A registered member, with no operation this suite requires."""


class ConstructionCase(NamedTuple):
    """One entry point and the check that constructs it."""

    family: str
    tag: str
    check: Callable[[], None]


class FakeTable(NamedTuple):
    """The columns and rows a fake telemetry backend was asked to build."""

    columns: tuple[str, ...]
    data: tuple[tuple[JsonValue, ...], ...]


class BrokenMemberError(RuntimeError):
    """Raised by a deliberately broken member instead of a sink being built."""


def _table(
    *,
    columns: Sequence[str],
    data: Sequence[Sequence[JsonValue]],
) -> FakeTable:
    """Return a fake table from columns and rows."""
    return FakeTable(tuple(columns), tuple(tuple(row) for row in data))


class FakeWandbRun:
    """Records the run calls the W&B sink makes."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float | FakeTable], int | None]] = []
        self.finished = False

    def log(
        self,
        data: Mapping[str, float | FakeTable],
        *,
        step: int | None = None,
    ) -> None:
        """Record one logged mapping and its step."""
        self.logged.append((dict(data), step))

    def finish(self) -> None:
        """Record that the run was finished."""
        self.finished = True


class FakeWandb:
    """Records the module-level calls the W&B sink makes."""

    def __init__(self) -> None:
        self.run = FakeWandbRun()
        self.initialized: tuple[str, str | None, str | None] | None = None

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
    ) -> FakeWandbRun:
        """Record the run settings and return the run."""
        self.initialized = (project, entity, name)

        return self.run


class FakeTrackio:
    """Records the calls the Trackio sink makes."""

    def __init__(self) -> None:
        self.initialized: tuple[str, str | None, str | None, str | None] | None = None
        self.logged: list[tuple[dict[str, float | FakeTable], int | None]] = []
        self.finished = False

    def init(
        self,
        *,
        project: str,
        name: str | None,
        space_id: str | None,
        server_url: str | None,
    ) -> None:
        """Record the run settings."""
        self.initialized = (project, name, space_id, server_url)

    def log(
        self,
        data: Mapping[str, float | FakeTable],
        *,
        step: int | None = None,
    ) -> None:
        """Record one logged mapping and its step."""
        self.logged.append((dict(data), step))

    def finish(self) -> None:
        """Record that the run was finished."""
        self.finished = True


class FakeMlflowMetric(NamedTuple):
    """One metric entry the fake tracking client was asked to log."""

    key: str
    value: float
    timestamp: int
    step: int


class FakeMlflowExperiment(NamedTuple):
    """An experiment record of the fake tracking client."""

    experiment_id: str


class FakeMlflowRunInfo(NamedTuple):
    """The identifying part of a run record of the fake tracking client."""

    run_id: str


class FakeMlflowRun(NamedTuple):
    """A run record of the fake tracking client."""

    info: FakeMlflowRunInfo


class FakeMlflow:
    """Records the tracking-client calls the MLflow sink makes."""

    def __init__(self) -> None:
        self.tracking_uris: list[str | None] = []
        self.started: tuple[str, str | None] | None = None
        self.metrics: list[tuple[str, str, float, int]] = []
        self.tables: list[tuple[dict[str, tuple[JsonValue, ...]], str]] = []
        self.ended: list[str] = []

    def connect(self, *, tracking_uri: str | None) -> FakeMlflow:
        """Record the tracking store a client is bound to and return the client."""
        self.tracking_uris.append(tracking_uri)

        return self

    def get_experiment_by_name(self, name: str) -> FakeMlflowExperiment | None:
        """Return no experiment, so the sink creates its own."""
        del name

        return None

    def create_experiment(self, name: str) -> str:
        """Return the identifier of a created experiment."""
        return f"id-{name}"

    def create_run(self, experiment_id: str, *, run_name: str | None) -> FakeMlflowRun:
        """Record the started run and return its record."""
        self.started = (experiment_id, run_name)

        return FakeMlflowRun(FakeMlflowRunInfo("run"))

    def log_batch(self, run_id: str, *, metrics: Sequence[FakeMlflowMetric]) -> None:
        """Record each scalar of one batch and its step."""
        self.metrics.extend(
            (run_id, metric.key, metric.value, metric.step) for metric in metrics
        )

    def log_table(
        self,
        run_id: str,
        data: Mapping[str, Sequence[JsonValue]],
        artifact_file: str,
    ) -> None:
        """Record one table artifact."""
        del run_id
        self.tables.append(
            ({column: tuple(values) for column, values in data.items()}, artifact_file)
        )

    def set_terminated(self, run_id: str) -> None:
        """Record that the run was ended."""
        self.ended.append(run_id)


@contextmanager
def _injected(name: str, module: ModuleType) -> Iterator[None]:
    """Install an inert backend module under one import name.

    Parameters
    ----------
    name
        The import name the factory asks for.
    module
        The inert module bound to that name for the duration.
    """
    previous = sys.modules.get(name)
    sys.modules[name] = module

    try:
        yield

    finally:
        if previous is None:
            del sys.modules[name]

        else:
            sys.modules[name] = previous


class Declaration(NamedTuple):
    """What the project manifest declares about its own component families."""

    distribution: str
    families: tuple[str, ...]


def _declaration(tree: Path) -> Declaration:
    """Read the distribution name and the declared families from the manifest.

    Parameters
    ----------
    tree
        The repository checkout holding the project manifest.

    Returns
    -------
    Declaration
        The distribution name, and the families it declares an entry-point group
        for, in lexical order.
    """
    manifest = tomllib.loads((tree / "pyproject.toml").read_text(encoding="utf-8"))
    groups = manifest["project"]["entry-points"]

    return Declaration(
        str(manifest["project"]["name"]),
        tuple(
            sorted(
                str(group).removeprefix(_GROUP_PREFIX)
                for group in groups
                if str(group).startswith(_GROUP_PREFIX)
            )
        ),
    )


_DISTRIBUTION, _FAMILIES = _declaration(Path(__file__).resolve().parents[2])


def _project_tags(family: str) -> tuple[str, ...]:
    """Return the tags this project's own distribution registers in one family.

    Parameters
    ----------
    family
        The family to discover.

    Returns
    -------
    tuple[str, ...]
        Tags in lexical order, empty where the project registers none.
    """
    return tuple(
        sorted(
            point.name
            for point in Registry[_DiscoveredMember](family).entries()
            if point.dist is not None and point.dist.name == _DISTRIBUTION
        )
    )


def _project_pairs() -> tuple[FamilyTag, ...]:
    """Return every project-registered family and tag pair, in lexical order."""
    return tuple(
        sorted((family, tag) for family in _FAMILIES for tag in _project_tags(family))
    )


def _exercise_sink(declaration: SinkSpec) -> None:
    """Build one declared sink and run its whole contract against an inert backend."""
    (sink,) = resolve_metric_sinks({"case": declaration}, SinkContext("case-point"))
    sink.log_scalars(_SCALARS, _STEP)
    sink.log_table(_TABLE_NAME, _ROWS)
    sink.close()


def _check_wandb_sink() -> None:
    client = FakeWandb()
    module = ModuleType(_WANDB)
    module.__dict__.update({"init": client.init, "Table": _table})

    with _injected(_WANDB, module):
        _exercise_sink(
            WandbSinkSpec(project="induction", entity="lofi", run_name="case")
        )

    assert client.initialized == ("induction", "lofi", "case")
    assert client.run.logged == [
        (dict(_SCALARS), _STEP),
        ({_TABLE_NAME: FakeTable(_COLUMNS, _TABLE_DATA)}, None),
    ]
    assert client.run.finished


def _trackio_module(client: FakeTrackio) -> ModuleType:
    """Return an inert Trackio module exposing one recording client's operations."""
    module = ModuleType(_TRACKIO)
    module.__dict__.update(
        {
            "init": client.init,
            "log": client.log,
            "finish": client.finish,
            "Table": _table,
        }
    )

    return module


def _check_trackio_sink() -> None:
    targets: tuple[tuple[str | None, str | None], ...] = (
        ("lofi/induction", None),
        (None, "https://trackio.invalid/runs"),
    )

    for space_id, server_url in targets:
        client = FakeTrackio()

        with _injected(_TRACKIO, _trackio_module(client)):
            _exercise_sink(
                TrackioSinkSpec(
                    project="induction",
                    run_name="case",
                    space_id=space_id,
                    server_url=server_url,
                )
            )

        assert client.initialized == ("induction", "case", space_id, server_url)
        assert client.logged == [
            (dict(_SCALARS), _STEP),
            ({_TABLE_NAME: FakeTable(_COLUMNS, _TABLE_DATA)}, None),
        ]
        assert client.finished


def _mlflow_module(client: FakeMlflow) -> ModuleType:
    """Return an inert MLflow module exposing one recording client's operations."""
    module = ModuleType(_MLFLOW)
    module.__dict__.update({"MlflowClient": client.connect})

    return module


def _mlflow_entities() -> ModuleType:
    """Return an inert MLflow entities module exposing the metric-entry constructor."""
    module = ModuleType(f"{_MLFLOW}.entities")
    module.__dict__.update({"Metric": FakeMlflowMetric})

    return module


def _check_mlflow_sink() -> None:
    for tracking_uri in ("memory://induction", None):
        client = FakeMlflow()

        with (
            _injected(_MLFLOW, _mlflow_module(client)),
            _injected(f"{_MLFLOW}.entities", _mlflow_entities()),
        ):
            _exercise_sink(
                MlflowSinkSpec(
                    experiment_name="induction",
                    tracking_uri=tracking_uri,
                )
            )

        expected = {
            column: tuple(row.get(column) for row in _ROWS) for column in _COLUMNS
        }

        assert client.tracking_uris == [tracking_uri]
        assert client.started == ("id-induction", "case-point")
        assert client.metrics == [
            ("run", name, value, _STEP) for name, value in _SCALARS.items()
        ]
        assert client.tables == [(expected, f"{_TABLE_NAME}.json")]
        assert [tuple(table) for table, _ in client.tables] == [_COLUMNS]
        assert client.ended == ["run"]


def _broken_sink(settings: Mapping[str, JsonValue]) -> MetricSink:
    """Raise instead of building a sink, naming the broken tag."""
    del settings

    raise BrokenMemberError(_BROKEN_TAG)


CASES: Final = (
    ConstructionCase(_SINK, _MLFLOW, _check_mlflow_sink),
    ConstructionCase(_SINK, _TRACKIO, _check_trackio_sink),
    ConstructionCase(_SINK, _WANDB, _check_wandb_sink),
)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{case.family}:{case.tag}" for case in CASES],
)
def test_each_declared_case_resolves_constructs_and_runs(
    case: ConstructionCase,
) -> None:
    case.check()


def test_every_declared_family_is_one_registry_group_segment() -> None:
    assert _FAMILIES
    assert all("." not in family for family in _FAMILIES)
    assert all(
        Registry[_DiscoveredMember](family).group == f"{_GROUP_PREFIX}{family}"
        for family in _FAMILIES
    )


def test_every_project_entry_point_loads_through_its_family_registry() -> None:
    pairs = _project_pairs()

    assert pairs

    for family, tag in pairs:
        assert Registry[_DiscoveredMember](family).get(tag) is not None


def test_a_broken_local_member_propagates_its_construction_failure() -> None:

    registry = installed(_SINK, {_BROKEN_TAG: _broken_sink})
    assert registry.available() == (_BROKEN_TAG,)
    assert (_SINK, _BROKEN_TAG) not in _project_pairs()

    with pytest.raises(BrokenMemberError, match=_BROKEN_TAG):
        registry.get(_BROKEN_TAG)({})

    assert registry.available() == (_BROKEN_TAG,)
