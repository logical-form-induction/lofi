import json
import os
import sqlite3
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import pytest
import torch

from lofi._types import ItemId
from lofi.data import Derivation, DerivationNode, TokenSpan
from lofi.deduce import parse
from lofi.nn.tokens.pretrained import (
    CrossingPieceError,
    WordRows,
    ZeroPieceWordError,
)
from lofi.nn.tokens.window import ContextOverflowError
from lofi.probe import (
    AlignedInput,
    AlignmentExclusion,
    AlignmentPolicy,
    ProbeRow,
    TokenizerAlignment,
    UnpublishedSiteError,
    observed_pass,
)
from lofi.probe._probing import GradientMode
from lofi.run import (
    RESOLVED_NAME,
    FailureKind,
    MetricsRecorder,
    RandomStreams,
    SeedPolicy,
    Selection,
    SweepPoint,
    _entrypoint,
    operations,
    plan_training,
    read_failure,
    resolve_device_policy,
)
from lofi.run.probing import (
    PROBE_TABLE,
    PROBES_NAME,
    ProbeExecution,
    TrainedParser,
    UnprobeableRunError,
    _gold,
    aligned_population,
    counterfactual_generator,
    pooled_leaves,
    probe_document,
    read_candidate_sentences,
    read_probe_rows,
    report_probe_rows,
    write_probe_rows,
)
from lofi.spec import RunSpec, RunSpecCodec
from lofi.stats import BenjaminiHochberg, Dependence, p_value_against_replicates
from lofi.train import prepare_training

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.nn.ports import SubwordPooler, TokenEncoder
    from lofi.train import ParserFittingPlan, PreparedTraining

CONFIG: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)
PACKAGED_CODEC: Final = RunSpecCodec(search_path=(CONFIG.parent, CONFIG.parent.parent))

INTERVENTION: Final = """\
probes:
  patch:
    kind: intervention
    interventions:
      - site: {component: parser, role: RECEIVER_EDGE, key: [receiver]}
        intervention: {kind: zero}
"""


class RecordingSink:
    def __init__(self) -> None:
        self.tables: dict[str, list[Mapping[str, JsonValue]]] = {}
        self.closed = False

    def log_scalars(self, mapping: Mapping[str, float], step: int) -> None:
        del mapping, step

    def log_table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        self.tables[name] = list(rows)

    def close(self) -> None:
        self.closed = True


def lofi(*argv: str) -> int:
    return _entrypoint.main(list(argv))


def run_directory(root: Path) -> Path:
    (directory,) = (
        path for path in (root / "runs").iterdir() if not path.name.startswith(".")
    )
    return directory


def digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def probed_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("probed")
    status = lofi(
        "run",
        "--config",
        str(CONFIG),
        "--set",
        "probes.diag.kind=diagnostics",
        "--experiment-seed",
        "5",
        "--root",
        str(root),
    )

    assert status == 0

    return run_directory(root)


def test_lofi_run_with_a_declared_probe_writes_its_rows(probed_run: Path) -> None:
    rows = read_probe_rows(probed_run / PROBES_NAME)
    measured = [row for row in rows if not row.measure.endswith("/refused")]

    assert rows
    assert {(row.probe, row.kind) for row in rows} == {("diag", "diagnostics")}
    assert measured
    assert all(
        row.site is not None and row.site.startswith("parser/") for row in measured
    )
    assert all(row.count >= 1 for row in rows)


def test_a_probe_declaring_an_unpublished_role_is_refused_before_training(
    tmp_path: Path,
) -> None:
    status = lofi(
        "run",
        "--config",
        str(CONFIG),
        "--set",
        "probes.diag.kind=diagnostics",
        "--set",
        'probes.diag.roles=["ATTENTION_WEIGHTS"]',
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path),
    )
    directory = run_directory(tmp_path)
    failure = read_failure(directory)
    checkpoints = directory / "checkpoints"

    assert status == 1
    assert failure is not None
    assert failure.kind is FailureKind.PREPARATION
    assert "ATTENTION_WEIGHTS" in failure.diagnostic
    assert not checkpoints.exists() or not any(checkpoints.iterdir())
    assert not (directory / PROBES_NAME).exists()


TOY_PROBE: Final = """\
from typing import Literal, NamedTuple

from lofi.probe import ProbeValue
from lofi.probe._probing import ProbeBuildContext
from lofi.registry import ComponentPlugin
from lofi.spec import ProbeSpec


class ToySettings(NamedTuple):
    scale: float


class ToySpec(ProbeSpec):
    kind: Literal["toy"] = "toy"
    scale: float = 2.0

    def settings(self) -> ToySettings:
        return ToySettings(self.scale)


class Toy:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def admit(self, published) -> None:
        del published

    def run(self, point):
        return (
            ProbeValue(
                "components",
                None,
                self.scale * len(point.components),
                len(point.passes),
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )


def build(settings: ToySettings, context: ProbeBuildContext, /) -> Toy:
    if context.role != "mine":
        raise ValueError("the probe was built under another role")
    return Toy(settings.scale)


PLUGIN = ComponentPlugin(spec=ToySpec, build=build)
"""


def test_adding_a_probe_takes_one_module_and_one_entry_point_line(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    distribution = plugins / "toy_probe-0.1.dist-info"
    distribution.mkdir(parents=True)
    (plugins / "toy_probe.py").write_text(TOY_PROBE, encoding="utf-8", newline="\n")
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: toy-probe\nVersion: 0.1\n",
        encoding="utf-8",
        newline="\n",
    )
    (distribution / "entry_points.txt").write_text(
        "[lofi.probe]\ntoy = toy_probe:PLUGIN\n", encoding="utf-8", newline="\n"
    )
    root = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lofi.run._entrypoint import main; sys.exit(main())",
            "run",
            "--config",
            str(CONFIG),
            "--set",
            "probes.mine.kind=toy",
            "--set",
            "probes.mine.scale=0.5",
            "--experiment-seed",
            "5",
            "--root",
            str(root),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(plugins)},
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    rows = read_probe_rows(run_directory(root) / PROBES_NAME)

    assert rows == (
        ProbeRow(
            "mine",
            "toy",
            "components",
            None,
            1.0,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def test_the_probe_rows_reach_the_declared_sinks(tmp_path: Path) -> None:
    document = json.loads(PACKAGED_CODEC(CONFIG).model_dump_json())
    document["probes"] = {"diag": {"kind": "diagnostics"}}
    blueprint = plan_training(RunSpec.model_validate(document))
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    sink = RecordingSink()
    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        tmp_path,
        streams,
        resolve_device_policy(blueprint.run_spec),
        scoring=None,
        sinks=(sink,),
        recorder=MetricsRecorder(tmp_path, SweepPoint(document)),
    )
    written = json.loads((tmp_path / PROBES_NAME).read_text(encoding="utf-8"))

    assert written
    assert sink.tables == {PROBE_TABLE: written}


def test_a_run_declaring_no_probe_writes_and_reports_no_probe_rows(
    tmp_path: Path,
) -> None:
    run_spec = PACKAGED_CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    sink = RecordingSink()
    operations.TRAINING_OPERATIONS.execute(
        blueprint,
        tmp_path,
        streams,
        resolve_device_policy(blueprint.run_spec),
        scoring=None,
        sinks=(sink,),
        recorder=MetricsRecorder(
            tmp_path, SweepPoint(json.loads(run_spec.model_dump_json()))
        ),
    )

    assert not (tmp_path / PROBES_NAME).exists()
    assert PROBE_TABLE not in sink.tables


@pytest.fixture(scope="module")
def patched_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("patched")
    overlay = root / "intervention.yaml"
    overlay.write_text(INTERVENTION, encoding="utf-8", newline="\n")
    status = lofi(
        "run",
        "--config",
        str(CONFIG),
        "--extra",
        str(overlay),
        "--experiment-seed",
        "5",
        "--root",
        str(root / "store"),
    )

    assert status == 0

    return run_directory(root / "store")


def test_lofi_probe_reads_the_weights_the_run_trained(
    patched_run: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "rows.json"

    assert lofi("probe", "--run", str(patched_run), "--out", str(destination)) == 0
    assert read_probe_rows(destination) == read_probe_rows(patched_run / PROBES_NAME)


def test_lofi_probe_reports_to_sinks_it_declares_and_closes_them(
    patched_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    store = tmp_path / "mlflow.db"
    overlay = tmp_path / "tracking.yaml"
    overlay.write_text(
        "sinks:\n"
        "  tracking:\n"
        "    kind: mlflow\n"
        "    experiment_name: lofi\n"
        f"    tracking_uri: sqlite:///{store}\n",
        encoding="utf-8",
        newline="\n",
    )
    destination = tmp_path / "rows.json"

    status = lofi(
        "probe",
        "--run",
        str(patched_run),
        "--out",
        str(destination),
        "--extra",
        str(overlay),
    )

    with sqlite3.connect(store) as database:
        runs = database.execute("SELECT status FROM runs").fetchall()

    assert status == 0
    assert read_probe_rows(destination)
    assert runs == [("FINISHED",)]


def test_lofi_probe_refuses_a_run_that_did_not_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = lofi(
        "run",
        "--config",
        str(CONFIG),
        "--set",
        "probes.diag.kind=diagnostics",
        "--set",
        'probes.diag.roles=["ATTENTION_WEIGHTS"]',
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path / "store"),
    )
    failed = run_directory(tmp_path / "store")
    destination = tmp_path / "rows.json"

    capsys.readouterr()
    refused = lofi(
        "probe",
        "--run",
        str(failed),
        "--out",
        str(destination),
        "--set",
        'probes.diag.roles=["CHART_CELL"]',
    )

    assert status == 1
    assert refused == 2
    assert "records no completed run" in capsys.readouterr().err
    assert not destination.exists()


def test_lofi_probe_adds_probes_to_a_run_trained_without_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    trained = lofi(
        "run", "--config", str(CONFIG), "--experiment-seed", "5", "--root", str(store)
    )
    run = run_directory(store)
    bare = tmp_path / "bare.json"
    probed = tmp_path / "probed.json"
    capsys.readouterr()

    refused = lofi("probe", "--run", str(run), "--out", str(bare))
    message = capsys.readouterr().err
    status = lofi(
        "probe",
        "--run",
        str(run),
        "--out",
        str(probed),
        "--set",
        "probes.diag.kind=diagnostics",
    )

    assert trained == 0
    assert not (run / PROBES_NAME).exists()
    assert refused == 2
    assert "declares no probe" in message
    assert not bare.exists()
    assert status == 0
    assert {(row.probe, row.kind) for row in read_probe_rows(probed)} == {
        ("diag", "diagnostics")
    }


def test_lofi_probe_executes_against_the_trained_point_and_leaves_the_run(
    probed_run: Path, tmp_path: Path
) -> None:
    overlay = tmp_path / "intervention.yaml"
    overlay.write_text(INTERVENTION, encoding="utf-8", newline="\n")
    before = digests(probed_run)
    destination = tmp_path / "rows.json"

    status = lofi(
        "probe",
        "--run",
        str(probed_run),
        "--out",
        str(destination),
        "--extra",
        str(overlay),
    )
    rows = read_probe_rows(destination)
    recorded = read_probe_rows(probed_run / PROBES_NAME)

    assert status == 0
    assert digests(probed_run) == before
    assert [row for row in rows if row.probe == "diag"] == list(recorded)
    assert [
        (row.probe, row.measure, row.site) for row in rows if row.probe == "patch"
    ] == [("patch", "zero/effect", "parser/RECEIVER_EDGE/receiver")]


@pytest.mark.parametrize(
    "overrides",
    [
        ("--set", "trainer.epochs=3"),
        ("--set", 'probes.diag.roles=["ATTENTION_WEIGHTS"]'),
    ],
)
def test_lofi_probe_refuses_what_the_trained_point_cannot_answer(
    probed_run: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    overrides: tuple[str, ...],
) -> None:
    destination = tmp_path / "rows.json"
    status = lofi(
        "probe", "--run", str(probed_run), "--out", str(destination), *overrides
    )

    assert status == 2
    assert not destination.exists()
    assert capsys.readouterr().err.startswith("lofi probe: ")


def test_lofi_probe_refuses_an_existing_destination_and_an_untrained_directory(
    probed_run: Path, tmp_path: Path
) -> None:
    taken = tmp_path / "taken.json"
    taken.write_text("[]\n", encoding="utf-8", newline="\n")

    assert lofi("probe", "--run", str(probed_run), "--out", str(taken)) == 2
    assert taken.read_text(encoding="utf-8") == "[]\n"
    assert (
        lofi("probe", "--run", str(tmp_path), "--out", str(tmp_path / "rows.json")) == 2
    )


@pytest.fixture(scope="module")
def prepared() -> PreparedTraining:
    run_spec = PACKAGED_CODEC(
        CONFIG, overrides=("data.datasets.train.dataset.size=16",)
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()

    return prepare_training(blueprint, streams, resolve_device_policy(run_spec))


def with_probes(probes: dict[str, JsonValue]) -> RunSpec:
    return PACKAGED_CODEC(
        CONFIG,
        overrides=(
            "data.datasets.train.dataset.size=16",
            *(f"probes.{k}={json.dumps(v)}" for k, v in probes.items()),
        ),
    )


def test_a_run_without_probes_composes_and_executes_nothing(
    prepared: PreparedTraining,
) -> None:
    execution = ProbeExecution(PACKAGED_CODEC(CONFIG), prepared)

    assert execution.roles == ()
    assert execution.execute() == ((), {})


def test_the_trained_parser_offers_its_instrumentable_components_and_held_out_passes(
    prepared: PreparedTraining,
) -> None:
    point = TrainedParser(prepared)
    layouts = [held_out(GradientMode.DISABLED) for held_out in point.passes]

    assert sorted(point.components) == ["parser", "terms"]
    assert len(layouts) == 2
    assert all(layout.spans is not None for layout in layouts)
    assert all(gold is not None for layout in layouts for gold in layout.gold)
    assert sum(len(layout.words) for layout in layouts) == 4


@pytest.mark.parametrize(
    "gradients",
    [GradientMode.DISABLED, GradientMode.ENABLED],
)
def test_a_held_out_pass_restores_every_mode_and_uses_its_declared_gradient_mode(
    prepared: PreparedTraining,
    request: pytest.FixtureRequest,
    gradients: GradientMode,
) -> None:
    point = TrainedParser(prepared)
    modules = prepared.model.modules
    parser = modules["parser"]
    parser.eval()
    modules["controller"].train()

    def restore() -> None:
        for module in modules.values():
            module.train()

    request.addfinalizer(restore)
    modes = [
        (role, name, module.training)
        for role, root in modules.items()
        for name, module in root.named_modules()
    ]
    observed = frozenset(
        site
        for site, spec in point.components["parser"].site_map().items()
        if spec.differentiable
    )

    grad_modes: list[bool] = []

    def record(module: torch.nn.Module, inputs: object) -> None:
        del module, inputs
        grad_modes.append(torch.is_grad_enabled())

    hooks = [
        module.register_forward_pre_hook(record)
        for root in modules.values()
        for module in root.modules()
    ]

    try:
        trace, _ = observed_pass(
            point,
            point.passes[0],
            observe=observed,
            patch={},
            gradients=gradients,
        )

    finally:
        for hook in hooks:
            hook.remove()

    assert grad_modes
    assert all(grad_modes) is (gradients is GradientMode.ENABLED)
    assert modules["controller"].training
    assert not parser.training
    assert [
        (role, name, module.training)
        for role, root in modules.items()
        for name, module in root.named_modules()
    ] == modes
    assert trace.sites()
    assert not any(trace[site].tensor.requires_grad for site in trace.sites())


def test_a_unary_chain_s_span_is_labeled_by_its_highest_node() -> None:
    derivation = Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 2), "S", rule="apply", children=(1, 3)),
            1: DerivationNode(TokenSpan(0, 1), "NP", rule="lift", children=(2,)),
            2: DerivationNode(TokenSpan(0, 1), "N"),
            3: DerivationNode(TokenSpan(1, 2), "S\\NP"),
        },
    )
    example = SimpleNamespace(
        supervision=SimpleNamespace(derivation=derivation),
        item=SimpleNamespace(item_id=ItemId("item")),
    )
    plan = SimpleNamespace(corpus=SimpleNamespace(examples=(example,)))

    gold = _gold(cast("ParserFittingPlan", plan))

    assert dict(gold[ItemId("item")]) == {(0, 2): "S", (0, 1): "NP", (1, 2): "S\\NP"}


def test_every_declared_arm_executes_against_the_parser_in_role_order(
    prepared: PreparedTraining,
) -> None:
    run_spec = with_probes(
        {
            "z": {"kind": "diagnostics", "roles": ["CHART_CELL"]},
            "a": {"kind": "classifier", "epochs": 5},
            "m": {
                "kind": "intervention",
                "interventions": [
                    {
                        "site": {
                            "component": "parser",
                            "role": "RECEIVER_EDGE",
                            "key": ["receiver"],
                        },
                        "intervention": {"kind": "noise", "sigma": 0.5},
                    }
                ],
            },
        }
    )
    execution = ProbeExecution(run_spec, prepared)
    rows = execution.execute().rows

    assert execution.roles == ("a", "m", "z")
    assert [row.probe for row in rows] == sorted(row.probe for row in rows)
    assert {row.kind for row in rows} == {"classifier", "intervention", "diagnostics"}
    assert ("m", "noise/effect") in {(row.probe, row.measure) for row in rows}


def test_a_probe_naming_an_unpatchable_site_is_refused_at_composition(
    prepared: PreparedTraining,
) -> None:
    run_spec = with_probes(
        {
            "m": {
                "kind": "intervention",
                "interventions": [
                    {
                        "site": {
                            "component": "parser",
                            "role": "INSIDE_SCORE",
                            "key": ["inside"],
                        },
                        "intervention": {"kind": "zero"},
                    }
                ],
            }
        }
    )

    with pytest.raises(ValueError, match="parser/INSIDE_SCORE/inside"):
        ProbeExecution(run_spec, prepared)


def test_a_role_published_where_no_held_out_pass_reaches_it_is_refused(
    prepared: PreparedTraining,
) -> None:
    point = TrainedParser(prepared)
    published = {
        site.role
        for component in point.components.values()
        for site in component.site_map()
    }
    run_spec = with_probes({"d": {"kind": "diagnostics", "roles": ["GATE"]}})

    assert "GATE" in {role.name for role in published}

    with pytest.raises(UnpublishedSiteError, match="GATE"):
        ProbeExecution(run_spec, prepared)


def test_probe_rows_round_trip_and_reach_every_sink_as_one_table(
    tmp_path: Path,
) -> None:
    rows = (
        ProbeRow(
            "diag",
            "diagnostics",
            "finite-readings/live",
            "parser/LOG_PARTITION/log_z",
            1.0,
            2,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        ProbeRow(
            "diag",
            "diagnostics",
            "cutpoint-entropy/refused",
            None,
            2.0,
            2,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    path = tmp_path / PROBES_NAME
    sinks = (RecordingSink(), RecordingSink())

    write_probe_rows(path, rows)
    report_probe_rows(sinks, rows)

    assert read_probe_rows(path) == rows
    assert all(
        sink.tables == {PROBE_TABLE: json.loads(path.read_text(encoding="utf-8"))}
        for sink in sinks
    )


def test_an_inferential_probe_row_round_trips_real_null_and_bh_fdr_metadata(
    tmp_path: Path,
) -> None:
    p_value = p_value_against_replicates(
        2.0,
        (0.0, 1.0, 2.0),
        tail="greater",
        null="label permutation",
        family="permutation",
        exact=True,
        seed=None,
    )
    decision = BenjaminiHochberg().apply((p_value,), 0.05, Dependence.PRDS)
    row = ProbeRow(
        "causal",
        "causal",
        "iia",
        None,
        0.75,
        8,
        p_value.provenance,
        p_value.value,
        decision.adjusted[0],
        decision.reject[0],
        "cell-wise-iia",
        Dependence.PRDS,
    )
    path = tmp_path / PROBES_NAME

    write_probe_rows(path, (row,))

    assert read_probe_rows(path) == (row,)
    assert json.loads(path.read_text(encoding="utf-8"))[0]["null"] == {
        "name": "label permutation",
        "family": "permutation",
        "replicates": 3,
        "tail": "greater",
        "exact": True,
        "seed": None,
        "matching": {},
    }


class FailingAligner:
    """Aligns every word to its own row except the words it is told to fail on."""

    def __init__(self, identity: str, failures: Mapping[str, Exception]) -> None:
        self._identity = identity
        self._failures = failures

    @property
    def identity(self) -> str:
        return self._identity

    def align(self, words: tuple[str, ...], /) -> WordRows:
        for word in words:
            if word in self._failures:
                raise self._failures[word]

        return WordRows(tuple((row,) for row in range(len(words))), (), len(words))


def test_an_unalignable_sentence_is_excluded_by_reason_under_each_checkpoint() -> None:
    sentences = (("a", "b"), ("a", "gap"), ("a", "cross"), ("long", "b"))
    first = FailingAligner(
        "first",
        {
            "gap": ZeroPieceWordError((1,)),
            "cross": CrossingPieceError("a cross", (0, 3), (0, 1)),
            "long": ContextOverflowError(("long", "b"), rows=1, needed=2),
        },
    )
    second = FailingAligner(
        "second", {"long": ContextOverflowError(("long", "b"), rows=1, needed=2)}
    )
    inputs, exclusions = aligned_population(sentences, (first, second))

    assert [aligned.tokens for aligned in inputs] == [("a", "b")]
    assert inputs[0].tokenizers == ("first", "second")
    assert exclusions == (
        AlignmentExclusion(("a", "gap"), "first", "zero-piece-word"),
        AlignmentExclusion(("a", "cross"), "first", "crossing-piece"),
        AlignmentExclusion(("long", "b"), "first", "context-overflow"),
        AlignmentExclusion(("long", "b"), "second", "context-overflow"),
    )


def test_an_alignment_failure_of_no_declared_reason_still_stops_the_population() -> (
    None
):
    broken = FailingAligner("broken", {"b": ValueError("inconsistent dimensions")})

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        aligned_population((("a", "b"),), (broken,))


POPULATION: Final[dict[str, JsonValue]] = {
    "sentences": "candidates.txt",
    "unary_depth": 0,
    "readings": 4,
    "render": "declared",
    "pieces": "all",
    "specials": "exclude",
    "contiguous": True,
    "seed": 2,
}


def with_counterfactuals(population: dict[str, JsonValue]) -> RunSpec:
    return PACKAGED_CODEC(
        CONFIG,
        overrides=(
            "data.datasets.train.dataset.size=16",
            f"counterfactuals={json.dumps(population)}",
        ),
    )


def test_the_counterfactual_generator_derives_under_the_run_s_own_grammar(
    prepared: PreparedTraining,
) -> None:
    plan = cast("ParserFittingPlan", prepared.plan)
    generator = counterfactual_generator(prepared, with_counterfactuals(POPULATION))

    assert generator.grammar == plan.grammar.grammar.fingerprint
    assert generator.readings == POPULATION["readings"]


def test_the_grammar_s_declared_threads_reach_the_counterfactual_parser() -> None:
    run_spec = PACKAGED_CODEC(
        CONFIG,
        overrides=(
            "data.datasets.train.dataset.size=16",
            "data.datasets.train.dataset.grammar.budget.threads=2",
            f"counterfactuals={json.dumps(POPULATION)}",
        ),
    )
    blueprint = plan_training(run_spec)
    streams = RandomStreams(SeedPolicy(3, samplers=blueprint.sampler_streams))
    streams.apply()
    threaded = prepare_training(blueprint, streams, resolve_device_policy(run_spec))
    generator = counterfactual_generator(threaded, run_spec)
    tokens = ("word-0", "word-1")
    words = TokenizerAlignment("words", ((0,), (1,)), (), len(tokens))

    with patch("lofi.probe._counterfactuals.parse", wraps=parse) as parser:
        generator.generate(
            (AlignedInput(tokens, (words,)),),
            policy=AlignmentPolicy("all", "exclude", contiguous=True),
            dataset="unit",
            seed=0,
        )

    assert parser.call_args is not None
    assert parser.call_args.kwargs["threads"] == 2


def test_a_run_declaring_no_counterfactual_population_builds_no_generator(
    prepared: PreparedTraining,
) -> None:
    with pytest.raises(ValueError, match="counterfactuals"):
        counterfactual_generator(prepared, PACKAGED_CODEC(CONFIG))


def test_candidate_sentences_are_read_one_per_line(tmp_path: Path) -> None:
    path = tmp_path / "candidates.txt"
    path.write_text(
        "word-0 word-1\nword-3  word-4 word-5\n", encoding="utf-8", newline="\n"
    )

    assert read_candidate_sentences(path) == (
        ("word-0", "word-1"),
        ("word-3", "word-4", "word-5"),
    )

    path.write_text("word-0\n\nword-1\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="line 2"):
        read_candidate_sentences(path)


def test_pooled_leaves_are_the_trained_point_s_own_encoding_of_the_sentences(
    prepared: PreparedTraining,
) -> None:
    sentences = (("word-0", "word-1"), ("word-3", "word-4", "word-5"))
    leaves = pooled_leaves(prepared, sentences)
    modules = prepared.model.modules
    tokens = cast("TokenEncoder", modules["token_encoder"])
    pooler = cast("SubwordPooler", modules["subword_pooler"])

    with torch.no_grad():
        expected = pooler.pool(tokens.encode_words(sentences))

    assert leaves.axes == expected.axes
    assert leaves.size("word") == 3
    assert torch.equal(leaves.tensor, expected.tensor)


def test_a_probe_document_may_change_the_counterfactual_population_alone(
    probed_run: Path,
) -> None:
    population = json.dumps(POPULATION)
    composed = probe_document(
        probed_run,
        Selection(
            probed_run / RESOLVED_NAME,
            None,
            (f"counterfactuals={population}",),
            (),
            (CONFIG.parent, CONFIG.parent.parent),
        ),
    )

    assert composed.counterfactuals is not None
    assert composed.counterfactuals.settings().seed == POPULATION["seed"]

    with pytest.raises(UnprobeableRunError, match="trainer"):
        probe_document(
            probed_run,
            Selection(
                probed_run / RESOLVED_NAME,
                None,
                ("trainer.epochs=3",),
                (),
                (CONFIG.parent, CONFIG.parent.parent),
            ),
        )
