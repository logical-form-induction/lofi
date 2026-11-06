import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, get_args

import didactic.api as dx
import pytest
from didactic.settings import InterpolationError, OriginKind
from tests.conformance.config_loader import (
    ConfigLoaderConformance,
    ConfigLoaderFixture,
    Layer,
    Request,
    leaves,
)

from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ConfigDocumentError,
    ConfigError,
    ConfigSource,
    ConfigValueError,
    DatasetSpec,
    FeaturizerSpec,
    LeafOrigin,
    LossSpec,
    MalformedOverrideError,
    MissingSettingError,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    RunSpecCodec,
    SettingParseError,
    SinkSpec,
    SpecLoader,
    TermEncoderSpec,
    TrainerSpec,
    UnknownSettingError,
    UnresolvedFragmentError,
)
from lofi.spec import _loader as loader_module
from lofi.spec._loader import SOURCE_OF_KIND
from lofi.spec.grammar import (
    ABFormalismSpec,
    CFGFormalismSpec,
    GrammarDecl,
    GrammarSpec,
    TLGFormalismSpec,
    UndirectedCCGFormalismSpec,
    declare_grammar,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl

TREE: Final = Path(__file__).resolve().parents[3]

TIMEOUT_SECONDS: Final = 120.0

NESTING_CAP: Final = 64
"""The depth the configuration engine evaluates a nested expression to."""


class LoaderParser(ParserSpec):
    kind: Literal["loader-parser"] = "loader-parser"
    beam: int = 1


class LoaderTransformer(TermEncoderSpec):
    kind: Literal["loader-transformer"] = "loader-transformer"
    num_heads: int = 4
    reach: int = 1


class LoaderRnn(TermEncoderSpec):
    kind: Literal["loader-rnn"] = "loader-rnn"
    state_size: int = 32
    reach: str = "near"


class LoaderLoss(LossSpec):
    kind: Literal["loader-loss"] = "loader-loss"


class LoaderAuxiliaryLoss(LossSpec):
    kind: Literal["loader-auxiliary-loss"] = "loader-auxiliary-loss"


class LoaderDataset(DatasetSpec):
    kind: Literal["loader-dataset"] = "loader-dataset"


class LoaderSizedDataset(DatasetSpec):
    kind: Literal["loader-sized-dataset"] = "loader-sized-dataset"
    size: int = 1

    @dx.validates("size")
    @classmethod
    def _size_is_positive(cls, size: int) -> int:
        if size < 1:
            msg = "a dataset holds at least one example"

            raise ValueError(msg)

        return size


class LoaderFeaturizer(FeaturizerSpec):
    kind: Literal["loader-featurizer"] = "loader-featurizer"


class LoaderTrainer(TrainerSpec):
    kind: Literal["loader-trainer"] = "loader-trainer"
    epochs: int = 1
    batch_size: int = 8
    warmup: int = 0

    @dx.validates("warmup")
    @classmethod
    def _warmup_is_not_negative(cls, warmup: int) -> int:
        if warmup < 0:
            msg = "warmup counts steps and cannot be negative"

            raise ValueError(msg)

        return warmup


class LoaderSgd(OptimizerSpec):
    kind: Literal["loader-sgd"] = "loader-sgd"
    lr: float
    momentum: float = 0.0

    @dx.validates("momentum")
    @classmethod
    def _momentum_is_a_fraction(cls, momentum: float) -> float:
        if not 0.0 <= momentum < 1.0:
            msg = "momentum is a fraction of the previous step"

            raise ValueError(msg)

        return momentum


class LoaderSink(SinkSpec):
    kind: Literal["loader-sink"] = "loader-sink"
    every: int = 1
    label: str = ""


RUN_DOCUMENT: Final[dict[str, JsonValue]] = {
    "defaults": [{"model.encoders.terms": "transformer"}],
    "model": {
        "language": {"from": "data"},
        "parser": {"kind": "loader-parser", "beam": 4},
    },
    "loss": {"terms": [{"objective": {"kind": "loader-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "loader-dataset"},
                "featurizer": {"kind": "loader-featurizer"},
            }
        }
    },
    "trainer": {"kind": "loader-trainer", "epochs": 1, "batch_size": 8},
    "parse": {"retain": "forest", "reachability": "all"},
    "release": {"checkpoint": "validation-best"},
    "optimizer": {"kind": "loader-sgd", "lr": 0.1, "momentum": 0.0},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
    "sinks": {"log": {"kind": "loader-sink", "every": "${trainer.epochs}"}},
}

FAST_PROFILE: Final[dict[str, JsonValue]] = {
    "trainer": {"epochs": 2, "batch_size": 16, "warmup": "${trainer.epochs}"},
    "optimizer": {"lr": 0.2},
    "sinks": {"log": {"label": "fast"}},
}

TRANSFORMER_FRAGMENT: Final[dict[str, JsonValue]] = {
    "kind": "loader-transformer",
    "num_heads": 8,
}

RNN_FRAGMENT: Final[dict[str, JsonValue]] = {"kind": "loader-rnn", "state_size": 64}


def write(path: Path, document: JsonValue) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    write(tmp_path / "run.json", RUN_DOCUMENT)
    write(tmp_path / "profiles" / "fast.json", FAST_PROFILE)
    write(
        tmp_path / "model" / "encoders" / "terms" / "transformer.json",
        TRANSFORMER_FRAGMENT,
    )
    write(tmp_path / "model" / "encoders" / "terms" / "rnn.json", RNN_FRAGMENT)

    return tmp_path


@pytest.fixture
def loader() -> SpecLoader:
    return SpecLoader()


def test_the_run_spec_codec_fixes_the_boundary_schema(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    validate = RunSpec.model_validate.__func__
    settings_from_overrides = loader_module.settings_from_overrides

    def discover() -> tuple[str, ...]:
        events.append("discover")

        return ()

    def typed(
        schema: type[dx.Model], overrides: tuple[str, ...]
    ) -> tuple[str | loader_module.TypedSetting, ...]:
        events.append("typed-overrides")

        return settings_from_overrides(schema, overrides)

    def counted(cls: type[RunSpec], payload: Mapping[str, JsonValue]) -> RunSpec:
        events.append("validate")

        return validate(cls, payload)

    monkeypatch.setattr(loader_module, "load_plugins", discover)
    monkeypatch.setattr(loader_module, "settings_from_overrides", typed)
    monkeypatch.setattr(RunSpec, "model_validate", classmethod(counted))
    codec = RunSpecCodec(search_path=(config_dir,))
    spec = codec(
        config_dir / "run.json",
        profile="fast",
        extra={"trainer.batch_size": 32},
        overrides=("trainer.epochs=3", "optimizer.lr=0.4"),
    )

    assert codec.search_path == (config_dir,)
    assert events == [
        "discover",
        "typed-overrides",
        "typed-overrides",
        "validate",
    ]
    assert spec.model.encoders["terms"] == LoaderTransformer(num_heads=8)
    assert spec.trainer == LoaderTrainer(epochs=3, batch_size=32, warmup=3)
    assert spec.optimizer == LoaderSgd(lr=0.4, momentum=0.0)
    assert spec.sinks == {"log": LoaderSink(every=3, label="fast")}


def test_every_composition_loads_the_installed_declarations_first(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scans = 0

    def discover() -> tuple[str, ...]:
        nonlocal scans
        scans += 1

        return ()

    monkeypatch.setattr(loader_module, "load_plugins", discover)
    codec = RunSpecCodec(search_path=(config_dir,))

    codec(config_dir / "run.json")
    SpecLoader(search_path=(config_dir,)).compose_traced(
        config_dir / "run.json", schema=RunSpec
    )

    assert scans == 2


def test_failed_declaration_discovery_is_retried_before_validation(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scans = 0
    validations = 0
    validate = RunSpec.model_validate.__func__

    def discover() -> tuple[str, ...]:
        nonlocal scans
        scans += 1

        if scans == 1:
            msg = "plugin import failed"
            raise RuntimeError(msg)

        return ()

    def counted(cls: type[RunSpec], payload: Mapping[str, JsonValue]) -> RunSpec:
        nonlocal validations
        validations += 1

        return validate(cls, payload)

    monkeypatch.setattr(loader_module, "load_plugins", discover)
    monkeypatch.setattr(RunSpec, "model_validate", classmethod(counted))
    codec = RunSpecCodec(search_path=(config_dir,))

    with pytest.raises(RuntimeError, match="plugin import failed"):
        codec(config_dir / "run.json")

    assert validations == 0

    codec(config_dir / "run.json")

    assert scans == 2
    assert validations == 1


def test_a_fresh_component_plugin_is_available_to_the_first_validation(
    tmp_path: Path,
) -> None:
    suffix = tmp_path.name.replace("-", "_")
    module = f"lofi_declaration_plugin_{suffix}"
    distribution = f"lofi-declaration-plugin-{suffix}"
    tag = f"fresh-sink-{suffix}"
    (tmp_path / f"{module}.py").write_text(
        "\n".join(
            (
                "from typing import Literal, NamedTuple",
                "from lofi.registry import ComponentPlugin",
                "from lofi.spec.components.run import SinkSpec",
                "",
                "LOADS = globals().get('LOADS', 0) + 1",
                "",
                "class FreshSinkSettings(NamedTuple):",
                "    label: str",
                "",
                "class FreshSinkSpec(SinkSpec):",
                f'    kind: Literal["{tag}"] = "{tag}"',
                "    label: str",
                "",
                "    def settings(self) -> FreshSinkSettings:",
                "        return FreshSinkSettings(self.label)",
                "",
                "def _build(settings: FreshSinkSettings, context: object, /) -> str:",
                "    del context",
                "    return settings.label",
                "",
                "PLUGIN = ComponentPlugin(spec=FreshSinkSpec, build=_build)",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    metadata = tmp_path / f"{distribution.replace('-', '_')}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.sink]\n{tag} = {module}:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )
    document: dict[str, JsonValue] = {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "fresh-run-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "fresh-run-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "fresh-run-dataset"},
                    "featurizer": {"kind": "unlabeled"},
                }
            }
        },
        "trainer": {"kind": "fresh-run-trainer"},
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "validation-best"},
        "optimizer": {"kind": "fresh-run-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "sinks": {"log": {"kind": tag, "label": "loaded"}},
    }
    path = write(tmp_path / "run.json", document)
    probe = """
import importlib
import json
import sys
from typing import Literal

from lofi.spec import (
    DatasetSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpecCodec,
    TrainerSpec,
)


class FreshParser(ParserSpec):
    kind: Literal["fresh-run-parser"] = "fresh-run-parser"


class FreshLoss(LossSpec):
    kind: Literal["fresh-run-loss"] = "fresh-run-loss"


class FreshDataset(DatasetSpec):
    kind: Literal["fresh-run-dataset"] = "fresh-run-dataset"


class FreshTrainer(TrainerSpec):
    kind: Literal["fresh-run-trainer"] = "fresh-run-trainer"


class FreshOptimizer(OptimizerSpec):
    kind: Literal["fresh-run-optimizer"] = "fresh-run-optimizer"


module, path = sys.argv[1:]
before = module in sys.modules
codec = RunSpecCodec()
first = codec(path)
second = codec(path)
plugin = importlib.import_module(module)
blocked = (
    "wandb",
    "trackio",
    "mlflow",
)
print(json.dumps({
    "before": before,
    "after": module in sys.modules,
    "loads": plugin.LOADS,
    "label": first.sinks["log"].label,
    "same": first.sinks["log"] == second.sinks["log"],
    "blocked": [name for name in blocked if name in sys.modules],
}))
"""
    python_path = os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-c", probe, module, str(path)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": python_path},
        timeout=TIMEOUT_SECONDS,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "before": False,
        "after": True,
        "loads": 1,
        "label": "loaded",
        "same": True,
        "blocked": [],
    }


def test_a_loader_reports_the_search_path_it_was_given(tmp_path: Path) -> None:
    assert SpecLoader().search_path == ()
    assert SpecLoader(search_path=(tmp_path,)).search_path == (tmp_path,)


def test_every_engine_rung_is_reported_as_one_source() -> None:
    assert set(SOURCE_OF_KIND) == set(get_args(OriginKind.__value__))
    assert set(SOURCE_OF_KIND.values()) == set(ConfigSource)


def test_the_layers_compose_in_precedence_order(
    config_dir: Path, loader: SpecLoader
) -> None:
    spec = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        profile="fast",
        extra={"trainer.batch_size": 32, "optimizer.lr": 0.3},
        overrides=("optimizer.lr=0.4", "trainer.epochs=3"),
    ).spec

    assert spec.model.parser == LoaderParser(beam=4)
    assert spec.model.encoders["terms"] == LoaderTransformer(num_heads=8)
    assert spec.trainer == LoaderTrainer(epochs=3, batch_size=32, warmup=3)
    assert spec.optimizer == LoaderSgd(lr=0.4, momentum=0.0)
    assert spec.sinks == {"log": LoaderSink(every=3, label="fast")}


def test_provenance_names_the_layer_at_every_depth(
    config_dir: Path, loader: SpecLoader
) -> None:
    provenance = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        profile="fast",
        extra={"trainer.batch_size": 32},
        overrides=("optimizer.lr=0.4",),
    ).provenance
    fragment = str(config_dir / "model" / "encoders" / "terms" / "transformer.json")

    assert provenance.origin_of("trainer.epochs") == LeafOrigin(
        source=ConfigSource.PROFILE, layer="fast"
    )
    assert provenance.origin_of("trainer.batch_size") == LeafOrigin(
        source=ConfigSource.EXTRA
    )
    assert provenance.origin_of("optimizer.lr") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer="optimizer.lr=0.4"
    )
    assert provenance.origin_of("optimizer.momentum") == LeafOrigin(
        source=ConfigSource.FILE, layer=str(config_dir / "run.json")
    )
    assert provenance.origin_of("model.parser.beam") == LeafOrigin(
        source=ConfigSource.FILE, layer=str(config_dir / "run.json")
    )
    assert provenance.origin_of("model.encoders.terms.num_heads") == LeafOrigin(
        source=ConfigSource.GROUP, layer=fragment
    )
    assert provenance.origin_of("model.encoders.terms.kind") == LeafOrigin(
        source=ConfigSource.GROUP, layer=fragment
    )
    assert provenance.origin_of("scheduler") == LeafOrigin(source=ConfigSource.DEFAULT)


def test_provenance_names_the_layer_of_a_leaf_inside_a_dict_slot(
    config_dir: Path, loader: SpecLoader
) -> None:
    provenance = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        profile="fast",
        extra={"sinks.log.every": 5},
        overrides=("sinks.log.label=hello",),
    ).provenance

    assert provenance.origin_of("sinks.log.kind") == LeafOrigin(
        source=ConfigSource.FILE, layer=str(config_dir / "run.json")
    )
    assert provenance.origin_of("sinks.log.every") == LeafOrigin(
        source=ConfigSource.EXTRA
    )
    assert provenance.origin_of("sinks.log.label") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer="sinks.log.label=hello"
    )

    provenance = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, profile="fast"
    ).provenance

    assert provenance.origin_of("sinks.log.every") == LeafOrigin(
        source=ConfigSource.FILE,
        layer=str(config_dir / "run.json"),
        expression="${trainer.epochs}",
    )
    assert provenance.origin_of("sinks.log.label") == LeafOrigin(
        source=ConfigSource.PROFILE, layer="fast"
    )


def test_a_leaf_the_schema_computes_is_attributed_to_the_schema(
    config_dir: Path, loader: SpecLoader
) -> None:
    provenance = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=("model.parser.beam=2",)
    ).provenance

    assert provenance.origin_of("model.parser.declared_output") == LeafOrigin(
        source=ConfigSource.DEFAULT
    )

    with pytest.raises(UnknownSettingError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("model.parser.declared_output=1",),
        )

    assert refused.value.path == "model.parser.declared_output"


def test_a_leaf_an_expression_produced_reports_the_expression(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, profile="fast"
    )

    assert composition.spec.trainer.warmup == 2
    assert composition.provenance.origin_of("trainer.warmup") == LeafOrigin(
        source=ConfigSource.PROFILE, layer="fast", expression="${trainer.epochs}"
    )


def test_an_override_holding_an_expression_reports_it(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=("trainer.warmup=${trainer.batch_size}",),
    )

    assert composition.spec.trainer.warmup == 8
    assert composition.provenance.origin_of("trainer.warmup") == LeafOrigin(
        source=ConfigSource.OVERRIDE,
        layer="trainer.warmup=${trainer.batch_size}",
        expression="${trainer.batch_size}",
    )


def test_an_override_layer_is_named_by_the_expression_as_typed(
    config_dir: Path, loader: SpecLoader
) -> None:
    provenance = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=(
            "optimizer.lr=4e-1",
            "optimizer.lr=0.5",
            "trainer.epochs=02",
            "sinks.log.label=a=b",
        ),
    ).provenance

    assert provenance.origin_of("optimizer.lr") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer="optimizer.lr=0.5"
    )
    assert provenance.origin_of("trainer.epochs") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer="trainer.epochs=02"
    )
    assert provenance.origin_of("sinks.log.label") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer="sinks.log.label=a=b"
    )


def test_an_override_appends_a_checked_tagged_union_element_with_its_origin(
    config_dir: Path, loader: SpecLoader
) -> None:
    expression = (
        'loss.terms+=[{"objective":{"kind":"loader-auxiliary-loss"},"weight":2}]'
    )

    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=(expression,)
    )

    assert composition.spec.loss is not None
    assert tuple(term.objective for term in composition.spec.loss.terms) == (
        LoaderLoss(),
        LoaderAuxiliaryLoss(),
    )
    assert tuple(term.weight for term in composition.spec.loss.terms) == (1.0, 2.0)
    assert composition.provenance.origin_of("loss.terms[0]") == LeafOrigin(
        source=ConfigSource.FILE, layer=str(config_dir / "run.json")
    )
    assert composition.provenance.origin_of("loss.terms[1]") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer=expression
    )


def test_an_override_prepends_a_checked_tagged_union_element(
    config_dir: Path, loader: SpecLoader
) -> None:
    expression = (
        'loss.+terms=[{"objective":{"kind":"loader-auxiliary-loss"},"weight":2}]'
    )

    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=(expression,)
    )

    assert composition.spec.loss is not None
    assert tuple(term.objective for term in composition.spec.loss.terms) == (
        LoaderAuxiliaryLoss(),
        LoaderLoss(),
    )
    assert composition.provenance.origin_of("loss.terms[0]") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer=expression
    )
    assert composition.provenance.origin_of("loss.terms[1]") == LeafOrigin(
        source=ConfigSource.FILE, layer=str(config_dir / "run.json")
    )


def test_an_unmarked_list_override_replaces_the_list_whole(
    config_dir: Path, loader: SpecLoader
) -> None:
    expression = 'loss.terms=[{"objective":{"kind":"loader-auxiliary-loss"}}]'

    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=(expression,)
    )

    assert composition.spec.loss is not None
    assert tuple(term.objective for term in composition.spec.loss.terms) == (
        LoaderAuxiliaryLoss(),
    )
    assert composition.provenance.origin_of("loss.terms[0]") == LeafOrigin(
        source=ConfigSource.OVERRIDE, layer=expression
    )


def test_list_extension_refuses_a_non_list_target(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("trainer.epochs+=[2]",),
        )

    assert refused.value.path == "trainer.epochs"
    assert "is not a list and cannot be appended" in refused.value.reason


def test_list_extension_refuses_an_unknown_tagged_union_element(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=('loss.terms+=[{"objective":{"kind":"loader-nothing"}}]',),
        )

    assert refused.value.path == "loss.terms[0].objective.kind"
    assert "LossSpec registers" in refused.value.reason
    assert "loader-auxiliary-loss" in refused.value.reason
    assert "loader-loss" in refused.value.reason


def test_provenance_covers_exactly_the_leaves_of_the_composed_spec(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, profile="fast"
    )

    held = leaves(json.loads(composition.spec.model_dump_json()))

    assert frozenset(composition.provenance.paths) == held


def test_composing_and_tracing_answer_the_same_request_alike(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, profile="fast"
    )

    assert (
        composition.spec
        == loader.compose_traced(
            config_dir / "run.json", schema=RunSpec, profile="fast"
        ).spec
    )
    assert (
        composition.provenance
        == loader.compose_traced(
            config_dir / "run.json", schema=RunSpec, profile="fast"
        ).provenance
    )


def test_a_group_selected_by_override_replaces_the_defaults_entry(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=("model/encoders/terms=rnn",),
    )

    assert composition.spec.model.encoders["terms"] == LoaderRnn(state_size=64)
    assert composition.provenance.origin_of(
        "model.encoders.terms.state_size"
    ) == LeafOrigin(
        source=ConfigSource.GROUP,
        layer=str(config_dir / "model" / "encoders" / "terms" / "rnn.json"),
    )


@pytest.mark.parametrize(
    ("selection", "text", "expected"),
    [
        ((), "3", LoaderTransformer(num_heads=8, reach=3)),
        (("model/encoders/terms=rnn",), "far", LoaderRnn(state_size=64, reach="far")),
    ],
)
def test_an_override_the_variants_declare_differently_reads_as_the_selected_one(
    config_dir: Path,
    loader: SpecLoader,
    selection: tuple[str, ...],
    text: str,
    expected: TermEncoderSpec,
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=(*selection, f"model.encoders.terms.reach={text}"),
    )

    assert composition.spec.model.encoders["terms"] == expected
    assert composition.provenance.origin_of("model.encoders.terms.reach") == (
        LeafOrigin(
            source=ConfigSource.OVERRIDE, layer=f"model.encoders.terms.reach={text}"
        )
    )


def test_an_override_the_selected_variant_cannot_read_is_refused_at_its_leaf(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("model.encoders.terms.reach=far",),
        )

    assert refused.value.path == "model.encoders.terms.reach"


def test_an_override_the_variants_declare_differently_is_refused_with_no_variant(
    config_dir: Path, loader: SpecLoader
) -> None:
    document = {key: value for key, value in RUN_DOCUMENT.items() if key != "defaults"}
    path = write(config_dir / "unencoded.json", document)

    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            path, schema=RunSpec, overrides=("model.encoders.terms.reach=3",)
        )

    assert refused.value.path == "model.encoders.terms.reach"
    assert "model.encoders.terms.kind" in refused.value.reason


def test_a_group_at_the_root_is_selected_by_a_leading_slash(
    config_dir: Path, loader: SpecLoader
) -> None:
    document = {key: value for key, value in RUN_DOCUMENT.items() if key != "optimizer"}
    path = write(config_dir / "unoptimized.json", document)
    write(
        config_dir / "optimizer" / "steady.json",
        {"kind": "loader-sgd", "lr": 0.05, "momentum": 0.9},
    )

    composition = loader.compose_traced(
        path, schema=RunSpec, overrides=("/optimizer=steady",)
    )

    assert composition.spec.optimizer == LoaderSgd(lr=0.05, momentum=0.9)
    assert composition.provenance.origin_of("optimizer.momentum") == LeafOrigin(
        source=ConfigSource.GROUP, layer=str(config_dir / "optimizer" / "steady.json")
    )

    with pytest.raises(UnresolvedFragmentError):
        loader.compose_traced(path, schema=RunSpec, overrides=("/optimizer=absent",))


def test_the_sources_of_a_path_name_every_layer_that_set_it_lowest_first(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        profile="fast",
        extra={"optimizer.lr": 0.3},
        overrides=(
            "optimizer.lr=4e-1",
            "model/encoders/terms=rnn",
            "optimizer.lr=0.5",
        ),
    )
    document = str(config_dir / "run.json")

    assert composition.sources["optimizer.lr"] == (
        LeafOrigin(source=ConfigSource.FILE, layer=document),
        LeafOrigin(source=ConfigSource.PROFILE, layer="fast"),
        LeafOrigin(source=ConfigSource.EXTRA),
        LeafOrigin(source=ConfigSource.OVERRIDE, layer="optimizer.lr=4e-1"),
        LeafOrigin(source=ConfigSource.OVERRIDE, layer="optimizer.lr=0.5"),
    )
    assert composition.sources["model.encoders.terms.state_size"] == (
        LeafOrigin(
            source=ConfigSource.GROUP,
            layer=str(config_dir / "model" / "encoders" / "terms" / "rnn.json"),
        ),
    )
    assert "model.encoders.terms.num_heads" not in composition.sources
    assert composition.sources["data.datasets.train.dataset.kind"] == (
        LeafOrigin(source=ConfigSource.FILE, layer=document),
    )


def test_an_empty_mapping_a_layer_sets_is_a_path_of_its_sources(
    config_dir: Path, loader: SpecLoader
) -> None:
    composition = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=("sinks={}",)
    )

    assert composition.sources["sinks"] == (
        LeafOrigin(source=ConfigSource.OVERRIDE, layer="sinks={}"),
    )


@pytest.mark.parametrize("name", ["null", "Null", "~", ""])
def test_a_group_selection_of_nothing_leaves_the_slot_unfilled(
    config_dir: Path, loader: SpecLoader, name: str
) -> None:
    spec = loader.compose_traced(
        config_dir / "run.json", schema=RunSpec, overrides=(f"/scheduler={name}",)
    ).spec

    assert spec.scheduler is None


def test_a_fragment_is_found_under_the_search_path(
    tmp_path: Path, config_dir: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    write(
        elsewhere / "model" / "encoders" / "terms" / "cnn.json",
        {"kind": "loader-rnn", "state_size": 128},
    )
    searching = SpecLoader(search_path=(elsewhere,))

    spec = searching.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=("model/encoders/terms=cnn",),
    ).spec

    assert spec.model.encoders["terms"] == LoaderRnn(state_size=128)


def test_an_unknown_key_in_the_document_is_refused_naming_its_path(
    config_dir: Path, loader: SpecLoader
) -> None:
    document = dict(RUN_DOCUMENT)
    document["trainer"] = {"kind": "loader-trainer", "epoch": 3}
    write(config_dir / "run.json", document)

    with pytest.raises(UnknownSettingError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.path == "trainer.epoch"
    assert "trainer.epochs" in refused.value.declared
    assert all(path.startswith("trainer.") for path in refused.value.declared)


def test_a_setting_of_an_unselected_variant_names_the_variant_it_belongs_to(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(UnknownSettingError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=(
                "model/encoders/terms=rnn",
                "model.encoders.terms.num_heads=2",
            ),
        )

    declaring = {
        str(tag)
        for tag, variant in TermEncoderSpec.__variants__.items()
        if "num_heads" in variant.__field_specs__
    }

    assert refused.value.path == "model.encoders.terms.num_heads"
    assert "loader-transformer" in refused.value.owners
    assert set(refused.value.owners) == declaring
    assert f"belongs to {', '.join(refused.value.owners)}," in str(refused.value)


@pytest.mark.parametrize("fragment", ["composition", "type-raising"])
def test_a_packaged_rule_fragment_extends_the_application_rules(
    tmp_path: Path, fragment: str
) -> None:
    codec = RunSpecCodec(search_path=(tmp_path, PACKAGED_CONFIG_ROOT))
    rules = []

    for name in ("application", fragment):
        document = tmp_path / f"{name}.yaml"
        document.write_text(
            "defaults:\n"
            "  - run/synthetic-parser\n"
            f"  - data.datasets.train.dataset.grammar.rules: {name}\n",
            encoding="utf-8",
            newline="\n",
        )
        dataset = cast(
            "SyntheticCorpusDecl", codec(document).data.datasets["train"].dataset
        )
        rules.append(declare_grammar(dataset.grammar).rules.rules)

    base, extended = rules

    assert len(extended) > len(base)
    assert extended[: len(base)] == base


def test_a_packaged_run_names_the_transformer_as_the_owner_of_heads() -> None:
    codec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))

    with pytest.raises(UnknownSettingError) as refused:
        codec(
            PACKAGED_CONFIG_ROOT / "run" / "synthetic-parser.yaml",
            overrides=("model.encoders.terms.heads=2",),
        )

    declaring = {
        str(tag)
        for tag, variant in TermEncoderSpec.__variants__.items()
        if "heads" in variant.__field_specs__
    }

    assert refused.value.path == "model.encoders.terms.heads"
    assert "transformer" in refused.value.owners
    assert set(refused.value.owners) == declaring
    assert "model.encoders.terms.width" in refused.value.declared


def test_a_setting_no_variant_declares_names_no_owner(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(UnknownSettingError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("model.encoders.terms.depth=2",),
        )

    assert refused.value.owners == ()


def test_an_unknown_key_at_the_root_lists_root_settings(
    config_dir: Path, loader: SpecLoader
) -> None:
    write(config_dir / "run.json", {**RUN_DOCUMENT, "seed": 1})

    with pytest.raises(UnknownSettingError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.path == "seed"
    assert "trainer" in refused.value.declared
    assert not any("." in path for path in refused.value.declared)


def test_an_override_that_does_not_read_is_refused_before_composition(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(SettingParseError) as refused:
        loader.compose_traced(
            config_dir / "run.json", schema=RunSpec, overrides=("trainer.epochs=x",)
        )

    assert refused.value.path == "trainer.epochs"
    assert refused.value.expected == "int"
    assert refused.value.text == "x"


def test_a_document_value_of_the_wrong_type_is_refused_naming_its_path(
    config_dir: Path, loader: SpecLoader
) -> None:
    document = dict(RUN_DOCUMENT)
    document["trainer"] = {"kind": "loader-trainer", "epochs": "seven"}
    write(config_dir / "run.json", document)

    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.path == "trainer.epochs"
    assert refused.value.reason == "expects int; got 'seven'"
    assert str(refused.value) == "trainer.epochs was refused: expects int; got 'seven'"


PACKAGED_GRAMMARS: Final = PACKAGED_CONFIG_ROOT / "data/datasets/train/dataset/grammar"


@pytest.mark.parametrize(
    ("name", "declaration"),
    [
        ("ab", ABFormalismSpec),
        ("cfg", CFGFormalismSpec),
        ("undirected", UndirectedCCGFormalismSpec),
        ("nl-diamond", TLGFormalismSpec),
        ("directed", GrammarDecl),
        ("ccgbank", GrammarDecl),
    ],
)
def test_a_grammar_document_composes_to_the_variant_its_formalism_selects(
    loader: SpecLoader, name: str, declaration: type[GrammarSpec]
) -> None:
    grammar = loader.compose_traced(
        PACKAGED_GRAMMARS / f"{name}.yaml", schema=GrammarSpec
    ).spec

    assert type(grammar) is declaration


def test_a_composed_grammar_carries_the_fragments_its_defaults_select(
    loader: SpecLoader,
) -> None:
    grammar = loader.compose_traced(
        PACKAGED_GRAMMARS / "ccgbank.yaml", schema=GrammarSpec
    ).spec
    signature = declare_grammar(grammar).signature

    assert signature.name == "ccgbank"
    assert [functor.name for functor in signature.functors] == ["/", "\\"]


@pytest.mark.parametrize("role", ["train", "dev"])
def test_a_refusal_inside_a_dataset_role_names_the_role(
    config_dir: Path, loader: SpecLoader, role: str
) -> None:
    document = dict(RUN_DOCUMENT)
    datasets: dict[str, JsonValue] = {
        name: {"dataset": {"kind": "loader-sized-dataset", "size": 4}}
        for name in ("train", "dev")
    }
    datasets[role] = {"dataset": {"kind": "loader-sized-dataset", "size": 0}}
    document["data"] = {"datasets": datasets}
    write(config_dir / "run.json", document)

    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.path == f"data.datasets.{role}.dataset.size"
    assert refused.value.reason == "a dataset holds at least one example"


def test_a_malformed_override_is_refused_naming_the_expression(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(MalformedOverrideError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("trainer.epochs=2", "trainer.epochs"),
        )

    assert refused.value.expression == "trainer.epochs"
    assert "trainer.epochs" in str(refused.value)


def test_a_missing_fragment_is_refused_with_the_files_tried(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(UnresolvedFragmentError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("model/encoders/terms=cnn",),
        )

    assert refused.value.path == "model.encoders.terms"
    assert refused.value.slot == "model.encoders.terms"
    assert refused.value.name == "cnn"
    assert refused.value.available == ("rnn", "transformer")
    assert refused.value.tried
    assert all(
        Path(tried).parent == config_dir / "model" / "encoders" / "terms"
        for tried in refused.value.tried
    )


def test_a_missing_profile_is_refused_as_an_unresolved_fragment(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(UnresolvedFragmentError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec, profile="slow")

    assert refused.value.slot == "profiles"
    assert refused.value.name == "slow"
    assert refused.value.available == ("fast",)


def test_a_missing_root_fragment_is_refused_at_the_defaults_list(
    config_dir: Path, loader: SpecLoader
) -> None:
    write(config_dir / "run.json", {**RUN_DOCUMENT, "defaults": ["base"]})

    with pytest.raises(UnresolvedFragmentError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.slot == "defaults"
    assert refused.value.name == "base"
    assert refused.value.available == ()


def test_a_missing_document_is_refused_as_a_whole(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigDocumentError) as refused:
        loader.compose_traced(config_dir / "nothing.json", schema=RunSpec)

    assert refused.value.document == config_dir / "nothing.json"
    assert str(config_dir / "nothing.json") in refused.value.reason


@pytest.mark.parametrize(
    "malformed",
    [
        Path("run.json"),
        Path("profiles") / "fast.json",
        Path("model") / "encoders" / "terms" / "transformer.json",
    ],
    ids=["document", "profile", "fragment"],
)
def test_a_file_that_does_not_parse_is_refused_as_a_whole(
    config_dir: Path, loader: SpecLoader, malformed: Path
) -> None:
    (config_dir / malformed).write_text("{not json", encoding="utf-8", newline="\n")

    with pytest.raises(ConfigDocumentError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec, profile="fast")

    assert refused.value.document == config_dir / "run.json"
    assert "does not parse" in refused.value.reason
    assert "line 1" in refused.value.reason


def test_an_unknown_variant_is_refused_at_the_discriminator(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("model.parser.kind=loader-nothing",),
        )

    assert refused.value.path == "model.parser.kind"
    assert "loader-nothing" in refused.value.reason


def test_a_required_setting_no_source_set_is_refused_as_missing(
    config_dir: Path, loader: SpecLoader
) -> None:
    document = dict(RUN_DOCUMENT)
    document["optimizer"] = {"kind": "loader-sgd"}
    write(config_dir / "run.json", document)

    with pytest.raises(MissingSettingError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.path == "optimizer.lr"


def test_a_value_a_validator_refuses_is_reported_at_its_leaf(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json", schema=RunSpec, overrides=("trainer.warmup=-1",)
        )

    assert refused.value.path == "trainer.warmup"
    assert "negative" in refused.value.reason


def test_every_value_the_validators_refuse_is_named_in_the_reason(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("trainer.warmup=-1", "optimizer.momentum=2"),
        )

    assert refused.value.path == "trainer.warmup"
    assert refused.value.reason == (
        "warmup counts steps and cannot be negative; "
        "also optimizer.momentum: momentum is a fraction of the previous step"
    )


def test_an_unresolvable_expression_is_reported_at_its_leaf(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=("trainer.warmup=${trainer.nothing}",),
        )

    assert refused.value.path == "trainer.warmup"
    assert "trainer.nothing" in refused.value.reason


def test_a_two_node_reference_cycle_names_both_of_its_nodes(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=(
                "trainer.warmup=${trainer.batch_size}",
                "trainer.batch_size=${trainer.warmup}",
            ),
        )

    assert "cycle" in refused.value.reason
    assert "trainer.warmup" in refused.value.reason
    assert "trainer.batch_size" in refused.value.reason


def test_a_longer_reference_cycle_names_every_node_it_passes_through(
    config_dir: Path, loader: SpecLoader
) -> None:
    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=(
                "trainer.warmup=${trainer.batch_size}",
                "trainer.batch_size=${trainer.epochs}",
                "trainer.epochs=${sinks.log.every}",
                "sinks.log.every=${trainer.warmup}",
            ),
        )

    substrate = refused.value.__cause__
    cycle = (
        ("sinks", "log", "every"),
        ("trainer", "warmup"),
        ("trainer", "batch_size"),
        ("trainer", "epochs"),
        ("sinks", "log", "every"),
    )
    rendered = (
        "Interpolation cycle detected: sinks.log.every -> trainer.warmup -> "
        "trainer.batch_size -> trainer.epochs -> sinks.log.every; at config key "
        "'trainer.epochs'"
    )

    assert isinstance(substrate, InterpolationError)
    assert substrate.cycle_path == cycle
    assert refused.value.path == "trainer.epochs"
    assert refused.value.reason == rendered
    assert str(refused.value) == f"trainer.epochs was refused: {rendered}"


def test_an_expression_nested_past_the_engine_cap_reports_the_cap_by_number(
    config_dir: Path, loader: SpecLoader
) -> None:
    nested = "${oc.create:" * (NESTING_CAP + 1)
    closed = "}" * (NESTING_CAP + 1)

    with pytest.raises(ConfigValueError) as refused:
        loader.compose_traced(
            config_dir / "run.json",
            schema=RunSpec,
            overrides=(f"trainer.warmup={nested}3{closed}",),
        )

    assert refused.value.path == "trainer.warmup"
    assert str(NESTING_CAP) in refused.value.reason


def test_an_expression_nested_to_the_engine_cap_still_resolves(
    config_dir: Path, loader: SpecLoader
) -> None:
    nested = "${oc.create:" * NESTING_CAP
    closed = "}" * NESTING_CAP
    run = loader.compose_traced(
        config_dir / "run.json",
        schema=RunSpec,
        overrides=(f"trainer.warmup={nested}3{closed}",),
    ).spec

    assert json.loads(run.model_dump_json())["trainer"]["warmup"] == 3


def test_a_document_that_is_not_a_mapping_is_refused_as_a_whole(
    config_dir: Path, loader: SpecLoader
) -> None:
    write(config_dir / "run.json", [1, 2])

    with pytest.raises(ConfigDocumentError) as refused:
        loader.compose_traced(config_dir / "run.json", schema=RunSpec)

    assert refused.value.document == config_dir / "run.json"
    assert "mapping" in refused.value.reason


def test_every_refusal_at_a_leaf_is_a_config_error(
    config_dir: Path, loader: SpecLoader
) -> None:
    requests: tuple[tuple[str, ...], ...] = (
        ("trainer.epochs=x",),
        ("trainer.epoch=1",),
        ("model/encoders/terms=cnn",),
        ("model.parser.kind=loader-nothing",),
        ("trainer.warmup=-1",),
        ("trainer.warmup=${trainer.nothing}",),
    )

    for overrides in requests:
        with pytest.raises(ConfigError) as refused:
            loader.compose_traced(
                config_dir / "run.json", schema=RunSpec, overrides=overrides
            )

        assert refused.value.path in str(refused.value)


REFUSAL_SCRIPT: Final = """\
import json
import os
import sys
from pathlib import Path
from typing import Literal

from lofi.spec import (
    ConfigError,
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    SpecLoader,
    TrainerSpec,
)


class ProbeParser(ParserSpec):
    kind: Literal["probe-parser"] = "probe-parser"


class ProbeLoss(LossSpec):
    kind: Literal["probe-loss"] = "probe-loss"


class ProbeDataset(DatasetSpec):
    kind: Literal["probe-dataset"] = "probe-dataset"


class ProbeFeaturizer(FeaturizerSpec):
    kind: Literal["probe-featurizer"] = "probe-featurizer"


class ProbeTrainer(TrainerSpec):
    kind: Literal["probe-trainer"] = "probe-trainer"
    epochs: int = 1


class ProbeSgd(OptimizerSpec):
    kind: Literal["probe-sgd"] = "probe-sgd"
    lr: float


directory = Path(os.environ["LOFI_PROBE_DIRECTORY"])
loader = SpecLoader()
print(sys.flags.optimize)

probes = (
    ("mistyped.json", ()),
    ("typed.json", ("optimizer.lr=fast",)),
    ("typed.json", ("trainer.epochs=${nothing}",)),
)

for name, overrides in probes:
    try:
        loader.compose_traced(directory / name, schema=RunSpec, overrides=overrides)
    except ConfigError as refused:
        print(type(refused).__name__, refused.path)
    else:
        print("composed")
"""

TYPED_DOCUMENT: Final[dict[str, JsonValue]] = {
    "model": {
        "language": {"from": "data"},
        "parser": {"kind": "probe-parser"},
    },
    "loss": {"terms": [{"objective": {"kind": "probe-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "probe-dataset"},
                "featurizer": {"kind": "probe-featurizer"},
            }
        }
    },
    "trainer": {"kind": "probe-trainer", "epochs": 7},
    "optimizer": {"kind": "probe-sgd", "lr": 0.1},
}

MISTYPED_DOCUMENT: Final[dict[str, JsonValue]] = {
    **TYPED_DOCUMENT,
    "trainer": {"kind": "probe-trainer", "epochs": "seven"},
}


def run_refusals(directory: Path, *, optimize: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-"],
        input=REFUSAL_SCRIPT,
        env={
            **os.environ,
            "LOFI_PROBE_DIRECTORY": str(directory),
            "PYTHONOPTIMIZE": optimize,
        },
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=TREE,
        timeout=TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr

    return result.stdout.splitlines()


def test_a_refusal_names_the_same_path_with_and_without_optimization(
    tmp_path: Path,
) -> None:
    write(tmp_path / "typed.json", TYPED_DOCUMENT)
    write(tmp_path / "mistyped.json", MISTYPED_DOCUMENT)

    plain = run_refusals(tmp_path, optimize="")
    optimized = run_refusals(tmp_path, optimize="1")

    assert plain[0] == "0"
    assert optimized[0] == "1"
    assert plain[1:] == optimized[1:]
    assert plain[1:] == [
        "ConfigValueError trainer.epochs",
        "SettingParseError optimizer.lr",
        "ConfigValueError trainer.epochs",
    ]


LEAF: Final = "trainer.epochs"

UNKNOWN: Final = "trainer.epoch"

PROFILE: Final = "fast"


def _epochs(spec: RunSpec) -> JsonValue:
    return json.loads(spec.model_dump_json())["trainer"]["epochs"]


def _ladder(document: Path) -> tuple[Layer, ...]:
    return (
        Layer(ConfigSource.FILE, Request(path=document), 1),
        Layer(ConfigSource.PROFILE, Request(path=document, profile=PROFILE), 2),
        Layer(
            ConfigSource.EXTRA,
            Request(path=document, profile=PROFILE, extra={LEAF: 30}),
            30,
        ),
        Layer(
            ConfigSource.OVERRIDE,
            Request(
                path=document,
                profile=PROFILE,
                overrides=(f"{LEAF}=40",),
                extra={LEAF: 30},
            ),
            40,
        ),
    )


class TestSpecLoader(ConfigLoaderConformance):
    @pytest.fixture
    def loader_fixture(self, config_dir: Path) -> ConfigLoaderFixture[RunSpec]:
        document = config_dir / "run.json"
        written = count(1)

        def reseat(resolved: RunSpec) -> Request:
            reseated = config_dir / f"resolved-{next(written)}.json"
            reseated.write_text(
                resolved.model_dump_json(), encoding="utf-8", newline="\n"
            )

            return Request(path=reseated)

        return ConfigLoaderFixture(
            loader=SpecLoader(),
            schema=RunSpec,
            leaf=LEAF,
            read=_epochs,
            ladder=_ladder(document),
            resolved=Request(path=document, profile=PROFILE),
            reseat=reseat,
            unknown=Request(path=document, overrides=(f"{UNKNOWN}=3",)),
            unknown_leaf=UNKNOWN,
        )
