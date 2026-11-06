import json
from typing import TYPE_CHECKING, Literal, cast

import didactic.api as dx
import pytest
from tests.conformance.component_plugin import plugin_context

from lofi.formalism import BoundedConsumer
from lofi.nn.build import build_component
from lofi.nn.parsers.flat import FlatSpanEncoder
from lofi.run import Axis, GridSweepExpander, Sweep
from lofi.spec import (
    ConfigValueError,
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    RunSpecCodec,
    SubwordPoolerSpec,
    TrainerSpec,
)
from lofi.spec._ports import Dependency
from lofi.spec.parsers.flat import FlatParserSpec, FlatSpanFeature

FEATURES: tuple[FlatSpanFeature, ...] = ("left", "right", "bilinear")
PARSER: dict[str, JsonValue] = {
    "kind": "flat",
    "width": 17,
    "features": [*FEATURES],
    "projection_initialization": "identity",
}

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.nn.ports import SubwordPooler


class _RunLoss(LossSpec):
    kind: Literal["flat-spec-test-loss"] = "flat-spec-test-loss"


class _RunDataset(DatasetSpec):
    kind: Literal["flat-spec-test-dataset"] = "flat-spec-test-dataset"


class _RunFeaturizer(FeaturizerSpec):
    kind: Literal["flat-spec-test-featurizer"] = "flat-spec-test-featurizer"


class _RunTrainer(TrainerSpec):
    kind: Literal["flat-spec-test-trainer"] = "flat-spec-test-trainer"


class _RunOptimizer(OptimizerSpec):
    kind: Literal["flat-spec-test-optimizer"] = "flat-spec-test-optimizer"


class _RunPooler(SubwordPoolerSpec):
    kind: Literal["flat-spec-test-pooler"] = "flat-spec-test-pooler"


class _Pooled:
    def __init__(self, width: int) -> None:
        self._width = width

    @property
    def output_width(self) -> int:
        return self._width


def _pooled(width: int) -> SubwordPooler:
    return cast("SubwordPooler", _Pooled(width))


def _document(parser: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    model: dict[str, JsonValue] = {
        "language": {"from": "data"},
        "subword_pooler": {"kind": "flat-spec-test-pooler"},
    }

    if parser is not None:
        model["parser"] = dict(parser)

    return {
        "model": model,
        "loss": {"terms": [{"objective": {"kind": "flat-spec-test-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "flat-spec-test-dataset"},
                    "featurizer": {"kind": "flat-spec-test-featurizer"},
                }
            }
        },
        "trainer": {"kind": "flat-spec-test-trainer"},
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "flat-spec-test-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def _write(path: Path, document: JsonValue) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


def test_the_tag_reaches_the_flat_variant_through_the_family_root() -> None:
    declared = ParserSpec.model_validate(PARSER)

    assert type(declared) is FlatParserSpec
    assert declared == FlatParserSpec(
        width=17,
        features=FEATURES,
        projection_initialization="identity",
    )
    assert ParserSpec.__variants__["flat"] is FlatParserSpec


def test_a_flat_encoder_combines_no_cells_and_bounds_no_grammar() -> None:
    declared = FlatParserSpec(
        width=4,
        features=FEATURES,
        projection_initialization="identity",
    )

    assert not isinstance(declared, BoundedConsumer)
    assert declared.chart_outputs() == {"inside", "cells"}


def test_endpoint_features_and_projection_initialization_are_declared() -> None:
    declared = FlatParserSpec(
        width=19,
        features=("right", "bilinear"),
        projection_initialization="xavier-uniform",
    )

    assert declared.width == 19
    assert declared.features == ("right", "bilinear")
    assert declared.projection_initialization == "xavier-uniform"
    assert declared.dependencies() == (Dependency("subword_pooler", "leaves"),)


def test_a_flat_declaration_requires_the_cell_width() -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        ParserSpec.model_validate(
            {
                "kind": "flat",
                "features": [*FEATURES],
                "projection_initialization": "identity",
            }
        )


@pytest.mark.parametrize("missing", ["features", "projection_initialization"])
def test_a_flat_declaration_requires_its_architecture_choices(missing: str) -> None:
    document = dict(PARSER)
    document.pop(missing)

    with pytest.raises(dx.ValidationError, match=missing):
        ParserSpec.model_validate(document)


@pytest.mark.parametrize("width", [0, -1])
def test_a_nonpositive_cell_width_is_refused_at_the_boundary(width: int) -> None:
    with pytest.raises(
        dx.ValidationError, match="a span cell has at least one coordinate"
    ):
        ParserSpec.model_validate({**PARSER, "width": width})


@pytest.mark.parametrize("features", [[], ["left", "left"]])
def test_endpoint_features_are_nonempty_and_distinct(
    features: list[JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError, match="endpoint feature"):
        ParserSpec.model_validate({**PARSER, "features": features})


def test_an_unknown_flat_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="depth"):
        ParserSpec.model_validate({**PARSER, "depth": 2})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "flat", "width": 17, '
        '"features": ["left", "right", "bilinear"], '
        '"projection_initialization": "identity", "declared_output": null}'
    )
    declared = ParserSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert ParserSpec.model_validate_json(declared.model_dump_json()) == declared


def test_a_loaded_run_declaration_builds_the_flat_span_encoder(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "run.json",
        _document(
            {
                "kind": "flat",
                "width": 23,
                "features": ["bilinear"],
                "projection_initialization": "xavier-uniform",
            }
        ),
    )
    run = RunSpecCodec()(path)
    parser = run.model.parser

    assert parser is not None
    built = build_component(
        cast("FlatParserSpec", parser),
        plugin_context(component="parser", producers={"leaves": _pooled(11)}),
    )

    assert type(run) is RunSpec
    assert type(run.model.parser) is FlatParserSpec
    assert type(built) is FlatSpanEncoder
    assert built.input_width == 11
    assert built.width == 23
    assert built.features == ("bilinear",)
    assert built.projection_initialization == "xavier-uniform"


def test_a_sweep_holds_recursive_and_flat_parsers_as_one_group_axis(
    tmp_path: Path,
) -> None:
    run_path = _write(tmp_path / "run.json", _document())
    _write(
        tmp_path / "model" / "parser" / "inside-outside.json",
        {"kind": "inside-outside", "width": 29},
    )
    _write(
        tmp_path / "model" / "parser" / "flat.json",
        {
            "kind": "flat",
            "width": 29,
            "features": ["bilinear"],
            "projection_initialization": "identity",
        },
    )
    codec = RunSpecCodec(search_path=(tmp_path,))

    def compose(overrides: tuple[str, ...], _seed: int) -> Mapping[str, JsonValue]:
        return cast(
            "dict[str, JsonValue]",
            json.loads(codec(run_path, overrides=overrides).model_dump_json()),
        )

    points = GridSweepExpander(compose).expand(
        Sweep(
            grid=(
                Axis(
                    group="model.parser",
                    values=("inside-outside", "flat"),
                ),
            ),
            expected_points=2,
        )
    )
    recursive = points[0].config
    ablation = points[1].config
    recursive_parser = cast("dict[str, JsonValue]", recursive["model"])["parser"]
    ablation_parser = cast("dict[str, JsonValue]", ablation["model"])["parser"]
    cast("dict[str, JsonValue]", recursive["model"]).pop("parser")
    cast("dict[str, JsonValue]", ablation["model"]).pop("parser")

    assert points[0].overrides == ("model/parser=inside-outside",)
    assert points[1].overrides == ("model/parser=flat",)
    assert recursive_parser == {
        "kind": "inside-outside",
        "width": 29,
        "composition": {
            "kind": "mean-field",
            "leaf": {"kind": "mlp", "layers": 1, "activation": "tanh"},
            "cell": {"kind": "mlp", "layers": 1, "activation": "tanh"},
        },
        "reduction": {"kind": "log-sum-exp"},
        "attend": {"kind": "marginal"},
        "split_left": {"kind": "mlp", "layers": 1, "activation": "tanh"},
        "split_right": {"kind": "mlp", "layers": 1, "activation": "tanh"},
        "share_splits": False,
        "outside_weights": "contextual",
        "root_outside": "learned",
        "root_init_gain": None,
        "depth": 1,
        "layer_sharing": "independent",
        "transition": {"kind": "mlp", "layers": 1, "activation": "tanh"},
        "declared_output": None,
    }
    assert ablation_parser == {
        "kind": "flat",
        "width": 29,
        "features": ["bilinear"],
        "projection_initialization": "identity",
        "declared_output": None,
    }
    assert recursive == ablation


def test_a_run_naming_an_unknown_parser_is_refused_at_load_time(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "run.json",
        _document({"kind": "recursive-but-unregistered"}),
    )

    with pytest.raises(ConfigValueError) as refused:
        RunSpecCodec()(path)

    assert refused.value.path == "model.parser.kind"
    assert "recursive-but-unregistered" in refused.value.reason
