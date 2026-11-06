import argparse
import enum
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import didactic.api as dx
import pytest
from didactic.settings import (
    CoercionError,
    ConfigError,
    OverrideSyntaxError,
    compose,
)
from hypothesis import given
from hypothesis import strategies as st

from lofi.spec import (
    PARSERS,
    AmbiguousSettingError,
    ComputeDtype,
    DatasetSpec,
    FeaturizerSpec,
    LeafKind,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    SettingParseError,
    SinkSpec,
    TermEncoderSpec,
    TrainerSpec,
    TypedSetting,
    UndeclaredLeafError,
    UnknownSettingError,
    annotation_at,
    declared_paths,
    leaf_kind,
    parse_text,
    settings_from_environment,
    settings_from_namespace,
    settings_from_overrides,
    typed_setting,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.spec import Annotation


class Color(enum.Enum):
    RED = 1
    BLUE = 2


class Corner(dx.Model, extra="forbid"):
    x: int = 0
    y: int = 0


class Shape(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    pass


class Circle(Shape):
    kind: Literal["circle"] = "circle"
    radius: float = 1.0
    label: str | None = None


class Square(Shape):
    kind: Literal["square"] = "square"
    side: int = 1
    label: str | None = None


class Canvas(dx.Model, extra="forbid"):
    title: str = ""
    depth: int = 0
    scale: float = 1.0
    visible: bool = True
    origin: Corner = dx.field(default_factory=Corner)
    shape: Shape | None = None
    mode: Literal["draft", "final"] = "draft"
    color: Color = Color.RED
    dtype: ComputeDtype = ComputeDtype.FLOAT32
    output: Path = Path("out")
    tags: tuple[str, ...] = ()
    sizes: frozenset[int] = frozenset()
    layers: dict[str, Corner] = dx.field(default_factory=dict)
    note: Annotated[str, "a note"] = ""
    nested: dict[str, dict[str, int]] = dx.field(default_factory=dict)


class Stamped(dx.Model, extra="forbid"):
    when: datetime


class Ledger(dx.Model, extra="forbid"):
    stamps: dict[str, Stamped] = dx.field(default_factory=dict)


class Pinned(dx.Model, extra="forbid"):
    shape: Circle = dx.field(default_factory=Circle)


class Either(dx.Model, extra="forbid"):
    value: int | str = 0


class Diverging(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    pass


class WideByInt(Diverging):
    kind: Literal["by-int"] = "by-int"
    width: int = 1


class WideByText(Diverging):
    kind: Literal["by-text"] = "by-text"
    width: str = "one"


class Holder(dx.Model, extra="forbid"):
    slot: Diverging


class CoerceParser(ParserSpec):
    kind: Literal["coerce-parser"] = "coerce-parser"
    beam: int = 1


class CoerceEncoder(TermEncoderSpec):
    kind: Literal["coerce-encoder"] = "coerce-encoder"
    num_heads: int = 4
    dtype: ComputeDtype = ComputeDtype.FLOAT32


class CoerceLoss(LossSpec):
    kind: Literal["coerce-loss"] = "coerce-loss"


class CoerceDataset(DatasetSpec):
    kind: Literal["coerce-dataset"] = "coerce-dataset"
    root: Path = Path("data")
    partitions: tuple[str, ...] = ("train",)


class CoerceFeaturizer(FeaturizerSpec):
    kind: Literal["coerce-featurizer"] = "coerce-featurizer"


class CoerceTrainer(TrainerSpec):
    kind: Literal["coerce-trainer"] = "coerce-trainer"
    epochs: int = 1
    run_name: str | None = None
    amp: bool = False


class CoerceSgd(OptimizerSpec):
    kind: Literal["coerce-sgd"] = "coerce-sgd"
    lr: float = 0.1


class CoerceSink(SinkSpec):
    kind: Literal["coerce-sink"] = "coerce-sink"
    project: str = "lofi"


BASE_DOCUMENT: Final[dict[str, JsonValue]] = {
    "model": {
        "language": {"from": "data"},
        "parser": {"kind": "coerce-parser"},
    },
    "loss": {"terms": [{"objective": {"kind": "coerce-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "coerce-dataset"},
                "featurizer": {"kind": "coerce-featurizer"},
            }
        }
    },
    "trainer": {"kind": "coerce-trainer"},
    "parse": {"retain": "forest", "reachability": "all"},
    "release": {"checkpoint": "resume-latest"},
    "optimizer": {"kind": "coerce-sgd"},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
}


def read(path: str, text: str) -> JsonValue:
    return typed_setting(Canvas, path, text).value


@pytest.mark.parametrize(
    ("path", "text", "value"),
    [
        ("title", " spaced ", " spaced "),
        ("title", "123", "123"),
        ("depth", "42", 42),
        ("depth", "-7", -7),
        ("scale", "1e-3", 0.001),
        ("scale", "2", 2.0),
        ("visible", "yes", True),
        ("visible", "Off", False),
        ("visible", "0", False),
        ("mode", "final", "final"),
        ("color", "BLUE", 2),
        ("color", "1", 1),
        ("dtype", "float16", "float16"),
        ("dtype", "BFLOAT16", "bfloat16"),
        ("output", "2024", "2024"),
        ("output", "a/b.txt", "a/b.txt"),
        ("tags", "a, b ,c", ["a", "b", "c"]),
        ("tags", '["x", "y"]', ["x", "y"]),
        ("tags", "", []),
        ("tags", '"p,q",r', ['"p,q"', "r"]),
        ("sizes", "1,2,3", [1, 2, 3]),
        ("sizes", "[4]", [4]),
        ("note", "kept", "kept"),
        ("origin", '{"x": 1, "y": 2}', {"x": 1, "y": 2}),
        ("origin.x", "9", 9),
        ("layers", '{"a": {"x": 1}}', {"a": {"x": 1}}),
        ("layers.a.x", "5", 5),
        ("layers.a", '{"y": 3}', {"y": 3}),
        ("nested.a.b", "6", 6),
        ("shape", "null", None),
        ("shape", '{"kind": "circle"}', {"kind": "circle"}),
        ("shape.kind", "square", "square"),
        ("shape.radius", "2.5", 2.5),
        ("shape.side", "3", 3),
        ("shape.label", "~", None),
        ("shape.label", " NULL ", None),
        ("shape.label", "   ", None),
        ("shape.label", "named", "named"),
    ],
)
def test_text_reads_as_the_value_the_annotation_declares(
    path: str, text: str, value: JsonValue
) -> None:
    assert read(path, text) == value


def test_an_item_that_does_not_read_is_named_on_its_own() -> None:
    with pytest.raises(SettingParseError) as caught:
        read("sizes", "1,two")

    assert caught.value.path == "sizes"
    assert caught.value.expected == "int"
    assert caught.value.text == "two"


def test_a_typed_setting_pairs_the_path_with_its_value() -> None:
    assert typed_setting(Canvas, "depth", "3") == TypedSetting("depth", 3)


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    [
        ("depth", "three", "int"),
        ("depth", "1.5", "int"),
        ("scale", "fast", "float"),
        ("visible", "maybe", "bool"),
        ("mode", "sketch", "one of ['draft', 'final']"),
        ("color", "GREEN", "Color"),
        ("dtype", "float8", "ComputeDtype"),
        ("tags", "[1,", "valid JSON for tuple[str, ...]: Expecting value"),
        ("origin", "1,2", "JSON object text for Corner"),
        ("origin", "[1, 2]", "JSON object text for Corner"),
        ("origin", "{oops", "valid JSON for Corner: Expecting property name"),
        ("shape", "circle", "JSON object text for Shape"),
        ("shape.radius", "wide", "float"),
    ],
)
def test_text_that_does_not_read_names_the_path_type_and_text(
    path: str, text: str, expected: str
) -> None:
    with pytest.raises(SettingParseError) as caught:
        read(path, text)

    error = caught.value

    assert error.path == path
    assert error.text == text
    assert error.expected.startswith(expected)
    assert path in str(error)
    assert repr(text) in str(error)


@pytest.mark.parametrize(
    ("path", "declared"),
    [
        ("width", ("color", "depth", "dtype", "layers", "mode", "nested")),
        ("origin.z", ("origin.x", "origin.y")),
        ("depth.more", ("depth",)),
        ("tags.0", ("tags",)),
        ("shape.area", ("shape.kind", "shape.label", "shape.radius", "shape.side")),
        ("shape.kind.x", ("shape.kind",)),
        ("origin..x", ("color", "depth")),
        (".depth", ("color", "depth")),
    ],
)
def test_a_path_naming_no_setting_lists_what_is_declared_beside_it(
    path: str, declared: tuple[str, ...]
) -> None:
    with pytest.raises(UnknownSettingError) as caught:
        read(path, "1")

    assert caught.value.path == path
    assert set(declared) <= set(caught.value.declared)


def test_a_field_the_variants_declare_differently_is_ambiguous() -> None:
    with pytest.raises(AmbiguousSettingError) as caught:
        typed_setting(Holder, "slot.width", "3")

    assert caught.value.path == "slot.width"
    assert caught.value.variants == ("by-int", "by-text")


def test_a_field_the_variants_declare_alike_reads_once() -> None:
    assert annotation_at(Canvas, "shape.label") == (str | None)
    assert read("shape.label", "x") == "x"


def test_a_root_declaring_no_variants_lists_nothing_below_the_slot() -> None:
    class Empty(dx.TaggedUnion, discriminator="kind", extra="forbid"):
        pass

    class Owner(dx.Model, extra="forbid"):
        slot: Empty

    with pytest.raises(UnknownSettingError) as caught:
        typed_setting(Owner, "slot.width", "3")

    assert caught.value.declared == ("slot.kind",)


@pytest.mark.parametrize(
    ("annotation", "kind"),
    [
        (str, LeafKind.TEXT),
        (bool, LeafKind.BOOLEAN),
        (int, LeafKind.INTEGER),
        (float, LeafKind.FLOAT),
        (Path, LeafKind.PATH),
        (Literal["a", "b"], LeafKind.LITERAL),
        (Color, LeafKind.ENUM),
        (ComputeDtype, LeafKind.ENUM),
        (int | None, LeafKind.OPTIONAL),
        (Shape | None, LeafKind.OPTIONAL),
        (int | str, LeafKind.CHOICE),
        (bool | float, LeafKind.CHOICE),
        (tuple[int, ...], LeafKind.SEQUENCE),
        (frozenset[str], LeafKind.SEQUENCE),
        (dict[str, int], LeafKind.MAPPING),
        (Corner, LeafKind.MODEL),
        (Shape, LeafKind.MODEL),
        (Circle, LeafKind.MODEL),
        (Annotated[int, "x"], LeafKind.INTEGER),
        (dx.Embed[Corner], LeafKind.MODEL),
    ],
)
def test_every_declared_annotation_form_falls_under_one_kind(
    annotation: Annotation, kind: LeafKind
) -> None:
    assert leaf_kind(annotation, path="field") is kind


@pytest.mark.parametrize(
    "annotation",
    [
        datetime,
        int | datetime,
        int | str | None,
        list[int],
        set[int],
        tuple[int, str],
        dict[int, str],
        bytes,
        complex,
        object,
        datetime | None,
        tuple[datetime, ...],
        frozenset[complex],
        dict[str, datetime],
        dict[str, tuple[int | datetime, ...]],
        tuple[datetime, ...] | None,
    ],
)
def test_an_annotation_of_no_declared_kind_has_no_parser(
    annotation: Annotation,
) -> None:
    with pytest.raises(UndeclaredLeafError) as caught:
        leaf_kind(annotation, path="field")

    assert caught.value.path == "field"
    assert caught.value.annotation in str(caught.value)

    with pytest.raises(UndeclaredLeafError):
        parse_text("1", annotation, path="field")


def test_a_schema_declaring_an_unparsed_type_turns_the_walk_red() -> None:
    with pytest.raises(UndeclaredLeafError) as caught:
        typed_setting(Stamped, "when", "2026-01-01")

    assert caught.value.path == "when"
    assert caught.value.annotation == "datetime"


def test_a_scalar_union_reads_text_as_its_first_member_that_reads_it() -> None:
    assert typed_setting(Either, "value", "1") == TypedSetting("value", 1)
    assert typed_setting(Either, "value", "derived") == TypedSetting("value", "derived")
    assert parse_text("2.5", int | float, path="field") == 2.5

    with pytest.raises(SettingParseError, match="field"):
        parse_text("many", int | float, path="field")


def test_the_walk_reaches_an_unparsed_type_held_in_a_mapping() -> None:
    paths = dict(declared_paths(Ledger))

    assert paths["stamps.*"] is Stamped
    assert paths["stamps.*.when"] is datetime

    with pytest.raises(UndeclaredLeafError) as caught:
        leaf_kind(paths["stamps.*.when"], path="stamps.*.when")

    assert caught.value.path == "stamps.*.when"
    assert typed_setting(Ledger, "stamps", "{}") == TypedSetting("stamps", {})

    with pytest.raises(UndeclaredLeafError):
        typed_setting(Ledger, "stamps.a.when", "2026-01-01")


def test_a_parser_is_declared_for_every_kind() -> None:
    assert set(PARSERS) == set(LeafKind)


def test_the_run_spec_walk_reaches_every_registered_variant() -> None:
    paths = dict(declared_paths(RunSpec))

    assert paths["model.parser.beam"] is int
    assert paths["model.encoders.*.num_heads"] is int
    assert paths["model.encoders.*.kind"] is str
    assert paths["data.datasets.*.dataset.root"] is Path
    assert paths["trainer.run_name"] == (str | None)
    assert paths["sinks"] == dict[str, SinkSpec]
    assert paths["sinks.*"] is SinkSpec
    assert paths["sinks.*.kind"] is str
    assert paths["sinks.*.project"] is str
    assert "model.decoders.*.kind" in paths
    assert "probes.*.kind" in paths
    assert "evaluators.*.kind" in paths
    assert "model.parser.declared_output" not in paths


def test_every_setting_of_the_run_spec_has_a_declared_parser() -> None:
    paths = declared_paths(RunSpec)
    undeclared: list[str] = []

    for path, annotation in paths:
        try:
            leaf_kind(annotation, path=path)

        except UndeclaredLeafError as found:
            undeclared.append(str(found))

    assert len(paths) > len(RunSpec.__field_specs__)
    assert undeclared == []


def test_a_variant_declared_as_a_field_is_walked_as_itself() -> None:
    assert [path for path, _ in declared_paths(Pinned)] == [
        "shape",
        "shape.kind",
        "shape.radius",
        "shape.label",
    ]
    assert annotation_at(Pinned, "shape.kind") == Literal["circle"]
    assert typed_setting(Pinned, "shape.radius", "2") == TypedSetting(
        "shape.radius", 2.0
    )

    with pytest.raises(UnknownSettingError) as caught:
        typed_setting(Pinned, "shape.side", "3")

    assert caught.value.declared == ("shape.kind", "shape.label", "shape.radius")


def test_the_walk_lists_a_model_met_again_within_itself_as_a_slot() -> None:
    class Wrap(dx.Model, extra="forbid"):
        label: str = ""
        inner: "Wrap | None" = None

    class Tree(dx.Model, extra="forbid"):
        root: Wrap = dx.field(default_factory=Wrap)

    assert [path for path, _ in declared_paths(Tree)] == [
        "root",
        "root.label",
        "root.inner",
    ]
    assert annotation_at(Tree, "root.inner.inner.label") is str


def test_the_walk_lists_a_variant_holding_its_own_root_as_a_slot() -> None:
    class Part(dx.TaggedUnion, discriminator="kind", extra="forbid"):
        pass

    class Composite(Part):
        kind: Literal["composite"] = "composite"
        parts: tuple[Part, ...] = ()

    class Assembly(dx.Model, extra="forbid"):
        part: Part

    assert [path for path, _ in declared_paths(Assembly)] == [
        "part",
        "part.kind",
        "part.parts",
    ]


def test_the_walk_lists_a_mapping_entry_under_a_wildcard_segment() -> None:
    paths = dict(declared_paths(Canvas))

    assert paths["layers"] == dict[str, Corner]
    assert paths["layers.*"] is Corner
    assert paths["layers.*.x"] is int
    assert paths["nested.*"] == dict[str, int]
    assert paths["nested.*.*"] is int
    assert annotation_at(Canvas, "layers.*.x") is paths["layers.*.x"]
    assert "tags.*" not in paths


def test_the_walk_lists_a_differently_declared_field_per_declaration() -> None:
    paths = [(path, annotation) for path, annotation in declared_paths(Holder)]

    assert paths == [
        ("slot", Diverging),
        ("slot.kind", str),
        ("slot.width", int),
        ("slot.width", str),
    ]


@pytest.mark.parametrize(
    "path",
    ["title", "depth", "scale", "visible", "mode", "color", "output", "shape.label"],
)
def test_an_expression_is_kept_as_text_whatever_the_leaf(path: str) -> None:
    assert read(path, "${trainer.epochs}") == "${trainer.epochs}"
    assert read(path, "run-${trainer.run_name}") == "run-${trainer.run_name}"


@pytest.mark.parametrize("path", ["origin", "layers", "shape", "layers.a"])
def test_an_expression_fills_an_object_slot_as_text(path: str) -> None:
    assert read(path, "${model.parser}") == "${model.parser}"

    with pytest.raises(SettingParseError):
        read(path, "model.parser")


def test_a_whole_expression_fills_a_sequence_slot_as_text() -> None:
    assert read("sizes", "${trainer.sizes}") == "${trainer.sizes}"
    assert read("sizes", "1,${trainer.epochs}") == [1, "${trainer.epochs}"]
    assert read("tags", "a-${x},b") == ["a-${x}", "b"]


@given(st.integers())
def test_an_integer_survives_the_round_trip(value: int) -> None:
    assert read("depth", str(value)) == value


@given(st.floats(allow_nan=False))
def test_a_float_survives_the_round_trip(value: float) -> None:
    assert read("scale", repr(value)) == value


@given(value=st.booleans())
def test_a_boolean_survives_the_round_trip(*, value: bool) -> None:
    assert read("visible", str(value)) is value


@given(st.text())
def test_text_and_a_path_survive_the_round_trip(value: str) -> None:
    assert read("title", value) == value
    assert read("output", value) == value


@given(st.sampled_from(["draft", "final"]))
def test_a_literal_member_survives_the_round_trip(value: str) -> None:
    assert read("mode", value) == value


@given(st.sampled_from(list(Color)))
def test_an_enumeration_member_survives_the_round_trip(member: Color) -> None:
    assert read("color", member.name) == member.value
    assert read("color", str(member.value)) == member.value


@given(st.one_of(st.none(), st.text(min_size=1)))
def test_an_optional_survives_the_round_trip(value: str | None) -> None:
    text = "null" if value is None else value
    read_back = read("shape.label", text)

    if value is not None and value.strip().lower() in {"", "null", "~"}:
        assert read_back is None
    else:
        assert read_back == value


@given(st.lists(st.integers()))
def test_a_sequence_survives_the_round_trip_as_json_and_as_items(
    values: list[int],
) -> None:
    assert read("sizes", json.dumps(values)) == values
    assert read("sizes", ",".join(str(v) for v in values)) == values


@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_a_mapping_survives_the_round_trip(values: dict[str, int]) -> None:
    assert read("nested.a", json.dumps(values)) == values


ENVIRONMENT: Final = {
    "LOFI_CHECK": "1",
    "LOFI_TRAINER__EPOCHS": "5",
    "LOFI_OPTIMIZER__LR": "1e-3",
    "LOFI_DATA__DATASETS__TRAIN__DATASET__PARTITIONS": "train,dev",
    "LOFI_TRAINER__RUN_NAME": "",
    "HOME": "/nowhere",
    "TRAINER__EPOCHS": "9",
}


def test_the_environment_is_read_under_the_prefix_and_separator() -> None:
    assert settings_from_environment(RunSpec, ENVIRONMENT) == (
        TypedSetting("data.datasets.train.dataset.partitions", ["train", "dev"]),
        TypedSetting("optimizer.lr", 0.001),
        TypedSetting("trainer.epochs", 5),
        TypedSetting("trainer.run_name", None),
    )


def test_a_top_level_field_is_set_whole_from_its_variable() -> None:
    assert settings_from_environment(RunSpec, {"LOFI_TRAINER": "{}"}) == (
        TypedSetting("trainer", {}),
    )


def test_a_declared_switch_is_not_a_setting_and_an_undeclared_one_is() -> None:
    assert settings_from_environment(RunSpec, {"LOFI_CHECK": "1"}) == ()

    with pytest.raises(UnknownSettingError) as caught:
        settings_from_environment(RunSpec, {"LOFI_CHEK": "1"})

    assert caught.value.path == "chek"

    with pytest.raises(UnknownSettingError):
        settings_from_environment(RunSpec, {"LOFI_CHECK": "1"}, switches=frozenset())

    assert (
        settings_from_environment(
            RunSpec,
            {"LOFI_CHECK": "1", "LOFI_TRACE": "1"},
            switches=frozenset({"LOFI_CHECK", "LOFI_TRACE"}),
        )
        == ()
    )


def test_the_prefix_and_separator_are_parameters() -> None:
    environment = {"APP-TRAINER.EPOCHS": "3", "LOFI_TRAINER__EPOCHS": "4"}

    assert settings_from_environment(
        RunSpec, environment, prefix="APP-", separator="."
    ) == (TypedSetting("trainer.epochs", 3),)


def test_a_variable_naming_no_setting_is_refused() -> None:
    with pytest.raises(UnknownSettingError) as caught:
        settings_from_environment(RunSpec, {"LOFI_TRAINER__EPOCH": "3"})

    assert caught.value.path == "trainer.epoch"


def test_a_variable_that_does_not_read_is_refused() -> None:
    with pytest.raises(SettingParseError) as caught:
        settings_from_environment(RunSpec, {"LOFI_TRAINER__EPOCHS": "many"})

    assert caught.value.path == "trainer.epochs"
    assert caught.value.text == "many"


def test_a_namespace_is_read_by_dotted_or_double_underscore_names() -> None:
    namespace = argparse.Namespace(
        trainer__epochs="7", **{"optimizer.lr": "0.5", "trainer.amp": None}
    )

    assert settings_from_namespace(RunSpec, namespace) == (
        TypedSetting("trainer.epochs", 7),
        TypedSetting("optimizer.lr", 0.5),
    )


def test_a_namespace_holding_a_typed_value_is_refused() -> None:
    with pytest.raises(TypeError, match="trainer__epochs"):
        settings_from_namespace(RunSpec, argparse.Namespace(trainer__epochs=7))


def test_a_namespace_argument_naming_no_setting_is_refused() -> None:
    with pytest.raises(UnknownSettingError) as caught:
        settings_from_namespace(RunSpec, argparse.Namespace(trainer__epoch="7"))

    assert caught.value.path == "trainer.epoch"


def test_overrides_are_read_and_group_selections_pass_through() -> None:
    expressions = [
        "trainer.epochs=9",
        "model/encoders/terms=rnn",
        "trainer.run_name=abc",
        "model.encoders.terms.dtype=float64",
    ]

    assert settings_from_overrides(RunSpec, expressions) == (
        TypedSetting("trainer.epochs", 9),
        "model/encoders/terms=rnn",
        TypedSetting("trainer.run_name", "abc"),
        TypedSetting("model.encoders.terms.dtype", "float64"),
    )


@pytest.mark.parametrize("expression", ["trainer.epochs", "=3", "trainer..epochs=3"])
def test_a_malformed_override_is_refused_by_the_engine_grammar(
    expression: str,
) -> None:
    with pytest.raises(OverrideSyntaxError):
        settings_from_overrides(RunSpec, [expression])


def test_an_override_that_does_not_read_is_refused() -> None:
    with pytest.raises(SettingParseError) as caught:
        settings_from_overrides(RunSpec, ["optimizer.lr=fast"])

    assert caught.value.path == "optimizer.lr"
    assert caught.value.expected == "float"
    assert caught.value.text == "fast"


def test_typed_settings_compose_without_the_engine_reading_text() -> None:
    settings = settings_from_overrides(
        RunSpec, ["trainer.epochs=3", "optimizer.lr=0.25", "trainer.amp=on"]
    )

    spec = compose(None, schema=RunSpec, base=BASE_DOCUMENT, overrides=settings)

    assert spec.trainer == CoerceTrainer(epochs=3, amp=True)
    assert spec.optimizer == CoerceSgd(lr=0.25)


def test_the_engine_refuses_text_a_typed_pair_carries() -> None:
    with pytest.raises(CoercionError) as caught:
        compose(
            None,
            schema=RunSpec,
            base=BASE_DOCUMENT,
            overrides=[("optimizer.lr", "0.25")],
        )

    assert caught.value.path == "optimizer.lr"


def test_environment_settings_compose_under_the_project_prefix() -> None:
    settings = settings_from_environment(
        RunSpec,
        {
            "LOFI_DATA__DATASETS__TRAIN__DATASET__ROOT": "corpora/2024",
            "LOFI_DATA__DATASETS__TRAIN__DATASET__PARTITIONS": "train,test",
            "LOFI_SINKS": '{"main": {"kind": "coerce-sink"}}',
        },
    )

    spec = compose(None, schema=RunSpec, base=BASE_DOCUMENT, overrides=settings)

    assert spec.data.datasets["train"].dataset == CoerceDataset(
        root=Path("corpora/2024"), partitions=("train", "test")
    )
    assert spec.sinks == {"main": CoerceSink()}


def test_an_expression_read_by_the_adapter_resolves_in_the_engine() -> None:
    settings = settings_from_overrides(
        RunSpec,
        [
            "trainer.epochs=4",
            "model.parser.beam=${trainer.epochs}",
            "data.datasets.train.dataset.partitions=${trainer.epochs},dev",
        ],
    )

    assert settings[1] == TypedSetting("model.parser.beam", "${trainer.epochs}")
    assert settings[2] == TypedSetting(
        "data.datasets.train.dataset.partitions", ["${trainer.epochs}", "dev"]
    )

    spec = compose(None, schema=RunSpec, base=BASE_DOCUMENT, overrides=settings)

    assert spec.model.parser == CoerceParser(beam=4)
    assert spec.data.datasets["train"].dataset.partitions == ("4", "dev")


def test_an_expression_fills_a_model_or_sequence_slot_in_the_engine() -> None:
    settings = settings_from_overrides(
        Canvas, ["origin.x=3", "layers.a=${origin}", "tags=${sizes}", "sizes=1,2"]
    )

    canvas = compose(None, schema=Canvas, overrides=settings)

    assert canvas.layers == {"a": Corner(x=3)}
    assert canvas.tags == ("1", "2")


def test_an_expression_at_a_union_slot_is_refused_by_the_engine() -> None:
    settings = settings_from_overrides(RunSpec, ["sinks.main=${model.parser}"])

    assert settings == (TypedSetting("sinks.main", "${model.parser}"),)

    with pytest.raises(ConfigError) as caught:
        compose(None, schema=RunSpec, base=BASE_DOCUMENT, overrides=settings)

    assert caught.value.path == "sinks.main"
