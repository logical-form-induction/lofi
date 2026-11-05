import json
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run._boundary.sweep import (
    Axis,
    FractionalFactorialDesignSpec,
    LatinHypercubeDesignSpec,
    RandomDesignSpec,
    Sweep,
    SweepPoint,
    overlay_settings,
    point_id,
    read_overlay,
)
from lofi.spec import ConfigDocumentError

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_a_sweep_declares_each_dimension() -> None:
    batch = Axis(path="trainer.batch_size", values=(16, 32))
    width = Axis(path="model.width", values=(128, 256))
    depth = Axis(path="model.depth", values=(2, 4))
    sweep = Sweep(
        grid=(batch,),
        yoked=((width, depth),),
        exclude=({"trainer.batch_size": 32, "model.width": 256},),
        seeds=(3, 5),
    )

    assert sweep.grid == (batch,)
    assert sweep.yoked == ((width, depth),)
    assert sweep.exclude[0]["model.width"] == 256
    assert sweep.seeds == (3, 5)


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"kind": "random", "points": 3, "seed": 7}, RandomDesignSpec),
        (
            {"kind": "latin-hypercube", "points": 4, "seed": 11},
            LatinHypercubeDesignSpec,
        ),
        (
            {
                "kind": "fractional-factorial",
                "generators": [
                    {"factor": "e", "word": ["a", "b", "c"]},
                    {"factor": "f", "word": ["b", "c", "d"]},
                ],
            },
            FractionalFactorialDesignSpec,
        ),
    ],
)
def test_a_sweep_declares_an_open_assignment_design(
    document: dict[str, JsonValue], expected: type[object]
) -> None:
    sweep = Sweep.model_validate({"design": document})

    assert isinstance(sweep.design, expected)


def test_an_axis_requires_a_path_and_a_value() -> None:
    with pytest.raises(dx.ValidationError):
        Axis(path="", values=(1,))

    with pytest.raises(dx.ValidationError):
        Axis(path="model.width", values=())


def test_yoked_axes_have_equal_lengths() -> None:
    with pytest.raises(dx.ValidationError):
        Sweep(
            yoked=(
                (
                    Axis(path="model.width", values=(64, 128)),
                    Axis(path="model.depth", values=(2,)),
                ),
            )
        )


def test_a_sweep_requires_distinct_nonnegative_seeds() -> None:
    for seeds in ((), (-1,), (3, 3)):
        with pytest.raises(dx.ValidationError):
            Sweep(seeds=seeds)


def test_a_sweep_declares_a_nonnegative_expected_point_count_or_none() -> None:
    assert Sweep().expected_points is None
    assert Sweep(expected_points=0).expected_points == 0
    assert Sweep(expected_points=12).expected_points == 12

    with pytest.raises(dx.ValidationError):
        Sweep(expected_points=-1)


def test_a_sweep_refuses_an_empty_exclusion_clause() -> None:
    axis = Axis(path="model.width", values=(64, 128))

    with pytest.raises(dx.ValidationError, match="names no axis"):
        Sweep(grid=(axis,), exclude=({},))


def test_a_sweep_refuses_a_clause_naming_an_undeclared_path() -> None:
    axis = Axis(path="model.width", values=(64, 128))

    with pytest.raises(dx.ValidationError, match="undeclared"):
        Sweep(grid=(axis,), exclude=({"model.width": 64, "model.depth": 2},))

    with pytest.raises(dx.ValidationError, match="undeclared"):
        Sweep(exclude=({"model.width": 64},))


def test_a_sweep_refuses_a_path_repeated_across_grid_and_yoked_axes() -> None:
    width = Axis(path="model.width", values=(64, 128))
    depth = Axis(path="model.depth", values=(2, 4))

    with pytest.raises(dx.ValidationError, match="repeat a key"):
        Sweep(grid=(width,), yoked=((width, depth),))


def test_a_sweep_refuses_assignments_whose_paths_nest() -> None:
    whole = Axis(path="model", values=("small", "large"))
    width = Axis(path="model.width", values=(64, 128))
    depth = Axis(path="model.depth", values=(2, 4))

    with pytest.raises(dx.ValidationError, match="conflict"):
        Sweep(grid=(whole, width))

    with pytest.raises(dx.ValidationError, match="conflict"):
        Sweep(grid=(whole,), yoked=((width, depth),))


def test_point_identity_is_invariant_to_key_order() -> None:
    left: dict[str, JsonValue] = {
        "model": {"width": 64, "depth": 2},
        "seed": 3,
    }
    right: dict[str, JsonValue] = {
        "seed": 3,
        "model": {"depth": 2, "width": 64},
    }

    assert point_id(left) == point_id(right)
    assert point_id(left, seed=4) == point_id(right, seed=4)
    assert len(point_id(left)) == 16


def test_point_identity_is_the_leading_hex_of_the_configuration_digest() -> None:
    config: dict[str, JsonValue] = {
        "model": {"width": 64, "depth": 2},
        "trainer": {"epochs": 3},
    }
    document = json.dumps(
        {"configuration": config, "seed": 3},
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256(document.encode()).hexdigest()

    assert point_id(config, seed=3) == digest[:16]
    assert point_id(config, seed=3) == "9145dd6292a57bfb"
    assert point_id(config) == "759b70499acc26bc"
    assert set(point_id(config)) <= set("0123456789abcdef")


def test_point_identity_leaves_out_the_observation_sections() -> None:
    config: dict[str, JsonValue] = {
        "model": {"width": 64, "depth": 2},
        "trainer": {"epochs": 3},
    }
    observed: dict[str, JsonValue] = {
        **config,
        "sinks": {"tracking": {"kind": "mlflow", "experiment_name": "lofi"}},
    }

    assert point_id(observed, seed=3) == point_id(config, seed=3)
    assert point_id({**config, "sinks": {}}) == point_id(config)
    assert point_id({**config, "probes": {}}) != point_id(config)


def test_point_identity_distinguishes_seeds() -> None:
    config: dict[str, JsonValue] = {"model": {"width": 64}}

    assert point_id(config) == point_id(config, seed=0)
    assert point_id(config, seed=1) != point_id(config, seed=2)
    assert point_id(config, seed=1) != point_id({"model": {"width": 65}}, seed=1)


def test_point_identity_reads_any_mapping() -> None:
    config: dict[str, JsonValue] = {"model": {"width": 64}}

    assert point_id(MappingProxyType(config)) == point_id(config)


def test_a_point_copies_its_resolved_configuration() -> None:
    source: dict[str, JsonValue] = {"model": {"width": 64}, "seed": 3}
    point = SweepPoint(source, overrides=("model.width=64",), seed=3)
    source["model"] = {"width": 128}
    returned = point.config
    returned["seed"] = 9

    assert point.config == {"model": {"width": 64}, "seed": 3}
    assert point.point_id == point_id(point.config, seed=3)
    assert point.overrides == ("model.width=64",)
    assert point.seed == 3


def test_points_are_identified_by_configuration_and_seed_only() -> None:
    config: dict[str, JsonValue] = {"model": {"width": 64}}
    first = SweepPoint(config, overrides=("model.width=64",), seed=1)
    reordered = SweepPoint({"model": {"width": 64}}, overrides=(), seed=1)
    reseeded = SweepPoint(config, overrides=("model.width=64",), seed=2)

    assert first == reordered
    assert hash(first) == hash(reordered)
    assert first != reseeded
    assert hash(first) != hash(reseeded)
    assert first.point_id != reseeded.point_id
    assert len({first, reordered, reseeded}) == 2
    assert first != config


def test_an_overlay_is_read_from_json_toml_and_yaml_by_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        '{"trainer": {"epochs": 1}}', encoding="utf-8", newline="\n"
    )
    (tmp_path / "b.toml").write_text(
        "[trainer]\nepochs = 2\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "c.yaml").write_text(
        "trainer:\n  epochs: 3\n", encoding="utf-8", newline="\n"
    )

    assert read_overlay(tmp_path / "a.json") == {"trainer": {"epochs": 1}}
    assert read_overlay(tmp_path / "b.toml") == {"trainer": {"epochs": 2}}
    assert read_overlay(tmp_path / "c.yaml") == {"trainer": {"epochs": 3}}


def test_an_absent_or_unparseable_overlay_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8", newline="\n")

    with pytest.raises(ConfigDocumentError) as absent:
        read_overlay(tmp_path / "absent.json")

    with pytest.raises(ConfigDocumentError) as unparseable:
        read_overlay(broken)

    assert absent.value.document == tmp_path / "absent.json"
    assert unparseable.value.document == broken


def test_an_overlay_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    listed = tmp_path / "listed.json"
    listed.write_text("[1, 2]", encoding="utf-8", newline="\n")

    with pytest.raises(ConfigDocumentError) as refused:
        read_overlay(listed)

    assert refused.value.document == listed


def test_overlay_settings_are_the_leaves_with_later_overlays_winning(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "trainer": {"epochs": 1, "kind": "t"},
                "loss": {"terms": [{"objective": {"kind": "l"}}]},
                "probes": {},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    second.write_text(
        json.dumps({"trainer": {"epochs": 2}, "seed": None}),
        encoding="utf-8",
        newline="\n",
    )

    assert overlay_settings((first, second)) == {
        "trainer.epochs": 2,
        "trainer.kind": "t",
        "loss.terms": [{"objective": {"kind": "l"}}],
        "probes": {},
        "seed": None,
    }
    assert overlay_settings(()) == {}


def test_an_axis_declares_a_group_or_a_path_but_not_both() -> None:
    group = Axis(
        group="data.datasets.train.dataset.grammar", values=("directed", "undirected")
    )
    leaf = Axis(
        path="data.datasets.train.dataset.grammar.budget.unary_rounds", values=(1, 2)
    )

    assert group.selects_group
    assert not leaf.selects_group
    assert group.key == "data.datasets.train.dataset.grammar"
    assert leaf.key == "data.datasets.train.dataset.grammar.budget.unary_rounds"
    assert group.slot == ("data", "datasets", "train", "dataset", "grammar")
    assert leaf.slot == (
        "data",
        "datasets",
        "train",
        "dataset",
        "grammar",
        "budget",
        "unary_rounds",
    )

    with pytest.raises(dx.ValidationError, match="exactly one"):
        Axis(path="trainer.epochs", group="trainer", values=(1,))

    with pytest.raises(dx.ValidationError, match="exactly one"):
        Axis(values=(1,))


def test_a_group_at_the_root_of_the_document_is_an_axis() -> None:
    axis = Axis(group="device_policy", values=("cpu", "cuda-mixed"))

    assert axis.selects_group
    assert axis.slot == ("device_policy",)
    assert axis.points == ("cpu", "cuda-mixed")


def test_an_axis_path_and_group_are_dotted_with_no_empty_segment() -> None:
    with pytest.raises(dx.ValidationError, match="dotted"):
        Axis(group="data/datasets/train/dataset/grammar", values=("directed",))

    with pytest.raises(dx.ValidationError, match="dotted"):
        Axis(path="data/datasets/train/dataset.grammar", values=("directed",))

    with pytest.raises(dx.ValidationError, match="empty segment"):
        Axis(group="data..grammar", values=("directed",))

    with pytest.raises(dx.ValidationError, match="empty segment"):
        Axis(path="data..grammar", values=(1,))


def test_a_config_group_axis_takes_fragment_names_or_nothing() -> None:
    axis = Axis(
        group="data.datasets.train.dataset.grammar.signature", values=("narrow", None)
    )

    assert axis.values == ("narrow", None)

    unnamed: tuple[tuple[JsonValue, ...], ...] = (
        (256,),
        (True,),
        ("",),
        (["narrow"],),
        ({"kind": "narrow"},),
    )

    for values in unnamed:
        with pytest.raises(dx.ValidationError, match="fragment names"):
            Axis(group="data.datasets.train.dataset.grammar", values=values)

    assert Axis(path="model.encoders.terms.width", values=(256,)).values == (256,)


def test_a_leaf_assignment_composes_with_the_group_selection_above_it() -> None:
    selection = Axis(group="model.encoders.terms", values=("hrr", "transformer"))
    width = Axis(path="model.encoders.terms.width", values=(64, 128))
    sweep = Sweep(grid=(selection, width))

    assert sweep.grid == (selection, width)

    with pytest.raises(dx.ValidationError, match="conflict"):
        Sweep(grid=(width, Axis(path="model.encoders.terms.width.scale", values=(2,))))


def test_nested_group_selections_cross_only_where_they_are_yoked() -> None:
    grammar = Axis(
        group="data.datasets.train.dataset.grammar", values=("directed", "undirected")
    )
    signature = Axis(
        group="data.datasets.train.dataset.grammar.signature", values=("narrow", None)
    )

    with pytest.raises(dx.ValidationError, match="conflict"):
        Sweep(grid=(grammar, signature))

    with pytest.raises(dx.ValidationError, match="conflict"):
        Sweep(grid=(grammar,), yoked=((signature, Axis(path="a.b", values=(1, 2))),))

    assert Sweep(yoked=((grammar, signature),)).yoked == ((grammar, signature),)


def test_a_group_and_a_path_naming_one_slot_are_one_key() -> None:
    with pytest.raises(dx.ValidationError, match="repeat a key"):
        Sweep(
            grid=(
                Axis(group="model.encoders.terms", values=("hrr",)),
                Axis(path="model.encoders.terms", values=({"kind": "hrr"},)),
            )
        )


def test_a_grid_axis_refuses_a_repeated_value() -> None:
    with pytest.raises(dx.ValidationError, match="repeats a value"):
        Sweep(grid=(Axis(path="model.width", values=(64, 128, 64)),))

    with pytest.raises(dx.ValidationError, match="repeats a value"):
        Sweep(grid=(Axis(group="device_policy", values=("cpu", "cpu")),))

    with pytest.raises(dx.ValidationError, match="repeats a value"):
        Sweep(
            grid=(Axis(path="loss", values=({"a": 1, "b": [2]}, {"b": [2], "a": 1})),)
        )


def test_values_of_different_types_are_different_grid_values() -> None:
    axis = Axis(path="model.scale", values=(1, 1.0, True, "1", [1]))

    assert Sweep(grid=(axis,)).grid == (axis,)


def test_a_yoked_group_refuses_a_repeated_tuple() -> None:
    with pytest.raises(dx.ValidationError, match="repeats a tuple"):
        Sweep(
            yoked=(
                (
                    Axis(path="model.width", values=(4, 4, 8)),
                    Axis(path="model.depth", values=(4, 4, 8)),
                ),
            )
        )


def test_a_yoked_axis_may_repeat_a_value_its_partner_tells_apart() -> None:
    encoder = Axis(group="model.encoders.terms", values=("rnn", "hrr", "rnn"))
    decoder = Axis(
        group="model.decoders.primary",
        values=("diffusion", "diffusion", "rnn"),
    )

    assert Sweep(yoked=((encoder, decoder),)).yoked == ((encoder, decoder),)


def test_axis_points_keep_the_type_and_length_they_were_declared_with() -> None:
    axis = Axis(
        path="data.datasets.train.dataset.grammar.signature.primitives",
        values=(["S", "NP"], ["S"], "S", 2),
    )
    pair, single, scalar, number = axis.points

    assert pair == ["S", "NP"]
    assert type(pair) is list
    assert single == ["S"]
    assert type(single) is list
    assert len(single) == 1
    assert scalar == "S"
    assert type(scalar) is str
    assert number == 2
    assert type(number) is int


def test_a_one_element_axis_point_does_not_come_back_as_its_element() -> None:
    axis = Axis(
        path="data.datasets.train.dataset.grammar.goal.categories", values=(["S"],)
    )
    (point,) = axis.points

    assert point != "S"
    assert not isinstance(point, str)
    assert point == ["S"]


def test_a_nested_axis_point_keeps_every_level_as_a_list() -> None:
    axis = Axis(
        path="model.primitive_types",
        values=([["e", "t"], ["e", "v", "t"]],),
    )
    (point,) = axis.points

    assert point == [["e", "t"], ["e", "v", "t"]]
    assert type(point) is list
    assert [type(inner) for inner in point] == [list, list]


def test_a_leaf_axis_point_may_be_a_whole_mapping() -> None:
    loss: JsonValue = {
        "terms": [
            {"objective": {"kind": "grammaticality"}, "weight": 1.0},
            {"objective": {"kind": "coherence", "estimator": "exact"}, "weight": 0.1},
        ]
    }
    axis = Axis(path="loss", values=(loss, {"terms": []}))
    whole, empty = axis.points

    assert whole == loss
    assert isinstance(whole, dict)
    assert isinstance(whole["terms"], list)
    assert [type(term) for term in whole["terms"]] == [dict, dict]
    assert empty == {"terms": []}


def test_a_sweep_round_trips_through_its_own_document() -> None:
    sweep = Sweep(
        grid=(
            Axis(path="a.b", values=(["S", "NP"], ["S"], "S")),
            Axis(path="c", values=({"kind": "x", "widths": [1, 2]},)),
            Axis(group="device_policy", values=("cpu", None)),
        ),
        exclude=({"a.b": ["S"]}, {"c": {"kind": "x", "widths": [1, 2]}}),
        expected_points=4,
    )
    again = Sweep.model_validate(json.loads(sweep.model_dump_json()))

    assert again == sweep
    assert again.grid[0].points == (["S", "NP"], ["S"], "S")
    assert again.grid[1].points == ({"kind": "x", "widths": [1, 2]},)
    assert again.grid[2].selects_group
    assert again.clauses == ({"a.b": ["S"]}, {"c": {"kind": "x", "widths": [1, 2]}})


def test_a_config_group_axis_refuses_a_list_point() -> None:
    with pytest.raises(dx.ValidationError, match="fragment names"):
        Axis(
            group="data.datasets.train.dataset.grammar.signature", values=(["narrow"],)
        )


def test_a_yoked_group_counts_points_rather_than_their_elements() -> None:
    primitives = Axis(
        path="data.datasets.train.dataset.grammar.signature.primitives",
        values=(["S", "NP"], ["S", "NP", "N"]),
    )
    rounds = Axis(
        path="data.datasets.train.dataset.grammar.budget.unary_rounds", values=(1, 2)
    )
    ((listed, scalar),) = Sweep(yoked=((primitives, rounds),)).yoked
    sized = [len(point) for point in listed.points if isinstance(point, list)]

    assert sized == [2, 3]
    assert len(listed.points) == len(scalar.points) == 2
    assert scalar.points == (1, 2)

    with pytest.raises(dx.ValidationError, match=r"different lengths: \[1, 2\]"):
        Sweep(
            yoked=(
                (
                    Axis(path="a.b", values=(["S", "NP"],)),
                    Axis(path="c.d", values=(1, 2)),
                ),
            )
        )

    with pytest.raises(dx.ValidationError, match=r"different lengths: \[2, 3\]"):
        Sweep(
            yoked=(
                (
                    Axis(path="a.b", values=(["S"], ["S", "NP"], ["S", "NP", "N"])),
                    Axis(path="c.d", values=(1, 2)),
                ),
            )
        )


def test_exclusion_clause_values_keep_the_type_they_were_declared_with() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=(["S", "NP"], ["S"], "S")),),
        exclude=({"a.b": ["S"]}, {"a.b": "S"}),
    )
    listed, scalar = sweep.clauses

    assert listed == {"a.b": ["S"]}
    assert type(listed["a.b"]) is list
    assert scalar == {"a.b": "S"}
    assert type(scalar["a.b"]) is str


def test_an_exclusion_clause_names_a_group_axis_by_its_slot() -> None:
    decoder = Axis(group="model.decoders.primary", values=("rnn", "hrr"))
    sweep = Sweep(grid=(decoder,), exclude=({"model.decoders.primary": "rnn"},))

    assert sweep.clauses == ({"model.decoders.primary": "rnn"},)

    with pytest.raises(dx.ValidationError, match="undeclared"):
        Sweep(grid=(decoder,), exclude=({"model/decoders/primary": "rnn"},))
