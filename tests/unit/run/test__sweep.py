import copy
import json
import os
import subprocess
import sys
import tempfile
from json import loads
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
import yaml
from didactic.settings import MissingFragmentError
from didactic.settings import compose as compose_settings
from hypothesis import given, settings
from tests.conformance.component_plugin import plugin_context
from tests.conformance.sweep import SweepConformance, SweepFixture
from tests.declarations import (
    SMALL,
    Candidates,
    DataKind,
    DataSection,
    candidate_sets,
    category_language,
    exceeds_shipped_ranks,
    run_document,
    train_spec,
)
from tests.unit.run.conftest import write_priced_experiment

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
from lofi.data.synthetic.spec.corpus import fingerprint as corpus_fingerprint
from lofi.nn.arity import ArityUnsupportedError
from lofi.nn.build import build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.run import (
    Axis,
    ExperimentError,
    FractionalFactorialDesignSpec,
    GeneratorSpec,
    GridDesignSpec,
    LatinHypercubeDesignSpec,
    MeasuredRole,
    PointManifest,
    RandomDesignSpec,
    Sweep,
    SweepExpander,
    SweepPoint,
    expand_experiment,
    measure_point,
    point_id,
    read_experiment,
    resolve_device_policy,
    resolve_point,
    resolved_document,
)
from lofi.run._sweep import (
    GridSweepExpander,
    ShadowedAxisError,
    UnexpectedPointCountError,
    unique_points,
)
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ConfigSource,
    LeafOrigin,
    LossSpec,
    OptimizerSpec,
    Provenance,
    RunSpec,
    RunSpecCodec,
    SpecLoader,
    TrainerSpec,
    config_search_path,
)
from lofi.spec._plugins import load_plugins
from lofi.spec.grammar import GrammarDecl, GrammarIso, declare_grammar
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch import nn

    from lofi._types import JsonValue
    from lofi.spec import ConfiguredSpec

TREE: Final = Path(__file__).resolve().parents[3]

RENDER_SCRIPT: Final = """
import sys

from tests.unit.run.test__sweep import expansion_bytes

sys.stdout.buffer.write(expansion_bytes())
"""


def leaf(config: Mapping[str, JsonValue], path: str) -> JsonValue:
    """Return the value a resolved document holds at a dotted path."""
    node: JsonValue = dict(config)

    for segment in path.split("."):
        if not isinstance(node, dict):
            msg = f"{path!r} runs past a leaf"
            raise TypeError(msg)

        node = node[segment]

    return node


def _read(text: str) -> JsonValue:
    try:
        return cast("JsonValue", json.loads(text))

    except json.JSONDecodeError:
        return text


def compose(overrides: tuple[str, ...], _seed: int) -> dict[str, JsonValue]:
    resolved: dict[str, JsonValue] = {"run": {"name": "base"}}

    for expression in overrides:
        key, _, text = expression.partition("=")
        *branches, leaf = key.split(".")
        node = resolved

        for branch in branches:
            child = node.setdefault(branch, {})

            if not isinstance(child, dict):
                msg = f"{branch!r} is a leaf"
                raise TypeError(msg)

            node = child

        node[leaf] = _read(text)

    return resolved


def fixture_sweep() -> Sweep:
    return Sweep(
        grid=(
            Axis(path="trainer.batch_size", values=(16, 32)),
            Axis(path="optimizer.lr", values=(0.1, 0.01)),
        ),
        yoked=(
            (
                Axis(path="model.width", values=(128, 256)),
                Axis(path="model.depth", values=(2, 4)),
            ),
        ),
        exclude=(
            {"trainer.batch_size": 32, "model.width": 256},
            {"optimizer.lr": 0.01},
        ),
        seeds=(1, 2, 3),
    )


HAND_ENUMERATED: Final[tuple[tuple[tuple[str, ...], int], ...]] = tuple(
    (overrides, seed)
    for overrides in (
        (
            "trainer.batch_size=16",
            "optimizer.lr=0.1",
            "model.width=128",
            "model.depth=2",
        ),
        (
            "trainer.batch_size=16",
            "optimizer.lr=0.1",
            "model.width=256",
            "model.depth=4",
        ),
        (
            "trainer.batch_size=32",
            "optimizer.lr=0.1",
            "model.width=128",
            "model.depth=2",
        ),
    )
    for seed in (1, 2, 3)
)


def expansion_bytes() -> bytes:
    points = GridSweepExpander(compose).expand(fixture_sweep())
    rows = [
        {
            "point_id": point.point_id,
            "seed": point.seed,
            "overrides": list(point.overrides),
            "config": point.config,
        }
        for point in points
    ]

    return json.dumps(rows, sort_keys=True).encode()


def expansion_from_a_fresh_interpreter(seed: int) -> bytes:
    finished = subprocess.run(
        [sys.executable, "-"],
        input=RENDER_SCRIPT.encode(),
        capture_output=True,
        check=True,
        cwd=TREE,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
    )

    return finished.stdout


class TestGridSweepExpander(SweepConformance):
    @pytest.fixture
    def sweep_fixture(self) -> SweepFixture:
        return SweepFixture(
            expansion_from_a_fresh_interpreter,
            point_id,
            unique_points,
            GridSweepExpander(compose).expand(fixture_sweep()),
        )


def test_the_expander_implements_the_protocol() -> None:
    assert isinstance(GridSweepExpander(compose), SweepExpander)


def test_the_fixture_expands_to_the_hand_enumerated_points_in_order() -> None:
    points = GridSweepExpander(compose).expand(fixture_sweep())

    assert tuple((point.overrides, point.seed) for point in points) == HAND_ENUMERATED


def test_every_override_round_trips_through_the_composer() -> None:
    points = GridSweepExpander(compose).expand(fixture_sweep())

    assert all(point.config == compose(point.overrides, point.seed) for point in points)
    assert points[0].config == {
        "run": {"name": "base"},
        "trainer": {"batch_size": 16},
        "optimizer": {"lr": 0.1},
        "model": {"width": 128, "depth": 2},
    }


def test_the_composer_runs_once_per_surviving_seeded_assignment() -> None:
    calls: list[tuple[tuple[str, ...], int]] = []

    def counting(overrides: tuple[str, ...], seed: int) -> dict[str, JsonValue]:
        calls.append((overrides, seed))

        return compose(overrides, seed)

    GridSweepExpander(counting).expand(fixture_sweep())

    assert tuple(calls) == HAND_ENUMERATED


def test_traced_provenance_is_carried_to_every_seed_of_its_assignment() -> None:
    calls: list[tuple[tuple[str, ...], int]] = []
    provenance = Provenance(
        origins={"run.name": LeafOrigin(source=ConfigSource.DEFAULT)}
    )

    def resolved(overrides: tuple[str, ...], seed: int) -> Provenance:
        calls.append((overrides, seed))

        return provenance

    points = GridSweepExpander(compose, provenance=resolved).expand(fixture_sweep())

    assert tuple(calls) == HAND_ENUMERATED
    assert all(point.provenance == provenance for point in points)


def test_scalars_render_as_stable_override_text() -> None:
    sweep = Sweep(
        grid=(
            Axis(path="a", values=("text", "with spaces", "x=y")),
            Axis(path="b", values=(True, False, None)),
            Axis(path="c", values=(3, 2.5, 1e-05)),
        ),
    )
    points = GridSweepExpander(compose).expand(sweep)

    assert points[0].overrides == ("a=text", "b=true", "c=3")
    assert points[-1].overrides == ("a=x=y", "b=null", "c=1e-05")
    assert {point.overrides[1] for point in points} == {"b=true", "b=false", "b=null"}
    assert {point.overrides[2] for point in points} == {"c=3", "c=2.5", "c=1e-05"}
    assert len(points) == 27


def test_a_single_path_clause_and_a_multi_path_clause_each_exclude() -> None:
    width = Axis(path="w", values=(1, 2))
    depth = Axis(path="d", values=(1, 2))
    both = Sweep(grid=(width, depth), exclude=({"w": 1, "d": 2},))
    single = Sweep(grid=(width, depth), exclude=({"w": 1},))
    expander = GridSweepExpander(compose)

    assert [p.overrides for p in expander.expand(both)] == [
        ("w=1", "d=1"),
        ("w=2", "d=1"),
        ("w=2", "d=2"),
    ]
    assert [p.overrides for p in expander.expand(single)] == [
        ("w=2", "d=1"),
        ("w=2", "d=2"),
    ]


def test_a_random_design_draws_a_reproducible_subset_without_replacement() -> None:
    sweep = Sweep(
        design=RandomDesignSpec(points=3, seed=17),
        grid=(Axis(path="a", values=(0, 1, 2)), Axis(path="b", values=(0, 1))),
    )
    expander = GridSweepExpander(compose)

    first = expander.expand(sweep)
    again = expander.expand(sweep)

    assert first == again
    assert len(first) == len({point.overrides for point in first}) == 3


def test_a_latin_hypercube_stratifies_every_equal_sized_axis() -> None:
    sweep = Sweep(
        design=LatinHypercubeDesignSpec(points=4, seed=19),
        grid=(
            Axis(path="a", values=(0, 1, 2, 3)),
            Axis(path="b", values=(10, 11, 12, 13)),
        ),
    )

    points = GridSweepExpander(compose).expand(sweep)

    assert {point.overrides[0] for point in points} == {"a=0", "a=1", "a=2", "a=3"}
    assert {point.overrides[1] for point in points} == {
        "b=10",
        "b=11",
        "b=12",
        "b=13",
    }


def test_a_resolution_iv_six_factor_fraction_has_sixteen_points() -> None:
    sweep = Sweep(
        design=FractionalFactorialDesignSpec(
            generators=(
                GeneratorSpec(factor="e", word=("a", "b", "c")),
                GeneratorSpec(factor="f", word=("b", "c", "d")),
            )
        ),
        grid=tuple(Axis(path=name, values=(0, 1)) for name in "abcdef"),
    )

    points = GridSweepExpander(compose).expand(sweep)

    assert len(points) == 16
    for point in points:
        assigned = {
            expression.partition("=")[0]: int(expression.partition("=")[2])
            for expression in point.overrides
        }
        assert assigned["e"] == assigned["a"] ^ assigned["b"] ^ assigned["c"]
        assert assigned["f"] == assigned["b"] ^ assigned["c"] ^ assigned["d"]


def test_a_clause_matches_only_when_every_pair_agrees() -> None:
    sweep = Sweep(
        grid=(Axis(path="w", values=(1, 2)), Axis(path="d", values=(1, 2))),
        exclude=({"w": 1, "d": 1}, {"w": 2, "d": 2}),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("w=1", "d=2"),
        ("w=2", "d=1"),
    ]


def test_a_clause_value_must_match_in_type_as_well_as_value() -> None:
    sweep = Sweep(
        grid=(Axis(path="flag", values=(True, 1, 1.0)),),
        exclude=({"flag": 1},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("flag=true",),
        ("flag=1.0",),
    ]


def test_a_list_valued_point_renders_as_a_json_array_override() -> None:
    sweep = Sweep(grid=(Axis(path="a.b", values=(["S", "NP"], ["S"], "S")),))
    points = GridSweepExpander(compose).expand(sweep)

    assert [point.overrides for point in points] == [
        ('a.b=["S", "NP"]',),
        ('a.b=["S"]',),
        ("a.b=S",),
    ]


def test_a_list_valued_point_reaches_the_composer_as_a_list() -> None:
    sweep = Sweep(grid=(Axis(path="a.b", values=(["S", "NP"], ["S"], "S")),))
    points = GridSweepExpander(compose).expand(sweep)
    resolved = [leaf(point.config, "a.b") for point in points]

    assert resolved == [["S", "NP"], ["S"], "S"]
    assert [type(value) for value in resolved] == [list, list, str]


def test_yoked_list_points_reach_the_composer_as_lists() -> None:
    sweep = Sweep(
        yoked=(
            (
                Axis(path="a.b", values=(["S", "NP"], ["S"])),
                Axis(path="c.d", values=(1, 2)),
            ),
        ),
        exclude=({"a.b": ["S"], "c.d": 2},),
    )
    points = GridSweepExpander(compose).expand(sweep)
    assigned = leaf(points[0].config, "a.b")

    assert [point.overrides for point in points] == [('a.b=["S", "NP"]', "c.d=1")]
    assert assigned == ["S", "NP"]
    assert type(assigned) is list


def test_a_clause_naming_a_list_excludes_the_point_that_lists_it() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=(["S", "NP"], ["S"])),),
        exclude=({"a.b": ["S"]},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ('a.b=["S", "NP"]',),
    ]


def test_a_scalar_clause_does_not_exclude_the_one_element_list_of_it() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=(["S"], "S")),),
        exclude=({"a.b": "S"},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ('a.b=["S"]',),
    ]


def test_a_list_clause_discriminates_the_type_of_the_elements_it_names() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=([1], [1.0], [True])),),
        exclude=({"a.b": [1.0]},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("a.b=[1]",),
        ("a.b=[true]",),
    ]


def test_a_list_clause_discriminates_the_type_below_the_first_level() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=([[1]], [[1.0]])),),
        exclude=({"a.b": [[1.0]]},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("a.b=[[1]]",),
    ]


def test_a_list_clause_excludes_only_the_list_that_agrees_at_every_position() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=(["S", "NP"], ["S", "VP"], ["NP", "NP"])),),
        exclude=({"a.b": ["S", "NP"]},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ('a.b=["S", "VP"]',),
        ('a.b=["NP", "NP"]',),
    ]


def test_a_list_clause_does_not_exclude_a_list_of_another_length() -> None:
    sweep = Sweep(
        grid=(Axis(path="a.b", values=([1], [1, 1], [1, 1, 1])),),
        exclude=({"a.b": [1, 1]},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("a.b=[1]",),
        ("a.b=[1, 1, 1]",),
    ]


SWEEP_DOCUMENT: Final = """
grid:
  - path: model.primitive_types
    values: [[e, t], [e, v, t]]
  - path: trainer.epochs
    values: [1, 2]
exclude:
  - model.primitive_types: [e, t]
    trainer.epochs: 2
expected_points: 3
"""


def test_a_loaded_sweep_document_keeps_its_list_points_as_lists(
    tmp_path: Path,
) -> None:
    document = tmp_path / "sweep.yaml"
    document.write_text(SWEEP_DOCUMENT, encoding="utf-8", newline="\n")
    sweep = SpecLoader().compose_traced(document, schema=Sweep).spec
    types, epochs = sweep.grid

    assert types.points == (["e", "t"], ["e", "v", "t"])
    assert epochs.points == (1, 2)
    points = GridSweepExpander(compose).expand(sweep)

    assert sweep.clauses == (
        {"model.primitive_types": ["e", "t"], "trainer.epochs": 2},
    )
    assert [point.overrides for point in points] == [
        ('model.primitive_types=["e", "t"]', "trainer.epochs=1"),
        ('model.primitive_types=["e", "v", "t"]', "trainer.epochs=1"),
        ('model.primitive_types=["e", "v", "t"]', "trainer.epochs=2"),
    ]
    assert [leaf(point.config, "model.primitive_types") for point in points] == [
        ["e", "t"],
        ["e", "v", "t"],
        ["e", "v", "t"],
    ]


def test_a_clause_can_exclude_across_grid_and_yoked_paths() -> None:
    sweep = Sweep(
        grid=(Axis(path="g", values=(1, 2)),),
        yoked=((Axis(path="y", values=(1, 2)), Axis(path="z", values=(3, 4))),),
        exclude=({"g": 2, "z": 4},),
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("g=1", "y=1", "z=3"),
        ("g=1", "y=2", "z=4"),
        ("g=2", "y=1", "z=3"),
    ]


def test_expected_points_accepts_the_computed_count_and_rejects_another() -> None:
    accepted = fixture_sweep().with_(expected_points=len(HAND_ENUMERATED))
    rejected = fixture_sweep().with_(expected_points=len(HAND_ENUMERATED) + 1)
    expander = GridSweepExpander(compose)

    assert len(expander.expand(accepted)) == len(HAND_ENUMERATED)

    with pytest.raises(UnexpectedPointCountError) as caught:
        expander.expand(rejected)

    assert caught.value.expected == len(HAND_ENUMERATED) + 1
    assert caught.value.actual == len(HAND_ENUMERATED)
    assert str(len(HAND_ENUMERATED)) in str(caught.value)


def test_expected_points_counts_unique_points() -> None:
    def ignoring_v(overrides: tuple[str, ...], seed: int) -> dict[str, JsonValue]:
        return compose(tuple(o for o in overrides if not o.startswith("v=")), seed)

    sweep = Sweep(
        grid=(Axis(path="w", values=(1, 2)), Axis(path="v", values=(1, 2))),
        expected_points=2,
    )

    assert [p.overrides for p in GridSweepExpander(ignoring_v).expand(sweep)] == [
        ("w=1", "v=1"),
        ("w=2", "v=1"),
    ]


def test_a_sweep_with_no_axis_yields_one_point_per_seed() -> None:
    points = GridSweepExpander(compose).expand(Sweep(seeds=(4, 5)))

    assert [(p.overrides, p.seed, p.config) for p in points] == [
        ((), 4, {"run": {"name": "base"}}),
        ((), 5, {"run": {"name": "base"}}),
    ]


def test_post_composition_augmentation_precedes_deduplication_and_point_count() -> None:
    def augment(point: SweepPoint) -> tuple[SweepPoint, ...]:
        return tuple(
            SweepPoint(
                point.config | {"twin": position},
                overrides=point.overrides,
                provenance=point.provenance,
                seed=point.seed,
            )
            for position in range(4)
        )

    points = GridSweepExpander(compose, augment=augment).expand(
        Sweep(expected_points=4)
    )

    assert [point.config["twin"] for point in points] == [0, 1, 2, 3]


class _OneShotAxis:
    def __init__(self, key: str, points: Sequence[int]) -> None:
        self.key = key
        self.selects_group = False
        self.points: Iterator[int] = iter(points)


class _OneShotSweep:
    def __init__(self) -> None:
        self.design = GridDesignSpec()
        self.grid = iter((_OneShotAxis("g", (1, 2)),))
        self.yoked = iter(((_OneShotAxis("y", (1, 2)), _OneShotAxis("z", (3, 4))),))
        self.clauses = iter(({"g": 2, "z": 4},))
        self.seeds = iter((7, 8))
        self.expected_points = 6


def test_expansion_consumes_each_input_sequence_once() -> None:
    points = GridSweepExpander(compose).expand(cast("Sweep", _OneShotSweep()))

    assert [(p.overrides, p.seed) for p in points] == [
        (("g=1", "y=1", "z=3"), 7),
        (("g=1", "y=1", "z=3"), 8),
        (("g=1", "y=2", "z=4"), 7),
        (("g=1", "y=2", "z=4"), 8),
        (("g=2", "y=1", "z=3"), 7),
        (("g=2", "y=1", "z=3"), 8),
    ]


def test_expansion_mutates_neither_the_sweep_nor_the_resolved_mappings() -> None:
    resolved: list[dict[str, JsonValue]] = []

    def recording(overrides: tuple[str, ...], seed: int) -> dict[str, JsonValue]:
        resolved.append(compose(overrides, seed))

        return resolved[-1]

    sweep = fixture_sweep()
    before = copy.deepcopy(sweep.model_dump())
    points = GridSweepExpander(recording).expand(sweep)
    snapshots = [copy.deepcopy(mapping) for mapping in resolved]

    for point in points:
        point.config["run"] = "changed"

    assert sweep.model_dump() == before
    assert sweep == fixture_sweep()
    assert resolved == snapshots
    assert all(point.config["run"] == {"name": "base"} for point in points)


def test_points_differing_only_in_seed_are_distinct() -> None:
    config: dict[str, JsonValue] = {"model": {"width": 64}}
    first = SweepPoint(config, seed=1)
    second = SweepPoint(config, seed=2)
    again = SweepPoint({"model": {"width": 64}}, seed=1)

    assert first != second
    assert first.point_id != second.point_id
    assert hash(first) != hash(second)
    assert first == again
    assert hash(first) == hash(again)
    assert unique_points((first, second, again)) == (first, second)


class AxisLoss(LossSpec):
    kind: Literal["sweep-axis-loss"] = "sweep-axis-loss"


class AxisTrainer(TrainerSpec):
    kind: Literal["sweep-axis-trainer"] = "sweep-axis-trainer"


class AxisOptimizer(OptimizerSpec):
    kind: Literal["sweep-axis-optimizer"] = "sweep-axis-optimizer"


PACKAGED_BASE: Final[dict[str, JsonValue]] = {
    "model": {"language": {"from": "data"}},
    "loss": {"terms": [{"objective": {"kind": "sweep-axis-loss"}}]},
    "trainer": {"kind": "sweep-axis-trainer"},
    "optimizer": {"kind": "sweep-axis-optimizer"},
    "parse": {"retain": "forest", "reachability": "all"},
    "release": {"checkpoint": "resume-latest"},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
}

PACKAGED_GROUPS: Final = MappingProxyType(
    {
        "model.parser": "inside-outside",
        "data.datasets.train.dataset": "synthetic",
        "data.datasets.train.dataset.grammar": "directed",
        "data.datasets.train.dataset.grammar.signature": "narrow",
        "data.datasets.train.dataset.grammar.rules": "application",
        "data.datasets.train.dataset.grammar.lexicon": "unambiguous",
        "data.datasets.train.featurizer": "span-constraint",
    }
)


TERM_ENCODER_VOCABULARY: Final = TokenVocabulary(
    ("token",),
    reserved=ReservedTokens({"<unk>": 0}),
    unknown="<unk>",
)

ARCHITECTURES: Final = ("hrr", "transformer")
WIDTHS: Final = (64, 128)
GRAMMARS: Final = (
    ("narrow", "application", "unambiguous"),
    ("wide", "composition", "ambiguous"),
)


@pytest.fixture(autouse=True)
def _declarations() -> None:
    """Make every packaged component declaration available to composition."""
    load_plugins()


def composed_run(overrides: tuple[str, ...], _seed: int) -> RunSpec:
    """Compose a run specification from the packaged groups and a point's overrides."""
    return compose_settings(
        None,
        schema=RunSpec,
        base=PACKAGED_BASE,
        groups=PACKAGED_GROUPS,
        overrides=overrides,
        search_path=(PACKAGED_CONFIG_ROOT,),
    )


def compose_packaged(overrides: tuple[str, ...], seed: int) -> dict[str, JsonValue]:
    return resolved_document(composed_run(overrides, seed))


def grammar_sweep() -> Sweep:
    return Sweep(
        grid=(
            Axis(group="model.encoders.terms", values=ARCHITECTURES),
            Axis(path="model.encoders.terms.width", values=WIDTHS),
        ),
        yoked=(
            (
                Axis(
                    group="data.datasets.train.dataset.grammar.signature",
                    values=tuple(signature for signature, _, _ in GRAMMARS),
                ),
                Axis(
                    group="data.datasets.train.dataset.grammar.rules",
                    values=tuple(rules for _, rules, _ in GRAMMARS),
                ),
                Axis(
                    group="data.datasets.train.dataset.grammar.lexicon",
                    values=tuple(lexicon for _, _, lexicon in GRAMMARS),
                ),
            ),
        ),
        seeds=(0, 1),
        expected_points=16,
    )


def corpus(spec: RunSpec) -> SyntheticCorpusDecl:
    dataset = spec.data.datasets["train"].dataset

    if not isinstance(dataset, SyntheticCorpusDecl):
        msg = "the composed dataset is not a synthetic corpus"
        raise TypeError(msg)

    return dataset


def grammar(spec: RunSpec) -> GrammarDecl:
    """Return the current formalism declaration lowered to the common grammar."""
    return declare_grammar(corpus(spec).grammar)


def term_encoder(spec: RunSpec) -> MeasuredRole:
    """Build the named term encoder over its dataset learner language."""
    encoder = spec.model.encoders["terms"]
    dataset = corpus(spec)
    assembly = GrammarIso().forward(declare_grammar(dataset.grammar), dataset.language)
    view = NeuralCodecView(assembly.language.codec.at(assembly.language.category_sort))
    built = build_component(
        cast("ConfiguredSpec[object]", encoder),
        plugin_context(
            codec=view,
            component="terms",
            vocabulary=TERM_ENCODER_VOCABULARY,
            policy=resolve_device_policy(spec),
        ),
    )

    return MeasuredRole(
        cast("str", encoder.kind), tuple(cast("nn.Module", built).parameters())
    )


def declared(point: SweepPoint) -> tuple[str, int]:
    """Return the named term encoder kind and width a point declares."""
    model = cast("dict[str, JsonValue]", point.config["model"])
    encoders = cast("dict[str, JsonValue]", model["encoders"])
    held = cast("dict[str, JsonValue]", encoders["terms"])

    return cast("str", held["kind"]), cast("int", held["width"])


def measured(point: SweepPoint) -> PointManifest:
    spec = RunSpec.model_validate(point.config)

    return measure_point(
        point,
        {"terms": term_encoder(spec)},
        corpus_fingerprint=corpus_fingerprint(corpus(spec)),
    )


def test_architecture_width_and_grammar_cross_into_measured_points() -> None:
    points = GridSweepExpander(compose_packaged).expand(grammar_sweep())
    manifests = tuple(measured(point) for point in points)
    sizes = {
        (*declared(point), manifest.corpus_fingerprint): manifest.parameter_count
        for point, manifest in zip(points, manifests, strict=True)
    }

    assert len(points) == 16
    assert len({point.point_id for point in points}) == 16
    assert {manifest.seed for manifest in manifests} == {0, 1}
    assert {width for _, width, _ in sizes} == set(WIDTHS)
    assert {manifest.roles["terms"].kind for manifest in manifests} == {
        architecture for architecture, _, _ in sizes
    }
    assert len({architecture for architecture, _, _ in sizes}) == len(ARCHITECTURES)
    assert len({manifest.corpus_fingerprint for manifest in manifests}) == len(GRAMMARS)
    assert len(sizes) == 8
    assert all(count > 0 for count in sizes.values())

    for architecture, _, fingerprint in sizes:
        narrow = sizes[architecture, 64, fingerprint]

        assert narrow < sizes[architecture, 128, fingerprint]

    for width, fingerprint in {(key[1], key[2]) for key in sizes}:
        assert len({sizes[a, width, fingerprint] for a, _, _ in sizes}) == len(
            ARCHITECTURES
        )


def test_two_points_on_different_grammars_have_distinct_identifiers() -> None:
    shared = (
        "model.encoders/terms=hrr",
        "model.encoders.terms.width=64",
    )
    ids = {
        SweepPoint(
            compose_packaged(
                (
                    *shared,
                    f"data/datasets/train/dataset/grammar/signature={signature}",
                    f"data/datasets/train/dataset/grammar/rules={rules}",
                    f"data/datasets/train/dataset/grammar/lexicon={lexicon}",
                ),
                0,
            ),
            seed=0,
        ).point_id
        for signature, rules, lexicon in GRAMMARS
    }

    assert len(ids) == len(GRAMMARS)


def test_a_grammar_the_config_groups_do_not_hold_is_refused_with_the_names() -> None:
    sweep = Sweep(
        grid=(
            Axis(
                group="data.datasets.train.dataset.grammar.signature",
                values=("narrow", "absent"),
            ),
        )
    )

    with pytest.raises(MissingFragmentError) as refusal:
        GridSweepExpander(compose_packaged).expand(sweep)

    assert "['narrow', 'wide']" in str(refusal.value)


PRIMITIVES: Final[tuple[list[JsonValue], ...]] = (
    ["S", "NP", "N", "conj"],
    ["S", "NP", "N", "PP", "ADJ", "conj"],
)
WITHHELD: Final[list[JsonValue]] = ["S", "NP", "N", "PP", "conj"]
SIGNATURE_PRIMITIVES: Final = "data.datasets.train.dataset.grammar.signature.primitives"
GOAL_CATEGORIES: Final = "data.datasets.train.dataset.grammar.goal.categories"
UNARY_ROUNDS: Final = "data.datasets.train.dataset.grammar.budget.unary_rounds"
SURVIVORS: Final[tuple[tuple[list[JsonValue], int], ...]] = (
    (PRIMITIVES[0], 1),
    (PRIMITIVES[1], 1),
    (PRIMITIVES[1], 2),
)


def primitive_sweep() -> Sweep:
    """Return a sweep crossing list-valued inventories with a scalar budget.

    One clause names the longer budget on the smaller inventory, so the
    exclusion turns on a list value and the survivors are the three the
    sweep declares.
    """
    return Sweep(
        grid=(
            Axis(path=SIGNATURE_PRIMITIVES, values=PRIMITIVES),
            Axis(path=GOAL_CATEGORIES, values=(["S"],)),
            Axis(path=UNARY_ROUNDS, values=(1, 2)),
        ),
        exclude=({SIGNATURE_PRIMITIVES: PRIMITIVES[0], UNARY_ROUNDS: 2},),
        seeds=(0,),
        expected_points=len(SURVIVORS),
    )


def test_list_and_scalar_points_resolve_into_a_validating_specification() -> None:
    sweep = primitive_sweep()
    inventories, goals, rounds_axis = sweep.grid
    points = GridSweepExpander(compose_packaged).expand(sweep)

    assert inventories.points == PRIMITIVES
    assert goals.points == (["S"],)
    assert rounds_axis.points == (1, 2)
    assert len(points) == len(SURVIVORS)

    for point, (primitives, rounds) in zip(points, SURVIVORS, strict=True):
        resolved = leaf(point.config, SIGNATURE_PRIMITIVES)
        categories = leaf(point.config, GOAL_CATEGORIES)
        spec = RunSpec.model_validate(point.config)
        declared = grammar(spec)

        assert resolved == primitives
        assert type(resolved) is list
        assert categories == ["S"]
        assert type(categories) is list
        assert leaf(point.config, UNARY_ROUNDS) == rounds
        assert declared.signature.primitives == tuple(primitives)
        assert declared.goal.categories == ("S",)
        assert declared.budget.unary_rounds == rounds
        assert resolved_document(spec) == point.config


def test_the_two_primitive_inventories_build_distinguishable_corpora() -> None:
    sweep = Sweep(
        grid=(
            Axis(
                path=SIGNATURE_PRIMITIVES,
                values=(PRIMITIVES[0], WITHHELD, PRIMITIVES[1]),
            ),
        ),
        exclude=({SIGNATURE_PRIMITIVES: WITHHELD},),
        seeds=(0,),
        expected_points=len(PRIMITIVES),
    )
    points = GridSweepExpander(compose_packaged).expand(sweep)
    corpora = [corpus(RunSpec.model_validate(point.config)) for point in points]
    grammars = [declare_grammar(built.grammar) for built in corpora]
    fingerprints = [corpus_fingerprint(built) for built in corpora]

    assert len(points) == len(PRIMITIVES)
    assert [built.signature.primitives for built in grammars] == [
        tuple(inventory) for inventory in PRIMITIVES
    ]
    assert tuple(WITHHELD) not in [built.signature.primitives for built in grammars]
    assert len(set(fingerprints)) == len(PRIMITIVES)


def sources_of(
    table: Mapping[str, tuple[LeafOrigin, ...]],
) -> Callable[[tuple[str, ...], int], Mapping[str, tuple[LeafOrigin, ...]]]:
    """Return a source resolver that reports one table for every point."""
    return lambda _overrides, _seed: table


def test_a_group_axis_renders_a_selection_and_a_root_slot_with_a_leading_slash() -> (
    None
):
    sweep = Sweep(
        grid=(
            Axis(group="device_policy", values=("cpu", None)),
            Axis(group="model.encoders.terms", values=("rnn",)),
        )
    )

    assert [p.overrides for p in GridSweepExpander(compose).expand(sweep)] == [
        ("/device_policy=cpu", "model/encoders/terms=rnn"),
        ("/device_policy=null", "model/encoders/terms=rnn"),
    ]


def test_a_mapping_point_is_rendered_as_the_object_it_is_and_can_be_excluded() -> None:
    kl: JsonValue = {"terms": [{"objective": {"kind": "kl"}, "weight": 0.1}]}
    ce: JsonValue = {"terms": [{"objective": {"kind": "ce"}, "weight": 0.1}]}
    sweep = Sweep(
        grid=(Axis(path="loss", values=(kl, ce)), Axis(path="w", values=(1, 2))),
        exclude=({"loss": {"terms": [{"weight": 0.1, "objective": {"kind": "ce"}}]}},),
    )
    points = GridSweepExpander(compose).expand(sweep)

    assert [leaf(p.config, "loss") for p in points] == [kl, kl]
    assert [p.overrides[0] for p in points] == [f"loss={json.dumps(kl)}"] * 2


def test_a_mapping_clause_must_agree_in_type_at_every_depth() -> None:
    sweep = Sweep(
        grid=(Axis(path="loss", values=({"weight": 1}, {"weight": 1.0})),),
        exclude=({"loss": {"weight": 1.0}},),
    )

    assert [
        leaf(p.config, "loss") for p in GridSweepExpander(compose).expand(sweep)
    ] == [{"weight": 1}]


def test_a_mapping_clause_must_name_the_same_keys_as_the_point() -> None:
    sweep = Sweep(
        grid=(Axis(path="loss", values=({"weight": 1}, {"weight": 2})),),
        exclude=({"loss": {"weight": 1, "scale": 2}},),
    )
    points = GridSweepExpander(compose).expand(sweep)

    assert [leaf(p.config, "loss") for p in points] == [{"weight": 1}, {"weight": 2}]


@pytest.mark.parametrize(
    ("source", "layer"),
    [
        (ConfigSource.FILE, "run.yaml"),
        (ConfigSource.PROFILE, "cluster"),
        (ConfigSource.EXTRA, ""),
        (
            ConfigSource.OVERRIDE,
            "data.datasets.train.dataset.corruption.proportion.value=0.25",
        ),
    ],
)
def test_a_declaring_source_under_a_group_axis_is_refused_naming_both(
    source: ConfigSource, layer: str
) -> None:
    table = {
        "data.datasets.train.dataset.corruption.proportion.value": (
            LeafOrigin(source=ConfigSource.GROUP, layer="quarter.yaml"),
            LeafOrigin(source=source, layer=layer),
        )
    }
    sweep = Sweep(
        grid=(
            Axis(
                group="data.datasets.train.dataset.corruption.proportion",
                values=("quarter",),
            ),
        )
    )

    with pytest.raises(ShadowedAxisError) as refused:
        GridSweepExpander(compose, sources=sources_of(table)).expand(sweep)

    assert refused.value.axis == "data.datasets.train.dataset.corruption.proportion"
    assert (
        refused.value.path == "data.datasets.train.dataset.corruption.proportion.value"
    )
    assert refused.value.source is source
    assert refused.value.layer == layer
    assert "data.datasets.train.dataset.corruption.proportion.value" in str(
        refused.value
    )


def test_a_leaf_axis_varies_the_document_value_it_is_applied_over() -> None:
    table = {
        "trainer.epochs": (
            LeafOrigin(source=ConfigSource.FILE, layer="run.yaml"),
            LeafOrigin(source=ConfigSource.OVERRIDE, layer="trainer.epochs=1"),
        )
    }
    sweep = Sweep(grid=(Axis(path="trainer.epochs", values=(1,)),))

    points = GridSweepExpander(compose, sources=sources_of(table)).expand(sweep)

    assert [p.overrides for p in points] == [("trainer.epochs=1",)]


@pytest.mark.parametrize(
    ("path", "source", "layer"),
    [
        ("trainer.epochs", ConfigSource.PROFILE, "cluster"),
        ("trainer.epochs", ConfigSource.EXTRA, ""),
        ("trainer.epochs", ConfigSource.OVERRIDE, "trainer.epochs=7"),
        ("trainer", ConfigSource.OVERRIDE, "trainer={}"),
    ],
)
def test_a_leaf_axis_refuses_a_source_it_would_silently_discard(
    path: str, source: ConfigSource, layer: str
) -> None:
    table = {path: (LeafOrigin(source=source, layer=layer),)}
    sweep = Sweep(grid=(Axis(path="trainer.epochs", values=(1, 2)),))

    with pytest.raises(ShadowedAxisError) as refused:
        GridSweepExpander(compose, sources=sources_of(table)).expand(sweep)

    assert (refused.value.path, refused.value.source) == (path, source)


def test_a_document_value_surviving_under_a_mapping_point_is_refused() -> None:
    point: JsonValue = {"terms": [{"weight": 1.0}]}
    own = f"loss={json.dumps(point)}"
    overruled = {
        "loss.terms": (
            LeafOrigin(source=ConfigSource.FILE, layer="run.yaml"),
            LeafOrigin(source=ConfigSource.OVERRIDE, layer=own),
        )
    }
    surviving = {
        **overruled,
        "loss.scale": (LeafOrigin(source=ConfigSource.FILE, layer="run.yaml"),),
    }
    sweep = Sweep(grid=(Axis(path="loss", values=(point,)),))

    assert len(GridSweepExpander(compose, sources=sources_of(overruled)).expand(sweep))

    with pytest.raises(ShadowedAxisError, match=r"loss\.scale"):
        GridSweepExpander(compose, sources=sources_of(surviving)).expand(sweep)


def test_fragments_defaults_siblings_and_the_points_own_overrides_are_not_shadows() -> (
    None
):
    table = {
        "model.encoders.terms.width": (
            LeafOrigin(source=ConfigSource.DEFAULT),
            LeafOrigin(source=ConfigSource.GROUP, layer="rnn.yaml"),
            LeafOrigin(
                source=ConfigSource.OVERRIDE,
                layer="model.encoders.terms.width=8",
            ),
        ),
        "model.decoders": (LeafOrigin(source=ConfigSource.FILE, layer="run"),),
        "model.width": (LeafOrigin(source=ConfigSource.FILE, layer="run.yaml"),),
    }
    sweep = Sweep(
        grid=(
            Axis(group="model.encoders.terms", values=("rnn",)),
            Axis(path="model.encoders.terms.width", values=(8,)),
        )
    )

    points = GridSweepExpander(compose, sources=sources_of(table)).expand(sweep)

    assert [p.overrides for p in points] == [
        ("model/encoders/terms=rnn", "model.encoders.terms.width=8")
    ]


SYNTHETIC_DEFAULTS: Final[list[JsonValue]] = [
    "run/synthetic-parser",
    {"model/controller": "mlp"},
    {"data.datasets.train.dataset": "synthetic"},
]


def fragment(slot: str, name: str) -> dict[str, JsonValue]:
    """Return the packaged fragment a group selection names."""
    directory = PACKAGED_CONFIG_ROOT.joinpath(*slot.split("."))

    return cast(
        "dict[str, JsonValue]",
        yaml.safe_load((directory / f"{name}.yaml").read_text(encoding="utf-8")),
    )


def test_a_group_axis_the_run_document_shadows_is_refused_naming_the_leaf(
    tmp_path: Path,
) -> None:
    run: dict[str, JsonValue] = {
        "defaults": [
            *SYNTHETIC_DEFAULTS,
            {"data.datasets.train.dataset.corruption.operators": "swap-deletion"},
            {"trainer": "parser"},
        ],
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "size": 20_480,
                        "corruption": {"proportion": {"value": 0.25}},
                    }
                }
            }
        },
    }
    sweep: dict[str, JsonValue] = {
        "grid": [
            {
                "group": "data.datasets.train.dataset.corruption.proportion",
                "values": ["quarter", "half"],
            }
        ],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    experiment = read_experiment(write_priced_experiment(tmp_path, run, sweep))

    with pytest.raises(ShadowedAxisError) as refused:
        expand_experiment(experiment, tmp_path)

    assert (
        refused.value.path == "data.datasets.train.dataset.corruption.proportion.value"
    )
    assert refused.value.source is ConfigSource.FILE
    assert refused.value.layer.endswith("run.json")
    assert "data.datasets.train.dataset.corruption.proportion.value" in str(
        refused.value
    )

    run["data"] = {"datasets": {"train": {"dataset": {"size": 20_480}}}}
    points = expand_experiment(
        read_experiment(write_priced_experiment(tmp_path, run, sweep)), tmp_path
    )

    assert [
        leaf(p.config, "data.datasets.train.dataset.corruption.proportion.value")
        for p in points[::3]
    ] == [
        fragment("data.datasets.train.dataset.corruption.proportion", name)["value"]
        for name in ("quarter", "half")
    ]


@pytest.mark.parametrize(
    ("slot", "names"),
    [
        ("device_policy", ("cpu", "cuda-mixed")),
        ("device_policy", ("cpu", "mps")),
    ],
)
def test_a_top_level_group_axis_expands_to_the_values_of_its_fragments(
    tmp_path: Path, slot: str, names: tuple[str, str]
) -> None:
    sweep: dict[str, JsonValue] = {
        "grid": [{"group": slot, "values": list(names)}],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    run: dict[str, JsonValue] = {
        "defaults": SYNTHETIC_DEFAULTS,
        "data": {"datasets": {"train": {"dataset": {"size": 20_480}}}},
    }
    path = write_priced_experiment(tmp_path, run, sweep)
    points = expand_experiment(read_experiment(path), tmp_path)

    assert len({p.point_id for p in points}) == 6

    for point, name in zip(points[::3], names, strict=True):
        expected = fragment(slot, name)

        for key, value in expected.items():
            assert leaf(point.config, f"{slot}.{key}") == value


def test_an_experiment_yoking_a_repeated_tuple_is_refused(tmp_path: Path) -> None:
    sweep: dict[str, JsonValue] = {
        "yoked": [
            [
                {"path": "model.encoders.terms.width", "values": [4, 4, 8]},
                {
                    "path": "model.decoders.application.width",
                    "values": [4, 4, 8],
                },
            ]
        ],
        "seeds": [0, 1, 2],
        "expected_points": 6,
    }
    path = write_priced_experiment(tmp_path, {"defaults": SYNTHETIC_DEFAULTS}, sweep)

    with pytest.raises(ExperimentError, match="repeats a tuple"):
        read_experiment(path)


def test_an_experiment_that_declares_no_point_count_is_refused(tmp_path: Path) -> None:
    sweep: dict[str, JsonValue] = {
        "grid": [{"path": "model.encoders.terms.width", "values": [4, 8]}],
        "seeds": [0, 1, 2],
    }
    path = write_priced_experiment(tmp_path, {"defaults": SYNTHETIC_DEFAULTS}, sweep)

    with pytest.raises(ExperimentError, match="expected_points"):
        read_experiment(path)


def shipped_point_ids(document: str) -> bytes:
    """Return a shipped experiment's point ids and any point written as JSON text.

    The ids are computed by expansion and by direct composition. The direct
    route composes each point's overrides with the run codec, outside the
    expander, so the two agree only when every point is the configuration its
    overrides name.
    """
    experiment = read_experiment(TREE / document)
    points = expand_experiment(experiment, TREE)
    codec = RunSpecCodec(search_path=config_search_path(TREE))
    composed = {
        overrides: resolved_document(codec(experiment.config.path, overrides=overrides))
        for overrides in dict.fromkeys(point.overrides for point in points)
    }

    sweep = experiment.sweep
    axes = (*sweep.grid, *(axis for group in sweep.yoked for axis in group))

    return json.dumps(
        {
            "encoded": [
                point
                for axis in axes
                for point in axis.points
                if isinstance(point, str) and point.lstrip().startswith(("{", "["))
            ],
            "declared": experiment.sweep.expected_points,
            "expanded": [p.point_id for p in points],
            "composed": [point_id(composed[p.overrides], seed=p.seed) for p in points],
        }
    ).encode()


SHIPPED_SCRIPT: Final = """
import sys

from tests.unit.run.test__sweep import shipped_point_ids

sys.stdout.buffer.write(shipped_point_ids(sys.argv[1]))
"""


@pytest.mark.parametrize("document", ["docs/examples/experiments/synthetic.yaml"])
def test_a_shipped_experiment_without_json_strings_expands_to_its_point_ids(
    document: str,
) -> None:
    finished = subprocess.run(
        [sys.executable, "-", document],
        input=SHIPPED_SCRIPT.encode(),
        capture_output=True,
        check=True,
        cwd=TREE,
    )
    ids = json.loads(finished.stdout)

    assert ids["encoded"] == []
    assert len(ids["expanded"]) == ids["declared"]
    assert ids["expanded"] == ids["composed"]


@settings(max_examples=SMALL.examples, deadline=None)
@given(candidate_sets(SMALL, category_sort="Cat"))
def test_a_sweep_over_generated_candidate_grammars_trains_each_candidate(
    drawn: Candidates,
) -> None:
    language = category_language(drawn.source, registered_language("propositional"))
    language_document = cast("dict[str, JsonValue]", loads(language.model_dump_json()))
    logic = cast("dict[str, JsonValue]", language_document["logic"])
    logic["prover"] = cast(
        "dict[str, JsonValue]",
        loads(
            json.dumps(language.logic.prover.model_dump(exclude={"declared_output"}))
        ),
    )
    section = DataSection(
        DataKind.STRINGS,
        {
            "datasets": {
                "train": {
                    "dataset": {
                        "kind": "synthetic",
                        "retained": {"validation": 0.25},
                        "grammar": loads(drawn.source.model_dump_json()),
                        "language": language_document,
                        "size": SMALL.size,
                        "seed": 1,
                        "length": {
                            "kind": "uniform",
                            "min_length": 2,
                            "max_length": 3,
                        },
                    },
                    "featurizer": {"kind": "unlabeled"},
                }
            }
        },
        None,
    )
    run = run_document(section)
    model = cast("dict[str, JsonValue]", run["model"])
    model["parser"] = {"kind": "tagged", "width": 4, "unary_depth": 1}
    spaces = cast("dict[str, dict[str, JsonValue]]", model["spaces"])
    combinators = cast("dict[str, JsonValue]", spaces["categories"]["combinators"])
    combinators["decoders"] = {}
    sweep = Sweep(
        yoked=(
            (
                Axis(
                    path="data.datasets.train.dataset",
                    values=tuple(
                        {
                            **section.dataset,
                            "grammar": {
                                **cast(
                                    "dict[str, JsonValue]", section.dataset["grammar"]
                                ),
                                "rules": loads(grammar.rules.model_dump_json()),
                            },
                        }
                        for grammar in drawn.candidates
                    ),
                ),
                Axis(
                    path="model.spaces.categories.combinators.decoders",
                    values=tuple(
                        dict.fromkeys(
                            (
                                rule.name
                                for rule in GrammarIso()
                                .forward(
                                    grammar,
                                    category_language(
                                        grammar,
                                        registered_language("propositional"),
                                    ),
                                )
                                .grammar.rules
                                if rule.arity == 2
                            ),
                            "application",
                        )
                        for grammar in drawn.candidates
                    ),
                ),
            ),
        )
    )
    codec = RunSpecCodec()

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / "base.yaml"
        base.write_text(
            yaml.safe_dump(run, sort_keys=False), encoding="utf-8", newline="\n"
        )
        points = GridSweepExpander(
            lambda overrides, _seed: resolved_document(codec(base, overrides=overrides))
        ).expand(sweep)

    assert len(points) == len(drawn.candidates)

    for point, candidate in zip(points, drawn.candidates, strict=True):
        spec = resolve_point(point, codec)

        dataset = cast("SyntheticCorpusDecl", spec.data.datasets["train"].dataset)

        assert loads(dataset.grammar.model_dump_json()) == loads(
            candidate.model_dump_json()
        )

        try:
            metrics = train_spec(spec)
        except ArityUnsupportedError:
            if not exceeds_shipped_ranks(candidate):
                raise
            continue

        assert metrics["optimizer/loss_finite"] == 1.0
