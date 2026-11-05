import json
from itertools import combinations
from pathlib import Path
from typing import Final, Literal, cast

import didactic.api as dx
import pytest
import torch
from tests.conformance.latent_predictor import (
    LatentPredictorSubject,
    LatentPredictorSuite,
)

from lofi.judgment import (
    ColumnInventory,
    ColumnLevel,
    ColumnLift,
    DesignCodec,
    FitSchedule,
    JointTypeTensor,
    JudgmentDesign,
    JudgmentRow,
    LexicalPossession,
    LexicalPrimitiveInventory,
    MemorizedCells,
    OrdinalHead,
    PinnedMap,
    PinnedPrefix,
    PossessionAndClasses,
    PrimitiveTypeMap,
    RowColumns,
    TableAblationSpec,
    TableColumns,
    TableRows,
    TableRung,
    fit_rung,
    ladder_of,
    mean_log_density,
    slot_types,
    spec_digest,
)
from lofi.spec.optimizers.adam import AdamOptimizerSpec

FIXTURES: Final = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment" / "table"
)
CONTROL: Final = "trivial_memorization"
DIRECT: Final = "structured_direct"
SEED: Final = 11
STEPS: Final = 400
LEARNING_RATE: Final = 0.05
WEIGHT_DECAY: Final = 0.005
INITIAL_SCALE: Final = 0.5
SCHEDULE = FitSchedule(
    optimizer=AdamOptimizerSpec(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
    steps=STEPS,
    batch_size=None,
    initial_scale=INITIAL_SCALE,
)


def ladder_index() -> tuple[str, tuple[str, ...]]:
    index = json.loads((FIXTURES / "ladder.json").read_text(encoding="utf-8"))

    return index["control"], tuple(index["rungs"])


def rung_spec(name: str) -> TableAblationSpec:
    return TableAblationSpec.model_validate_json(
        (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
    )


def judgments() -> tuple[int, tuple[dict[str, str | int | bool], ...]]:
    fixture = json.loads((FIXTURES / "judgments.json").read_text(encoding="utf-8"))

    return fixture["levels"], tuple(fixture["rows"])


def design_of(
    levels: int, rows: tuple[dict[str, str | int | bool], ...]
) -> JudgmentDesign:
    codec = DesignCodec(levels, factors=("verb", "frame"))

    return codec.encode(
        JudgmentRow(
            str(row["participant"]),
            f"{row['verb']}|{row['frame']}",
            int(row["response"]),
            RowColumns({"verb": str(row["verb"]), "frame": str(row["frame"])}, {}),
        )
        for row in rows
    )


def split() -> tuple[int, JudgmentDesign, JudgmentDesign]:
    levels, rows = judgments()
    fitted = tuple(row for row in rows if not row["held_out"])
    held_out = tuple(row for row in rows if row["held_out"])

    return levels, design_of(levels, fitted), design_of(levels, held_out)


def one_rung(name: str, *, seed: int = SEED) -> TableRung:
    generator = torch.Generator().manual_seed(seed)
    (built,) = ladder_of((rung_spec(name),), generator=generator)

    return built


def small_inventory() -> ColumnInventory:
    return ColumnInventory(
        primitive_types=("NP", "S_that"),
        null_type="NONE",
        levels=(
            ColumnLevel(label="_ NP", arguments=("NP",)),
            ColumnLevel(label="_ NP S_that", arguments=("NP", "S_that")),
        ),
    )


def small_spec(
    lift: JointTypeTensor | MemorizedCells | LexicalPossession,
) -> TableAblationSpec:
    return TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell")),
        columns=TableColumns(attribute="frame", inventory=small_inventory()),
        width=2,
        lift=lift,
        schedule=SCHEDULE,
    )


def small_design(pairs: tuple[tuple[str, str, int], ...]) -> JudgmentDesign:
    codec = DesignCodec(5, factors=("verb", "frame"))

    return codec.encode(
        JudgmentRow(
            f"a{position}",
            f"{verb}|{frame}",
            level,
            RowColumns({"verb": verb, "frame": frame}, {}),
        )
        for position, (verb, frame, level) in enumerate(pairs)
    )


ROWS: Final = (
    ("think", "_ NP", 0),
    ("tell", "_ NP S_that", 3),
    ("think", "_ NP S_that", 2),
    ("tell", "_ NP", 4),
)


class TestEachRungObeysTheLatentPredictorLaws(LatentPredictorSuite):
    @pytest.fixture(params=list(ladder_index()[1]))
    def latent_predictor_subject(
        self, request: pytest.FixtureRequest
    ) -> LatentPredictorSubject:
        levels, rows = judgments()

        return LatentPredictorSubject(
            one_rung(str(request.param)), design_of(levels, rows[:40])
        )


def test_the_ladder_index_names_the_control_and_every_committed_rung() -> None:
    control, rungs = ladder_index()
    committed = frozenset(path.stem for path in FIXTURES.glob("*.json")) - {
        "ladder",
        "judgments",
    }

    assert control in rungs
    assert frozenset(rungs) == committed
    assert len(rungs) == len(committed)


def test_every_rung_loads_through_one_entry_point_with_its_own_fingerprint() -> None:
    _, rungs = ladder_index()
    specs = {name: rung_spec(name) for name in rungs}
    fingerprints = {spec_digest(spec) for spec in specs.values()}

    assert len(fingerprints) == len(rungs)
    assert len(set(specs.values())) == len(rungs)


def test_two_rungs_differ_in_the_lift_and_in_nothing_else() -> None:
    _, rungs = ladder_index()
    specs = [rung_spec(name) for name in rungs]

    for lower, upper in combinations(specs, 2):
        assert lower.lift != upper.lift
        assert lower.columns == upper.columns
        assert lower.rows == upper.rows
        assert lower.width == upper.width


def test_every_rung_implements_the_one_crossed_term_the_comparison_declares() -> None:
    _, rungs = ladder_index()
    declared = {one_rung(name).terms for name in rungs}

    assert len(declared) == 1
    assert len(next(iter(declared))) == 1


def test_the_decomposing_rungs_read_one_primitive_type_map() -> None:
    _, rungs = ladder_index()
    specs = [rung_spec(name) for name in rungs]
    generator = torch.Generator().manual_seed(SEED)
    built = ladder_of(specs, generator=generator)
    decomposing = [
        rung for rung, spec in zip(built, specs, strict=True) if spec.lift.decomposes
    ]
    shared = [
        parameter
        for parameter in decomposing[0].parameters()
        if all(
            any(parameter is other for other in rung.parameters())
            for rung in decomposing[1:]
        )
    ]
    inventory = specs[0].columns.inventory

    assert len(decomposing) == 4
    assert len(shared) == 1
    assert tuple(shared[0].shape) == (inventory.type_count, specs[0].width)


def test_a_rung_that_does_not_decompose_shares_no_parameter_with_one_that_does() -> (
    None
):
    _, rungs = ladder_index()
    specs = [rung_spec(name) for name in rungs]
    generator = torch.Generator().manual_seed(SEED)
    built = ladder_of(specs, generator=generator)
    flat = [
        (spec.lift.decomposes, rung) for rung, spec in zip(built, specs, strict=True)
    ]
    labeling = [rung for decomposes, rung in flat if not decomposes]
    decomposing = [rung for decomposes, rung in flat if decomposes]

    assert len(labeling) == 3

    for rung in labeling:
        for other in decomposing:
            assert not any(
                parameter is shared
                for parameter in rung.parameters()
                for shared in other.parameters()
            )


def test_a_ladder_refuses_rungs_declared_over_other_columns() -> None:
    _, rungs = ladder_index()
    first = rung_spec(rungs[0])
    narrowed = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=first.rows.levels),
        columns=TableColumns(attribute="frame", inventory=small_inventory()),
        width=first.width,
        lift=first.lift,
        schedule=first.schedule,
    )
    generator = torch.Generator().manual_seed(SEED)

    with pytest.raises(ValueError, match="rungs of one ladder"):
        ladder_of((first, narrowed), generator=generator)


def test_a_ladder_refuses_an_empty_sequence_of_rungs() -> None:
    generator = torch.Generator().manual_seed(SEED)

    with pytest.raises(ValueError, match="at least one rung"):
        ladder_of((), generator=generator)


def test_a_rung_refuses_a_map_built_over_other_columns_or_at_another_width() -> None:
    spec = small_spec(JointTypeTensor())
    generator = torch.Generator().manual_seed(SEED)
    elsewhere = PrimitiveTypeMap(
        ColumnInventory(
            primitive_types=("NP",),
            null_type="NONE",
            levels=(ColumnLevel(label="_ NP", arguments=("NP",)),),
        ),
        spec.width,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )
    wider = PrimitiveTypeMap(
        spec.columns.inventory,
        spec.width + 1,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )

    with pytest.raises(ValueError, match="built over other columns"):
        TableRung(
            spec,
            elsewhere,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match="reads a map of width"):
        TableRung(spec, wider, generator=generator, initial_scale=INITIAL_SCALE)


def test_the_padding_of_a_short_argument_sequence_takes_the_null_type() -> None:
    inventory = small_inventory()
    slots = slot_types(inventory, device=torch.device("cpu"))

    assert tuple(slots.shape) == (len(inventory.levels), inventory.arity)
    assert slots[0].tolist() == [0, inventory.null_index]
    assert slots[1].tolist() == [0, 1]
    assert inventory.null_index not in range(len(inventory.primitive_types))


def test_the_joint_lift_is_the_outer_product_of_the_slots_it_reads() -> None:
    inventory = small_inventory()
    generator = torch.Generator().manual_seed(SEED)
    mapped = PrimitiveTypeMap(
        inventory, 2, generator=generator, initial_scale=INITIAL_SCALE
    )
    slots = slot_types(inventory, device=torch.device("cpu"))
    joint = mapped.joint(slots)
    vectors = mapped.vectors

    assert tuple(joint.shape) == (2, 2**inventory.arity)
    assert torch.allclose(
        joint[0], torch.outer(vectors[0], vectors[inventory.null_index]).flatten()
    )
    assert torch.allclose(joint[1], torch.outer(vectors[0], vectors[1]).flatten())


def test_the_summed_lift_adds_one_vector_for_every_slot_including_the_empty_one() -> (
    None
):
    inventory = small_inventory()
    generator = torch.Generator().manual_seed(SEED)
    mapped = PrimitiveTypeMap(
        inventory, 2, generator=generator, initial_scale=INITIAL_SCALE
    )
    slots = slot_types(inventory, device=torch.device("cpu"))
    summed = mapped.summed(slots)
    vectors = mapped.vectors

    assert torch.allclose(summed[0], vectors[0] + vectors[inventory.null_index])
    assert torch.allclose(summed[1], vectors[0] + vectors[1])


def test_a_rung_scores_a_row_by_the_verb_and_the_frame_the_row_records() -> None:
    spec = small_spec(JointTypeTensor())
    generator = torch.Generator().manual_seed(SEED)
    (rung,) = ladder_of((spec,), generator=generator)
    design = small_design(ROWS)
    scores = rung.latent_scores(design)
    table = rung.table()
    verbs = {verb: index for index, verb in enumerate(spec.rows.levels)}
    frames = {label: index for index, label in enumerate(spec.columns.inventory.labels)}

    assert tuple(table.shape) == (
        len(spec.rows.levels),
        len(spec.columns.inventory.levels),
    )

    for position, (verb, frame, _) in enumerate(ROWS):
        assert scores[position] == table[verbs[verb], frames[frame]]


def test_a_design_recording_a_verb_the_comparison_never_declared_is_refused() -> None:
    spec = small_spec(JointTypeTensor())
    generator = torch.Generator().manual_seed(SEED)
    (rung,) = ladder_of((spec,), generator=generator)
    design = small_design((("promise", "_ NP", 1),))

    with pytest.raises(ValueError, match="does not declare"):
        rung.latent_scores(design)


def test_a_design_without_the_factor_the_comparison_reads_is_refused() -> None:
    spec = small_spec(JointTypeTensor())
    generator = torch.Generator().manual_seed(SEED)
    (rung,) = ladder_of((spec,), generator=generator)
    codec = DesignCodec(5, factors=("frame",))
    design = codec.encode(
        (JudgmentRow("a0", "think|_ NP", 1, RowColumns({"frame": "_ NP"}, {})),)
    )

    with pytest.raises(ValueError, match="reads the factor 'verb'"):
        rung.latent_scores(design)


def test_the_two_inventories_are_indexed_apart_when_they_use_the_same_names() -> None:
    frames = small_inventory()
    collided = LexicalPrimitiveInventory(names=frames.primitive_types)
    spec = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell")),
        columns=TableColumns(attribute="frame", inventory=frames),
        width=2,
        lift=LexicalPossession(lexical_primitives=collided),
        schedule=SCHEDULE,
    )
    wider = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell")),
        columns=TableColumns(attribute="frame", inventory=frames),
        width=2,
        lift=LexicalPossession(
            lexical_primitives=LexicalPrimitiveInventory(
                names=(*frames.primitive_types, "cause")
            )
        ),
        schedule=SCHEDULE,
    )
    shapes = {
        tuple(parameter.shape)
        for parameter in one_rung_of(spec).parameters()
        if parameter.ndim == 2
    }
    widened = {
        tuple(parameter.shape)
        for parameter in one_rung_of(wider).parameters()
        if parameter.ndim == 2
    }

    assert spec_digest(spec) != spec_digest(wider)
    assert (frames.type_count, spec.width) in shapes
    assert (frames.type_count, spec.width) in widened
    assert (len(spec.rows.levels), collided.count) in shapes
    assert (len(spec.rows.levels), collided.count + 1) in widened


def one_rung_of(spec: TableAblationSpec) -> TableRung:
    generator = torch.Generator().manual_seed(SEED)
    (built,) = ladder_of((spec,), generator=generator)

    return built


def test_sigmoid_pinning_keeps_special_rows_exact_and_every_regular_column_free() -> (
    None
):
    generator = torch.Generator().manual_seed(SEED)
    mapping = PinnedMap(
        5,
        4,
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=2,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )
    logits = mapping.logits

    assert logits is not None
    assert tuple(logits.shape) == (3, 4)
    assert tuple(mapping.parameters()) == (logits,)
    assert torch.equal(mapping()[:2], torch.eye(2, 4, dtype=torch.float64))
    assert torch.all(mapping()[2:] > 0.0)
    assert torch.all(mapping()[2:] < 1.0)

    before = mapping()[:2].clone()
    initial_logits = logits.detach().clone()
    weights = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    optimizer = torch.optim.SGD(mapping.parameters(), lr=0.25)
    loss = (mapping()[2:] * weights).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.all(logits.grad != 0.0)

    optimizer.step()

    assert not torch.equal(logits, initial_logits)
    assert torch.equal(mapping()[:2], before)
    assert torch.equal(mapping()[:2], torch.eye(2, 4, dtype=torch.float64))


def test_softmax_pinning_zeros_only_reserved_columns_and_normalizes_regular_rows() -> (
    None
):
    generator = torch.Generator().manual_seed(SEED)
    mapping = PinnedMap(
        5,
        4,
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=2,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )
    values = mapping()
    logits = mapping.logits

    assert logits is not None
    assert tuple(logits.shape) == (3, 2)
    assert torch.equal(values[:2], torch.eye(2, 4, dtype=torch.float64))
    assert torch.equal(values[2:, :2], torch.zeros(3, 2, dtype=torch.float64))
    assert torch.all(values[2:, 2:] > 0.0)
    assert torch.allclose(values[2:].sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_an_all_special_map_is_an_exact_identity_with_no_parameters() -> None:
    generator = torch.Generator().manual_seed(SEED)
    primitives = PinnedMap(
        3,
        3,
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=3,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )
    clusters = PinnedMap(
        3,
        3,
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=3,
        generator=generator,
        initial_scale=INITIAL_SCALE,
    )
    identity = torch.eye(3, dtype=torch.float64)

    assert tuple(primitives.parameters()) == ()
    assert tuple(clusters.parameters()) == ()
    assert torch.equal(primitives(), identity)
    assert torch.equal(clusters(), identity)


def test_a_public_pinned_map_refuses_invalid_target_mode_and_cardinality() -> None:
    generator = torch.Generator().manual_seed(SEED)
    bad_target = cast(
        "Literal['semantic_primitives', 'item_clusters']",
        "frames",
    )
    bad_mode = cast(
        "Literal['independent_sigmoid', 'normalized_softmax']",
        "sigmoid",
    )

    with pytest.raises(ValueError, match="target"):
        PinnedMap(
            3,
            3,
            target=bad_target,
            mode="independent_sigmoid",
            cardinality=1,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match="mode"):
        PinnedMap(
            3,
            3,
            target="semantic_primitives",
            mode=bad_mode,
            cardinality=1,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match=r"target.*requires mode"):
        PinnedMap(
            3,
            3,
            target="semantic_primitives",
            mode="normalized_softmax",
            cardinality=1,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match="cardinality"):
        PinnedMap(
            3,
            3,
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=-1,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match=r"cardinality.*item cardinality"):
        PinnedMap(
            2,
            4,
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=3,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match=r"cardinality.*inventory cardinality"):
        PinnedMap(
            4,
            2,
            target="semantic_primitives",
            mode="independent_sigmoid",
            cardinality=3,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )

    with pytest.raises(ValueError, match=r"cardinality.*regular rows"):
        PinnedMap(
            3,
            2,
            target="item_clusters",
            mode="normalized_softmax",
            cardinality=2,
            generator=generator,
            initial_scale=INITIAL_SCALE,
        )


def test_the_composite_routes_primitive_and_cluster_prefixes_separately() -> None:
    primitive_prefix = PinnedPrefix(
        target="semantic_primitives",
        mode="independent_sigmoid",
        cardinality=1,
    )
    cluster_prefix = PinnedPrefix(
        target="item_clusters",
        mode="normalized_softmax",
        cardinality=2,
    )
    lexical = LexicalPrimitiveInventory(names=("cause", "attitude", "motion"))
    spec = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think", "tell", "promise")),
        columns=TableColumns(attribute="frame", inventory=small_inventory()),
        width=2,
        lift=PossessionAndClasses(
            lexical_primitives=lexical,
            classes=4,
            primitive_prefix=primitive_prefix,
            cluster_prefix=cluster_prefix,
        ),
        schedule=SCHEDULE,
    )
    maps = {
        module.target: module
        for module in one_rung_of(spec).modules()
        if isinstance(module, PinnedMap)
    }
    primitives = maps["semantic_primitives"]
    clusters = maps["item_clusters"]

    assert primitives.mode == "independent_sigmoid"
    assert primitives.cardinality == 1
    assert primitives.inventory == lexical.count
    assert primitives.logits is not None
    assert tuple(primitives.logits.shape) == (2, lexical.count)
    assert clusters.mode == "normalized_softmax"
    assert clusters.cardinality == 2
    assert clusters.inventory == 4
    assert clusters.logits is not None
    assert tuple(clusters.logits.shape) == (1, 2)
    assert torch.equal(clusters()[2:, :2], torch.zeros(1, 2, dtype=torch.float64))


@pytest.mark.parametrize(
    ("field", "value"),
    [("steps", 0), ("batch_size", 0), ("initial_scale", 0.0)],
)
def test_a_fit_schedule_refuses_nonpositive_quantities(
    field: str, value: float
) -> None:
    payload = SCHEDULE.model_dump()
    payload[field] = value

    with pytest.raises(dx.ValidationError):
        FitSchedule.model_validate(payload)


def test_a_density_over_a_design_of_other_levels_is_refused() -> None:
    rung = one_rung_of(small_spec(JointTypeTensor()))
    design = small_design(ROWS)

    with pytest.raises(ValueError, match="was given a design over"):
        mean_log_density(rung, OrdinalHead(7), design)


def test_a_fit_moves_a_rung_toward_the_responses_the_design_records() -> None:
    levels, fitted, _ = split()
    rung = one_rung(DIRECT)
    head = OrdinalHead(levels)

    with torch.no_grad():
        before = float(mean_log_density(rung, head, fitted))

    fit_rung(rung, head, fitted)

    with torch.no_grad():
        after = float(mean_log_density(rung, head, fitted))

    assert after > before


def test_a_fit_leaves_the_control_saying_nothing_about_a_held_out_cell() -> None:
    levels, fitted, _ = split()
    spec = rung_spec(CONTROL)
    rung = one_rung(CONTROL)
    head = OrdinalHead(levels)
    before = rung.table().detach().clone()
    fit_rung(rung, head, fitted)
    after = rung.table().detach()
    verbs = {verb: index for index, verb in enumerate(spec.rows.levels)}
    frames = {label: index for index, label in enumerate(spec.columns.inventory.labels)}
    _, rows = judgments()
    seen = {
        (verbs[str(row["verb"])], frames[str(row["frame"])])
        for row in rows
        if not row["held_out"]
    }
    unseen = {
        (verbs[str(row["verb"])], frames[str(row["frame"])])
        for row in rows
        if row["held_out"]
    }

    assert unseen

    for verb, frame in unseen:
        assert after[verb, frame] == before[verb, frame]

    assert all(after[verb, frame] != before[verb, frame] for verb, frame in seen)


def test_the_control_is_beaten_on_held_out_cells_by_the_structured_direct_rung() -> (
    None
):
    levels, fitted, held_out = split()
    densities = {}

    for name in (CONTROL, DIRECT):
        rung = one_rung(name)
        head = OrdinalHead(levels)
        fit_rung(rung, head, fitted)

        with torch.no_grad():
            densities[name] = float(mean_log_density(rung, head, held_out))

    assert densities[DIRECT] > densities[CONTROL]


def test_the_structured_direct_rung_reads_the_frame_a_held_out_row_records() -> None:
    levels, rows = judgments()
    _, fitted, held_out = split()
    rung = one_rung(DIRECT)
    head = OrdinalHead(levels)
    fit_rung(rung, head, fitted)
    labels = rung_spec(DIRECT).columns.inventory.labels
    shifted = tuple(
        {
            **row,
            "frame": labels[(labels.index(str(row["frame"])) + 1) % len(labels)],
        }
        for row in rows
        if row["held_out"]
    )

    with torch.no_grad():
        recorded = float(mean_log_density(rung, head, held_out))
        permuted = float(mean_log_density(rung, head, design_of(levels, shifted)))

    assert recorded > permuted


class Unparameterized(ColumnLift, extra="forbid"):
    lift: Literal["unparameterized"] = "unparameterized"


def test_a_lift_this_module_parameterizes_no_way_is_refused_at_construction() -> None:
    spec = TableAblationSpec(
        rows=TableRows(attribute="verb", levels=("think",)),
        columns=TableColumns(attribute="frame", inventory=small_inventory()),
        width=2,
        lift=Unparameterized(),
        schedule=SCHEDULE,
    )
    generator = torch.Generator().manual_seed(SEED)

    with pytest.raises(TypeError, match="no parameterization"):
        ladder_of((spec,), generator=generator)
