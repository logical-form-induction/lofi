from itertools import pairwise
from math import log2
from typing import Final

import pytest
import torch

from lofi.probe._description import (
    VOITA_BLOCKS,
    BlockSchedule,
    CodeBlock,
    Codelength,
    ProbeComplexity,
    measure_complexity,
    online_codelength,
    uniform_codelength,
)
from lofi.probe._selectivity import (
    ClassifierProbe,
    ControlTask,
    ProbeCapacity,
    ProbeResult,
    SpanInventory,
)

CLASSES: Final = 2
FEATURES: Final = 4
SPANS: Final = 512

# a linear probe fitted long enough to find a planted coordinate from a short prefix
READER: Final = ProbeCapacity(hidden=0, epochs=150, learning_rate=0.3, seed=3)

# four blocks, so a nine-level sweep runs twenty-seven fits rather than ninety
COARSE: Final = BlockSchedule((0.125, 0.25, 0.5, 1.0))

LABELS: Final = torch.randint(
    CLASSES, (SPANS,), generator=torch.Generator().manual_seed(1)
).tolist()

VECTORS: Final = torch.randn(
    SPANS, FEATURES, generator=torch.Generator().manual_seed(7), dtype=torch.float64
)


def planted(level: int) -> SpanInventory:
    """Return spans of which ``level`` in every eight state their label outright."""
    vectors = VECTORS.clone()

    for at in range(SPANS):
        if at % 8 < level:
            vectors[at, 0] = 3.0 * (2 * LABELS[at] - 1)

    return SpanInventory(
        vectors, [f"span-{at}" for at in range(SPANS)], LABELS, classes=CLASSES
    )


def squares(spans: int, classes: int) -> SpanInventory:
    """Return spans of one coordinate whose label is the coordinate's class."""
    draws = torch.Generator().manual_seed(4)
    labels = torch.randint(classes, (spans,), generator=draws).tolist()
    vectors = torch.tensor([[float(label)] for label in labels], dtype=torch.float64)

    return SpanInventory(
        vectors, [f"span-{at}" for at in range(spans)], labels, classes=classes
    )


class _Reported:
    """A probe report stating whatever it is built with.

    Parameters
    ----------
    accuracy : float
        What to report as the real-task accuracy.
    control_accuracy : float
        What to report as the control-task accuracy.
    selectivity : float
        What to report as the gap, whether or not it is one.
    """

    def __init__(
        self, accuracy: float, control_accuracy: float, selectivity: float
    ) -> None:
        self.accuracy = accuracy
        self.control_accuracy = control_accuracy
        self.selectivity = selectivity
        self.control_seed = 0
        self.capacity = READER
        self.parameters = 1


def test_a_schedule_cuts_its_blocks_where_its_fractions_fall() -> None:
    doubled = (1, 2, 4, 8, 16, 32, 62, 125, 250, 500, 1000)

    assert VOITA_BLOCKS.boundaries(1000) == doubled
    assert VOITA_BLOCKS.boundaries(10) == (1, 2, 5, 10)
    assert VOITA_BLOCKS.boundaries(1) == (1,)
    assert len(VOITA_BLOCKS) == 11
    assert BlockSchedule((0.5, 1.0)).boundaries(8) == (4, 8)


@pytest.mark.parametrize(
    "fractions",
    [
        (),
        (0.5,),
        (0.0, 1.0),
        (-0.25, 1.0),
        (0.5, 0.25, 1.0),
        (0.5, 0.5, 1.0),
        (0.5, 2.0),
        (float("inf"), 1.0),
    ],
)
def test_a_schedule_refuses_boundaries_that_do_not_ascend_to_one(
    fractions: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="block"):
        BlockSchedule(fractions)


def test_a_schedule_refuses_a_boundary_that_is_not_a_number() -> None:
    with pytest.raises(TypeError):
        BlockSchedule(("half", 1.0))  # ty: ignore[invalid-argument-type]


def test_a_schedule_refuses_a_count_of_labels_it_cannot_cut() -> None:
    with pytest.raises(ValueError, match="positive"):
        VOITA_BLOCKS.boundaries(0)

    with pytest.raises(TypeError):
        VOITA_BLOCKS.boundaries(4.0)  # ty: ignore[invalid-argument-type]


def test_a_schedule_is_equal_and_hashes_on_the_boundaries_it_states() -> None:
    schedule = BlockSchedule((0.5, 1.0))

    assert schedule == BlockSchedule([0.5, 1.0])
    assert schedule != BlockSchedule((0.25, 1.0))
    assert schedule != (0.5, 1.0)
    assert len({schedule, BlockSchedule((0.5, 1.0))}) == 1
    assert eval(repr(schedule)) == schedule  # noqa: S307


def test_the_uniform_code_charges_the_class_count_in_bits_for_every_span() -> None:
    assert uniform_codelength(squares(64, 4)) == 128.0
    assert uniform_codelength(squares(5, 2)) == 5.0

    with pytest.raises(TypeError):
        uniform_codelength(squares(4, 2).vectors)  # ty: ignore[invalid-argument-type]


def test_a_code_of_one_block_is_the_uniform_code() -> None:
    inventory = squares(64, 4)
    code = online_codelength(inventory, READER, schedule=BlockSchedule((1.0,)))

    assert code.blocks == (CodeBlock(0, 64, 128.0),)
    assert code.bits == 128.0
    assert code.uniform == 128.0
    assert code.compression == 1.0
    assert code.information == 0.0
    assert code.labels == 64


def test_every_block_is_charged_for_its_own_labels_and_no_others() -> None:
    """Spans that say nothing, balanced in every prefix, cost one bit each."""
    spans = 64
    inventory = SpanInventory(
        torch.ones(spans, 1, dtype=torch.float64),
        [f"same-{at}" for at in range(spans)],
        [at % 2 for at in range(spans)],
        classes=2,
    )
    code = online_codelength(
        inventory, READER, schedule=BlockSchedule((0.25, 0.5, 1.0))
    )

    assert [len(block) for block in code.blocks] == [16, 16, 32]

    for block in code.blocks:
        assert block.bits == pytest.approx(float(len(block)), abs=1e-3)

    assert code.bits == pytest.approx(64.0, abs=1e-3)
    assert code.uniform == 64.0


def test_the_first_block_is_charged_under_the_uniform_code() -> None:
    inventory = planted(8)
    code = online_codelength(inventory, READER, schedule=VOITA_BLOCKS)
    first = code.blocks[0]

    assert first.start == 0
    assert first.end == VOITA_BLOCKS.boundaries(SPANS)[0]
    assert first.bits == len(first) * log2(CLASSES)


def test_a_code_records_its_blocks_in_order_and_the_schedule_that_cut_them() -> None:
    code = online_codelength(planted(4), READER, schedule=COARSE)

    assert code.schedule == COARSE
    assert [block.start for block in code.blocks] == [0, 64, 128, 256]
    assert [block.end for block in code.blocks] == [64, 128, 256, 512]
    assert code.labels == SPANS
    assert code.bits == pytest.approx(sum(block.bits for block in code.blocks))


def test_a_code_shortens_at_every_level_of_planted_signal() -> None:
    lengths = [
        online_codelength(planted(level), READER, schedule=COARSE).bits
        for level in range(9)
    ]

    assert all(later < earlier for earlier, later in pairwise(lengths))


def test_a_representation_that_states_nothing_costs_more_than_knowing_nothing() -> None:
    code = online_codelength(planted(0), READER, schedule=COARSE)

    assert code.bits > code.uniform
    assert code.information < 0.0
    assert code.compression < 1.0


def test_the_bits_do_not_fall_to_a_probe_that_is_merely_wider() -> None:
    wide = ProbeCapacity(hidden=64, epochs=150, learning_rate=0.3, seed=3)
    memorized = online_codelength(planted(0), wide, schedule=COARSE)
    read = online_codelength(planted(8), wide, schedule=COARSE)

    assert memorized.information < 0.0
    assert memorized.bits > online_codelength(planted(0), READER, schedule=COARSE).bits
    assert read.bits == pytest.approx(
        online_codelength(planted(8), READER, schedule=COARSE).bits, abs=1.0
    )


def test_a_code_refuses_a_capacity_or_a_schedule_it_cannot_read() -> None:
    inventory = squares(16, 2)

    with pytest.raises(TypeError):
        online_codelength(inventory, READER.seed, schedule=COARSE)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        online_codelength(inventory, READER, schedule=COARSE.fractions)  # ty: ignore[invalid-argument-type]


def test_a_block_transmits_at_least_one_label_at_a_cost_it_can_report() -> None:
    block = CodeBlock(2, 6, 8.0)

    assert len(block) == 4
    assert block == CodeBlock(2, 6, 8.0)
    assert block != CodeBlock(2, 6, 9.0)
    assert block != (2, 6, 8.0)
    assert len({block, CodeBlock(2, 6, 8.0)}) == 1
    assert eval(repr(block)) == block  # noqa: S307

    with pytest.raises(ValueError, match="no label"):
        CodeBlock(6, 6, 1.0)

    with pytest.raises(ValueError, match="no bits"):
        CodeBlock(0, 6, -1.0)


def test_a_codelength_refuses_blocks_that_do_not_cover_the_labels_once() -> None:
    with pytest.raises(ValueError, match="no block"):
        Codelength((), 8.0, COARSE)

    with pytest.raises(ValueError, match="twice or not"):
        Codelength((CodeBlock(1, 4, 4.0),), 8.0, COARSE)

    with pytest.raises(ValueError, match="twice or not"):
        Codelength((CodeBlock(0, 4, 4.0), CodeBlock(5, 8, 4.0)), 8.0, COARSE)

    with pytest.raises(ValueError, match="not positive"):
        Codelength((CodeBlock(0, 4, 4.0),), 0.0, COARSE)

    with pytest.raises(TypeError):
        Codelength((CodeBlock(0, 4, 4.0),), 8.0, COARSE.fractions)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        Codelength(((0, 4, 4.0),), 8.0, COARSE)  # ty: ignore[invalid-argument-type]


def test_a_complexity_carries_the_bits_beside_the_accuracy_and_the_selectivity() -> (
    None
):
    fitting = planted(8)
    held_out = SpanInventory(
        VECTORS[:64], [f"held-{at}" for at in range(64)], LABELS[:64], classes=CLASSES
    )
    probe = ClassifierProbe(READER, ControlTask(seed=11))
    measured = measure_complexity(probe, fitting, held_out, schedule=COARSE)
    reported = probe.run(fitting, held_out)
    alone = online_codelength(fitting, READER, schedule=COARSE)

    assert measured.accuracy == reported.accuracy
    assert measured.control_accuracy == reported.control_accuracy
    assert measured.selectivity == reported.selectivity
    assert measured.description_length == alone.bits
    assert measured.codelength.blocks == alone.blocks
    assert measured.report.control_seed == 11
    assert "selectivity" in repr(measured)


def test_a_complexity_refuses_bits_reported_without_both_accuracies() -> None:
    code = online_codelength(squares(16, 2), READER, schedule=BlockSchedule((1.0,)))
    honest = ProbeResult(
        0.75, 0.25, control_seed=1, capacity=READER, parameters=READER.parameters(1, 2)
    )

    assert ProbeComplexity(honest, code).selectivity == 0.5

    with pytest.raises(ValueError, match="not its accuracy"):
        ProbeComplexity(_Reported(0.75, 0.25, 0.75), code)

    with pytest.raises(ValueError, match="not a proportion"):
        ProbeComplexity(_Reported(1.5, 0.25, 1.25), code)

    with pytest.raises(TypeError):
        ProbeComplexity(honest, code.bits)  # ty: ignore[invalid-argument-type]
